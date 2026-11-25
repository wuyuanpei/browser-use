"""System1 backends for TypeSafe Jev and structured-output browser-use chat models."""

import asyncio
import base64
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, create_model

from browser_use.agent.jev.decision import TARGET_OPERATIONS, request_body
from browser_use.agent.jev.questions import LLM_SYSTEM1_INSTRUCTION
from browser_use.agent.jev.views import JEV_RETRY_DELAYS, MAX_CHOICE_OPTIONS, MAX_REQUEST_CHARS, JevSettings, System1Reply
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage, SystemMessage, UserMessage
from browser_use.llm.views import ChatInvokeCompletion


class System1Error(RuntimeError):
	"""A System1 transport, response envelope, or structured model output error."""

	def __init__(self, message: str, attempts: list[dict[str, Any]] | None = None):
		super().__init__(message[:300])
		self.attempts = attempts or []


def image_rejection_status(attempts: list[dict[str, Any]]) -> int | None:
	"""Exclude authentication failures, timeouts and rate limits from image rejections."""
	status = attempts[-1].get('status') if attempts else None
	if isinstance(status, int) and 400 <= status < 500 and status not in (401, 403, 408, 429):
		return status
	return None


class System1Backend(Protocol):
	@property
	def name(self) -> Literal['jev', 'llm']: ...

	async def ask(
		self, state: dict[str, Any], questions: dict[str, dict[str, Any]], *, images: dict[str, str] | None = None
	) -> System1Reply: ...

	async def aclose(self) -> None: ...


def _elapsed(start: float) -> int:
	return round((time.monotonic() - start) * 1000)


def _validate_questions(questions: dict[str, dict[str, Any]]) -> None:
	if not 1 <= len(questions) <= 32:
		raise ValueError('System1 requires 1 to 32 questions')
	for question in questions.values():
		criteria = question.get('criteria')
		if question.get('type') != 'choice' or not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_CHOICE_OPTIONS:
			raise ValueError('Each choice question requires 1 to 255 options')


class JevBackend:
	name: Literal['jev'] = 'jev'

	def __init__(self, settings: JevSettings, api_key: str):
		if not api_key:
			raise ValueError('Jev requires an API key')
		self.model = settings.jev_model
		self.last_attempts: list[dict[str, Any]] = []
		self._url = f'{settings.jev_base_url}/v1/systemone'
		self._client = httpx.AsyncClient(
			headers={'Authorization': f'Bearer {api_key}'},
			follow_redirects=False,
			timeout=settings.jev_timeout,
			trust_env=urlsplit(self._url).hostname not in ('localhost', '127.0.0.1', '::1'),
		)

	async def aclose(self) -> None:
		await self._client.aclose()

	async def ask(
		self, state: dict[str, Any], questions: dict[str, dict[str, Any]], *, images: dict[str, str] | None = None
	) -> System1Reply:
		attempts: list[dict[str, Any]] = []
		self.last_attempts = attempts
		_validate_questions(questions)
		envelope = request_body(state, questions, self.model, with_image=bool(images))
		body = json.dumps(envelope, ensure_ascii=False, allow_nan=False)
		if len(body) > MAX_REQUEST_CHARS:
			raise ValueError('System1 request exceeds the character limit')
		image_bytes = 0
		if images:
			if images.keys() != {'page-screenshot'} or 'assets' not in envelope:
				raise ValueError('A Jev image request requires a page.screenshot reference and a page-screenshot part')
			jpeg = base64.b64decode(images['page-screenshot'], validate=True)
			image_bytes = len(jpeg)
			# Build once: both in-memory parts and the generated multipart boundary replay on a retry.
			request = self._client.build_request(
				'POST',
				self._url,
				files=[
					('request', (None, body, 'application/json')),
					('page-screenshot', ('page-screenshot.jpg', jpeg, 'image/jpeg')),
				],
				headers={'TypeSafe-Danger-Zone-Images': 'true'},
			)
		else:
			request = self._client.build_request(
				'POST', self._url, content=body.encode('utf-8'), headers={'Content-Type': 'application/json'}
			)
		start = time.monotonic()
		for index in range(len(JEV_RETRY_DELAYS) + 1):
			attempt_start = time.monotonic()
			attempt: dict[str, Any] = {'attempt': index + 1, 'status': None, 'error_type': None, 'latency_ms': 0}
			attempts.append(attempt)
			status = None
			error_type = None
			response = None
			try:
				response = await self._client.send(request)
				status = response.status_code
			except httpx.TransportError as error:
				# Error strings and response bodies may contain headers or other private data.
				error_type = type(error).__name__
			except asyncio.CancelledError:
				error_type = 'CancelledError'
				raise
			finally:
				attempt.update(status=status, error_type=error_type, latency_ms=_elapsed(attempt_start))
			if response is not None and response.is_success:
				try:
					payload = response.json()
					if not isinstance(payload, dict) or not isinstance(payload.get('answers'), dict):
						raise ValueError('Missing answers')
					if not questions.keys() <= payload['answers'].keys():
						raise ValueError('Incomplete answers')
					usage = payload.get('usage') or {}
					if not isinstance(usage, dict):
						raise ValueError('Invalid usage')
					return System1Reply(
						answers=payload['answers'],
						model=payload.get('model'),
						input_tokens=usage.get('input_tokens'),
						output_tokens=usage.get('output_tokens'),
						image_count=int(bool(images)),
						image_bytes=image_bytes,
						has_probabilities=True,
						latency_ms=_elapsed(start),
						attempts=attempts,
					)
				except (ValueError, TypeError, UnicodeError):
					raise System1Error('Invalid Jev response envelope or missing answers', attempts) from None
			if status not in (None, 429, 503, 529) or index == len(JEV_RETRY_DELAYS):
				reason = f'HTTP {status}' if status is not None else error_type
				raise System1Error(f'Jev request failed ({reason})', attempts)
			await asyncio.sleep(JEV_RETRY_DELAYS[index])
		raise AssertionError('Jev retry loop did not terminate')


_Invoke = Callable[..., Awaitable[ChatInvokeCompletion[Any]]]


class LLMBackend:
	name: Literal['llm'] = 'llm'

	def __init__(self, settings: JevSettings, *, invoke: _Invoke | None = None):
		if settings.system1_llm is None:
			raise ValueError('system1_llm is required')
		self._llm = settings.system1_llm
		self._two_stage = settings.system1_llm_two_stage
		self._invoke = invoke

	async def aclose(self) -> None:
		"""The Agent owns the chat model; this backend has no independent connection."""

	async def _choose(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
		fields: dict[str, Any] = {
			key: (Literal.__getitem__(tuple(question['criteria'])), ...)  # pyright: ignore[reportAttributeAccessIssue]
			for key, question in questions.items()
		}
		schema = create_model('System1Choices', __config__=ConfigDict(extra='forbid'), **fields)
		content = json.dumps({'state': state, 'questions': questions}, ensure_ascii=False, allow_nan=False)
		messages: list[BaseMessage] = [SystemMessage(content=LLM_SYSTEM1_INSTRUCTION), UserMessage(content=content)]
		if self._invoke is None:
			response = await self._llm.ainvoke(messages, output_format=schema)
		else:
			response = await self._invoke(self._llm, messages, output_format=schema)
		raw = response.completion
		if isinstance(raw, str):
			choices = schema.model_validate_json(raw)
		else:
			choices = schema.model_validate(raw.model_dump() if isinstance(raw, BaseModel) else raw)
		return {name: {'choice': choice} for name, choice in choices.model_dump().items()}

	async def ask(
		self, state: dict[str, Any], questions: dict[str, dict[str, Any]], *, images: dict[str, str] | None = None
	) -> System1Reply:
		if images:
			raise ValueError('Screenshot inputs require the Jev backend')
		_validate_questions(questions)
		if 'operation' not in questions:
			raise ValueError('System1 requires an operation question')
		start = time.monotonic()
		attempts: list[dict[str, Any]] = []

		async def choose(batch: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
			began = time.monotonic()
			try:
				result = await self._choose(state, batch)
			except Exception as error:
				error_name = type(error).__name__
				attempts.append(
					{'attempt': len(attempts) + 1, 'status': None, 'error_type': error_name, 'latency_ms': _elapsed(began)}
				)
				raise System1Error(f'System1 LLM call failed ({error_name})', attempts) from None
			attempts.append({'attempt': len(attempts) + 1, 'status': None, 'error_type': None, 'latency_ms': _elapsed(began)})
			return result

		if self._two_stage:
			answers = await choose({'operation': questions['operation']})
			operation = answers['operation']['choice']
			if operation in TARGET_OPERATIONS:
				name = f'{operation.lower()}_target'
				if name not in questions:
					raise System1Error('No target question for the selected operation', attempts)
				answers.update(await choose({name: questions[name]}))
		else:
			answers = await choose(questions)
		return System1Reply(
			answers=answers, model=self._llm.model, has_probabilities=False, attempts=attempts, latency_ms=_elapsed(start)
		)


def make_backend(
	settings: JevSettings, api_key: str | None, agent_llm: BaseChatModel, *, invoke: _Invoke | None = None
) -> System1Backend:
	if settings.system1 == 'llm':
		return LLMBackend(settings, invoke=invoke)
	return JevBackend(settings, api_key or '')
