"""Run Browser Use in an owned browser and publish benchmark observations."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar
from uuid import uuid4

from .decision_guard import decision_validity, read_document, same_document, stale_disposition, target_guard
from .models import PROVIDER, RunSettings, redact_secrets
from .observation import READ_STATE, document_key, fingerprint, indexed_elements
from .tasks import EnvironmentBlocked
from .token_usage import TOKEN_FIELDS, token_usage

if TYPE_CHECKING:
	from browser_use.agent.service import Agent
	from browser_use.browser.session import BrowserSession, CDPSession
	from browser_use.browser.views import BrowserStateSummary

	from .chrome_attachment import ChromeAttachment
	from .tasks import CaseProbe

MAX_STEPS = 100
DOCUMENT_READY_TIMEOUT = 5.0
DOCUMENT_READY_POLL = 0.05
OBSERVATION_RETRIES = 3
OBSERVATION_RETRY_DELAY = 1.0
OBSERVATION_FAILURE_LIMIT = 5
OBSERVATION_RECOVERY_TIMEOUT = 20.0
VIEWPORT = {'width': 1366, 'height': 780}
BrowserMode = Literal['isolated', 'attached_chrome']
_Result = TypeVar('_Result')


def _attached_tools(excluded: list[str], mode: str):
	"""Keep the shared browser's keyboard actions restricted to verified System1 submission."""
	from browser_use import Tools
	from browser_use.tools.views import SendKeysAction

	if mode != 'isolated':
		return Tools(exclude_actions=excluded)

	class EnterKey(SendKeysAction):
		keys: Literal['Enter']

	# Only System1 can use the isolated runner's keyboard action. Its dispatch verifies the original field.
	tools = Tools(exclude_actions=[name for name in excluded if name != 'send_keys'])
	tools.registry.registry.actions['send_keys'].param_model = EnterKey
	return tools


class RunCancelled(RuntimeError):
	"""Cooperative cancellation at a boundary before browser input."""

	is_cancellation = True


class BrowserUnavailable(EnvironmentBlocked):
	"""The shared browser cannot start another case until it recovers."""


class PageObservationUnavailable(EnvironmentBlocked):
	def __init__(self, message: str, target_id: str | None):
		super().__init__(message)
		self.target_id = target_id


@dataclass(frozen=True)
class Preview:
	image: str | None
	error: str | None
	sample: dict[str, Any]


@dataclass(frozen=True)
class RunEvent:
	name: str
	payload: dict[str, Any]
	snapshot: dict[str, Any]
	preview: Preview | None


if os.name == 'nt':

	class _WindowsBrowserLoop(asyncio.ProactorEventLoop):
		"""Own the native watchdog's subprocess transports until their pipes close."""

		def __init__(self):
			super().__init__()
			self._child_transports = []

		async def subprocess_exec(self, *args, **kwargs):
			transport, protocol = await super().subprocess_exec(*args, **kwargs)
			self._child_transports.append(transport)
			return transport, protocol

		def close(self):
			# Browser Use retains only the browser PID. The asyncio pipe handles
			# must be released on their owning loop before closing its proactor.
			for transport in self._child_transports:
				transport.close()
			self._child_transports.clear()
			if not self.is_closed():
				self.run_until_complete(asyncio.sleep(0))
			super().close()


class _BrowserLoop:
	"""Keep CDP and its owning BrowserSession on one loop between HTTP requests."""

	def __init__(self):
		self.loop = _WindowsBrowserLoop() if os.name == 'nt' else asyncio.new_event_loop()
		self.thread = threading.Thread(target=self._serve, name='benchmark-browser', daemon=True)
		self.thread.start()

	def _serve(self):
		asyncio.set_event_loop(self.loop)
		try:
			self.loop.run_forever()
		finally:
			pending = asyncio.all_tasks(self.loop)
			for task in pending:
				task.cancel()
			if pending:
				self.loop.run_until_complete(asyncio.wait(pending, timeout=3))
			self.loop.run_until_complete(self.loop.shutdown_asyncgens())
			self.loop.close()

	def run(
		self, operation: Coroutine[Any, Any, _Result], *, cancel: Callable[[], str | None] | None = None, timeout: float = 90
	) -> _Result:
		async def bounded() -> _Result:
			if cancel is None:
				return await asyncio.wait_for(operation, timeout=timeout)
			task = asyncio.create_task(operation)
			try:
				async with asyncio.timeout(timeout):
					while not task.done():
						reason = cancel()
						if reason:
							raise RunCancelled(str(reason))
						await asyncio.wait({task}, timeout=0.12)
					return task.result()
			finally:
				if not task.done():
					task.cancel()
					await asyncio.gather(task, return_exceptions=True)

		return asyncio.run_coroutine_threadsafe(bounded(), self.loop).result()

	def close(self):
		self.loop.call_soon_threadsafe(self.loop.stop)
		self.thread.join()


class _BoundBrowser:
	"""Read observations and prepare fixtures on the currently owned CDP target."""

	def __init__(self, bridge):
		self.bridge = bridge
		self.native: BrowserSession | None = None
		self.bound: CDPSession | None = None

	async def bind(self) -> str:
		browser = self.native
		if browser is None or not (target := browser.agent_focus_target_id):
			raise PageObservationUnavailable('Browser Use has no active page', None)
		try:
			bound = await browser.get_or_create_cdp_session(target, focus=False)
		except ValueError as error:
			raise PageObservationUnavailable('Browser session is unavailable before sampling', target) from error
		self.bound = bound
		return bound.target_id

	async def send(self, method, params):
		browser = self.native
		bound = self.bound
		if browser is None or bound is None or browser.agent_focus_target_id != bound.target_id:
			raise PageObservationUnavailable('Browser target changed before sampling', bound.target_id if bound else None)
		try:
			active = await browser.get_or_create_cdp_session(bound.target_id, focus=False)
		except ValueError as error:
			raise PageObservationUnavailable('Browser session is unavailable before sampling', bound.target_id) from error
		if active.session_id != bound.session_id:
			raise PageObservationUnavailable('Browser session changed before sampling', bound.target_id)
		group, name = method.split('.')
		command = getattr(getattr(bound.cdp_client.send, group), name)
		result = await command(params=params, session_id=bound.session_id)
		if browser.agent_focus_target_id != bound.target_id:
			raise PageObservationUnavailable('Browser target changed while sampling', bound.target_id)
		return result

	def call(self, method, **params):
		timeout = params.pop('_response_timeout', 20)
		return self.bridge.run(self.send(method, params), timeout=timeout)

	def evaluate(self, expression: str) -> Any:
		result = self.call('Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
		return self._evaluation_value(result)

	@staticmethod
	def _evaluation_value(result: dict[str, Any]) -> Any:
		if result.get('exceptionDetails'):
			raise RuntimeError(f'Browser evaluation failed: {result["exceptionDetails"].get("text", "")}')
		return result['result'].get('value')

	async def _read_page(self, target: str) -> dict[str, Any]:
		async def sample():
			try:
				return self._evaluation_value(
					await self.send('Runtime.evaluate', {'expression': READ_STATE, 'returnByValue': True, 'awaitPromise': True})
				)
			except RuntimeError as error:
				message = str(error).casefold()
				if not isinstance(error, EnvironmentBlocked) and any(
					text in message for text in ('execution context was destroyed', 'cannot find context with specified id')
				):
					return None
				raise

		page = await sample()
		if isinstance(page, dict):
			return page
		# Navigation can leave the page temporarily without a body. Keep the same bound tab and session.
		try:
			async with asyncio.timeout(DOCUMENT_READY_TIMEOUT):
				while True:
					await asyncio.sleep(DOCUMENT_READY_POLL)
					page = await sample()
					if isinstance(page, dict):
						return page
		except TimeoutError as error:
			raise PageObservationUnavailable('Browser document is unavailable', target) from error

	def observe(
		self, *, cancel: Callable[[], str | None] | None = None, timeout: float = OBSERVATION_RECOVERY_TIMEOUT
	) -> dict[str, Any]:
		async def read():
			target = await self.bind()
			return target, await self._read_page(target)

		try:
			target, page = self.bridge.run(read(), cancel=cancel, timeout=timeout)
		except TimeoutError as error:
			raise PageObservationUnavailable(
				'Browser observation timed out', self.bound.target_id if self.bound else None
			) from error
		page.update(target_id=target, document_id=document_key(page)[1], observed_at=datetime.now(timezone.utc).isoformat())
		page['dom_fingerprint'] = page['fingerprint'] = fingerprint(page)
		return page

	def screenshot(self, timeout: int = 20) -> str:
		return self.call(
			'Page.captureScreenshot', format='jpeg', quality=72, captureBeyondViewport=False, _response_timeout=timeout
		)['data']


class BrowserUseRun:
	"""Own one browser session, its decisions and the inspector's published evidence."""

	def __init__(
		self,
		url: str,
		goal: str,
		settings: RunSettings,
		*,
		scenario: str | None = None,
		on_event: Callable[[RunEvent], None] | None = None,
		should_cancel: Callable[[], str | None] | None = None,
		probe: CaseProbe | None = None,
		record_dir: Path | None = None,
		jev_artifact_dir: Path | None = None,
		chrome: ChromeAttachment | None = None,
	):
		self._started = time.monotonic()
		self._deadline = settings.deadline_monotonic
		if self._deadline is None and settings.time_limit_seconds is not None:
			self._deadline = self._started + settings.time_limit_seconds
		self._secrets = (settings.execution.api_key, settings.evaluation.api_key, settings.jev.api_key if settings.jev else None)
		self._on_event, self._should_cancel, self._probe = on_event, should_cancel, probe
		self._record_dir = record_dir
		self._chrome = chrome
		self._stale_streak: list[str] = []
		self._pending_observation: tuple[str, str | None] | None = None
		self._observe_failures = 0
		self._observe_failed_since: float | None = None
		self._jev_dir = None
		if settings.jev is not None:
			self._jev_dir = jev_artifact_dir or Path.cwd() / 'artifacts' / 'jev-runs' / uuid4().hex
			self._jev_dir.mkdir(parents=True, exist_ok=True)
		self._model_role: ContextVar[str | None] = ContextVar('benchmark_jev_model_role', default=None)
		self._prediction_text_call: dict[str, Any] | None = None
		self._model_call_lock = threading.RLock()
		self._planner_parse_feedback: tuple[int, dict[str, Any]] | None = None
		self._model_token_usage = dict.fromkeys((*TOKEN_FIELDS, 'missing_usage_calls'), 0)
		config = settings.system1_config()
		self._state: dict[str, Any] = dict(
			goal=goal,
			system1_provider=config['provider'],
			system1_model=config['model'],
			system1_base_url=config['base_url'],
			execution_mode=settings.execution_mode,
			screenshot_mode=settings.screenshot_policy == 'always',
			screenshot_policy=settings.screenshot_policy,
			vision_escalation=settings.vision_escalation,
			candidate_pipeline=settings.effective_candidate_pipeline,
			min_read_operation_probability=settings.min_read_operation_probability if settings.jev is not None else None,
			model_profile=settings.profile,
			browser_mode='attached_chrome' if chrome is not None else 'isolated',
			text_model=settings.evaluation.model,
			text_model_base_url=settings.evaluation.public()['base_url'],
			status='ready',
			page={},
			decision=None,
			decisions=[],
			history=[],
			model_calls=[],
			text_calls=[],
			decision_rounds=0,
			stale_retries=0,
			forced_after_stale=0,
			stale_reasons={},
			elapsed_ms=0,
			event_seq=0,
			observation_index=-1,
			plan=[goal],
			plan_index=0,
			record=bool(record_dir),
			free_step_supported=True,
		)
		if self._jev_dir is not None:
			self._state.update(jev_trace_path=str(self._jev_dir / 'jev_trace.jsonl'), planner=settings.execution.public())
		if scenario is not None:
			self._state['scenario'] = scenario
		self._closed = False
		self._browser_start_dispatched = False
		self._check_cancelled()
		self._temp = tempfile.TemporaryDirectory(prefix='browser-use-benchmark-', ignore_cleanup_errors=True)
		self._bridge = chrome.bridge if chrome is not None else _BrowserLoop()
		self._browser = _BoundBrowser(self._bridge)
		self._native: Agent | None = None
		self._summary: BrowserStateSummary | None = None
		try:
			# BrowserStartEvent launches and attaches CDP on the native event bus.
			# Let that operation settle before issuing kill() on the same session.
			self._bridge.run(self._start(url, settings), timeout=110)
			self._check_cancelled()
			if probe:
				probe.prepare(self._browser.evaluate)
			self._check_cancelled()
			self.observe('initial')
		except BaseException:
			with suppress(BrowserUnavailable):
				self.close()
			raise

	def _check_cancelled(self):
		reason = self._should_cancel() if self._should_cancel else None
		if reason:
			raise RunCancelled(reason)
		if self._chrome is not None:
			self._chrome.check_connection()

	def _emit(self, name: str, *, preview: Preview | None = None, **payload: Any):
		self._state['event_seq'] += 1
		self._state['elapsed_ms'] = round((time.monotonic() - self._started) * 1000)
		if self._on_event:
			payload = redact_secrets({'event_seq': self._state['event_seq'], **payload}, *self._secrets)
			self._on_event(RunEvent(name, payload, self.snapshot(), preview))

	def snapshot(self) -> dict[str, Any]:
		self._sync_planner_parse_feedback()
		executor = getattr(self._native, '_jev', None)
		final_observation = executor.final_browser_observation if executor is not None else None
		roles = executor.usage_by_role() if executor is not None else {}
		system1_usage = roles.get('system1', {})
		disabled = system1_usage.get('vision_disabled')
		runs = executor.runs if executor is not None else []
		reviews = [run for run in runs if run.review_delegation]
		with self._model_call_lock:
			public_state = self._state.copy()
			calls = public_state.pop('model_calls').copy()
			usage = self._model_token_usage.copy()
		snapshot = redact_secrets(
			{
				**public_state,
				**(
					{
						'jev_usage_by_role': roles,
						'replans': executor.redelegations,
						'research': {'pages': [record.model_dump() for record in executor.captures.records]},
					}
					if executor is not None
					else {}
				),
				**({'final_browser_observation': final_observation.model_dump()} if final_observation is not None else {}),
				'vision_disabled': disabled['status'] if disabled else None,
				'escalations': system1_usage.get('escalations', 0),
				'escalations_executed': system1_usage.get('escalations_executed', 0),
				'escalations_by_reason': system1_usage.get('escalations_by_reason', {}),
				'planner_screenshot_requests': sum(run.screenshot_declared is True for run in runs),
				'review_delegations': len(reviews),
				'screenshot_decisions_missing': (
					sum(run.screenshot_declared is None for run in reviews) if self._state['screenshot_policy'] == 'smart' else 0
				),
				'token_usage': {**usage, 'complete': usage['missing_usage_calls'] == 0},
				'elements': indexed_elements(self._state['page'].get('actions', [])),
			},
			*self._secrets,
		)
		snapshot['model_calls'] = calls
		return snapshot

	def _public_call(self, record: dict[str, Any]) -> dict[str, Any]:
		cleaned = redact_secrets(record, *self._secrets)
		return deepcopy(record) if cleaned is record else cleaned

	def _account_model_call(self, record: dict[str, Any], sign: int) -> None:
		usage = token_usage([record])
		for key in self._model_token_usage:
			self._model_token_usage[key] += sign * usage[key]

	def _update_model_call(self, call_id: int, **metadata: Any) -> None:
		# A previously published snapshot can still reference the old record.
		with self._model_call_lock:
			calls = self._state['model_calls']
			calls[call_id - 1] = {**calls[call_id - 1], **self._public_call(metadata)}

	def _sync_planner_parse_feedback(self) -> None:
		# Planner marks this error after the provider call has already ended.
		with self._model_call_lock:
			if self._planner_parse_feedback is not None:
				call_id, failure = self._planner_parse_feedback
				if failure['planner_feedback_retry']:
					self._update_model_call(call_id, parse_failures=[failure])
					self._planner_parse_feedback = None

	@contextmanager
	def model_call(self, kind: str, **metadata: Any) -> Iterator[dict[str, Any]]:
		"""Record one real execution or verification request with the same sequencing."""
		with self._model_call_lock:
			calls = self._state['model_calls']
			call_id = len(calls) + 1
			record: dict[str, Any] = dict(
				id=call_id,
				kind=kind,
				status='pending',
				observation_index=self._state['observation_index'],
				event_seq=self._state['event_seq'] + 1,
				**metadata,
			)
			pending = self._public_call(record)
			calls.append(pending)
			self._account_model_call(pending, 1)
		self._emit('model_start', call=pending)
		started = time.perf_counter()
		try:
			yield record
			record['status'] = 'success'
		except BaseException as error:
			record.update(status='error', error=str(error) or type(error).__name__)
			raise
		finally:
			record['latency_ms'] = round((time.perf_counter() - started) * 1000)
			completed = self._public_call(record)
			with self._model_call_lock:
				calls[call_id - 1] = completed
				self._account_model_call(pending, -1)
				self._account_model_call(completed, 1)
			self._emit('model_end', call=completed)

	async def _start(self, url: str, settings: RunSettings):
		self._check_cancelled()
		from browser_use import Agent, BrowserSession, ChatDeepSeek
		from browser_use.llm.exceptions import ModelOutputFormatError

		self._check_cancelled()
		if self._chrome is not None:
			browser = await self._chrome.begin_case()
			mode = settings.execution_mode if settings.jev is not None and self._jev_dir is not None else 'direct'
			tools = _attached_tools(self._chrome.excluded_tools, mode)
		else:
			executable = os.getenv('BROWSER_USE_EXECUTABLE_PATH')
			profile = settings.user_data_dir or str(Path(self._temp.name) / 'profile')
			options: dict[str, Any] = dict(
				user_data_dir=profile,
				keep_alive=True,
				enable_default_extensions=False,
				viewport=VIEWPORT,
				headless=settings.headless,
			)
			if executable:
				options['executable_path'] = executable
			browser = BrowserSession(**options)
			tools = None
		self._browser.native = browser
		execution = settings.execution
		llm = ChatDeepSeek(
			model=execution.model,
			api_key=execution.api_key,
			base_url=execution.base_url,
			timeout=execution.timeout,
			max_tokens=execution.max_tokens,
			thinking=False,
		)
		original = llm.ainvoke

		async def traced(messages, output_format=None, **kwargs):
			self._sync_planner_parse_feedback()
			request = dict(model=execution.model, messages=[message.model_dump(mode='json') for message in messages])
			role = self._model_role.get()
			kind = role or PROVIDER
			with self.model_call(kind, model=execution.model, request=request, **({'role': role} if role else {})) as record:
				try:
					reply = await original(messages, output_format=output_format, **kwargs)
				except ModelOutputFormatError as error:
					failure = error.trace_record(role=role or 'planner', model=execution.model)
					record['parse_failures'] = [failure]
					if role == 'planner':
						with self._model_call_lock:
							self._planner_parse_feedback = (record['id'], failure)
					if error.usage is not None:
						record['response'] = {'usage': error.usage.model_dump(mode='json', exclude_none=True)}
					raise
				record['response'] = reply.model_dump(mode='json', exclude_none=True)
				return reply

		async def check_stop():
			self._check_cancelled()
			return False

		object.__setattr__(llm, 'ainvoke', traced)
		jev_config = {}
		if settings.jev is not None and self._jev_dir is not None:
			from browser_use.agent.jev import JevSettings

			assert settings.execution_mode in ('isolated', 'collaborative')
			# The baseline deadline started before this constructor and includes browser setup.
			budget = None if self._deadline is None else self._deadline - time.monotonic()
			if budget is not None and budget <= 0:
				raise RunCancelled('timeout')
			jev_config['jev'] = JevSettings.model_validate(
				{
					'mode': settings.execution_mode,
					'screenshot_policy': settings.screenshot_policy,
					'vision_escalation': settings.vision_escalation is not False,
					'candidate_pipeline': settings.candidate_pipeline,
					'min_read_operation_probability': settings.min_read_operation_probability,
					'jev_base_url': settings.jev.base_url,
					'jev_model': settings.jev.model,
					'jev_api_key': settings.jev.api_key,
					'trace_path': self._jev_dir / 'jev_trace.jsonl',
					'capture_dir': self._jev_dir / 'jev_captures',
					'time_budget_seconds': budget,
					**settings.jev_probability_options(),
				}
			)
		self._native = Agent(
			task=self._state['goal'],
			llm=llm,
			browser_session=browser,
			tools=tools,
			use_judge=False,
			directly_open_url=False,
			enable_signal_handler=False,
			register_external_agent_status_raise_error_callback=check_stop,
			use_vision=False,
			max_actions_per_step=1,
			enable_planning=False,
			file_system_path=str(Path(self._temp.name) / 'agent-files'),
			**jev_config,
		)
		if executor := getattr(self._native, '_jev', None):
			self._trace_jev_calls(executor)
		self._check_cancelled()
		self._browser_start_dispatched = True
		if self._chrome is None:
			await browser.start()
		self._check_cancelled()
		await browser.navigate_to(url)
		self._check_cancelled()
		await self._browser.bind()

	def _trace_jev_calls(self, executor):
		"""Capture actual requests on this run's native clients; the executor owns costs and roles."""
		from browser_use.agent.jev.backends import System1Error, image_rejection_status
		from browser_use.agent.jev.decision import request_body

		invoke = executor.usage.invoke

		async def with_role(step, role, llm, messages, **kwargs):
			token = self._model_role.set(role)
			try:
				return await invoke(step, role, llm, messages, **kwargs)
			finally:
				self._model_role.reset(token)

		executor.usage.invoke = with_role
		ask = executor.backend.ask
		model = self._state['system1_model']

		async def capture_jev(state, questions, *, images=None):
			request = request_body(state, questions, model, with_image=bool(images))
			image_files = {}
			if images:
				assert self._jev_dir is not None
				name = f'system1_images/{executor.step}-{len(self._state["model_calls"]) + 1}.jpg'
				path = self._jev_dir / name
				path.parent.mkdir(exist_ok=True)
				path.write_bytes(base64.b64decode(images['page-screenshot'], validate=True))
				image_files['page-screenshot'] = name
			trigger = executor.current_vision_trigger if images else None
			escalated_from = None
			if trigger and trigger.startswith('escalation:') and self._prediction_text_call is not None:
				first = self._prediction_text_call
				self._update_model_call(first['id'], outcome='escalated', escalation_from=trigger.removeprefix('escalation:'))
				escalated_from = first['id']
			with self.model_call(
				'jev',
				role='system1',
				model=model,
				request=request,
				bu_step=executor.step,
				request_images=image_files,
				request_format='multipart/form-data' if images else 'application/json',
				vision=bool(images),
				vision_trigger=trigger,
				**({'escalated_from_call_id': escalated_from} if escalated_from is not None else {}),
			) as record:
				if not images:
					self._prediction_text_call = record
				try:
					reply = await ask(state, questions, images=images)
				except BaseException as error:
					record['attempts'] = getattr(error, 'attempts', getattr(executor.backend, 'last_attempts', []))
					if images and isinstance(error, System1Error) and image_rejection_status(error.attempts) is not None:
						record['vision_rejected'] = True
					raise
				record['attempts'] = reply.attempts
				record['response'] = {
					'model': reply.model,
					'answers': json.loads(reply.model_dump_json(include={'answers'}))['answers'],
					'usage': {'input_tokens': reply.input_tokens, 'output_tokens': reply.output_tokens},
				}
				return reply

		executor.backend.ask = capture_jev

	def request_image(self, reference: str | None) -> bytes:
		"""Read the JPEG bound to this live run's exact model request."""
		if reference and self._jev_dir is not None:
			for call in self._state['model_calls']:
				name = call.get('request_images', {}).get('page-screenshot')
				if name and call['request']['state']['page'].get('screenshot') == reference:
					return (self._jev_dir / name).read_bytes()
		raise ValueError('Unknown Jev screenshot')

	def _observation_recovery_exhausted(self) -> bool:
		return self._observe_failures >= OBSERVATION_FAILURE_LIMIT or (
			self._observe_failed_since is not None
			and time.monotonic() - self._observe_failed_since >= OBSERVATION_RECOVERY_TIMEOUT
		)

	def _sample_with_retries(self, phase: str, cancel: Callable[[], str | None] | None):
		started = self._observe_failed_since if self._observe_failed_since is not None else time.monotonic()
		deadline = started + OBSERVATION_RECOVERY_TIMEOUT
		for retry in range(OBSERVATION_RETRIES + 1):
			try:
				# Each attempt binds to the currently focused tab before reading its document.
				page = self._browser.observe(cancel=cancel, timeout=max(0, deadline - time.monotonic()))
				sample = self._probe.inspect(self._browser.evaluate, page) if self._on_event and self._probe else {}
			except PageObservationUnavailable as error:
				self._observe_failures += 1
				self._observe_failed_since = started
				if self._observation_recovery_exhausted() or retry == OBSERVATION_RETRIES:
					raise
				delay = min(OBSERVATION_RETRY_DELAY, max(0, deadline - time.monotonic()))
				self._emit(
					'observation_retry',
					phase=phase,
					attempt=self._observe_failures,
					target_id=error.target_id,
					reason=str(error),
				)
				self._bridge.run(asyncio.sleep(delay), cancel=cancel, timeout=OBSERVATION_RETRY_DELAY + 1)
			else:
				self._observe_failures = 0
				self._observe_failed_since = None
				return page, sample
		raise AssertionError('Observation retry loop must return a sample or raise its last error')

	def observe(self, phase: str = 'step', reason: str | None = None):
		cancel = self._should_cancel if phase != 'final' and self._state['status'] == 'ready' else None
		page, sample = self._sample_with_retries(phase, cancel)
		self._state.pop('observation_error', None)
		metadata: dict[str, Any] = {'stale_reason': reason} if reason else {}
		self._state['page'] = page
		self._state['observation_index'] += 1
		if self._on_event:
			try:
				preview = Preview(self._browser.screenshot(timeout=15), None, sample)
			except Exception as error:
				preview = Preview(None, str(error), sample)
			self._emit('observation', preview=preview, phase=phase, **metadata)
			return

		image = self._browser.screenshot()
		page['screenshot'] = image
		self._state.setdefault('observations', []).append(
			dict(
				page=page,
				phase=phase,
				**metadata,
				step=len(self._state['history']),
				elements=indexed_elements(page['actions']),
				event_seq=self._state['event_seq'] + 1,
			)
		)
		self._emit('observation', phase=phase, **metadata)
		if self._record_dir:
			self._record_dir.mkdir(parents=True, exist_ok=True)
			if phase == 'initial':
				self._record_started = time.monotonic()
			elapsed = round((time.monotonic() - self._record_started) * 1000)
			(self._record_dir / f'{elapsed:06d}.jpg').write_bytes(base64.b64decode(image, validate=True))

	async def _predict(self, *, retry_timeout: bool = True):
		from browser_use.agent.views import AgentStepInfo

		agent = self._native
		if agent is None:
			raise RuntimeError('Browser Use agent is unavailable')
		agent.step_start_time = time.time()
		browser = self._browser.native
		assert browser is not None
		before = await read_document(browser)
		summary = await agent._prepare_context(AgentStepInfo(agent.state.n_steps - 1, MAX_STEPS))
		document = await read_document(browser)
		# BrowserStateSummary has a URL but no document identity. A navigation
		# during native capture makes that observation ambiguous even at the same URL.
		stable = same_document(before, document) and same_document(document, {**document, 'url': summary.url})
		document['url'] = summary.url
		self._summary = summary
		agent.state.last_model_output = agent.state.last_result = None
		# Only a text call from this prediction can be the first half of an image retry.
		self._prediction_text_call = None
		try:
			await agent._get_next_action(summary)
		except TimeoutError:
			if not retry_timeout or agent._jev is None:
				raise
			# No action has been dispatched. Reobserve before the single prediction retry.
			self._emit('prediction_retry', reason='model_timeout')
			return await self._predict(retry_timeout=False)
		model_output = agent.state.last_model_output
		if model_output is None:
			raise RuntimeError('Browser Use did not return a decision')
		actions = model_output.action
		if len(actions) != 1:
			raise ValueError('Expected one Browser Use action per inspector step')
		raw = actions[0].model_dump(exclude_none=True)
		operation, args = next(iter(raw.items()))
		# Native indices and the inspector's indices enumerate different trees.
		# Match DOM identity before drawing a target on the inspector screenshot.
		target = ''
		index = args.get('index')
		node = summary.dom_state.selector_map.get(index) if isinstance(index, int) else None
		if node is not None:
			try:
				obj = await self._browser.send('DOM.resolveNode', {'backendNodeId': node.backend_node_id})
				object_id = obj['object']['objectId']
				try:
					result = await self._browser.send(
						'Runtime.callFunctionOn',
						{
							'objectId': object_id,
							'returnByValue': True,
							'functionDeclaration': 'function(){return window.__benchmarkObserver?.ids?.get(this) ?? null}',
						},
					)
				finally:
					await self._browser.send('Runtime.releaseObject', {'objectId': object_id})
				identity = result.get('result', {}).get('value')
				elements = indexed_elements(self._state['page']['actions'], include_node_refs=True)
				target = next((e['index'] for e in elements if e['node_ref'] == identity), '')
			except (RuntimeError, KeyError):
				pass  # Closed frame or remote child target cannot be annotated on this screenshot.
		return raw, operation, target, target_guard(node) if node is not None else None, document, stable

	async def _act(self):
		agent = self._native
		if agent is None:
			raise RuntimeError('Browser Use agent is unavailable')
		await agent._execute_actions()
		await agent._post_process()
		await agent._finalize(self._summary)
		return [r.model_dump(mode='json', exclude_none=True) for r in (agent.state.last_result or [])]

	def command(self, name, body=None):
		if name not in {'predict', 'act', 'tick'}:
			raise ValueError(f'Unknown Browser Use command: {name}')
		try:
			self._check_cancelled()
			if self._state['status'] in {'done', 'blocked'}:
				return self.snapshot()
			if self._pending_observation is not None:
				try:
					self.observe(*self._pending_observation)
				except PageObservationUnavailable:
					if self._observation_recovery_exhausted():
						raise
				else:
					self._pending_observation = None
				return self.snapshot()
			if name in {'predict', 'tick'} and self._state['decision'] is None:
				self._choose()
			if name in {'act', 'tick'}:
				self._execute((body or {}).get('fingerprint'))
		except RunCancelled:
			self._state.update(decision=None, status='ready')
			raise
		finally:
			self._state['elapsed_ms'] = round((time.monotonic() - self._started) * 1000)
		return self.snapshot()

	def _observe_after_change(self, phase: str, reason: str | None = None):
		try:
			self.observe(phase, reason)
		except (PageObservationUnavailable, BrowserUnavailable) as error:
			if isinstance(error, BrowserUnavailable) and self._state['status'] == 'ready':
				raise
			failure = {'phase': phase, 'reason': str(error), 'target_id': getattr(error, 'target_id', None)}
			self._state['observation_error'] = failure
			self._emit('observation_error', **failure)
			if self._state['status'] == 'ready':
				if self._observation_recovery_exhausted():
					raise
				self._pending_observation = (phase, reason)

	def _choose(self):
		if self._state['decision_rounds'] >= MAX_STEPS:
			raise RunCancelled('timeout')
		calls_before = len(self._state['model_calls'])
		try:
			raw, operation, target, guard, document, stable = self._bridge.run(
				self._predict(), cancel=self._should_cancel, timeout=150
			)
		except TimeoutError as error:
			if str(error):
				raise
			raise TimeoutError('Browser Use prediction timed out before dispatch') from error
		executor = getattr(self._native, '_jev', None)
		source = executor.step_source() if executor is not None else 'llm'
		role = 'system1' if source == 'system1' else 'planner'
		calls = self._state['model_calls'][calls_before:]
		call = (
			next((c for c in reversed(calls) if c.get('role') == role), None)
			if executor is not None
			else next((c for c in reversed(calls) if c['kind'] == PROVIDER), None)
		)
		decision = dict(
			operation=operation,
			target=target,
			choice=operation,
			native_action=raw,
			target_guard=guard,
			document_guard=document,
			document_stable_during_capture=stable,
			requires_fresh_target=executor.has_reviewed_click() if executor is not None else False,
			provider='jev' if source == 'system1' else PROVIDER,
			decision_source=source,
			model_call_id=call['id'] if call is not None else None,
			fingerprint=self._state['page']['fingerprint'],
			observation_index=self._state['observation_index'],
			probabilities={},
		)
		self._state['decisions'].append(decision)
		self._state['decision_rounds'] += 1
		self._state.update(decision=decision, status='predicted')
		if call is not None:
			if source == 'system1':
				assert executor is not None
				answers = call['response']['answers']
				selected = answers['operation']
				target_answer = answers.get(selected['choice'].lower() + '_target') or {}
				decision.update(
					system1_operation=selected['choice'],
					operation_probabilities=selected.get('probabilities', {}),
					target_confidence=target_answer.get('confidence'),
					native_target_key=target_answer.get('choice'),
					native_target_probabilities=target_answer.get('probabilities'),
					latency_ms=call['latency_ms'],
				)
				if (reviewed_target := executor.planner_reviewed_click_key()) is not None:
					decision.update(
						native_target_key=reviewed_target,
						target_confidence=None,
						target_review_model_call_id=next(
							(c['id'] for c in reversed(calls) if c.get('role') == 'planner' and c['status'] == 'success'), None
						),
					)
			self._update_model_call(call['id'], decision=decision, operation=operation, target=target)
		self._emit('decision', decision=decision)

	def _execute(self, expected_fingerprint):
		decision = self._state['decision']
		if decision is None:
			raise ValueError('Predict a Browser Use action first')
		self._check_cancelled()
		page = decision['document_guard']
		if expected_fingerprint is not None and expected_fingerprint != decision['fingerprint']:
			reason = 'ui_out_of_sync'
		elif not decision['document_stable_during_capture']:
			reason = 'document_changed'
		else:

			async def check():
				browser = self._browser.native
				assert browser is not None
				return await decision_validity(browser, decision['target_guard'], page, await read_document(browser))

			reason = self._bridge.run(check(), cancel=self._should_cancel, timeout=20)
		disposition = stale_disposition(len(self._stale_streak), reason)
		if decision.get('requires_fresh_target') and reason is not None and disposition in {'execute', 'force'}:
			disposition = 'retry'
		if disposition != 'execute':
			assert reason is not None
			reasons = self._state['stale_reasons']
			reasons[reason] = reasons.get(reason, 0) + 1
			if disposition != 'ui_sync':
				self._stale_streak.append(reason)
		if disposition in {'retry', 'ui_sync'}:
			if disposition == 'retry':
				self._state['stale_retries'] += 1
			decision['stale_reason'] = reason
			if (call_id := decision.get('model_call_id')) is not None:
				self._update_model_call(call_id, decision=decision)
			self._state.update(decision=None, status='ready')
			self._observe_after_change('stale_retry', reason=reason)
			return
		self._check_cancelled()
		operation = decision['operation']
		action = dict(
			kind=operation,
			operation=operation,
			decision_source=decision.get('decision_source'),
			model_call_id=decision.get('model_call_id'),
			target=decision['target'],
			label=decision['target'],
			native_action=decision['native_action'],
			source_url=page['url'],
			source_document=page.get('document_id'),
			step=len(self._state['history']) + 1,
		)
		if disposition == 'force':
			self._state['forced_after_stale'] += 1
			action.update(forced_after_stale=True, stale_reasons=list(self._stale_streak))
		elif reason == 'target_unverifiable':
			action['target_unverifiable'] = True
		self._stale_streak.clear()
		self._emit('action_started', action=action)
		self._state.update(decision=None, status='ready')
		try:
			results = self._bridge.run(self._act(), cancel=self._should_cancel, timeout=150)
		except BaseException as error:
			self._emit('action_result', action=action, error=str(error) or type(error).__name__)
			raise
		result = results[-1] if results else {}
		action.update(result=results, elapsed_ms=round((time.monotonic() - self._started) * 1000))
		if operation not in {'done', 'finish'}:
			self._state['history'].append(action)
		self._emit('action_executed', action=action)
		if result.get('is_done'):
			answer = result.get('extracted_content') or result.get('long_term_memory') or ''
			self._state.update(
				status='blocked' if result.get('success') is False else 'done', plan_index=1, result={'markdown': answer}
			)
		elif not results:
			self._state.update(status='blocked', stop_reason='Browser Use returned no action result')
		self._emit('action_result', action=action, result=results)
		if result.get('is_done'):
			self.finish_execution()
		self._observe_after_change('step')

	def close(self):
		if self._closed:
			return
		self._closed = True
		try:
			if self._browser.native is not None:
				self._bridge.run(self._close_browser(), timeout=25)
		finally:
			if self._chrome is None:
				self._bridge.close()
			self._temp.cleanup()

	def record_timeout_evidence(self):
		"""Expose captured material without assigning a successful result to an unfinished run."""
		executor = getattr(self._native, '_jev', None)
		captures = executor.captures.records if executor is not None else []
		lines = ['# 任务超时后的部分证据', '', '下列内容是在中断前保存的页面材料。剩余条件和最终结果尚未确认。']
		for page in captures[-8:]:
			lines.extend(['', f'## {page.title or "已保存页面"}', '', page.url, '', page.summary or page.text[:1500]])
		if page := self._state.get('page'):
			if page.get('text'):
				lines.extend(['', '## 最后一次网页采样', '', page['url'], '', page['text'][:1500]])
		self._state['partial_result'] = {'markdown': '\n'.join(lines)}
		self._emit('partial_result')

	def finish_execution(self):
		"""Flush Jev's last action outcome and role totals while the test page is still observable."""
		executor = getattr(self._native, '_jev', None)
		if executor is not None:
			self._bridge.run(executor.aclose(), timeout=20)

	async def _close_browser(self):
		try:
			if self._native is not None:
				if executor := getattr(self._native, '_jev', None):
					await executor.aclose()
				await self._native.eventbus.stop(clear=True, timeout=1)
		finally:
			browser = self._browser.native
			if self._chrome is not None:
				await self._chrome.finish_case()
			elif self._browser_start_dispatched and browser is not None:
				await browser.kill()
