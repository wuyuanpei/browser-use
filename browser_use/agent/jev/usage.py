"""Attribute actual model requests independently of planner decision rounds."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel

from browser_use.agent.jev.trace import JevTrace
from browser_use.agent.jev.views import JEV_INPUT_USD_PER_MTOK, System1Reply
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelOutputFormatError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeUsage
from browser_use.tokens.service import TokenCost

logger = logging.getLogger(__name__)
Role = Literal['planner', 'system1', 'text', 'summary', 'answer', 'risk', 'other']


_planner_feedback_scope = ContextVar('jev_planner_feedback_scope', default=False)


@contextmanager
def planner_feedback_scope():
	# The retry controller records rejected planner calls after it selects the feedback disposition.
	token = _planner_feedback_scope.set(True)
	try:
		yield
	finally:
		_planner_feedback_scope.reset(token)


class RoleUsage(BaseModel):
	decisions: int = 0
	calls: int = 0
	latency_ms: int = 0
	reported_usage_calls: int = 0
	missing_usage_calls: int = 0
	prompt_tokens: int = 0
	completion_tokens: int = 0
	cached_tokens: int = 0
	cost_usd: float | None = 0.0

	def record(self, *, prompt: int | None, completion: int | None, cached: int | None, latency: int, cost: float | None):
		self.calls += 1
		self.latency_ms += latency
		self.prompt_tokens += prompt or 0
		self.completion_tokens += completion or 0
		self.cached_tokens += cached or 0
		if prompt is None or completion is None:
			self.missing_usage_calls += 1
		else:
			self.reported_usage_calls += 1
		self.cost_usd = None if self.cost_usd is None or cost is None else self.cost_usd + cost


class UsageLedger:
	def __init__(self, token_cost: TokenCost, trace: JevTrace):
		self.cost = token_cost
		self.trace = trace
		self.cursor = len(token_cost.usage_history)
		self.vision_requests = 0
		self.vision_rejections = 0
		self.vision_disabled: dict[str, int] | None = None
		self.roles = {name: RoleUsage() for name in ('planner', 'system1', 'text', 'summary', 'answer', 'risk', 'other')}

	async def _price(self, model: str, usage: ChatInvokeUsage | None) -> float | None:
		if usage is None or not self.cost.include_cost:
			return None
		try:
			pricing = await self.cost.calculate_cost(model, usage)
			return pricing.total_cost if pricing else None
		except Exception:
			logger.warning('Pricing unavailable for %s', model)
			return None

	async def _record_llm(
		self,
		role: Role,
		model: str,
		usage: ChatInvokeUsage | None,
		step: int,
		start: float | None,
		ok: bool,
		error: str | None,
		attempt: int,
	):
		latency = round(1000 * (monotonic() - start)) if start is not None else 0
		prompt = usage.prompt_tokens if usage else None
		completion = usage.completion_tokens if usage else None
		cached = usage.prompt_cached_tokens if usage else None
		cost = await self._price(model, usage)
		self.roles[role].record(prompt=prompt, completion=completion, cached=cached, latency=latency, cost=cost)
		self.trace.write(
			'llm_call',
			step,
			role=role,
			model=model,
			ok=ok,
			error=error,
			latency_ms=latency if start is not None else None,
			attempt=attempt,
			prompt_tokens=prompt,
			completion_tokens=completion,
			cached_tokens=cached,
			cost_usd=cost,
		)

	async def sync_other(self, step: int):
		entries = self.cost.usage_history
		if self.cursor > len(entries):
			self.cursor = 0
		pending = entries[self.cursor :]
		self.cursor = len(entries)
		for entry in pending:
			await self._record_llm('other', entry.model, entry.usage, step, None, True, None, 1)

	async def _claim_usage(self, step: int, model: str, usage: ChatInvokeUsage | None):
		# Claim this response's entry. Unrelated calls still belong to 'other'.
		entries = self.cost.usage_history
		pending = entries[self.cursor :]
		self.cursor = len(entries)
		claimed = False
		for entry in reversed(pending):
			if not claimed and usage is not None and entry.model == model and entry.usage == usage:
				claimed = True
			else:
				await self._record_llm('other', entry.model, entry.usage, step, None, True, None, 1)

	async def record_format_error(
		self,
		step: int,
		role: Role,
		model: str,
		error: ModelOutputFormatError,
		*,
		start: float,
		attempt: int,
		planner_feedback_retry: bool = False,
	):
		record = error.trace_record(role=role, model=model, planner_feedback_retry=planner_feedback_retry)
		self.trace.write('llm_output_parse_error', step, **record)
		# TokenCost tracks successful returns. A rejected provider reply is billable as well.
		if error.usage is not None:
			self.cost.add_usage(model, error.usage)
		await self._claim_usage(step, model, error.usage)
		await self._record_llm(role, model, error.usage, step, start, False, type(error).__name__, attempt)

	async def invoke(
		self, step: int, role: Role, llm: BaseChatModel, messages: list[BaseMessage], *, attempt: int = 1, **kwargs: Any
	):
		await self.sync_other(step)
		start = monotonic()
		try:
			response = await llm.ainvoke(messages, **kwargs)
		except ModelOutputFormatError as error:
			if role != 'planner' or not _planner_feedback_scope.get():
				await self.record_format_error(step, role, llm.model, error, start=start, attempt=attempt)
			raise
		except BaseException as exc:
			await self._record_llm(role, llm.model, None, step, start, False, type(exc).__name__, attempt)
			raise
		await self._claim_usage(step, llm.model, response.usage)
		await self._record_llm(role, llm.model, response.usage, step, start, True, None, attempt)
		return response

	def record_jev(self, reply: System1Reply | None, elapsed_ms: int, attempts: int = 1, *, vision: bool = False):
		# A transport retry is a separate HTTP request. Only the final reply reports usage.
		attempts = max(1, len(reply.attempts)) if reply is not None else attempts
		if vision or (reply is not None and reply.image_count):
			self.vision_requests += attempts
		for i in range(max(1, attempts)):
			final = i == attempts - 1 and reply is not None
			prompt = reply.input_tokens if final and reply else None
			completion = reply.output_tokens if final and reply else None
			cost = prompt * JEV_INPUT_USD_PER_MTOK / 1_000_000 if prompt is not None else None
			self.roles['system1'].record(
				prompt=prompt, completion=completion, cached=None, latency=elapsed_ms if i == 0 else 0, cost=cost
			)

	def record_vision_rejection(self, status: int, step: int) -> None:
		self.vision_rejections += 1
		if self.vision_disabled is None:
			self.vision_disabled = {'status': status, 'step': step}

	def summary(self) -> dict[str, dict]:
		rows = {role: counters.model_dump() for role, counters in self.roles.items()}
		rows['system1'].update(
			vision_requests=self.vision_requests,
			text_requests=self.roles['system1'].calls - self.vision_requests,
			vision_rejections=self.vision_rejections,
			vision_disabled=self.vision_disabled.copy() if self.vision_disabled else None,
		)
		# Native BU does not publish an entry or its latency when no usage is returned.
		# Keep these observed totals usable without implying complete coverage for other.
		rows['other'].update(call_coverage='reported_usage_only', latency_coverage='unavailable')
		return rows
