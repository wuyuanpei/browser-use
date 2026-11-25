"""Run bounded System1 subgoals inside Browser Use's native Agent step loop."""

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from browser_use.agent.jev import decision, rules, state
from browser_use.agent.jev.backends import JevBackend, System1Error, image_rejection_status, make_backend
from browser_use.agent.jev.checks import CheckGate
from browser_use.agent.jev.helpers import ModelHelpers
from browser_use.agent.jev.prompts import render_handback
from browser_use.agent.jev.research import CaptureStore
from browser_use.agent.jev.trace import JevTrace
from browser_use.agent.jev.usage import Role, UsageLedger
from browser_use.agent.jev.views import (
	JEV_INPUT_USD_PER_MTOK,
	MAX_REQUEST_CHARS,
	NO_ACTION_HANDBACK_LIMIT,
	NO_EFFECT_LIMIT,
	PAGE_TEXT_LIMIT,
	REPEAT_LIMIT,
	REPEAT_WINDOW,
	SUBGOAL_FIELD_LIMITS,
	ActionRecord,
	Candidate,
	CollectPageParams,
	DelegateSubgoalParams,
	DelegateSubgoalParamsSmart,
	FinishParams,
	Handback,
	HandbackReason,
	JevSettings,
	Observation,
	Operation,
	SubgoalRun,
	System1Reply,
)
from browser_use.agent.jev.vision import PreparedImage, prepare_screenshot
from browser_use.agent.views import ActionResult, AgentOutput
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage, UserMessage
from browser_use.utils import get_browser_use_version

if TYPE_CHECKING:
	from browser_use.agent.service import Agent

logger = logging.getLogger(__name__)
_SOURCE = Literal['system1', 'llm']
Stop = tuple[HandbackReason, str]


@dataclass
class RouteResult:
	model_output: AgentOutput | None = None
	context_messages: list[UserMessage] = field(default_factory=list)


@dataclass
class _System1Request:
	state: dict[str, Any]
	questions: dict[str, dict[str, Any]]
	model: str | None
	image: PreparedImage | None
	stats: dict[str, Any]


def _is_review_delegation(previous: SubgoalRun | None, rejected: bool) -> bool:
	return previous is not None and (rejected or (previous.handback is not None and previous.handback.reason != 'step_done'))


def _is_repeated_empty(runs: list[SubgoalRun], identity: str, vision: bool = False) -> bool:
	recent = runs[-NO_ACTION_HANDBACK_LIMIT:]
	return len(recent) == NO_ACTION_HANDBACK_LIMIT and all(
		(r.identity, r.vision) == (identity, vision) and not r.executed_actions for r in recent
	)


class JevExecutor:
	def __init__(self, agent: 'Agent', settings: JevSettings):
		self.agent = agent
		self.settings = settings
		self.trace = JevTrace(Path(settings.trace_path or agent.agent_directory / 'jev_trace.jsonl'), agent.id)
		self.gate = CheckGate(settings.checks, self.trace)
		self.runs: list[SubgoalRun] = []
		self._first_run = 0
		self.task_actions: list[ActionRecord] = []
		self.last_observation: Observation | None = None
		self._routed_target: tuple[int, str | None] | None = None
		self.observations: list[Observation] = []
		self.pending_record: ActionRecord | None = None
		self._record_target: Candidate | None = None
		self._handback_message: str | None = None
		self._source: _SOURCE = 'llm'
		self._planner_attempt = 0
		self._planner_page_action = False
		self._vision_disabled: dict[str, Any] | None = None
		self.current_vision_trigger: str | None = None
		self._closed = False
		self.finish_only = False
		self.defer_final_answer = False
		self.finalized_step: int | None = None
		self._finalize_lock = asyncio.Lock()
		self.usage = UsageLedger(agent.token_cost_service, self.trace)
		self.helpers = ModelHelpers(settings, agent.llm, self.invoke_llm, agent.settings.llm_timeout or 60)
		for model in self.helpers.models.values():
			agent.token_cost_service.register_llm(model)
		if settings.system1_llm is not None:
			agent.token_cost_service.register_llm(settings.system1_llm)
		self.backend = self._make_backend()
		self.captures = CaptureStore(Path(settings.capture_dir or agent.agent_directory / 'jev_captures'), agent.id)
		self.trace.write(
			'run_config',
			mode=settings.mode,
			system1=settings.system1,
			jev_model=settings.jev_model,
			jev_base_url=settings.jev_base_url,
			checks={k: settings.checks.mode(k) for k in ('risky_action', 'submit', 'completion', 'confidence', 'scope')},
			models={k: v.model for k, v in self.helpers.models.items()},
			system1_llm=settings.system1_llm.model if settings.system1_llm else None,
			bu_version=get_browser_use_version(),
			**settings.model_dump(
				include={
					'min_operation_probability',
					'min_target_probability',
					'unable_probability_handback',
					'default_max_actions',
					'max_candidates',
					'answer_timeout',
					'jev_timeout',
					'screenshot_policy',
					'vision_max_side',
					'vision_jpeg_quality',
				}
			),
		)

	@property
	def step(self) -> int:
		return self.agent.state.n_steps

	@property
	def active_run(self) -> SubgoalRun | None:
		return self.runs[-1] if self.runs and self.runs[-1].status == 'active' else None

	def step_source(self) -> _SOURCE:
		return self._source

	def _make_backend(self):
		return make_backend(
			self.settings, self.settings.jev_api_key or os.getenv('TYPESAFE_API_KEY'), self.agent.llm, invoke=self._invoke_system1
		)

	def on_new_task(self):
		"""Start a new delegation history, reopening the backend after a completed run."""
		self._first_run = len(self.runs)
		if not self._closed:
			return
		if self.active_run:
			self.active_run.status = 'handed_back'
		self._handback_message = None
		self.last_observation = None
		self._planner_page_action = False
		self.finish_only = False
		self.backend = self._make_backend()
		self._closed = False

	def last_step_planner_page_action(self) -> bool:
		return self._planner_page_action

	def usage_by_role(self) -> dict[str, dict]:
		return self.usage.summary()

	async def invoke_llm(self, role: Role, llm: BaseChatModel, messages: list[BaseMessage], **kwargs: Any):
		if role == 'planner':
			self._planner_attempt += 1
			kwargs.setdefault('attempt', self._planner_attempt)
		return await self.usage.invoke(self.step, role, llm, messages, **kwargs)

	async def _invoke_system1(self, llm, messages, **kwargs):
		return await self.invoke_llm('system1', llm, messages, **kwargs)

	def register_actions(self):
		registry = self.agent.tools.registry
		delegate_model = DelegateSubgoalParamsSmart if self.settings.screenshot_policy == 'smart' else DelegateSubgoalParams
		models = {'delegate_subgoal': delegate_model, 'finish': FinishParams, 'collect_page': CollectPageParams}
		for name, model in models.items():
			existing = registry.registry.actions.get(name)
			if name in registry.exclude_actions or (existing and existing.param_model is not model):
				raise ValueError(f'Jev action "{name}" is excluded or conflicts with a registered action')

		@registry.action(
			'Hand one stage of the task to System1, which performs all page actions of that stage.',
			param_model=delegate_model,
			terminates_sequence=True,
		)
		async def delegate_subgoal(params: DelegateSubgoalParams, browser_session: BrowserSession):
			return await self.delegate(params, browser_session)

		@registry.action(
			'End the task. A separate helper writes the final answer from the saved pages and the history.',
			param_model=FinishParams,
			terminates_sequence=True,
		)
		async def finish(params: FinishParams):
			return ActionResult(
				is_done=True,
				success=params.success,
				extracted_content=params.reason,
				long_term_memory=f'Finished (success={params.success}): {params.reason}',
			)

		@registry.action('Internal action used only by System1: save the visible page text.', param_model=CollectPageParams)
		async def collect_page(params: CollectPageParams, browser_session: BrowserSession):
			return await self.collect(browser_session)

	def _refuse(self, reason: str, detail: str, identity: str) -> ActionResult:
		self.trace.write('delegate_refused', self.step, reason=reason, identity=identity)
		return ActionResult(error=detail)

	async def delegate(self, params: DelegateSubgoalParams, browser_session: BrowserSession) -> ActionResult:
		data = params.model_dump()
		truncated = []
		for name, limit in SUBGOAL_FIELD_LIMITS.items():
			if data.get(name) and len(data[name]) > limit:
				data[name] = data[name][:limit]
				truncated.append(name)
		data['max_actions'] = (
			self.settings.default_max_actions if params.max_actions is None else min(30, max(1, params.max_actions))
		)
		params = type(params).model_validate(data)
		identity = '\n'.join(' '.join(s.split()) for s in (params.objective, params.completion))
		previous = self.runs[-1] if len(self.runs) > self._first_run else None
		rejected = bool(
			previous and previous.identity == identity and previous.handback and previous.handback.reason == 'step_done'
		)
		if rejected and not (params.review_note or '').strip():
			return self._refuse(
				'review_note_missing',
				'review_note is required when you re-delegate the same subgoal after rejecting STEP_DONE.',
				identity,
			)
		policy = self.settings.screenshot_policy
		smart = params if policy == 'smart' and isinstance(params, DelegateSubgoalParamsSmart) else None
		declared = smart.screenshot_required if smart else None
		reason = smart.screenshot_reason if smart else None
		source: Literal['policy', 'planner'] | None = None
		if policy == 'always':
			source = 'policy'
		elif declared is True:
			source = 'planner'
		if _is_repeated_empty(self.runs[self._first_run :], identity, source is not None):
			return self._refuse(
				'no_action_twice',
				'This subgoal was handed back twice in a row without any executed action. Rewrite it, delegate something else first, or call finish.',
				identity,
			)
		run_id = f'sg{len(self.runs) + 1}'
		if params.start_url is not None or params.start_tab is not None:
			if params.start_url is not None:
				try:
					parsed = urlsplit(params.start_url)
					valid = parsed.scheme in {'http', 'https'} and bool(parsed.hostname)
				except ValueError:
					valid = False
				if not valid or params.start_tab is not None:
					return self._refuse('invalid_start', 'Choose one http(s) start_url or one open start_tab.', identity)
				name, arguments = 'navigate', {'url': params.start_url, 'new_tab': False}
			else:
				if not any(t.target_id[-4:] == params.start_tab for t in await browser_session.get_tabs()):
					return self._refuse('invalid_start', 'The requested start_tab is not open.', identity)
				name, arguments = 'switch', {'tab_id': params.start_tab}
			try:
				navigation = await self.agent.tools.registry.execute_action(name, arguments, browser_session=browser_session)
				error = navigation.error if isinstance(navigation, ActionResult) else None
			except Exception as exc:
				error = str(exc)[:300]
			self.trace.write(
				'planner_navigation',
				self.step,
				run_id=run_id,
				kind='url' if params.start_url else 'tab',
				url=params.start_url,
				has_query=bool(params.start_url and urlsplit(params.start_url).query),
				ok=not error,
				error=error,
			)
			if error:
				return ActionResult(error=error)
		url = await browser_session.get_current_page_url()
		run = SubgoalRun(
			id=run_id,
			params=params,
			identity=identity,
			max_actions=data['max_actions'],
			started_step=self.step,
			start_url=url,
			start_site=rules.site_of(url),
			vision=source is not None,
			vision_source=source,
			vision_reason=reason,
			review_delegation=_is_review_delegation(previous, rejected),
			screenshot_declared=declared,
			rejected_fingerprint=previous.handback.fingerprint if rejected and previous and previous.handback else None,
		)
		self.runs.append(run)
		previous_reason = previous.handback.reason if previous and previous.handback else None
		if policy == 'smart':
			if run.review_delegation and declared is None:
				self.trace.write(
					'screenshot_decision_missing', self.step, run_id=run.id, previous_handback_reason=previous_reason
				)
			if declared is not None and not (reason or '').strip():
				self.trace.write('screenshot_reason_missing', self.step, run_id=run.id, screenshot_required=declared)
		self.trace.write(
			'subgoal_start',
			self.step,
			run_id=run.id,
			**params.model_dump(exclude={'start_url', 'screenshot_required', 'screenshot_reason'}),
			truncated_fields=truncated,
			start_url=url,
			start_site=run.start_site,
			is_rejection_retry=rejected,
			screenshot_required=declared,
			screenshot_reason=reason,
			vision=run.vision,
			review_delegation=run.review_delegation,
			previous_handback_reason=previous_reason,
		)
		return ActionResult(long_term_memory=f'Delegated {run.id} to System1: {params.objective}')

	async def collect(self, session: BrowserSession) -> ActionResult:
		obs = self.last_observation
		if obs is not None and obs.step == self.step:
			page = {'url': obs.url, 'title': obs.title, 'text': obs.visible_text, 'links': obs.links}
		else:
			page = await state.read_visible_text(session)
		if not rules.content_lines(page['text']):
			return ActionResult(error='No visible page text to collect.')
		if self.captures.is_saved(page['url'], page['text']):
			return ActionResult(long_term_memory='The visible text is already saved; nothing new to collect.')
		run = self.active_run
		record = self.captures.add(run.id if run else None, self.step, page)
		try:
			record.summary = await self.helpers.summary(record.model_dump(include={'url', 'title', 'text', 'links'}))
			record.summary_status = 'ok'
		except Exception as exc:
			record.summary_status = 'failed'
			record.summary_error = type(exc).__name__
		self.captures.write(record)
		self.trace.write(
			'capture',
			self.step,
			capture_id=record.id,
			url=record.url,
			chars=len(record.text),
			file=str(self.captures.path(record)),
			summary_status=record.summary_status,
		)
		return ActionResult(long_term_memory=f'Saved capture {record.id}: {record.url} | {record.title}')

	def _planner(self, page: tuple[str, str] | None = None) -> RouteResult:
		self._source = 'llm'
		self._planner_attempt = 0
		self.usage.roles['planner'].decisions += 1
		context = self.captures.planner_state(
			self.settings.mode, self.settings.system1, self.runs[-1] if len(self.runs) > self._first_run else None, page
		)
		messages = [UserMessage(content=context)]
		if self._handback_message:
			messages.append(UserMessage(content=self._handback_message))
		return RouteResult(context_messages=messages)

	def _handback(
		self, run: SubgoalRun, stop: Stop, obs: Observation | None, choice: decision.Decision | None = None
	) -> RouteResult:
		reason, detail = stop
		result = Handback(
			reason=reason,
			detail=detail,
			step=self.step,
			fingerprint=obs.fingerprint if obs else None,
			operation=choice.operation if choice else None,
			operation_probability=choice.operation_probability if choice else None,
			target_label=choice.target.element()[:120] if choice and choice.target else None,
			target_probability=choice.target_probability if choice else None,
		)
		run.status = 'handed_back'
		run.handback = result
		self.trace.write(
			'handback',
			self.step,
			run_id=run.id,
			**result.model_dump(exclude={'step'}),
			actions_used=len(run.executed_actions),
			actions_discarded=sum(r.status == 'discarded' for r in run.actions),
		)
		self._handback_message = render_handback(
			run,
			_is_repeated_empty(self.runs[self._first_run :], run.identity, run.vision),
			screenshot_policy=self.settings.screenshot_policy,
			vision_disabled_status=self._vision_disabled['status'] if self._vision_disabled else None,
		)
		self._log_handback(run, result)
		return self._planner((obs.url, obs.visible_text) if obs else None)

	def _log_handback(self, run: SubgoalRun, result: Handback):
		logger.info('System1 %s handed control to the planner (%s): %s', run.id, result.reason, result.detail)

	def _precheck(self, run: SubgoalRun, obs: Observation) -> Stop | None:
		executed = run.executed_actions
		if self.finish_only:
			return 'step_limit', 'The run reached its step limit or its failure limit. Call finish now.'
		if executed and executed[-1].status in ('executed_failed', 'unknown'):
			return 'action_error', f'The last action failed or its effect is unknown: {executed[-1].error}'
		if self.agent.state.consecutive_failures > 0:
			return 'step_error', 'The previous step failed before its action finished. Review the current page.'
		if obs.is_pdf or obs.state_error:
			return 'unsupported_page', f'System1 cannot work on this page: {obs.state_error or "PDF"}.'
		site = rules.site_of(obs.url)
		if self.gate.evaluate(
			'scope', bool(site and run.start_site and site != run.start_site), f'{run.start_site} -> {site}', self.step
		):
			return 'out_of_scope', f'The page left the starting site ({run.start_site} -> {site}).'
		if run.no_effect_streak >= NO_EFFECT_LIMIT:
			return (
				'no_effect',
				f'The last {run.no_effect_streak} actions did not change the page. Consider a different route or control.',
			)
		if len(executed) >= run.max_actions:
			return 'action_budget', f'The action budget of this subgoal ({run.max_actions}) is used up. Review the result.'
		return None

	def _choice_check(self, run: SubgoalRun, obs: Observation, choice: decision.Decision) -> Stop | None:
		op, op_p, target_p = choice.operation, choice.operation_probability, choice.target_probability
		limit = self.settings.unable_probability_handback
		unable_p = choice.unable_probability
		if self.gate.evaluate(
			'confidence', unable_p is not None and unable_p > limit, f'UNABLE p={unable_p}, limit={limit}', self.step
		):
			return 'unable_probability', f'System1 gave UNABLE a probability of {unable_p}, above the limit {limit}.'
		if op == 'UNABLE':
			return 'unable', 'System1 reports that no offered operation can advance the completion condition.'
		if op == 'STEP_DONE':
			self.gate.evaluate(
				'completion', decision.completion_blocked(obs, run), 'STEP_DONE without the required page action.', self.step
			)
			return 'step_done', 'System1 requests review: it reports that the completion condition is visible.'
		low = (op_p is not None and op_p < self.settings.min_operation_probability) or (
			target_p is not None and target_p < self.settings.min_target_probability
		)
		if self.gate.evaluate('confidence', low, f'operation p={op_p}, target p={target_p}', self.step):
			return 'low_confidence', f'System1 was not confident: operation p={op_p}, target p={target_p}.'
		return None

	def _repeat(self, run: SubgoalRun, obs: Observation, choice: decision.Decision, text: str | None) -> bool:
		key = choice.target.key if choice.target else None
		return (
			sum(
				r.fingerprint_before == obs.fingerprint
				and r.operation == choice.operation
				and r.target_key == key
				and r.text == text
				for r in run.executed_actions[-REPEAT_WINDOW:]
			)
			>= REPEAT_LIMIT - 1
		)

	async def _action_check(self, run: SubgoalRun, obs: Observation, choice: decision.Decision) -> tuple[Stop | None, str | None]:
		op, target = choice.operation, choice.target
		repeated: Stop = ('repeated_action', 'System1 was about to repeat the same action on an unchanged page a third time.')
		if op != 'TYPE_TEXT' and self._repeat(run, obs, choice, None):
			return repeated, None
		executed = run.executed_actions
		if op == 'WAIT' and executed and executed[-1].operation == 'WAIT':
			return ('wait_limit', 'System1 chose WAIT twice in a row.'), None
		submission = op == 'PRESS_ENTER' or (op == 'CLICK' and target and target.is_submit)
		if self.gate.evaluate('submit', bool(submission and not run.params.may_submit), f'{op} requires may_submit', self.step):
			return (
				'submit_not_allowed',
				'System1 tried to submit but may_submit is false. Re-delegate with may_submit=true if submitting is required.',
			), None
		stop = await self._risky_action(run, obs, choice)
		if stop:
			return stop, None
		if op == 'TYPE_TEXT' and target:
			try:
				text = await self.helpers.text(self.agent.task, run, obs, target, [a for a in self.task_actions if a.executed])
			except Exception:
				text = None
			if text is None:
				return (
					'text_unavailable',
					f'The text helper could not determine a value for {target.element()}. Put the exact value into inputs, or obtain it first.',
				), None
			return (repeated if self._repeat(run, obs, choice, text) else None), text
		return None, None

	async def _risky_action(self, run: SubgoalRun, obs: Observation, choice: decision.Decision) -> Stop | None:
		if self.gate.is_off('risky_action'):
			return None
		target = choice.target
		effect = target.risky_effect if target and choice.operation == 'CLICK' else None
		label = target.element() if target else ''
		via = None
		if choice.operation == 'PRESS_ENTER' and run.executed_actions:
			key = run.executed_actions[-1].target_key
			target = next((c for c in obs.candidates.get('TYPE_TEXT', []) if c.key == key), None)
			inherited = rules.enter_effect(target, obs.candidates.get('CLICK', [])) if target else None
			if inherited and target:
				effect, control = inherited
				via = control.element()
				label = f'Press Enter in {target.element()}'
		if not effect or not target:
			return None
		detail = f'{effect}: {label}'
		if self.settings.checks.mode('risky_action') == 'record':
			self.gate.evaluate('risky_action', True, detail, self.step)
			return None
		if not run.params.may_submit:
			self.gate.evaluate('risky_action', True, detail, self.step)
			return 'risk_not_allowed', f'System1 chose an irreversible control ({detail}) but may_submit is false.'
		proposed = {'effect': effect, 'element': label, 'role': target.role, 'context': target.context}
		if via:
			proposed['via'] = via
		started = monotonic()
		error = None
		try:
			review = await self.helpers.authorize(
				self.agent.task, run, obs, proposed, [a for a in self.task_actions if a.executed]
			)
			allowed, reason = review.authorized, review.reason
		except Exception as exc:
			error = type(exc).__name__
			allowed, reason = False, f'Authorization failed: {error}'
		self.trace.write(
			'risk_authorization',
			self.step,
			run_id=run.id,
			trigger='enter' if via else 'click',
			effect=effect,
			target_label=label,
			authorized=allowed,
			reason=reason,
			latency_ms=round(1000 * (monotonic() - started)),
			error=error,
		)
		if not allowed:
			self.gate.evaluate('risky_action', True, detail, self.step)
			return 'risk_denied', f'You did not authorize {detail} ({reason}).'
		return None

	def _vision_wanted(self, run: SubgoalRun) -> bool:
		policy = self.settings.screenshot_policy
		return self._vision_disabled is None and (policy == 'always' or (policy == 'smart' and run.vision))

	def _system1_request(
		self,
		run: SubgoalRun,
		obs: Observation,
		task: str,
		registered: set[str],
		offered: list[Operation],
		research: dict | None,
		image: PreparedImage | None,
	) -> _System1Request:
		system_state = decision.build_state(
			obs, run, task, self.task_actions, research, vision=(str(uuid4()), image) if image else None
		)
		questions = decision.build_questions(obs, run, offered, research=research, registered_actions=registered, vision=image)
		counts = {name: len(q['criteria']) for name, q in questions.items() if name != 'operation'}
		model = self.settings.jev_model if self.backend.name == 'jev' else None
		stats: dict[str, Any] = {
			'run_id': run.id,
			'backend': self.backend.name,
			'questions': list(questions),
			'request_chars': decision.request_size(system_state, questions, model, with_image=image is not None),
			'offered_operations': offered,
			'candidate_counts': counts,
			'vision': image is not None,
			'vision_trigger': run.vision_source if image else None,
			'image_bytes': image.bytes if image else 0,
			'image_size': [image.width, image.height] if image else None,
			'dpr': image.scale if image else None,
			'rect_count': len(
				{key for name in counts for key, criterion in questions[name]['criteria'].items() if 'rect' in criterion}
			),
			'unselectable_options': obs.unselectable_options,
			'outcome': 'handback',
			'operation': None,
			'operation_probability': None,
			'target_label': None,
			'target_key': None,
			'target_probability': None,
			'returned_model': None,
			'unable_probability': None,
			'choice_is_argmax': None,
			'answers': {},
			'attempts': [],
			'bu_action': None,
			'input_tokens': None,
			'output_tokens': None,
			'cost_usd': None,
		}
		if self._vision_disabled is not None:
			stats['vision_disabled'] = True
		return _System1Request(system_state, questions, model, image, stats)

	async def _ask_once(self, request: _System1Request) -> System1Reply | None:
		"""Send this request, attributing every transport attempt to its text or image payload."""
		stats = request.stats
		start = monotonic()
		self.current_vision_trigger = stats['vision_trigger']
		try:
			if request.image is None:
				reply = await self.backend.ask(request.state, request.questions)
			else:
				reply = await self.backend.ask(
					request.state, request.questions, images={'page-screenshot': request.image.jpeg_b64}
				)
		except asyncio.CancelledError:
			elapsed_ms = round((monotonic() - start) * 1000)
			attempts = self.backend.last_attempts if isinstance(self.backend, JevBackend) else []
			if attempts:
				self.usage.record_jev(None, elapsed_ms, len(attempts), vision=request.image is not None)
			stats.update(outcome='interrupted', error='CancelledError', latency_ms=elapsed_ms, attempts=attempts)
			self._record_decision(request, None)
			raise
		except System1Error as exc:
			elapsed_ms = round((monotonic() - start) * 1000)
			stats.update(error=str(exc), attempts=exc.attempts, latency_ms=elapsed_ms)
			if self.backend.name == 'jev':
				self.usage.record_jev(None, elapsed_ms, max(1, len(exc.attempts)), vision=request.image is not None)
			return None
		finally:
			self.current_vision_trigger = None

		stats.update(
			returned_model=reply.model,
			input_tokens=reply.input_tokens,
			output_tokens=reply.output_tokens,
			latency_ms=reply.latency_ms,
			attempts=reply.attempts,
			answers=reply.answers,
		)
		if self.backend.name == 'jev':
			self.usage.record_jev(reply, reply.latency_ms, vision=request.image is not None)
			stats['cost_usd'] = (
				reply.input_tokens * JEV_INPUT_USD_PER_MTOK / 1_000_000 if reply.input_tokens is not None else None
			)
		return reply

	def _disable_vision(self, run: SubgoalRun, status: int) -> None:
		self.trace.write('vision_rejected', self.step, run_id=run.id, status=status)
		self.usage.record_vision_rejection(status, self.step)
		if self._vision_disabled is None:
			self._vision_disabled = {'status': status, 'step': self.step, 'run_id': run.id}
			self.trace.write('vision_disabled', **self._vision_disabled)

	async def route(self, agent: 'Agent', browser_state_summary: BrowserStateSummary) -> RouteResult:
		self._source = 'llm'
		self._routed_target = (self.step, agent.browser_session.agent_focus_target_id)
		await self.usage.sync_other(self.step)
		run = self.active_run
		if not run:
			# A collaborative planner may have navigated since the most recent System1 step.
			# Only an existing capture requires text comparison on a planner-only step.
			page = None
			if self.captures.records:
				try:
					current = await state.read_visible_text(agent.browser_session)
					page = (current['url'], current['text'])
				except Exception:
					logger.debug('Cannot compare the current viewport with previous captures', exc_info=True)
			return self._planner(page)
		if self.pending_record and self.pending_record.status == 'predicted':
			self.pending_record.status = 'discarded'
			self._emit_result()
		elif self.pending_record and self.pending_record.status == 'dispatched':
			self.pending_record.status = 'unknown'
			self.pending_record.error = 'No execution result was received'
		try:
			obs = await state.observe(agent.browser_session, browser_state_summary, self.settings, self.step)
		except Exception as exc:
			self._emit_result()
			return self._handback(run, ('unsupported_page', f'Could not observe the current page: {type(exc).__name__}'), None)
		self.last_observation = obs
		self.observations = [*self.observations[-11:], obs]
		await self._complete_record(run, obs, browser_state_summary)
		stop = self._precheck(run, obs)
		if stop:
			return self._handback(run, stop, obs)

		registered = set(agent.tools.registry.registry.actions)
		research = self.captures.research_state(obs)
		offered = decision.offered_operations(
			obs, run, registered, self.settings.checks, collect_available=bool(research and research['collect_available'])
		)
		image = None
		if self._vision_wanted(run):
			image = await asyncio.to_thread(
				prepare_screenshot,
				browser_state_summary.screenshot,
				obs.viewport_width,
				obs.viewport_height,
				max_side=self.settings.vision_max_side,
				quality=self.settings.vision_jpeg_quality,
			)
			if image is None:
				self.trace.write('vision_unavailable', self.step, run_id=run.id, vision_trigger=run.vision_source)
		request = self._system1_request(run, obs, agent.task, registered, offered, research, image)
		for name, count in request.stats['candidate_counts'].items():
			if count > self.settings.max_candidates:
				return self._handback(
					run,
					(
						'too_many_candidates',
						f'The page offers {count} targets for {name}, more than {self.settings.max_candidates}. Narrow this stage.',
					),
					obs,
				)
		size_error: Stop = ('state_too_large', 'The page is too large for one System1 request.')
		if request.stats['request_chars'] > MAX_REQUEST_CHARS:
			return self._handback(run, size_error, obs)

		self.usage.roles['system1'].decisions += 1
		run.decisions += 1
		reply = await self._ask_once(request)
		rejected = image_rejection_status(request.stats['attempts']) if reply is None and image is not None else None
		if rejected is not None:
			self._disable_vision(run, rejected)
			request.stats['outcome'] = 'vision_rejected'
			self._record_decision(request, None)
			request = self._system1_request(run, obs, agent.task, registered, offered, research, None)
			request.stats['fallback_from'] = 'vision_rejected'
			if request.stats['request_chars'] > MAX_REQUEST_CHARS:
				return self._handback(run, size_error, obs)
			reply = await self._ask_once(request)

		stats = request.stats
		if reply is None:
			detail = f'The System1 request failed: {stats["error"]}'
			if rejected is not None:
				detail += f' Screenshots were disabled after Jev rejected an image (HTTP {rejected}).'
			self._record_decision(request, None)
			return self._handback(run, ('system1_error', detail), obs)
		if request.image is not None:
			run.vision_decisions += 1
		choice = None
		text = None
		try:
			threshold = (
				self.settings.unable_probability_handback if self.settings.checks.mode('confidence') == 'enforce' else None
			)
			choice = decision.combine(reply, request.questions, obs, unable_threshold=threshold)
		except decision.InvalidAnswer as exc:
			stop = ('invalid_answer', f'System1 returned an invalid answer: {exc}')
			stats['choice_is_argmax'] = decision.argmax_flags(reply)
		else:
			target = choice.target
			stats.update(
				operation=choice.operation,
				operation_probability=choice.operation_probability,
				target_key=target.key if target else None,
				target_label=target.element()[:120] if target else None,
				target_probability=choice.target_probability,
				unable_probability=choice.unable_probability,
				choice_is_argmax=choice.choice_is_argmax,
			)
			stop = self._choice_check(run, obs, choice)
			if not stop:
				stop, text = await self._action_check(run, obs, choice)
		if stop:
			self._record_decision(request, reply)
			return self._handback(run, stop, obs, choice)
		assert choice is not None
		target = choice.target
		bu_action = decision.to_bu_action(choice.operation, target, text)
		action = agent.ActionModel.model_validate(bu_action)
		output = agent.AgentOutput(
			thinking=None,
			evaluation_previous_goal=None,
			next_goal=run.params.objective,
			memory=decision.memory_line(
				run.id, choice.operation, target, text, choice.operation_probability, choice.target_probability
			),
			action=[action],
		)
		record = ActionRecord(
			n=len(run.actions) + 1,
			step=self.step,
			run_id=run.id,
			operation=choice.operation,
			bu_action=bu_action,
			target_key=target.key if target else None,
			target_label=target.element()[:120] if target else None,
			text=text,
			fingerprint_before=obs.fingerprint,
			url_before=obs.url,
		)
		run.actions.append(record)
		self.task_actions.append(record)
		self.pending_record = record
		self._record_target = target
		self._source = 'system1'
		stats.update(outcome='execute', bu_action=bu_action)
		self._record_decision(request, reply)
		logger.info('System1 %s selected %s', run.id, output.memory)
		return RouteResult(model_output=output)

	def _record_decision(self, request: _System1Request, reply: System1Reply | None) -> None:
		stats = request.stats
		self.trace.write('system1_decision', self.step, **stats)
		directory = self.agent.settings.save_conversation_path
		if not directory:
			return
		suffix = '_vision_rejected' if stats['outcome'] == 'vision_rejected' else ''
		path = Path(directory) / f'system1_{self.agent.id}_{self.step}{suffix}.json'
		path.parent.mkdir(parents=True, exist_ok=True)
		record = {
			'request': decision.request_body(
				request.state, request.questions, request.model, with_image=request.image is not None
			),
			'reply': reply.model_dump() if reply else None,
			'decision': {k: v for k, v in stats.items() if k != 'answers'},
		}
		if request.image is not None:
			jpeg = path.with_suffix('.jpg')
			jpeg.write_bytes(base64.b64decode(request.image.jpeg_b64, validate=True))
			record['request_images'] = {'page-screenshot': jpeg.name}
		path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')

	def on_dispatch(self, agent: 'Agent'):
		record = self.pending_record
		if self._source == 'system1' and record and record.step == self.step and record.status == 'predicted':
			record.status = 'dispatched'

	def on_dispatch_error(self, agent: 'Agent', error: BaseException):
		record = self.pending_record
		if self._source != 'system1' or not record or record.step != self.step:
			return
		if record.status == 'dispatched':
			# This notification runs after the tools.act boundary; even cancellation may follow a committed action.
			record.status = 'unknown'
			record.error = f'{type(error).__name__}: action was interrupted after dispatch'
		elif record.status == 'predicted':
			record.status = 'discarded'
			self._emit_result()

	def on_executed(self, agent: 'Agent', results: list[ActionResult]):
		self._planner_page_action = False
		if self._source == 'llm':
			output = agent.state.last_model_output
			for i, result in enumerate(results):
				if not result.error:
					self._handback_message = None
					if output and i < len(output.action):
						name = next(iter(output.action[i].model_dump(exclude_unset=True)), '')
						if name not in {'delegate_subgoal', 'finish', 'done', 'collect_page'}:
							self._planner_page_action = True
			return
		record = self.pending_record
		if not record or record.step != self.step or record.status != 'dispatched':
			return
		if not results:
			record.status = 'unknown'
			record.error = 'The dispatched action returned no result'
			return
		result = results[-1]
		error = result.error
		name, args = next(iter(record.bu_action.items()))
		index = args.get('index')
		if (
			not error
			and name in ('click', 'input', 'select_dropdown')
			and (result.extracted_content or '').startswith(f'Element index {index} not available')
		):
			error = result.extracted_content
		if (
			not error
			and name == 'select_dropdown'
			and result.long_term_memory != f"Selected dropdown option '{args['text']}' at index {index}"
		):
			error = result.extracted_content or 'Dropdown selection failed'
		record.status = 'executed_failed' if error else 'executed_ok'
		record.error = error

	async def _complete_record(self, run: SubgoalRun, obs: Observation, summary: BrowserStateSummary):
		record = self.pending_record
		if not record:
			return
		target = self._record_target
		record.url_after = obs.url
		record.page_changed = obs.fingerprint != record.fingerprint_before
		if record.operation == 'SELECT' and target and target.index is not None and record.status == 'executed_ok':
			node = summary.dom_state.selector_map.get(target.index)
			# A different document may legally reuse a backend node id.
			if node is not None and rules.document_key(obs.url) == rules.document_key(record.url_before):
				current = await state.selected_option(self.agent.browser_session, node, live=True)
				if current is not None:
					n = int(target.key.rsplit(':', 1)[1])
					record.verified = current[0] == n and current[1] == target.option_text
					if not record.verified:
						record.status = 'executed_failed'
						record.error = f'The page did not keep option {n} "{target.option_text}"; current: {current}.'
		changes = record.operation in ('TYPE_TEXT', 'SELECT', 'PRESS_ENTER') or (
			record.operation == 'CLICK' and target and target.kind == 'element' and not target.editable
		)
		record.applied = record.status == 'executed_ok' and bool(changes)
		run.applied = run.applied or record.applied
		if record.executed and record.operation not in ('WAIT', 'COLLECT'):
			run.no_effect_streak = 0 if record.page_changed else run.no_effect_streak + 1
		self._emit_result()

	def _emit_result(self):
		record = self.pending_record
		if record:
			fields = record.model_dump(
				include={
					'run_id',
					'n',
					'operation',
					'target_label',
					'status',
					'error',
					'page_changed',
					'url_after',
					'verified',
					'applied',
				}
			)
			self.trace.write('action_result', self.step, record_step=record.step, **fields)
		self.pending_record = None
		self._record_target = None

	async def finalize_if_done(self, agent: 'Agent'):
		async with self._finalize_lock:
			await self._finalize_current(agent)

	async def _finalize_current(self, agent: 'Agent'):
		if not agent.history.history:
			return
		item = agent.history.history[-1]
		if not item.metadata or item.metadata.step_number == self.finalized_step or not item.result or not item.model_output:
			return
		result = item.result[-1]
		if not result.is_done or len(item.result) > len(item.model_output.action):
			return
		last = item.model_output.action[len(item.result) - 1].model_dump(exclude_unset=True)
		if 'finish' in last:
			trigger = 'finish'
		elif (
			'done' in last
			and self.settings.mode == 'collaborative'
			and self.captures.records
			and agent.output_model_schema is None
		):
			trigger = 'done_rewrite'
		else:
			return
		last_page = {'url': item.state.url, 'title': item.state.title, 'text': ''}
		try:
			page = await state.read_visible_text(agent.browser_session)
			focus = agent.browser_session.agent_focus_target_id
			target_matches = not self._routed_target or self._routed_target == (item.metadata.step_number, focus)
			if not last_page['url'] and focus and self._routed_target == (item.metadata.step_number, focus):
				# A newly focused tab can have an empty cached target URL while its live document is ready.
				last_page.update(url=page['url'], title=page['title'])
			if target_matches and page['url'] == last_page['url']:
				last_page['text'] = decision.page_text(page['text'], PAGE_TEXT_LIMIT)
		except Exception:
			logger.debug('Unable to read the final page', exc_info=True)
		reason = result.extracted_content or result.long_term_memory or ''
		recent = []
		for obs in self.observations:
			evidence = {'url': obs.url, 'title': obs.title, 'text': obs.visible_text[:1500]}
			if not recent or recent[-1] != evidence:
				recent.append(evidence)
		payload = {
			'original_goal': agent.task,
			'collected_pages': [r.model_dump(include={'id', 'url', 'title', 'text', 'summary'}) for r in self.captures.records],
			'action_history': [decision.action_state(a, include_run=True) for a in self.task_actions if a.executed],
			'observation_history': recent,
			'last_page': last_page,
			'completion_review': {'success': result.success, 'reason': reason},
		}
		error = None
		try:
			result.extracted_content = await self.helpers.answer(payload)
		except Exception as exc:
			error = type(exc).__name__
			result.success = False
			result.extracted_content = f'Final answer could not be generated ({error}). Planner reason: {reason}'
		if agent.state.last_result and agent.state.last_result[-1].is_done:
			agent.state.last_result[-1].extracted_content = result.extracted_content
			agent.state.last_result[-1].success = result.success
		self.finalized_step = item.metadata.step_number
		self.trace.write(
			'final_answer',
			item.metadata.step_number,
			trigger=trigger,
			ok=error is None,
			chars=len(result.extracted_content),
			error=error,
			last_page_url=last_page['url'],
		)

	async def aclose(self):
		if self._closed:
			return
		self._closed = True
		try:
			await self.backend.aclose()
		finally:
			if self.pending_record:
				if self.pending_record.status == 'predicted':
					self.pending_record.status = 'discarded'
				elif self.pending_record.status == 'dispatched':
					self.pending_record.status = 'unknown'
				self._emit_result()
			await self.usage.sync_other(self.step)
			roles = self.usage_by_role()
			costs = [r['cost_usd'] for r in roles.values()]
			self.trace.write(
				'usage_summary', self.step, roles=roles, total_cost_usd=sum(costs) if all(c is not None for c in costs) else None
			)
