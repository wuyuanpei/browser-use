"""Run Browser Use in an owned browser and publish benchmark observations."""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .models import PROVIDER, RunSettings, redact_secrets
from .observation import READ_STATE, document_key, fingerprint, indexed_elements
from .tasks import EnvironmentBlocked
from .token_usage import token_usage

if TYPE_CHECKING:
	from browser_use.agent.service import Agent
	from browser_use.browser.session import BrowserSession, CDPSession
	from browser_use.browser.views import BrowserStateSummary

	from .chrome_attachment import ChromeAttachment
	from .tasks import CaseProbe

MAX_STEPS = 100
VIEWPORT = {'width': 1366, 'height': 780}
BrowserMode = Literal['isolated', 'attached_chrome']


class RunCancelled(RuntimeError):
	"""Cooperative cancellation at a boundary before browser input."""

	is_cancellation = True


class BrowserUnavailable(EnvironmentBlocked):
	"""The shared browser cannot start another case until it recovers."""


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

	def run(self, operation, *, cancel=None, timeout=90):
		async def bounded():
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
			raise RuntimeError('Browser Use has no active page')
		bound = await browser.get_or_create_cdp_session(target, focus=False)
		self.bound = bound
		return bound.target_id

	async def send(self, method, params):
		browser = self.native
		bound = self.bound
		if browser is None or bound is None or browser.agent_focus_target_id != bound.target_id:
			raise RuntimeError('Browser target changed before sampling')
		active = await browser.get_or_create_cdp_session(bound.target_id, focus=False)
		if active.session_id != bound.session_id:
			raise RuntimeError('Browser session changed before sampling')
		group, name = method.split('.')
		command = getattr(getattr(bound.cdp_client.send, group), name)
		result = await command(params=params, session_id=bound.session_id)
		if browser.agent_focus_target_id != bound.target_id:
			raise RuntimeError('Browser target changed while sampling')
		return result

	def call(self, method, **params):
		timeout = params.pop('_response_timeout', 20)
		return self.bridge.run(self.send(method, params), timeout=timeout)

	def evaluate(self, expression: str) -> Any:
		result = self.call('Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
		if result.get('exceptionDetails'):
			raise RuntimeError(f'Browser evaluation failed: {result["exceptionDetails"].get("text", "")}')
		return result['result'].get('value')

	def observe(self) -> dict[str, Any]:
		target = self.bridge.run(self.bind())
		page = self.evaluate(READ_STATE)
		if not isinstance(page, dict):
			raise RuntimeError('Browser document is unavailable')
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
		chrome: ChromeAttachment | None = None,
	):
		self._secrets = (settings.execution.api_key, settings.evaluation.api_key)
		self._on_event, self._should_cancel, self._probe = on_event, should_cancel, probe
		self._record_dir = record_dir
		self._chrome = chrome
		self._state: dict[str, Any] = dict(
			goal=goal,
			system1_provider=PROVIDER,
			system1_model=settings.execution.model,
			system1_base_url=settings.execution.public()['base_url'],
			execution_mode='direct',
			screenshot_mode=False,
			screenshot_policy='off',
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
			elapsed_ms=0,
			event_seq=0,
			observation_index=-1,
			plan=[goal],
			plan_index=0,
			record=bool(record_dir),
			free_step_supported=True,
		)
		if scenario is not None:
			self._state['scenario'] = scenario
		self._started = time.monotonic()
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
		return redact_secrets(
			{
				**self._state,
				'token_usage': token_usage(self._state['model_calls']),
				'elements': indexed_elements(self._state['page'].get('actions', [])),
			},
			*self._secrets,
		)

	@contextmanager
	def model_call(self, kind: str, **metadata: Any) -> Iterator[dict[str, Any]]:
		"""Record one real execution or verification request with the same sequencing."""
		calls = self._state['model_calls']
		record = dict(
			id=len(calls) + 1,
			kind=kind,
			status='pending',
			observation_index=self._state['observation_index'],
			event_seq=self._state['event_seq'] + 1,
			**metadata,
		)
		calls.append(record)
		self._emit('model_start', call=record)
		started = time.perf_counter()
		try:
			yield record
			record['status'] = 'success'
		except BaseException as error:
			record.update(status='error', error=str(error) or type(error).__name__)
			raise
		finally:
			record['latency_ms'] = round((time.perf_counter() - started) * 1000)
			self._emit('model_end', call=record)

	async def _start(self, url: str, settings: RunSettings):
		self._check_cancelled()
		from browser_use import Agent, BrowserSession, ChatDeepSeek, Tools

		self._check_cancelled()
		if self._chrome is not None:
			browser = await self._chrome.begin_case()
			tools = Tools(exclude_actions=self._chrome.excluded_tools)
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
			request = dict(model=execution.model, messages=[message.model_dump(mode='json') for message in messages])
			with self.model_call(PROVIDER, model=execution.model, request=request) as record:
				reply = await original(messages, output_format=output_format, **kwargs)
				record['response'] = reply.model_dump(mode='json', exclude_none=True)
				return reply

		async def check_stop():
			self._check_cancelled()
			return False

		object.__setattr__(llm, 'ainvoke', traced)
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
		)
		self._check_cancelled()
		self._browser_start_dispatched = True
		if self._chrome is None:
			await browser.start()
		self._check_cancelled()
		await browser.navigate_to(url)
		self._check_cancelled()
		await self._browser.bind()

	def observe(self, phase: str = 'step'):
		page = self._browser.observe()
		self._state['page'] = page
		self._state['observation_index'] += 1
		if self._on_event:
			sample = self._probe.inspect(self._browser.evaluate, page) if self._probe else {}
			try:
				preview = Preview(self._browser.screenshot(timeout=15), None, sample)
			except Exception as error:
				preview = Preview(None, str(error), sample)
			self._emit('observation', preview=preview, phase=phase)
			return

		image = self._browser.screenshot()
		page['screenshot'] = image
		self._state.setdefault('observations', []).append(
			dict(
				page=page,
				phase=phase,
				step=len(self._state['history']),
				elements=indexed_elements(page['actions']),
				event_seq=self._state['event_seq'] + 1,
			)
		)
		self._emit('observation', phase=phase)
		if self._record_dir:
			self._record_dir.mkdir(parents=True, exist_ok=True)
			if phase == 'initial':
				self._record_started = time.monotonic()
			elapsed = round((time.monotonic() - self._record_started) * 1000)
			(self._record_dir / f'{elapsed:06d}.jpg').write_bytes(base64.b64decode(image, validate=True))

	async def _predict(self):
		from browser_use.agent.views import AgentStepInfo

		agent = self._native
		if agent is None:
			raise RuntimeError('Browser Use agent is unavailable')
		agent.step_start_time = time.time()
		summary = await agent._prepare_context(AgentStepInfo(agent.state.n_steps - 1, MAX_STEPS))
		self._summary = summary
		agent.state.last_model_output = agent.state.last_result = None
		await agent._get_next_action(summary)
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
		return raw, operation, target

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

	def _choose(self):
		if self._state['decision_rounds'] >= MAX_STEPS:
			raise RunCancelled('timeout')
		raw, operation, target = self._bridge.run(self._predict(), cancel=self._should_cancel, timeout=150)
		decision = dict(
			operation=operation,
			target=target,
			choice=operation,
			native_action=raw,
			provider=self._state['system1_provider'],
			fingerprint=self._state['page']['fingerprint'],
			observation_index=self._state['observation_index'],
			probabilities={},
		)
		self._state['decisions'].append(decision)
		self._state['decision_rounds'] += 1
		self._state.update(decision=decision, status='predicted')
		if self._state['model_calls']:
			self._state['model_calls'][-1].update(decision=decision, operation=operation, target=target)
		self._emit('decision', decision=decision)

	def _execute(self, expected_fingerprint):
		decision = self._state['decision']
		if decision is None:
			raise ValueError('Predict a Browser Use action first')
		self._check_cancelled()
		page = self._state['page']
		current = self._browser.observe()
		if (
			current['target_id'] != page['target_id']
			or current.get('document_id') != page.get('document_id')
			or current['fingerprint'] != decision['fingerprint']
			or (expected_fingerprint is not None and expected_fingerprint != decision['fingerprint'])
		):
			self._state.update(decision=None, status='ready')
			self.observe('stale_retry')
			return
		operation = decision['operation']
		action = dict(
			kind=operation,
			operation=operation,
			target=decision['target'],
			label=decision['target'],
			native_action=decision['native_action'],
			source_url=page['url'],
			source_document=page.get('document_id'),
			step=len(self._state['history']) + 1,
		)
		self._emit('action_started', action=action)
		self._state.update(decision=None, status='ready')
		try:
			results = self._bridge.run(self._act(), cancel=self._should_cancel, timeout=150)
		except BaseException as error:
			self._emit('action_result', action=action, error=str(error) or type(error).__name__)
			raise
		result = results[-1] if results else {}
		action.update(result=results, elapsed_ms=round((time.monotonic() - self._started) * 1000))
		if operation != 'done':
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
		self.observe('step')

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

	async def _close_browser(self):
		try:
			if self._native is not None:
				await self._native.eventbus.stop(clear=True, timeout=1)
		finally:
			browser = self._browser.native
			if self._chrome is not None:
				await self._chrome.finish_case()
			elif self._browser_start_dispatched and browser is not None:
				await browser.kill()
