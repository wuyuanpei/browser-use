"""Exercise recovery, discarded decisions and exact dispatch boundaries against local Chromium."""

import asyncio
import json
import time
from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer

from browser_use import Agent
from browser_use.agent.jev import JevChecks, JevSettings
from browser_use.agent.jev.service import _is_repeated_empty
from browser_use.agent.jev.views import ActionRecord, DelegateSubgoalParams, SubgoalRun
from browser_use.agent.views import AgentStepInfo
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import FakeJev, browser_eval, serve_page
from tests.ci.jev_support import jev_browser as _jev_browser_fixture  # noqa: F401
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401


def planner_action(name, **params):
	return json.dumps({'action': [{name: params}]})


def subgoal(name):
	return planner_action(
		'delegate_subgoal',
		objective=name,
		completion=f'{name} complete',
		reason='Check the page before delegating.',
		may_submit=True,
		requires_action=True,
	)


def test_only_adjacent_empty_subgoals_count():
	def run(n, name):
		return SubgoalRun(
			id=f'sg{n}',
			params=DelegateSubgoalParams(objective=name, completion='Done', reason='review'),
			identity=name,
			started_step=n,
			start_url='http://localhost',
			start_site='localhost',
			max_actions=15,
		)

	a, b = run(1, 'A'), run(2, 'B')
	assert _is_repeated_empty([a, a], 'A')
	assert not _is_repeated_empty([a, b, a], 'A')
	a.actions.append(
		ActionRecord(
			n=1,
			step=1,
			run_id=a.id,
			operation='WAIT',
			bu_action={'wait': {}},
			url_before=a.start_url,
			fingerprint_before='same',
			status='discarded',
		)
	)
	assert _is_repeated_empty([a, a], 'A')
	a.actions[-1].status = 'unknown'
	assert not _is_repeated_empty([a, a], 'A')


async def test_replan_after_intervening_prerequisite(jev_browser, httpserver: HTTPServer, tmp_path: Path):
	url = serve_page(
		httpserver,
		'/replan',
		"""<button onclick="document.querySelector('#save').disabled=false;this.disabled=true">Unlock</button>
<button id="save" disabled onclick="document.querySelector('output').textContent='A saved'">Save final</button><output></output>""",
	)
	fake = FakeJev(
		httpserver,
		[
			{'operation': 'UNABLE'},
			{'operation': 'CLICK', 'target': 'Unlock'},
			{'operation': 'STEP_DONE'},
			{'operation': 'UNABLE'},
			{'operation': 'CLICK', 'target': 'Save final'},
			{'operation': 'STEP_DONE'},
		],
	)
	planner = create_mock_llm(
		[subgoal('A'), subgoal('B'), subgoal('A'), subgoal('A'), planner_action('finish', success=True, reason='Save observed')]
	)
	agent = Agent(
		task=f'Visit {url} and save A after unlocking.',
		llm=planner,
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(
			jev_api_key='test',
			jev_base_url=httpserver.url_for('/'),
			answer_llm=create_mock_llm(['{"text":"A saved"}']),
			trace_path=tmp_path / 'trace.jsonl',
		),
	)
	observed = []

	async def inspect(at_step):
		if at_step.history.is_done():
			observed.append(await browser_eval(jev_browser, 'document.querySelector("output").textContent'))

	history = await agent.run(max_steps=12, on_step_end=inspect)
	assert not fake.errors
	assert observed == ['A saved']
	assert history.final_result() == 'A saved'
	assert agent._jev is not None
	runs = agent._jev.runs
	assert [r.params.objective for r in runs] == ['A', 'B', 'A', 'A']
	assert [len(r.executed_actions) for r in runs] == [0, 1, 0, 1]
	assert not any(e['event'] == 'delegate_refused' for e in _events(tmp_path / 'trace.jsonl'))


def _events(path: Path) -> list[dict]:
	return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


async def test_cancellation_before_native_dispatch_uses_no_budget(jev_browser, httpserver: HTTPServer, tmp_path: Path):
	url = serve_page(
		httpserver,
		'/counter',
		"""<button onclick="document.querySelector('output').textContent='1'">Count</button><output>0</output>""",
	)
	fake = FakeJev(httpserver, [{'operation': 'CLICK', 'target': 'Count'}, {'operation': 'CLICK', 'target': 'Count'}])
	check_entered = asyncio.Event()
	release = asyncio.Event()

	async def pre_action_check():
		check_entered.set()
		await release.wait()
		return False

	agent = Agent(
		task=f'Press Count on {url}',
		llm=create_mock_llm(),
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(jev_api_key='test', jev_base_url=httpserver.url_for('/'), trace_path=tmp_path / 'trace.jsonl'),
	)
	assert agent._jev is not None
	try:
		await jev_browser.navigate_to(url)
		await agent._jev.delegate(
			DelegateSubgoalParams(objective='Press Count', completion='Counter is one', reason='Start'), jev_browser
		)
		summary = await agent._prepare_context(AgentStepInfo(0, 8))
		await agent._get_next_action(summary)
		agent.register_external_agent_status_raise_error_callback = pre_action_check
		task = asyncio.create_task(agent._execute_actions())
		await asyncio.wait_for(check_entered.wait(), timeout=10)
		task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await task
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == '0'
		assert agent._jev.runs[-1].executed_actions == []
		agent.register_external_agent_status_raise_error_callback = None
		await agent._get_next_action(await agent._prepare_context(AgentStepInfo(0, 8)))
		await agent._execute_actions()
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == '1'
		assert [r.status for r in agent._jev.runs[-1].actions] == ['discarded', 'executed_ok']
		assert len(fake.requests) == 2 and not fake.errors
	finally:
		release.set()
		await close_driver(agent)


async def close_driver(agent):
	# Private-step drivers, unlike Agent.run(), own the lifetime of the Agent event dispatcher.
	try:
		await agent.eventbus.stop(clear=True, timeout=3)
	finally:
		await agent._jev.aclose()


async def routed(agent, *, execute=True):
	executor = agent._jev
	assert executor is not None
	agent.step_start_time = time.time()
	step_info = AgentStepInfo(agent.state.n_steps - 1, 30)
	summary = await agent._prepare_context(step_info)
	out = await executor.route(agent, summary)
	if out.model_output is not None and execute:
		agent.state.last_model_output = out.model_output
		await agent._execute_actions()
		await agent._post_process()
		await agent._finalize(summary)
	return out


async def start_stage(
	session,
	server,
	tmp_path,
	*,
	goal='Exercise controls',
	max_actions=15,
	checks=None,
	max_candidates=200,
	body=None,
	requires=False,
):
	url = serve_page(
		server,
		'/stage',
		body
		if body is not None
		else """<button onclick="document.querySelector('output').textContent='Enabled'">Enable</button>
<button onclick="document.querySelector('output').textContent='Default'">Reset</button>
<button>Plain</button><output>Default</output>""",
	)
	await session.navigate_to(url)
	settings = JevSettings(
		jev_api_key='test',
		jev_base_url=server.url_for('/'),
		trace_path=tmp_path / 'trace.jsonl',
		max_candidates=max_candidates,
		checks=checks or JevChecks(),
	)
	agent = Agent(task=goal, llm=create_mock_llm(), browser_session=session, use_judge=False, jev=settings)
	e = agent._jev
	assert e is not None
	params = DelegateSubgoalParams(
		objective='Use controls',
		completion='State observed',
		reason='Proceed',
		may_submit=True,
		max_actions=max_actions,
		requires_action=requires,
	)
	out = await e.delegate(params, session)
	assert not out.error
	return agent, e, params


@pytest.mark.parametrize(
	('case', 'steps', 'limit', 'expected'),
	[
		('budget', [{'operation': 'CLICK', 'target': 'Enable'}], 1, 'action_budget'),
		('wait', [{'operation': 'WAIT'}, {'operation': 'WAIT'}], 15, 'wait_limit'),
		('stuck', [{'operation': 'CLICK', 'target': 'Plain'}] * 3, 15, 'no_effect'),
		(
			'repeat',
			[{'operation': 'CLICK', 'target': label} for label in ('Enable', 'Reset', 'Enable', 'Reset', 'Enable')],
			15,
			'repeated_action',
		),
		('confidence', [{'operation': 'CLICK', 'target': 'Enable', 'p': 0.75, 'confidence': 0.95}], 15, 'low_confidence'),
		(
			'unable_probability',
			[{'operation': 'CLICK', 'target': 'Enable', 'p': 0.55, 'unable_p': 0.4}],
			15,
			'unable_probability',
		),
		('unable', [{'operation': 'UNABLE'}], 15, 'unable'),
	],
)
async def test_handbacks_use_actual_page_and_results(case, steps, limit, expected, jev_browser, httpserver, tmp_path):
	fake = FakeJev(httpserver, steps)
	agent, e, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		max_actions=limit,
		checks=JevChecks(confidence='off') if case == 'unable' else JevChecks(),
	)
	try:
		for _ in range(len(steps) + 1):
			if (await routed(agent)).model_output is None:
				break
		handback = e.runs[-1].handback
		assert handback and handback.reason == expected, handback
		assert not fake.errors
		if case == 'repeat':
			assert len(e.runs[-1].executed_actions) == 4
		elif case == 'confidence':
			assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == 'Default'
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('case', ['step_limit', 'step_error', 'out_of_scope', 'too_many_candidates', 'state_too_large'])
async def test_prechecks_do_not_call_model(case, jev_browser, httpserver, tmp_path):
	agent, e, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		max_candidates=1 if case == 'too_many_candidates' else 200,
		goal='long original goal ' * 8000 if case == 'state_too_large' else 'Check controls',
	)
	fake = FakeJev(httpserver, [])
	try:
		summary = await agent._prepare_context(AgentStepInfo(0, 20))
		if case == 'step_limit':
			e.finish_only = True
		elif case == 'step_error':
			agent.state.consecutive_failures = 1
		elif case == 'out_of_scope':
			cross = httpserver.url_for('/stage').replace('localhost', '127.0.0.1')
			await jev_browser.navigate_to(cross)
			summary = await agent._prepare_context(AgentStepInfo(0, 20))
		out = await e.route(agent, summary)
		assert out.model_output is None
		assert e.runs[-1].handback and e.runs[-1].handback.reason == case
		assert fake.requests == []
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('case', ['invalid_choice', 'non_argmax', 'http_failure'])
async def test_invalid_answer_and_http_failure(case, jev_browser, httpserver, tmp_path):
	from werkzeug.wrappers import Response

	def reply(request):
		body = request.get_json()
		answers = {}
		for name, q in body['questions'].items():
			keys = list(q['criteria'])
			choice = keys[0]
			probs = {k: float(k == choice) for k in keys}
			if name == 'operation':
				if case == 'invalid_choice':
					choice = 'missing-choice'
				else:
					probs[choice], probs[keys[-1]] = 0.05, 0.95
			answers[name] = {'choice': choice, 'probabilities': probs, 'confidence': 0.95}
		return Response(
			json.dumps({'answers': answers, 'usage': {'input_tokens': 10, 'output_tokens': 1}}), content_type='application/json'
		)

	handler = httpserver.expect_request('/v1/systemone', method='POST')
	if case == 'http_failure':
		handler.respond_with_data('service error', status=500)
	else:
		handler.respond_with_handler(reply)
	agent, e, _ = await start_stage(jev_browser, httpserver, tmp_path)
	try:
		assert (await routed(agent)).model_output is None
		assert e.runs[-1].handback and e.runs[-1].handback.reason == (
			'system1_error' if case == 'http_failure' else 'invalid_answer'
		)
		assert not e.runs[-1].executed_actions
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('mode', ['record', 'off'])
async def test_record_and_off_checks_preserve_action(mode, jev_browser, httpserver, tmp_path):
	agent, e, _ = await start_stage(
		jev_browser, httpserver, tmp_path, checks=JevChecks.model_validate({'default': mode}), requires=True
	)
	fake = FakeJev(httpserver, [{'operation': 'CLICK', 'target': 'Enable', 'p': 0.72}, {'operation': 'STEP_DONE'}])
	try:
		await jev_browser.navigate_to(httpserver.url_for('/stage').replace('localhost', '127.0.0.1'))
		assert (await routed(agent)).model_output is not None
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == 'Enabled'
		assert (await routed(agent)).model_output is None
		would = [v['name'] for v in _events(tmp_path / 'trace.jsonl') if v['event'] == 'check' and v['outcome'] == 'would_block']
		assert ('scope' in would and 'confidence' in would) if mode == 'record' else not would
		assert not fake.errors
	finally:
		await close_driver(agent)


async def test_rejected_step_done_keeps_handback_and_requires_note(jev_browser, httpserver, tmp_path):
	agent, e, params = await start_stage(jev_browser, httpserver, tmp_path)
	fake = FakeJev(
		httpserver, [{'operation': 'STEP_DONE'}, {'operation': 'CLICK', 'target': 'Enable'}, {'operation': 'STEP_DONE'}]
	)
	try:
		first = await routed(agent)
		assert first.model_output is None and '<system1_handback>' in str(first.context_messages[-1].content)
		assert (await e.delegate(params, jev_browser)).error
		retry = await routed(agent)
		assert '<system1_handback>' in str(retry.context_messages[-1].content)
		assert not (await e.delegate(params.model_copy(update={'review_note': 'Enable still required.'}), jev_browser)).error
		await routed(agent)
		assert 'STEP_DONE' not in fake.requests[1]['questions']['operation']['criteria']
		await routed(agent)
		assert 'STEP_DONE' in fake.requests[2]['questions']['operation']['criteria']
		assert not fake.errors
	finally:
		await close_driver(agent)


async def test_dispatched_wait_cancelled_is_counted_and_handed_back(jev_browser, httpserver, tmp_path):
	fake = FakeJev(httpserver, [{'operation': 'WAIT'}])
	agent, e, _ = await start_stage(jev_browser, httpserver, tmp_path)
	try:
		routing = await routed(agent, execute=False)
		assert routing.model_output is not None
		agent.state.last_model_output = routing.model_output
		task = asyncio.create_task(agent._execute_actions())
		async with asyncio.timeout(10):
			# Observe the native tools.act boundary without replacing the execution hook.
			while e.pending_record and e.pending_record.status == 'predicted':  # noqa: ASYNC110
				await asyncio.sleep(0.001)
		assert e.pending_record and e.pending_record.status == 'dispatched'
		task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await task
		assert [r.status for r in e.runs[-1].executed_actions] == ['unknown']
		assert (await routed(agent)).model_output is None
		assert e.runs[-1].handback and e.runs[-1].handback.reason == 'action_error'
		assert len(fake.requests) == 1
	finally:
		await close_driver(agent)


async def test_removed_dom_target_is_failure_even_without_native_error(jev_browser, httpserver, tmp_path):
	fake = FakeJev(httpserver, [{'operation': 'CLICK', 'target': 'Enable'}])
	agent, e, _ = await start_stage(jev_browser, httpserver, tmp_path)
	try:
		out = await routed(agent, execute=False)
		assert out.model_output
		agent.state.last_model_output = out.model_output
		await browser_eval(jev_browser, 'document.querySelector("button").remove()')
		await jev_browser.get_browser_state_summary()
		await agent._execute_actions()
		assert agent.state.last_result and e.pending_record
		assert e.pending_record.status == 'executed_failed'
		assert (await routed(agent)).model_output is None
		assert e.runs[-1].handback and e.runs[-1].handback.reason == 'action_error'
		assert not e.runs[-1].applied and len(fake.requests) == 1
	finally:
		await close_driver(agent)


async def test_public_follow_up_resumes_backend_and_uses_new_goal(jev_browser, httpserver, tmp_path):
	url = serve_page(
		httpserver,
		'/continue',
		"""<button onclick="document.querySelector('output').textContent='second instruction done'">Continue</button><output>first-page unique evidence</output>""",
	)
	fake = FakeJev(
		httpserver,
		[
			{'operation': 'COLLECT'},
			{'operation': 'STEP_DONE'},
			{'operation': 'CLICK', 'target': 'Continue'},
			{'operation': 'STEP_DONE'},
		],
	)
	first = planner_action(
		'delegate_subgoal', objective='Read the original text', completion='Original text is captured', reason='Collect evidence'
	)
	finish = planner_action('finish', success=True, reason='Checked current page')
	new_goal = 'Continue the previous task by pressing Continue.'
	agent = Agent(
		task=f'Read the original page at {url}',
		llm=create_mock_llm([first, finish, subgoal('Press Continue'), finish]),
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(
			jev_api_key='test',
			jev_base_url=httpserver.url_for('/'),
			trace_path=tmp_path / 'trace.jsonl',
			capture_dir=tmp_path / 'captures',
			summary_llm=create_mock_llm(['{"summary":"Saved the first page"}']),
			answer_llm=create_mock_llm(['{"text":"Original text collected"}', '{"text":"Second instruction completed"}']),
		),
	)
	first_history = await agent.run(max_steps=7)
	assert first_history.final_result() == 'Original text collected'
	executor = agent._jev
	assert executor is not None and len(executor.captures.records) == 1
	original_backend = executor.backend
	agent.add_new_task(new_goal)
	assert executor.backend is not original_backend and executor.active_run is None
	marker = []

	async def verify(at_step):
		if at_step.history.is_done():
			marker.append(await browser_eval(jev_browser, 'document.querySelector("output").textContent'))

	second_history = await agent.run(max_steps=8, on_step_end=verify)
	assert marker == ['second instruction done']
	assert second_history.final_result() == 'Second instruction completed'
	assert len(fake.requests) == 4 and fake.requests[2]['state']['original_goal'] == new_goal
	assert len(executor.captures.records) == 1 and executor.captures.records[0].text
	assert len([v for v in _events(tmp_path / 'trace.jsonl') if v['event'] == 'final_answer']) == 2


async def test_native_step_timeout_counts_interrupted_system1_request(jev_browser, httpserver, tmp_path):
	from threading import Event

	from werkzeug.wrappers import Response

	arrived, release = Event(), Event()

	def delayed_reply(request):
		arrived.set()
		release.wait(timeout=30)
		return Response('{}', content_type='application/json')

	httpserver.expect_request('/v1/systemone', method='POST').respond_with_handler(delayed_reply)
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path)
	agent.settings.step_timeout = 8
	try:
		await agent._execute_step(0, 10, AgentStepInfo(0, 10))
		assert arrived.is_set() and agent.state.consecutive_failures == 1
		assert not executor.runs[-1].executed_actions
		assert (await routed(agent)).model_output is None
		assert executor.runs[-1].handback and executor.runs[-1].handback.reason == 'step_error'
		stats = executor.usage_by_role()['system1']
		assert stats['decisions'] == stats['calls'] == stats['missing_usage_calls'] == 1
		events = [row for row in _events(tmp_path / 'trace.jsonl') if row['event'] == 'system1_decision']
		assert len(events) == 1 and events[0]['outcome'] == 'interrupted'
		assert events[0]['attempts'][0]['error_type'] == 'CancelledError'
	finally:
		release.set()
		await close_driver(agent)
