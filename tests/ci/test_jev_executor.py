"""Exercise recovery, discarded decisions and exact dispatch boundaries against local Chromium."""

import asyncio
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from browser_use import Agent
from browser_use.agent.jev import JevChecks, JevSettings, backends, service
from browser_use.agent.jev.questions import VISION_RULES
from browser_use.agent.jev.service import _is_repeated_empty, _is_review_delegation
from browser_use.agent.jev.views import ActionRecord, DelegateSubgoalParams, DelegateSubgoalParamsSmart, Handback, SubgoalRun
from browser_use.agent.views import AgentStepInfo
from browser_use.benchmark.tasks import INSPECT_PAGE
from browser_use.browser.events import TypeTextEvent
from browser_use.browser.session import BrowserSession
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import FakeJev, browser_eval, serve_page
from tests.ci.jev_support import jev_browser as _jev_browser_fixture  # noqa: F401
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401

REVIEW_REMINDER = 'Your next delegate_subgoal is a review: set screenshot_required (true or false) with screenshot_reason.'


class _VisionJev(FakeJev):
	"""Run the same scripted choices against the JSON part of a real multipart request."""

	def _handle(self, request: Request) -> Response:
		if request.mimetype == 'multipart/form-data':
			request = Request.from_values(method='POST', json=json.loads(request.form['request']))
		return super()._handle(request)


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


@pytest.mark.parametrize(('filler_count', 'operation'), [(30, 'CLICK'), (215, 'CLICK'), (619, 'CLICK'), (619, 'TYPE_TEXT')])
async def test_dense_inbox_keeps_compose_and_search(jev_browser, httpserver, tmp_path, filler_count, operation):
	filler = ''.join(
		f'<button style="height:22px;overflow:hidden;white-space:nowrap">邮件 {i}</button>' for i in range(filler_count)
	)
	url = serve_page(
		httpserver,
		'/dense-inbox',
		'<div style="display:flex;flex-direction:column-reverse">'
		f'<div style="display:grid;grid-template-columns:repeat(32,minmax(0,1fr));width:1100px">{filler}</div>'
		"<button onclick=\"document.querySelector('output').textContent='草稿已打开'\">写信</button>"
		'<input type="search" placeholder="搜索邮件" oninput="document.querySelector(\'output\').textContent=this.value">'
		'<output></output></div>',
	)
	search = operation == 'TYPE_TEXT'
	target = '搜索邮件' if search else '写信'
	fake = FakeJev(httpserver, [{'operation': operation, 'target': target}, {'operation': 'STEP_DONE'}])
	objective = '在搜索邮件框输入会议' if search else '打开写信编辑器'
	planner = create_mock_llm([subgoal(objective), planner_action('finish', success=True, reason='Inbox action applied')])
	agent = Agent(
		task=f'Visit {url} and {objective}.',
		llm=planner,
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(
			jev_api_key='test',
			jev_base_url=httpserver.url_for('/'),
			text_llm=create_mock_llm(['{"text":"会议"}']),
			answer_llm=create_mock_llm(['{"text":"Inbox action applied"}']),
			trace_path=tmp_path / 'trace.jsonl',
		),
	)
	observed = []

	async def inspect(at_step):
		if at_step.history.is_done():
			observed.append(await browser_eval(jev_browser, 'document.querySelector("output").textContent'))

	history = await agent.run(max_steps=5, on_step_end=inspect)
	assert fake.requests and not fake.errors
	questions = fake.requests[0]['questions']
	assert len(questions['click_target']['criteria']) == min(filler_count + 2, 200)
	assert any('搜索邮件' in entry['element'] for entry in questions['type_text_target']['criteria'].values())
	first = next(event for event in _events(tmp_path / 'trace.jsonl') if event['event'] == 'system1_decision')
	assert first['observed_candidate_counts']['click_target'] == filler_count + 2
	assert history.is_successful() and observed == (['会议'] if search else ['草稿已打开'])


@pytest.mark.parametrize('mode,expected', [('append', '前缀关键点'), ('replace', '关键点')])
async def test_jev_applies_explicit_rich_text_mode(jev_browser, httpserver, tmp_path, mode, expected):
	url = serve_page(
		httpserver,
		'/rich-text',
		'<div id="editor" aria-label="正文" role="textbox" contenteditable="true">'
		'<p id="tail">前缀</p><ul id="items"><li>第一项</li><li>第二项</li><li>第三项</li></ul></div>'
		'<script>if (location.search.includes("append")) {'
		'editor.prepend(items);'
		'editor.focus(); getSelection().collapse(tail.firstChild, tail.textContent.length);'
		'document.execCommand("bold"); document.execCommand("italic");'
		'let drop = true; editor.addEventListener("beforeinput", e => {'
		'if (drop && e.data === "关") { drop = false; e.preventDefault(); }'
		'}); }</script>',
	)
	fake = FakeJev(httpserver, [{'operation': 'TYPE_TEXT', 'target': '正文'}, {'operation': 'STEP_DONE'}])
	plan = planner_action(
		'delegate_subgoal',
		objective='写入关键点',
		completion='正文显示关键点',
		reason='按任务填写',
		inputs='关键点',
		text_mode=mode,
		requires_action=True,
	)
	agent = Agent(
		task=f'Open {url}?{mode} and use {mode} to write 关键点 in the rich text editor.',
		llm=create_mock_llm([plan, planner_action('finish', success=True, reason='Text present')]),
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(
			jev_api_key='local-test',
			jev_base_url=httpserver.url_for('/'),
			trace_path=tmp_path / 'trace.jsonl',
			text_llm=create_mock_llm(['{"text":"关键点"}']),
			answer_llm=create_mock_llm(['{"text":"Text present"}']),
		),
	)
	proof = []

	async def inspect(running):
		if running.history.is_done():
			proof.append(
				await browser_eval(
					jev_browser,
					'[(document.querySelector("#tail") || editor).textContent, editor.querySelectorAll("li").length, '
					'[...editor.querySelectorAll("i, em")].some(el => el.textContent === "关键点"), '
					'[...editor.querySelectorAll("b, strong")].some(el => el.textContent === "关键点")]',
				)
			)
			proof.append((await browser_eval(jev_browser, INSPECT_PAGE))['facts']['rich_text'][0])

	await agent.run(max_steps=5, on_step_end=inspect)
	assert not fake.errors
	assert proof[0] == [expected, 3 if mode == 'append' else 0, mode == 'append', mode == 'append']
	assert proof[1]['text'].endswith(expected)
	assert proof[1]['format'] == ([{'bold': True, 'italic': True, 'runs': ['关键点']}] if mode == 'append' else [])
	assert proof[1]['unordered_lists'] == ([['第一项', '第二项', '第三项']] if mode == 'append' else [])


async def test_replacing_empty_editor_preserves_managed_nodes(jev_browser, httpserver):
	url = serve_page(
		httpserver,
		'/empty-editor',
		'<div id="editor" contenteditable="true"><div data-editor-block><br></div></div>'
		'<script>editor.addEventListener("beforeinput", e => {'
		'if (!editor.querySelector("[data-editor-block]")) e.preventDefault();'
		'});</script>',
	)
	await jev_browser.navigate_to(url)
	summary = await jev_browser.get_browser_state_summary()
	target = next(node for node in summary.dom_state.selector_map.values() if node.attributes.get('id') == 'editor')
	event = jev_browser.event_bus.dispatch(TypeTextEvent(node=target, text='hello', clear=True))
	await event.event_result(raise_if_any=True, raise_if_none=False)
	assert await browser_eval(jev_browser, 'editor.querySelector("[data-editor-block]").innerText') == 'hello'


async def test_append_does_not_fall_back_to_focused_field(jev_browser, httpserver, monkeypatch):
	from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

	url = serve_page(httpserver, '/append', '<input id="target" value="prefix"><input id="other" value="unrelated">')
	await jev_browser.navigate_to(url)
	summary = await jev_browser.get_browser_state_summary()
	target = next(node for node in summary.dom_state.selector_map.values() if node.attributes.get('id') == 'target')
	await browser_eval(jev_browser, 'other.focus()')

	async def lose_target(*_args, **_kwargs):
		raise RuntimeError('editable target detached')

	monkeypatch.setattr(DefaultActionWatchdog, '_input_text_element_node_impl', lose_target)
	monkeypatch.setattr(DefaultActionWatchdog, '_click_element_node_impl', lose_target)
	event = jev_browser.event_bus.dispatch(TypeTextEvent(node=target, text='suffix', clear=False))
	with pytest.raises(RuntimeError, match='editable target detached'):
		await event.event_result(raise_if_any=True, raise_if_none=False)
	assert await browser_eval(jev_browser, '[target.value, other.value]') == ['prefix', 'unrelated']


async def test_jev_reobserves_when_categories_load_during_native_capture(jev_browser, httpserver, monkeypatch, tmp_path):
	url = serve_page(
		httpserver,
		'/load-category',
		'<main id="panel">Nothing loaded yet</main><output></output>',
	)
	fake = FakeJev(httpserver, [{'operation': 'CLICK', 'target': 'Beauty'}, {'operation': 'STEP_DONE'}])
	plan = create_mock_llm([subgoal('Find Beauty'), planner_action('finish', success=True, reason='Beauty selected')])
	agent = Agent(
		task=f'Open {url} and select Beauty.',
		llm=plan,
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(
			jev_api_key='test',
			jev_base_url=httpserver.url_for('/'),
			trace_path=tmp_path / 'trace.jsonl',
			answer_llm=create_mock_llm(['{"text":"Beauty selected"}']),
		),
	)
	executor = agent._jev
	assert executor is not None
	capture = jev_browser.get_browser_state_summary
	switched = False

	async def change_during_capture(**kwargs):
		nonlocal switched
		summary = await capture(**kwargs)
		if not switched and executor.active_run and summary.url == url:
			switched = True
			await browser_eval(
				jev_browser,
				"""panel.innerHTML = `<button onclick="this.parentElement.textContent='Beauty selected'">Beauty</button>`""",
			)
		return summary

	monkeypatch.setattr(BrowserSession, 'get_browser_state_summary', lambda self, **kwargs: change_during_capture(**kwargs))
	history = await agent.run(max_steps=6)
	assert switched and history.is_successful()
	assert not fake.errors
	request = fake.requests[0]
	assert 'Beauty' in request['state']['page']['text'] and 'Nothing loaded yet' not in request['state']['page']['text']
	assert any('Beauty' in v['element'] for v in request['questions']['click_target']['criteria'].values())
	rows = _events(tmp_path / 'trace.jsonl')
	handback = next(row for row in rows if row['event'] == 'handback' and row['reason'] == 'step_done')
	same_step = next(row for row in rows if row['event'] == 'system1_decision' and row['step'] == handback['step'])
	assert handback['fingerprint'] == same_step['observation_fingerprint']


async def test_read_only_operation_can_use_split_confidence(jev_browser, httpserver):
	url = serve_page(httpserver, '/split-confidence', '<main style="height:200vh"><p>Visible article</p></main>')
	fake = FakeJev(
		httpserver,
		[
			{'operation': 'COLLECT', 'operation_probs': {'COLLECT': 0.58, 'SCROLL_DOWN': 0.41, 'UNABLE': 0.01}},
			{'operation': 'STEP_DONE'},
		],
	)
	planner = create_mock_llm(
		[
			planner_action(
				'delegate_subgoal', objective='Read the article', completion='Article saved', reason='Capture visible text'
			),
			planner_action('finish', success=True, reason='Article saved'),
		]
	)
	agent = Agent(
		task=f'Visit {url} and save the visible article.',
		llm=planner,
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(
			jev_api_key='test',
			jev_base_url=httpserver.url_for('/'),
			summary_llm=create_mock_llm(['{"text":"Visible article"}']),
			answer_llm=create_mock_llm(['{"text":"Article saved"}']),
		),
	)

	history = await agent.run(max_steps=5)
	assert not fake.errors
	assert any('collect_page' in action for action in history.model_actions())


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


async def routed(agent, *, execute=True, summary=None):
	executor = agent._jev
	assert executor is not None
	agent.step_start_time = time.time()
	step_info = AgentStepInfo(agent.state.n_steps - 1, 30)
	if summary is None:
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
	screenshot_policy: Literal['off', 'always', 'smart'] = 'off',
	vision_escalation=True,
	min_read_operation_probability=0.6,
	screenshot_fields=None,
	save_conversation_path=None,
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
		screenshot_policy=screenshot_policy,
		vision_escalation=vision_escalation,
		min_read_operation_probability=min_read_operation_probability,
		checks=checks or JevChecks(),
	)
	agent = Agent(
		task=goal,
		llm=create_mock_llm(),
		browser_session=session,
		use_judge=False,
		jev=settings,
		save_conversation_path=save_conversation_path,
	)
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
	if screenshot_policy == 'smart':
		params = DelegateSubgoalParamsSmart.model_validate({**params.model_dump(exclude_unset=True), **(screenshot_fields or {})})
		out = await agent.tools.registry.execute_action(
			'delegate_subgoal', params.model_dump(exclude_unset=True), browser_session=session
		)
	else:
		out = await e.delegate(params, session)
	assert not out.error
	return agent, e, params


async def test_missing_screenshot_uses_text_in_the_same_decision(jev_browser, httpserver, tmp_path):
	fake = FakeJev(httpserver, [{'operation': 'STEP_DONE'}])
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, screenshot_policy='always')
	try:
		summary = await agent._prepare_context(AgentStepInfo(0, 10))
		result = await executor.route(agent, replace(summary, screenshot=None))
		handback = executor.runs[-1].handback
		assert handback is not None and handback.reason == 'step_done'
		assert 'Screenshots: 0 of 1 System1 decisions' in result.context_messages[-1].content
		assert len(fake.requests) == 1 and 'screenshot' not in fake.requests[0]['state']['page']
		events = _events(tmp_path / 'trace.jsonl')
		assert any(row['event'] == 'vision_unavailable' for row in events)
		row = next(row for row in events if row['event'] == 'system1_decision')
		assert row['vision'] is False and row['vision_trigger'] is None and row['image_bytes'] == row['rect_count'] == 0
		usage = executor.usage_by_role()['system1']
		assert (usage['text_requests'], usage['vision_requests']) == (1, 0)
	finally:
		await close_driver(agent)


async def scroll_stage(jev_browser, httpserver, tmp_path, **settings):
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, **settings)
	fixture = Path(__file__).resolve().parents[2] / 'browser_use/benchmark/static/fixture.html'
	httpserver.expect_request('/region-fixture').respond_with_data(fixture.read_bytes(), content_type='text/html; charset=utf-8')
	await jev_browser.navigate_to(httpserver.url_for('/region-fixture') + '?scenario=scroll-panel')
	return agent, executor


async def test_region_scroll_moves_only_panel_and_consumes_budget(jev_browser, httpserver, tmp_path):
	fake = _VisionJev(httpserver, [{'operation': 'SCROLL_DOWN', 'target': 'Filters'}])
	agent, executor = await scroll_stage(jev_browser, httpserver, tmp_path, max_actions=1, screenshot_policy='always')
	try:
		assert (await routed(agent)).model_output is not None
		assert await browser_eval(jev_browser, "document.querySelector('#filter-panel').scrollTop > 0 && scrollY === 0")
		run = executor.runs[-1]
		assert len(run.executed_actions) == 1 and 'scroll_region' in run.executed_actions[0].bu_action
		assert (await routed(agent)).model_output is None and run.handback is not None and run.handback.reason == 'action_budget'
		effect = run.executed_actions[0].scroll_effect
		assert effect and effect.measurable and effect.root_delta_y == 0
		assert effect.target_delta_y is not None and effect.target_delta_y > 0
		assert run.executed_actions[0].page_changed is True
		criteria = fake.requests[0]['questions']['scroll_down_target']['criteria']
		assert 'rect' in criteria[run.executed_actions[0].target_key]
		assert 'scroll_region' not in str(agent._planner_output_format().model_json_schema())
	finally:
		await close_driver(agent)


@pytest.mark.parametrize(
	'mutation',
	[
		'p.remove()',
		'p.replaceWith(p.cloneNode(true))',
		'p.scrollTop=p.scrollHeight',
		"document.body.insertAdjacentHTML('beforeend','<div style=\"position:fixed;inset:0;background:gray;z-index:2147483647\"></div>')",
	],
)
async def test_region_scroll_rechecks_selected_node_before_wheel(mutation, jev_browser, httpserver, tmp_path):
	FakeJev(httpserver, [{'operation': 'SCROLL_DOWN', 'target': 'Filters'}])
	agent, executor = await scroll_stage(jev_browser, httpserver, tmp_path)
	try:
		summary = await agent._prepare_context(AgentStepInfo(0, 10))
		out = await routed(agent, summary=summary, execute=False)
		assert out.model_output is not None
		await browser_eval(jev_browser, f"(()=>{{const p=document.querySelector('#filter-panel');{mutation};return true}})()")
		agent.state.last_model_output = out.model_output
		await agent._execute_actions()
		await agent._post_process()
		await agent._finalize(summary)
		run = executor.runs[-1]
		assert len(run.executed_actions) == 1 and run.executed_actions[0].status == 'executed_failed'
		assert (await routed(agent)).model_output is None and run.handback is not None and run.handback.reason == 'action_error'
	finally:
		await close_driver(agent)


async def test_cancelled_region_wheels_still_trigger_no_effect(jev_browser, httpserver, tmp_path):
	FakeJev(httpserver, [{'operation': 'SCROLL_DOWN', 'target': 'Filters'}] * 2)
	agent, executor = await scroll_stage(jev_browser, httpserver, tmp_path)
	try:
		await browser_eval(
			jev_browser,
			"let wheelCount=0; const counter=document.createElement('output'); document.body.prepend(counter); "
			"document.querySelector('#filter-panel').addEventListener('wheel',e=>{e.preventDefault(); "
			"counter.textContent='Wheel '+ ++wheelCount;},{passive:false}); true",
		)
		await routed(agent)
		await routed(agent)
		assert (await routed(agent)).model_output is None
		assert await browser_eval(jev_browser, "document.querySelector('output').textContent") == 'Wheel 2'
		run = executor.runs[-1]
		assert len(run.executed_actions) == 2 and run.handback is not None and run.handback.reason == 'no_effect'
		for action in run.executed_actions:
			assert action.scroll_effect and action.scroll_effect.target_delta_y == 0 and action.page_changed is False
	finally:
		await close_driver(agent)


async def test_page_scroll_after_root_locks_counts_no_effect_and_escalates(jev_browser, httpserver, tmp_path):
	fake = _VisionJev(httpserver, [{'operation': 'SCROLL_DOWN', 'target': 'the whole page'}] * 2 + [{'operation': 'STEP_DONE'}])
	agent, executor, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		screenshot_policy='smart',
		body=(
			'<main style="height:220vh"><p id="activity">Initial content</p></main>'
			'<aside aria-label="Side panel" style="position:fixed;left:0;top:100px;width:180px;height:220px;overflow:auto">'
			'<div style="height:800px">Scrollable content</div></aside>'
		),
	)
	try:
		run = executor.runs[-1]
		for n in range(2):
			summary = await agent._prepare_context(AgentStepInfo(agent.state.n_steps - 1, 30))
			prediction = await routed(agent, execute=False, summary=summary)
			assert prediction.model_output is not None and run.no_effect_streak == n
			assert run.actions[-1].target_key == 'page'
			# The page locked after prediction, when the offered page action was already selected.
			await browser_eval(jev_browser, "document.documentElement.style.overflowY='hidden'")
			agent.state.last_model_output = prediction.model_output
			await agent._execute_actions()
			await agent._post_process()
			await agent._finalize(summary)
			assert await browser_eval(
				jev_browser, "scrollY === 0 && getComputedStyle(document.documentElement).overflowY === 'hidden'"
			)
			await browser_eval(jev_browser, f"activity.textContent='New viewport text {n}'")
			if n == 0:
				await browser_eval(jev_browser, "document.documentElement.style.overflowY=''")

		assert (await routed(agent)).model_output is None
		assert run.no_effect_escalated and run.vision_source == 'escalation'
		assert run.handback and run.handback.reason == 'step_done'
		assert (
			len(fake.requests) == 3
			and fake.requests[-1]['state']['current_subgoal']['screenshot_guidance']['source'] == 'escalation'
		)
		for request, expected in zip(
			fake.requests, ('Initial content', 'New viewport text 0', 'New viewport text 1'), strict=True
		):
			assert expected in request['state']['page']['text']
		assert 'page' not in fake.requests[-1]['questions']['scroll_down_target']['criteria']
		assert 'Side panel' in str(fake.requests[-1]['questions']['scroll_down_target']['criteria'])
		events = _events(tmp_path / 'trace.jsonl')
		results = [event for event in events if event['event'] == 'action_result' and event['operation'] == 'SCROLL_DOWN']
		assert len(results) == 2
		for event in results:
			assert event['status'] == 'executed_ok' and event['page_changed'] is False
			assert event['scroll_effect'] == {
				'target_key': 'page',
				'root_delta_y': 0,
				'center_delta_y': None,
				'target_delta_y': None,
				'url_changed': False,
				'measurable': True,
			}
		locked = next(event for event in events if event['event'] == 'scroll_page_suppressed')
		assert locked['reason'] == 'root_locked_without_center_scroll' and locked['overflow_y'] == 'hidden'
	finally:
		await close_driver(agent)


@pytest.mark.parametrize(
	('operation', 'threshold', 'execute'), [('SCROLL_DOWN', 0.6, True), ('CLICK', 0.6, False), ('SCROLL_DOWN', 0.8, False)]
)
async def test_read_threshold_only_changes_selected_read_operations(
	operation, threshold, execute, jev_browser, httpserver, tmp_path
):
	FakeJev(
		httpserver,
		[
			{
				'operation': operation,
				'target': 'the whole page' if operation == 'SCROLL_DOWN' else 'Plain',
				'operation_probs': {operation: 0.72, 'WAIT': 0.28},
			},
			{'operation': 'STEP_DONE'},
		],
	)
	agent, executor, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		min_read_operation_probability=threshold,
		body='<main style="height:220vh"><button>Plain</button></main>',
	)
	try:
		result = await routed(agent)
		assert (result.model_output is not None) is execute
		if execute:
			assert await browser_eval(jev_browser, 'scrollY') > 0
			assert (await routed(agent)).model_output is None
			action = executor.runs[-1].executed_actions[0]
			effect = action.scroll_effect
			assert action.page_changed and effect and effect.measurable
			assert effect.root_delta_y is not None and effect.root_delta_y > 0
		else:
			handback = executor.runs[-1].handback
			assert handback is not None and handback.reason == 'low_confidence'
		events = _events(tmp_path / 'trace.jsonl')
		assert next(r for r in events if r['event'] == 'run_config')['min_read_operation_probability'] == threshold
		assert next(r for r in events if r['event'] == 'system1_decision')['read_operation_probability'] == (
			0.72 if operation == 'SCROLL_DOWN' else None
		)
	finally:
		await close_driver(agent)


def image_requests(server):
	return server.expect_request('/v1/systemone', method='POST', headers={'TypeSafe-Danger-Zone-Images': 'true'})


LOW_CLICK = {'operation': 'CLICK', 'target': 'Plain', 'p': 0.62, 'target_p': 0.68}


@pytest.mark.parametrize(
	('reason', 'first'),
	[
		('low_confidence', LOW_CLICK),
		('unable_probability', {'operation': 'UNABLE'}),
		('unable', {'operation': 'UNABLE'}),
		('repeated_action', {'operation': 'CLICK', 'target': 'Plain'}),
	],
)
async def test_runtime_image_retry_uses_same_observation(reason, first, jev_browser, httpserver, tmp_path, monkeypatch):
	fake = _VisionJev(
		httpserver, [first, {'operation': 'CLICK', 'target': 'Enable'}, {'operation': 'STEP_DONE'}, {'operation': 'STEP_DONE'}]
	)
	conversations = tmp_path / 'conversations'
	agent, executor, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		screenshot_policy='smart',
		screenshot_fields={'screenshot_required': False, 'screenshot_reason': 'Labels should suffice.'},
		checks=JevChecks(confidence='off') if reason == 'unable' else JevChecks(),
		save_conversation_path=conversations,
	)
	try:
		summary = await agent._prepare_context(AgentStepInfo(0, 10))
		run = executor.runs[-1]
		if reason == 'repeated_action':
			obs = await service.state.observe(jev_browser, summary, executor.settings, executor.step)
			button = next(c for c in obs.candidates['CLICK'] if c.name == 'Plain')
			for n in (1, 2):
				run.actions.append(
					ActionRecord(
						n=n,
						step=1,
						run_id=run.id,
						operation='CLICK',
						target_key=button.key,
						bu_action={'click': {'index': button.index}},
						fingerprint_before=obs.fingerprint,
						url_before=obs.url,
						status='executed_ok',
					)
				)
		observe = AsyncMock(wraps=service.state.observe)
		prepare = Mock(wraps=service.prepare_screenshot)
		monkeypatch.setattr(service.state, 'observe', observe)
		monkeypatch.setattr(service, 'prepare_screenshot', prepare)
		assert (await routed(agent, summary=summary)).model_output is not None
		assert observe.await_count == prepare.call_count == 1
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == 'Enabled'
		assert len(fake.requests) == 2 and not fake.errors
		text_packet, image_packet = fake.requests
		image_state = json.loads(json.dumps(image_packet['state']))
		assert image_state['page'].pop('screenshot') in image_packet['assets']
		assert image_state['page'].pop('viewport')
		guidance = image_state['current_subgoal'].pop('screenshot_guidance')
		assert guidance == {'required': True, 'source': 'escalation', 'reason': run.vision_reason}
		assert image_state == text_packet['state'] and 'assets' not in text_packet
		assert {k: list(q['criteria']) for k, q in text_packet['questions'].items()} == {
			k: list(q['criteria']) for k, q in image_packet['questions'].items()
		}
		assert run.vision and not run.vision_at_start and run.screenshot_declared is False and run.vision_source == 'escalation'
		assert run.decisions == run.vision_decisions == 1
		assert run.escalations == [{'step': run.started_step, 'reason': reason, 'outcome': 'executed'}]
		events = _events(tmp_path / 'trace.jsonl')
		rows = [r for r in events if r['event'] == 'system1_decision']
		assert [(r['outcome'], r['vision_trigger'], r['escalation_from'], r['planner_declared']) for r in rows] == [
			('escalated', None, reason, False),
			('execute', f'escalation:{reason}', reason, False),
		]
		assert next(r for r in events if r['event'] == 'run_config')['vision_escalation'] is True
		assert len(list(conversations.glob('*_escalated.json'))) == 1
		normal = [f for f in conversations.glob('system1_*.json') if not f.stem.endswith('_escalated')]
		assert (
			len(normal) == 1
			and (conversations / json.loads(normal[0].read_text(encoding='utf-8'))['request_images']['page-screenshot']).is_file()
		)
		assert 'System1 switched to images after:' in str((await routed(agent)).context_messages[-1].content)
		assert executor.usage_by_role()['system1']['escalations_executed'] == 1
		assert not (
			await executor.delegate(
				DelegateSubgoalParams(objective='Next', completion='New stage', reason='Advance'), jev_browser
			)
		).error
		await routed(agent)
		assert [r['vision_trigger'] for r in _events(tmp_path / 'trace.jsonl') if r['event'] == 'system1_decision'] == [
			None,
			f'escalation:{reason}',
			f'escalation:{reason}',
			None,
		]
		assert not executor.runs[-1].vision and not executor.runs[-1].escalations
	finally:
		await close_driver(agent)


async def test_image_retry_reuses_targets_after_request_size_halving(jev_browser, httpserver, tmp_path, monkeypatch):
	fake = _VisionJev(httpserver, [LOW_CLICK, {'operation': 'CLICK', 'target': 'Enable'}])
	fillers = ''.join(f'<button aria-label="{i} {"x" * 115}">Other</button>' for i in range(48))
	body = "<button>Plain</button><button onclick=\"document.querySelector('output').value='Enabled'\">Enable</button>"
	agent, executor, _ = await start_stage(
		jev_browser, httpserver, tmp_path, screenshot_policy='smart', max_candidates=32, body=body + fillers + '<output></output>'
	)
	try:
		measured_sizes = []
		request_size = service.decision.request_size

		def measure_request(*args, **kwargs):
			size = request_size(*args, **kwargs)
			if not measured_sizes:
				# Force exactly the 32-target request over budget, independent of Chrome's assigned node IDs.
				monkeypatch.setattr(service, 'MAX_REQUEST_CHARS', size - 1)
			measured_sizes.append(size)
			return size

		monkeypatch.setattr(service.decision, 'request_size', measure_request)
		select_targets = Mock(wraps=service.decision.select_targets)
		monkeypatch.setattr(service.decision, 'select_targets', select_targets)
		assert (await routed(agent)).model_output is not None
		assert await browser_eval(jev_browser, 'document.querySelector("output").value') == 'Enabled'
		text, image = fake.requests
		keys = list(text['questions']['click_target']['criteria'])
		assert 1 < len(keys) < executor.settings.max_candidates
		assert list(image['questions']['click_target']['criteria']) == keys
		limits = [call.args[-1] for call in select_targets.call_args_list]
		assert limits == [32 >> i for i in range(len(limits))] and limits[-1] == len(keys)
		full_text_size, limited_text_size, image_size = measured_sizes
		assert limits == [32, 16]
		assert max(limited_text_size, image_size) <= service.MAX_REQUEST_CHARS < full_text_size
		rows = [r for r in _events(tmp_path / 'trace.jsonl') if r['event'] == 'system1_decision']
		for row, packet in zip(rows, fake.requests, strict=True):
			assert row['request_chars'] == len(json.dumps(packet, ensure_ascii=False)) <= service.MAX_REQUEST_CHARS
			assert row['observed_candidate_counts']['click_target'] > executor.settings.max_candidates
			assert row['candidate_counts']['click_target'] == len(keys)
	finally:
		await close_driver(agent)


async def test_failed_image_choice_uses_second_stop_and_logs_both_checks(jev_browser, httpserver, tmp_path):
	fake = _VisionJev(httpserver, [LOW_CLICK, {'operation': 'UNABLE'}])
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, screenshot_policy='smart')
	try:
		out = await routed(agent)
		run = executor.runs[-1]
		assert len(fake.requests) == 2 and run.handback and run.handback.reason == 'unable_probability'
		assert 'System1 retried with the viewport image after low_confidence; it still stopped: unable_probability.' in str(
			out.context_messages[-1].content
		)
		assert run.escalations[-1]['outcome'] == 'handed_back' and not run.vision and executor.pending_record is None
		assert [
			r['request_vision'] for r in _events(tmp_path / 'trace.jsonl') if r['event'] == 'check' and r['name'] == 'confidence'
		] == [False, True]
	finally:
		await close_driver(agent)


@pytest.mark.parametrize(
	('policy', 'enabled', 'declared', 'disabled', 'expected_vision'),
	[
		('off', True, None, False, False),
		('always', True, None, False, True),
		('smart', False, False, False, False),
		('smart', True, True, False, True),
		('smart', True, False, True, False),
	],
)
async def test_image_retry_policy_gates(policy, enabled, declared, disabled, expected_vision, jev_browser, httpserver, tmp_path):
	fake = _VisionJev(httpserver, [LOW_CLICK])
	agent, executor, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		screenshot_policy=policy,
		vision_escalation=enabled,
		screenshot_fields={'screenshot_required': declared, 'screenshot_reason': 'Review the target.'},
	)
	try:
		run = executor.runs[-1]
		if disabled:
			executor._disable_vision(run, 415)
		await routed(agent)
		assert run.handback and run.handback.reason == 'low_confidence'
		assert len(fake.requests) == 1 and ('assets' in fake.requests[0]) is expected_vision and not run.escalations
		assert next(r for r in _events(tmp_path / 'trace.jsonl') if r['event'] == 'run_config')['vision_escalation'] is enabled
	finally:
		await close_driver(agent)


@pytest.mark.parametrize(
	'reason',
	[
		'invalid_answer',
		'action_error',
		'step_error',
		'submit_not_allowed',
		'risk_not_allowed',
		'risk_denied',
		'out_of_scope',
		'action_budget',
		'text_unavailable',
		'system1_error',
		'too_many_candidates',
		'state_too_large',
		'unsupported_page',
		'wait_limit',
		'step_limit',
		'step_done',
		'repeated_action',
	],
)
def test_non_visual_stops_and_repeated_typing_never_escalate(reason):
	choice = service.decision.Decision(
		operation='TYPE_TEXT', operation_probability=0.95, target_probability=0.95, unable_probability=0.02
	)
	assert service._escalation_reason((reason, 'stop'), choice) is None


async def test_cancelled_image_retry_archives_actual_attempts(jev_browser, httpserver, tmp_path):
	arrived = asyncio.Event()
	release = threading.Event()
	loop = asyncio.get_running_loop()

	def block(request):
		assert request.files['page-screenshot'].read()
		loop.call_soon_threadsafe(arrived.set)
		release.wait(timeout=15)
		return Response('{}', mimetype='application/json')

	image_requests(httpserver).respond_with_handler(block)
	FakeJev(httpserver, [LOW_CLICK])
	conversations = tmp_path / 'conversations'
	agent, executor, _ = await start_stage(
		jev_browser, httpserver, tmp_path, screenshot_policy='smart', save_conversation_path=conversations
	)
	task = asyncio.create_task(routed(agent, execute=False))
	try:
		await asyncio.wait_for(arrived.wait(), 15)
		task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await task
		assert executor.pending_record is None
		assert [r['outcome'] for r in _events(tmp_path / 'trace.jsonl') if r['event'] == 'system1_decision'] == [
			'escalated',
			'interrupted',
		]
		assert len(list(conversations.glob('system1_*.json'))) == 2 and len(list(conversations.glob('system1_*.jpg'))) == 1
		assert executor.runs[-1].escalations[-1]['outcome'] == 'retry_failed'
		usage = executor.usage_by_role()['system1']
		assert (usage['decisions'], usage['vision_requests'], usage['text_requests'], usage['escalations_executed']) == (
			1,
			1,
			1,
			0,
		)
	finally:
		release.set()
		await close_driver(agent)


async def test_repeated_type_text_uses_one_text_helper_and_one_jev_request(jev_browser, httpserver, tmp_path, monkeypatch):
	fake = _VisionJev(httpserver, [{'operation': 'TYPE_TEXT', 'target': 'Town'}])
	agent, executor, _ = await start_stage(
		jev_browser, httpserver, tmp_path, screenshot_policy='smart', body='<label>Town <input></label>'
	)
	try:
		run = executor.runs[-1]
		summary = await agent._prepare_context(AgentStepInfo(0, 10))
		obs = await service.state.observe(jev_browser, summary, executor.settings, executor.step)
		target = obs.candidates['TYPE_TEXT'][0]
		for n in (1, 2):
			run.actions.append(
				ActionRecord(
					n=n,
					step=1,
					run_id=run.id,
					operation='TYPE_TEXT',
					target_key=target.key,
					bu_action={'input': {'index': target.index, 'text': 'Luzern'}},
					text='Luzern',
					fingerprint_before=obs.fingerprint,
					url_before=obs.url,
					status='executed_ok',
				)
			)
		helper = AsyncMock(return_value='Luzern')
		monkeypatch.setattr(executor.helpers, 'text', helper)
		await routed(agent, summary=summary)
		assert len(fake.requests) == 1 and helper.await_count == 1 and not run.escalations
		assert run.handback and run.handback.reason == 'repeated_action'
	finally:
		await close_driver(agent)


async def test_discarded_upgraded_prediction_is_not_an_execution(jev_browser, httpserver, tmp_path):
	fake = _VisionJev(httpserver, [LOW_CLICK, {'operation': 'CLICK', 'target': 'Enable'}])
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, screenshot_policy='smart')
	try:
		assert (await routed(agent, execute=False)).model_output is not None
		assert len(fake.requests) == 2 and executor.usage_by_role()['system1']['escalations_executed'] == 0
		await executor.aclose()
		run = executor.runs[-1]
		assert not run.vision_at_start and run.vision
		assert _is_repeated_empty([run, run], run.identity, False) and not _is_repeated_empty([run, run], run.identity, True)
		assert run.escalations[-1]['outcome'] == 'retry_failed'
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == 'Default'
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('budget_exhausted', [False, True])
async def test_no_effect_once_checks_remaining_budget(budget_exhausted, jev_browser, httpserver, tmp_path):
	fake = _VisionJev(httpserver, [{'operation': 'CLICK', 'target': 'Plain'}])
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, screenshot_policy='smart')
	try:
		run = executor.runs[-1]
		run.no_effect_streak = 2
		if budget_exhausted:
			run.actions.append(
				ActionRecord(
					n=1,
					step=1,
					run_id=run.id,
					operation='WAIT',
					bu_action={'wait': {}},
					fingerprint_before='old',
					url_before=run.start_url,
					status='executed_ok',
				)
			)
			run.max_actions = 1
		await routed(agent)
		assert run.no_effect_escalated and run.vision and run.vision_source == 'escalation'
		assert executor.usage_by_role()['system1']['escalations_by_reason'] == {'no_effect': 1}
		if budget_exhausted:
			assert run.handback and run.handback.reason == 'action_budget' and not fake.requests
			assert run.escalations[-1]['outcome'] == 'handed_back'
		else:
			assert (
				len(fake.requests) == 1
				and fake.requests[0]['state']['current_subgoal']['screenshot_guidance']['source'] == 'escalation'
			)
			assert run.escalations[-1]['outcome'] == 'executed'
			run.no_effect_streak = 2
			await routed(agent)
			assert run.handback and run.handback.reason == 'no_effect' and len(fake.requests) == 1 and len(run.escalations) == 1
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('failure', ['missing', 415, 503])
async def test_image_retry_fails_without_third_request(failure, jev_browser, httpserver, tmp_path, monkeypatch):
	monkeypatch.setattr(backends, 'JEV_RETRY_DELAYS', (0, 0))
	if isinstance(failure, int):
		image_requests(httpserver).respond_with_data(status=failure)
	else:
		monkeypatch.setattr(service, 'prepare_screenshot', Mock(return_value=None))
	FakeJev(httpserver, [LOW_CLICK])
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, screenshot_policy='smart')
	try:
		await routed(agent)
		run = executor.runs[-1]
		assert run.handback and run.handback.reason == 'low_confidence' and run.decisions == 1 and not run.vision_decisions
		types = [r.mimetype for r, _ in httpserver.log if r.path == '/v1/systemone']
		count = {'missing': 0, 415: 1, 503: 3}[failure]
		assert types == ['application/json'] + ['multipart/form-data'] * count
		outcome = {'missing': 'vision_unavailable', 415: 'vision_rejected', 503: 'retry_failed'}[failure]
		assert run.escalations[0]['outcome'] == outcome
		assert (executor._vision_disabled is not None) == (failure == 415)
		if failure == 503:
			assert 'Retrying with the image failed:' in run.handback.detail
		if failure == 'missing':
			assert any(
				r['event'] == 'vision_unavailable' and r['vision_trigger'] == 'escalation:low_confidence'
				for r in _events(tmp_path / 'trace.jsonl')
			)
	finally:
		await close_driver(agent)


@pytest.mark.parametrize(
	('previous_reason', 'rejected', 'review'),
	[(None, False, False), ('step_done', False, False), ('step_done', True, True)]
	+ [
		(reason, False, True)
		for reason in ('unable', 'low_confidence', 'no_effect', 'repeated_action', 'action_budget', 'system1_error')
	],
)
def test_review_delegation(previous_reason, rejected, review):
	previous = None
	if previous_reason is not None:
		previous = SubgoalRun(
			id='sg1',
			identity='stage',
			params=DelegateSubgoalParams(objective='stage', completion='done', reason='review'),
			max_actions=1,
			started_step=1,
			start_url='http://localhost',
			start_site='localhost',
			handback=Handback(reason=previous_reason, detail='observed', step=2),
		)
	assert _is_review_delegation(previous, rejected) is review


async def test_smart_sends_each_stage_its_declared_input(jev_browser, httpserver, tmp_path, monkeypatch):
	reason = 'Find Enable in the correct row.'
	fake = _VisionJev(httpserver, [{'operation': 'CLICK', 'target': 'Enable'}] + [{'operation': 'STEP_DONE'}] * 3)
	agent, executor, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		screenshot_policy='smart',
		screenshot_fields={'screenshot_required': True, 'screenshot_reason': reason},
	)
	triggers = []
	ask = executor.backend.ask

	async def capture(state, questions, *, images=None):
		triggers.append(executor.current_vision_trigger)
		return await ask(state, questions, images=images)

	monkeypatch.setattr(executor.backend, 'ask', capture)
	try:
		await routed(agent)
		done = await routed(agent)
		assert f'2 of 2 System1 decisions in this subgoal used the viewport image (requested by you: {reason})' in str(
			done.context_messages[-1].content
		)
		assert REVIEW_REMINDER not in str(done.context_messages[-1].content)
		for n, declaration in enumerate(({}, {'screenshot_required': False, 'screenshot_reason': 'Label is sufficient.'}), 1):
			args = {'objective': f'Read stage {n}', 'completion': 'Page visible', 'reason': 'Advanced', **declaration}
			out = await agent.tools.registry.execute_action('delegate_subgoal', args, browser_session=jev_browser)
			assert not out.error
			result = await routed(agent)
			assert '(text-only stage)' in str(result.context_messages[-1].content)
		assert triggers == ['planner', 'planner', None, None] and executor.current_vision_trigger is None
		assert [req.mimetype for req, _ in httpserver.log if req.path == '/v1/systemone'] == [
			'multipart/form-data',
			'multipart/form-data',
			'application/json',
			'application/json',
		]
		for packet in fake.requests[:2]:
			assert packet['state']['current_subgoal']['screenshot_guidance'] == {
				'required': True,
				'reason': reason,
				'source': 'planner',
			}
			assert packet['state']['page']['screenshot'] in packet['assets']
		for packet in fake.requests[2:]:
			assert 'screenshot_guidance' not in packet['state']['current_subgoal']
			assert 'assets' not in packet and 'screenshot' not in packet['state']['page']
		events = _events(tmp_path / 'trace.jsonl')
		assert [r['vision_trigger'] for r in events if r['event'] == 'system1_decision'] == triggers
		starts = [r for r in events if r['event'] == 'subgoal_start']
		assert [
			(r['screenshot_required'], r['vision'], r['review_delegation'], r['previous_handback_reason']) for r in starts
		] == [(True, True, False, None), (None, False, False, 'step_done'), (False, False, False, 'step_done')]
		assert not any(r['event'] == 'screenshot_decision_missing' for r in events)
		assert not fake.errors
	finally:
		await close_driver(agent)


async def test_smart_review_missing_fields_and_changed_vision(jev_browser, httpserver, tmp_path):
	fake = _VisionJev(httpserver, [{'operation': op} for op in ('UNABLE', 'UNABLE', 'STEP_DONE', 'UNABLE')])
	agent, executor, params = await start_stage(
		jev_browser, httpserver, tmp_path, screenshot_policy='smart', vision_escalation=False, checks=JevChecks(confidence='off')
	)
	try:
		first = await routed(agent)
		assert REVIEW_REMINDER in str(first.context_messages[-1].content)
		assert not (await executor.delegate(params, jev_browser)).error
		assert executor.runs[-1].review_delegation and executor.runs[-1].screenshot_declared is None
		await routed(agent)
		assert (await executor.delegate(params, jev_browser)).error

		requested = DelegateSubgoalParamsSmart.model_validate(
			{**params.model_dump(), 'screenshot_required': True, 'screenshot_reason': '  '}
		)
		assert not (await executor.delegate(requested, jev_browser)).error
		done = await routed(agent)
		assert REVIEW_REMINDER not in str(done.context_messages[-1].content)
		default_reason = 'Planner requested the image without a specific reason.'
		assert fake.requests[2]['state']['current_subgoal']['screenshot_guidance']['reason'] == default_reason
		assert f'(requested by you: {default_reason})' in str(done.context_messages[-1].content)
		assert (await executor.delegate(requested, jev_browser)).error
		rejected = requested.model_copy(
			update={'screenshot_required': False, 'screenshot_reason': None, 'review_note': 'The result is still missing.'}
		)
		assert not (await executor.delegate(rejected, jev_browser)).error
		assert executor.runs[-1].review_delegation and executor.runs[-1].screenshot_declared is False
		await routed(agent)
		assert 'assets' not in fake.requests[-1]
		executor.on_new_task()
		assert not (await executor.delegate(params, jev_browser)).error
		assert not executor.runs[-1].review_delegation
		events = _events(tmp_path / 'trace.jsonl')
		assert [(r['run_id'], r['previous_handback_reason']) for r in events if r['event'] == 'screenshot_decision_missing'] == [
			('sg2', 'unable')
		]
		assert [(r['run_id'], r['screenshot_required']) for r in events if r['event'] == 'screenshot_reason_missing'] == [
			('sg3', True),
			('sg4', False),
		]
		assert [r['reason'] for r in events if r['event'] == 'delegate_refused'] == ['no_action_twice', 'review_note_missing']
		assert not fake.errors
	finally:
		await close_driver(agent)


async def test_smart_image_rejection_overrides_later_planner_requests(jev_browser, httpserver, tmp_path):
	fake = _VisionJev(httpserver, [{'operation': 'CLICK', 'target': 'Enable'}, {'operation': 'UNABLE'}, {'operation': 'UNABLE'}])
	agent, executor, params = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		screenshot_policy='smart',
		screenshot_fields={'screenshot_required': True, 'screenshot_reason': 'Identify the row by color.'},
	)
	try:
		assert (await routed(agent)).model_output is not None
		httpserver.expect_oneshot_request(
			'/v1/systemone', method='POST', headers={'TypeSafe-Danger-Zone-Images': 'true'}
		).respond_with_data(status=415)
		first = await routed(agent)
		assert 'disabled for this task after Jev rejected an image (HTTP 415); 1 of 2' in str(first.context_messages[-1].content)
		assert REVIEW_REMINDER not in str(first.context_messages[-1].content)
		assert not (await executor.delegate(params, jev_browser)).error
		assert executor.runs[-1].screenshot_declared is True
		second = await routed(agent)
		assert 'disabled for this task after Jev rejected an image (HTTP 415); 0 of 1' in str(second.context_messages[-1].content)
		assert REVIEW_REMINDER not in str(second.context_messages[-1].content)
		rows = [r for r in _events(tmp_path / 'trace.jsonl') if r['event'] == 'system1_decision']
		assert [r['vision_trigger'] for r in rows] == ['planner', 'planner', None, None]
		assert rows[1]['outcome'] == rows[2]['fallback_from'] == 'vision_rejected'
		assert 'screenshot_guidance' not in fake.requests[1]['state']['current_subgoal']
		assert [req.mimetype for req, _ in httpserver.log if req.path == '/v1/systemone'] == [
			'multipart/form-data',
			'multipart/form-data',
			'application/json',
			'application/json',
		]
		assert executor.current_vision_trigger is None and not fake.errors
	finally:
		await close_driver(agent)


async def test_rejected_image_falls_back_to_text_for_the_task(jev_browser, httpserver, tmp_path, monkeypatch):
	wire_jpeg = tmp_path / 'received-image.jpg'
	wire_image_request = tmp_path / 'received-image-request.json'

	def reject(request):
		wire_jpeg.write_bytes(request.files['page-screenshot'].read())
		wire_image_request.write_text(request.form['request'], encoding='utf-8')
		return Response(status=415)

	image_requests(httpserver).respond_with_handler(reject)
	click = {'operation': 'CLICK', 'target': 'Enable'}
	fake = FakeJev(httpserver, [click, click, {'operation': 'STEP_DONE'}, {'operation': 'CLICK', 'target': 'Reset'}])
	conversations = tmp_path / 'conversations'
	agent, executor, params = await start_stage(
		jev_browser, httpserver, tmp_path, screenshot_policy='always', save_conversation_path=conversations
	)
	off_agent = Agent(
		task=agent.task,
		llm=create_mock_llm(),
		browser_session=jev_browser,
		use_judge=False,
		jev=executor.settings.model_copy(update={'screenshot_policy': 'off', 'trace_path': tmp_path / 'off.jsonl'}),
	)
	try:
		# Send a real off request against this exact summary without executing its prediction.
		assert off_agent._jev is not None
		await off_agent._jev.delegate(params, jev_browser)
		summary = await agent._prepare_context(AgentStepInfo(0, 30))
		assert (await off_agent._jev.route(off_agent, summary)).model_output is not None
		off_body = httpserver.log[-1][0].get_data()
		(tmp_path / 'received-off.json').write_bytes(off_body)
		cut = len(httpserver.log)

		observe = AsyncMock(wraps=service.state.observe)
		prepare = Mock(wraps=service.prepare_screenshot)
		monkeypatch.setattr(service.state, 'observe', observe)
		monkeypatch.setattr(service, 'prepare_screenshot', prepare)
		run = executor.runs[-1]
		step = executor.step
		assert (await routed(agent, summary=summary)).model_output is not None
		observe.assert_awaited_once_with(jev_browser, summary, executor.settings, step)
		prepare.assert_called_once()
		assert (run.decisions, run.vision_decisions) == (1, 0) and run.handback is None
		assert [action.operation for action in run.executed_actions] == ['CLICK']
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == 'Enabled'

		received = [(req, res.status_code) for req, res in httpserver.log[cut:] if req.path == '/v1/systemone']
		assert [(req.mimetype, status) for req, status in received] == [('multipart/form-data', 415), ('application/json', 200)]
		text_body = received[1][0].get_data()
		(tmp_path / 'received-fallback.json').write_bytes(text_body)
		assert text_body == off_body
		wire = json.loads(text_body)
		assert 'assets' not in wire and not {'screenshot', 'viewport'} & wire['state']['page'].keys()
		assert all(
			'rect' not in item for key, q in wire['questions'].items() if key != 'operation' for item in q['criteria'].values()
		)
		assert VISION_RULES not in wire['questions']['operation']['instructions']
		events = _events(tmp_path / 'trace.jsonl')
		assert not any(row['event'] == 'handback' for row in events)
		assert [
			(r['event'], r['status'], r['run_id'], r['step'])
			for r in events
			if r['event'] in {'vision_rejected', 'vision_disabled'}
		] == [('vision_rejected', 415, run.id, step), ('vision_disabled', 415, run.id, step)]
		rejected, fallback = (r for r in events if r['event'] == 'system1_decision')
		assert (rejected['outcome'], rejected['vision'], rejected['vision_trigger']) == ('vision_rejected', True, 'policy')
		assert (fallback['outcome'], fallback['vision'], fallback['vision_trigger'], fallback['fallback_from']) == (
			'execute',
			False,
			None,
			'vision_rejected',
		)
		assert rejected['image_bytes'] == wire_jpeg.stat().st_size and rejected['rect_count'] > 0
		assert fallback['image_bytes'] == fallback['rect_count'] == 0 and fallback['image_size'] is None

		saved = [json.loads(path.read_text(encoding='utf-8')) for path in conversations.glob('system1_*.json')]
		assert len(saved) == 2
		records = {r['decision']['outcome']: r for r in saved}
		image_record = records['vision_rejected']
		assert image_record['request'] == json.loads(wire_image_request.read_bytes())
		ref = image_record['request']['state']['page']['screenshot']
		part = image_record['request']['assets'][ref]['part']
		assert (conversations / image_record['request_images'][part]).read_bytes() == wire_jpeg.read_bytes()
		assert records['execute']['request'] == wire and not records['execute'].get('request_images')
		usage = executor.usage_by_role()['system1']
		assert (
			usage['decisions'],
			usage['calls'],
			usage['vision_requests'],
			usage['text_requests'],
			usage['vision_rejections'],
		) == (1, 2, 1, 1, 1)
		assert usage['vision_disabled'] == {'status': 415, 'step': step}

		prepare.reset_mock()
		next_step = await routed(agent)
		assert next_step.model_output is None and run.handback and run.handback.reason == 'step_done'
		assert 'HTTP 415); 0 of 2 System1 decisions' in next_step.context_messages[-1].content
		await close_driver(agent)
		agent.add_new_task('Reset the controls')
		assert executor.active_run is None
		assert not (await executor.delegate(params, jev_browser)).error
		assert (await routed(agent)).model_output is not None
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == 'Default'
		prepare.assert_not_called()
		events = _events(tmp_path / 'trace.jsonl')
		assert [r['event'] for r in events if r['event'].startswith('vision_')] == ['vision_rejected', 'vision_disabled']
		subsequent = [r for r in events if r['event'] == 'system1_decision'][2:]
		assert len(subsequent) == 2 and all(
			r['vision'] is False and r['vision_trigger'] is None and r['vision_disabled'] is True for r in subsequent
		)
		assert len(fake.requests) == 4 and not fake.errors
		assert fake.requests[-1]['state']['original_goal'] == 'Reset the controls'
		assert executor.usage_by_role()['system1']['vision_disabled'] == usage['vision_disabled']
	finally:
		await close_driver(off_agent)
		await close_driver(agent)


@pytest.mark.parametrize('status', [400, 413, 415])
async def test_last_attempt_image_rejection_falls_back(jev_browser, httpserver, tmp_path, monkeypatch, status):
	# The first attempt failed transiently. Only the final status decides whether to disable images.
	monkeypatch.setattr(backends, 'JEV_RETRY_DELAYS', (0, 0))
	httpserver.expect_oneshot_request('/v1/systemone', method='POST').respond_with_data(status=503)
	image_requests(httpserver).respond_with_data(status=status)
	fake = FakeJev(httpserver, [{'operation': 'CLICK', 'target': 'Enable'}])
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, screenshot_policy='always')
	try:
		assert (await routed(agent)).model_output is not None
		assert len(fake.requests) == 1 and not fake.errors
		usage = executor.usage_by_role()['system1']
		assert (
			usage['decisions'],
			usage['calls'],
			usage['vision_requests'],
			usage['text_requests'],
			usage['vision_rejections'],
		) == (1, 3, 2, 1, 1)
		assert usage['vision_disabled']['status'] == status
	finally:
		await close_driver(agent)


@pytest.mark.parametrize(
	('failure', 'attempts'), [(401, 1), (403, 1), (408, 1), (429, 3), (500, 1), (503, 3), ('transport', 3), ('malformed', 1)]
)
async def test_other_image_failures_keep_vision(jev_browser, httpserver, tmp_path, monkeypatch, failure, attempts):
	monkeypatch.setattr(backends, 'JEV_RETRY_DELAYS', (0, 0))
	agent, executor, params = await start_stage(jev_browser, httpserver, tmp_path, screenshot_policy='always')
	transports = []

	def accepted_image(request):
		answers = {}
		for name, question in json.loads(request.form['request'])['questions'].items():
			keys = question['criteria']
			choice = 'STEP_DONE' if name == 'operation' else next(iter(keys))
			answers[name] = {'choice': choice, 'confidence': 1, 'probabilities': {key: float(key == choice) for key in keys}}
		return Response(json.dumps({'answers': answers}), mimetype='application/json')

	try:
		with monkeypatch.context() as network:
			if failure == 'transport':

				async def disconnect(request):
					transports.append(request.headers['Content-Type'].split(';')[0])
					raise httpx.ReadError('connection closed', request=request)

				assert isinstance(executor.backend, backends.JevBackend)
				network.setattr(executor.backend._client, 'send', disconnect)
			else:
				for _ in range(attempts):
					httpserver.expect_oneshot_request('/v1/systemone', method='POST').respond_with_data(
						'not JSON' if failure == 'malformed' else '', status=200 if failure == 'malformed' else failure
					)
			handback = await routed(agent)
			assert handback.model_output is None and REVIEW_REMINDER not in str(handback.context_messages[-1].content)
		failed = executor.runs[-1]
		assert failed.handback and failed.handback.reason == 'system1_error'
		assert (failed.decisions, failed.vision_decisions) == (1, 0)

		image_requests(httpserver).respond_with_handler(accepted_image)
		assert not (await executor.delegate(params, jev_browser)).error
		accepted = await routed(agent)
		run = executor.runs[-1]
		assert run.handback and run.handback.reason == 'step_done'
		assert (run.decisions, run.vision_decisions) == (1, 1)
		assert (
			'Screenshots: 1 of 1 System1 decisions in this subgoal used the viewport image.'
			in accepted.context_messages[-1].content
		)
		request_types = transports + [req.mimetype for req, _ in httpserver.log if req.path == '/v1/systemone']
		assert request_types == ['multipart/form-data'] * (attempts + 1)
		usage = executor.usage_by_role()['system1']
		assert usage['vision_requests'] == usage['calls'] == attempts + 1
		assert usage['text_requests'] == usage['vision_rejections'] == 0 and usage['vision_disabled'] is None
		events = _events(tmp_path / 'trace.jsonl')
		assert not any(r['event'] in ('vision_disabled', 'vision_rejected') for r in events)
	finally:
		await close_driver(agent)


@pytest.mark.parametrize(('status', 'attempts'), [(500, 1), (503, 3)])
async def test_fallback_text_failure_reports_image_rejection(jev_browser, httpserver, tmp_path, monkeypatch, status, attempts):
	monkeypatch.setattr(backends, 'JEV_RETRY_DELAYS', (0, 0))
	image_requests(httpserver).respond_with_data(status=415)
	httpserver.expect_request('/v1/systemone', method='POST').respond_with_data(status=status)
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, screenshot_policy='always')
	try:
		out = await routed(agent)
		run = executor.runs[-1]
		assert out.model_output is None and run.handback and run.handback.reason == 'system1_error'
		assert run.handback.detail.endswith('Screenshots were disabled after Jev rejected an image (HTTP 415).')
		assert (
			'Screenshots: disabled for this task after Jev rejected an image (HTTP 415); 0 of 1 System1 decisions in this subgoal used the viewport image.'
			in out.context_messages[-1].content
		)
		assert not run.actions and executor.pending_record is None
		usage = executor.usage_by_role()['system1']
		assert (usage['decisions'], usage['vision_requests'], usage['text_requests'], usage['vision_rejections']) == (
			1,
			1,
			attempts,
			1,
		)
		assert usage['vision_disabled']['status'] == 415
	finally:
		await close_driver(agent)


async def test_cancelled_fallback_is_archived_as_text(jev_browser, httpserver, tmp_path):
	arrived = asyncio.Event()
	release = threading.Event()
	loop = asyncio.get_running_loop()

	def wait_for_cancellation(request):
		assert request.mimetype == 'application/json'
		loop.call_soon_threadsafe(arrived.set)
		release.wait(timeout=10)
		return Response('{}', mimetype='application/json')

	image_requests(httpserver).respond_with_data(status=415)
	httpserver.expect_request('/v1/systemone', method='POST').respond_with_handler(wait_for_cancellation)
	conversations = tmp_path / 'conversations'
	agent, executor, _ = await start_stage(
		jev_browser, httpserver, tmp_path, screenshot_policy='always', save_conversation_path=conversations
	)
	task = asyncio.create_task(routed(agent, execute=False))
	try:
		await asyncio.wait_for(arrived.wait(), timeout=10)
		task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await task
		rejected, interrupted = (r for r in _events(tmp_path / 'trace.jsonl') if r['event'] == 'system1_decision')
		assert rejected['outcome'] == 'vision_rejected' and rejected['vision'] is True
		assert interrupted['outcome'] == 'interrupted' and interrupted['fallback_from'] == 'vision_rejected'
		assert interrupted['vision'] is False and interrupted['image_bytes'] == 0 and interrupted['vision_trigger'] is None
		assert [a['error_type'] for a in interrupted['attempts']] == ['CancelledError']
		records = [json.loads(path.read_text(encoding='utf-8')) for path in conversations.glob('system1_*.json')]
		assert len(records) == 2
		interrupted_record = next(r for r in records if r['decision']['outcome'] == 'interrupted')
		assert 'assets' not in interrupted_record['request'] and not interrupted_record.get('request_images')
		assert executor.pending_record is None and not executor.runs[-1].actions
		usage = executor.usage_by_role()['system1']
		assert (usage['decisions'], usage['calls'], usage['vision_requests'], usage['text_requests']) == (1, 2, 1, 1)
	finally:
		release.set()
		task.cancel()
		await asyncio.gather(task, return_exceptions=True)
		await close_driver(agent)


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
			handback_result = await routed(agent)
			if handback_result.model_output is None:
				assert REVIEW_REMINDER not in str(handback_result.context_messages[-1].content)
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


@pytest.mark.parametrize('case', ['step_limit', 'step_error', 'out_of_scope', 'candidate_limit', 'state_too_large'])
async def test_prechecks_and_candidate_limit(case, jev_browser, httpserver, tmp_path):
	agent, e, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		max_candidates=1 if case == 'candidate_limit' else 200,
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
		handback = e.runs[-1].handback
		if case == 'candidate_limit':
			assert handback and handback.reason == 'unable_probability'
			assert len(fake.requests) == 1
			assert len(fake.requests[0]['questions']['click_target']['criteria']) == 1
		else:
			assert handback and handback.reason == case
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
