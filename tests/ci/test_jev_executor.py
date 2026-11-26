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
from browser_use.benchmark.tasks import INSPECT_PAGE
from browser_use.browser.events import TypeTextEvent
from browser_use.browser.session import BrowserSession
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


@pytest.mark.parametrize(
	('filler_count', 'operation'), [(30, 'CLICK'), (215, 'CLICK'), (402, 'TYPE_TEXT'), (619, 'CLICK'), (619, 'TYPE_TEXT')]
)
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


async def test_input_does_not_emit_blur_until_the_browser_changes_focus(jev_browser, httpserver):
	url = serve_page(
		httpserver,
		'/inline-search',
		'<input id="query" onblur="window.closedByBlur=true" '
		'onkeydown="if (event.key===\'Enter\' && !window.closedByBlur) out.value=this.value">'
		'<input id="other"><output id="out"></output>',
	)
	await jev_browser.navigate_to(url)
	summary = await jev_browser.get_browser_state_summary(include_screenshot=False)
	node = next(n for n in summary.dom_state.selector_map.values() if n.attributes.get('id') == 'query')
	event = jev_browser.event_bus.dispatch(TypeTextEvent(node=node, text='会议', clear=True))
	await event.event_result(raise_if_any=True, raise_if_none=False)
	assert await browser_eval(jev_browser, '[document.activeElement===query, Boolean(window.closedByBlur)]') == [True, False]
	agent = Agent(task='Submit an inline search', llm=create_mock_llm(), browser_session=jev_browser)
	result = await agent.multi_act([agent.ActionModel.model_validate({'send_keys': {'keys': 'Enter'}})])
	assert not result[-1].error and await browser_eval(jev_browser, 'out.value') == '会议'
	assert await browser_eval(jev_browser, 'other.focus(); Boolean(window.closedByBlur)') is True


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
	capture = jev_browser.get_browser_state_summary
	executor = agent._jev
	assert executor is not None
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


@pytest.mark.parametrize(
	('votes', 'expected'),
	[
		({'SCROLL_DOWN': 0.40, 'COLLECT': 0.37, 'WAIT': 0.01, 'UNABLE': 0.22}, 'scroll'),
		({'SCROLL_DOWN': 0.71, 'COLLECT': 0.01, 'CLICK': 0.01, 'WAIT': 0.15, 'UNABLE': 0.12}, 'scroll'),
		({'SCROLL_DOWN': 0.40, 'COLLECT': 0.33, 'WAIT': 0.01, 'UNABLE': 0.26}, 'unable_probability'),
		({'SCROLL_DOWN': 0.34, 'COLLECT': 0.10, 'CLICK': 0.31, 'WAIT': 0.09, 'UNABLE': 0.16}, 'low_confidence'),
	],
)
async def test_read_confidence_distinguishes_abstention_from_other_actions(jev_browser, httpserver, tmp_path, votes, expected):
	step = {'operation': 'SCROLL_DOWN', 'target': 'whole page', 'operation_probs': votes}
	fake = FakeJev(httpserver, [step, step])
	agent, executor, _ = await start_stage(
		jev_browser, httpserver, tmp_path, body='<button>Details</button><main style="height:300vh">Article</main>'
	)
	try:
		response = await routed(agent)
		assert not fake.errors
		if expected == 'scroll':
			assert response.model_output is not None and len(fake.requests) == 1
			assert await browser_eval(jev_browser, 'scrollY') > 0
			row = next(r for r in _events(tmp_path / 'trace.jsonl') if r['event'] == 'system1_decision')
			assert row['read_operation_probability'] == pytest.approx(votes['SCROLL_DOWN'] + votes['COLLECT'])
			assert row['operation_score'] >= 0.8
		else:
			handback = executor.runs[-1].handback
			assert response.model_output is None and handback is not None and handback.reason == expected
			assert executor.task_actions == []
			if expected == 'unable_probability':
				assert len(fake.requests) == 1
	finally:
		await close_driver(agent)


async def test_low_confidence_gets_one_recheck_before_handback(jev_browser, httpserver, tmp_path):
	fake = FakeJev(
		httpserver,
		[
			{'operation': 'CLICK', 'target': 'Enable', 'p': 0.55, 'target_p': 0.95},
			{'operation': 'CLICK', 'target': 'Enable', 'p': 0.95},
		],
	)
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path)
	try:
		assert (await routed(agent)).model_output is not None
		assert len(fake.requests) == 2 and not fake.errors
		assert (
			fake.requests[0]['questions']['click_target']['criteria'] == fake.requests[1]['questions']['click_target']['criteria']
		)
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == 'Enabled'
		assert len(executor.runs[-1].executed_actions) == 1
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('abstain', [False, True])
async def test_confident_click_operation_checks_the_ambiguous_target_again(jev_browser, httpserver, tmp_path, abstain):
	fake = FakeJev(
		httpserver,
		[
			{'operation': 'CLICK', 'target': 'Enable', 'target_p': 0.6},
			{'operation': 'CLICK', 'target': 'None of these targets is supported' if abstain else 'Enable'},
		],
	)
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path)
	try:
		result = await routed(agent)
		assert not fake.errors and len(fake.requests) == 2
		review = fake.requests[1]['questions']['click_target']
		assert review['criteria']['__no_grounded_target__'] == 'None of these targets is supported'
		assert (result.model_output is not None) is not abstain
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == (
			'Default' if abstain else 'Enabled'
		)
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('separate', [False, True])
async def test_click_review_groups_only_the_same_native_pointer_event(jev_browser, httpserver, tmp_path, monkeypatch, separate):
	from browser_use.agent.jev import state

	monkeypatch.setattr(state, '_drop_click_wrappers', lambda clicks, _nodes: clicks)
	step = {'operation': 'CLICK', 'target': {'contains_same_label': 'true'}}
	fake = FakeJev(httpserver, [{**step, 'target_p': 0.55}, step])
	width = '50%' if separate else '100%'
	agent, executor, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		body='<style>.control{cursor:pointer;width:280px;height:32px}.caption{display:block;height:32px}</style>'
		'<div class="control" onclick="out.value+=\'Clicked\'">'
		f'<span class="caption" style="width:{width}">Open</span></div><output id="out"></output>',
	)
	try:
		assert (await routed(agent)).model_output is not None and not fake.errors
		before, review = (request['questions']['click_target']['criteria'] for request in fake.requests)
		count = lambda choices: sum(isinstance(c, dict) and '"Open"' in c['element'] for c in choices.values())
		assert count(before) == 2 and count(review) == (2 if separate else 1)
		assert await browser_eval(jev_browser, 'out.value') == 'Clicked'
		assert len(executor.task_actions) == 1
	finally:
		await close_driver(agent)


async def test_reviewed_click_is_discarded_if_point_moves_before_dispatch(jev_browser, httpserver, tmp_path):
	step = {'operation': 'CLICK', 'target': 'Enable'}
	FakeJev(httpserver, [{**step, 'target_p': 0.6}, step])
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path)
	try:
		decision = await routed(agent, execute=False)
		assert decision.model_output is not None and executor.has_reviewed_click()
		await browser_eval(jev_browser, 'document.querySelector("button").style.marginTop="50px"')
		agent.state.last_model_output = decision.model_output
		await agent._execute_actions()
		assert agent.state.last_result
		error = agent.state.last_result[-1].error
		assert error and 'reviewed click target changed' in error
		assert executor.task_actions[-1].status == 'discarded'
		assert await browser_eval(jev_browser, 'document.querySelector("output").textContent') == 'Default'
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('answer', ['container', 'unknown', 'outside'])
async def test_planner_resolves_a_persisting_control_label_split(jev_browser, httpserver, tmp_path, monkeypatch, answer):
	from browser_use.agent.jev import state
	from browser_use.agent.jev.helpers import ClickTargetReply

	monkeypatch.setattr(state, '_drop_click_wrappers', lambda clicks, _nodes: clicks)
	step = {'operation': 'CLICK', 'target': {'contains_same_label': 'true'}, 'target_p': 0.55}
	fake = FakeJev(httpserver, [step, step])
	url = '<style>.control{cursor:pointer;width:280px;height:32px}.caption{display:block;width:50%;height:32px}</style>'
	agent, executor, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		body=url + '<div class="control" onclick="out.value+=\'Opened\'"><span class="caption">Open</span></div>'
		'<output id="out"></output>',
	)
	chosen = []

	async def resolve(_goal, _run, _obs, candidates):
		key = next(k for k, c in candidates.items() if c.get('context', {}).get('contains_same_label'))
		chosen.append(key)
		return ClickTargetReply(
			key={'container': key, 'unknown': None, 'outside': 'not-a-candidate'}[answer], reason='Observed menu container'
		)

	try:
		monkeypatch.setattr(executor.helpers, 'review_click_target', resolve)
		response = await routed(agent, execute=False)
		assert chosen and len(chosen) == 1 and not fake.errors
		assert (response.model_output is not None) == (answer == 'container')
		assert executor.planner_reviewed_click_key() == (chosen[0] if response.model_output is not None else None)
		if response.model_output is not None:
			agent.state.last_model_output = response.model_output
			await agent._execute_actions()
		assert await browser_eval(jev_browser, 'out.value') == ('Opened' if answer == 'container' else '')
	finally:
		await close_driver(agent)


async def test_workbench_waits_for_a_live_delayed_body(jev_browser, httpserver):
	from browser_use.benchmark.engine import _BoundBrowser

	await jev_browser.navigate_to(serve_page(httpserver, '/body-replaced', '<main>Original document</main>'))
	bound = _BoundBrowser(None)
	bound.native = jev_browser
	target = await bound.bind()
	await browser_eval(
		jev_browser,
		'(function(){const body=document.body;body.remove(); setTimeout(()=>document.documentElement.append(body),180);})()',
	)
	first = await bound.send('Runtime.evaluate', {'expression': 'document.body === null', 'returnByValue': True})
	assert first['result']['value'] is True
	assert 'Original document' in (await bound._read_page(target))['text']


@pytest.mark.parametrize(('target_probability', 'expected'), [(0.55, ''), (0.95, 'Quarterly update')])
async def test_same_label_does_not_merge_different_item_actions(jev_browser, httpserver, tmp_path, target_probability, expected):
	step = {'operation': 'CLICK', 'target': {'nearby_text': 'Quarterly update'}, 'target_p': target_probability}
	fake = FakeJev(httpserver, [step, step])
	cards = ''.join(
		f'<div><h2>{title}</h2><div><button onclick="out.value=\'{title}\'">Expand</button></div></div>'
		for title in ('Weekly review', 'Quarterly update')
	)
	agent, executor, _ = await start_stage(jev_browser, httpserver, tmp_path, body=f'<main>{cards}</main><output id="out">')
	try:
		result = await routed(agent)
		assert not fake.errors
		assert (result.model_output is not None) == bool(expected)
		assert await browser_eval(jev_browser, 'document.querySelector("output").value') == expected
		if not expected:
			handback = executor.runs[-1].handback
			assert handback is not None and handback.reason == 'low_confidence'
			assert executor.task_actions == []
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('case', ['same', 'different', 'uncertain_focus', 'unexpected_newline'])
async def test_typing_may_include_votes_for_focusing_the_same_field(jev_browser, httpserver, tmp_path, monkeypatch, case):
	step = {
		'operation': 'TYPE_TEXT',
		'operation_probs': {'TYPE_TEXT': 0.71, 'CLICK': 0.28, 'WAIT': 0.01},
		'target': 'Query',
		'target_p': 1.0,
		'focus_target': 'Other' if case == 'different' else 'Query',
		'focus_target_p': 0.6 if case == 'uncertain_focus' else 0.85,
	}
	fake = FakeJev(httpserver, [step, step])
	agent, executor, _ = await start_stage(
		jev_browser, httpserver, tmp_path, body='<input id="query" aria-label="Query"><button>Other</button>'
	)

	async def provide_text(*_args):
		return 'keyword\n' if case == 'unexpected_newline' else 'keyword'

	try:
		run = executor.active_run
		assert run is not None
		run.params.inputs = 'keyword'
		monkeypatch.setattr(executor.helpers, 'text', provide_text)
		result = await routed(agent)
		assert not fake.errors
		if case == 'same':
			assert result.model_output is not None and len(fake.requests) == 1
			assert await browser_eval(jev_browser, 'document.querySelector("#query").value') == 'keyword'
		else:
			handback = executor.runs[-1].handback
			assert result.model_output is None and handback is not None and handback.reason == 'low_confidence'
			assert executor.task_actions == []
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('change', ['none', 'focus', 'node', 'after_prediction'])
async def test_enter_handoff_preserves_only_the_verified_typed_field(jev_browser, httpserver, tmp_path, monkeypatch, change):
	steps = [
		{'operation': 'TYPE_TEXT', 'target': 'Query'},
		{'operation': 'UNABLE'},
		{'operation': 'PRESS_ENTER' if change in ('none', 'after_prediction') else 'UNABLE'},
	]
	fake = FakeJev(httpserver, steps)
	agent, executor, params = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		body='<form onsubmit="event.preventDefault(); document.querySelector(\'output\').value=query.value">'
		'<input id="query" aria-label="Query"><input id="other" aria-label="Other">'
		'<button type="submit">Search</button><output></output></form>',
	)

	async def provide_text(*_args):
		return 'keyword'

	try:
		monkeypatch.setattr(executor.helpers, 'text', provide_text)
		assert (await routed(agent)).model_output is not None
		assert (await routed(agent)).model_output is None
		next_stage = params.model_copy(update={'objective': 'Submit the query', 'may_submit': True})
		assert not (await executor.delegate(next_stage, jev_browser)).error
		if change == 'focus':
			await browser_eval(jev_browser, 'other.focus()')
		elif change == 'node':
			await browser_eval(jev_browser, 'query.replaceWith(query.cloneNode()); query.focus()')
		result = await routed(agent, execute=change != 'after_prediction')
		if change == 'after_prediction':
			await browser_eval(jev_browser, 'other.focus()')
			agent.state.last_model_output = result.model_output
			await agent._execute_actions()
			assert agent.state.last_result
			error = agent.state.last_result[0].error
			assert error and 'focus' in error.lower()
		choice = fake.requests[-1]['questions']['operation']['criteria']
		assert ('PRESS_ENTER' in choice) == (change in ('none', 'after_prediction'))
		assert not fake.errors
		assert await browser_eval(jev_browser, 'document.querySelector("output").value') == (
			'keyword' if change == 'none' else ''
		)
	finally:
		await close_driver(agent)


async def test_uncertain_navigation_gives_planner_both_observed_links(jev_browser, httpserver, tmp_path):
	step = {'operation': 'CLICK', 'target': {'nearby_text': 'Documentation'}, 'target_p': 0.55}
	fake = FakeJev(httpserver, [step, step])
	first = '/topic/' + 'a' * 150
	second = '/topic/' + 'b' * 150
	agent, executor, _ = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		body=f'<article>Documentation<a href="{first}">Topic</a></article>'
		f'<article>Discussion<a href="{second}">Topic</a></article>',
	)
	try:
		result = await routed(agent)
		assert result.model_output is None and not fake.errors
		message = '\n'.join(str(item.content) for item in result.context_messages)
		assert httpserver.url_for(first) in message and httpserver.url_for(second) in message
		handback = executor.runs[-1].handback
		assert handback is not None and handback.reason == 'low_confidence'
		assert executor.task_actions == []
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('last_probe', ['same', 'new_evidence', 'new_action', 'requires_action'])
async def test_reworded_probes_must_add_evidence(jev_browser, httpserver, tmp_path, last_probe):
	last = {
		'new_action': {'operation': 'CLICK', 'target': 'Details'},
		'requires_action': {'operation': 'UNABLE'},
	}.get(last_probe, {'operation': 'STEP_DONE'})
	fake = FakeJev(httpserver, [{'operation': 'STEP_DONE'}] * 3 + [last])
	agent, executor, params = await start_stage(
		jev_browser,
		httpserver,
		tmp_path,
		body="<main>Stable result</main><button onclick=\"document.querySelector('main').textContent='Details'\">Details</button>",
	)
	try:
		for goal in (
			'Review the result again',
			'Check this observation from another perspective',
			'Verify the original data one last time',
		):
			assert (await routed(agent)).model_output is None
			assert not (await executor.delegate(params.model_copy(update={'objective': goal}), jev_browser)).error
		if last_probe == 'new_evidence':
			await browser_eval(jev_browser, 'document.querySelector("main").textContent = "Fresh result"')
		elif last_probe == 'same':
			await browser_eval(
				jev_browser, 'const button = document.querySelector("button"); button.replaceWith(button.cloneNode(true))'
			)
		elif last_probe == 'requires_action':
			run = executor.active_run
			assert run is not None
			run.params.requires_action = True
		response = await routed(agent)
		assert len(fake.requests) == 4
		should_finish = last_probe in {'same', 'requires_action'}
		assert executor.finish_only is should_finish
		assert (agent._planner_output_format() is agent.PlannerDoneAgentOutput) is should_finish
		assert any('Only the finish action is valid' in str(m.content) for m in response.context_messages) is should_finish
		if last_probe == 'new_action':
			assert response.model_output is not None
			assert await browser_eval(jev_browser, 'document.querySelector("main").textContent') == 'Details'
		else:
			assert response.model_output is None
			handback = executor.runs[-1].handback
			assert handback is not None and handback.reason == ('verification_stalled' if should_finish else 'step_done')
	finally:
		await close_driver(agent)


@pytest.mark.parametrize('case', ['still_uncertain', 'stale', 'unauthorized'])
async def test_confidence_recheck_preserves_execution_guards(jev_browser, httpserver, tmp_path, monkeypatch, case):
	low = {'operation': 'CLICK', 'target': 'Send', 'p': 0.55, 'target_p': 0.95}
	fake = FakeJev(httpserver, [low, low] if case == 'still_uncertain' else [low, {'operation': 'CLICK', 'target': 'Send'}])
	agent, executor, params = await start_stage(
		jev_browser, httpserver, tmp_path, body='<form><button type="submit">Send</button></form><output></output>'
	)
	try:
		if case == 'unauthorized':
			run = executor.active_run
			assert run is not None
			run.params.may_submit = False
		if case == 'stale':
			ask = executor.backend.ask

			async def change_during_recheck(system_state, questions):
				response = await ask(system_state, questions)
				if len(fake.requests) == 2:
					await browser_eval(jev_browser, 'document.querySelector("output").textContent = "Page changed"')
				return response

			monkeypatch.setattr(executor.backend, 'ask', change_during_recheck)
		out = await routed(agent)
		expected = {'still_uncertain': 'low_confidence', 'stale': 'observation_changed', 'unauthorized': 'submit_not_allowed'}
		handback = executor.runs[-1].handback
		assert out.model_output is None and handback is not None and handback.reason == expected[case]
		assert len(fake.requests) == 2
		assert executor.task_actions == []
		if case == 'still_uncertain':
			fake.steps.append(low)
			assert not (await executor.delegate(params.model_copy(update={'objective': 'Try the form again'}), jev_browser)).error
			assert (await routed(agent)).model_output is None
			assert len(fake.requests) == 3 and executor.task_actions == []
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
