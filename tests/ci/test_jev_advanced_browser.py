"""End-to-end Jev actions and independent page/server outcomes with local Chromium."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.agent.jev import JevSettings
from browser_use.agent.jev.helpers import RiskAuthorization
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.profile import ViewportSize
from browser_use.llm.views import ChatInvokeCompletion
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import (
	AUTOCOMPLETE_PAGE,
	CHAT_PAGE,
	COMPOSE_PAGE,
	SEARCH_FORM_PAGE,
	SELECT_PAGE,
	FakeJev,
	browser_eval,
	serve_newtab_site,
	serve_page,
)
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401


def action(name: str, **params: Any) -> str:
	return json.dumps({'memory': 'Review visible outcome', 'action': [{name: params}]})


def delegate(*, requires_action: bool = True, may_submit: bool = True, **details: Any) -> str:
	return action(
		'delegate_subgoal',
		objective='Complete this stage of the task',
		completion='The requested state is visible',
		reason='Use the current page',
		requires_action=requires_action,
		may_submit=may_submit,
		**details,
	)


def finish(success: bool = True) -> str:
	return action('finish', success=success, reason='Review complete')


@dataclass
class Result:
	final_page: Any
	events: list[dict[str, Any]]
	history: AgentHistoryList
	requests: list[dict[str, Any]]
	answer_payloads: list[dict[str, Any]]
	planner_calls: AsyncMock

	def by_event(self, kind: str) -> list[dict[str, Any]]:
		return [event for event in self.events if event['event'] == kind]


async def run_case(
	server: HTTPServer,
	tmp_path: Path,
	url: str,
	system1_steps: list[dict[str, Any]],
	planner_outputs: list[str],
	proof: str,
	*,
	mode: str = 'isolated',
	checks: str = 'enforce',
	backend: str = 'jev',
	write: str | None = 'Hello',
	summary: str = 'Brief summary',
	schema: type[BaseModel] | None = None,
) -> Result:
	"""Run Agent.run and read final live values before BU closes the browser."""
	fake = FakeJev(server, system1_steps)
	llm_requests: list[dict[str, Any]] = []
	chooser = None
	if backend == 'llm':
		chooser = create_mock_llm()
		script = iter(system1_steps)

		async def choose(messages, output_format, **kwargs):
			body = json.loads(messages[-1].content)
			llm_requests.append(body)
			step = next(script)
			op = step['operation']
			choices = {}
			for name, spec in body['questions'].items():
				if name == 'operation':
					choices[name] = op
				elif name == op.lower() + '_target':
					choices[name] = FakeJev._find(spec['criteria'], step['target'])
				else:
					choices[name] = next(iter(spec['criteria']))
			return ChatInvokeCompletion(completion=output_format.model_validate(choices), usage=None)

		cast(AsyncMock, chooser.ainvoke).side_effect = choose
	answer = create_mock_llm([json.dumps({'text': 'The final page was reviewed.'})])
	plan = create_mock_llm(planner_outputs)
	answer_calls = cast(AsyncMock, answer.ainvoke)
	planner_calls = cast(AsyncMock, plan.ainvoke)
	text = create_mock_llm([json.dumps({'text': write})] * 8)
	summarizer = create_mock_llm([json.dumps({'summary': summary})] * 8)
	trace = tmp_path / 'jev.jsonl'
	settings = JevSettings.model_validate(
		{
			'mode': mode,
			'system1': backend,
			'system1_llm': chooser,
			'checks': checks,
			'jev_api_key': 'local-test',
			'jev_base_url': server.url_for('/'),
			'trace_path': trace,
			'capture_dir': tmp_path / 'captures',
			'text_llm': text,
			'summary_llm': summarizer,
			'answer_llm': answer,
		}
	)
	browser = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=False,
			enable_default_extensions=False,
			viewport=ViewportSize(width=1280, height=720),
			executable_path=os.environ.get('JEV_TEST_CHROME_PATH'),
		)
	)
	agent = Agent(
		task=f'Open {url} and perform the requested stage.',
		llm=plan,
		jev=settings,
		browser_session=browser,
		use_judge=False,
		use_vision=False,
		flash_mode=True,
		enable_signal_handler=False,
		message_compaction=False,
		output_model_schema=schema,
	)
	final_proof = []

	async def capture(running: Agent):
		if running.history.is_done():
			final_proof.append(await browser_eval(running.browser_session, proof))

	history = await agent.run(max_steps=max(12, len(system1_steps) + len(planner_outputs) + 4), on_step_end=capture)
	events = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
	assert not fake.errors, (fake.errors, [r['state']['page']['url'] for r in fake.requests])
	assert len(final_proof) == 1, [(evt.get('reason'), evt.get('detail')) for evt in events if evt['event'] == 'handback']
	payloads = [json.loads(call.args[0][-1].content) for call in answer_calls.call_args_list]
	return Result(
		final_proof[0], events, history, llm_requests if chooser is not None else fake.requests, payloads, planner_calls
	)


@pytest.mark.parametrize('rollback', [False, True])
async def test_select_observes_the_exact_choice_after_dispatch(httpserver: HTTPServer, tmp_path: Path, rollback: bool):
	url = serve_page(httpserver, '/choices', SELECT_PAGE)
	target: str | dict[str, str] = 'Second' if rollback else {'element': ':2]', 'select': 'Duplicate'}
	steps = [{'operation': 'SELECT', 'target': target}]
	if not rollback:
		steps.append({'operation': 'STEP_DONE'})
	res = await run_case(
		httpserver,
		tmp_path,
		url,
		steps,
		[delegate(), finish(not rollback)],
		"Object.fromEntries(['duplicate','reverting'].map(id=>{const s=document.getElementById(id);return [id,[s.selectedIndex,s.value]];}))",
	)
	records = res.by_event('action_result')
	assert len(records) == 1 and records[0]['operation'] == 'SELECT'
	assert records[0]['verified'] is (not rollback), (res.final_page, records[0])
	if rollback:
		assert records[0]['status'] == 'executed_failed' and not records[0]['applied']
		assert res.final_page['reverting'] == [0, 'first'] and res.by_event('handback')[-1]['reason'] == 'action_error'
	else:
		assert records[0]['status'] == 'executed_ok' and records[0]['applied']
		assert res.final_page['duplicate'] == [1, '2']


@pytest.mark.parametrize(
	('page', 'allowed', 'checks'),
	[
		('chat', True, 'enforce'),
		('chat', False, 'enforce'),
		('chat', True, 'record'),
		('compose', True, 'enforce'),
		('search', True, 'enforce'),
	],
)
async def test_enter_outcome_on_real_form(httpserver: HTTPServer, tmp_path: Path, page: str, allowed: bool, checks: str):
	pages = {
		'chat': (CHAT_PAGE, 'Message', '/api/messages'),
		'compose': (COMPOSE_PAGE, 'Body', '/api/send'),
		'search': (SEARCH_FORM_PAGE, 'Search query', ''),
	}
	body, field, endpoint = pages[page]
	url = serve_page(httpserver, '/form', body)
	deliveries = []

	def delivered(request: Request) -> Response:
		deliveries.append(request.form.to_dict() if request.form else request.get_data(as_text=True))
		return Response(status=204)

	if endpoint:
		httpserver.expect_request(endpoint, method='POST').respond_with_handler(delivered)
	confirm = page != 'search' and checks == 'enforce'
	outputs = [delegate(inputs='Hello')]
	if confirm:
		outputs.append(json.dumps({'authorized': allowed, 'reason': 'Checked recipients and content'}))
	outputs.append(finish(allowed))
	steps = [{'operation': 'TYPE_TEXT', 'target': field}, {'operation': 'PRESS_ENTER'}]
	if allowed:
		steps.append({'operation': 'STEP_DONE'})
	res = await run_case(httpserver, tmp_path, url, steps, outputs, 'document.body.innerText', checks=checks)
	decisions = res.by_event('risk_authorization')
	if confirm:
		assert len(decisions) == 1 and decisions[0]['trigger'] == 'enter' and decisions[0]['authorized'] == allowed
		assert decisions[0]['effect'] == 'send'
		risk_call = next(
			c
			for c in res.planner_calls.call_args_list
			if RiskAuthorization in c.args[1:] or c.kwargs.get('output_format') is RiskAuthorization
		)
		risk_input = json.loads(risk_call.args[0][-1].content)
		assert any(f['value'] == 'Hello' for f in risk_input['visible_fields'])
	else:
		assert not decisions
	assert len(deliveries) == int(bool(endpoint) and allowed)
	if page == 'chat' and allowed:
		assert deliveries[0] == 'Hello' and 'Hello' in res.final_page
	if page == 'compose':
		assert isinstance(deliveries[0], dict)
		assert deliveries[0]['body'] == 'Hello' and 'Sent' in res.final_page
	if page == 'search':
		assert 'Results for Hello' in res.final_page
	if not allowed:
		assert res.by_event('handback')[-1]['reason'] == 'risk_denied'
	if checks == 'record':
		assert any(e['name'] == 'risky_action' and e['outcome'] == 'would_block' for e in res.by_event('check'))
	summary_roles = res.by_event('usage_summary')[-1]['roles']
	assert summary_roles['risk']['calls'] == int(confirm) and summary_roles['text']['calls'] == 1
	assert summary_roles['planner']['calls'] == 2


@pytest.mark.parametrize('cross_site', [False, True])
async def test_switch_tab_and_cross_site_handback(httpserver: HTTPServer, tmp_path: Path, cross_site: bool):
	url = serve_newtab_site(httpserver)
	steps = [{'operation': 'CLICK', 'target': 'Other site' if cross_site else 'Open result'}]
	if not cross_site:
		steps += [{'operation': 'CLICK', 'target': {'role': 'tab', 'element': '/tabs'}}, {'operation': 'STEP_DONE'}]
	res = await run_case(
		httpserver, tmp_path, url, steps, [delegate(), finish(not cross_site)], '[location.hostname,location.pathname]'
	)
	if cross_site:
		assert res.final_page == ['127.0.0.1', '/tab-result']
		assert res.by_event('handback')[-1]['reason'] == 'out_of_scope'
	else:
		assert res.final_page == ['localhost', '/tabs']
		assert res.requests[1]['state']['page']['url'].endswith('/tab-result')
		assert res.by_event('handback')[-1]['reason'] == 'step_done'


class Report(BaseModel):
	code: str


@pytest.mark.parametrize('structured', [False, True])
async def test_collect_on_a_collaborative_finish_on_b(httpserver: HTTPServer, tmp_path: Path, structured: bool):
	a = serve_page(httpserver, '/evidence', '<article>Archive record ALPHA-83</article>', 'Archive')
	b = serve_page(httpserver, '/receipt', '<p>Receipt BETA-42</p>', 'Receipt')
	done = (
		action('done', data={'code': 'BETA-42'}, success=True)
		if structured
		else action('done', text='Planner review', success=True)
	)
	res = await run_case(
		httpserver,
		tmp_path,
		a,
		[{'operation': 'COLLECT'}, {'operation': 'STEP_DONE'}],
		[delegate(requires_action=False), action('navigate', url=b, new_tab=False), done],
		'[location.pathname,document.body.innerText]',
		mode='collaborative',
		schema=Report if structured else None,
	)
	assert res.final_page == ['/receipt', 'Receipt BETA-42']
	captures = [json.loads(path.read_text(encoding='utf-8')) for path in (tmp_path / 'captures').glob('*.json')]
	assert len(captures) == 1 and captures[0]['url'] == a and 'ALPHA-83' in captures[0]['text']
	if structured:
		assert not res.answer_payloads and not res.by_event('final_answer')
		assert res.history.structured_output == Report(code='BETA-42')
	else:
		assert len(res.answer_payloads) == 1
		payload = res.answer_payloads[0]
		assert payload['last_page']['url'] == b and 'BETA-42' in payload['last_page']['text']
		browser_observation = payload['final_browser_observation']
		assert browser_observation['url'] == b and 'BETA-42' in browser_observation['dom_text']
		assert 'ALPHA-83' not in browser_observation['dom_text']
		assert len(payload['collected_pages']) == 1 and 'ALPHA-83' in payload['collected_pages'][0]['text']
		assert res.by_event('final_answer')[0]['trigger'] == 'done_rewrite'
		assert res.history.final_result() == 'The final page was reviewed.'


async def test_complete_autocomplete_by_typing_then_selecting_suggestion(httpserver: HTTPServer, tmp_path: Path):
	url = serve_page(httpserver, '/autocomplete', AUTOCOMPLETE_PAGE)
	steps = [
		{'operation': 'TYPE_TEXT', 'target': 'Where from?'},
		# The fixture reveals suggestions asynchronously after the input event.
		{'operation': 'WAIT'},
		{'operation': 'CLICK', 'target': 'Zurich (ZRH)'},
		{'operation': 'STEP_DONE'},
	]
	res = await run_case(
		httpserver,
		tmp_path,
		url,
		steps,
		[delegate(inputs='Zurich'), finish()],
		'[document.querySelector("#origin").value,document.querySelector("#choices").hidden]',
		write='Zurich',
	)
	assert res.final_page == ['Zurich (ZRH)', True]
	assert [e['operation'] for e in res.by_event('action_result')] == ['TYPE_TEXT', 'WAIT', 'CLICK']


def workflow_site(server: HTTPServer) -> tuple[str, list[dict]]:
	"""One three-page task with a form, two source documents, and an observable final submission."""
	serve_page(server, '/source-a', '<article>Reference ALPHA-83</article><a href="/source-b">Next source</a>', 'First source')
	serve_page(server, '/source-b', '<article>Reference BETA-42</article>', 'Second source')
	url = serve_page(
		server,
		'/workflow',
		"""<label>Airport <input id="airport" oninput="suggestion.hidden=false"></label>
<button hidden id="suggestion" onclick="airport.value='ZRH';this.hidden=true">Zurich airport</button>
<button onclick="calendar.hidden=false">Open calendar</button><div id="calendar" hidden>
<button onclick="selection.textContent='June 18';calendar.hidden=true">June 18</button></div><output id="selection"></output>
<label>Type <select id="kind" onchange="extra.hidden=this.value!=='work'"><option>Choose</option>
<option value="work">Work</option></select></label><label id="extra" hidden>Team details <input id="details"></label>
<a href="/source-a" target="_blank">Open source</a><button id="send">Send</button><output id="result"></output>
<script>send.onclick=async()=>{const data={airport:airport.value,date:selection.textContent,type:kind.value,details:details.value};
const r=await fetch('/api/workflow',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
if(r.ok)result.textContent='Submitted';};</script>""",
		'Workflow',
	)
	submissions = []

	def save(request: Request) -> Response:
		submissions.append(request.get_json())
		return Response(status=204)

	server.expect_request('/api/workflow', method='POST').respond_with_handler(save)
	return url, submissions


def workflow_steps() -> list[dict[str, Any]]:
	return [
		{'operation': 'TYPE_TEXT', 'target': 'Airport'},
		{'operation': 'CLICK', 'target': 'Zurich airport'},
		{'operation': 'CLICK', 'target': 'Open calendar'},
		{'operation': 'CLICK', 'target': 'June 18'},
		{'operation': 'SELECT', 'target': 'Work'},
		{'operation': 'TYPE_TEXT', 'target': 'Team details'},
		{'operation': 'CLICK', 'target': 'Open source'},
		{'operation': 'COLLECT'},
		{'operation': 'CLICK', 'target': 'Next source'},
		{'operation': 'COLLECT'},
		{'operation': 'CLICK', 'target': {'role': 'tab', 'element': '/workflow'}},
		{'operation': 'CLICK', 'target': 'Send'},
		{'operation': 'STEP_DONE'},
	]


async def run_native_workflow(url: str) -> tuple[str, list[str]]:
	"""Drive the native action schema using indices from the current BU snapshot."""
	plan = create_mock_llm()
	browser = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=False,
			enable_default_extensions=False,
			viewport=ViewportSize(width=1280, height=720),
			executable_path=os.environ.get('JEV_TEST_CHROME_PATH'),
		)
	)
	steps = iter(workflow_steps())
	source_text = []

	async def native(messages, output_format, **kwargs):
		step = next(steps)
		op, target = step['operation'], step.get('target')
		if op == 'STEP_DONE':
			raw = action('done', text='Submitted', success=True)
		elif op == 'COLLECT':
			source_text.append(await browser_eval(browser, 'document.querySelector("article").textContent'))
			raw = action('extract', query='Read the reference in this article', extract_links=False)
		elif isinstance(target, dict):
			tab = next(t for t in await browser.get_tabs() if t.url == url)
			raw = action('switch', tab_id=tab.target_id[-4:])
		else:
			snapshot = await browser.get_browser_state_summary(include_screenshot=False)
			if op == 'SELECT':
				idx = next(i for i, n in snapshot.dom_state.selector_map.items() if n.tag_name == 'select')
				raw = action('select_dropdown', index=idx, text=target)
			else:
				tag_types = {'input'} if op == 'TYPE_TEXT' else {'a', 'button'}
				idx = next(
					i
					for i, n in snapshot.dom_state.selector_map.items()
					if n.tag_name in tag_types and n.ax_node and (n.ax_node.name or '').strip() == target
				)
				raw = action('input', index=idx, text='Zurich', clear=True) if op == 'TYPE_TEXT' else action('click', index=idx)
		return ChatInvokeCompletion(completion=output_format.model_validate_json(raw), usage=None)

	cast(AsyncMock, plan.ainvoke).side_effect = native
	extraction = create_mock_llm(['Reference ALPHA-83', 'Reference BETA-42'])
	agent = Agent(
		task=f'Open {url} and submit the reviewed form',
		llm=plan,
		browser_session=browser,
		page_extraction_llm=extraction,
		use_judge=False,
		use_vision=False,
		flash_mode=True,
		message_compaction=False,
		enable_signal_handler=False,
		jev=JevSettings(mode='off'),
	)
	final = []

	async def check(running: Agent):
		if running.history.is_done():
			final.append(await browser_eval(browser, 'document.querySelector("#result").textContent'))

	history = await agent.run(max_steps=19, on_step_end=check)
	assert history.final_result() == 'Submitted' and len(final) == 1
	return final[0], source_text


@pytest.mark.parametrize(
	('mode', 'backend'),
	[
		('isolated', 'jev'),
		('isolated', 'llm'),
		('collaborative', 'jev'),
		('collaborative', 'llm'),
		('off', 'native'),
	],
)
async def test_multistep_workflow_all_modes(httpserver: HTTPServer, tmp_path: Path, mode: str, backend: str):
	url, submissions = workflow_site(httpserver)
	if mode == 'off':
		proof, originals = await run_native_workflow(url)
		assert originals == ['Reference ALPHA-83', 'Reference BETA-42']
	else:
		final = finish() if mode == 'isolated' else action('done', text='Checked the submitted receipt', success=True)
		outputs = [
			delegate(inputs='Airport Zurich, team details Zurich'),
			json.dumps({'authorized': True, 'reason': 'The user requested this submission'}),
			final,
		]
		steps = workflow_steps()
		res = await run_case(
			httpserver,
			tmp_path,
			url,
			steps,
			outputs,
			'document.querySelector("#result").textContent',
			mode=mode,
			backend=backend,
			write='Zurich',
		)
		proof = res.final_page
		assert res.by_event('handback')[-1]['reason'] == 'step_done'
		assert len(res.answer_payloads) == 1
		originals = res.answer_payloads[0]['collected_pages']
		assert len(originals) == 2 and 'ALPHA-83' in originals[0]['text'] and 'BETA-42' in originals[1]['text']
		assert len(res.by_event('risk_authorization')) == 1 and res.by_event('risk_authorization')[0]['authorized']
		assert len(res.requests) == len(steps)
		roles = res.by_event('usage_summary')[-1]['roles']
		assert {k: v['calls'] for k, v in roles.items() if k != 'other'} == {
			'planner': 2,
			'system1': len(steps),
			'text': 2,
			'summary': 2,
			'answer': 1,
			'risk': 1,
		}
	assert proof == 'Submitted'
	assert submissions == [{'airport': 'ZRH', 'date': 'June 18', 'type': 'work', 'details': 'Zurich'}]


async def test_helper_failure_preserves_original_and_leaves_input_untouched(httpserver: HTTPServer, tmp_path: Path):
	url = serve_page(httpserver, '/notes', '<p>Source record ALPHA-83</p><label>Note <input id="note"></label>')
	res = await run_case(
		httpserver,
		tmp_path,
		url,
		[{'operation': 'COLLECT'}, {'operation': 'TYPE_TEXT', 'target': 'Note'}],
		[delegate(requires_action=False), finish(False)],
		'document.querySelector("#note").value',
		write=None,
		summary='',
	)
	assert res.final_page == ''
	assert res.by_event('handback')[-1]['reason'] == 'text_unavailable'
	assert [e['operation'] for e in res.by_event('action_result')] == ['COLLECT']
	capture_files = list((tmp_path / 'captures').glob('*.json'))
	assert len(capture_files) == 1
	capture = json.loads(capture_files[0].read_text(encoding='utf-8'))
	assert capture['summary_status'] == 'failed' and 'ALPHA-83' in capture['text']
	assert 'ALPHA-83' in res.answer_payloads[0]['collected_pages'][0]['text']


@pytest.mark.parametrize(
	('kind', 'may_submit', 'applies'),
	[
		('recipient', False, True),
		('recipient', True, True),
		('form', False, False),
		('form', True, True),
		('search', False, False),
		('search', True, True),
	],
)
async def test_enter_distinguishes_in_place_confirmation_from_submission(httpserver, tmp_path, kind, may_submit, applies):
	field = {'recipient': 'Recipients', 'form': 'Reference', 'search': 'Search records'}[kind]
	input_type = 'search' if kind == 'search' else 'text'
	control = f'<label>{field}<input id="field" type="{input_type}"></label>'
	js = (
		"const field = document.querySelector('#field');"
		"const show = () => { document.querySelector('#result').textContent = field.value; };"
	)
	if kind == 'form':
		control = f'<form id="entry">{control}<button type="submit">Apply</button></form>'
		js += "document.querySelector('#entry').addEventListener('submit', e => { e.preventDefault(); show(); });"
	else:
		js += "field.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); show(); } });"
	url = serve_page(httpserver, '/confirm-enter', f'{control}<output id="result"></output><script>{js}</script>')
	steps = [{'operation': 'TYPE_TEXT', 'target': field}, {'operation': 'PRESS_ENTER'}]
	if applies:
		steps.append({'operation': 'STEP_DONE'})
	res = await run_case(
		httpserver,
		tmp_path,
		url,
		steps,
		[delegate(may_submit=may_submit, inputs='Alice'), finish(applies)],
		'document.querySelector("#result").textContent',
		write='Alice',
	)
	assert res.final_page == ('Alice' if applies else '')
	assert res.by_event('handback')[-1]['reason'] == ('step_done' if applies else 'submit_not_allowed')


@pytest.mark.parametrize('checks', ['enforce', 'record'])
async def test_submit_rule_measures_the_actual_search_effect(httpserver: HTTPServer, tmp_path: Path, checks: str):
	url = serve_page(httpserver, '/search', SEARCH_FORM_PAGE)
	steps = [{'operation': 'TYPE_TEXT', 'target': 'Search query'}, {'operation': 'PRESS_ENTER'}]
	if checks == 'record':
		steps.append({'operation': 'STEP_DONE'})
	res = await run_case(
		httpserver,
		tmp_path,
		url,
		steps,
		[delegate(may_submit=False), finish(checks == 'record')],
		'document.querySelector("#result").textContent',
		checks=checks,
	)
	check = next(c for c in res.by_event('check') if c['name'] == 'submit')
	assert check['outcome'] == ('blocked' if checks == 'enforce' else 'would_block')
	if checks == 'enforce':
		assert res.final_page == '' and res.by_event('handback')[-1]['reason'] == 'submit_not_allowed'
	else:
		assert res.final_page == 'Results for Hello'
