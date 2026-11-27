"""Exercise iframe editors against an isolated local Chromium."""

import html
import json
from collections.abc import Iterator

import pytest
from pytest_httpserver import HTTPServer

from browser_use import Agent
from browser_use.agent.jev.state import evidence_key, observe
from browser_use.agent.jev.views import JevSettings
from browser_use.browser.events import TypeTextEvent
from browser_use.browser.session import BrowserSession
from browser_use.dom.serializer.clickable_elements import ClickableElementDetector
from browser_use.dom.views import EnhancedDOMTreeNode
from browser_use.tools.service import Tools
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import FakeJev, browser_eval, serve_page
from tests.ci.jev_support import jev_browser as _jev_browser_fixture  # noqa: F401
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401

TOKEN = 'WP7 typed marker'
EDITOR_TEXT = 'Initial editor text'
FRAME_STYLE = 'width: 540px; height: 260px; border: 7px solid black'
FRAME_BODY = f'<p>{EDITOR_TEXT}</p>'
READ_FROM_CHILD = """
<script>
addEventListener('message', event => {
	if (event.data !== 'wp7-read') return;
	event.source.postMessage({
		text: document.body.innerText,
		designMode: document.designMode,
		bodyEditable: document.body.isContentEditable,
	}, '*');
});
</script>
"""
READ_FRAME = """
new Promise(resolve => {
	const frame = document.querySelector('iframe');
	const receive = event => {
		if (event.source !== frame.contentWindow || !event.data || !('bodyEditable' in event.data)) return;
		removeEventListener('message', receive);
		resolve(event.data);
	};
	addEventListener('message', receive);
	frame.contentWindow.postMessage('wp7-read', '*');
	setTimeout(() => {
		removeEventListener('message', receive);
		resolve(null);
	}, 2500);
})
"""


def walk_dom(node: EnhancedDOMTreeNode) -> Iterator[EnhancedDOMTreeNode]:
	yield node
	for child in node.children_and_shadow_roots:
		yield from walk_dom(child)
	if node.content_document is not None:
		yield from walk_dom(node.content_document)


def editor_page(server: HTTPServer, cross_origin: HTTPServer, case: str) -> str:
	if case == 'P3':
		editor = f'<div id="editor" aria-label="Top editor" contenteditable="true">{EDITOR_TEXT}</div>'
	elif case == 'P4':
		src = serve_page(cross_origin, '/editor', FRAME_BODY + "<script>document.designMode='on'</script>" + READ_FROM_CHILD)
		viewer = serve_page(cross_origin, '/viewer', '<p>Unrelated cross-origin viewer</p>')
		editor = (
			f'<iframe id="editor" title="Message editor" style="{FRAME_STYLE}" src="{src}"></iframe>'
			f'<iframe title="Read-only viewer" style="width: 240px; height: 150px" src="{viewer}"></iframe>'
		)
	else:
		inner = (
			f"<body>{FRAME_BODY}<script>document.designMode='on'</script>"
			if case == 'P1'
			else f'<style>html,body{{height:100%;margin:0}}</style><body contenteditable="true">{FRAME_BODY}'
		)
		editor = f'<iframe id="editor" title="Message editor" style="{FRAME_STYLE}" srcdoc="{html.escape(inner)}"></iframe>'
	return serve_page(server, f'/{case}', editor + '<button type="button">Ordinary button</button>')


async def read_editor(browser: BrowserSession, case: str):
	if case == 'P4':
		return await browser_eval(browser, READ_FRAME)
	return await browser_eval(
		browser,
		"""(() => {
			const editor = document.querySelector('#editor');
			const doc = editor.contentDocument;
			const body = doc ? doc.body : editor;
			return {text: body.innerText, designMode: doc?.designMode, bodyEditable: body.isContentEditable};
		})()""",
	)


def describe_node(node: EnhancedDOMTreeNode) -> dict:
	props = node.ax_node.properties if node.ax_node else []
	return {
		'tag': node.tag_name,
		'id': node.attributes.get('id'),
		'ax_role': node.ax_node.role if node.ax_node else None,
		'ax': {p.name: p.value for p in props or [] if p.name in {'editable', 'focusable'}},
		'contenteditable': node.attributes.get('contenteditable'),
		'interactive': ClickableElementDetector.is_interactive(node),
		'frame': node.frame_id,
		'target': node.target_id,
	}


@pytest.mark.parametrize('case', ['P1', 'P2', 'P3', 'P4'])
async def test_probe_native_iframe_editor(jev_browser: BrowserSession, httpserver: HTTPServer, case: str):
	# A distinct loopback host and port also exercise Chromium's out-of-process frame path.
	with HTTPServer(host='127.0.0.1', port=0, threaded=True) as cross_origin:
		await jev_browser.navigate_to(editor_page(httpserver, cross_origin, case))
		before = await read_editor(jev_browser, case)
		assert before and before['text'] == EDITOR_TEXT and before['bodyEditable']
		summary = await jev_browser.get_browser_state_summary(include_screenshot=False)
		observation = await observe(jev_browser, summary, JevSettings(screenshot_policy='off'), 1)
		watchdog = jev_browser._dom_watchdog
		assert watchdog is not None and watchdog.enhanced_dom_tree is not None
		tree = list(walk_dom(watchdog.enhanced_dom_tree))
		editor = next(node for node in tree if node.attributes.get('id') == 'editor')
		frame_roots = (
			[node for node in walk_dom(editor.content_document) if node.tag_name in {'body', 'html'}]
			if editor.content_document is not None
			else []
		)
		target = editor if case == 'P3' else next((node for node in frame_roots if node.tag_name == 'body'), None)
		target_index = next((index for index, node in summary.dom_state.selector_map.items() if node is target), None)
		top_roots = [node for node in tree if node.tag_name in {'body', 'html'} and node not in frame_roots]
		selector_map = [{'index': index, **describe_node(node)} for index, node in summary.dom_state.selector_map.items()]
		if target_index is not None:
			result = await Tools().registry.execute_action(
				'input', {'index': target_index, 'text': TOKEN, 'clear': False}, browser_session=jev_browser
			)
			typed = {'mode': 'indexed BU input', 'error': result.error}
		elif target is not None:
			# The index route is unavailable. Probe the same BU input event using the exact observed DOM node.
			event = jev_browser.event_bus.dispatch(TypeTextEvent(node=target, text=TOKEN, clear=False))
			try:
				await event.event_result(raise_if_any=True, raise_if_none=False)
				typed = {'mode': 'BU input event', 'error': None}
			except Exception as error:
				typed = {'mode': 'BU input event', 'error': str(error)}
		else:
			typed = {'mode': 'unavailable', 'error': None}
		after = await read_editor(jev_browser, case)
		assert after is not None
		data = {
			'case': case,
			'frame_roots': [describe_node(node) for node in frame_roots],
			'top_roots': [describe_node(node) for node in top_roots],
			'selector_map': selector_map,
			'type_text': [candidate.element() for candidate in observation.candidates['TYPE_TEXT']],
			'target_index': target_index,
			'input': typed,
			'before': before,
			'after': after,
			'marker_found': TOKEN in after['text'],
		}
		print('WP7_PROBE ' + json.dumps(data, ensure_ascii=False))
		assert all(not node['interactive'] for node in data['top_roots'])
		assert all(node['tag'] != 'html' for node in selector_map)
		assert any(node['tag'] == 'button' for node in selector_map)
		assert data['marker_found']
		assert 'audit' not in observation.model_dump()
		assert 'Unrelated cross-origin viewer' not in observation.visible_text
		assert target_index is not None and typed['mode'] == 'indexed BU input'
		expected_name = 'Top editor' if case == 'P3' else 'Message editor (iframe body)'
		assert any(
			candidate.index == target_index and candidate.role == 'textbox' and candidate.name == expected_name
			for candidate in observation.candidates['TYPE_TEXT']
		)
		if case != 'P3':
			assert '[iframe] ' + EDITOR_TEXT in observation.visible_text
			updated = await jev_browser.get_browser_state_summary(include_screenshot=False)
			observed = await observe(jev_browser, updated, JevSettings(screenshot_policy='off'), 2)
			assert EDITOR_TEXT in iframe_text(observed.visible_text)
			assert TOKEN in iframe_text(observed.visible_text)
			assert evidence_key(observed) != evidence_key(observation)


def iframe_text(text: str) -> str:
	return '\n'.join(line.removeprefix('[iframe] ') for line in text.splitlines() if line.startswith('[iframe] '))


def planner_action(name: str, **params: object) -> str:
	return json.dumps({'action': [{name: params}]})


async def test_jev_rejects_text_redirected_into_another_field(jev_browser, httpserver, tmp_path):
	url = serve_page(
		httpserver,
		'/redirect-text',
		"""
		<div id="editor" role="textbox" aria-label="Intended editor" contenteditable="true"></div>
		<input id="other" aria-label="Other field">
		<script>
		editor.addEventListener('beforeinput', event => {
			if (!event.data) return;
			other.value += event.data;
			event.preventDefault();
		});
		</script>
		""",
	)
	fake = FakeJev(httpserver, [{'operation': 'TYPE_TEXT', 'target': 'Intended editor'}, {'operation': 'STEP_DONE'}])
	plan = planner_action(
		'delegate_subgoal',
		objective='Type into the intended editor',
		completion='The intended editor contains the marker',
		reason='The user specified the intended editor.',
		inputs=TOKEN,
		text_mode='append',
		requires_action=True,
	)
	trace = tmp_path / 'trace.jsonl'
	agent = Agent(
		task=f'Open {url} and put the marker in the intended editor.',
		llm=create_mock_llm([plan, planner_action('finish', success=False, reason='Review the typed result')]),
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(
			jev_api_key='local-test',
			jev_base_url=httpserver.url_for('/'),
			trace_path=trace,
			min_operation_probability=0,
			min_target_probability=0,
			text_llm=create_mock_llm([json.dumps({'text': TOKEN})]),
			answer_llm=create_mock_llm(['{"text":"Reviewed the typed result."}']),
		),
	)
	await agent.run(max_steps=5)
	assert not fake.errors
	target_text, other_text = await browser_eval(jev_browser, '[editor.innerText, other.value]')
	assert TOKEN not in target_text and TOKEN in other_text
	records = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
	result = next(r for r in records if r['event'] == 'action_result' and r['operation'] == 'TYPE_TEXT')
	assert result['status'] == 'executed_failed'
	assert result['error'] == 'typed text not found in the target'
	assert not result['applied'] and result['verified'] is False
	assert any(r['event'] == 'handback' and r['reason'] == 'action_error' for r in records)


@pytest.mark.parametrize('scenario', ['visibility', 'none', 'limit'])
async def test_candidate_audit_reports_visible_labels_once(jev_browser, httpserver, tmp_path, scenario):
	body = '<button type="button">Ordinary button</button>'
	if scenario == 'visibility':
		body += """
			<button disabled>Disabled only</button>
			<p>The button named Outside only is further down.</p>
			<button style="position:absolute;top:1800px">Outside only</button>
		"""
	if scenario == 'limit':
		body += '<button type="button">Another button</button>'
	url = serve_page(httpserver, '/candidate-audit', body)
	fake = FakeJev(httpserver, [{'operation': 'WAIT'}, {'operation': 'STEP_DONE'}])
	plan = planner_action(
		'delegate_subgoal',
		objective='Inspect the fixture',
		completion='Identify the action choices',
		reason='Inspect the available controls.',
	)
	trace = tmp_path / 'audit.jsonl'
	agent = Agent(
		task=f'Open {url} and inspect the page controls.',
		llm=create_mock_llm([plan, planner_action('finish', success=True, reason='Inspected the page')]),
		browser_session=jev_browser,
		use_judge=False,
		jev=JevSettings(
			jev_api_key='local-test',
			jev_base_url=httpserver.url_for('/'),
			trace_path=trace,
			min_operation_probability=0,
			min_target_probability=0,
			max_candidates=1 if scenario == 'limit' else 200,
			answer_llm=create_mock_llm(['{"text":"Inspected the page."}']),
		),
	)
	await agent.run(max_steps=5)
	assert len(fake.requests) == 2 and not fake.errors
	records = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
	events = [record for record in records if record['event'] == 'candidate_audit']
	if scenario == 'visibility':
		assert len(events) == 1
		assert {(row['operation'], row['label'], row['reason']) for row in events[0]['omitted']} >= {
			('CLICK', 'Disabled only', 'disabled'),
			('CLICK', 'Outside only', 'outside_viewport'),
		}
	elif scenario == 'limit':
		assert len(events) == 1 and len(events[0]['omitted']) == 1
		[omitted] = events[0]['omitted']
		assert omitted['operation'] == 'CLICK' and omitted['reason'] == 'candidate_limit'
		assert omitted['label'] in {'Ordinary button', 'Another button'}
		assert all(
			omitted['label'] not in choice['element']
			for choice in fake.requests[0]['questions']['click_target']['criteria'].values()
		)
	else:
		assert events == []


@pytest.mark.parametrize(('case', 'text_mode'), [('P1', 'append'), ('P2', 'append'), ('P4', 'append'), ('P4', None)])
async def test_collect_saves_the_typed_iframe_body(jev_browser, httpserver, tmp_path, case, text_mode):
	with HTTPServer(host='127.0.0.1', port=0, threaded=True) as cross_origin:
		url = editor_page(httpserver, cross_origin, case)
		fake = FakeJev(
			httpserver,
			[
				{'operation': 'TYPE_TEXT', 'target': 'Message editor (iframe body)'},
				{'operation': 'COLLECT'},
				{'operation': 'STEP_DONE'},
			],
		)
		plan = planner_action(
			'delegate_subgoal',
			objective='Type and collect the editor text',
			completion='Collect the visible written text',
			reason='Record the contents of the editor.',
			inputs=TOKEN,
			requires_action=True,
			**({'text_mode': text_mode} if text_mode is not None else {}),
		)
		capture_dir = tmp_path / 'captures'
		trace = tmp_path / 'trace.jsonl'
		agent = Agent(
			task=f'Open {url}, type the marker in the editor body, then collect the editor text.',
			llm=create_mock_llm(
				[plan, planner_action('finish', success=True, reason='Captured the editor', unverified_requirements=[])]
			),
			browser_session=jev_browser,
			use_judge=False,
			jev=JevSettings(
				jev_api_key='local-test',
				jev_base_url=httpserver.url_for('/'),
				capture_dir=capture_dir,
				trace_path=trace,
				min_operation_probability=0,
				min_target_probability=0,
				text_llm=create_mock_llm([json.dumps({'text': TOKEN})]),
				summary_llm=create_mock_llm(['{"summary":"The editor contains the marker."}']),
				answer_llm=create_mock_llm(['{"text":"Captured the editor."}']),
			),
		)
		editor_readbacks = []

		async def inspect(running):
			if running.history.action_names()[-1:] == ['input']:
				editor_readbacks.append(await read_editor(jev_browser, case))

		history = await agent.run(max_steps=6, on_step_end=inspect)
		assert not fake.errors and history.is_successful()
		assert any(reading and TOKEN in reading['text'] for reading in editor_readbacks)
		(capture,) = capture_dir.glob('*.json')
		saved = json.loads(capture.read_text(encoding='utf-8'))
		assert EDITOR_TEXT in iframe_text(saved['text'])
		assert TOKEN in iframe_text(saved['text'])
		records = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
		result = next(r for r in records if r['event'] == 'action_result' and r['operation'] == 'TYPE_TEXT')
		assert result['status'] == 'executed_ok' and result['verified'] and result['applied']
