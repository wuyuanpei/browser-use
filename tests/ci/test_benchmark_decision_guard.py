import os
from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer

from browser_use import BrowserSession
from browser_use.benchmark.decision_guard import decision_validity, read_document, stale_disposition, target_guard
from browser_use.benchmark.engine import BrowserUseRun, RunEvent
from browser_use.benchmark.models import EvaluationModel, ExecutionModel, RunSettings
from browser_use.benchmark.token_usage import aggregate_token_usage, token_usage


def test_stale_limit():
	assert stale_disposition(0, 'target_detached') == 'retry'
	assert stale_disposition(1, 'document_changed') == 'retry'
	assert stale_disposition(2, 'target_state_changed') == 'force'
	assert stale_disposition(2, 'ui_out_of_sync') == 'ui_sync'
	assert stale_disposition(2, 'target_unverifiable') == 'execute'


def test_cached_tokens_are_optional_in_archives():
	calls = [
		{'kind': 'browser_use', 'response': {'usage': {'prompt_tokens': 15, 'completion_tokens': 2, 'prompt_cached_tokens': 10}}},
		{'kind': 'text', 'response': {'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'prompt_cache_hit_tokens': 5}}},
	]
	usage = token_usage(calls)
	assert usage['other_cached_input_tokens'] == 15
	archived = {k: v for k, v in usage.items() if k != 'other_cached_input_tokens'}
	totals = aggregate_token_usage([{'token_usage': archived}, {'token_usage': usage}])
	assert totals['other_cached_input_tokens'] == 15
	assert totals['other_input_tokens'] == 50
	assert totals['complete']


async def test_decision_guard_on_live_document(httpserver: HTTPServer, tmp_path: Path):
	html = """<link rel="icon" href="data:,"><body><span id="news"></span><button id="target" aria-label="choose">Choose <b>one</b></button>
	<input type="checkbox" id="toggle"><div id="host"></div><script>
	host.attachShadow({mode:'open'}).innerHTML='<button id="shadowbtn">Shadow button</button>';
	setInterval(()=>news.textContent=String(performance.now()),100);
	</script></body>"""
	httpserver.expect_request('/guard').respond_with_data(html, content_type='text/html')
	browser = BrowserSession(
		headless=True,
		user_data_dir=str(tmp_path / 'chrome'),
		enable_default_extensions=False,
		executable_path=os.environ.get('BROWSER_USE_EXECUTABLE_PATH'),
	)
	await browser.start()
	try:
		await browser.navigate_to(httpserver.url_for('/guard'))
		summary = await browser.get_browser_state_summary(include_screenshot=False)
		guards = {node.attributes.get('id'): target_guard(node) for node in summary.dom_state.selector_map.values()}
		doc = await read_document(browser)
		session = await browser.get_or_create_cdp_session(focus=False)

		async def change(expression):
			reply = await session.cdp_client.send.Runtime.evaluate(
				params={'expression': expression, 'awaitPromise': True}, session_id=session.session_id
			)
			assert 'exceptionDetails' not in reply

		async def check(element):
			return await decision_validity(browser, guards[element], doc, await read_document(browser))

		await change('new Promise(resolve => setTimeout(resolve, 160))')
		assert await check('target') is None
		await change("target.style.marginLeft='50px'; location.hash='moved'")
		assert await check('target') is None
		assert await check('shadowbtn') is None
		await change("target.setAttribute('aria-label', 'cancel')")
		assert await check('target') == 'target_changed'
		await change("target.setAttribute('aria-label','choose'); target.textContent='Cancel'")
		assert await check('target') == 'target_text_changed'
		await change('toggle.checked=true')
		assert await check('toggle') == 'target_state_changed'
		await change('target.remove()')
		assert await check('target') == 'target_detached'
		# A route change after the first document sample overrides a stale target-state result.
		start_of_check = await read_document(browser)
		await change("history.replaceState(null, '', '?during-target-check')")
		assert await decision_validity(browser, guards['toggle'], doc, start_of_check) == 'document_changed'
		await browser.navigate_to(httpserver.url_for('/guard?second'))
		assert await check('shadowbtn') == 'document_changed'
	finally:
		await browser.kill()


@pytest.mark.parametrize('requires_fresh_target', [False, True])
def test_third_stale_executes_and_records_native_result(
	httpserver: HTTPServer, httpserver_ipv4: HTTPServer, tmp_path: Path, requires_fresh_target: bool
):
	httpserver.expect_request('/task').respond_with_data(
		'<link rel="icon" href="data:,"><p>Finished</p>', content_type='text/html'
	)
	# Keep the model service available while Chrome retains a connection to the test page.
	model_server = httpserver_ipv4
	model_server.expect_request('/chat/completions', method='POST').respond_with_json(
		{
			'id': 'local',
			'object': 'chat.completion',
			'created': 1,
			'model': 'deepseek-flash',
			'choices': [
				{
					'index': 0,
					'finish_reason': 'tool_calls',
					'message': {
						'role': 'assistant',
						'tool_calls': [
							{
								'id': 'step',
								'type': 'function',
								'function': {
									'name': 'AgentOutput',
									'arguments': '{"action":[{"done":{"text":"Finished","success":true}}]}',
								},
							}
						],
					},
				}
			],
		}
	)
	endpoint = dict(model='deepseek-flash', base_url=model_server.url_for('/'), api_key='local-test')
	settings = RunSettings(
		profile='debug',
		execution=ExecutionModel(**endpoint, timeout=10, max_tokens=300),
		evaluation=EvaluationModel(**endpoint, reasoning={}),
		headless=True,
		user_data_dir=str(tmp_path / 'agent'),
	)
	events: list[RunEvent] = []
	run = BrowserUseRun(httpserver.url_for('/task'), 'Read the page', settings, on_event=events.append)
	try:
		for step in range(2):
			run.command('predict')
			run._browser.evaluate(f"history.replaceState(null,'','?version={step}')")
			state = run.command('act')
			assert state['stale_retries'] == step + 1
			assert not state['forced_after_stale']
		run.command('predict')
		run._browser.evaluate("history.replaceState(null,'','?version=ui')")
		state = run.command('act', {'fingerprint': 'obsolete'})
		assert state['stale_retries'] == 2 and state['forced_after_stale'] == 0
		assert events[-1].payload['stale_reason'] == 'ui_out_of_sync'
		run.command('predict')
		run._state['decision']['requires_fresh_target'] = requires_fresh_target
		run._browser.evaluate("history.replaceState(null,'','?version=third')")
		state = run.command('tick')
		if requires_fresh_target:
			assert state['status'] == 'ready' and state['stale_retries'] == 3 and not state['forced_after_stale']
			assert not any(event.name == 'action_started' for event in events)
			return
		assert state['status'] == 'done'
		assert state['stale_reasons'] == {'document_changed': 3, 'ui_out_of_sync': 1}
		assert state['forced_after_stale'] == 1
		(action,) = (event.payload['action'] for event in events if event.name == 'action_result')
		assert action['forced_after_stale'] and action['stale_reasons'] == ['document_changed'] * 3
		assert action['result'][-1]['is_done'] and action['result'][-1]['success']
	finally:
		run.close()
