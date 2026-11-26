"""PR #1 workbench integration with the 0.13.14 Jev Agent, using only local model services."""

import asyncio
import csv
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from browser_use.benchmark.baseline import BaselineManager
from browser_use.benchmark.chrome_attachment import _OwnedClient
from browser_use.benchmark.engine import BrowserUnavailable, BrowserUseRun
from browser_use.benchmark.models import (
	EvaluationModel,
	ExecutionModel,
	ModelEndpoint,
	RunSettings,
	load_profile_environment,
	runtime_settings,
)
from browser_use.benchmark.token_usage import aggregate_token_usage, token_usage


@pytest.mark.parametrize('actual_context', ['dashboard', 'other'])
async def test_chrome_120_default_tab_requires_dashboard_context(actual_context, monkeypatch):
	client = _OwnedClient('ws://127.0.0.1', 'http://127.0.0.1:8824')
	client.context_id = 'dashboard'
	unsupported_id = RuntimeError({'code': -32000, 'message': 'Failed to find browser context with id dashboard'})
	request = AsyncMock(
		side_effect=[
			unsupported_id,
			{'targetId': 'new'},
			{'targetInfo': {'url': 'about:blank', 'browserContextId': actual_context}},
			{},
		]
	)
	monkeypatch.setattr(client, '_request', request)
	cleanup = AsyncMock()
	monkeypatch.setattr(client, 'close_case', cleanup)
	if actual_context == 'dashboard':
		assert await client.new_page() == 'new'
		assert request.await_args_list[-1].args == ('Target.attachToTarget', {'targetId': 'new', 'flatten': True})
	else:
		with pytest.raises(BrowserUnavailable, match='未落在测试台所在的 Chrome 资料'):
			await client.new_page()
		cleanup.assert_awaited_once()
		assert request.await_count == 3
	assert request.await_args_list[1].args == ('Target.createTarget', {'url': 'about:blank', 'background': True})


@pytest.mark.parametrize('recover', [True, False])
def test_workbench_resamples_unavailable_document_once(monkeypatch, recover):
	from browser_use.benchmark.engine import _BoundBrowser

	browser = _BoundBrowser(SimpleNamespace(run=lambda call: call))
	monkeypatch.setattr(browser, 'bind', lambda: 'owned-tab')
	pages = [None, {'url': 'about:blank', 'text': '', 'actions': []} if recover else None]
	calls = []

	def sample(_script):
		calls.append(True)
		return pages.pop(0)

	monkeypatch.setattr(browser, 'evaluate', sample)
	if recover:
		assert browser.observe()['target_id'] == 'owned-tab'
	else:
		with pytest.raises(RuntimeError, match='Browser document is unavailable'):
			browser.observe()
	assert len(calls) == 2


def test_workbench_explains_prediction_deadline_before_dispatch(monkeypatch):
	def expire(operation, **_kwargs):
		operation.close()
		raise TimeoutError

	run = object.__new__(BrowserUseRun)
	monkeypatch.setattr(run, '_bridge', SimpleNamespace(run=expire), raising=False)
	run._state = {'decision_rounds': 0, 'model_calls': []}
	run._should_cancel = None
	with pytest.raises(TimeoutError, match='prediction timed out before dispatch'):
		run._choose()
	assert run._state['decision_rounds'] == 0


def test_modes_and_model_profiles_are_independent(monkeypatch, tmp_path: Path):
	for name in ('TYPESAFE_API_KEY', 'REPORT_TEXT_MODEL', 'REPORT_TEXT_MODEL_BASE_URL', 'REPORT_TEXT_MODEL_REASONING'):
		monkeypatch.delenv(name, raising=False)
	monkeypatch.setenv('PLANNER_API_KEY', 'mock-plan-key')
	monkeypatch.setenv('TYPESAFE_API_KEY', '')
	monkeypatch.setenv('TEXT_MODEL_API_KEY', 'mock-eval-key')
	monkeypatch.setenv('REPORT_TEXT_MODEL_API_KEY', 'mock-report-key')
	assert runtime_settings().execution_mode == 'direct'
	with pytest.raises(ValueError, match='直接执行'):
		runtime_settings(mode='direct', screenshot_policy='always')
	with pytest.raises(ValueError, match='直接执行'):
		runtime_settings(mode='direct', screenshot_policy='smart')
	with pytest.raises(ValueError, match='TYPESAFE_API_KEY'):
		runtime_settings(mode='isolated')
	path = tmp_path / 'profile.env'
	path.write_text('TYPESAFE_API_KEY=mock-local-key\nUNRELATED_CONFIG=ignored\n', encoding='utf-8')
	load_profile_environment(path)
	for mode in ('isolated', 'collaborative'):
		assert runtime_settings('jev', mode, screenshot_policy='always', vision_escalation=True).vision_escalation is None
		assert runtime_settings('jev', mode, screenshot_policy='off', vision_escalation=False).vision_escalation is None
		assert runtime_settings('jev', mode, screenshot_policy='smart').vision_escalation is True
		assert runtime_settings('jev', mode, screenshot_policy='smart', vision_escalation=False).vision_escalation is False
		for profile in ('debug', 'final_report'):
			settings = runtime_settings('jev', mode, model_profile=profile)
			assert settings.execution_mode == mode and settings.profile == profile
			assert settings.jev is not None
			assert settings.system1_config()['provider'] == 'jev' and settings.jev.api_key == 'mock-local-key'
			public = json.dumps(settings.model_dump())
			assert all(key not in public for key in ('mock-local-key', 'mock-plan-key', 'mock-eval-key', 'mock-report-key'))
	assert runtime_settings().system1_config()['provider'] == 'browser_use'


def test_baseline_archives_jev_policy_frames_and_report_column(tmp_path, monkeypatch):
	for name in ('PLANNER_API_KEY', 'TEXT_MODEL_API_KEY', 'TYPESAFE_API_KEY'):
		monkeypatch.setenv(name, 'mock-benchmark-key')
	manager = BaselineManager(tmp_path, 'http://127.0.0.1:8824')
	monkeypatch.setattr(manager, 'work', lambda *_: None)
	manager.start([manager.tasks[0]['id']], 1, 'jev', 'isolated', screenshot_policy='always')
	batch = manager.batch
	assert batch and batch['screenshot_policy'] == 'always' and batch['screenshot_mode']
	run_id = batch['id']
	folder = manager.folder(run_id)
	manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
	assert manifest['screenshot_policy'] == 'always' and manifest['jev_model'] == batch['system1']['model']
	assert manifest['vision_escalation'] is None
	row = batch['attempts'][0]
	assert row['screenshot_policy'] == 'always' and row['jev_model'] == manifest['jev_model']
	assert row['vision_disabled'] is None
	assert all(record['min_read_operation_probability'] == 0.6 for record in (manifest, batch, row))
	row['vision_requests'] = 3
	manager.write_summary(batch)
	with (folder / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
		summary = next(csv.DictReader(stream))
		assert summary['vision_requests'] == '3' and summary['vision_disabled'] == ''
	assert '带图请求' in (folder / 'report.zh.md').read_text(encoding='utf-8')
	name = 'system1_images/2-3.jpg'
	frame = manager.attempt_folder(run_id, row['id']) / name
	frame.parent.mkdir(parents=True)
	frame.write_bytes(b'archived model frame')
	assert manager.frame(run_id, row['id'], name) == frame.read_bytes()
	with pytest.raises(ValueError, match='Invalid screenshot name'):
		manager.frame(run_id, row['id'], 'system1_images/../trace.json')


def test_server_accepts_visual_presets_and_freezes_escalation_for_reset_and_batch(tmp_path, monkeypatch):
	from browser_use.benchmark import server

	for name in ('PLANNER_API_KEY', 'TEXT_MODEL_API_KEY', 'TYPESAFE_API_KEY'):
		monkeypatch.setenv(name, 'local-key')
	manager = BaselineManager(tmp_path, server.ORIGIN)
	monkeypatch.setattr(manager, 'work', lambda *_: None)
	monkeypatch.setattr(server, 'BASELINE', manager)
	monkeypatch.setattr(server, 'AGENT', None)
	monkeypatch.setattr(server, 'CHROME', None)
	opened = []

	def open_run(url, goal, settings, **kwargs):
		opened.append((url, settings.vision_escalation))
		return SimpleNamespace(snapshot=lambda: {}, close=lambda: None)

	monkeypatch.setattr(server, 'BrowserUseRun', open_run)
	preset_ids = ['112_visual_cards', '113_visual_icons', '114_visual_cards_plain', '115_scroll_panel', '116_dense']
	for task in (task for task in manager.tasks if task['id'] in preset_ids):
		scenario = task['url'].split('scenario=')[-1]
		server.command(
			'reset',
			{
				'scenario': scenario,
				'goal': task['goal'],
				'execution_mode': 'isolated',
				'screenshot_policy': 'smart',
				'vision_escalation': False,
			},
		)
	assert opened == [
		(server.ORIGIN + '/fixture.html?scenario=' + scene, False)
		for scene in ('visual-cards', 'visual-icons', 'visual-cards-plain', 'scroll-panel', 'dense')
	]
	batch = server.command(
		'baseline/start',
		{'task_ids': preset_ids, 'execution_mode': 'isolated', 'screenshot_policy': 'smart', 'vision_escalation': False},
	)['batch']
	manifest = json.loads((manager.folder(batch['id']) / 'manifest.json').read_bytes())
	assert manager._settings is not None and manager._settings.vision_escalation is False
	assert batch['vision_escalation'] is False and manifest['vision_escalation'] is False
	assert manifest['suite_version'] == '2026-11-26.1'


@pytest.mark.parametrize(
	('policy', 'pipeline', 'effective'),
	[
		('off', None, 'pr5_with_regions'),
		('always', None, 'extended'),
		('off', 'extended', 'extended'),
		('smart', 'pr5', 'pr5'),
	],
)
def test_batch_api_freezes_candidate_pipeline(tmp_path, monkeypatch, policy, pipeline, effective):
	from browser_use.benchmark import server

	for name in ('PLANNER_API_KEY', 'TEXT_MODEL_API_KEY', 'TYPESAFE_API_KEY'):
		monkeypatch.setenv(name, 'local-test-key')
	manager = BaselineManager(tmp_path, server.ORIGIN)
	monkeypatch.setattr(manager, 'work', lambda *_: None)
	monkeypatch.setattr(server, 'BASELINE', manager)
	monkeypatch.setattr(server, 'CHROME', None)
	monkeypatch.setattr(server, 'AGENT', None)
	body = {'task_ids': ['116_dense'], 'execution_mode': 'isolated', 'system1_provider': 'jev', 'screenshot_policy': policy}
	if pipeline is not None:
		body['candidate_pipeline'] = pipeline
	batch = server.command('baseline/start', body)['batch']
	manifest = json.loads((manager.folder(batch['id']) / 'manifest.json').read_bytes())
	assert manager._settings is not None and manager._settings.effective_candidate_pipeline == effective
	assert manager._settings.model_dump()['effective_candidate_pipeline'] == effective
	assert manifest['candidate_pipeline_requested'] == pipeline
	assert all(row['candidate_pipeline'] == effective for row in (manifest, batch, batch['attempts'][0]))


def test_jev_retry_and_legacy_token_totals():
	legacy = {'kind': 'jev', 'response': {'usage': {'input_tokens': 7}}}
	new = {
		'kind': 'jev',
		'role': 'system1',
		'attempts': [{'status': 429}, {'status': 200}],
		'response': {'usage': {'input_tokens': 11, 'output_tokens': 3}},
	}
	usage = token_usage([legacy, new])
	assert usage['jev_input_tokens'] == 18 and usage['jev_output_tokens'] == 3 and usage['jev_total_tokens'] == 21
	assert usage['missing_usage_calls'] == 1
	archived = token_usage([legacy])
	del archived['jev_output_tokens']
	assert aggregate_token_usage([{'token_usage': archived}, {'token_usage': usage}])['jev_output_tokens'] == 3


_VISUAL_REASON = 'Find Save beside the current record using its row on screen.'


def _save_work_fixture(
	httpserver,
	httpserver_ipv4,
	tmp_path,
	monkeypatch,
	policy,
	*,
	image_error=None,
	vision_escalation=None,
	planner_parse_failures=0,
	hold_planner_retry=None,
	omit_first_parse_usage=False,
	redelegate=False,
):
	# Windows may supply a system proxy for localhost. The model server and page both run inside this test.
	monkeypatch.setenv('NO_PROXY', 'localhost,127.0.0.1,::1')
	httpserver.expect_request('/task').respond_with_data(
		'<link rel="icon" href="data:,"><title>Save work</title><button id="save" onclick="document.querySelector(\'output\').textContent=\'Saved\'; fetch(\'/save\',{method:\'POST\'})">Save</button><output>Ready</output>',
		content_type='text/html',
	)
	saved = []

	def save(_):
		saved.append(True)
		return Response('ok')

	httpserver.expect_request('/save', method='POST').respond_with_handler(save)
	declarations = iter(
		(
			{'screenshot_required': False, 'screenshot_reason': 'The Save label is enough.'},
			{},
			{'screenshot_required': True, 'screenshot_reason': _VISUAL_REASON},
		)
	)

	def completion(request):
		nonlocal planner_parse_failures, omit_first_parse_usage
		body = request.get_json()
		assert not any(
			part.get('type') == 'image_url'
			for message in body['messages']
			if isinstance(message.get('content'), list)
			for part in message['content']
		)
		if 'tool_choice' not in body:
			evidence = json.loads(body['messages'][-1]['content'])['independent_evidence']
			final = evidence['samples'][-1]['evidence']['sampled_page']
			text = evidence['texts'][final['text']['text_ref']]['text']
			verdict = {'verdict': 'met' if 'Saved' in text else 'not_met', 'reason': text}
			return Response(
				json.dumps({'choices': [{'message': {'content': json.dumps(verdict)}, 'finish_reason': 'stop'}]}),
				mimetype='application/json',
			)
		tool = body['tool_choice']['function']['name']
		if tool != 'AnswerReply' and hold_planner_retry is not None and planner_parse_failures == 0:
			retry_started, release_retry = hold_planner_retry
			retry_started.set()
			release_retry.wait(timeout=30)
		if tool != 'AnswerReply' and planner_parse_failures > 0:
			planner_parse_failures -= 1
			action = None
		elif tool == 'AnswerReply':
			action = {'text': 'The page shows Saved.'}
		elif redelegate and (
			handbacks := [
				message
				for message in body['messages']
				if message['role'] == 'user' and '<system1_handback>' in str(message['content'])
			]
		):
			if 'sg2' in str(handbacks[-1]['content']):
				action = {'action': [{'finish': {'success': True, 'reason': 'Observed Saved'}}]}
			else:
				action = {
					'action': [
						{
							'delegate_subgoal': {
								'objective': 'Verify the save',
								'completion': 'Saved is visible',
								'reason': 'Check the result',
							}
						}
					]
				}
		elif (policy != 'smart' or saved) and any(
			message['role'] == 'user' and '<system1_handback>' in str(message['content']) for message in body['messages']
		):
			action = {'action': [{'finish': {'success': True, 'reason': 'Observed Saved'}}]}
		else:
			action = {
				'action': [
					{
						'delegate_subgoal': {
							'objective': 'Save the record',
							'completion': 'The page shows Saved',
							'reason': 'Apply the requested change',
							'requires_action': True,
							'may_submit': True,
							**(next(declarations) if policy == 'smart' else {}),
						}
					}
				]
			}
		response = {
			'id': 'local',
			'object': 'chat.completion',
			'created': 1,
			'model': 'mock',
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
									'name': tool,
									'arguments': json.dumps(action) if action is not None else '{"action":[',
								},
							}
						],
					},
				}
			],
			'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15},
		}
		if action is None and omit_first_parse_usage:
			response.pop('usage')
			omit_first_parse_usage = False
		return Response(json.dumps(response), mimetype='application/json')

	requests = []

	def choose(request):
		if request.mimetype == 'multipart/form-data':
			assert policy != 'off'
			data = json.loads(request.form['request'])
			jpeg = request.files['page-screenshot'].read()
			assert jpeg.startswith(b'\xff\xd8') and request.headers['TypeSafe-Danger-Zone-Images'] == 'true'
		else:
			assert request.mimetype == 'application/json' and (policy != 'always' or image_error)
			data, jpeg = request.get_json(), None
		requests.append((data, jpeg))
		if jpeg and image_error:
			message = 'Image input rejected' if image_error == 415 else 'Authentication failed'
			return Response(json.dumps({'error': message}), status=image_error, mimetype='application/json')
		questions = data['questions']
		answers = {}
		for name, question in questions.items():
			criteria = question['criteria']
			if name == 'operation':
				choice = 'STEP_DONE' if 'Saved' in data['state']['page']['text'] else 'CLICK'
				if policy == 'smart' and not jpeg:
					choice = 'UNABLE'
			elif name == 'click_target':
				choice = next(k for k, v in criteria.items() if '"Save"' in v['element'])
			else:
				choice = next(iter(criteria))
			assert choice in criteria
			answers[name] = {'choice': choice, 'confidence': 1.0, 'probabilities': {k: float(k == choice) for k in criteria}}
		return Response(
			json.dumps({'model': 'local-jev', 'answers': answers, 'usage': {'input_tokens': 25, 'output_tokens': 2}}),
			mimetype='application/json',
		)

	httpserver_ipv4.expect_request('/chat/completions', method='POST').respond_with_handler(completion)
	httpserver_ipv4.expect_request('/v1/systemone', method='POST').respond_with_handler(choose)
	endpoint = dict(model='mock', base_url=httpserver_ipv4.url_for('/').rstrip('/'), api_key='dummy-llm-key')
	settings = RunSettings(
		profile='debug',
		execution_mode='isolated',
		screenshot_policy=policy,
		vision_escalation=vision_escalation,
		execution=ExecutionModel(**endpoint, timeout=10, max_tokens=500),
		evaluation=EvaluationModel(**endpoint, reasoning={}),
		jev=ModelEndpoint(model='local-jev', base_url=httpserver_ipv4.url_for('/').rstrip('/'), api_key='dummy-jev-key'),
		headless=True,
		user_data_dir=str(tmp_path / 'profile'),
	)
	return settings, saved, requests


@pytest.mark.parametrize(('policy', 'redelegate'), [('off', False), ('always', False), ('off', True)])
def test_workbench_real_page_repredicts_and_keeps_per_role_evidence(
	httpserver: HTTPServer, httpserver_ipv4: HTTPServer, tmp_path: Path, monkeypatch, policy, redelegate
):
	settings, saved, requests = _save_work_fixture(
		httpserver, httpserver_ipv4, tmp_path, monkeypatch, policy, redelegate=redelegate
	)
	run = BrowserUseRun(
		httpserver.url_for('/task'),
		'Press Save and report the outcome.',
		settings,
		on_event=lambda _: None,
		jev_artifact_dir=tmp_path / 'attempt',
	)
	try:
		native = run._native
		assert native is not None
		predict = native._get_next_action
		timeout = True

		async def time_out_once(summary):
			nonlocal timeout
			if timeout:
				timeout = False
				raise TimeoutError('simulated planner call timeout')
			await predict(summary)

		monkeypatch.setattr(native, '_get_next_action', time_out_once)
		first = run.command('tick')
		assert not timeout
		assert first['history'][-1]['operation'] == 'delegate_subgoal'
		run.record_timeout_evidence()
		partial = run.snapshot()
		assert not partial.get('result')
		assert 'Save work' in partial['partial_result']['markdown'] or 'Ready' in partial['partial_result']['markdown']
		discarded = run.command('predict')['decision']
		assert discarded['decision_source'] == 'system1'
		run._browser.evaluate("history.replaceState(null,'','?after=stale')")
		retry = run.command('act')
		assert retry['stale_retries'] == 1 and not saved
		run.command('tick')
		if redelegate:
			run.command('tick')
		final = run.command('tick')
		assert final['replans'] == int(redelegate)
		assert final['status'] == 'done' and final['result']['markdown'] == 'The page shows Saved.' and len(saved) == 1
		assert run._browser.evaluate("document.querySelector('output').textContent") == 'Saved'
		assert final['execution_mode'] == 'isolated' and final['system1_provider'] == 'jev'
		calls = {c['id']: c for c in final['model_calls']}
		assert final['screenshot_policy'] == policy and final['screenshot_mode'] == (policy == 'always')
		assert final['vision_disabled'] is None
		jev_calls = [call for call in calls.values() if call['kind'] == 'jev']
		assert len(jev_calls) == len(requests) == (4 if redelegate else 3)
		for call, (body, jpeg) in zip(jev_calls, requests, strict=True):
			assert call['request'] == body
			assert call['vision'] == (policy == 'always')
			if jpeg:
				ref = body['state']['page']['screenshot']
				assert body['assets'][ref]['part'] == 'page-screenshot'
				name = call['request_images']['page-screenshot']
				assert name == f'system1_images/{call["bu_step"]}-{call["id"]}.jpg'
				assert (tmp_path / 'attempt' / name).read_bytes() == run.request_image(ref) == jpeg
			else:
				assert not call['request_images'] and 'assets' not in body
		assert calls[discarded['model_call_id']]['decision']['stale_reason'] == 'document_changed'
		for decision in final['decisions']:
			expected_role = 'system1' if decision['decision_source'] == 'system1' else 'planner'
			assert calls[decision['model_call_id']]['role'] == expected_role
		answer_call = next(c for c in calls.values() if c['role'] == 'answer')
		assert 'decision' not in answer_call
		roles = final['jev_usage_by_role']
		if redelegate:
			assert (roles['planner']['calls'], roles['system1']['calls'], roles['answer']['calls']) == (3, 4, 1)
			assert roles['system1']['prompt_tokens'] == 100 and final['token_usage']['jev_total_tokens'] == 108
		else:
			assert (roles['planner']['calls'], roles['system1']['calls'], roles['answer']['calls']) == (2, 3, 1)
			assert roles['system1']['prompt_tokens'] == 75 and final['token_usage']['jev_total_tokens'] == 81
		assert roles['system1']['vision_requests'] == (len(requests) if policy == 'always' else 0)
		events = [json.loads(line) for line in Path(final['jev_trace_path']).read_text(encoding='utf-8').splitlines()]
		assert events[-1]['event'] == 'usage_summary'
		assert [e['status'] for e in events if e['event'] == 'action_result'] == ['discarded', 'executed_ok']
		assert all(secret not in json.dumps(final) for secret in ('dummy-llm-key', 'dummy-jev-key'))
	finally:
		run.close()


def _run_save_batch(httpserver, tmp_path, monkeypatch, settings):
	monkeypatch.setattr('browser_use.benchmark.baseline.runtime_settings', lambda *args, **kwargs: settings)
	running_status = []

	def open_run(*args, on_event, **kwargs):
		def observe(event):
			call = event.payload.get('call', {})
			if event.name == 'model_start' and call.get('kind') == 'jev':
				running_status.append((call['vision'], event.snapshot['vision_disabled']))
			on_event(event)

		return BrowserUseRun(*args, on_event=observe, **kwargs)

	manager = BaselineManager(tmp_path / 'batches', httpserver.url_for('/'), run_factory=open_run)
	task = {
		'id': 'save_fixture',
		'name': 'Save work',
		'url': httpserver.url_for('/task'),
		'goal': 'Press Save and report the outcome.',
	}
	manager.tasks = [task]
	initial = manager.start(
		['save_fixture'],
		1,
		'jev',
		'isolated',
		screenshot_policy=settings.screenshot_policy,
		vision_escalation=settings.vision_escalation,
	)
	assert manager.thread is not None
	manager.thread.join(timeout=90)
	assert not manager.thread.is_alive()
	return manager, manager.read_run(initial['batch']['id']), running_status


@contextmanager
def _open_inspector(manager, settings, tmp_path, monkeypatch):
	from browser_use.benchmark import server

	monkeypatch.setenv('TYPESAFE_API_KEY', 'dummy-local-key')
	monkeypatch.setattr(server, 'BASELINE', manager)
	with server.InspectorServer(('127.0.0.1', 0), server.Handler) as httpd:
		port = httpd.server_address[1]
		origin = f'http://127.0.0.1:{port}'
		monkeypatch.setattr(server, 'ORIGIN', origin)
		monkeypatch.setattr(server, 'PORT', port)
		thread = threading.Thread(target=httpd.serve_forever, daemon=True)
		thread.start()
		try:
			viewer = BrowserUseRun(
				origin + '/#baseline',
				'Inspect the saved run.',
				settings.model_copy(
					update={
						'execution_mode': 'direct',
						'jev': None,
						'screenshot_policy': 'off',
						'vision_escalation': None,
						'user_data_dir': str(tmp_path / 'inspector-profile'),
					}
				),
			)
			try:
				deadline = time.monotonic() + 10
				while not viewer._browser.evaluate("Boolean(document.querySelector('#history .trace-row'))"):
					assert time.monotonic() < deadline, 'Inspector did not render the archived model calls'
					time.sleep(0.05)
				yield viewer._browser.evaluate
			finally:
				viewer.close()
		finally:
			httpd.shutdown()
			thread.join()


def test_baseline_recovers_after_jev_rejects_image(httpserver, httpserver_ipv4, tmp_path, monkeypatch):
	settings, saved, requests = _save_work_fixture(httpserver, httpserver_ipv4, tmp_path, monkeypatch, 'always', image_error=415)
	manager, batch, running_status = _run_save_batch(httpserver, tmp_path, monkeypatch, settings)
	run_id = batch['id']
	row = batch['attempts'][0]
	assert row['status'] == 'passed' and saved == [True]
	assert row['vision_disabled'] == 415 and row['vision_requests'] == 1
	assert running_status == [(True, None), (False, 415), (False, 415)]
	snapshot = manager.read_attempt(run_id, row['id'])
	assert snapshot['vision_disabled'] == 415
	assert snapshot['jev_usage_by_role']['system1']['decisions'] == 2
	calls = snapshot['model_calls']
	rejected, fallback, done = (call for call in calls if call['kind'] == 'jev')
	assert rejected['status'] == 'error' and rejected['vision_rejected'] is True and rejected['attempts'][-1]['status'] == 415
	assert fallback['id'] == rejected['id'] + 1 and fallback['observation_index'] == rejected['observation_index']
	assert fallback['status'] == done['status'] == 'success'
	assert fallback['response']['answers']['operation']['choice'] == 'CLICK'
	assert done['response']['answers']['operation']['choice'] == 'STEP_DONE'
	assert sum(call['role'] == 'planner' for call in calls if 'role' in call) == 2
	assert [body for body, _ in requests] == [call['request'] for call in (rejected, fallback, done)]
	image, text, later = (jpeg for _, jpeg in requests)
	assert text is None and later is None
	assert image == manager.frame(run_id, row['id'], rejected['request_images']['page-screenshot'])
	folder = manager.folder(run_id)
	with (folder / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
		assert next(csv.DictReader(stream))['vision_disabled'] == '415'
	assert '| 415 |' in (folder / 'report.zh.md').read_text(encoding='utf-8')
	with _open_inspector(manager, settings, tmp_path, monkeypatch) as evaluate:
		hint = "document.querySelector('#screenshot-mode-hint').textContent"
		assert '415' not in evaluate(hint)
		evaluate(f'document.querySelector(\'[data-call-id="{rejected["id"]}"]\').click()')
		assert 'HTTP 415' in evaluate(hint)
		evaluate(
			"const mode=document.querySelector('#execution-mode'); mode.value='isolated'; mode.dispatchEvent(new Event('change')); document.querySelector('#screenshot-smart').click()"
		)
		assert evaluate(hint) == 'Planner 在复盘时决定是否向 Jev 提交截图。'


def test_smart_batch_counts_reviews_and_displays_planner_declarations(httpserver, httpserver_ipv4, tmp_path, monkeypatch):
	settings, saved, requests = _save_work_fixture(
		httpserver, httpserver_ipv4, tmp_path, monkeypatch, 'smart', vision_escalation=False
	)
	manager, batch, running_status = _run_save_batch(httpserver, tmp_path, monkeypatch, settings)
	row = batch['attempts'][0]
	assert row['status'] == 'passed' and saved == [True]
	assert row['vision_escalation'] is False and batch['vision_escalation'] is False
	assert running_status == [(False, None), (False, None), (True, None), (True, None)]
	assert [bool(image) for _, image in requests] == [False, False, True, True]
	snapshot = manager.read_attempt(batch['id'], row['id'])
	jev = [c for c in snapshot['model_calls'] if c['kind'] == 'jev']
	assert [c['vision_trigger'] for c in jev] == [None, None, 'planner', 'planner']
	assert jev[-2]['request']['state']['current_subgoal']['screenshot_guidance']['reason'] == _VISUAL_REASON
	counts = {'planner_screenshot_requests': 1, 'review_delegations': 2, 'screenshot_decisions_missing': 1}
	folder = manager.folder(batch['id'])
	with (folder / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
		csv_row = next(csv.DictReader(stream))
	for key, count in counts.items():
		assert snapshot[key] == row[key] == batch[key] == int(csv_row[key]) == count
	report = (folder / 'report.zh.md').read_text(encoding='utf-8')
	assert 'Planner 请求截图 | 复盘委托 | 复盘漏填' in report and '| 1 | 2 | 1 |' in report
	with _open_inspector(manager, settings, tmp_path, monkeypatch) as evaluate:
		kinds = evaluate("[...document.querySelectorAll('#history .trace-kind')].map(e => e.textContent)")
		assert kinds.count('Jev（截图·Planner）') == 2
		assert 'Planner 请求截图 1 · 复盘委托 2 · 复盘漏填 1' in evaluate(
			"document.querySelector('#baseline-summary').textContent"
		)
		declarations = evaluate("[...document.querySelectorAll('.screenshot-declaration')].map(e => e.innerText)")
		assert ['screenshot_required: ' + label for label in ('false', '未填', 'true')] == [
			d.splitlines()[0] for d in declarations
		]
		assert _VISUAL_REASON in declarations[-1]


def test_workbench_auth_failure_is_not_an_image_rejection(httpserver, httpserver_ipv4, tmp_path, monkeypatch):
	settings, saved, requests = _save_work_fixture(httpserver, httpserver_ipv4, tmp_path, monkeypatch, 'always', image_error=401)
	manager, batch, running_status = _run_save_batch(httpserver, tmp_path, monkeypatch, settings)
	row = batch['attempts'][0]
	assert row['vision_disabled'] is None and not saved and running_status == [(True, None)]
	snapshot = manager.read_attempt(batch['id'], row['id'])
	(jev,) = (call for call in snapshot['model_calls'] if call['kind'] == 'jev')
	assert jev['status'] == 'error' and jev['attempts'][-1]['status'] == 401 and not jev.get('vision_rejected')
	assert len(requests) == 1 and requests[0][1]


def test_escalation_and_recovered_parse_error_in_workbench(httpserver, httpserver_ipv4, tmp_path, monkeypatch):
	settings, saved, requests = _save_work_fixture(
		httpserver, httpserver_ipv4, tmp_path, monkeypatch, 'smart', planner_parse_failures=1
	)
	manager, batch, _ = _run_save_batch(httpserver, tmp_path, monkeypatch, settings)
	row = batch['attempts'][0]
	assert row['status'] == 'passed' and saved == [True]
	snapshot = manager.read_attempt(batch['id'], row['id'])
	folder = manager.folder(batch['id'])
	manifest = json.loads((folder / 'manifest.json').read_bytes())
	assert all(level['vision_escalation'] is True for level in (manifest, batch, row, snapshot))
	reason = 'unable_probability'
	for level in (batch, row, snapshot):
		assert level['escalations'] == level['escalations_executed'] == 1
		assert level['escalations_by_reason'] == {reason: 1}
	assert [bool(image) for _, image in requests] == [False, True, True]
	calls = snapshot['model_calls']
	text, image, sticky = [call for call in calls if call['kind'] == 'jev']
	assert text['outcome'] == 'escalated' and text['escalation_from'] == reason
	assert image['escalated_from_call_id'] == text['id'] and image['bu_step'] == text['bu_step']
	assert image['observation_index'] == text['observation_index']
	assert image['vision_trigger'] == sticky['vision_trigger'] == f'escalation:{reason}'
	assert 'escalated_from_call_id' not in sticky
	assert image['response']['answers']['operation']['choice'] == 'CLICK'
	assert snapshot['history'][-1]['model_call_id'] == image['id']
	first, corrected, finish = [c for c in calls if c['kind'] == 'planner']
	(failure,) = first['parse_failures']
	assert first['status'] == 'error' and failure['raw_text'] == '{"action":[' and failure['planner_feedback_retry']
	assert failure['role'] == 'planner' and not {'attempts', 'retried', 'same_as_previous'}.intersection(failure)
	assert 'Return a valid JSON action' in str(corrected['request']['messages'][-1]['content'])
	assert corrected['status'] == finish['status'] == 'success'
	assert [call['response']['usage']['prompt_tokens'] for call in (first, corrected, finish)] == [12, 12, 12]
	assert snapshot['token_usage']['planner_input_tokens'] == snapshot['jev_usage_by_role']['planner']['prompt_tokens'] == 36
	events = [json.loads(line) for line in Path(snapshot['jev_trace_path']).read_text(encoding='utf-8').splitlines()]
	(logged,) = [event for event in events if event['event'] == 'llm_output_parse_error']
	assert all(logged[key] == value for key, value in failure.items())
	with (folder / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
		archived = next(csv.DictReader(stream))
	assert archived['escalations_by_reason'] == f'{reason}:1'
	assert archived['escalations'] == archived['escalations_executed'] == '1'
	assert f'| 1 | 1 | {reason}:1 |' in (folder / 'report.zh.md').read_text(encoding='utf-8')
	with _open_inspector(manager, settings, tmp_path, monkeypatch) as evaluate:
		use_vision = "document.querySelector('#vision-escalation')"
		assert evaluate(use_vision + '.checked && ' + use_vision + '.disabled')
		evaluate(
			"const mode=document.querySelector('#execution-mode'); mode.value='isolated'; mode.dispatchEvent(new Event('change')); document.querySelector('#screenshot-smart').click()"
		)
		assert evaluate('!' + use_vision + '.disabled')
		kinds = evaluate("[...document.querySelectorAll('#history .trace-kind')].map(e => e.textContent)")
		assert kinds.count('Jev（文本→升级）') == 1 and kinds.count('Jev（截图·升级）') == 2
		notes = evaluate("[...document.querySelectorAll('#history .trace-note')].map(e => e.textContent)")
		assert notes.count(f'升级原因：{reason}') == 3 and '解析失败 1 次，已重试' in notes
		evaluate(f'document.querySelector(\'[data-call-id="{first["id"]}"]\').click()')
		assert json.loads(evaluate("document.querySelector('#model-output').textContent"))['parse_failures'] == [failure]


def test_exhausted_planner_parse_attempts_are_archived(httpserver, httpserver_ipv4, tmp_path, monkeypatch):
	settings, saved, _ = _save_work_fixture(
		httpserver, httpserver_ipv4, tmp_path, monkeypatch, 'off', planner_parse_failures=2, omit_first_parse_usage=True
	)
	manager, batch, _ = _run_save_batch(httpserver, tmp_path, monkeypatch, settings)
	row = batch['attempts'][0]
	assert row['execution_status'] == 'blocked' and not saved
	snapshot = manager.read_attempt(batch['id'], row['id'])
	first, second = [call for call in snapshot['model_calls'] if call['kind'] == 'planner']
	assert first['status'] == second['status'] == 'error'
	assert [call['parse_failures'][0]['planner_feedback_retry'] for call in (first, second)] == [True, False]
	assert first['parse_failures'][0]['usage'] is None
	assert second['response']['usage']['prompt_tokens'] == snapshot['token_usage']['planner_input_tokens'] == 12
	assert snapshot['token_usage']['missing_usage_calls'] == snapshot['jev_usage_by_role']['planner']['missing_usage_calls'] == 1
	assert not snapshot['token_usage']['complete']
	with _open_inspector(manager, settings, tmp_path, monkeypatch) as evaluate:
		assert '解析失败，已放弃' in evaluate("[...document.querySelectorAll('#history .trace-note')].map(e => e.textContent)")


def test_cancelled_planner_retry_keeps_earlier_raw_reply_and_usage(httpserver, httpserver_ipv4, tmp_path, monkeypatch):
	retry_started, release_retry = threading.Event(), threading.Event()
	settings, saved, _ = _save_work_fixture(
		httpserver,
		httpserver_ipv4,
		tmp_path,
		monkeypatch,
		'off',
		planner_parse_failures=1,
		hold_planner_retry=(retry_started, release_retry),
	)
	run = BrowserUseRun(
		httpserver.url_for('/task'),
		'Press Save and report the outcome.',
		settings,
		jev_artifact_dir=tmp_path / 'attempt',
	)
	try:

		async def cancel_second_attempt():
			prediction = asyncio.create_task(run._predict())
			try:
				assert await asyncio.to_thread(retry_started.wait, 25)
			finally:
				prediction.cancel()
				with pytest.raises(asyncio.CancelledError):
					await prediction
				release_retry.set()

		run._bridge.run(cancel_second_attempt(), timeout=35)
		snapshot = run.snapshot()
		planner, cancelled = snapshot['model_calls']
		assert (
			not saved
			and planner['status'] == 'error'
			and cancelled['status'] == 'error'
			and cancelled['error'] == 'CancelledError'
		)
		(failure,) = planner['parse_failures']
		assert failure['planner_feedback_retry'] and failure['raw_text'] == '{"action":['
		assert planner['response']['usage']['prompt_tokens'] == snapshot['token_usage']['planner_input_tokens'] == 12
		assert snapshot['jev_usage_by_role']['planner']['prompt_tokens'] == 12
		assert snapshot['token_usage']['missing_usage_calls'] == 1
	finally:
		release_retry.set()
		run.close()
