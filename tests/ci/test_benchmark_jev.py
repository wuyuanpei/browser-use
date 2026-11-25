"""PR #1 workbench integration with the 0.13.14 Jev Agent, using only local model services."""

import csv
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from browser_use.benchmark.baseline import BaselineManager
from browser_use.benchmark.engine import BrowserUseRun
from browser_use.benchmark.models import (
	EvaluationModel,
	ExecutionModel,
	ModelEndpoint,
	RunSettings,
	load_profile_environment,
	runtime_settings,
)
from browser_use.benchmark.token_usage import aggregate_token_usage, token_usage


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
		assert runtime_settings('jev', mode, screenshot_policy='always').screenshot_policy == 'always'
		assert runtime_settings('jev', mode, screenshot_policy='smart').screenshot_policy == 'smart'
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
	row = batch['attempts'][0]
	assert row['screenshot_policy'] == 'always' and row['jev_model'] == manifest['jev_model']
	assert row['vision_disabled'] is None
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


def _save_work_fixture(httpserver, httpserver_ipv4, tmp_path, monkeypatch, policy, *, image_error=None):
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
		if tool == 'AnswerReply':
			action = {'text': 'The page shows Saved.'}
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
							{'id': 'step', 'type': 'function', 'function': {'name': tool, 'arguments': json.dumps(action)}}
						],
					},
				}
			],
			'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15},
		}
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
		execution=ExecutionModel(**endpoint, timeout=10, max_tokens=500),
		evaluation=EvaluationModel(**endpoint, reasoning={}),
		jev=ModelEndpoint(model='local-jev', base_url=httpserver_ipv4.url_for('/').rstrip('/'), api_key='dummy-jev-key'),
		headless=True,
		user_data_dir=str(tmp_path / 'profile'),
	)
	return settings, saved, requests


@pytest.mark.parametrize('policy', ['off', 'always'])
def test_workbench_real_page_repredicts_and_keeps_per_role_evidence(
	httpserver: HTTPServer, httpserver_ipv4: HTTPServer, tmp_path: Path, monkeypatch, policy
):
	settings, saved, requests = _save_work_fixture(httpserver, httpserver_ipv4, tmp_path, monkeypatch, policy)
	run = BrowserUseRun(
		httpserver.url_for('/task'),
		'Press Save and report the outcome.',
		settings,
		on_event=lambda _: None,
		jev_artifact_dir=tmp_path / 'attempt',
	)
	try:
		first = run.command('tick')
		assert first['history'][-1]['operation'] == 'delegate_subgoal'
		discarded = run.command('predict')['decision']
		assert discarded['decision_source'] == 'system1'
		run._browser.evaluate("history.replaceState(null,'','?after=stale')")
		retry = run.command('act')
		assert retry['stale_retries'] == 1 and not saved
		run.command('tick')
		final = run.command('tick')
		assert final['status'] == 'done' and final['result']['markdown'] == 'The page shows Saved.' and len(saved) == 1
		assert run._browser.evaluate("document.querySelector('output').textContent") == 'Saved'
		assert final['execution_mode'] == 'isolated' and final['system1_provider'] == 'jev'
		calls = {c['id']: c for c in final['model_calls']}
		assert final['screenshot_policy'] == policy and final['screenshot_mode'] == (policy == 'always')
		assert final['vision_disabled'] is None
		jev_calls = [call for call in calls.values() if call['kind'] == 'jev']
		assert len(jev_calls) == len(requests) == 3
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
		assert (roles['planner']['calls'], roles['system1']['calls'], roles['answer']['calls']) == (2, 3, 1)
		assert roles['system1']['vision_requests'] == (3 if policy == 'always' else 0)
		assert roles['system1']['prompt_tokens'] == 75 and final['token_usage']['jev_total_tokens'] == 81
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
	initial = manager.start(['save_fixture'], 1, 'jev', 'isolated', screenshot_policy=settings.screenshot_policy)
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
	settings, saved, requests = _save_work_fixture(httpserver, httpserver_ipv4, tmp_path, monkeypatch, 'smart')
	manager, batch, running_status = _run_save_batch(httpserver, tmp_path, monkeypatch, settings)
	row = batch['attempts'][0]
	assert row['status'] == 'passed' and saved == [True]
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
