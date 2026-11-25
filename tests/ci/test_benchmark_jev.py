"""PR #1 workbench integration with the 0.13.14 Jev Agent, using only local model services."""

import json
from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

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
	with pytest.raises(ValueError, match='TYPESAFE_API_KEY'):
		runtime_settings(mode='isolated')
	path = tmp_path / 'profile.env'
	path.write_text('TYPESAFE_API_KEY=mock-local-key\nUNRELATED_CONFIG=ignored\n', encoding='utf-8')
	load_profile_environment(path)
	for mode in ('isolated', 'collaborative'):
		for profile in ('debug', 'final_report'):
			settings = runtime_settings('jev', mode, model_profile=profile)
			assert settings.execution_mode == mode and settings.profile == profile
			assert settings.jev is not None
			assert settings.system1_config()['provider'] == 'jev' and settings.jev.api_key == 'mock-local-key'
			public = json.dumps(settings.model_dump())
			assert all(key not in public for key in ('mock-local-key', 'mock-plan-key', 'mock-eval-key', 'mock-report-key'))
	assert runtime_settings().system1_config()['provider'] == 'browser_use'


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


def test_workbench_real_page_repredicts_and_keeps_per_role_evidence(
	httpserver: HTTPServer, httpserver_ipv4: HTTPServer, tmp_path: Path, monkeypatch
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

	def completion(request):
		body = request.get_json()
		tool = body['tool_choice']['function']['name']
		if tool == 'AnswerReply':
			action = {'text': 'The page shows Saved.'}
		elif any(message['role'] == 'user' and '<system1_handback>' in str(message['content']) for message in body['messages']):
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

	def choose(request):
		data = request.get_json()
		questions = data['questions']
		answers = {}
		for name, question in questions.items():
			criteria = question['criteria']
			if name == 'operation':
				choice = 'STEP_DONE' if 'Saved' in data['state']['page']['text'] else 'CLICK'
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
	endpoint = dict(model='mock', base_url=httpserver_ipv4.url_for('/'), api_key='dummy-llm-key')
	settings = RunSettings(
		profile='debug',
		execution_mode='isolated',
		execution=ExecutionModel(**endpoint, timeout=10, max_tokens=500),
		evaluation=EvaluationModel(**endpoint, reasoning={}),
		jev=ModelEndpoint(model='local-jev', base_url=httpserver_ipv4.url_for('/').rstrip('/'), api_key='dummy-jev-key'),
		headless=True,
		user_data_dir=str(tmp_path / 'profile'),
	)
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
		assert calls[discarded['model_call_id']]['decision']['stale_reason'] == 'document_changed'
		for decision in final['decisions']:
			expected_role = 'system1' if decision['decision_source'] == 'system1' else 'planner'
			assert calls[decision['model_call_id']]['role'] == expected_role
		answer_call = next(c for c in calls.values() if c['role'] == 'answer')
		assert 'decision' not in answer_call
		roles = final['jev_usage_by_role']
		assert (roles['planner']['calls'], roles['system1']['calls'], roles['answer']['calls']) == (2, 3, 1)
		assert roles['system1']['prompt_tokens'] == 75 and final['token_usage']['jev_total_tokens'] == 81
		events = [json.loads(line) for line in Path(final['jev_trace_path']).read_text(encoding='utf-8').splitlines()]
		assert events[-1]['event'] == 'usage_summary'
		assert [e['status'] for e in events if e['event'] == 'action_result'] == ['discarded', 'executed_ok']
		assert all(secret not in json.dumps(final) for secret in ('dummy-llm-key', 'dummy-jev-key'))
	finally:
		run.close()
