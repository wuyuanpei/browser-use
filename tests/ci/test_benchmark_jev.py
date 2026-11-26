"""PR #1 workbench integration with the 0.13.14 Jev Agent, using only local model services."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from browser_use.benchmark.chrome_attachment import _OwnedClient
from browser_use.benchmark.engine import BrowserUnavailable, BrowserUseRun, _attached_tools
from browser_use.benchmark.evaluation import build_context
from browser_use.benchmark.models import (
	EvaluationModel,
	ExecutionModel,
	ModelEndpoint,
	RunSettings,
	load_profile_environment,
	runtime_settings,
)
from browser_use.benchmark.tasks import CaseProbe
from browser_use.benchmark.token_usage import aggregate_token_usage, token_usage


def test_attached_isolated_executor_only_accepts_enter():
	excluded = ['send_keys', 'evaluate', 'upload_file']
	for mode in ('direct', 'collaborative'):
		assert 'send_keys' not in _attached_tools(excluded, mode).registry.registry.actions
	actions = _attached_tools(excluded, 'isolated').registry.registry.actions
	assert not {'evaluate', 'upload_file'} & actions.keys()
	params = actions['send_keys'].param_model
	assert params.model_validate({'keys': 'Enter'}).model_dump()['keys'] == 'Enter'
	for keys in ('Control+w', 'Tab', 'Enter Enter'):
		with pytest.raises(ValidationError):
			params.model_validate({'keys': keys})


def test_workbench_records_the_planner_selected_target_without_system1_confidence(monkeypatch):
	run = BrowserUseRun.__new__(BrowserUseRun)
	run._state = {
		'decision_rounds': 0,
		'model_calls': [],
		'decisions': [],
		'page': {'fingerprint': 'page'},
		'observation_index': 1,
	}
	executor = SimpleNamespace(
		step_source=lambda: 'system1', has_reviewed_click=lambda: True, planner_reviewed_click_key=lambda: '4'
	)
	monkeypatch.setattr(
		run,
		'_native',
		SimpleNamespace(_jev=executor),
		raising=False,
	)
	run._should_cancel = None
	run._emit = lambda *_args, **_kwargs: None

	async def predict():
		answers = {
			'operation': {'choice': 'CLICK', 'probabilities': {'CLICK': 1.0}},
			'click_target': {'choice': '3', 'confidence': 0.6, 'probabilities': {'3': 0.6, '4': 0.4}},
		}
		run._state['model_calls'].extend(
			[
				{'id': 1, 'role': 'system1', 'response': {'answers': answers}, 'latency_ms': 1},
				{'id': 2, 'role': 'planner', 'status': 'success'},
			]
		)
		return {'click': {'index': 4}}, 'click', '4', {}, {}, True

	monkeypatch.setattr(run, '_predict', predict)
	monkeypatch.setattr(run, '_bridge', SimpleNamespace(run=lambda call, **_: asyncio.run(call)), raising=False)
	run._choose()
	chosen = run._state['decision']
	assert chosen['native_target_key'] == '4'
	assert chosen['target_confidence'] is None
	assert chosen['target_review_model_call_id'] == 2
	assert chosen['model_call_id'] == 1
	assert chosen['native_target_probabilities'] == {'3': 0.6, '4': 0.4}


@pytest.mark.parametrize(
	('sample_changes', 'status', 'phase', 'included'),
	[
		({}, 'completed', 'final', True),
		({'source_document': 11.5}, 'completed', 'final', False),
		({'target_id': 'other-tab'}, 'completed', 'final', False),
		({'url': 'https://example.test/other'}, 'completed', 'final', False),
		({}, 'error', 'final', False),
		({}, 'completed', 'step', False),
	],
)
def test_final_native_dom_evidence_requires_a_matching_independent_document(sample_changes, status, phase, included):
	url = 'https://example.test/feed'
	dom = 'Rendered entry that the independently sampled viewport does not display'
	state = {
		'goal': 'Summarize the rendered entries',
		'final_browser_observation': {
			'url': url,
			'source_document': 10.5,
			'target_id': 'owner-tab',
			'step': 3,
			'dom_text': dom,
		},
	}
	page = {'url': url, 'source_document': 10.5, 'target_id': 'owner-tab', 'text': 'First entry', **sample_changes}
	samples = [{'phase': phase, 'sample_status': status, 'evidence': {'sampled_page': page}}]
	context = build_context(state, samples, task_started_at=None, task_ended_at=None, execution_status='done')
	evidence = context['independent_evidence']
	if included:
		native = evidence['final_native_dom']
		assert native['independent_sample_id'] == 'sample-1' and native['step'] == 3
		reading = evidence['texts'][native['text']['text_ref']]
		assert reading['text'] == dom
		assert evidence['sources'][reading['source_id']] == {'source_url': url, 'source_document': 10.5}
	else:
		assert 'final_native_dom' not in evidence and dom not in json.dumps(context)


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
@pytest.mark.parametrize('context_transition', [False, True])
def test_workbench_waits_for_the_owned_document(monkeypatch, recover, context_transition):
	from browser_use.benchmark import engine

	browser = engine._BoundBrowser(SimpleNamespace(run=lambda call, **kwargs: asyncio.run(call)))
	monkeypatch.setattr(engine, 'DOCUMENT_READY_POLL', 0.001)
	monkeypatch.setattr(engine, 'DOCUMENT_READY_TIMEOUT', 0.3 if recover else 0.008)
	bind = AsyncMock(return_value='owned-tab')
	monkeypatch.setattr(browser, 'bind', bind)
	calls = []

	async def sample(method, params):
		calls.append(method)
		page = {'url': 'about:blank', 'text': '', 'actions': []} if recover and len(calls) == 3 else None
		if page is None and context_transition:
			raise RuntimeError({'code': -32000, 'message': 'Execution context was destroyed'})
		return {'result': {'value': page}}

	monkeypatch.setattr(browser, 'send', sample)
	if recover:
		assert browser.observe()['target_id'] == 'owned-tab'
		assert calls == ['Runtime.evaluate'] * 3
	else:
		with pytest.raises(engine.PageObservationUnavailable, match='Browser document is unavailable'):
			browser.observe()
		assert calls
	bind.assert_awaited_once()


@pytest.mark.parametrize('terminal', [False, True])
def test_post_action_observation_failure_does_not_replay_an_action(monkeypatch, terminal):
	from browser_use.benchmark import engine

	document = {'target_id': 'owned-tab', 'document_id': 10.5, 'url': 'about:blank'}
	run = object.__new__(BrowserUseRun)
	run._state = {
		'status': 'predicted',
		'decision': {
			'document_guard': document,
			'document_stable_during_capture': True,
			'target_guard': None,
			'operation': 'finish' if terminal else 'click',
			'target': '',
			'native_action': {},
		},
		'history': [],
	}
	run._pending_observation = None
	run._check_cancelled = lambda: None
	run._should_cancel = lambda: None
	monkeypatch.setattr(run, '_browser', SimpleNamespace(native=object()), raising=False)
	monkeypatch.setattr(run, '_bridge', SimpleNamespace(run=lambda call, **kwargs: asyncio.run(call)), raising=False)
	run._stale_streak = []
	run._started = 0
	emitted = []
	run._emit = lambda name, **fields: emitted.append((name, fields))
	run._act = AsyncMock(return_value=[{'is_done': terminal, 'success': True, 'extracted_content': 'Recorded'}])
	run.finish_execution = lambda: None
	run.snapshot = lambda: run._state
	run._choose = lambda: pytest.fail('An incomplete observation should not trigger another prediction or action')
	observations = []

	def observe(phase='step', reason=None):
		observations.append(phase)
		if len(observations) == 1:
			raise engine.PageObservationUnavailable('Browser document is unavailable', 'owned-tab')

	run.observe = observe
	monkeypatch.setattr(engine, 'read_document', AsyncMock(return_value=document))
	run._execute(None)
	assert run._state['status'] == ('done' if terminal else 'ready')
	assert [event for event, _ in emitted][-2:] == ['action_result', 'observation_error']
	run.command('tick')
	run._act.assert_awaited_once()
	assert observations == (['step'] if terminal else ['step', 'step'])


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


def test_workbench_probability_floors_can_be_set_to_zero(monkeypatch):
	monkeypatch.setenv('PLANNER_API_KEY', 'mock-plan-key')
	monkeypatch.setenv('TEXT_MODEL_API_KEY', 'mock-text-key')
	monkeypatch.setenv('TYPESAFE_API_KEY', 'mock-jev-key')
	operation = 'BROWSER_USE_BENCHMARK_JEV_MIN_OPERATION_PROBABILITY'
	target = 'BROWSER_USE_BENCHMARK_JEV_MIN_TARGET_PROBABILITY'
	monkeypatch.delenv(operation, raising=False)
	monkeypatch.delenv(target, raising=False)
	assert runtime_settings('jev', 'isolated').jev_probability_options() == {
		'min_operation_probability': 0.8,
		'min_target_probability': 0.8,
	}
	monkeypatch.setenv(operation, '0')
	monkeypatch.setenv(target, '0')
	assert runtime_settings('jev', 'isolated').jev_probability_options() == {
		'min_operation_probability': 0.0,
		'min_target_probability': 0.0,
	}
	monkeypatch.setenv(operation, '0.8')
	assert runtime_settings('jev', 'isolated').jev_probability_options() == {
		'min_operation_probability': 0.8,
		'min_target_probability': 0.0,
	}
	assert runtime_settings().jev_probability_options() == {}
	monkeypatch.setenv(target, '1.1')
	with pytest.raises(ValidationError, match='min_target_probability'):
		runtime_settings('jev', 'isolated')


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
		'<link rel="icon" href="data:,"><title>Save work</title>'
		"<button id=\"save\" onclick=\"document.querySelector('output').textContent='Saved'; "
		"fetch('/save',{method:'POST'})\">Save</button><button>Cancel</button><output>Ready</output>",
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
		elif handbacks := [
			message
			for message in body['messages']
			if message['role'] == 'user' and '<system1_handback>' in str(message['content'])
		]:
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
			probabilities = {k: float(k == choice) for k in criteria}
			if name == 'operation' and choice == 'CLICK':
				probabilities.update(CLICK=0.75, UNABLE=0.25)
			elif name == 'click_target':
				other = next(k for k, v in criteria.items() if '"Cancel"' in v['element'])
				probabilities.update({choice: 0.65, other: 0.35})
			answers[name] = {'choice': choice, 'confidence': probabilities[choice], 'probabilities': probabilities}
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
		min_operation_probability=0,
		min_target_probability=0,
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
		assert run._native is not None
		predict = run._native._get_next_action
		timeout = True

		async def time_out_once(summary):
			nonlocal timeout
			if timeout:
				timeout = False
				raise TimeoutError('simulated planner call timeout')
			await predict(summary)

		monkeypatch.setattr(run._native, '_get_next_action', time_out_once)
		first = run.command('tick')
		assert not timeout
		assert first['history'][-1]['operation'] == 'delegate_subgoal'
		run.record_timeout_evidence()
		partial = run.snapshot()
		assert not partial.get('result')
		assert 'Save work' in partial['partial_result']['markdown'] or 'Ready' in partial['partial_result']['markdown']
		discarded = run.command('predict')['decision']
		assert discarded['decision_source'] == 'system1'
		assert discarded['operation_probabilities']['CLICK'] == 0.75 and discarded['target_confidence'] == 0.65
		run._browser.evaluate("history.replaceState(null,'','?after=stale')")
		retry = run.command('act')
		assert retry['stale_retries'] == 1 and not saved
		run.command('tick')
		run.command('tick')
		final = run.command('tick')
		assert final['replans'] == 1
		assert final['status'] == 'done' and final['result']['markdown'] == 'The page shows Saved.' and len(saved) == 1
		assert run._browser.evaluate("document.querySelector('output').textContent") == 'Saved'
		sample = CaseProbe({}).inspect(run._browser.evaluate, run._browser.observe())
		context = build_context(
			final, [{'phase': 'final', **sample}], task_started_at=None, task_ended_at=None, execution_status='done'
		)
		evidence = context['independent_evidence']
		native = evidence['final_native_dom']
		assert 'Saved' in evidence['texts'][native['text']['text_ref']]['text']
		assert final['execution_mode'] == 'isolated' and final['system1_provider'] == 'jev'
		calls = {c['id']: c for c in final['model_calls']}
		assert calls[discarded['model_call_id']]['decision']['stale_reason'] == 'document_changed'
		for decision in final['decisions']:
			expected_role = 'system1' if decision['decision_source'] == 'system1' else 'planner'
			assert calls[decision['model_call_id']]['role'] == expected_role
		answer_call = next(c for c in calls.values() if c['role'] == 'answer')
		assert 'decision' not in answer_call
		roles = final['jev_usage_by_role']
		assert (roles['planner']['calls'], roles['system1']['calls'], roles['answer']['calls']) == (3, 4, 1)
		assert roles['system1']['prompt_tokens'] == 100 and final['token_usage']['jev_total_tokens'] == 108
		events = [json.loads(line) for line in Path(final['jev_trace_path']).read_text(encoding='utf-8').splitlines()]
		assert events[-1]['event'] == 'usage_summary'
		assert [e['status'] for e in events if e['event'] == 'action_result'] == ['discarded', 'executed_ok']
		assert all(secret not in json.dumps(final) for secret in ('dummy-llm-key', 'dummy-jev-key'))
	finally:
		run.close()
