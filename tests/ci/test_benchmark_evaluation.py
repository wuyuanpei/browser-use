import json
from contextlib import contextmanager
from typing import Any

import pytest
from pytest_httpserver import HTTPServer

from browser_use.benchmark.evaluation import derive_hints, evaluate_frozen, freeze_context
from browser_use.benchmark.models import EvaluationModel


@pytest.mark.parametrize('first_response', ['invalid HTTP JSON', 'invalid model JSON'])
def test_evaluate_frozen_retries_invalid_json(httpserver: HTTPServer, first_response: str):
	endpoint = '/chat/completions'
	first_reply = httpserver.expect_ordered_request(endpoint, method='POST')
	if first_response == 'invalid HTTP JSON':
		first_reply.respond_with_data('invalid JSON', content_type='application/json')
	else:
		first_reply.respond_with_json({'choices': [{'message': {'content': 'invalid JSON'}, 'finish_reason': 'stop'}]})
	verdict = {'verdict': 'met', 'reason': 'The final sample contains the requested result.'}
	httpserver.expect_ordered_request(endpoint, method='POST').respond_with_json(
		{'choices': [{'message': {'content': json.dumps(verdict)}, 'finish_reason': 'stop'}]}
	)
	config = EvaluationModel(
		model='local-verifier', base_url=httpserver.url_for('/').rstrip('/'), api_key='local-key', reasoning={}
	)
	original_context = {'goal': 'Read the final sample.', 'task_started_at': '2026-11-27T00:30:36.297+08:00'}
	frozen = freeze_context(original_context)
	calls: list[dict[str, Any]] = []

	@contextmanager
	def model_call(kind: str, **metadata: Any):
		trace = {'kind': kind, **metadata}
		calls.append(trace)
		yield trace

	result = evaluate_frozen(frozen, config, model_call)

	assert result['attempts'] == 2
	assert result['evaluation_status'] == 'completed'
	assert result['verdict'] == 'met'
	assert [call['evaluation_attempt'] for call in calls] == [1, 2]
	assert calls[0]['request'] == calls[1]['request']
	request_context = json.loads(calls[0]['request']['messages'][1]['content'])
	assert request_context['derived_hints']['date_hints'][0]['date'] == '2026-11-27'
	assert request_context['derived_hints']['question_ordinals'] == []
	assert frozen['context'] == original_context


def test_derive_hints_uses_the_task_start_offset_for_dates():
	hints = derive_hints({'task_started_at': '2026-11-27T00:30:36.297+08:00'})

	assert len(hints['date_hints']) == 15
	assert hints['date_hints'][0] == {'offset_days': 0, 'date': '2026-11-27', 'weekday': 'Friday'}
	assert hints['date_hints'][2] == {'offset_days': 2, 'date': '2026-11-29', 'weekday': 'Sunday'}
	assert hints['date_hints'][14] == {'offset_days': 14, 'date': '2026-12-11', 'weekday': 'Friday'}
	assert derive_hints({})['date_hints'] == []


@pytest.mark.parametrize('option_names_vary', [False, True])
def test_derive_hints_counts_twenty_question_groups_in_sample_order(option_names_vary: bool):
	controls = {
		f'control-{number * 3 + option + 1}': {
			'source_id': 'document-1',
			'reading': {
				'name': f'{number * 3 + option}-q-{number}-abcd' if option_names_vary else f'q-{number}',
				'type': 'radio' if number < 10 else 'checkbox',
				'form_owner': None,
			},
		}
		for number in range(20)
		for option in range(3)
	}
	context = {
		'independent_evidence': {
			'control_records': controls,
			'samples': [
				{
					'sample_id': 'sample-1',
					'evidence': {
						'form_state': {
							'source_url': 'https://wj.qq.com/s2/test/' if option_names_vary else 'http://localhost/form',
							'controls': list(controls),
						},
					},
				},
			],
		},
	}

	groups = derive_hints(context)['question_ordinals']

	assert len(groups) == 1
	assert groups[0]['source_id'] == 'document-1'
	assert groups[0]['sample_id'] == 'sample-1'
	ordinals = groups[0]['by_name']
	assert len(ordinals) == 20
	assert ordinals['q-0'] == 1
	assert ordinals['q-16'] == 17
	assert ordinals['q-19'] == 20
