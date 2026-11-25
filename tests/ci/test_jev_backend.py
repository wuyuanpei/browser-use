import asyncio
import json
import threading
import time

import pytest
from werkzeug.wrappers import Response

from browser_use.agent.jev import JevSettings
from browser_use.agent.jev.backends import JevBackend, System1Error

QUESTIONS = {'operation': {'type': 'choice', 'criteria': {'WAIT': 'Wait'}, 'instructions': 'Choose'}}
REPLY = {
	'model': 'jev-test',
	'answers': {'operation': {'choice': 'WAIT', 'probabilities': {'WAIT': 1}, 'confidence': 0.96}},
	'usage': {'input_tokens': 235, 'output_tokens': 4},
}
PATH = '/v1/systemone'
KEY = 'local-test-only'


@pytest.fixture
async def backend(httpserver):
	instance = JevBackend(JevSettings(jev_base_url=httpserver.url_for('')), KEY)
	try:
		yield instance
	finally:
		await instance.aclose()


async def test_reply_and_authenticated_request(httpserver, backend):
	state = {'original_goal': '已保存'}
	httpserver.expect_oneshot_request(
		PATH,
		method='POST',
		headers={'Authorization': f'Bearer {KEY}'},
		json={'model': 'jev-latest', 'state': state, 'questions': QUESTIONS},
	).respond_with_json(REPLY)
	result = await backend.ask(state, QUESTIONS)
	assert (result.model, result.answers, result.input_tokens, result.output_tokens, result.has_probabilities) == (
		'jev-test',
		REPLY['answers'],
		235,
		4,
		True,
	)
	assert [a['status'] for a in result.attempts] == [200]
	httpserver.expect_oneshot_request(PATH).respond_with_json({**REPLY, 'usage': {'input_tokens': 0}})
	second = await backend.ask({}, QUESTIONS)
	assert second.input_tokens == 0 and second.output_tokens is None


async def test_retry_backoff(httpserver, backend):
	for status in (429, 529):
		httpserver.expect_ordered_request(PATH).respond_with_data(KEY, status=status)
	httpserver.expect_ordered_request(PATH).respond_with_json(REPLY)
	reply = await backend.ask({}, QUESTIONS)
	assert [a['status'] for a in reply.attempts] == [429, 529, 200]
	assert reply.latency_ms >= 1490


@pytest.mark.parametrize(('statuses', 'expected'), [([503, 503, 503], 3), ([400], 1), ([302], 1)])
async def test_failed_response_has_safe_error_and_attempts(httpserver, backend, statuses, expected):
	for code in statuses:
		httpserver.expect_ordered_request(PATH).respond_with_data(KEY, status=code)
	with pytest.raises(System1Error) as exc:
		await backend.ask({}, QUESTIONS)
	assert len(httpserver.log) == expected
	assert [a['status'] for a in exc.value.attempts] == statuses
	assert KEY not in str(exc.value) and len(str(exc.value)) <= 300


async def test_timeout_retries(httpserver_ipv4):
	def slow(_):
		time.sleep(0.5)
		return Response(json.dumps(REPLY), mimetype='application/json')

	httpserver_ipv4.expect_request(PATH).respond_with_handler(slow)
	client = JevBackend(JevSettings(jev_base_url=httpserver_ipv4.url_for(''), jev_timeout=0.2), KEY)
	try:
		with pytest.raises(System1Error) as exc:
			await client.ask({}, QUESTIONS)
		assert len(exc.value.attempts) == 3
		assert all(attempt['error_type'] == 'ReadTimeout' for attempt in exc.value.attempts)
		assert KEY not in str(exc.value)
	finally:
		await client.aclose()


@pytest.mark.parametrize('raw', ['not JSON', '[]', '{"answers":{}}', '{"answers":[]}'])
async def test_invalid_envelope_is_not_retried(httpserver, backend, raw):
	httpserver.expect_oneshot_request(PATH).respond_with_data(raw)
	with pytest.raises(System1Error) as exc:
		await backend.ask({}, QUESTIONS)
	assert len(exc.value.attempts) == len(httpserver.log) == 1


async def test_invalid_requests_are_rejected_before_post(httpserver, backend):
	question = QUESTIONS['operation']
	for questions in (
		{},
		{str(i): question for i in range(33)},
		{'operation': {**question, 'criteria': {str(i): '' for i in range(256)}}},
	):
		with pytest.raises(ValueError):
			await backend.ask({}, questions)
	with pytest.raises(ValueError):
		await backend.ask({'text': 'x' * 120_000}, QUESTIONS)
	with pytest.raises(ValueError):
		await backend.ask({'probability': float('nan')}, QUESTIONS)
	assert not httpserver.log


async def test_cancelled_request_retains_actual_attempt(httpserver_ipv4):
	received, release = threading.Event(), threading.Event()

	def wait_for_cancellation(_):
		received.set()
		release.wait(12)
		return Response(json.dumps(REPLY), mimetype='application/json')

	httpserver_ipv4.expect_oneshot_request(PATH).respond_with_handler(wait_for_cancellation)
	client = JevBackend(JevSettings(jev_base_url=httpserver_ipv4.url_for('')), KEY)
	task = asyncio.create_task(client.ask({}, QUESTIONS))
	try:
		assert await asyncio.to_thread(received.wait, 10)
		task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await task
		assert len(client.last_attempts) == 1
		assert client.last_attempts[0]['error_type'] == 'CancelledError'
	finally:
		release.set()
		await client.aclose()
