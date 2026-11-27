"""Regression tests for the benchmark attempt archive."""

import json
import statistics
import threading
import time

import pytest

from browser_use.benchmark import archive
from browser_use.benchmark.engine import BrowserUseRun
from browser_use.benchmark.token_usage import TOKEN_FIELDS, token_usage


def test_terminal_publication_writes_even_inside_the_throttle_window():
	memory = []
	disk = []
	publisher = archive.AttemptPublisher(memory.append, disk.append)
	for i in range(100):
		publisher.publish({'event_seq': i})

	assert 1 <= publisher.memory_publishes <= 2
	assert 1 <= publisher.disk_writes <= 2
	memory_count, disk_count = publisher.memory_publishes, publisher.disk_writes
	publisher.publish({'event_seq': 100}, force=True)
	assert publisher.memory_publishes == memory_count + 1
	assert publisher.disk_writes == disk_count + 1
	assert memory[-1] == disk[-1] == {'event_seq': 100}


def _run_with_publication(on_event):
	run = BrowserUseRun.__new__(BrowserUseRun)
	run._secrets = ('wp1-test-credential', '', None)
	run._on_event = on_event
	run._started = time.monotonic()
	run._model_call_lock = threading.RLock()
	run._planner_parse_feedback = None
	run._model_token_usage = dict.fromkeys((*TOKEN_FIELDS, 'missing_usage_calls'), 0)
	run._native = None
	run._chrome = None
	run._should_cancel = None
	run._state = {
		'model_calls': [],
		'page': {},
		'event_seq': 0,
		'observation_index': 0,
		'status': 'done',
		'screenshot_policy': 'off',
	}
	return run


def test_completed_calls_preserve_snapshots_and_do_not_archive_credentials(tmp_path):
	events_path = tmp_path / 'events.jsonl'
	trace_path = tmp_path / 'trace.json'
	secret = 'wp1-test-credential'
	memory = []
	publisher = archive.AttemptPublisher(memory.append, lambda snapshot: archive.atomic_json(trace_path, snapshot))

	def on_event(event):
		with events_path.open('a', encoding='utf-8') as stream:
			stream.write(json.dumps({'event': event.name, **event.payload}) + '\n')
		publisher.publish(event.snapshot)

	run = _run_with_publication(on_event)
	with run.model_call('planner', request={'headers': {'Authorization': secret}}) as call:
		pending = run.command('tick')
		assert pending['model_calls'][0]['status'] == 'pending'
		assert pending['token_usage']['missing_usage_calls'] == 1
		call['response'] = {'content': secret, 'usage': {'input_tokens': 17, 'output_tokens': 3}}
	first = run.command('tick')
	run._update_model_call(call['id'], decision={'stale_reason': 'document_changed'})
	with pytest.raises(ValueError, match=secret):
		with run.model_call('planner', request={'text': secret}):
			raise ValueError(secret)
	final = run.command('tick')
	publisher.publish(final, force=True)

	assert 'decision' not in first['model_calls'][0]
	assert final['model_calls'][0]['decision']['stale_reason'] == 'document_changed'
	assert pending['model_calls'][0]['status'] == 'pending'
	assert final['token_usage'] == token_usage(final['model_calls'])
	assert final['token_usage']['planner_total_tokens'] == 20
	assert final['token_usage']['missing_usage_calls'] == 1
	assert [item['status'] for item in final['model_calls']] == ['success', 'error']
	for archived in (json.dumps(final), events_path.read_text(encoding='utf-8'), trace_path.read_text(encoding='utf-8')):
		assert secret not in archived and '[REDACTED]' in archived


def test_nonterminal_event_cost_does_not_scale_with_large_finished_requests(tmp_path):
	def median_event_cost(number_of_calls):
		def write(snapshot):
			archive.atomic_json(tmp_path / 'trace.json', snapshot)

		publisher = archive.AttemptPublisher(lambda _: None, write)
		run = _run_with_publication(None)
		for index in range(number_of_calls):
			with run.model_call('jev', request={'messages': [{'text': str(index) + 'x' * 40_000}]}):
				pass
		run._on_event = lambda event: publisher.publish(event.snapshot)
		run._emit('observation')
		elapsed = []
		for _ in range(30):
			started = time.perf_counter()
			run._emit('nonterminal_event')
			elapsed.append(time.perf_counter() - started)
		return statistics.median(elapsed)

	small = median_event_cost(10)
	large = median_event_cost(100)
	assert large <= small * 3
	assert large < 0.05


def test_compact_live_archive_preserves_observations(tmp_path):
	path = tmp_path / 'evaluation.json'
	sample = {'observations': [{'page': {'text': '示例', 'actions': [1, 2]}}]}
	archive.atomic_json(path, sample, indent=None)

	recorded = path.read_text(encoding='utf-8')
	assert recorded.count('\n') == 0
	assert json.loads(recorded) == sample
