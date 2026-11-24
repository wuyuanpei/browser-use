import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from browser_use.agent.jev import JevChecks, JevSettings
from browser_use.agent.jev.trace import JevTrace
from browser_use.agent.jev.views import ActionRecord, DelegateSubgoalParams, SubgoalRun


def test_configuration():
	assert JevSettings.model_validate({'checks': 'record'}).checks.mode('scope') == 'record'
	checks = JevChecks(default='record', confidence='enforce')
	assert [checks.mode(k) for k in ('risky_action', 'submit', 'completion', 'confidence', 'scope')] == [
		'record',
		'record',
		'record',
		'enforce',
		'record',
	]
	assert JevSettings(jev_base_url='http://localhost:8080/base/').jev_base_url == 'http://localhost:8080/base'
	assert JevSettings.model_validate({'mode': 'off', 'jev_base_url': 'ftp://disabled', 'max_candidates': -1}).mode == 'off'


@pytest.mark.parametrize(
	'config',
	[
		{'system1': 'llm'},
		{'system1_llm_two_stage': True},
		{'jev_base_url': 'ftp://x'},
		{'default_max_actions': 31},
		{'max_candidates': 256},
		{'random_field': 'invalid'},
	],
)
def test_invalid_config(config):
	with pytest.raises(ValidationError):
		JevSettings.model_validate(config)


def test_actions_count_execution_attempts():
	run = SubgoalRun(
		id='sg1',
		params=DelegateSubgoalParams(objective='click', completion='saved', reason='test'),
		identity='click\nsaved',
		max_actions=15,
		started_step=1,
		start_url='https://example.org',
		start_site='example.org',
	)
	for n, status in enumerate(('predicted', 'dispatched', 'executed_ok', 'executed_failed', 'discarded', 'unknown'), 1):
		run.actions.append(
			ActionRecord.model_validate(
				dict(
					n=n,
					step=n,
					run_id=run.id,
					operation='CLICK',
					bu_action={'click': {'index': 2}},
					url_before=run.start_url,
					fingerprint_before='a',
					status=status,
				)
			)
		)
	assert [a.n for a in run.executed_actions] == [3, 4, 6]


def test_trace_redacts_credentials_and_keeps_unicode(tmp_path: Path):
	secret = 'test-secret-value'
	settings = JevSettings(jev_api_key=secret)
	assert secret not in repr(settings) and secret not in str(settings.model_dump())
	trace = JevTrace(tmp_path / 'new' / 'jev.jsonl', 'agent-42')
	trace.write('run_config', api_key=secret, nested=[{'Authorization': secret, 'JEV_API_KEY': secret, 'title': '已保存'}])
	trace.write('action_result', step=4, status='executed_ok')
	raw = trace.path.read_text(encoding='utf-8')
	assert secret not in raw and '已保存' in raw
	first, second = [json.loads(line) for line in raw.splitlines()]
	assert first['nested'] == [{'title': '已保存'}] and first['step'] is None and first['agent_id'] == 'agent-42'
	assert datetime.fromisoformat(first['ts']).utcoffset() == timedelta(0)
	assert second['step'] == 4 and second['event'] == 'action_result'
