import base64
import json
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from PIL import Image
from pydantic import ValidationError

from browser_use import Agent
from browser_use.agent.jev import JevChecks, JevSettings
from browser_use.agent.jev import state as jev_state
from browser_use.agent.jev.trace import JevTrace
from browser_use.agent.jev.views import ActionRecord, Candidate, DelegateSubgoalParams, Observation, SubgoalRun
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import SerializedDOMState
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import FakeJev


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


def test_screenshot_configuration():
	settings = JevSettings()
	assert (settings.screenshot_policy, settings.vision_max_side, settings.vision_jpeg_quality) == ('off', 1600, 72)
	assert JevSettings(screenshot_policy='always').screenshot_policy == 'always'
	assert JevSettings(screenshot_policy='smart').screenshot_policy == 'smart'
	with pytest.raises(ValidationError, match='requires the Jev backend'):
		JevSettings(system1='llm', system1_llm=create_mock_llm([]), screenshot_policy='always')
	with pytest.raises(ValidationError, match='requires the Jev backend'):
		JevSettings(system1='llm', system1_llm=create_mock_llm([]), screenshot_policy='smart')


async def fixture_request(httpserver, policy, tmp_path, monkeypatch):
	settings = JevSettings.model_validate(
		{
			'jev_api_key': 'test-only',
			'jev_base_url': httpserver.url_for(''),
			'trace_path': tmp_path / 'trace.jsonl',
			**({'screenshot_policy': policy} if policy is not None else {}),
		}
	)
	button = Candidate(key='7', kind='element', index=7, role='button', name='保存', rect={'x': 11, 'y': 25, 'w': 80, 'h': 30})
	obs = Observation(
		step=3,
		url='https://example.org/edit',
		title='草稿',
		scroll={'y': 30, 'viewport_height': 70},
		viewport_width=100,
		viewport_height=70,
		visible_text='草稿\n保存',
		links=[],
		candidates={'CLICK': [button]},
		tabs=[],
		can_go_back=False,
		is_pdf=False,
		state_error=None,
		fingerprint='visible',
	)
	run = SubgoalRun(
		id='sg1',
		params=DelegateSubgoalParams(objective='保存草稿', completion='已保存', reason='完成编辑', requires_action=True),
		identity='保存草稿\n已保存',
		max_actions=3,
		started_step=1,
		start_url=obs.url,
		start_site='example.org',
	)
	png = BytesIO()
	Image.new('RGB', (100, 70), 'white').save(png, format='PNG')
	summary = BrowserStateSummary(
		url=obs.url,
		title=obs.title,
		tabs=[],
		dom_state=SerializedDOMState(None, {}),
		screenshot=base64.b64encode(png.getvalue()).decode('ascii'),
	)
	monkeypatch.setattr(jev_state, 'observe', AsyncMock(return_value=obs))
	if policy == 'always':
		monkeypatch.setattr('browser_use.agent.jev.service.uuid4', lambda: UUID('00000000-0000-4000-8000-000000000001'))
		httpserver.expect_request('/v1/systemone', method='POST').respond_with_json(
			{
				'answers': {
					'operation': {
						'choice': 'CLICK',
						'confidence': 1,
						'probabilities': {'CLICK': 1, 'WAIT': 0, 'COLLECT': 0, 'UNABLE': 0},
					},
					'click_target': {'choice': '7', 'confidence': 1, 'probabilities': {'7': 1}},
				}
			}
		)
	else:
		FakeJev(httpserver, [{'operation': 'CLICK', 'target': '保存'}])
	agent = Agent(
		task='Save the 草稿',
		llm=create_mock_llm(),
		jev=settings,
		use_judge=False,
		use_vision=False,
		enable_signal_handler=False,
		file_system_path=str(tmp_path / 'agent'),
	)
	executor = agent._jev
	assert executor is not None
	executor.runs.append(run)
	try:
		await executor.route(agent, summary)
	finally:
		await agent.eventbus.stop(clear=True, timeout=3)
		await executor.aclose()
	assert len(httpserver.log) == 1
	return httpserver.log[0][0]


@pytest.mark.parametrize('policy', [None, 'off'])
async def test_off_request_matches_original_http_body(httpserver, policy, tmp_path, monkeypatch):
	request = await fixture_request(httpserver, policy, tmp_path, monkeypatch)
	# Captured from d0a9379 by running the same executor route and reading its HTTP request.
	golden = Path(__file__).with_name('fixtures') / 'jev_off_request.json.golden'
	assert request.get_data() == golden.read_bytes()
	assert request.content_type == 'application/json'
	assert 'TypeSafe-Danger-Zone-Images' not in request.headers


async def test_always_request_matches_phase15_multipart(httpserver, tmp_path, monkeypatch):
	request = await fixture_request(httpserver, 'always', tmp_path, monkeypatch)
	# Captured from the reconstructed Phase 1.5 worktree with the same UUID and observation.
	golden = Path(__file__).with_name('fixtures') / 'jev_always_request.json.golden'
	assert request.mimetype == 'multipart/form-data' and request.headers['TypeSafe-Danger-Zone-Images'] == 'true'
	assert request.form['request'].encode() == golden.read_bytes()
	with Image.open(request.files['page-screenshot']) as jpeg:
		assert (jpeg.format, jpeg.size) == ('JPEG', (100, 70))


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
	first, second = (json.loads(line) for line in raw.splitlines())
	assert first['nested'] == [{'title': '已保存'}] and first['step'] is None and first['agent_id'] == 'agent-42'
	assert datetime.fromisoformat(first['ts']).utcoffset() == timedelta(0)
	assert second['step'] == 4 and second['event'] == 'action_result'
