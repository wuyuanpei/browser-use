"""Exercise smart screenshots through delegation, routing, and real HTTP serialization."""

import base64
import json
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import _multipart
from PIL import Image
from werkzeug.wrappers import Request, Response

from browser_use import Agent
from browser_use.agent.jev import JevChecks, JevSettings, state
from browser_use.agent.jev.service import JevExecutor
from browser_use.agent.jev.views import Candidate, Observation
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import SerializedDOMState
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import FakeJev
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401

GOLDEN = Path(__file__).with_name('fixtures') / 'jev_smart_l1_wire.golden'
IMAGE_IDS = [UUID('00000000-0000-4000-8000-000000000001'), UUID('00000000-0000-4000-8000-000000000002')]
VISUAL_REASON = 'Identify the Save button on the colored row.'
SAVE = {'operation': 'CLICK', 'target': '保存'}
PREVIEW = {'operation': 'CLICK', 'target': '预览'}


class _RecordingJev(FakeJev):
	def _handle(self, request: Request) -> Response:
		request.get_data()  # Keep the original bytes available after parsing multipart parts.
		if request.mimetype == 'multipart/form-data':
			request = Request.from_values(method='POST', json=json.loads(request.form['request']))
		return super()._handle(request)


@asynccontextmanager
async def _delegated(server, tmp_path, monkeypatch, declared, *, checks=None, changed_step=None):
	obs = Observation(
		step=1,
		url='https://example.org/edit',
		title='草稿',
		scroll={'y': 30, 'viewport_height': 70},
		viewport_width=100,
		viewport_height=70,
		visible_text='草稿\n保存\n预览',
		links=[],
		candidates={
			'CLICK': [
				Candidate(
					key='7', kind='element', index=7, role='button', name='保存', rect={'x': 11, 'y': 25, 'w': 40, 'h': 30}
				),
				Candidate(
					key='8', kind='element', index=8, role='button', name='预览', rect={'x': 55, 'y': 25, 'w': 40, 'h': 30}
				),
			]
		},
		tabs=[],
		can_go_back=False,
		is_pdf=False,
		state_error=None,
		fingerprint='draft',
	)
	png = BytesIO()
	Image.new('RGB', (100, 70), 'white').save(png, format='PNG')
	summary = BrowserStateSummary(
		url=obs.url,
		title=obs.title,
		tabs=[],
		dom_state=SerializedDOMState(None, {}),
		screenshot=base64.b64encode(png.getvalue()).decode('ascii'),
		jev_viewport={'url': obs.url, 'text': obs.visible_text},
	)

	async def observe(_browser, _summary, _settings, step):
		return obs.model_copy(update={'step': step, 'fingerprint': 'saved' if step == changed_step else obs.fingerprint})

	monkeypatch.setattr(state, 'observe', observe)
	monkeypatch.setattr(BrowserSession, 'get_current_page_url', AsyncMock(return_value=obs.url))
	monkeypatch.setattr(BrowserSession, 'cdp_client', property(lambda _session: None))
	agent = Agent(
		task='Save the 草稿',
		llm=create_mock_llm(),
		jev=JevSettings(
			jev_api_key='test-only',
			jev_base_url=server.url_for(''),
			trace_path=tmp_path / 'trace.jsonl',
			screenshot_policy='smart',
			checks=checks or JevChecks(),
		),
		use_judge=False,
		use_vision=False,
		enable_signal_handler=False,
		file_system_path=str(tmp_path / 'agent'),
	)
	executor = agent._jev
	assert executor is not None
	params = {'objective': '保存草稿', 'completion': '已保存', 'reason': '完成编辑', 'requires_action': True}
	if declared is not None:
		params.update(screenshot_required=declared, screenshot_reason=VISUAL_REASON if declared else 'Text labels are enough.')
	try:
		result = await agent.tools.registry.execute_action('delegate_subgoal', params, browser_session=agent.browser_session)
		assert not result.error
		yield agent, executor, summary
	finally:
		await agent.eventbus.stop(clear=True, timeout=3)
		await executor.aclose()


async def _route_and_complete_click(agent: Agent, executor: JevExecutor, summary: BrowserStateSummary):
	result = await executor.route(agent, summary)
	assert result.model_output is not None
	assert 'click' in result.model_output.action[0].model_dump(exclude_unset=True)
	# Supply a successful browser action result while routing and the next observation remain separate.
	executor.on_dispatch(agent)
	executor.on_executed(agent, [ActionResult()])
	agent.state.n_steps += 1


def _events(tmp_path):
	return [json.loads(line) for line in (tmp_path / 'trace.jsonl').read_text(encoding='utf-8').splitlines()]


async def test_smart_planner_true_matches_l1_wire_golden(httpserver, tmp_path, monkeypatch):
	ids = iter(IMAGE_IDS)
	monkeypatch.setattr('browser_use.agent.jev.service.uuid4', lambda: next(ids))
	monkeypatch.setattr(_multipart, 'os', SimpleNamespace(urandom=lambda length: bytes(range(length))))
	fake = _RecordingJev(httpserver, [SAVE, {'operation': 'STEP_DONE'}])
	async with _delegated(httpserver, tmp_path, monkeypatch, True) as (agent, executor, summary):
		await _route_and_complete_click(agent, executor, summary)
		assert (await executor.route(agent, summary)).model_output is None
		assert executor.runs[-1].handback is not None and executor.runs[-1].handback.reason == 'step_done'
	assert not fake.errors and not fake.steps
	golden = json.loads(GOLDEN.read_bytes())
	wire_requests = [request for request, _ in httpserver.log]
	assert len(wire_requests) == 2
	for request, packet, expected, screenshot_id in zip(wire_requests, fake.requests, golden['requests'], IMAGE_IDS, strict=True):
		assert request.method == 'POST' and request.path == '/v1/systemone'
		assert {name: request.headers[name] for name in expected['headers']} == expected['headers']
		assert request.get_data() == base64.b64decode(expected['body_base64'])
		assert list(request.form) == ['request'] and list(request.files) == ['page-screenshot']
		assert request.files['page-screenshot'].mimetype == 'image/jpeg'
		assert packet['state']['page']['screenshot'] == str(screenshot_id)
		assert packet['state']['current_subgoal']['screenshot_guidance'] == {
			'required': True,
			'reason': VISUAL_REASON,
			'source': 'planner',
		}
	assert [row['vision_trigger'] for row in _events(tmp_path) if row['event'] == 'system1_decision'] == ['planner', 'planner']


@pytest.mark.parametrize('declared', [False, None], ids=['declared-false', 'omitted'])
@pytest.mark.parametrize(
	('trigger', 'steps', 'completed', 'expected_requests'),
	[
		('low_confidence', [{**SAVE, 'p': 0.62}, SAVE], 0, 1),
		('unable', [{'operation': 'UNABLE'}, SAVE], 0, 1),
		('no_effect', [SAVE, PREVIEW, SAVE], 2, 2),
		('repeated_action', [SAVE, SAVE, PREVIEW, SAVE, PREVIEW], 3, 4),
	],
)
async def test_smart_runtime_triggers_stay_text_only(
	httpserver, tmp_path, monkeypatch, declared, trigger, steps, completed, expected_requests
):
	fake = _RecordingJev(httpserver, steps)
	checks = JevChecks(confidence='off') if trigger == 'unable' else None
	# In the repetition case, Preview returns to the page on which Save has already been tried twice.
	changed_step = 3 if trigger == 'repeated_action' else None
	async with _delegated(httpserver, tmp_path, monkeypatch, declared, checks=checks, changed_step=changed_step) as (
		agent,
		executor,
		summary,
	):
		for _ in range(completed):
			await _route_and_complete_click(agent, executor, summary)
		result = await executor.route(agent, summary)
		handback = executor.runs[-1].handback
	assert not fake.errors and len(fake.requests) == expected_requests
	assert all(
		req.content_type == 'application/json' and 'TypeSafe-Danger-Zone-Images' not in req.headers for req, _ in httpserver.log
	)
	assert all('assets' not in packet and 'screenshot' not in packet['state']['page'] for packet in fake.requests)
	assert result.model_output is None and handback is not None and handback.reason == trigger
	rows = _events(tmp_path)
	assert all(row['event'] != 'vision_escalation' and 'vision_escalation' not in row for row in rows)
	assert all(row.get('vision_trigger') is None for row in rows)
