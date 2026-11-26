import json
from unittest.mock import AsyncMock, Mock

import pytest

from browser_use import Agent
from browser_use.agent.jev import state
from browser_use.agent.jev.views import DelegateSubgoalParams, JevSettings, Observation, ScrollRegion, SubgoalRun
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import SerializedDOMState
from tests.ci.conftest import create_mock_llm


def viewport(text='A\nB', **changes):
	region = {'backendNodeId': 9, 'name': 'Panel', 'scrollTop': 0, 'scrollHeight': 600, 'clientHeight': 120, **changes}
	return {
		'url': 'https://example.org/',
		'title': 'Items',
		'document': 7,
		'target_id': 'tab',
		'text': text,
		'links': [],
		'scroll': [0, 0],
		'viewport': {'width': 800, 'height': 600},
		'regions': [region],
	}


def native_summary():
	return BrowserStateSummary(
		url='https://example.org/', title='Items', tabs=[], dom_state=SerializedDOMState(None, {}), screenshot='png'
	)


def test_region_size_remains_in_no_effect_fingerprint_but_diagnostics_do_not_change_alignment():
	before = viewport()
	after = {**viewport(scrollHeight=800, clientHeight=130), 'scrollDiagnostics': {'centerY': 90}}
	assert state._alignment(before) == state._alignment(after)
	assert state._alignment(before) != state._alignment({**after, 'links': [{'url': 'https://example.org/changed'}]})
	region = ScrollRegion(
		key='region:9', backend_node_id=9, target_id='tab', document=7, scroll_top=0, scroll_height=600, client_height=120
	)
	obs = Observation(
		step=1,
		url=before['url'],
		title='Items',
		scroll={'y': 0},
		visible_text='A\nB',
		links=[],
		candidates={},
		tabs=[],
		can_go_back=False,
		is_pdf=False,
		state_error=None,
		scroll_regions=[region],
		fingerprint='',
	)
	for change in ({'scroll_height': 800}, {'client_height': 130}):
		updated = obs.model_copy(update={'scroll_regions': [region.model_copy(update=change)]})
		assert state.fingerprint(obs) != state.fingerprint(updated)


@pytest.mark.parametrize(
	('field', 'values', 'reject'),
	[
		('name', ['Panel', 'Offers', 'Recent'], False),
		('scrollHeight', [600, 900, 1200], False),
		('clientHeight', [120, 140, 150], False),
		('backendNodeId', [9, 10, 11], True),
		('scrollTop', [0, 20, 40], True),
	],
)
async def test_capture_compares_region_identity_and_position(monkeypatch, field, values, reject):
	views = [viewport(**{field: v}) for v in values]
	read = AsyncMock(side_effect=views)
	monkeypatch.setattr(state, 'read_visible_text', read)
	browser = Mock()
	browser.get_browser_state_summary = AsyncMock(side_effect=[native_summary(), native_summary()])
	result = await state.capture_aligned_state(browser, include_recent_events=False)
	if reject:
		assert isinstance(result, state.UnalignedState) and result.screenshot is None
		part = 'regions.identity' if field == 'backendNodeId' else 'regions.scroll'
		assert [a['components'] for a in result.alignment_attempts] == [[part], [part]]
		assert read.await_count == 3
	else:
		assert not result.state_error and result.screenshot == 'png' and result.jev_viewport == views[1]
		assert read.await_count == 2


async def test_explicit_pr5_capture_reads_only_the_original_text_view(tmp_path, monkeypatch):
	agent = Agent(
		task='Read items',
		llm=create_mock_llm(),
		use_judge=False,
		jev=JevSettings(
			jev_api_key='test', candidate_pipeline='pr5', screenshot_policy='always', trace_path=tmp_path / 'trace.jsonl'
		),
	)
	e = agent._jev
	assert e is not None
	browser = agent.browser_session
	session = Mock()
	session.target_id = 'tab'
	legacy = {'url': 'https://example.org/', 'text': 'Items', 'links': [], 'title': 'Items', 'centerScroll': [0, 120]}
	session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={'result': {'value': legacy}})
	monkeypatch.setattr(type(browser), 'get_or_create_cdp_session', AsyncMock(return_value=session))
	monkeypatch.setattr(type(browser), 'get_browser_state_summary', AsyncMock(return_value=native_summary()))
	try:
		summary = await e.capture_browser_state(include_recent_events=False)
		assert summary.jev_viewport == {**legacy, 'target_id': 'tab'}
		reads = session.cdp_client.send.Runtime.evaluate.await_args_list
		assert len(reads) == 2 and all(
			c.kwargs['params'] == {'expression': state.VISIBLE_TEXT_SCRIPT, 'returnByValue': True} for c in reads
		)
	finally:
		await agent.eventbus.stop(clear=True, timeout=3)
		await e.aclose()


async def test_failed_capture_check_and_handback_have_the_same_two_attempts(tmp_path, monkeypatch):
	trace = tmp_path / 'trace.jsonl'
	agent = Agent(
		task='Read items',
		llm=create_mock_llm(),
		use_judge=False,
		jev=JevSettings(jev_api_key='test', screenshot_policy='always', trace_path=trace),
	)
	e = agent._jev
	assert e is not None
	e.runs.append(
		SubgoalRun(
			id='sg1',
			params=DelegateSubgoalParams(objective='Read', completion='Visible', reason='Read items'),
			identity='read',
			started_step=1,
			start_url='https://example.org/',
			start_site='example.org',
			max_actions=3,
		)
	)
	browser = agent.browser_session
	monkeypatch.setattr(type(browser), 'get_browser_state_summary', AsyncMock(side_effect=[native_summary(), native_summary()]))
	monkeypatch.setattr(type(browser), 'get_or_create_cdp_session', AsyncMock(side_effect=RuntimeError('No history in fixture')))
	monkeypatch.setattr(
		state,
		'read_visible_text',
		AsyncMock(
			side_effect=[viewport(), viewport('A\nC', name='Offers', scrollHeight=700), viewport('A\nC', backendNodeId=10)]
		),
	)
	try:
		summary = await e.capture_browser_state(include_recent_events=False)
		assert isinstance(summary, state.UnalignedState)
		await e.route(agent, summary)
		events = [json.loads(line) for line in trace.read_text('utf8').splitlines()]
		check = next(r for r in events if r['event'] == 'check' and r['name'] == 'snapshot_alignment')
		back = next(r for r in events if r['event'] == 'handback')
		assert back['reason'] == 'unsupported_page'
		assert (
			back['alignment_attempts']
			== check['alignment_attempts']
			== [
				{
					'attempt': 1,
					'components': ['text'],
					'ignored_components': ['regions.name', 'regions.size'],
					'text': {'before_changed_lines': 1, 'after_changed_lines': 1},
				},
				{
					'attempt': 2,
					'components': ['regions.identity'],
					'ignored_components': [],
					'text': {'before_changed_lines': 0, 'after_changed_lines': 0},
				},
			]
		)
	finally:
		await agent.eventbus.stop(clear=True, timeout=3)
		await e.aclose()
