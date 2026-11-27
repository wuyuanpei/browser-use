"""Geometry filtering leaves reachable targets available to the existing ranking."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from browser_use import Agent
from browser_use.agent.jev import decision, pruning, state
from browser_use.agent.jev.views import Candidate, DelegateSubgoalParams, DelegateSubgoalParamsSmart, JevSettings
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import browser_eval
from tests.ci.jev_support import jev_browser as _jev_browser_fixture  # noqa: F401
from tests.ci.jev_support import jev_httpserver as _jev_httpserver_fixture  # noqa: F401
from tests.ci.test_jev_executor import _VisionJev, close_driver


@pytest.mark.parametrize('maximum', [1, 200])
async def test_under_limit_does_not_sample_browser(maximum):
	browser, summary, obs = AsyncMock(), MagicMock(), MagicMock()
	clicks = [Candidate(key='tab:1', kind='tab', role='tab', name='Existing page')]
	obs.candidates = {'CLICK': clicks}
	retained, stats = await pruning.prune_click_targets(browser, summary, obs, maximum)
	assert retained is clicks and stats == dict(
		before=1, after_pruning=1, clipped=0, occluded=0, elapsed_ms=0.0, checked=0, budget_exhausted=False, js_ms=0.0
	)
	browser.get_or_create_cdp_session.assert_not_called()


async def test_dense_filter_before_limit_keeps_unrelated_target(jev_browser, httpserver):
	fixture = Path(__file__).resolve().parents[2] / 'browser_use/benchmark/static/fixture.html'
	httpserver.expect_request('/fixture.html').respond_with_data(fixture.read_bytes(), content_type='text/html; charset=utf-8')
	await jev_browser.navigate_to(httpserver.url_for('/fixture.html') + '?scenario=dense')
	settings = JevSettings()
	summary = await state.capture_aligned_state(jev_browser, include_recent_events=False)
	obs = await state.observe(jev_browser, summary, settings, 1)
	raw = obs.candidates['CLICK']
	target = next(
		c for c in raw if c.index is not None and summary.dom_state.selector_map[c.index].attributes.get('id') == 'target'
	)
	tab = Candidate(key='tab:1', kind='tab', role='tab', name='Another tab')
	raw.append(tab)
	terms = decision._candidate_terms('Open the archive ledger')
	assert target not in decision._limit_targets(raw, settings.max_candidates, terms)
	audit = {}
	filtered, stats = await pruning.prune_click_targets(
		jev_browser, summary, obs, settings.max_candidates, budget_ms=60_000, audit=audit
	)
	assert stats['before'] == len(raw) > 200
	assert stats['clipped'] == 100 and stats['occluded'] == 220
	# The original full-scan result is Continue and the 60 live references, in DOM order.
	expected = [
		c
		for c in raw
		if c is target
		or c is tab
		or (
			c.index is not None
			and (parent := summary.dom_state.selector_map[c.index].parent) is not None
			and parent.attributes.get('id') == 'dense-references'
		)
	]
	assert len(expected) == 62 and [c.key for c in filtered] == [c.key for c in expected]
	assert stats['checked'] == 381 and not stats['budget_exhausted'] and stats['js_ms'] > 0
	assert len(raw) - len(filtered) == len(audit) == stats['clipped'] + stats['occluded']
	assert stats['after_pruning'] == len(filtered)
	assert target in decision._limit_targets(filtered, settings.max_candidates, terms)
	assert tab in filtered
	assert [c for c in raw if c.key not in audit] == filtered


async def test_pruning_verifies_shadow_identity_and_all_five_points(jev_browser, httpserver):
	page = """<main id="host"></main><script>
	const root=document.querySelector('main').attachShadow({mode:'open'});
	root.innerHTML=`<style>
	.box{position:relative;display:inline-block;width:120px;height:45px}
	button,.overlay{position:absolute;inset:0;width:100%;height:100%}
	.overlay{background:transparent}.middle{inset:30%;width:40%;height:40%}
	</style>
	<div class="box"><button id="covered">Covered</button><div class="overlay"></div></div>
	<div class="box"><button id="partial">Partial</button><div class="overlay middle"></div></div>
	<div class="box"><button id="stale">Stale</button><div class="overlay"></div></div>`;
	</script>"""
	httpserver.expect_request('/shadow').respond_with_data(page, content_type='text/html')
	await jev_browser.navigate_to(httpserver.url_for('/shadow'))
	settings = JevSettings()
	summary = await state.capture_aligned_state(jev_browser, include_recent_events=False)
	obs = await state.observe(jev_browser, summary, settings, 1)
	by_name = {c.name: c.key for c in obs.candidates['CLICK']}
	assert {'Covered', 'Partial', 'Stale'} <= by_name.keys()
	await browser_eval(
		jev_browser,
		"const old=document.querySelector('main').shadowRoot.getElementById('stale'); old.replaceWith(old.cloneNode(true))",
	)
	audit = {}
	got, _ = await pruning.prune_click_targets(jev_browser, summary, obs, maximum=1, audit=audit)
	keys = {c.key for c in got}
	assert audit[by_name['Covered']] == 'occluded'
	assert by_name['Partial'] in keys and by_name['Stale'] in keys


async def test_dense_policy_preserves_pr5_off_wire_and_prunes_visual_policies(jev_browser, httpserver, tmp_path, monkeypatch):
	fixture = Path(__file__).resolve().parents[2] / 'browser_use/benchmark/static/fixture.html'
	httpserver.expect_request('/fixture.html').respond_with_data(fixture.read_bytes(), content_type='text/html; charset=utf-8')
	await jev_browser.navigate_to(httpserver.url_for('/fixture.html') + '?scenario=dense')
	summary = await state.capture_aligned_state(jev_browser, include_recent_events=False)
	obs = await state.observe(jev_browser, summary, JevSettings(), 1)
	goal = 'Open the archive ledger'
	completion = 'Archive ledger is open'
	terms = decision._candidate_terms(f'{goal} {completion}')
	pr5_targets = decision._limit_targets(obs.candidates['CLICK'], 200, terms)
	target = next(c for c in obs.candidates['CLICK'] if c.name == 'Continue')
	lookup = Mock(wraps=pruning._lookup_hints)
	monkeypatch.setattr(pruning, '_lookup_hints', lookup)
	visual_keys = []
	cases = [
		('off', None, 300),
		('always', None, 60_000),
		('smart', None, 60_000),
		('off', 'extended', 60_000),
		('off', 'extended', 0),
		('always', 'pr5', 300),
	]
	fake = _VisionJev(httpserver, [{'operation': 'STEP_DONE'} for _ in cases])
	for index, (policy, pipeline, budget) in enumerate(cases):
		path = tmp_path / f'{index}.jsonl'
		settings = JevSettings(
			jev_api_key='local-test',
			jev_base_url=httpserver.url_for('/'),
			screenshot_policy=policy,
			candidate_pipeline=pipeline,
			pruning_budget_ms=budget,
			trace_path=path,
		)
		agent = Agent(task=goal, llm=create_mock_llm(), browser_session=jev_browser, use_judge=False, jev=settings)
		executor = agent._jev
		assert executor is not None
		try:
			params = DelegateSubgoalParams(objective=goal, completion=completion, reason='Next stage')
			if policy == 'smart':
				params = DelegateSubgoalParamsSmart(
					**params.model_dump(), screenshot_required=False, screenshot_reason='Use the text labels'
				)
			await executor.delegate(params, jev_browser)
			lookup.reset_mock()
			await executor.route(agent, summary)
			assert len(fake.requests) == index + 1 and not fake.errors
			wire = fake.requests[-1]['questions']['click_target']['criteria']
			event = next(
				e
				for line in path.read_text(encoding='utf-8').splitlines()
				if (e := json.loads(line))['event'] == 'system1_decision'
			)
			stats = event['candidate_pruning']
			if (policy == 'off' and pipeline is None) or pipeline == 'pr5' or budget == 0:
				lookup.assert_not_called()
				assert stats['budget_exhausted'] is (budget == 0) and stats['checked'] == 0
				if policy == 'always':
					assert event['vision'] and list(wire) == [c.key for c in pr5_targets]
				else:
					assert list(wire.items()) == [(c.key, c.criterion()) for c in pr5_targets]
				assert target.key not in wire and stats['before'] == stats['after_pruning'] and stats['elapsed_ms'] == 0
			else:
				lookup.assert_called()
				assert target.key in wire and stats['clipped'] > 0 and stats['occluded'] > 0
				assert event['vision'] is (policy == 'always')
				visual_keys.append(list(wire))
		finally:
			await close_driver(agent)
	assert len(visual_keys) == 3 and visual_keys[0] == visual_keys[1] == visual_keys[2]


async def test_budget_expires_mid_candidate_and_keeps_unchecked_suffix(jev_browser, httpserver, monkeypatch):
	page = """<style>.box{position:relative;display:inline-block}.mask{position:absolute;inset:0}</style>
	<span class="box"><button>Covered one</button><i class="mask"></i></span>
	<button>Reachable</button>
	<span class="box"><button>Covered two</button><i class="mask"></i></span>
	<span class="box"><button>Covered three</button><i class="mask"></i></span>"""
	httpserver.expect_request('/budget').respond_with_data(page, content_type='text/html')
	await jev_browser.navigate_to(httpserver.url_for('/budget'))
	summary = await state.capture_aligned_state(jev_browser, include_recent_events=False)
	obs = await state.observe(jev_browser, summary, JevSettings(), 1)
	buttons = [c for c in obs.candidates['CLICK'] if c.tag == 'button']
	assert [c.name for c in buttons] == ['Covered one', 'Reachable', 'Covered two', 'Covered three']
	obs.candidates['CLICK'] = buttons
	# A deterministic page-local clock reaches its deadline inside the third candidate.
	source = pruning._MEASURE.replace('function classify(el) {', 'function classify(el) { ticks++;')
	monkeypatch.setattr(
		pruning, '_MEASURE', '((...args)=>{let ticks=0;const performance={now:()=>ticks};return ' + source + '(...args)})'
	)
	filtered, stats = await pruning.prune_click_targets(jev_browser, summary, obs, 2, budget_ms=3)
	assert [c.key for c in filtered] == [c.key for c in buttons[1:]]
	assert stats['checked'] == 2 and stats['occluded'] == 1 and stats['budget_exhausted'] and stats['js_ms'] == 3
	# With the same snapshot and sufficient time, reaching the limit also stops in rank order.
	terms = decision._candidate_terms('Reachable')
	kept, stats = await pruning.prune_click_targets(jev_browser, summary, obs, 1, budget_ms=500, goal_terms=terms)
	assert stats['checked'] == 1 and not stats['budget_exhausted']
	assert [c.name for c in decision._limit_targets(kept, 1, terms)] == ['Reachable']
