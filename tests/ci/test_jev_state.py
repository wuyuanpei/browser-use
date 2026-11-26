import asyncio
import base64
import io
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from browser_use import Agent
from browser_use.agent.jev.rules import enter_effect
from browser_use.agent.jev.state import observe, scroll_effect, selected_option
from browser_use.agent.jev.views import JevSettings
from browser_use.agent.jev.vision import prepare_screenshot
from browser_use.browser.session import BrowserSession
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import (
	COMPOSE_PAGE,
	DIALOG_PAGE,
	FORM_PAGE,
	SCROLL_BOX_PAGE,
	SEARCH_FORM_PAGE,
	SELECT_PAGE,
	TABLE_PAGE,
	TALL_PAGE,
	VISIBILITY_PAGE,
	browser_eval,
	serve_page,
	serve_two_page_site,
)
from tests.ci.jev_support import jev_browser as _jev_browser_fixture  # noqa: F401
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401


async def observe_page(jev_browser: BrowserSession, settings: JevSettings | None = None):
	summary = await jev_browser.get_browser_state_summary(include_screenshot=False)
	return await observe(jev_browser, summary, settings or JevSettings(), 1)


@pytest.mark.parametrize('mode', ['unsupported_parameter', 'ignored_parameter', 'connection_error'])
async def test_viewport_uses_legacy_result_only_for_unsupported_deep_serialization(mode):
	from browser_use.agent.jev import state

	legacy = {'text': 'Evidence', 'centerScroll': [0, 90], 'url': 'https://example.org/', 'links': []}
	first = {
		'unsupported_parameter': RuntimeError('Invalid serializationOptions: unknown parameter'),
		'ignored_parameter': {'result': {'objectId': 'legacy-array', 'type': 'object'}},
		'connection_error': ConnectionError('lost transport'),
	}[mode]
	session = Mock()
	session.target_id = 'target'
	runtime = session.cdp_client.send.Runtime
	runtime.evaluate = AsyncMock(side_effect=[first, {'result': {'value': legacy}}])
	runtime.releaseObject = AsyncMock()
	browser = Mock()
	browser.get_or_create_cdp_session = AsyncMock(return_value=session)
	value = await state.read_visible_text(browser)
	if mode == 'connection_error':
		assert runtime.evaluate.await_count == 1 and value['text'] == ''
	else:
		assert value == {**legacy, 'target_id': 'target'} and runtime.evaluate.await_count == 2
		assert runtime.evaluate.call_args.kwargs['params'] == {'expression': state.VISIBLE_TEXT_SCRIPT, 'returnByValue': True}
	assert runtime.releaseObject.await_count == int(mode == 'ignored_parameter')


@pytest.mark.parametrize(('scenario', 'count', 'result'), [('visual-cards', 3, 'Sage House'), ('visual-icons', 4, 'Starred')])
async def test_visual_fixture_has_indistinguishable_text_candidates(jev_browser, httpserver, scenario, count, result):
	fixture_path = Path(__file__).resolve().parents[2] / 'browser_use/benchmark/static/fixture.html'
	httpserver.expect_request('/fixture.html').respond_with_data(
		fixture_path.read_bytes(), content_type='text/html; charset=utf-8'
	)
	await jev_browser.navigate_to(httpserver.url_for('/fixture.html') + '?scenario=' + scenario)
	obs = await observe_page(jev_browser)
	buttons = obs.candidates['CLICK']
	assert len(buttons) == count
	assert buttons[0].role == 'button'
	assert all((b.name, b.context, b.role) == (buttons[0].name, buttons[0].context, buttons[0].role) for b in buttons)
	before = await browser_eval(jev_browser, 'document.body.innerText')
	assert result not in before
	await browser_eval(jev_browser, 'document.querySelectorAll("main button")[1].click()')
	assert result in await browser_eval(jev_browser, 'document.body.innerText')


async def test_noncentral_scroll_panel_has_directional_candidate_and_region_fingerprint(jev_browser, httpserver):
	from browser_use.agent.jev import state

	fixture = Path(__file__).resolve().parents[2] / 'browser_use/benchmark/static/fixture.html'
	httpserver.expect_request('/scroll-fixture').respond_with_data(fixture.read_bytes(), content_type='text/html; charset=utf-8')
	await jev_browser.navigate_to(httpserver.url_for('/scroll-fixture') + '?scenario=scroll-panel')
	summary = await state.capture_aligned_state(jev_browser, include_recent_events=False)
	assert not summary.state_error and summary.jev_viewport is not None and summary.jev_viewport['centerScroll']
	first = await observe(jev_browser, summary, JevSettings(), 1)
	# An already-enriched summary must also honor the pipeline selected for this observation.
	legacy = await observe(jev_browser, summary, JevSettings(screenshot_policy='always', candidate_pipeline='pr5'), 1)
	assert not legacy.scroll_regions
	assert all(not c.key.startswith('region:') for c in legacy.candidates['SCROLL_DOWN'])
	assert any(c.name == 'Main list' and c.index is not None for c in legacy.candidates['SCROLL_DOWN'])
	extended = await observe(jev_browser, summary, JevSettings(screenshot_policy='off', candidate_pipeline='extended'), 1)
	assert [r.key for r in extended.scroll_regions] == [r.key for r in first.scroll_regions]
	panel = next(c for c in first.candidates['SCROLL_DOWN'] if c.name == 'Filters')
	assert panel.key.startswith('region:') and panel.role == 'region' and panel.kind == 'scroll' and panel.rect
	assert sum(c.name == 'Filters' for c in first.candidates['SCROLL_DOWN']) == 1
	assert not any(c.name == 'Main list' for c in first.candidates['SCROLL_DOWN'])
	assert not any(c.key == panel.key for c in first.candidates['SCROLL_UP'])
	await browser_eval(jev_browser, "const p=document.querySelector('#filter-panel'); p.scrollTop=p.scrollHeight")
	last = await observe_page(jev_browser)
	assert any(c.key == panel.key for c in last.candidates['SCROLL_UP'])
	assert not any(c.key == panel.key for c in last.candidates['SCROLL_DOWN'])
	assert last.fingerprint != first.fingerprint
	# Holding other inputs fixed isolates the region's own contribution to no_effect.
	assert state.fingerprint(first.model_copy(update={'scroll_regions': last.scroll_regions})) != first.fingerprint


async def test_form_input_and_fingerprint(jev_browser: BrowserSession, httpserver: HTTPServer):
	await jev_browser.navigate_to(serve_page(httpserver, '/form', FORM_PAGE))
	agent = Agent(task='Fill in the form', llm=create_mock_llm(), browser_session=jev_browser)
	initial = await observe_page(jev_browser)
	field = next(c for c in initial.candidates['TYPE_TEXT'] if c.name == 'Full name')
	assert field.placeholder == 'Enter name' and field.states['required'] and field.context['group'] == 'Traveler details'
	assert field.context['form'] == 'Profile'
	buttons = {c.name: c for c in initial.candidates['CLICK']}
	assert buttons['Continue'].is_submit and not buttons['Preview'].is_submit
	assert buttons['Submit'].form_key == field.form_key and field.form_key
	assert initial.fingerprint == (await observe_page(jev_browser)).fingerprint
	result = await agent.multi_act(
		[agent.ActionModel.model_validate({'input': {'index': field.index, 'text': 'Alice', 'clear': True}})]
	)
	assert result and not result[-1].error
	next_obs = await observe_page(jev_browser)
	assert next(c for c in next_obs.candidates['TYPE_TEXT'] if c.key == field.key).value == 'Alice'
	assert initial.fingerprint != next_obs.fingerprint


async def test_table_and_dialog_context(jev_browser: BrowserSession, httpserver: HTTPServer):
	await jev_browser.navigate_to(serve_page(httpserver, '/table', TABLE_PAGE))
	first = await observe_page(jev_browser)
	rows = [c for c in first.candidates['CLICK'] if c.role == 'button' and c.name == 'Edit']
	assert len(rows) == 3 and all('table_row' in c.context for c in rows), [(c.tag, c.name, c.context) for c in rows]
	assert [c.context['table_row'] for c in rows] == ['Alpha', 'Beta', 'Gamma']
	assert {c.context['table_column'] for c in rows} == {'Action'}
	await jev_browser.navigate_to(serve_page(httpserver, '/dialog', DIALOG_PAGE))
	context = {c.name: c for c in (await observe_page(jev_browser)).candidates['CLICK']}
	assert context['Apply'].context['dialog'] == 'Preferences' and context['Apply'].context['section'] == 'Notifications'
	assert context['Apply'].dialog_key == context['Cancel'].dialog_key and context['Apply'].dialog_key
	assert context['Outside'].dialog_key is None


async def test_duplicate_post_controls_include_their_own_post(jev_browser, httpserver):
	await jev_browser.navigate_to(
		serve_page(
			httpserver,
			'/same-name',
			'<article><p>First AI post by Alice</p><div><button>展开</button></div></article>'
			'<article><p>Second AI post by Bob</p><div><button>展开</button></div></article>',
		)
	)
	observed = await observe_page(jev_browser)
	contexts = [c.criterion().get('context', {}) for c in observed.candidates['CLICK'] if c.name == '展开']
	assert len(contexts) == 2
	assert 'First AI post by Alice' in str(contexts[0]) and 'Second AI post by Bob' not in str(contexts[0])
	assert 'Second AI post by Bob' in str(contexts[1]) and 'First AI post by Alice' not in str(contexts[1])


async def test_click_targets_keep_one_wrapper_per_control(jev_browser, httpserver):
	await jev_browser.navigate_to(
		serve_page(
			httpserver,
			'/category-tabs',
			'<style>section{cursor:pointer;width:100px} section div{height:28px;line-height:28px}</style>'
			'<section onclick="result.value=\'first\'"><div class="outer"><div class="inner">美妆</div></div></section>'
			'<section onclick="result.value=\'second\'"><div class="outer"><div class="inner">美妆</div></div></section>'
			'<output id="result"></output>',
		)
	)
	obs = await observe_page(jev_browser)
	tabs = [c for c in obs.candidates['CLICK'] if c.role == 'div' and c.name == '美妆']
	assert len(tabs) == 2
	agent = Agent(task='Open the other category', llm=create_mock_llm(), browser_session=jev_browser)
	result = await agent.multi_act([agent.ActionModel.model_validate({'click': {'index': tabs[1].index}})])
	assert result and not result[-1].error
	assert await browser_eval(jev_browser, 'document.querySelector("output").value') == 'second'


async def test_clickable_list_item_owns_its_placeholder_link(jev_browser, httpserver):
	await jev_browser.navigate_to(
		serve_page(
			httpserver,
			'/folders',
			'<style>li{width:160px} li a{display:block;width:140px;font-size:16px}</style>'
			'<ul><li onclick="out.value=\'opened\'"><a href="#">已发送</a></li>'
			'<li><a href="/archive">已发送</a></li></ul><output id="out"></output>',
		)
	)
	obs = await observe_page(jev_browser)
	sent = [c for c in obs.candidates['CLICK'] if c.name == '已发送']
	assert [(c.role, None if not c.href else c.href.rsplit('/', 1)[-1]) for c in sent] == [
		('listitem', None),
		('link', 'archive'),
	]
	agent = Agent(task='Open sent messages', llm=create_mock_llm(), browser_session=jev_browser)
	result = await agent.multi_act([agent.ActionModel.model_validate({'click': {'index': sent[0].index}})])
	assert result and not result[-1].error
	assert await browser_eval(jev_browser, 'out.value') == 'opened'


async def test_options_match_values_and_actual_selection(jev_browser: BrowserSession, httpserver: HTTPServer):
	await jev_browser.navigate_to(serve_page(httpserver, '/select', SELECT_PAGE))
	summary = await jev_browser.get_browser_state_summary(include_screenshot=False)
	page = await observe(jev_browser, summary, JevSettings(), 1)
	caps = page.candidates['SELECT']
	london = next(c for c in caps if c.name == 'London')
	assert london.states['selected'] and london.key == f'{london.index}:2'
	assert all(c.name != 'Closed' for c in caps)
	duplicates = [c for c in caps if c.name == 'Other']
	assert [c.match_text for c in duplicates] == ['1', '2']
	assert london.index is not None
	node = summary.dom_state.selector_map[london.index]
	assert await selected_option(jev_browser, node) == (2, 'London')
	agent = Agent(task='Select the second Other', llm=create_mock_llm(), browser_session=jev_browser)
	second = duplicates[1]
	result = await agent.multi_act(
		[agent.ActionModel.model_validate({'select_dropdown': {'index': second.index, 'text': second.match_text}})]
	)
	assert result and not result[-1].error
	assert await browser_eval(jev_browser, '[duplicate.selectedIndex,duplicate.value]') == [1, '2']
	after = await observe_page(jev_browser)
	assert [c.states['selected'] for c in after.candidates['SELECT'] if c.name == 'Other'] == [False, True]


async def test_viewport_and_scroll_targets(jev_browser: BrowserSession, httpserver: HTTPServer):
	await jev_browser.navigate_to(serve_page(httpserver, '/tall', TALL_PAGE))
	before = await observe_page(jev_browser)
	assert 'Bottom button' not in {c.name for c in before.candidates['CLICK']}
	assert 'page' in {c.key for c in before.candidates['SCROLL_DOWN']}
	assert 'page' not in {c.key for c in before.candidates['SCROLL_UP']}
	await browser_eval(jev_browser, 'window.scrollTo(0,1600)')
	after = await observe_page(jev_browser)
	assert 'Bottom button' in {c.name for c in after.candidates['CLICK']}
	assert 'Top first' not in {c.name for c in after.candidates['CLICK']}
	assert before.fingerprint != after.fingerprint
	await jev_browser.navigate_to(serve_page(httpserver, '/box', SCROLL_BOX_PAGE))
	box = await observe_page(jev_browser)
	for key in ['SCROLL_DOWN', 'SCROLL_UP']:
		assert 'Results' in {c.name for c in box.candidates[key]}


async def test_jev_exposes_scroll_for_viewport_center_pane(jev_browser, httpserver):
	from browser_use import Tools

	await jev_browser.navigate_to(
		serve_page(
			httpserver,
			'/scroll-pane',
			'<style>body{margin:0;overflow:hidden}</style>'
			'<main id="pane" style="height:100vh;overflow:auto"><div style="height:200vh">Pane</div></main>',
		)
	)
	obs = await observe_page(jev_browser)
	page = next(choice for choice in obs.candidates['SCROLL_DOWN'] if choice.key == 'page')
	scroll = await Tools().scroll(down=True, pages=0.8, browser_session=jev_browser)
	assert not scroll.error
	actual = await browser_eval(jev_browser, 'pane.scrollTop')
	assert actual > 0
	effect = scroll_effect(obs, await observe_page(jev_browser), page)
	assert effect.page_changed is True and effect.measurable and effect.root_delta_y == 0 and effect.center_delta_y == actual


async def test_native_scroll_inside_center_iframe_has_unknown_effect(jev_browser, httpserver):
	from browser_use import Tools

	await jev_browser.navigate_to(
		serve_page(
			httpserver,
			'/scroll-frame',
			'<main style="height:220vh"><iframe id="pane" style="width:85vw;height:75vh" '
			'srcdoc="<main style=height:200vh>Inner scroll document</main>"></iframe></main>',
		)
	)
	before = await observe_page(jev_browser)
	page = next(choice for choice in before.candidates['SCROLL_DOWN'] if choice.key == 'page')
	scroll = await Tools().scroll(down=True, pages=0.8, browser_session=jev_browser)
	assert not scroll.error
	assert await browser_eval(jev_browser, 'pane.contentWindow.scrollY > 0 && scrollY === 0')
	effect = scroll_effect(before, await observe_page(jev_browser), page)
	assert effect.root_delta_y == 0 and effect.center_delta_y is None
	assert effect.page_changed is None and not effect.measurable


async def test_off_preserves_indexed_center_when_page_has_no_new_region(jev_browser, httpserver):
	from browser_use.agent.jev import state

	url = serve_page(
		httpserver,
		'/indexed-center',
		'<style>body{margin:0;overflow:hidden}</style>'
		'<main tabindex="0" aria-label="Central items" style="height:100vh;overflow:auto"><div style="height:200vh">Item</div></main>',
	)
	await jev_browser.navigate_to(url)
	summary = await state.capture_aligned_state(jev_browser, include_recent_events=False)
	assert summary.jev_viewport
	center = summary.jev_viewport['centerBackendNodeId']
	index = next(i for i, n in summary.dom_state.selector_map.items() if n.backend_node_id == center)
	off = await observe(jev_browser, summary, JevSettings(), 1)
	assert not off.scroll_regions
	assert [c.key for c in off.candidates['SCROLL_DOWN']] == ['page', str(index)]
	assert [c.key for c in off.candidates['SCROLL_UP']] == [str(index)]
	smart = await observe(jev_browser, summary, JevSettings(screenshot_policy='smart'), 1)
	assert [c.key for c in smart.candidates['SCROLL_DOWN']] == ['page']


async def test_core_scroll_reports_failure_when_no_scroll_was_executed(jev_browser, monkeypatch):
	from browser_use import Tools
	from browser_use.browser.events import ScrollEvent

	bus_type = type(jev_browser.event_bus)
	dispatch = bus_type.dispatch

	def fail_scroll(self, event, *args, **kwargs):
		if isinstance(event, ScrollEvent):
			raise RuntimeError('simulated scroll dispatch failure')
		return dispatch(self, event, *args, **kwargs)

	monkeypatch.setattr(bus_type, 'dispatch', fail_scroll)
	scroll = await Tools().scroll(down=True, pages=1.0, browser_session=jev_browser)
	assert scroll.error == 'Failed to execute scroll action.'
	assert not scroll.long_term_memory


@pytest.mark.parametrize('dpr', [1, 1.5])
async def test_screenshot_rects_after_scrolling_page_and_iframe(jev_browser: BrowserSession, httpserver: HTTPServer, dpr: float):
	url = serve_page(
		httpserver,
		'/image-rects',
		"""<style>
		body {margin:0; height:2200px}
		#page-button {position:absolute; left:-8px; top:690px; width:96px; height:60px; background:#fc4422; border:0}
		iframe {position:absolute; left:330px; top:850px; width:240px; height:140px; border:6px solid purple}
		</style>
		<button id="page-button" aria-label="Page target"></button>
		<iframe srcdoc="<style>body{margin:0;height:800px}
		button{position:absolute;left:31px;top:200px;width:80px;height:36px;background:#0cb45a;border:0}
		</style><button aria-label='Frame target'></button>"></iframe>""",
	)
	await jev_browser.navigate_to(url)
	cdp = await jev_browser.get_or_create_cdp_session()
	original = await browser_eval(jev_browser, '[innerWidth, innerHeight, devicePixelRatio]')

	async def set_metrics(width: int, height: int, scale: float):
		await cdp.cdp_client.send.Emulation.setDeviceMetricsOverride(
			params={'width': width, 'height': height, 'deviceScaleFactor': scale, 'mobile': False}, session_id=cdp.session_id
		)

	try:
		await set_metrics(1280, 720, dpr)
		await browser_eval(
			jev_browser,
			'scrollTo(0,700); frames[0].scrollTo(0,164); new Promise(ok=>requestAnimationFrame(()=>requestAnimationFrame(()=>ok(true))))',
		)
		summary = await jev_browser.get_browser_state_summary(include_screenshot=True)
		obs = await observe(jev_browser, summary, JevSettings(screenshot_policy='always'), 1)
		assert (obs.viewport_width, obs.viewport_height) == (1280, 720)
		image = prepare_screenshot(summary.screenshot, obs.viewport_width, obs.viewport_height, max_side=1600, quality=72)
		assert image is not None and image.scale == dpr
		# Independent DOM measurement includes the frame's content origin and both scroll offsets.
		expected = await browser_eval(
			jev_browser,
			"""(() => {
				const frame = document.querySelector('iframe'), fr = frame.getBoundingClientRect();
				const page = document.querySelector('button').getBoundingClientRect();
				const child = frames[0].document.querySelector('button').getBoundingClientRect();
				return {'Page target': {x:0, y:0, w:page.right, h:page.bottom},
					'Frame target': {x:fr.left+frame.clientLeft+child.left, y:fr.top+frame.clientTop+child.top,
						w:child.width, h:child.height}};
			})()""",
		)
		assert expected['Page target'] == {'x': 0, 'y': 0, 'w': 88, 'h': 50}
		assert expected['Frame target'] == {'x': 367, 'y': 192, 'w': 80, 'h': 36}
		colors = {'Page target': (252, 68, 34), 'Frame target': (12, 180, 90)}
		candidates = {c.name: c for c in obs.candidates['CLICK']}
		with Image.open(io.BytesIO(base64.b64decode(image.jpeg_b64))) as jpeg:
			for name, color in colors.items():
				rect = candidates[name].rect
				assert rect == expected[name], (name, rect, expected[name])
				assert rect is not None
				x, y, w, h = (rect[key] for key in ('x', 'y', 'w', 'h'))
				for point in ((x + 5, y + 5), (x + w - 6, y + h - 6)):
					pixel = jpeg.getpixel(point)
					assert isinstance(pixel, tuple)
					assert max(abs(a - b) for a, b in zip(pixel, color, strict=True)) <= 8
	finally:
		await set_metrics(*original)


async def test_visibility_and_password(jev_browser: BrowserSession, httpserver: HTTPServer):
	await jev_browser.navigate_to(serve_page(httpserver, '/visibility', VISIBILITY_PAGE))
	obs = await observe_page(jev_browser)
	assert 'Visible paragraph' in obs.visible_text
	assert 'Visibility hidden' not in obs.visible_text
	assert all(c.name != 'Disabled button' for c in obs.candidates['CLICK'])
	assert all(c.input_type != 'password' for c in obs.candidates['TYPE_TEXT'])


async def test_enter_effect_from_form_context(jev_browser: BrowserSession, httpserver: HTTPServer):
	await jev_browser.navigate_to(serve_page(httpserver, '/compose', COMPOSE_PAGE))
	obs = await observe_page(jev_browser)
	body = next(c for c in obs.candidates['TYPE_TEXT'] if c.name == 'Body')
	effect = enter_effect(body, obs.candidates['CLICK'])
	assert effect and effect[0] == 'send'
	without = await observe_page(jev_browser, JevSettings.model_validate({'checks': 'off'}))
	assert not any(c.risky_effect for c in without.candidates['CLICK'])
	await jev_browser.navigate_to(serve_page(httpserver, '/search', SEARCH_FORM_PAGE))
	search = await observe_page(jev_browser)
	assert enter_effect(search.candidates['TYPE_TEXT'][0], search.candidates['CLICK']) is None


async def test_navigation_history_and_other_tab_candidate(jev_browser: BrowserSession, httpserver: HTTPServer):
	site = serve_two_page_site(httpserver)
	await jev_browser.navigate_to('about:blank', new_tab=True)
	await jev_browser.navigate_to(site.first)
	assert not (await observe_page(jev_browser)).can_go_back
	await jev_browser.navigate_to(site.second)
	assert (await observe_page(jev_browser)).can_go_back
	previous = jev_browser.agent_focus_target_id
	assert previous
	await jev_browser.navigate_to(site.first, new_tab=True)
	current = await observe_page(jev_browser)
	tab = next(c for c in current.candidates['CLICK'] if c.kind == 'tab' and c.tab_id == previous[-4:])
	assert tab.href == site.second
	assert tab.name == next(t['title'] or t['url'] for t in current.tabs if t['tab_id'] == previous[-4:])


async def test_delayed_popup_is_a_switchable_candidate(jev_browser: BrowserSession, httpserver: HTTPServer):
	"""Release the popup after the native click returns, then exercise the new tab candidate."""
	loop = asyncio.get_running_loop()
	ready = asyncio.Event()

	def loaded(request: Request) -> Response:
		loop.call_soon_threadsafe(ready.set)
		return Response(status=204)

	httpserver.expect_request('/popup-ready', method='GET').respond_with_handler(loaded)
	detail = serve_page(
		httpserver,
		'/delayed-result',
		'<p>Delayed detail</p><script>addEventListener("load",()=>fetch("/popup-ready"))</script>',
	)
	listing = serve_page(
		httpserver,
		'/delayed-list',
		"""<button onclick="setTimeout(()=>{window.popupReady=true; if(window.releasePopup) openResult();},500)">Open later</button>
<script>function openResult(){window.open('/delayed-result','_blank');}</script>""",
	)
	await jev_browser.navigate_to(listing)
	agent = Agent(task='Open later', llm=create_mock_llm(), browser_session=jev_browser)
	summary = await jev_browser.get_browser_state_summary(include_screenshot=False)
	index = next(i for i, node in summary.dom_state.selector_map.items() if node.tag_name == 'button')
	outcome = await agent.multi_act([agent.ActionModel.model_validate({'click': {'index': index}})])
	assert outcome and not outcome[-1].error
	await browser_eval(jev_browser, 'window.releasePopup=true; if(window.popupReady) openResult(); true')
	await asyncio.wait_for(ready.wait(), timeout=30)
	observation = await observe_page(jev_browser)
	assert observation.url == listing
	candidate = next(c for c in observation.candidates['CLICK'] if c.kind == 'tab' and c.href == detail)
	switched = await agent.multi_act([agent.ActionModel.model_validate({'switch': {'tab_id': candidate.tab_id}})])
	assert switched and not switched[-1].error
	assert await browser_eval(jev_browser, 'document.body.innerText') == 'Delayed detail'
