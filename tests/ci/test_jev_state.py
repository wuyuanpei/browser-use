import asyncio
import base64
import io

import pytest
from PIL import Image
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from browser_use import Agent
from browser_use.agent.jev.rules import enter_effect
from browser_use.agent.jev.state import observe, selected_option
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
