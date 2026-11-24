"""Browser Use 0.13.14 local Chromium probes.

A1: absolute_position includes the frame/scroll offset and is relative to the viewport.
The snapshot client rectangle's x/y are clientLeft/clientTop. Selector order follows DOM order.
A2: automatic first navigation leaves about:blank directly behind the task URL.
A3: a full action output model reads mixed histories and executes the original native actions.
A4: main-document viewport text excludes hidden/offscreen text and iframe contents.
A5: report the local snapshot median. A6: live AX selection is available on native options.
"""

import logging
import statistics
import time
from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest_httpserver import HTTPServer

from browser_use import Agent
from browser_use.agent.jev.state import read_visible_text
from browser_use.agent.views import AgentHistoryList, AgentOutput, StepMetadata
from browser_use.browser.session import BrowserSession
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import SELECT_PAGE, TALL_PAGE, VISIBILITY_PAGE, browser_eval, serve_page
from tests.ci.jev_support import jev_browser as _jev_browser_fixture  # noqa: F401
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401

logger = logging.getLogger(__name__)


async def test_viewport_geometry_and_dom_order(jev_browser: BrowserSession, httpserver: HTTPServer):
	url = serve_page(httpserver, '/tall', TALL_PAGE)
	await jev_browser.navigate_to(url)
	first = await jev_browser.get_browser_state_summary()
	top = [n for n in first.dom_state.selector_map.values() if n.tag_name == 'button']
	assert [n.ax_node.name for n in top if n.ax_node and (n.ax_node.name or '').startswith('Top')] == [
		'Top first',
		'Top second',
		'Top third',
	]
	await browser_eval(jev_browser, 'window.scrollTo(0, 1600)')
	second = await jev_browser.get_browser_state_summary()
	visible = next(n for n in second.dom_state.selector_map.values() if n.ax_node and n.ax_node.name == 'Bottom button')
	assert second.page_info and visible.absolute_position
	actual_y = await browser_eval(jev_browser, 'document.querySelector("button:last-child").getBoundingClientRect().y')
	logger.info(
		'A1: scroll=%s, top at start=%s, bottom client=%s absolute=%s viewport y=%s',
		second.page_info.scroll_y,
		[n.ax_node.name for n in top if n.ax_node],
		visible.snapshot_node.clientRects if visible.snapshot_node else None,
		visible.absolute_position,
		actual_y,
	)
	assert abs(visible.absolute_position.y - actual_y) <= 2
	assert visible.snapshot_node and visible.snapshot_node.clientRects and visible.snapshot_node.clientRects.y != actual_y


async def test_initial_navigation_history(jev_browser: BrowserSession, httpserver: HTTPServer):
	url = serve_page(httpserver, '/landing', '<p>Landing</p>')
	await jev_browser.navigate_to('about:blank', new_tab=True)
	agent = Agent(task=f'Open {url} and read it', llm=create_mock_llm(), browser_session=jev_browser)
	await agent._execute_initial_actions()
	cdp = await jev_browser.get_or_create_cdp_session()
	history = await cdp.cdp_client.send.Page.getNavigationHistory(session_id=cdp.session_id)
	assert history['entries'][history['currentIndex']]['url'] == url
	previous = history['entries'][history['currentIndex'] - 1]['url'] if history['currentIndex'] > 0 else None
	logger.info('A2: history=%s', [entry['url'] for entry in history['entries']])
	assert previous == 'about:blank'


async def test_mixed_action_history(jev_browser: BrowserSession, httpserver: HTTPServer, tmp_path: Path):
	url = serve_page(httpserver, '/counter', '<button onclick="window.clicks++">Count</button><script>window.clicks=0;</script>')
	await jev_browser.navigate_to(url)
	agent = Agent(task='Count', llm=create_mock_llm(), browser_session=jev_browser)
	summary = await jev_browser.get_browser_state_summary()
	index = next(i for i, node in summary.dom_state.selector_map.items() if node.tag_name == 'button')
	full = agent.tools.registry.create_action_model()
	narrow = agent.tools.registry.create_action_model(include_actions=['done'])
	click = full.model_validate({'click': {'index': index}})
	result = await agent.multi_act([click])
	assert result and not result[0].error and await browser_eval(jev_browser, 'window.clicks') == 1
	full_output = AgentOutput.type_with_custom_actions(full)
	narrow_output = AgentOutput.type_with_custom_actions(narrow)
	done = narrow.model_validate({'done': {'success': True, 'text': 'Counted'}})
	for step, model_output in enumerate((full_output(action=[click]), narrow_output(action=[done])), 1):
		await agent._make_history_item(
			model_output, summary, result, StepMetadata(step_number=step, step_start_time=0, step_end_time=1)
		)
	path = tmp_path / 'history.json'
	agent.history.save_to_file(path)
	history = AgentHistoryList.load_from_file(path, full_output)
	actions = history.model_actions()
	assert len(actions) == 2 and actions[0]['click']['index'] == index
	assert actions[1]['done']['text'] == 'Counted' and actions[1]['done']['success'] is True
	with pytest.raises(ValidationError):
		AgentHistoryList.load_from_file(path, narrow_output)


async def test_visible_text_and_screenshot_time(jev_browser: BrowserSession, httpserver: HTTPServer):
	url = serve_page(httpserver, '/visible', VISIBILITY_PAGE)
	await jev_browser.navigate_to(url)
	text = (await read_visible_text(jev_browser))['text']
	assert 'Visible paragraph' in text
	assert all(
		label not in text for label in ['Display hidden', 'Visibility hidden', 'Offscreen paragraph', 'Nested frame paragraph']
	)
	durations = []
	for _ in range(5):
		started = time.monotonic()
		summary = await jev_browser.get_browser_state_summary(include_screenshot=True)
		assert summary.screenshot
		durations.append(time.monotonic() - started)
	logger.info('A5: median snapshot including screenshot %.3fs', statistics.median(durations))
	assert statistics.median(durations) < 10


async def test_selected_option_in_ax(jev_browser: BrowserSession, httpserver: HTTPServer):
	url = serve_page(httpserver, '/select-ax', SELECT_PAGE)
	await jev_browser.navigate_to(url)
	summary = await jev_browser.get_browser_state_summary()
	select = next(node for node in summary.dom_state.selector_map.values() if node.attributes.get('id') == 'city')
	options = [
		n
		for child in select.children
		for n in ([child] if child.tag_name == 'option' else child.children)
		if n.tag_name == 'option'
	]
	assert len(options) == 4
	assert await browser_eval(jev_browser, 'document.querySelector("#city").selectedIndex') == 1
	states = [
		[(a.name, a.value) for a in (node.ax_node.properties or []) if a.name == 'selected'] if node.ax_node else []
		for node in options
	]
	logger.info('A6: native option AX selection properties=%s', states)
	assert states[1] == [('selected', True)]
