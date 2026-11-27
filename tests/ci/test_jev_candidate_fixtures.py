from pathlib import Path
from types import SimpleNamespace

from browser_use.agent.jev.state import observe
from browser_use.agent.jev.views import JevSettings
from browser_use.benchmark import server
from browser_use.benchmark.tasks import task_catalog
from tests.ci.jev_support import browser_eval
from tests.ci.jev_support import jev_browser as _jev_browser_fixture  # noqa: F401
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401


async def test_visual_cards_plain_names_are_only_in_the_illustrations(jev_browser, httpserver):
	fixture = Path(__file__).resolve().parents[2] / 'browser_use/benchmark/static/fixture.html'
	httpserver.expect_request('/fixture.html').respond_with_data(fixture.read_bytes(), content_type='text/html; charset=utf-8')
	await jev_browser.navigate_to(httpserver.url_for('/fixture.html') + '?scenario=visual-cards-plain')
	summary = await jev_browser.get_browser_state_summary(include_screenshot=False)
	observation = await observe(jev_browser, summary, JevSettings(), 1)
	buttons = observation.candidates['CLICK']
	assert len(buttons) == 3
	assert buttons[0].name == 'View stay' and buttons[0].role == 'button'
	assert all(
		(button.name, button.role, button.context) == (buttons[0].name, buttons[0].role, buttons[0].context) for button in buttons
	)

	text, title, attributes, pictures = await browser_eval(
		jev_browser,
		"""[
			document.body.textContent,
			document.title,
			[...document.querySelectorAll('*')].flatMap(element => [...element.attributes]
				.filter(attribute => /^(aria-|title$|alt$)/.test(attribute.name)).map(attribute => attribute.value)),
			[...document.querySelectorAll('main canvas')].map(canvas => canvas.toDataURL()),
		]""",
	)
	for name in ('Amber Cottage', 'Sage House', 'Harbor Loft'):
		assert name not in text and name not in title and name not in observation.visible_text
		assert not any(name in value for value in attributes)
	assert len(pictures) == len(set(pictures)) == 3
	await browser_eval(jev_browser, 'document.querySelectorAll("main button")[1].click()')
	assert await browser_eval(jev_browser, 'document.querySelector(".detail-copy h1").innerText') == 'Sage House'


def test_candidate_fixture_tasks_are_available_in_baseline_and_free_modes(monkeypatch):
	expected = {
		'114_visual_cards_plain': ('visual-cards-plain', 'Open the Sage House stay.'),
		'115_scroll_panel': ('scroll-panel', 'Check the Rooftop terrace option in Filters and leave it checked.'),
		'116_dense': ('dense', 'Open the archive ledger and leave it open.'),
	}
	registered = {task['id']: task for task in task_catalog(server.ORIGIN) if task['id'] in expected}
	assert registered.keys() == expected.keys()
	monkeypatch.setattr(server, 'BASELINE', None)
	monkeypatch.setattr(server, 'CHROME', None)
	monkeypatch.setattr(server, 'AGENT', None)
	monkeypatch.setattr(server, 'runtime_settings', lambda *args, **kwargs: object())
	opened = []

	def open_run(url, goal, settings, *, scenario, **kwargs):
		opened.append((url, goal, scenario))
		return SimpleNamespace(snapshot=lambda: {}, close=lambda: None)

	monkeypatch.setattr(server, 'BrowserUseRun', open_run)
	for task_id, (scenario, goal) in expected.items():
		task = registered[task_id]
		assert task['version'] == '2026-11-26.1'
		assert task['url'] == f'{server.ORIGIN}/fixture.html?scenario={scenario}'
		assert task['goal'] == goal
		server.command('reset', {'scenario': scenario, 'goal': goal})
		assert opened[-1] == (task['url'], goal, scenario)
