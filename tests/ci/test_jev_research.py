import json

from browser_use.agent.jev.research import CaptureStore
from browser_use.agent.jev.views import Observation


def observation(url, text):
	return Observation(
		step=3,
		url=url,
		title='Current',
		visible_text=text,
		links=[],
		candidates={},
		scroll={},
		tabs=[],
		can_go_back=False,
		is_pdf=False,
		state_error=None,
		fingerprint='page',
	)


def test_capture_durability_incremental_page_and_source(tmp_path):
	store = CaptureStore(tmp_path / 'captures', 'agent-a')
	obs = observation('https://test.example/page#section', 'Title\nFirst item')
	assert not store.directory.exists()
	assert store.available(obs)
	p = store.add(
		'sg1',
		3,
		{
			'url': obs.url,
			'title': obs.title,
			'text': obs.visible_text,
			'links': [
				{'text': 'Next', 'url': 'https://test.example/next'},
				{'text': 'Next', 'url': 'https://test.example/next'},
				{'text': 'bad', 'url': 'javascript:alert(1)'},
			],
		},
	)
	file = store.path(p)
	assert file.name == 'agent-a_p0001.json'
	assert json.loads(file.read_text(encoding='utf-8'))['summary_status'] == 'pending'
	p.summary_status = 'failed'
	p.summary_error = 'TimeoutError'
	store.write(p)
	assert json.loads(file.read_text(encoding='utf-8'))['text'] == obs.visible_text
	assert not store.available(observation('https://test.example/page', 'First item'))
	assert store.available(observation('https://test.example/page', 'Title\nSecond item'))
	assert not store.available(observation('https://test.example/new', '  '))
	assert not store.available(observation('about:blank', 'Text'))
	research = store.research_state(obs)
	assert research and research['current_page_saved'] and research['collected_pages'][0]['id'] == 'p1'
	assert 'text' not in research['collected_pages'][0]
	assert p.links == [{'text': 'Next', 'url': 'https://test.example/next'}]
