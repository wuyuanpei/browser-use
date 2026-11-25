import json

import pytest

from browser_use.agent.jev import rules
from browser_use.agent.jev.checks import CheckGate
from browser_use.agent.jev.trace import JevTrace
from browser_use.agent.jev.views import Candidate, JevChecks


@pytest.mark.parametrize(
	('host', 'expected'),
	[
		('flights.google.com', 'google.com'),
		('www.google.com', 'google.com'),
		('news.bbc.co.uk', 'bbc.co.uk'),
		('a.b.example.com.cn', 'example.com.cn'),
		('co.uk', 'co.uk'),
		('localhost', 'localhost'),
		('127.0.0.1', '127.0.0.1'),
		('::1', '::1'),
	],
)
def test_site(host, expected):
	assert rules.registrable_domain(host) == expected


def test_document_identity_and_visible_links():
	assert rules.site_of('http://localhost:8080/a') == rules.site_of('http://localhost:9090/b') == 'localhost'
	assert rules.site_of('about:blank') is None
	assert not rules.http_url('http://[')
	assert not rules.http_url(None)
	assert rules.document_key('HTTPS://EXAMPLE.COM/a?b=1#c') == 'https://example.com/a?b=1'
	assert rules.content_lines('  \n text\treading \t   \n more ') == ['text reading', 'more']
	link = {'url': 'https://example.com/alice', 'text': 'Alice'}
	assert rules.observed_link_pairs(
		[
			link,
			link,
			{'url': 'javascript:foo', 'text': 'Alice'},
			{'url': link['url'], 'text': 'x' * 301},
			{'url': 'https://example.com/' + 'x' * 6000, 'text': 'long'},
			{'url': 'https://example.com/z', 'text': 'z'},
		]
	) == [link, {'url': 'https://example.com/z', 'text': 'z'}]


@pytest.mark.parametrize(
	('role', 'name', 'extra', 'effect'),
	[
		('button', '发送邮件', None, 'send'),
		('button', 'Delete account', None, 'delete'),
		('button', 'Postpone', None, None),
		('link', 'Delete', None, None),
		('switch', 'Enable notifications', None, 'account_setting'),
		('button', '启用自动回复功能', None, 'account_setting'),
		('menuitem', 'Add to favorites', None, 'bookmark'),
		('button', 'Follow', ['follow'], 'custom'),
		('button', 'Search', None, None),
		('button', 'Pay!', None, 'pay'),
		('button', 'Place order', None, 'order'),
		('button', 'Book now', None, 'booking'),
		('button', 'Publish', None, 'publish'),
		('button', 'Create share link', None, 'share'),
		('button', 'Unlike', None, 'like'),
	],
)
def test_risky_effect(role, name, extra, effect):
	assert rules.risky_effect(role, name, extra) == effect


@pytest.mark.parametrize(
	('tag', 'input_type', 'button_type', 'form', 'expected'),
	[
		('input', 'submit', None, False, True),
		('input', 'image', None, False, True),
		('input', 'text', None, True, False),
		('button', None, None, True, True),
		('button', None, 'submit', True, True),
		('button', None, 'button', True, False),
		('button', None, 'reset', True, False),
		('button', None, None, False, False),
	],
)
def test_submission(tag, input_type, button_type, form, expected):
	assert rules.is_ordinary_submit(tag, input_type, button_type, form) is expected


@pytest.mark.parametrize(
	('options', 'k', 'expected'),
	[
		([('Other', '1'), ('Other', '2')], 1, '2'),
		([('Other', None), ('Other', None)], 1, None),
		([('First', ''), ('Second', None)], 0, 'First'),
		([('FIRST', 'seCond'), ('First', 'SECond')], 1, None),
		([('New', 'ID'), ('Second', 'id')], 1, 'Second'),
	],
)
def test_select_identifies_exact_option(options, k, expected):
	assert rules.select_match_text(options, k) == expected


@pytest.mark.parametrize(
	('form', 'dialog', 'button_form', 'button_dialog', 'effect'),
	[
		('f', None, 'f', None, 'send'),
		('f', None, 'g', None, None),
		(None, None, None, None, 'send'),
		(None, 'd', None, None, None),
		(None, 'd', None, 'd', 'send'),
	],
)
def test_enter_effect(form, dialog, button_form, button_dialog, effect):
	field = Candidate(key='1', kind='element', role='textbox', name='Draft', form_key=form, dialog_key=dialog)
	control = Candidate(
		key='2', kind='element', role='button', name='Send', risky_effect='send', form_key=button_form, dialog_key=button_dialog
	)
	assert rules.enter_effect(field, [control]) == ((effect, control) if effect else None)
	assert rules.enter_effect(field, [control.model_copy(update={'risky_effect': None, 'name': 'Search'})]) is None


@pytest.mark.parametrize('mode', ['enforce', 'record', 'off'])
@pytest.mark.parametrize('triggered', [True, False])
def test_check_modes(tmp_path, mode, triggered):
	path = tmp_path / 'trace.jsonl'
	gate = CheckGate(JevChecks(default=mode), JevTrace(path, 'test'))
	assert gate.evaluate('scope', triggered, 'Changed site', 2) is (triggered and mode == 'enforce')
	assert gate.is_off('scope') is (mode == 'off')
	events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
	if triggered and mode != 'off':
		assert len(events) == 1
		assert events[0]['name'] == 'scope' and events[0]['step'] == 2
		assert events[0]['outcome'] == ('blocked' if mode == 'enforce' else 'would_block')
	else:
		assert not events
