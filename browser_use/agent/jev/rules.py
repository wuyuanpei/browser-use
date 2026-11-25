"""Pure rules shared by observation, decision routing, and evidence collection."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from browser_use.agent.jev.views import LINKS_LIMIT_CHARS, Candidate

TWO_LEVEL_SUFFIXES = frozenset(
	'com.cn net.cn org.cn gov.cn edu.cn ac.cn com.hk com.tw org.tw com.sg com.my co.in co.uk org.uk ac.uk gov.uk '
	'co.jp ne.jp or.jp ac.jp co.kr or.kr com.au net.au org.au edu.au gov.au co.nz org.nz com.br com.mx com.ar com.tr co.za'.split()
)

_EFFECT_LABELS = {
	'bookmark': ('收藏', '取消收藏', '已收藏', 'bookmark', 'add to favorites', 'remove from favorites'),
	'like': ('点赞', '取消点赞', 'like', 'unlike'),
	'send': ('发送', '发送邮件', 'send'),
	'publish': ('发布', 'publish', 'post'),
	'delete': ('删除', '确认删除', 'delete'),
	'share': ('创建分享', 'create share link'),
	'pay': ('支付', '付款', '立即支付', 'pay', 'pay now'),
	'order': ('下单', '提交订单', '立即购买', 'place order', 'buy now'),
	'booking': ('确认预订', '预订', 'confirm booking', 'book now', 'reserve'),
}
_SETTING_NAME = re.compile(r'(?:(?:启用|开启|停用|禁用).+功能|(?:enable|disable) .+)')
SEND_EFFECTS = frozenset({'send', 'publish', 'share'})


def http_url(value: object) -> bool:
	if not isinstance(value, str):
		return False
	try:
		parts = urlsplit(value)
		return parts.scheme in {'http', 'https'} and bool(parts.hostname)
	except ValueError:
		return False


def document_key(url: str) -> str:
	parts = urlsplit(url)
	return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ''))


def content_lines(text: str) -> list[str]:
	return [line for raw in text.splitlines() if (line := ' '.join(raw.split()))]


def observed_link_pairs(links: list[dict[str, str]]) -> list[dict[str, str]]:
	seen: set[tuple[str, str]] = set()
	result: list[dict[str, str]] = []
	remaining = LINKS_LIMIT_CHARS
	for link in links:
		url, text = link.get('url'), link.get('text')
		if not isinstance(url, str) or not http_url(url) or not isinstance(text, str) or not text.strip() or len(text) > 300:
			continue
		size = len(url) + len(text)
		if size <= remaining and (url, text) not in seen:
			result.append({'url': url, 'text': text})
			seen.add((url, text))
			remaining -= size
	return result


def registrable_domain(host: str) -> str:
	host = host.strip().rstrip('.').lower()
	if '.' not in host:
		return host
	try:
		ipaddress.ip_address(host)
		return host
	except ValueError:
		pass
	labels = host.split('.')
	count = 3 if '.'.join(labels[-2:]) in TWO_LEVEL_SUFFIXES else 2
	return '.'.join(labels[-count:])


def site_of(url: str) -> str | None:
	return registrable_domain(urlsplit(url).hostname or '') if http_url(url) else None


def is_ordinary_submit(tag: str, input_type: str | None, button_type: str | None, in_form: bool) -> bool:
	if tag == 'input':
		return input_type in {'submit', 'image'}
	return tag == 'button' and in_form and (button_type or 'submit').lower() == 'submit'


def select_match_text(options: list[tuple[str, str | None]], k: int) -> str | None:
	"""Return an argument that BU's first text-or-value match resolves to option k."""
	earlier: set[str] = set()
	for text, value in options[:k]:
		earlier.add(text.lower())
		earlier.add(text.lower() if value is None else value.lower())
	text, value = options[k]
	for key in (value, text):
		if key and key.lower() not in earlier:
			return key
	return None


def risky_effect(role: str, name: str, extra_labels: list[str] | None = None) -> str | None:
	if role not in {'button', 'menuitem', 'switch'}:
		return None
	normalized = name.strip().casefold()
	for effect, labels in _EFFECT_LABELS.items():
		for label in labels:
			if normalized.startswith(label):
				remainder = normalized[len(label) :]
				if (
					not remainder
					or not label.isascii()
					or remainder[0].isspace()
					or unicodedata.category(remainder[0]).startswith('P')
				):
					return effect
	if _SETTING_NAME.fullmatch(normalized):
		return 'account_setting'
	if extra_labels and any(normalized == label.strip().casefold() for label in extra_labels):
		return 'custom'
	return None


def enter_effect(field: Candidate, click_candidates: list[Candidate]) -> tuple[str, Candidate] | None:
	"""Find the control whose effect Enter in this form or editor can trigger."""
	if field.form_key is not None:
		controls = [c for c in click_candidates if c.form_key == field.form_key and c.risky_effect]
		controls.sort(key=lambda c: not c.is_submit)
	else:
		controls = [c for c in click_candidates if c.dialog_key == field.dialog_key and c.risky_effect in SEND_EFFECTS]
	for control in controls:
		if control.risky_effect:
			return control.risky_effect, control
	return None
