"""Project Browser Use's DOM snapshot into Jev's visible action candidates."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterator
from typing import Any
from urllib.parse import urljoin

from browser_use.agent.jev import rules
from browser_use.agent.jev.views import Candidate, JevSettings, Observation
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import EnhancedDOMTreeNode

logger = logging.getLogger(__name__)

VISIBLE_TEXT_SCRIPT = r"""(() => {
	const LIMIT = 50000;
	const vw = window.innerWidth || document.documentElement.clientWidth;
	const vh = window.innerHeight || document.documentElement.clientHeight;
	const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG']);
	const inViewport = (rects) => {
		for (const r of rects) {
			if (r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw) return true;
		}
		return false;
	};
	const blockCache = new Map();
	const isBlock = (el) => {
		if (!blockCache.has(el)) {
			const d = getComputedStyle(el).display;
			blockCache.set(el, !['inline', 'inline-block', 'inline-flex', 'inline-grid', 'contents'].includes(d));
		}
		return blockCache.get(el);
	};
	const blockOf = (el) => {
		let cur = el;
		while (cur && cur !== document.body && !isBlock(cur)) cur = cur.parentElement;
		return cur || document.body;
	};
	const lines = [];
	let current = null;
	let currentBlock = null;
	let total = 0;
	const root = document.body || document.documentElement;
	const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
	let node;
	while ((node = walker.nextNode()) && total < LIMIT) {
		const raw = node.nodeValue;
		if (!raw || !raw.trim()) continue;
		const el = node.parentElement;
		if (!el || SKIP.has(el.tagName.toUpperCase())) continue;
		const range = document.createRange();
		range.selectNodeContents(node);
		if (!inViewport(range.getClientRects())) continue;
		const style = getComputedStyle(el);
		if (style.visibility === 'hidden' || style.visibility === 'collapse') continue;
		const text = raw.replace(/\s+/g, ' ').trim();
		const block = blockOf(el);
		if (block !== currentBlock) {
			if (current) lines.push(current);
			current = text;
			currentBlock = block;
		} else {
			current += ' ' + text;
		}
		total += text.length + 1;
	}
	if (current) lines.push(current);
	const links = [];
	for (const a of document.querySelectorAll('a[href]')) {
		if (links.length >= 200) break;
		if (!/^https?:/i.test(a.href) || !inViewport(a.getClientRects())) continue;
		const text = (a.innerText || a.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim();
		if (text) links.push({ text: text.slice(0, 300), url: a.href });
	}
	return { url: location.href, title: document.title, text: lines.join('\n').slice(0, LIMIT), links, viewport: { width: vw, height: vh } };
})()"""


async def read_visible_text(browser_session: BrowserSession) -> dict[str, Any]:
	"""Read main-document text and links from a single viewport in one CDP request."""
	try:
		cdp = await browser_session.get_or_create_cdp_session()
		result = await cdp.cdp_client.send.Runtime.evaluate(
			params={'expression': VISIBLE_TEXT_SCRIPT, 'returnByValue': True}, session_id=cdp.session_id
		)
		value = result['result'].get('value')
		if isinstance(value, dict) and 'text' in value:
			return value
		logger.debug('Could not read viewport: %s', result.get('exceptionDetails', {}).get('text'))
	except Exception as exc:
		logger.debug('Could not read viewport: %s', exc)
	return {'text': '', 'links': [], 'viewport': {'width': 0, 'height': 0}, 'url': '', 'title': ''}


_STATES = frozenset({'checked', 'selected', 'expanded', 'pressed', 'disabled', 'required', 'readonly'})
_HEADINGS = frozenset({'h1', 'h2', 'h3', 'h4', 'h5', 'h6'})
_TEXT_INPUTS = frozenset({'text', 'search', 'email', 'url', 'tel', 'number'})


def _short(value: str | None, limit: int) -> str:
	return ' '.join((value or '').split())[:limit]


def _role(node: EnhancedDOMTreeNode) -> str:
	role = node.ax_node.role if node.ax_node else None
	if role and role not in {'none', 'generic'}:
		return role
	tag, attrs = node.tag_name, node.attributes
	if tag == 'input':
		kind = attrs.get('type', 'text').lower()
		if kind in {'checkbox', 'radio'}:
			return kind
		return 'button' if kind in {'button', 'reset', 'submit', 'image'} else 'textbox'
	if 'contenteditable' in attrs:
		return 'textbox'
	if tag == 'a' and attrs.get('href'):
		return 'link'
	return {'textarea': 'textbox', 'select': 'combobox'}.get(tag, tag)


def _name(node: EnhancedDOMTreeNode) -> str:
	attrs = node.attributes
	for value in (
		node.ax_node.name if node.ax_node else None,
		attrs.get('aria-label'),
		attrs.get('placeholder'),
		attrs.get('title'),
		attrs.get('alt'),
	):
		if value and value.strip():
			return _short(value, 120)
	if node.tag_name == 'input' and _role(node) == 'button' and attrs.get('value'):
		return _short(attrs['value'], 120)
	return _short(node.get_all_children_text(max_depth=3), 120)


def _states(node: EnhancedDOMTreeNode) -> dict[str, bool]:
	states = (
		{
			p.name: p.value is True or str(p.value).lower() in {'true', 'mixed'}
			for p in (node.ax_node.properties or [])
			if p.name in _STATES
		}
		if node.ax_node
		else {}
	)
	for attr in ('disabled', 'readonly', 'required'):
		if attr in node.attributes:
			states[attr] = True
	return states


def _descendants(node: EnhancedDOMTreeNode) -> Iterator[EnhancedDOMTreeNode]:
	pending = list(reversed(node.children))
	while pending:
		child = pending.pop()
		yield child
		pending.extend(reversed(child.children))


def _heading(node: EnhancedDOMTreeNode) -> bool:
	return node.tag_name in _HEADINGS or node.attributes.get('role') == 'heading'


def _context(node: EnhancedDOMTreeNode) -> tuple[dict[str, str], str | None, str | None]:
	"""Name scopes using the same tree that produced the executable action index."""
	context: dict[str, str] = {}
	form = dialog = group = table = row = cell = None
	child, parent, depth = node, node.parent, 0
	while parent:
		tag, attrs = parent.tag_name, parent.attributes
		if form is None and tag == 'form':
			form = parent
		if dialog is None and (tag == 'dialog' or attrs.get('role') in {'dialog', 'alertdialog'}):
			dialog = parent
		if group is None and (tag == 'fieldset' or attrs.get('role') in {'group', 'radiogroup'}):
			group = parent
		if cell is None and tag in {'td', 'th'}:
			cell = parent
		if row is None and tag == 'tr':
			row = parent
		if table is None and tag == 'table':
			table = parent
		if depth < 12 and 'section' not in context:
			siblings = parent.children
			idx = next((i for i, item in enumerate(siblings) if item is child), 0)
			prior_heading = next((s for s in reversed(siblings[max(0, idx - 15) : idx]) if _heading(s)), None)
			if prior_heading:
				context['section'] = _short(prior_heading.get_all_children_text(), 80)
		child, parent = parent, parent.parent
		depth += 1
	if node.attributes.get('form'):
		form_id = node.attributes['form']
		form = next((n for n in _descendants(child) if n.tag_name == 'form' and n.attributes.get('id') == form_id), None)
	if form:
		context['form'] = _short(
			form.attributes.get('aria-label') or form.attributes.get('name') or form.attributes.get('id'), 60
		)
	if dialog:
		label = dialog.attributes.get('aria-label')
		if not label:
			header = next((el for el in _descendants(dialog) if _heading(el)), None)
			label = header.get_all_children_text() if header else None
		context['dialog'] = _short(label, 60)
	if group:
		legend = next((el for el in group.children if el.tag_name == 'legend'), None)
		context['group'] = _short(legend.get_all_children_text() if legend else group.attributes.get('aria-label'), 60)
	if row and table and cell:
		cells = [el for el in row.children if el.tag_name in {'th', 'td'}]
		label_cell = next((el for el in cells if el.tag_name == 'th'), cells[0] if cells else None)
		if label_cell and label_cell is not cell:
			context['table_row'] = _short(label_cell.get_all_children_text(), 60)
		col = next((i for i, el in enumerate(cells) if el is cell), -1)
		headers_row = next(
			(el for el in _descendants(table) if el.tag_name == 'tr' and any(c.tag_name == 'th' for c in el.children)), None
		)
		if headers_row:
			headers = [el for el in headers_row.children if el.tag_name in {'th', 'td'}]
			if 0 <= col < len(headers) and headers[col].tag_name == 'th':
				context['table_column'] = _short(headers[col].get_all_children_text(), 60)
	return (
		{k: v for k, v in context.items() if v},
		str(form.backend_node_id) if form else None,
		str(dialog.backend_node_id) if dialog else None,
	)


def _options(node: EnhancedDOMTreeNode) -> list[EnhancedDOMTreeNode]:
	options = []
	for child in node.children:
		if child.tag_name == 'option':
			options.append(child)
		elif child.tag_name == 'optgroup':
			options.extend(opt for opt in child.children if opt.tag_name == 'option')
	return options


def _option_text(node: EnhancedDOMTreeNode) -> str:
	return ' '.join(''.join(child.node_value for child in node.children if child.tag_name == '#text').split())


async def selected_option(
	browser_session: BrowserSession, node: EnhancedDOMTreeNode, *, live: bool = False
) -> tuple[int, str] | None:
	"""Return the live one-based option index and label, including duplicate labels."""
	if not live:
		for ordinal, option in enumerate(_options(node), 1):
			if _states(option).get('selected'):
				return ordinal, _option_text(option)
	# Collapsed native selects can omit option AX nodes. Read the DOM property rather
	# than the selected HTML attribute, which describes the initial selection.
	try:
		cdp = await browser_session.cdp_client_for_node(node)
		resolved = await cdp.cdp_client.send.DOM.resolveNode(
			params={'backendNodeId': node.backend_node_id}, session_id=cdp.session_id
		)
		object_id = resolved['object'].get('objectId')
		if not object_id:
			return None
		try:
			result = await cdp.cdp_client.send.Runtime.callFunctionOn(
				params={
					'objectId': object_id,
					'functionDeclaration': 'function(){const i=this.selectedIndex;return i>=0?[i+1,this.options[i].text]:null;}',
					'returnByValue': True,
				},
				session_id=cdp.session_id,
			)
			value = result['result'].get('value')
			if isinstance(value, list) and len(value) == 2:
				return int(value[0]), ' '.join(str(value[1]).split())
		finally:
			await cdp.cdp_client.send.Runtime.releaseObject(params={'objectId': object_id}, session_id=cdp.session_id)
	except Exception as exc:
		logger.debug('Could not read selection for node %s: %s', node.backend_node_id, exc)
	return None


def _visible(node: EnhancedDOMTreeNode, width: int, height: int) -> bool:
	# In BU 0.13.14 the tree builder has already applied frame and scroll offsets.
	# DOMSnapshot.clientRects contains clientLeft/clientTop, not the element origin.
	rect = node.absolute_position
	return bool(
		node.is_visible is not False
		and rect
		and rect.width > 0
		and rect.height > 0
		and rect.x < width
		and rect.y < height
		and rect.x + rect.width > 0
		and rect.y + rect.height > 0
	)


def _viewport_rect(node: EnhancedDOMTreeNode, width: int, height: int) -> dict[str, int] | None:
	rect = node.absolute_position
	if rect is None:
		return None
	x, y = rect.x, rect.y
	# BU adds each iframe's border-box origin. Its document starts inside the border.
	parent = node.parent
	while parent is not None:
		if parent.tag_name in {'iframe', 'frame'} and parent.snapshot_node and (client := parent.snapshot_node.clientRects):
			x += client.x
			y += client.y
		parent = parent.parent
	left, top = max(0, math.floor(x)), max(0, math.floor(y))
	right, bottom = min(width, math.ceil(x + rect.width)), min(height, math.ceil(y + rect.height))
	if left >= right or top >= bottom:
		return None
	return {'x': left, 'y': top, 'w': right - left, 'h': bottom - top}


def _element(index: int, node: EnhancedDOMTreeNode, url: str, settings: JevSettings, width: int, height: int) -> Candidate:
	attrs, tag, role, states = node.attributes, node.tag_name, _role(node), _states(node)
	context, form_key, dialog_key = _context(node)
	kind = attrs.get('type', 'text').lower() if tag == 'input' else None
	editable = (
		kind != 'password'
		and not (states.get('disabled') or states.get('readonly'))
		and (
			tag == 'textarea'
			or kind in _TEXT_INPUTS
			or ('contenteditable' in attrs and attrs['contenteditable'].lower() != 'false')
			or (tag != 'select' and role in {'textbox', 'searchbox', 'combobox'})
		)
	)
	try:
		href = urljoin(url, attrs['href']) if attrs.get('href') else None
	except ValueError:
		href = None
	name = _name(node)
	return Candidate(
		key=str(index),
		kind='element',
		index=index,
		role=role,
		name=name,
		tag=tag,
		input_type=kind,
		value=('***' if kind == 'password' else attrs.get('value', '')[:120]) if tag in {'textarea', 'input'} else None,
		states=states,
		placeholder=_short(attrs.get('placeholder'), 80) or None,
		href=href[:200] if href and rules.http_url(href) else None,
		context=context,
		rect=_viewport_rect(node, width, height),
		editable=editable,
		is_submit=rules.is_ordinary_submit(tag, kind, attrs.get('type'), bool(form_key or 'form' in attrs)),
		risky_effect=rules.risky_effect(role, name, settings.risky_action_labels)
		if settings.checks.mode('risky_action') != 'off'
		else None,
		form_key=form_key,
		dialog_key=dialog_key,
	)


async def _add_options(browser: BrowserSession, node: EnhancedDOMTreeNode, field: Candidate, candidates: list[Candidate]) -> int:
	options = _options(node)
	pairs = [(_option_text(option), option.attributes.get('value')) for option in options]
	selection = await selected_option(browser, node)
	excluded = 0
	for k, option in enumerate(options):
		states = _states(option)
		if states.get('disabled') or (
			option.parent and option.parent.tag_name == 'optgroup' and 'disabled' in option.parent.attributes
		):
			continue
		match = rules.select_match_text(pairs, k)
		if match is None:
			excluded += 1
			continue
		text, _ = pairs[k]
		states['selected'] = selection is not None and selection[0] == k + 1
		context = dict(field.context)
		if field.name:
			context['select'] = field.name[:60]
		candidates.append(
			Candidate(
				key=f'{field.key}:{k + 1}',
				kind='option',
				role='option',
				name=text[:120],
				tag='option',
				index=field.index,
				states=states,
				value=selection[1][:120] if selection else None,
				option_text=text,
				match_text=match,
				context=context,
				form_key=field.form_key,
				dialog_key=field.dialog_key,
			)
		)
	return excluded


def fingerprint(obs: Observation) -> str:
	candidates = {c.key: c for items in obs.candidates.values() for c in items}
	payload = [
		obs.url,
		obs.title,
		obs.scroll['y'],
		sorted(t['url'] for t in obs.tabs),
		rules.content_lines(obs.visible_text),
		[[c.key, c.role, c.name, c.value, sorted(c.states.items())] for c in sorted(candidates.values(), key=lambda c: c.key)],
	]
	return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


async def observe(browser_session: BrowserSession, summary: BrowserStateSummary, settings: JevSettings, step: int) -> Observation:
	rendered = await read_visible_text(browser_session)
	# A newly focused target can have an empty cached URL before TargetInfoChanged arrives.
	page_url = summary.url or rendered['url']
	info = summary.page_info
	viewport = rendered.get('viewport') or {}
	width = info.viewport_width if info else int(viewport.get('width', 0))
	height = info.viewport_height if info else int(viewport.get('height', 0))
	above = info.pixels_above if info else summary.pixels_above
	below = info.pixels_below if info else summary.pixels_below
	scroll = {
		'y': info.scroll_y if info else above,
		'viewport_height': height,
		'page_height': info.page_height if info else above + height + below,
		'pixels_above': above,
		'pixels_below': below,
	}
	candidates: dict[str, list[Candidate]] = {op: [] for op in ('CLICK', 'TYPE_TEXT', 'SELECT', 'SCROLL_UP', 'SCROLL_DOWN')}
	page = Candidate(key='page', kind='scroll', role='page', name='the whole page')
	if above > 0:
		candidates['SCROLL_UP'].append(page)
	if below > 0:
		candidates['SCROLL_DOWN'].append(page)
	unselectable = 0
	for index, node in summary.dom_state.selector_map.items():
		if not _visible(node, width, height):
			continue
		field = _element(index, node, page_url, settings, width, height)
		if not field.states.get('disabled') and field.tag not in {'select', 'option'}:
			candidates['CLICK'].append(field)
		if field.editable:
			candidates['TYPE_TEXT'].append(field)
		if field.tag == 'select' and not field.states.get('disabled'):
			unselectable += await _add_options(browser_session, node, field, candidates['SELECT'])
		if node.is_actually_scrollable:
			candidates['SCROLL_UP'].append(field)
			candidates['SCROLL_DOWN'].append(field)
	tabs: list[dict[str, str]] = []
	for tab in summary.tabs:
		if tab.target_id != browser_session.agent_focus_target_id:
			key = f'tab:{len(tabs) + 1}'
			tab_id = tab.target_id[-4:]
			tabs.append({'key': key, 'tab_id': tab_id, 'title': tab.title, 'url': tab.url})
			candidates['CLICK'].append(
				Candidate(
					key=key, kind='tab', role='tab', name=tab.title[:120] or tab.url[:120], href=tab.url[:200], tab_id=tab_id
				)
			)
	go_back = False
	try:
		cdp = await browser_session.get_or_create_cdp_session()
		history = await cdp.cdp_client.send.Page.getNavigationHistory(session_id=cdp.session_id)
		current = history['currentIndex']
		go_back = current > 0 and rules.http_url(history['entries'][current - 1]['url'])
	except Exception as exc:
		logger.debug('Could not read navigation history: %s', exc)
	obs = Observation(
		step=step,
		url=page_url,
		title=summary.title or rendered['title'],
		# A viewport screenshot includes classic scrollbars. PageInfo describes the content area.
		viewport_width=viewport.get('width') or width,
		viewport_height=viewport.get('height') or height,
		scroll=scroll,
		visible_text=rendered['text'],
		links=rendered['links'],
		candidates=candidates,
		tabs=tabs,
		can_go_back=go_back,
		is_pdf=summary.is_pdf_viewer,
		state_error=summary.state_error,
		unselectable_options=unselectable,
		fingerprint='',
	)
	obs.fingerprint = fingerprint(obs)
	return obs
