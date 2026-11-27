"""Project Browser Use's DOM snapshot into Jev's visible action candidates."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, fields
from difflib import SequenceMatcher
from functools import partial
from typing import Any
from urllib.parse import urljoin

from browser_use.agent.jev import rules, scroll_regions
from browser_use.agent.jev.views import Candidate, JevSettings, Observation, ScrollEffect, ScrollObservation, ScrollRegion
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import EnhancedDOMTreeNode, SerializedDOMState

logger = logging.getLogger(__name__)

_VIEWPORT_BODY = r"""(() => {
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
	const html = document.documentElement;
	const nearestVerticalScroller = (hit) => {
		for (let el = hit; el; el = el.parentElement) {
			if (el === document.scrollingElement || !/^(auto|scroll)$/.test(getComputedStyle(el).overflowY)) continue;
			if (el.scrollHeight - el.clientHeight >= 1) return el;
		}
		return null;
	};
	const centerHit = document.elementFromPoint(vw / 2, vh / 2);
	const centerElement = nearestVerticalScroller(centerHit);
	const scrollAtCenter = centerElement ? [Math.max(0, centerElement.scrollTop),
		Math.max(0, centerElement.scrollHeight - centerElement.clientHeight - centerElement.scrollTop)] : null;
	// Browser Use's native scroll gesture uses the center of the layout viewport, excluding scrollbars.
	const gestureHit = document.elementFromPoint((html.clientWidth || vw) / 2, (html.clientHeight || vh) / 2);
	const gestureCenter = gestureHit === centerHit ? centerElement : nearestVerticalScroller(gestureHit);
	const rootStyle = getComputedStyle(html);
	const overflowFromBody = document.body && rootStyle.overflowX === 'visible' && rootStyle.overflowY === 'visible';
	const rootOverflowY = overflowFromBody ? getComputedStyle(document.body).overflowY : rootStyle.overflowY;
	const centerPath = [];
	for (let el = gestureCenter; el; el = el.parentElement) {
		let index = 0;
		for (let sibling = el.previousElementSibling; sibling; sibling = sibling.previousElementSibling) index++;
		centerPath.push(el.localName + ':' + index);
	}
	const scrollDiagnostics = {rootOverflowY, centerY: gestureCenter?.scrollTop ?? null,
		centerMatchesCandidate: gestureCenter === centerElement,
		centerPath: gestureCenter ? centerPath.reverse().join('/') : null,
		centerEmbedded: Boolean(gestureHit?.matches('iframe,frame') || gestureHit?.shadowRoot)};
	const viewport = {centerScroll: scrollAtCenter, url: location.href, title: document.title, document: performance.timeOrigin,
		scroll: [scrollX, scrollY], text: lines.join('\n').slice(0, LIMIT), links, viewport: {width: vw, height: vh},
		scrollDiagnostics};
"""
VISIBLE_TEXT_SCRIPT = _VIEWPORT_BODY + 'return viewport; })()'
_VIEWPORT_WITH_REGIONS = (
	_VIEWPORT_BODY
	+ scroll_regions.OBSERVE_JS
	+ 'return [JSON.stringify({...viewport, regions}), centerElement, ...regionNodes]; })()'
)


async def _legacy_viewport(cdp) -> dict:
	result = await cdp.cdp_client.send.Runtime.evaluate(
		params={'expression': VISIBLE_TEXT_SCRIPT, 'returnByValue': True}, session_id=cdp.session_id
	)
	value = result['result'].get('value')
	if not isinstance(value, dict) or 'text' not in value:
		raise ValueError('Invalid viewport response')
	return {**value, 'target_id': cdp.target_id}


async def read_visible_text(browser_session: BrowserSession, *, include_regions: bool = True) -> dict[str, Any]:
	"""Read main-document text and links from a single viewport in one CDP request."""
	try:
		cdp = await browser_session.get_or_create_cdp_session()
		if not include_regions:
			return await _legacy_viewport(cdp)
		try:
			result = await cdp.cdp_client.send.Runtime.evaluate(
				params={
					'expression': _VIEWPORT_WITH_REGIONS,
					'serializationOptions': {'serialization': 'deep', 'maxDepth': 2, 'additionalParameters': {'maxNodeDepth': 0}},
				},
				session_id=cdp.session_id,
			)
		except Exception as exc:
			message = str(exc).lower()
			if any(name in message for name in ('serializationoptions', 'maxnodedepth')) and any(
				word in message for word in ('invalid', 'unknown', 'unsupported', 'not found', 'unrecognized')
			):
				return await _legacy_viewport(cdp)
			raise
		remote = result['result']
		try:
			items = remote.get('deepSerializedValue', {}).get('value', [])
			if items and 'exceptionDetails' not in result:
				value = json.loads(items[0]['value'])
				center = items[1].get('value') or {}
				value['centerBackendNodeId'] = center.get('backendNodeId')
				for region, item in zip(value['regions'], items[2:], strict=True):
					region['backendNodeId'] = (item.get('value') or {}).get('backendNodeId')
				return {**value, 'target_id': cdp.target_id}
		finally:
			if object_id := remote.get('objectId'):
				await cdp.cdp_client.send.Runtime.releaseObject(params={'objectId': object_id}, session_id=cdp.session_id)
		if 'exceptionDetails' not in result and 'deepSerializedValue' not in remote:
			# Older endpoints ignore the serialization option and return an ordinary array handle.
			return await _legacy_viewport(cdp)
		logger.debug('Could not read viewport: %s', result.get('exceptionDetails', {}).get('text'))
	except Exception as exc:
		logger.debug('Could not read viewport: %s', exc)
	return {'text': '', 'links': [], 'viewport': {'width': 0, 'height': 0}, 'url': '', 'title': ''}


def _alignment(view: dict) -> dict:
	# The PR5 fields include the document identity. Only a region's node identity and scroll position
	# affect this comparison; labels, lazy-loaded dimensions and moving boxes can change independently.
	stable = {k: v for k, v in view.items() if k not in ('regions', 'scrollDiagnostics')}
	stable['regions'] = [(r['backendNodeId'], r['scrollTop']) for r in view.get('regions', []) if r.get('backendNodeId')]
	return stable


@dataclass(kw_only=True)
class UnalignedState(BrowserStateSummary):
	"""The two rejected comparisons belong to this exact failed observation."""

	alignment_attempts: list[dict[str, Any]]


def _alignment_attempt(attempt: int, before: dict, after: dict, summary_url: str) -> dict[str, Any]:
	old, new = _alignment(before), _alignment(after)
	components = sorted(
		k for k in old.keys() | new.keys() if k != 'regions' and (k not in old or k not in new or old[k] != new[k])
	)
	if ('url' not in components) and (not after.get('url') or (summary_url and summary_url != after['url'])):
		components.append('url')
	old_regions = {r['backendNodeId']: r for r in before.get('regions', []) if r.get('backendNodeId')}
	new_regions = {r['backendNodeId']: r for r in after.get('regions', []) if r.get('backendNodeId')}
	same_document = before.get('document') == after.get('document') and before.get('target_id') == after.get('target_id')
	if list(old_regions) != list(new_regions) or ((old_regions or new_regions) and not same_document):
		components.append('regions.identity')
	shared = old_regions.keys() & new_regions.keys() if same_document else set()
	ignored = []
	for name, keys in [
		('regions.scroll', ('scrollTop',)),
		('regions.name', ('name',)),
		('regions.size', ('scrollHeight', 'clientHeight')),
	]:
		if any(old_regions[node].get(k) != new_regions[node].get(k) for node in shared for k in keys):
			(components if name == 'regions.scroll' else ignored).append(name)
	removed = added = 0
	if 'text' in components:
		diff = SequenceMatcher(
			None, before.get('text', '').splitlines(keepends=True), after.get('text', '').splitlines(keepends=True)
		)
		for operation, i, j, k, end in diff.get_opcodes():
			if operation != 'equal':
				removed += j - i
				added += end - k
	return {
		'attempt': attempt,
		'components': components,
		'ignored_components': ignored,
		'text': {'before_changed_lines': removed, 'after_changed_lines': added},
	}


async def capture_aligned_state(
	browser: BrowserSession, *, include_recent_events: bool, include_regions: bool = True
) -> BrowserStateSummary:
	"""Bracket the native DOM capture with viewport reads to catch navigation and delayed content."""
	read = read_visible_text if include_regions else partial(read_visible_text, include_regions=False)
	before = await read(browser)
	rejected: list[tuple[dict, dict, str]] = []
	for _ in range(2):
		summary = await browser.get_browser_state_summary(include_screenshot=True, include_recent_events=include_recent_events)
		after = await read(browser)
		matches_url = not summary.url or summary.url == after['url']
		if matches_url and after['url'] and _alignment(before) == _alignment(after):
			summary.jev_viewport = after
			return summary
		rejected.append((before, after, summary.url))
		before = after

	# These selectors must not survive a capture made across two versions of the page.
	browser.update_cached_selector_map({})
	detail = 'Page changed while taking its DOM snapshot. Observe again before choosing an element.'
	data = {f.name: getattr(summary, f.name) for f in fields(BrowserStateSummary)}
	data.update(
		dom_state=SerializedDOMState(_root=None, selector_map={}),
		screenshot=None,
		url=after['url'] or summary.url,
		title=after['title'] or summary.title,
		state_error=summary.state_error or detail,
		jev_viewport=None,
	)
	return UnalignedState(
		**data, alignment_attempts=[_alignment_attempt(index, old, new, url) for index, (old, new, url) in enumerate(rejected, 1)]
	)


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


def _scope_text(scope: EnhancedDOMTreeNode) -> str:
	parts = []
	length = 0
	pending = [scope]
	for _ in range(80):
		if not pending or length >= 120:
			break
		node = pending.pop()
		if node.is_visible is False or 'hidden' in node.attributes or node.attributes.get('aria-hidden') == 'true':
			continue
		if node.tag_name == '#text':
			if text := _short(node.node_value, 120):
				parts.append(text)
				length += len(text) + 1
		elif node.tag_name not in {'script', 'style', 'noscript', 'template'}:
			pending.extend(reversed(node.children[:80]))
	return ' '.join(parts)[:120]


def _nearby_text(node: EnhancedDOMTreeNode, name: str) -> str:
	"""Describe the closest content around a repeated control without using page-wide text."""
	fallback = ''
	parent = node.parent
	for _ in range(6):
		if parent is None or parent.tag_name in {'body', 'html', '#document'}:
			break
		text = _scope_text(parent)
		if text and text != name:
			fallback = fallback or text
			if parent.tag_name in {'article', 'li'} or parent.attributes.get('role') in {'article', 'listitem'}:
				return text
		parent = parent.parent
	return fallback


def _covers_click(surface: EnhancedDOMTreeNode, clicked: EnhancedDOMTreeNode) -> bool:
	area, rect = clicked.absolute_position, surface.absolute_position
	if not area or not rect:
		return False
	x, y = area.x + area.width / 2, area.y + area.height / 2
	return rect.x < x < rect.x + rect.width and rect.y < y < rect.y + rect.height


def _drop_click_wrappers(clicks: list[Candidate], nodes: dict[int, EnhancedDOMTreeNode]) -> list[Candidate]:
	# Nested layout elements can describe the same click several times. Keep the inner
	# surface only when the outer candidate's click point is already inside it.
	indexed = {id(nodes[candidate.index]): candidate for candidate in clicks if candidate.index is not None}
	wrappers: dict[int, Candidate] = {}
	removed: set[str] = set()
	for candidate in clicks:
		if candidate.index is None or not candidate.name:
			continue
		node = nodes[candidate.index]
		parent = node.parent
		# A bare hash link can be the text surface of a clickable list item.
		# Keep links with their own navigation or handlers as separate controls.
		if (
			node.tag_name == 'a'
			and candidate.role == 'link'
			and node.attributes.get('href') == '#'
			and not node.has_js_click_listener
			and not {'target', 'download'} & node.attributes.keys()
			and parent is not None
			and parent.tag_name == 'li'
			and parent.has_js_click_listener
			and (owner := indexed.get(id(parent))) is not None
			and owner.role == 'listitem'
			and owner.name == candidate.name
			and _covers_click(node, parent)
		):
			removed.add(candidate.key)
		if (
			candidate.role in {'div', 'span'}
			and node.tag_name == candidate.role
			and not node.has_js_click_listener
			and not (node.snapshot_node and node.snapshot_node.is_clickable)
			and not {'role', 'tabindex', 'aria-label', 'aria-labelledby'} & node.attributes.keys()
			and not (candidate.editable or candidate.states or candidate.risky_effect or candidate.is_submit)
		):
			wrappers[id(node)] = candidate
	for candidate in wrappers.values():
		assert candidate.index is not None
		node = nodes[candidate.index]
		parent = node.parent
		while parent is not None and (outer := wrappers.get(id(parent))) is not None:
			if outer.name == candidate.name and outer.role == candidate.role and _covers_click(node, parent):
				removed.add(outer.key)
			parent = parent.parent
	return [candidate for candidate in clicks if candidate.key not in removed]


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
	if obs.scroll_regions:
		payload.append([(r.key, r.scroll_top, r.scroll_height, r.client_height) for r in obs.scroll_regions])
	return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def scroll_effect(before: Observation | None, after: Observation, target: Candidate | None) -> ScrollEffect:
	effect = ScrollEffect(target_key=target.key if target else None)
	if before is None:
		return effect
	if before.url and after.url:
		effect.url_changed = before.url != after.url
	first, last = before.scroll_observation, after.scroll_observation
	if first is None or last is None or (first.target_id, first.document) != (last.target_id, last.document):
		return effect

	effect.root_delta_y = last.root_y - first.root_y
	if first.center_id is not None and last.center_id is not None:
		same_center = first.center_id == last.center_id
	else:
		same_center = first.center_path is not None and first.center_path == last.center_path
	if same_center and first.center_y is not None and last.center_y is not None:
		effect.center_delta_y = last.center_y - first.center_y
	if target is None:
		return effect

	if target.key == 'page' or target.index == 0:
		# A gesture can enter a frame or shadow tree whose internal scroll position the top-level read cannot see.
		center_measured = effect.center_delta_y is not None or first.center_y is None and last.center_y is None
		effect.measurable = center_measured and not (first.center_embedded or last.center_embedded)
	elif target.key.startswith('region:'):
		start = next((r.scroll_top for r in before.scroll_regions if r.key == target.key), None)
		end = next((r.scroll_top for r in after.scroll_regions if r.key == target.key), None)
		if start is not None and end is not None:
			effect.target_delta_y = end - start
			effect.measurable = True
	elif target.index is not None and (start_indexed := first.indexed.get(target.index)) is not None:
		end_indexed = next((point for point in last.indexed.values() if point[:3] == start_indexed[:3]), None)
		if end_indexed is not None:
			effect.target_delta_y = end_indexed[3] - start_indexed[3]
			effect.measurable = True
	return effect


async def observe(browser_session: BrowserSession, summary: BrowserStateSummary, settings: JevSettings, step: int) -> Observation:
	pipeline = settings.effective_candidate_pipeline
	rendered = summary.jev_viewport
	if rendered is None:
		if summary.state_error:
			rendered = {'text': '', 'links': [], 'viewport': {}, 'title': ''}
		elif pipeline == 'pr5':
			rendered = await read_visible_text(browser_session, include_regions=False)
		else:
			rendered = await read_visible_text(browser_session)
	# A newly focused target can have an empty cached URL before TargetInfoChanged arrives.
	page_url = summary.url or rendered.get('url', '')
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
	center_above, center_below = rendered.get('centerScroll') or (0, 0)
	if center_above or center_below:
		scroll.update(center_pixels_above=round(center_above), center_pixels_below=round(center_below))
	target_id = rendered.get('target_id')
	center_id = rendered.get('centerBackendNodeId')
	diagnostics = rendered.get('scrollDiagnostics') or {}
	root_overflow_y = diagnostics.get('rootOverflowY')
	center_embedded = diagnostics.get('centerEmbedded', False)
	center_y = diagnostics.get('centerY')
	suppress_page = (
		root_overflow_y in ('hidden', 'clip')
		and rendered.get('centerScroll') is None
		and center_y is None
		and not center_embedded
	)
	scroll_observation = None
	if target_id and (document := rendered.get('document')) is not None and (position := rendered.get('scroll')):
		scroll_observation = ScrollObservation(
			target_id=target_id,
			document=document,
			root_y=position[1],
			root_overflow_y=root_overflow_y,
			center_id=center_id if diagnostics.get('centerMatchesCandidate') else None,
			center_path=diagnostics.get('centerPath'),
			center_y=center_y,
			center_embedded=center_embedded,
			page_suppressed=suppress_page and (above > 0 or below > 0),
		)
	if not suppress_page:
		if center_above or above > 0:
			candidates['SCROLL_UP'].append(page)
		if center_below or below > 0:
			candidates['SCROLL_DOWN'].append(page)
	regions: list[ScrollRegion] = []
	indexed_scrollables = {
		n.backend_node_id
		for n in summary.dom_state.selector_map.values()
		if n.target_id == target_id and n.is_actually_scrollable and _visible(n, width, height)
	}
	for region in rendered.get('regions', []) if pipeline != 'pr5' else []:
		backend_id = region.get('backendNodeId')
		if not backend_id or backend_id in indexed_scrollables or not target_id:
			continue
		key = f'region:{backend_id}'
		record = ScrollRegion(
			key=key,
			backend_node_id=backend_id,
			target_id=target_id,
			document=rendered['document'],
			scroll_top=region['scrollTop'],
			scroll_height=region['scrollHeight'],
			client_height=region['clientHeight'],
		)
		regions.append(record)
		candidate = Candidate(key=key, kind='scroll', role='region', name=region['name'], rect=region['rect'])
		if record.scroll_top > 0:
			candidates['SCROLL_UP'].append(candidate)
		if record.scroll_top + record.client_height < record.scroll_height:
			candidates['SCROLL_DOWN'].append(candidate)
	# Off preserves the PR5 candidate list when the page adds no region.
	keep_legacy_scrolls = pipeline == 'pr5' or (pipeline == 'pr5_with_regions' and not regions)
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
		center_represented = node.target_id == target_id and node.backend_node_id == center_id
		if node.is_actually_scrollable:
			# An iframe candidate scrolls its inner document. Its element's own scrollTop is unrelated.
			if scroll_observation and node.tag_name.lower() != 'iframe' and node.snapshot_node and node.snapshot_node.scrollRects:
				scroll_observation.indexed[index] = (
					node.target_id,
					node.frame_id,
					node.backend_node_id,
					node.snapshot_node.scrollRects.y,
				)
			if keep_legacy_scrolls or not center_represented:
				candidates['SCROLL_UP'].append(field)
				candidates['SCROLL_DOWN'].append(field)
	clicks = candidates['CLICK'] = _drop_click_wrappers(candidates['CLICK'], summary.dom_state.selector_map)
	names = Counter((field.role, field.name) for field in clicks if field.name)
	for field in clicks:
		if field.index is not None and names[field.role, field.name] > 1:
			if text := _nearby_text(summary.dom_state.selector_map[field.index], field.name):
				field.context['nearby_text'] = text
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
		scroll_regions=regions,
		scroll_observation=scroll_observation,
		tabs=tabs,
		can_go_back=go_back,
		is_pdf=summary.is_pdf_viewer,
		state_error=summary.state_error,
		unselectable_options=unselectable,
		fingerprint='',
	)
	obs.fingerprint = fingerprint(obs)
	return obs
