"""Validate the document and the model-visible target immediately before a benchmark action."""

import asyncio
from collections.abc import Iterator
from contextlib import suppress
from typing import Literal, TypedDict
from urllib.parse import urldefrag

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import EnhancedDOMTreeNode, NodeType

from .tasks import EnvironmentBlocked

STALE_FORCE_AFTER = 2
GUARD_ATTRS = (
	'id',
	'name',
	'type',
	'role',
	'aria-label',
	'title',
	'placeholder',
	'href',
	'aria-checked',
	'aria-pressed',
	'aria-selected',
	'aria-expanded',
	'disabled',
	'aria-disabled',
)


class DocumentGuard(TypedDict):
	target_id: str
	document_id: float
	url: str


class TargetGuard(TypedDict):
	target_id: str
	session_id: str | None
	backend_node_id: int
	tag: str
	attrs: dict[str, str]
	text: str
	checked: bool | str | None


async def read_document(browser: BrowserSession) -> DocumentGuard:
	"""Sample the same document identity used by the inspector without serializing the page."""
	session = await browser.get_or_create_cdp_session(focus=False)
	reply = await session.cdp_client.send.Runtime.evaluate(
		params={'expression': '[performance.timeOrigin, location.href]', 'returnByValue': True}, session_id=session.session_id
	)
	identity = reply['result'].get('value')
	if not isinstance(identity, list) or len(identity) != 2:
		raise RuntimeError('Browser document identity is unavailable')
	document_id, url = identity
	return DocumentGuard(target_id=session.target_id, document_id=document_id, url=url)


def same_document(left: DocumentGuard, right: DocumentGuard) -> bool:
	return (
		left['target_id'] == right['target_id']
		and left['document_id'] == right['document_id']
		and urldefrag(left['url']).url == urldefrag(right['url']).url
	)


def _text_parts(node: EnhancedDOMTreeNode) -> Iterator[str]:
	if node.node_type == NodeType.TEXT_NODE:
		yield node.node_value
		return
	for child in node.children:
		yield from _text_parts(child)
	for root in node.shadow_roots or ():
		if root.shadow_root_type == 'open':
			yield from _text_parts(root)


def target_guard(node: EnhancedDOMTreeNode) -> TargetGuard:
	attrs = {name: node.attributes[name] for name in GUARD_ATTRS if name in node.attributes}
	checked = None
	if node.tag_name == 'input' and attrs.get('type', '').lower() in {'checkbox', 'radio'} and node.ax_node:
		for prop in node.ax_node.properties or ():
			if prop.name == 'checked':
				checked = {'true': True, 'false': False}.get(str(prop.value).lower(), prop.value)
				break
	return TargetGuard(
		target_id=node.target_id,
		session_id=node.session_id,
		backend_node_id=node.backend_node_id,
		tag=node.tag_name,
		attrs=attrs,
		text=' '.join(' '.join(_text_parts(node)).split())[:40],
		checked=checked,
	)


_READ_TARGET = """function(names) {
	const parts = [];
	function collect(node) {
		if (node.nodeType === Node.TEXT_NODE) { parts.push(node.nodeValue); return; }
		for (const child of node.childNodes) collect(child);
		if (node.shadowRoot) collect(node.shadowRoot);
	}
	if (this.isConnected) collect(this);
	const attrs = Object.fromEntries(names.filter(name => this.hasAttribute(name)).map(name => [name, this.getAttribute(name)]));
	const toggle = this.localName === 'input' && ['checkbox', 'radio'].includes(this.type);
	return {connected: this.isConnected, tag: this.localName.toLowerCase(), attrs, text: parts.join(' '),
			checked: toggle ? (this.indeterminate ? 'mixed' : this.checked) : null};
}"""


async def _check_target(browser: BrowserSession, guard: TargetGuard) -> str | None:
	session = await browser.get_or_create_cdp_session(guard['target_id'], focus=False)
	if guard['session_id'] is not None and session.session_id != guard['session_id']:
		return 'target_unverifiable'
	cdp = session.cdp_client.send
	try:
		reply = await cdp.DOM.resolveNode(params={'backendNodeId': guard['backend_node_id']}, session_id=session.session_id)
	except EnvironmentBlocked:
		raise
	except RuntimeError as error:
		if any(marker in str(error).lower() for marker in ('no node with given id', 'could not find node', 'node is detached')):
			return 'target_detached'
		raise
	object_id = reply['object'].get('objectId')
	if not object_id:
		return 'target_unverifiable'
	try:
		result = await cdp.Runtime.callFunctionOn(
			params={
				'objectId': object_id,
				'functionDeclaration': _READ_TARGET,
				'arguments': [{'value': list(GUARD_ATTRS)}],
				'returnByValue': True,
			},
			session_id=session.session_id,
		)
		state = result['result'].get('value')
		if not isinstance(state, dict):
			return 'target_unverifiable'
	finally:
		with suppress(RuntimeError, TimeoutError):
			await cdp.Runtime.releaseObject(params={'objectId': object_id}, session_id=session.session_id)
	if not state['connected']:
		return 'target_detached'
	if state['tag'] != guard['tag'] or state['attrs'] != guard['attrs']:
		return 'target_changed'
	if guard['text'] and guard['text'] not in ' '.join(state['text'].split()):
		return 'target_text_changed'
	if guard['checked'] is not None and state['checked'] != guard['checked']:
		return 'target_state_changed'
	return None


async def decision_validity(
	browser: BrowserSession, guard: TargetGuard | None, decided: DocumentGuard, current: DocumentGuard
) -> str | None:
	if not same_document(decided, current):
		return 'document_changed'
	if guard is None:
		return None
	try:
		reason = await asyncio.wait_for(_check_target(browser, guard), timeout=5)
	except (EnvironmentBlocked, ConnectionError):
		raise
	except (TimeoutError, RuntimeError, KeyError, ValueError):
		reason = 'target_unverifiable'
	# A navigation can invalidate a target's CDP session while it is being read.
	return reason if same_document(decided, await read_document(browser)) else 'document_changed'


def stale_disposition(streak: int, reason: str | None) -> Literal['execute', 'ui_sync', 'retry', 'force']:
	if reason == 'ui_out_of_sync':
		return 'ui_sync'
	if reason is None or reason == 'target_unverifiable':
		return 'execute'
	return 'retry' if streak < STALE_FORCE_AFTER else 'force'
