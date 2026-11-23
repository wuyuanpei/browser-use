"""Project captured page data into the inspector and archive format."""

import hashlib
import json
from pathlib import Path
from typing import Any

READ_STATE = Path(__file__).with_name('snapshot.js').read_text(encoding='utf-8')
_INDEXED_OPERATIONS = {
	'click': 'CLICK',
	'switch_tab': 'CLICK',
	'fill': 'TYPE_TEXT',
	'select': 'SELECT',
	'scroll_region_up': 'SCROLL_UP',
	'scroll_region_down': 'SCROLL_DOWN',
}


def document_key(page: dict[str, Any]) -> tuple[str | None, Any]:
	"""A reload creates a new document even when its URL is unchanged."""
	marker = page.get('page_key') or page.get('marker') or []
	identity = page.get('document_id')
	return page.get('url'), identity if identity is not None else marker[0] if marker else None


def fingerprint(page: dict[str, Any]) -> str:
	fields = ('url', 'text', 'actions', 'scroll', 'content', 'links')
	content = {key: page[key] for key in fields if key in page}
	return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def indexed_elements(actions: list[dict[str, Any]], *, include_node_refs: bool = False) -> list[dict[str, Any]]:
	"""Number observed elements for display, coalescing operations on a single DOM node."""
	elements: dict[int, dict[str, Any]] = {}
	for action in actions:
		kind = action['kind']
		operation = _INDEXED_OPERATIONS.get(kind)
		if operation is None:
			continue
		node = action['node']
		if node not in elements:
			element = {k: action[k] for k in ('role', 'value', 'checked', 'selected', 'expanded', 'context') if k in action}
			element.update(index=str(len(elements) + 1), label=action['label'].split(' → ')[0], operations=[])
			if include_node_refs:
				element['node_ref'] = node
			elements[node] = element
		element = elements[node]
		if operation not in element['operations']:
			element['operations'].append(operation)
		if kind == 'select':
			options = element.setdefault('options', [])
			element['value'] = action.get('current_value', '')
			options.append(
				{'index': f'{element["index"]}:{len(options) + 1}', 'label': action['label'], 'value': action['value']}
			)
	return list(elements.values())
