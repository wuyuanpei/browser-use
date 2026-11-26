"""Filter physically unreachable click targets before the existing goal-ranked cap."""

import json
import logging
from time import perf_counter

from browser_use.agent.jev.decision import ranked_target_indices
from browser_use.agent.jev.views import Candidate, Observation
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import EnhancedDOMTreeNode, NodeType

logger = logging.getLogger(__name__)


# Deep serialization returns the backendNodeId of each actual element hit-tested here.
# Positions are only lookup hints; even if the document has changed, only verified IDs can be removed.
_MEASURE = r"""((paths, documentRoot, maximum, budget) => {
	const start=performance.now(), deadline=start+budget;
	const timeout=Symbol('budget');
	function checkTime() {
		if (performance.now() >= deadline) throw timeout;
	}
	const lists=new WeakMap();
	function locate(path) {
		let scope=document, el;
		for (const index of path) {
			if (!scope) return null;
			if (!lists.has(scope)) lists.set(scope,scope.querySelectorAll('*'));
			el=lists.get(scope)[index];
			scope=el?.shadowRoot;
		}
		return el;
	}
	function parentOf(el) {
		return el.assignedSlot || el.parentElement || el.getRootNode().host || null;
	}
	const clips = new WeakMap();
	const overflow = new Set(['hidden', 'auto', 'scroll', 'clip']);
	const unlimited = [-Infinity, -Infinity, Infinity, Infinity];
	function ancestorClip(parent) {
		checkTime();
		if (!parent) return unlimited;
		if (clips.has(parent)) return clips.get(parent);
		let bounds = ancestorClip(parentOf(parent));
		const style = getComputedStyle(parent);
		const x = overflow.has(style.overflowX), y = overflow.has(style.overflowY);
		if ((x || y) && parent !== document.scrollingElement && parent.getClientRects().length) {
			// The outer border box is conservative for transformed clip edges.
			const r = parent.getBoundingClientRect();
			let left=r.left, top=r.top, right=r.right, bottom=r.bottom;
			if (style.transform === 'none' && parent.offsetWidth && parent.offsetHeight) {
				const sx=r.width/parent.offsetWidth, sy=r.height/parent.offsetHeight;
				left+=parent.clientLeft*sx; top+=parent.clientTop*sy;
				right=left+parent.clientWidth*sx; bottom=top+parent.clientHeight*sy;
			}
			const margin = parseFloat(style.overflowClipMargin) || 0;
			bounds = [
				x ? Math.max(bounds[0], left-margin) : bounds[0],
				y ? Math.max(bounds[1], top-margin) : bounds[1],
				x ? Math.min(bounds[2], right+margin) : bounds[2],
				y ? Math.min(bounds[3], bottom+margin) : bounds[3],
			];
		}
		clips.set(parent, bounds);
		return bounds;
	}
	function exposed(el, left, top, right, bottom) {
		left=Math.max(0,left); top=Math.max(0,top);
		right=Math.min(innerWidth,right); bottom=Math.min(innerHeight,bottom);
		if (right<=left || bottom<=top) return null;
		const dx=Math.min(2,(right-left)/4), dy=Math.min(2,(bottom-top)/4);
		const points=[[(left+right)/2,(top+bottom)/2], [left+dx,top+dy], [right-dx,top+dy],
			[left+dx,bottom-dy], [right-dx,bottom-dy]];
		let unknown=false;
		for (const [x,y] of points) {
			checkTime();
			let hit=document.elementFromPoint(x,y);
			while (hit && hit.shadowRoot) {
				const next=hit.shadowRoot.elementFromPoint(x,y);
				if (!next || next===hit) break;
				hit=next;
			}
			if (!hit) unknown=true;
			for (let n=hit;n;n=parentOf(n)) if (n===el) return true;
		}
		return unknown ? null : false;
	}
	function classify(el) {
		if (!el || el.getClientRects().length !== 1) return null;
		const r=el.getBoundingClientRect();
		const c=ancestorClip(parentOf(el));
		const left=Math.max(r.left,c[0]), top=Math.max(r.top,c[1]), right=Math.min(r.right,c[2]), bottom=Math.min(r.bottom,c[3]);
		const clipped=right<=left || bottom<=top;
		// Positioned descendants can escape a clip. Keep the original five-point check for that box as well.
		const visible=clipped ? exposed(el,r.left,r.top,r.right,r.bottom) : exposed(el,left,top,right,bottom);
		if (visible===null) return null;
		return visible ? 'reachable' : clipped ? 'clipped' : 'occluded';
	}
	const results=[];
	let reachable=0, exhausted=false;
	try {
		for (let i=0;i<paths.length;i++) {
			checkTime();
			const el=locate(paths[i]);
			const result=classify(el);
			checkTime();
			if (result===null) continue;
			results.push([i,result,el]);
			if (result==='reachable' && ++reachable===maximum) break;
		}
	} catch (error) {
		if (error!==timeout) throw error;
		exhausted=true;
	}
	return [results,documentRoot ? document : document.documentElement,exhausted,performance.now()-start];
})"""


def _lookup_hints(root: EnhancedDOMTreeNode, by_node: dict[int, str]) -> dict[str, list[int]]:
	"""Light-DOM ordinals in each open shadow scope. Keys are verified against Chrome IDs after measurement."""
	paths = {}

	def visit(scope: EnhancedDOMTreeNode, prefix: list[int]) -> None:
		ordinal = 0
		stack = [scope]
		while stack:
			node = stack.pop()
			if node.node_type == NodeType.ELEMENT_NODE:
				if (key := by_node.get(id(node))) is not None:
					paths[key] = [*prefix, ordinal]
				for shadow in node.shadow_roots or ():
					if shadow.shadow_root_type == 'open':
						visit(shadow, [*prefix, ordinal])
				ordinal += 1
			stack.extend(reversed(node.children))

	visit(root, [])
	return paths


async def prune_click_targets(
	browser: BrowserSession,
	summary: BrowserStateSummary,
	obs: Observation,
	maximum: int,
	*,
	enabled: bool = True,
	budget_ms: float = 300,
	goal_terms: set[str] | None = None,
	audit: dict[str, str] | None = None,
) -> tuple[list[Candidate], dict]:
	clicks = obs.candidates['CLICK']
	stats = dict(
		before=len(clicks),
		after_pruning=len(clicks),
		clipped=0,
		occluded=0,
		elapsed_ms=0.0,
		checked=0,
		budget_exhausted=False,
		js_ms=0.0,
	)
	if not enabled or len(clicks) <= maximum:
		return clicks, stats
	if budget_ms == 0:
		stats['budget_exhausted'] = True
		return clicks, stats
	if not summary.dom_state._root:
		return clicks, stats
	start = perf_counter()
	try:
		root = summary.dom_state._root.original_node
		while root.parent:
			root = root.parent
		if root.target_id != browser.agent_focus_target_id or (
			root.node_type != NodeType.DOCUMENT_NODE and root.tag_name != 'html'
		):
			return clicks, stats
		nodes = summary.dom_state.selector_map
		by_node = {id(nodes[c.index]): c.key for c in clicks if c.index is not None and c.index in nodes}
		paths = _lookup_hints(root, by_node)
		ordered = [clicks[i] for i in ranked_target_indices(clicks, goal_terms or set()) if clicks[i].key in paths]
		if not ordered:
			return clicks, stats
		arguments = ','.join(
			(
				json.dumps([paths[c.key] for c in ordered]),
				json.dumps(root.node_type == NodeType.DOCUMENT_NODE),
				str(maximum),
				json.dumps(budget_ms, allow_nan=False),
			)
		)
		cdp = await browser.get_or_create_cdp_session(root.target_id, focus=False)
		result = await cdp.cdp_client.send.Runtime.evaluate(
			params={
				'expression': f'{_MEASURE}({arguments})',
				'serializationOptions': {'serialization': 'deep', 'maxDepth': 4, 'additionalParameters': {'maxNodeDepth': 0}},
			},
			session_id=cdp.session_id,
		)
		remote = result['result']
		try:
			groups = remote.get('deepSerializedValue', {}).get('value', [])
			if 'exceptionDetails' in result or len(groups) != 4:
				return clicks, stats
			rows, document, exhausted, elapsed = groups
			stats.update(budget_exhausted=exhausted['value'], js_ms=round(elapsed['value'], 3))
			if document.get('value', {}).get('backendNodeId') != root.backend_node_id:
				return clicks, stats
			removed: dict[str, str] = {}
			for row in rows.get('value', []):
				index, reason, live_node = row['value']
				candidate = ordered[index['value']]
				assert candidate.index is not None
				if live_node.get('value', {}).get('backendNodeId') != nodes[candidate.index].backend_node_id:
					continue
				stats['checked'] += 1
				if reason['value'] in ('clipped', 'occluded'):
					removed[candidate.key] = reason['value']
		finally:
			if object_id := remote.get('objectId'):
				await cdp.cdp_client.send.Runtime.releaseObject(params={'objectId': object_id}, session_id=cdp.session_id)
		if audit is not None:
			audit.update(removed)
		filtered = [c for c in clicks if c.key not in removed]
		stats.update(
			after_pruning=len(filtered),
			clipped=sum(r == 'clipped' for r in removed.values()),
			occluded=sum(r == 'occluded' for r in removed.values()),
		)
		return filtered, stats
	except Exception as exc:
		logger.debug('Click geometry unavailable; retaining the observed candidates: %s', exc)
		return clicks, stats
	finally:
		stats['elapsed_ms'] = round((perf_counter() - start) * 1000, 2)
