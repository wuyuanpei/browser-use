"""Observe vertical scroll regions and revalidate the selected DOM node before a wheel event."""

from browser_use.agent.jev.views import SCROLL_PAGES, ScrollRegion
from browser_use.browser.session import BrowserSession

# Shared by the viewport read and the action's callFunctionOn, with no page-side state.
GEOMETRY_JS = r"""
	const visible = e => e.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
	const visibleRegion = e => e && e !== document.body && e !== document.documentElement &&
		e.scrollHeight > e.clientHeight + 2 && /^(auto|scroll)$/.test(getComputedStyle(e).overflowY) &&
		!e.closest('[aria-hidden="true"],[inert]') && visible(e);
	const regionRect = e => {
		const bounds = e.getBoundingClientRect();
		let left = Math.max(0, bounds.left), right = Math.min(innerWidth, bounds.right);
		let top = Math.max(0, bounds.top), bottom = Math.min(innerHeight, bounds.bottom);
		for (let parent = e.parentElement; parent && parent !== document.body; parent = parent.parentElement) {
			const style = getComputedStyle(parent), rect = parent.getBoundingClientRect();
			if (/auto|scroll|hidden|clip/.test(style.overflowX)) {
				left = Math.max(left, rect.left);
				right = Math.min(right, rect.right);
			}
			if (/auto|scroll|hidden|clip/.test(style.overflowY)) {
				top = Math.max(top, rect.top);
				bottom = Math.min(bottom, rect.bottom);
			}
		}
		return left < right && top < bottom ? {left, top, right, bottom} : null;
	};
	const scrollPoint = (e, rect) => {
		if (!rect) return null;
		for (const [fx, fy] of [[.5,.5], [.1,.1], [.9,.1], [.1,.9], [.9,.9]]) {
			const x = rect.left + (rect.right - rect.left) * fx, y = rect.top + (rect.bottom - rect.top) * fy;
			let hit = document.elementFromPoint(x, y);
			if (!e.contains(hit)) continue;
			while (hit && !visibleRegion(hit)) hit = hit.parentElement;
			if (hit === e) return {x, y};
		}
		return null;
	};
"""

OBSERVE_JS = (
	GEOMETRY_JS
	+ r"""
	const regionName = (e, rect) => {
		const label = e.getAttribute('aria-label')?.trim();
		if (label) return label;
		const heading = Array.from(e.querySelectorAll('h1,h2,h3,h4,h5,h6,[role=heading]')).find(visible);
		if (heading?.innerText?.trim()) return heading.innerText.trim();
		const text = document.createTreeWalker(e, NodeFilter.SHOW_TEXT);
		const range = document.createRange();
		let snippet = '', node;
		while (snippet.length < 60 && (node = text.nextNode())) {
			const part = node.textContent.replace(/\s+/g, ' ').trim();
			if (!part || !visible(node.parentElement)) continue;
			range.selectNodeContents(node);
			const inBox = Array.from(range.getClientRects()).some(r =>
				r.width > 0 && r.height > 0 && r.left < rect.right && r.right > rect.left && r.top < rect.bottom && r.bottom > rect.top);
			if (inBox) snippet += (snippet ? ' ' : '') + part;
		}
		return snippet.slice(0, 60) || 'Scrollable region';
	};
	const regions = [], regionNodes = [];
	for (const e of document.body?.querySelectorAll('*') || []) {
		if (e === centerElement || !visibleRegion(e)) continue;
		const rect = regionRect(e);
		if (!scrollPoint(e, rect)) continue;
		const x = Math.floor(rect.left), y = Math.floor(rect.top);
		regionNodes.push(e);
		regions.push({
			name: regionName(e, rect).slice(0, 120), scrollTop: Math.round(e.scrollTop),
			scrollHeight: e.scrollHeight, clientHeight: e.clientHeight,
			rect: {x, y, w: Math.ceil(rect.right)-x, h: Math.ceil(rect.bottom)-y}
		});
		if (regions.length === 12) break;
	}
"""
)


async def scroll(browser: BrowserSession, region: ScrollRegion, down: bool) -> bool:
	if browser.agent_focus_target_id != region.target_id:
		return False
	cdp = await browser.get_or_create_cdp_session(target_id=region.target_id, focus=False)
	resolved = await cdp.cdp_client.send.DOM.resolveNode(
		params={'backendNodeId': region.backend_node_id}, session_id=cdp.session_id
	)
	object_id = resolved['object'].get('objectId')
	if not object_id:
		return False
	try:
		# The captured node identity survives sibling reordering but never follows a replacement into a wheel action.
		fn = (
			"""function(origin, down, pages) {"""
			+ GEOMETRY_JS
			+ """
			if (performance.timeOrigin !== origin || !this.isConnected || !visibleRegion(this)) return null;
			const remaining = down ? this.scrollHeight - this.clientHeight - this.scrollTop : this.scrollTop;
			if (remaining <= 0) return null;
			const point = scrollPoint(this, regionRect(this));
			return point && {...point, scrollTop:this.scrollTop, deltaY: (down ? 1 : -1) * Math.min(remaining, Math.max(1, this.clientHeight * pages))};
		}"""
		)
		result = await cdp.cdp_client.send.Runtime.callFunctionOn(
			params={
				'objectId': object_id,
				'functionDeclaration': fn,
				'returnByValue': True,
				'arguments': [{'value': region.document}, {'value': down}, {'value': SCROLL_PAGES}],
			},
			session_id=cdp.session_id,
		)
		point = result['result'].get('value')
		if not point:
			return False
		await cdp.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseWheel', 'x': point['x'], 'y': point['y'], 'deltaX': 0, 'deltaY': point['deltaY']},
			session_id=cdp.session_id,
		)
		# CDP acknowledges a wheel before the compositor updates scrollTop. Let the next observation
		# see the movement, while leaving a cancelled wheel to the usual no_effect accounting.
		await cdp.cdp_client.send.Runtime.callFunctionOn(
			params={
				'objectId': object_id,
				'awaitPromise': True,
				'returnByValue': True,
				'arguments': [{'value': point['scrollTop']}],
				'functionDeclaration': """function(before) {
					if (this.scrollTop !== before || !this.isConnected) return;
					return new Promise(resolve => {
						let frame;
						const timeout = setTimeout(() => { cancelAnimationFrame(frame); resolve(); }, 200);
						const check = () => {
							if (this.isConnected && this.scrollTop === before) frame = requestAnimationFrame(check);
							else { clearTimeout(timeout); resolve(); }
						};
						check();
					});
				}""",
			},
			session_id=cdp.session_id,
		)

		return True
	finally:
		await cdp.cdp_client.send.Runtime.releaseObject(params={'objectId': object_id}, session_id=cdp.session_id)
