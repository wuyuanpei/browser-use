"""Real localhost pages and a scripted HTTP stand-in for Jev's choice endpoint."""

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.browser.profile import ViewportSize


@pytest.fixture(name='httpserver')
def jev_httpserver():
	"""Serve browser pages and System1 POSTs concurrently despite Chromium's idle sockets."""
	with HTTPServer(host='localhost', port=0, threaded=True) as server:
		yield server


@pytest.fixture(scope='module', name='jev_browser')
async def jev_browser():
	"""An isolated browser without downloaded extensions or external model services."""
	session = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=True,
			enable_default_extensions=False,
			viewport=ViewportSize(width=1280, height=720),
			executable_path=os.environ.get('JEV_TEST_CHROME_PATH'),
		)
	)
	try:
		await session.start()
		yield session
	finally:
		await session.kill()


async def browser_eval(session: BrowserSession, expression: str) -> Any:
	cdp = await session.get_or_create_cdp_session()
	res = await cdp.cdp_client.send.Runtime.evaluate(
		params={'expression': expression, 'returnByValue': True, 'awaitPromise': True}, session_id=cdp.session_id
	)
	assert 'exceptionDetails' not in res, res.get('exceptionDetails')
	return res['result'].get('value')


TALL_PAGE = """<style>body{height:3000px}</style>
<button>Top first</button><button>Top second</button><button>Top third</button>
<button style="position:absolute;top:2000px">Bottom button</button>"""

VISIBILITY_PAGE = """<p>Visible paragraph</p><p hidden>Display hidden</p>
<p style="visibility:hidden">Visibility hidden</p><p style="position:absolute;top:3000px">Offscreen paragraph</p>
<iframe srcdoc="<p>Nested frame paragraph</p>"></iframe><button disabled>Disabled button</button>
<label>Password <input type="password"></label>"""

SELECT_PAGE = """<label>City <select id="city"><option>Paris</option><optgroup label="Other cities">
<option selected>London</option><option>Seattle</option><option disabled>Closed</option></optgroup></select></label>
<label>Duplicate <select id="duplicate"><option value="1">Other</option><option value="2">Other</option></select></label>
<label>Reverting <select id="reverting" onchange="window.changedAt=performance.now(); setTimeout(() => { this.selectedIndex = 0; window.revertedAt=performance.now(); this.dispatchEvent(new Event('reverted')); }, 100)">
<option value="first">First</option><option value="second">Second</option></select></label>"""

FORM_PAGE = """<form aria-label="Profile" onsubmit="event.preventDefault()"><fieldset><legend>Traveler details</legend>
<label for="name">Full name</label><input id="name" required placeholder="Enter name">
<button>Continue</button><button type="button">Preview</button><input type="submit" value="Submit"></fieldset></form>"""

TABLE_PAGE = """<table><thead><tr><th>Name</th><th>Action</th></tr></thead><tbody>
<tr><td>Alpha</td><td><button>Edit</button></td></tr><tr><td>Beta</td><td><button>Edit</button></td></tr>
<tr><td>Gamma</td><td><button>Edit</button></td></tr></tbody></table>"""

SCROLL_BOX_PAGE = """<div tabindex="0" aria-label="Results" style="height:100px;overflow:auto;">
<div style="height:800px;">Scrollable results</div></div>"""

DIALOG_PAGE = """<h2>Accounts</h2><button>Outside</button><div role="dialog" aria-label="Preferences">
<h3>Notifications</h3><button>Apply</button><button>Cancel</button></div>"""

AUTOCOMPLETE_PAGE = """<label>Where from?<input id="origin" oninput="setTimeout(() => choices.hidden=false,300)"></label>
<div id="choices" hidden><button onclick="originField.value=this.textContent; choices.hidden=true">Zurich (ZRH)</button></div>
<script>const originField=document.querySelector('#origin');</script>"""

SEARCH_FORM_PAGE = """<form onsubmit="event.preventDefault(); result.textContent='Results for '+query.value">
<label>Search query <input id="query"></label><button>Search</button></form><p id="result"></p>"""

CHAT_PAGE = """<label>Message <input id="message"></label><button id="send">Send</button><p id="messages"></p>
<script>async function deliver(){await fetch('/api/messages',{method:'POST',body:message.value});
messages.textContent=message.value;}send.onclick=deliver;message.onkeydown=e=>{if(e.key==='Enter')deliver();};</script>"""

COMPOSE_PAGE = """<form id="compose"><label>To <input name="to"></label><label>Subject <input name="subject"></label>
<label>Body <textarea name="body" id="bodyField"></textarea></label><button>Send</button></form><p id="result"></p>
<script>compose.onsubmit=async e=>{e.preventDefault();await fetch('/api/send',{method:'POST',body:new FormData(compose)});
result.textContent='Sent';};bodyField.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();compose.requestSubmit();}};</script>"""

DATE_PICKER_PAGE = """<button id="pick" onclick="calendar.hidden=false">Choose date</button><div id="calendar" hidden>
<button onclick="pick.textContent='June 18'">18</button><button onclick="calendar.hidden=true">Done</button></div>"""

REVEAL_PAGE = """<label>Type <select onchange="extra.hidden=this.value!=='other'">
<option value="">Choose</option><option value="other">Other</option></select></label>
<label id="extra" hidden>Details <input></label>"""


def serve_page(server: HTTPServer, path: str, body: str, title: str = 'Jev test') -> str:
	html = f'<!doctype html><html><head><meta charset="utf-8"><link rel="icon" href="data:,"><title>{title}</title></head><body>{body}</body></html>'
	server.expect_request(path, method='GET').respond_with_data(html, content_type='text/html; charset=utf-8')
	return server.url_for(path)


@dataclass
class TwoPageSite:
	first: str
	second: str
	save_requests: list[str] = field(default_factory=list)


def serve_two_page_site(server: HTTPServer) -> TwoPageSite:
	first = serve_page(server, '/first', '<a href="/second">Go to second page</a>', 'First page')
	second = serve_page(
		server,
		'/second',
		"""<button onclick="save()">Save</button><p id="status"></p>
<script>async function save(){const r=await fetch('/api/save',{method:'POST'});if(r.ok)document.querySelector('#status').textContent='Saved';}</script>""",
		'Second page',
	)
	site = TwoPageSite(first, second)

	def saved(request: Request) -> Response:
		site.save_requests.append(request.get_data(as_text=True))
		return Response(status=204)

	server.expect_request('/api/save', method='POST').respond_with_handler(saved)
	return site


@dataclass
class ShopSite:
	first: str
	cart: str
	cart_requests: list[str] = field(default_factory=list)
	orders: list[str] = field(default_factory=list)


def serve_shop_site(server: HTTPServer) -> ShopSite:
	products = ('Apple', 'Pear', 'Plum')
	catalog = ''.join(f'<p><a href="/product/{p}">{p} details</a></p>' for p in products)
	first = serve_page(server, '/shop', f'<main style="min-height:130vh">{catalog}<a href="/cart">Cart</a></main>', 'Shop')
	site = ShopSite(first, server.url_for('/cart'))
	for product in products:
		serve_page(
			server,
			f'/product/{product}',
			f"""<h1>{product}</h1><button onclick="fetch('/api/cart',{{method:'POST',body:'{product}'}}).then(()=>document.querySelector('output').textContent='Added')">Add to cart</button>
<output></output><a href="/shop">Back to list</a><a href="/cart">Cart</a>""",
			product,
		)

	def add(request: Request) -> Response:
		site.cart_requests.append(request.get_data(as_text=True))
		return Response(status=204)

	def order(request: Request) -> Response:
		site.orders.append(request.get_data(as_text=True))
		return Response(status=204)

	def cart(_request: Request) -> Response:
		return Response(
			f"""<title>Cart</title><h1>Cart</h1><p>{', '.join(site.cart_requests)}</p>
<button onclick="fetch('/api/order',{{method:'POST'}}).then(()=>this.textContent='Ordered')">Place order</button>""",
			content_type='text/html',
		)

	server.expect_request('/api/cart', method='POST').respond_with_handler(add)
	server.expect_request('/api/order', method='POST').respond_with_handler(order)
	server.expect_request('/cart', method='GET').respond_with_handler(cart)
	return site


def serve_newtab_site(server: HTTPServer) -> str:
	result = serve_page(server, '/tab-result', '<h1>Result</h1>', 'Result')
	cross = result.replace('localhost', '127.0.0.1')
	return serve_page(
		server,
		'/tabs',
		f"""<a href="/tab-result" target="_blank">Open result</a>
<a href="{cross}" target="_blank">Other site</a>
<button onclick="setTimeout(()=>window.open('/tab-result'),500)">Later</button>""",
		'List',
	)


def _distribute(keys: list[str], choice: str, p: float, unable_p: float | None) -> dict[str, float]:
	if choice not in keys or not 0 <= p <= 1:
		raise ValueError(f'Unexpected choice or probability: {choice!r}, {p}')
	if len(keys) == 1:
		return {choice: 1.0}
	probs = {choice: p}
	if unable_p is not None and 'UNABLE' in keys and choice != 'UNABLE':
		probs['UNABLE'] = unable_p
	others = [key for key in keys if key not in probs]
	remaining = 1.0 - sum(probs.values())
	if not others and abs(remaining) > 1e-8:
		raise ValueError('Scripted probabilities do not sum to one')
	probs.update({key: remaining / len(others) for key in others})
	if any(v < -1e-8 or v > p + 1e-8 for v in probs.values()):
		raise ValueError(f'Scripted choice {choice!r} does not have the largest probability')
	return probs


class FakeJev:
	"""Scripted HTTP responses using names or contextual fields from actual offered candidates."""

	def __init__(self, httpserver: HTTPServer, steps: list[dict[str, Any]]):
		self.steps = list(steps)
		self.requests: list[dict[str, Any]] = []
		self.errors: list[str] = []
		httpserver.expect_request('/v1/systemone', method='POST').respond_with_handler(self._handle)

	@staticmethod
	def _find(criteria: dict[str, Any], target: str | dict[str, str]) -> str:
		if isinstance(target, str):
			matches = [k for k, v in criteria.items() if target in (v.get('element', '') if isinstance(v, dict) else str(v))]
		else:
			matches = [
				k
				for k, v in criteria.items()
				if isinstance(v, dict)
				and all(
					str(expected) in str(v.get(field, v.get('context', {}).get(field, ''))) for field, expected in target.items()
				)
			]
		if len(matches) != 1:
			raise LookupError(f'Target {target!r} matched {matches}')
		return matches[0]

	def _handle(self, request: Request) -> Response:
		body = request.get_json()
		self.requests.append(body)
		step: dict[str, Any] = self.steps.pop(0) if self.steps else {'operation': 'UNABLE'}
		operation = step['operation']
		answers = {}
		try:
			for name, question in body['questions'].items():
				keys = list(question['criteria'])
				if name == 'operation':
					choice, p = operation, step.get('p', 0.95)
				elif name == operation.lower() + '_target':
					choice, p = self._find(question['criteria'], step['target']), step.get('target_p', 0.95)
				elif name == 'click_target' and 'focus_target' in step:
					choice, p = self._find(question['criteria'], step['focus_target']), step.get('focus_target_p', 0.95)
				else:
					choice, p = keys[0], 0.95
				probabilities = step.get('operation_probs') if name == 'operation' else None
				answers[name] = {
					'choice': choice,
					'confidence': step.get('confidence', p),
					'probabilities': {key: probabilities.get(key, 0.0) for key in keys}
					if probabilities is not None
					else _distribute(keys, choice, p, step.get('unable_p') if name == 'operation' else None),
				}
		except (LookupError, ValueError) as exc:
			self.errors.append(str(exc))
			return Response(str(exc), status=500)
		payload = {'model': 'jev-test', 'answers': answers, 'usage': {'input_tokens': 100, 'output_tokens': 10}}
		return Response(json.dumps(payload), content_type='application/json')
