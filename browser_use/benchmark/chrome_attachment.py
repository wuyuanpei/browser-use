"""Attach the local benchmark to a Chrome debugging connection the user has approved."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import secrets
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import websockets
from websockets.protocol import State

from browser_use.browser._cdp_timeout import TimeoutWrappedCDPClient
from browser_use.browser.events import AgentFocusChangedEvent, TabCreatedEvent
from browser_use.browser.session import BrowserSession
from browser_use.browser.session_manager import SessionManager
from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
from browser_use.browser.watchdogs.dom_watchdog import DOMWatchdog
from browser_use.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog
from browser_use.utils import create_task_with_error_handling

from .engine import VIEWPORT, BrowserUnavailable, _BrowserLoop
from .tasks import EnvironmentBlocked

_EXCLUDED_TOOLS = [
	'evaluate',
	'read_file',
	'write_file',
	'replace_file',
	'upload_file',
	'send_keys',
	'save_as_pdf',
	'close',
]


class ChromeScopeViolation(EnvironmentBlocked):
	"""A requested action would leave the set of benchmark-owned web pages."""


def _read_loopback_endpoint(active_port: Path) -> str:
	try:
		port, path = active_port.read_text(encoding='utf-8').splitlines()
		if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535:
			raise ValueError
		if not re.fullmatch(r'/devtools/browser(?:/[0-9a-fA-F-]+)?', path):
			raise ValueError
	except (OSError, ValueError):
		raise BrowserUnavailable('本机 Chrome 官方调试端点不可用。') from None
	return f'ws://127.0.0.1:{port}{path}'


def _require_web_url(url: str, fixture_origin: str) -> None:
	if url in ('about:blank', ''):
		return
	try:
		parsed = urlsplit(url)
		host = (parsed.hostname or '').lower().rstrip('.')
		if parsed.scheme in ('http', 'https') and host and host != 'passwords.google.com' and parsed.username is None:
			is_private = host == 'localhost' or host.endswith('.localhost') or host.isdecimal()
			with suppress(ValueError):
				is_private = is_private or not ipaddress.ip_address(host).is_global
			if not is_private or (f'{parsed.scheme}://{parsed.netloc}' == fixture_origin and parsed.path == '/fixture.html'):
				return
	except ValueError:
		pass
	raise ChromeScopeViolation('附着模式只允许测试标签中的普通网页。')


class _OwnedClient(TimeoutWrappedCDPClient):
	"""Allow page operations only for tabs created by the currently running benchmark case."""

	def __init__(self, endpoint: str, fixture_origin: str):
		super().__init__(endpoint, cdp_request_timeout_s=10)
		self.pages: set[str] = set()
		self.sessions: dict[str, str] = {}
		self.fixture_origin = fixture_origin
		self.context_id: str | None = None
		self._closing_case = False

	async def start(self) -> None:
		logger = logging.getLogger(__name__ + '.transport')
		logger.setLevel(logging.WARNING)
		# Chrome can hold the upgrade while waiting for a click in its own approval dialog.
		self.ws = await websockets.connect(
			self.url, proxy=None, open_timeout=75, close_timeout=3, max_size=self.max_ws_frame_size, logger=logger
		)
		self._message_handler_task = asyncio.create_task(self._handle_messages())

	@property
	def connected(self) -> bool:
		return (
			self.ws is not None
			and self.ws.state is State.OPEN
			and self._message_handler_task is not None
			and not self._message_handler_task.done()
		)

	def check_connection(self) -> None:
		if not self.connected:
			raise BrowserUnavailable('Chrome 连接已断开。请重新批准连接后再运行。')

	async def _request(self, method: str, params: Any = None, session_id: str | None = None) -> dict[str, Any]:
		self.check_connection()
		try:
			return await super().send_raw(method, params, session_id)
		except (ConnectionError, OSError, websockets.exceptions.ConnectionClosed):
			raise BrowserUnavailable('Chrome 连接已断开。请重新批准连接后再运行。') from None

	async def new_page(self) -> str:
		if not self.context_id:
			raise BrowserUnavailable('尚未确定 Chrome 测试台资料。')
		try:
			reply = await self._request(
				'Target.createTarget', {'url': 'about:blank', 'background': True, 'browserContextId': self.context_id}
			)
		except Exception:
			raise BrowserUnavailable('无法在测试台所在的 Chrome 资料中创建标签。') from None
		target_id = reply['targetId']
		self.pages.add(target_id)
		if (await self.page_info(target_id)).get('browserContextId') != self.context_id:
			await self.close_case()
			raise BrowserUnavailable('新任务标签未落在测试台所在的 Chrome 资料中。')
		await self._request('Target.attachToTarget', {'targetId': target_id, 'flatten': True})
		return target_id

	async def bind_dashboard_context(self, dashboard_url: str) -> None:
		# Read target metadata only to match the open dashboard. Existing targets are never attached.
		contexts = {page.get('browserContextId') for page in await self._page_targets() if page.get('url') == dashboard_url}
		if len(contexts) != 1 or not (context_id := next(iter(contexts))):
			raise BrowserUnavailable('请在指定 Chrome 资料的普通窗口中打开测试台，再批准连接。')
		self.context_id = context_id

	async def _page_targets(self) -> list[dict[str, Any]]:
		info = await self._request('Target.getTargets', {'filter': [{'type': 'page'}, {'exclude': True}]})
		return info['targetInfos']

	async def recover_unclosed_pages(self, previous: _OwnedClient) -> None:
		if not previous.pages:
			return
		remaining = [page for page in await self._page_targets() if page['targetId'] in previous.pages]
		if remaining and (
			self.context_id != previous.context_id or any(page.get('browserContextId') != self.context_id for page in remaining)
		):
			raise BrowserUnavailable('上次的测试标签仍在原 Chrome 资料中。请在该资料中打开连接页。')
		self.pages.update(page['targetId'] for page in remaining)

	def claim_popup(self, target: dict[str, Any]) -> bool:
		target_id = target['targetId']
		if (
			target_id in self.pages
			or target.get('type') != 'page'
			or target.get('openerId') not in self.pages
			or target.get('browserContextId') != self.context_id
		):
			return False
		self.pages.add(target_id)
		return True

	async def attach_popup(self, target: dict[str, Any]) -> None:
		target_id = target['targetId']
		if self._closing_case or target_id not in self.pages:
			return
		try:
			self.require_url(target.get('url', ''))
			await self._request('Target.attachToTarget', {'targetId': target_id, 'flatten': True})
		except ChromeScopeViolation:
			await self._request('Target.closeTarget', {'targetId': target_id})

	def require_url(self, url: str) -> None:
		_require_web_url(url, self.fixture_origin)

	async def page_info(self, target_id: str) -> dict[str, Any]:
		info = (await self._request('Target.getTargetInfo', {'targetId': target_id}))['targetInfo']
		self.require_url(info.get('url', ''))
		return info

	async def close_case(self) -> None:
		self._closing_case = True
		try:
			attempted: set[str] = set()
			loop = asyncio.get_running_loop()
			while pending := self.pages - attempted:
				self.check_connection()
				# Keep openers registered until closure is verified so queued popups stay attributable.
				await asyncio.gather(*(self._close_page(target_id) for target_id in pending))
				attempted.update(pending)
				# A successful close request can precede the target's removal in Chrome.
				deadline = loop.time() + 2
				while not self.pages.isdisjoint(attempted):
					live = {page['targetId'] for page in await self._page_targets()}
					self.pages.difference_update(attempted - live)
					if self.pages.isdisjoint(attempted) or loop.time() >= deadline:
						break
					await asyncio.sleep(0.05)
			if self.pages:
				raise BrowserUnavailable('Chrome 未确认测试标签已关闭。请重新连接后再运行。')
		except BrowserUnavailable:
			raise
		except (RuntimeError, TimeoutError):
			raise BrowserUnavailable('无法确认测试标签已关闭。请重新连接 Chrome。') from None
		finally:
			self.sessions.clear()
			self._closing_case = False

	async def _close_page(self, target_id: str) -> None:
		# A user or site may have already closed the tab. Confirm with target discovery after all requests.
		with suppress(RuntimeError, TimeoutError):
			await self._request('Target.closeTarget', {'targetId': target_id})

	async def send_raw(self, method: str, params: Any = None, session_id: str | None = None) -> dict[str, Any]:
		self.check_connection()
		params = params or {}
		if method == 'Target.setDiscoverTargets':
			# SessionManager registers events itself. No discovery of other Chrome tabs is needed.
			return {}
		if method == 'Target.getTargets':
			infos = [page for page in await self._page_targets() if page['targetId'] in self.pages]
			for page in infos:
				self.require_url(page.get('url', ''))
			return {'targetInfos': infos}
		if method in ('Target.getTargetInfo', 'Target.activateTarget', 'Target.closeTarget'):
			target_id = params.get('targetId') or self.sessions.get(session_id or '')
			if target_id not in self.pages:
				raise ChromeScopeViolation('标签不属于当前测试。')
			if method != 'Target.closeTarget':
				await self.page_info(target_id)
			return await self._request(method, params, session_id)
		if not session_id or session_id not in self.sessions:
			raise ChromeScopeViolation('操作没有绑定到当前测试标签。')
		if method.startswith(('Browser.', 'Storage.', 'Target.')) and method != 'Target.setAutoAttach':
			raise ChromeScopeViolation('该协议操作不适用于已打开的 Chrome。')
		if method.startswith('Network.') and 'cookie' in method.lower():
			raise ChromeScopeViolation('附着模式不读取浏览器凭据。')
		if method in ('DOM.setFileInputFiles', 'Page.handleJavaScriptDialog'):
			raise ChromeScopeViolation('附着模式不接管本机文件选择或浏览器确认。')
		if method == 'Page.navigate':
			self.require_url(params.get('url', ''))
		if method.startswith(('DOM', 'Runtime.', 'Accessibility.', 'Input.')) or method == 'Page.captureScreenshot':
			await self.page_info(self.sessions[session_id])
		return await self._request(method, params, session_id)


class _OwnedSessionManager(SessionManager):
	def __init__(self, browser: BrowserSession, client: _OwnedClient):
		super().__init__(browser)
		self.client = client

	async def start_monitoring(self) -> None:
		await super().start_monitoring()

		def attached(event, parent_session_id=None):
			target = event['targetInfo']
			target_id = target['targetId']
			owned_frame = target['type'] == 'iframe' and parent_session_id in self.client.sessions
			if target_id not in self.client.pages and not owned_frame:
				return
			self.client.sessions[event['sessionId']] = target_id
			if target['type'] == 'page' and target_id in self.client.pages:
				create_task_with_error_handling(
					self.client._request('Emulation.setFocusEmulationEnabled', {'enabled': True}, event['sessionId']),
					name='benchmark_owned_focus_emulation',
					logger_instance=self.logger,
				)
			create_task_with_error_handling(
				self._handle_target_attached(event), name='benchmark_owned_attach', logger_instance=self.logger
			)

		def detached(event, _parent_session_id=None):
			session_id = event['sessionId']
			if self.client.sessions.pop(session_id, None) is None:
				return
			create_task_with_error_handling(
				self._handle_target_detached(event), name='benchmark_owned_detach', logger_instance=self.logger
			)

		def created(event, _session_id=None):
			target = event['targetInfo']
			if self.client.claim_popup(target):
				create_task_with_error_handling(
					self.client.attach_popup(target), name='benchmark_owned_popup', logger_instance=self.logger
				)

		self.client.register.Target.attachedToTarget(attached)
		self.client.register.Target.detachedFromTarget(detached)
		self.client.register.Target.targetCreated(created)
		await self.client._request('Target.setDiscoverTargets', {'discover': True, 'filter': [{'type': 'page'}]})

	async def _initialize_existing_targets(self) -> None:
		"""Each case attaches its own fresh tabs."""

	async def ensure_valid_focus(self, timeout: float = 5.0) -> bool:
		target = self.browser_session.agent_focus_target_id
		return target in self.client.pages and await self.validate_session(target)

	async def _recover_agent_focus(self, crashed_target_id: str) -> None:
		if self.client._closing_case or self.browser_session.agent_focus_target_id is not None:
			return
		for target in reversed(self.get_all_page_targets()):
			if target.target_id != crashed_target_id and target.target_id in self.client.pages:
				await self.browser_session.get_or_create_cdp_session(target.target_id, focus=True)
				return


class _AttachedBrowser(BrowserSession):
	"""Reuse the Browser Use DOM and action watchdogs on a scoped browser connection."""

	def __init__(self) -> None:
		super().__init__(
			cdp_url='ws://127.0.0.1',
			keep_alive=True,
			viewport=VIEWPORT,
			enable_default_extensions=False,
			permissions=[],
		)

	async def initialize_owned(self, client: _OwnedClient) -> None:
		self._cdp_client_root = client
		manager = _OwnedSessionManager(self, client)
		self.session_manager = manager
		await manager.start_monitoring()
		# The ordinary session enables browser-wide permissions, downloads, storage and dialog handlers.
		for watchdog_class in (DOMWatchdog, DefaultActionWatchdog, ScreenshotWatchdog):
			watchdog_class.model_rebuild()
			watchdog = watchdog_class(event_bus=self.event_bus, browser_session=self)
			watchdog.attach_to_session()
			if isinstance(watchdog, DOMWatchdog):
				self._dom_watchdog = watchdog
			elif isinstance(watchdog, ScreenshotWatchdog):
				self._screenshot_watchdog = watchdog
		self._watchdogs_attached = True

	async def _cdp_create_new_page(self, url: str = 'about:blank', background: bool = False, new_window: bool = False) -> str:
		if url != 'about:blank' or new_window:
			raise ChromeScopeViolation('只能创建当前测试的空白标签。')
		client = self._cdp_client_root
		assert isinstance(client, _OwnedClient)
		return await client.new_page()

	async def on_NavigateToUrlEvent(self, event) -> None:
		client = self._cdp_client_root
		assert isinstance(client, _OwnedClient)
		client.require_url(event.url)
		await super().on_NavigateToUrlEvent(event)

	async def get_tabs(self):
		tabs = await super().get_tabs()
		client = self._cdp_client_root
		assert isinstance(client, _OwnedClient)
		for tab in tabs:
			client.require_url(tab.url)
		return tabs


class ChromeAttachment:
	"""Keep one approved connection and use its event loop across serial benchmark attempts."""

	excluded_tools = _EXCLUDED_TOOLS

	def __init__(self, active_port: Path):
		self._active_port = active_port
		self.bridge = _BrowserLoop()
		self._client: _OwnedClient | None = None
		self._browser: _AttachedBrowser | None = None
		self._anchor = secrets.token_urlsafe(18)

	@property
	def connected(self) -> bool:
		return self._browser is not None and self._client is not None and self._client.connected

	def check_connection(self) -> None:
		if not self.connected:
			raise BrowserUnavailable('请先批准并连接指定的本机 Chrome。')

	def anchor_url(self, origin: str) -> str:
		return f'{origin}/?chrome_attachment={self._anchor}#baseline'

	def connect(self, origin: str) -> None:
		dashboard_url = self.anchor_url(origin)
		self.bridge.run(self._connect(origin, dashboard_url), timeout=100)

	async def _connect(self, origin: str, dashboard_url: str) -> None:
		if self.connected:
			return
		await self._disconnect()
		previous = self._client
		endpoint = _read_loopback_endpoint(self._active_port)
		client = _OwnedClient(endpoint, origin)
		check_url = f'{origin}/fixture.html?scenario=research'
		# BrowserSession identifies this as an external connection, while the actual endpoint stays in the client.
		browser = _AttachedBrowser()
		try:
			await client.start()
			await client.bind_dashboard_context(dashboard_url)
			if previous is not None:
				await client.recover_unclosed_pages(previous)
			self._client, self._browser = client, browser
			await browser.initialize_owned(client)
			await self.finish_case()
			await self.begin_case()
			await browser.navigate_to(check_url)
			target_id = browser.agent_focus_target_id
			if not target_id or (await client.page_info(target_id))['url'] != check_url:
				raise BrowserUnavailable('已连接 Chrome，但测试页验证失败。')
			await self.finish_case()
			self._anchor = secrets.token_urlsafe(18)
		except BaseException as error:
			if self._client is client:
				with suppress(BrowserUnavailable):
					await self._disconnect()
			else:
				await client.stop()
			if isinstance(error, (asyncio.CancelledError, BrowserUnavailable)):
				raise
			raise BrowserUnavailable('Chrome 未获批准或测试页验证失败。') from None

	async def begin_case(self) -> BrowserSession:
		self.check_connection()
		assert self._client is not None and self._browser is not None
		if self._client.pages:
			raise BrowserUnavailable('上一测试的 Chrome 标签尚未清理。请重新连接后再运行。')
		try:
			target_id = await self._client.new_page()
			await self._browser.get_or_create_cdp_session(target_id, focus=True)
			await self._browser.event_bus.dispatch(TabCreatedEvent(target_id=target_id, url='about:blank'))
			await self._browser.event_bus.dispatch(AgentFocusChangedEvent(target_id=target_id, url='about:blank'))
			return self._browser
		except BaseException:
			with suppress(BrowserUnavailable):
				await self.finish_case()
			raise

	async def finish_case(self) -> None:
		if self._browser is not None:
			self._browser.agent_focus_target_id = None
			self._browser._cached_browser_state_summary = None
		try:
			if self._client is not None:
				await self._client.close_case()
		finally:
			if self._browser is not None and self._browser.session_manager is not None:
				await self._browser.session_manager.clear()

	def disconnect(self) -> None:
		self.bridge.run(self._disconnect(), timeout=30)

	async def _disconnect(self) -> None:
		client, browser = self._client, self._browser
		try:
			if client is not None and client.connected:
				await self.finish_case()
		finally:
			self._browser = None
			try:
				if browser is not None:
					await browser.event_bus.stop(clear=True, timeout=3)
			finally:
				if client is not None:
					await client.stop()
				# Keep unconfirmed IDs in memory for the next explicitly approved connection.
				if client is None or not client.pages:
					self._client = None

	def close(self) -> None:
		try:
			self.disconnect()
		finally:
			self.bridge.close()
