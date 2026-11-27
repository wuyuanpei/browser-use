"""Loopback-only HTTP interface for the Browser Use benchmark."""

import argparse
import atexit
import hashlib
import json
import os
import secrets
import socket
import threading
from contextlib import suppress
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from browser_use.agent.jev.views import ignore_legacy_vision_escalation

from .baseline import BaselineConflict, BaselineManager
from .chrome_attachment import ChromeAttachment
from .engine import MAX_STEPS, BrowserUnavailable, BrowserUseRun, RunCancelled
from .models import (
	DEBUG_PROFILE,
	PROVIDER,
	load_environment,
	load_profile_environment,
	model_profiles,
	redact_secrets,
	runtime_settings,
)
from .observation import READ_STATE

ROOT = Path(__file__).parent
PORT = 8824
ORIGIN = f'http://127.0.0.1:{PORT}'
ARTIFACTS = Path.cwd() / 'artifacts' / 'baselines'
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
AGENT: BrowserUseRun | None = None
BASELINE: BaselineManager | None = None
CHROME: ChromeAttachment | None = None
FREE_PAUSE = threading.Event()


class InspectorServer(ThreadingHTTPServer):
	"""Keep one process on the loopback port, including Windows socket reuse semantics."""

	allow_reuse_address = os.name != 'nt'
	allow_reuse_port = False

	def server_bind(self):
		if os.name == 'nt':
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
		super().server_bind()


def baseline():
	global BASELINE
	if BASELINE is None:
		BASELINE = BaselineManager(
			ARTIFACTS,
			ORIGIN,
			run_factory=partial(BrowserUseRun, chrome=CHROME),
			browser_mode='attached_chrome' if CHROME is not None else 'isolated',
		)
	return BASELINE


def response_state():
	"""公开运行状态和脱敏配置，不返回后端保存的密钥。"""
	state = AGENT.snapshot() if AGENT else {'page': None, 'status': 'idle', 'history': [], 'decision': None}
	profiles = model_profiles()
	debug = profiles[0]
	return {
		**state,
		'model_profile': state.get('model_profile', DEBUG_PROFILE),
		'model_profile_default': DEBUG_PROFILE,
		'model_profiles': profiles,
		'jev_configured': bool(os.getenv('TYPESAFE_API_KEY', '').strip()),
		'server_runtime': {
			'process_id': os.getpid(),
			'observation_sha256': hashlib.sha256(READ_STATE.encode()).hexdigest(),
		},
		'text_model': state.get('text_model', debug['text']['model']),
		'laya_base_url': os.environ.get('LAYA_BASE_URL', 'http://127.0.0.1:8791').rstrip('/'),
		'text_model_base_url': state.get('text_model_base_url', debug['text']['base_url']),
		'max_steps': MAX_STEPS,
		'baseline_active': bool(BASELINE and BASELINE.active),
		'chrome_attachment': {
			'configured': CHROME is not None,
			'connected': bool(CHROME and CHROME.connected),
			'anchor_url': CHROME.anchor_url(ORIGIN) if CHROME and not CHROME.connected else None,
		},
		'system1_options': [
			{
				'provider': PROVIDER,
				'label': 'Browser Use',
				**debug['planner'],
				'configured': any(profile['configured'] for profile in profiles),
			}
		],
		'system1_default': PROVIDER,
		'planner_default': debug['planner'],
	}


def close_browser():
	global AGENT
	agent, AGENT = AGENT, None
	if agent is not None:
		agent.close()


def public_error(error):
	return redact_secrets(str(error))


def command(name, body):
	"""路由自由任务和基线命令，浏览器所有权不跨模式抢占。"""
	global AGENT
	if name in ('reset', 'baseline/start'):
		body = ignore_legacy_vision_escalation(body)
	if name in ('chrome/connect', 'chrome/disconnect'):
		if CHROME is None:
			raise ValueError('本机 Chrome 附着模式未配置。')
		if BASELINE and BASELINE.active:
			raise BaselineConflict('批次运行中不能更换 Chrome 连接。')
		connect = name == 'chrome/connect'
		if not connect or not CHROME.connected:
			# The attachment retains ownership of tabs for cleanup even after the free task ends.
			with suppress(BrowserUnavailable):
				close_browser()
		if connect:
			CHROME.connect(ORIGIN)
		else:
			CHROME.disconnect()
		return response_state()
	if name.startswith('baseline/'):
		manager = baseline()
		if name == 'baseline/start':
			if CHROME is not None:
				CHROME.check_connection()
			# 在开始时将运行配置固定到新批次上。
			return manager.start(
				body.get('task_ids'),
				body.get('repeats', 1),
				body.get('system1_provider'),
				body.get('execution_mode'),
				screenshot_mode=body.get('screenshot_mode', False),
				screenshot_policy=body.get('screenshot_policy'),
				candidate_pipeline=body.get('candidate_pipeline'),
				model_profile=body.get('model_profile'),
				before_start=close_browser,
			)
		if name == 'baseline/stop':
			return manager.stop()
		if name == 'baseline/save':
			return manager.save(body.get('run_id'))
		# 打开目录沿用 POST 的本地访问及令牌保护，不接受任意路径。
		if name == 'baseline/open-folder':
			return manager.open_folder(body.get('run_id'))
		if name == 'baseline/rename':
			return manager.rename(body.get('run_id'), body.get('name'))
		if name == 'baseline/delete':
			return manager.delete(body.get('run_id'))
		raise ValueError('Unknown baseline command')
	if BASELINE and BASELINE.active:
		raise BaselineConflict('Baseline owns the browser; stop it before running a free task')
	# 每个新的显式推进请求视为继续；正在进行的请求由独立暂停事件打断边界。
	FREE_PAUSE.clear()
	if name == 'reset':
		scenario = body.get('scenario', 'flights')
		if scenario not in {
			'travel',
			'research',
			'flights',
			'custom',
			'visual-cards',
			'visual-icons',
			'visual-cards-plain',
			'scroll-panel',
			'dense',
		}:
			raise ValueError('Unknown demo scenario')
		goal = body.get('goal', '').strip()
		if not goal or len(goal) > 2000:
			raise ValueError('Enter 1–2,000 characters')
		default_url = (
			'https://www.google.com/travel/flights?hl=en'
			if scenario == 'flights'
			else f'{ORIGIN}/fixture.html?scenario={scenario}'
			if scenario != 'custom'
			else ''
		)
		url = body.get('url', default_url)
		if not isinstance(url, str):
			raise ValueError('Enter a valid http:// or https:// website URL')
		url = url.strip()
		try:
			parsed = urlparse(url)
			valid_url = (
				0 < len(url) <= 2048
				and parsed.scheme in {'http', 'https'}
				and bool(parsed.hostname)
				and not any(char.isspace() or ord(char) < 32 for char in url)
			)
			parsed.port  # Reject malformed port numbers before replacing the current task.
		except ValueError:
			valid_url = False
		if not valid_url:
			raise ValueError('Enter a valid http:// or https:// website URL (up to 2,048 characters)')
		# 配置验证必须先于 close_browser，错误配置不抢占旧任务。
		settings = runtime_settings(
			body.get('system1_provider'),
			body.get('execution_mode'),
			body.get('screenshot_mode', False),
			screenshot_policy=body.get('screenshot_policy'),
			candidate_pipeline=body.get('candidate_pipeline'),
			model_profile=body.get('model_profile'),
		)
		if CHROME is not None:
			CHROME.check_connection()
		close_browser()
		AGENT = BrowserUseRun(
			url,
			goal,
			settings,
			scenario=scenario,
			should_cancel=lambda: 'paused' if FREE_PAUSE.is_set() else None,
			record_dir=(Path.cwd() / 'artifacts' / 'frames' if body.get('record') else None),
			chrome=CHROME,
		)
	else:
		if AGENT is None:
			raise ValueError('Start a demo first')
		AGENT.command(name, body)
	return response_state()


class Handler(BaseHTTPRequestHandler):
	def send(self, status, content, mime='application/json'):
		content = content if isinstance(content, bytes) else content.encode()
		self.send_response(status)
		self.send_header('Content-Type', mime)
		self.send_header('Content-Length', str(len(content)))
		self.send_header('Cache-Control', 'no-store')
		self.send_header('X-Content-Type-Options', 'nosniff')
		self.end_headers()
		self.wfile.write(content)

	def do_GET(self):
		if self.headers.get('Host') != f'127.0.0.1:{PORT}':
			return self.send(403, 'Forbidden', 'text/plain')
		path = urlparse(self.path).path
		if path.startswith('/api/baseline'):
			try:
				manager = baseline()
				query = parse_qs(urlparse(self.path).query)
				run_id = query.get('run_id', [None])[0]
				attempt_id = query.get('attempt_id', [None])[0]
				if path == '/api/baseline':
					result = manager.state()
				elif path == '/api/baseline/run':
					result = manager.read_run(run_id)
				elif path == '/api/baseline/attempt':
					# 历史记录优先展示当时配置，不用当前环境变量覆盖。
					snapshot = manager.read_attempt(run_id, attempt_id)
					result = {
						**snapshot,
						'text_model': snapshot.get('text_model', os.getenv('TEXT_MODEL', 'deepseek-chat')),
						'laya_base_url': os.getenv('LAYA_BASE_URL', 'http://127.0.0.1:8791'),
						'text_model_base_url': snapshot.get(
							'text_model_base_url', os.getenv('TEXT_MODEL_BASE_URL', 'https://api.deepseek.com/v1')
						),
						'max_steps': MAX_STEPS,
					}
				elif path == '/api/baseline/frame':
					return self.send(200, manager.frame(run_id, attempt_id, query.get('name', [None])[0]), 'image/jpeg')
				else:
					raise ValueError('Unknown baseline route')
				return self.send(200, json.dumps(result))
			except (ValueError, OSError) as error:
				return self.send(400, json.dumps({'error': public_error(error)}))
		if path == '/api/state':
			with LOCK:
				return self.send(200, json.dumps(response_state()))
		if path == '/api/jev/frame':
			try:
				agent = AGENT
				if agent is None:
					raise ValueError('No active Jev run')
				reference = parse_qs(urlparse(self.path).query).get('ref', [None])[0]
				return self.send(200, agent.request_image(reference), 'image/jpeg')
			except (ValueError, OSError) as error:
				return self.send(404, json.dumps({'error': public_error(error)}))
		if path == '/demo.mp4':
			video = ROOT / 'docs' / 'demo.mp4'
			if video.exists():
				return self.send(200, video.read_bytes(), 'video/mp4')
		files = {
			'/': ('index.html', 'text/html'),
			'/app.js': ('app.js', 'text/javascript'),
			'/style.css': ('style.css', 'text/css'),
			'/fixture.html': ('fixture.html', 'text/html'),
		}
		if path not in files:
			return self.send(404, 'Not found', 'text/plain')
		name, mime = files[path]
		content = (ROOT / 'static' / name).read_text(encoding='utf-8').replace('__TOKEN__', TOKEN)
		self.send(200, content, mime + '; charset=utf-8')

	def do_POST(self):
		if (
			self.headers.get('Host') != f'127.0.0.1:{PORT}'
			or self.headers.get('X-Demo-Token') != TOKEN
			or self.headers.get('Origin') not in (None, ORIGIN)
		):
			return self.send(403, json.dumps({'error': 'Local demo requests only'}))
		# 暂停不等待浏览器锁，正在等待 HTTP 的线程会在下一安全边界读取事件。
		if self.path == '/api/pause':
			FREE_PAUSE.set()
			return self.send(200, json.dumps({'paused': True}))
		if not LOCK.acquire(blocking=False):
			return self.send(409, json.dumps({'error': 'A browser step is already running'}))
		try:
			length = int(self.headers.get('Content-Length', '0'))
			if not 0 < length < 8192:
				raise ValueError('Invalid request size')
			body = json.loads(self.rfile.read(length))
			result = command(self.path.removeprefix('/api/'), body)
			self.send(200, json.dumps(result))
		except BaselineConflict as error:
			self.send(409, json.dumps({'error': public_error(error)}))
		except RunCancelled as error:
			# 主动暂停不是模型错误；返回已经保存的状态供用户检查和继续。
			if str(error) == 'paused':
				self.send(200, json.dumps(response_state()))
			else:
				self.send(400, json.dumps({'error': public_error(error)}))
		except (ValueError, RuntimeError, TimeoutError) as error:
			self.send(400, json.dumps({'error': public_error(error)}))
		except Exception:
			self.send(
				500,
				json.dumps({'error': 'Local demo failed; no automatic retry. Reset to recover.'}),
			)
		finally:
			LOCK.release()

	def log_message(self, *_args):
		pass


def main():
	global PORT, ORIGIN, ARTIFACTS, CHROME
	parser = argparse.ArgumentParser(description='Run the Browser Use benchmark inspector')
	parser.add_argument('--port', type=int, default=PORT, help='Loopback port. Use 0 for an available port.')
	parser.add_argument('--artifacts', type=Path, default=ARTIFACTS, help='Directory for benchmark batches.')
	parser.add_argument('--profile-env', type=Path, action='append', default=[], help='Reuse a Jev model configuration file.')
	parser.add_argument(
		'--chrome-active-port-file',
		type=Path,
		help='Use DevToolsActivePort from an already running local Chrome instead of launching a browser.',
	)
	args = parser.parse_args()
	load_environment()
	default_profile = os.getenv('BROWSER_USE_BENCHMARK_PROFILE_ENV')
	profile_paths = args.profile_env or ([Path(default_profile)] if default_profile else [])
	for path in profile_paths:
		load_profile_environment(path)
	os.environ.setdefault('ANONYMIZED_TELEMETRY', 'false')
	ARTIFACTS = args.artifacts
	if active_port := args.chrome_active_port_file or os.getenv('BROWSER_USE_BENCHMARK_CHROME_ACTIVE_PORT_FILE'):
		CHROME = ChromeAttachment(Path(active_port))
		atexit.register(CHROME.close)
	atexit.register(close_browser)
	with InspectorServer(('127.0.0.1', args.port), Handler) as server:
		PORT = server.server_address[1]
		ORIGIN = f'http://127.0.0.1:{PORT}'
		baseline()
		print(f'Browser Use benchmark: {ORIGIN}/#baseline', flush=True)
		try:
			server.serve_forever()
		except KeyboardInterrupt:
			pass


if __name__ == '__main__':
	main()
