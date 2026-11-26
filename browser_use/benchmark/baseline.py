"""Sequential, server-owned baseline runs and durable local evidence."""

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .archive import atomic_json
from .engine import MAX_STEPS, VIEWPORT, BrowserMode, BrowserUnavailable, BrowserUseRun, RunCancelled, RunEvent
from .evaluation import attempt_status, build_context, evaluate_frozen, freeze_context, outcome_statistics
from .models import DEBUG_PROFILE, FINAL_REPORT_PROFILE, PROVIDER, RunSettings, redact_secrets, runtime_settings
from .observation import indexed_elements
from .tasks import VERSION, CaseProbe, EnvironmentBlocked, task_catalog
from .token_usage import TOKEN_FIELDS, aggregate_token_usage, token_usage

TERMINAL = {'completed', 'cancelled', 'error', 'interrupted'}
CONFIRMED_SUCCESS = json.loads(Path(__file__).with_name('baseline_success.json').read_text(encoding='utf-8'))


class BaselineConflict(ValueError):
	pass


def now():
	return datetime.now().astimezone().isoformat(timespec='milliseconds')


class BaselineManager:
	def __init__(
		self,
		root: Path,
		origin: str,
		*,
		run_factory: Callable[..., BrowserUseRun] = BrowserUseRun,
		browser_mode: BrowserMode = 'isolated',
		timeout: int = 300,
	):
		self.root = Path(root).resolve()
		self.tasks = task_catalog(origin)
		self._run_factory = run_factory
		self._browser_mode = browser_mode
		self._settings: RunSettings | None = None
		self.timeout = timeout
		self.lock = threading.RLock()
		self.io_lock = threading.Lock()
		self.stop_event = threading.Event()
		self.active = False
		self.batch: dict[str, Any] | None = None
		self.snapshots: dict[tuple[str, str], dict[str, Any]] = {}
		self.token_cache = {}
		self.thread = None
		self.recover()

	def clean(self, data: Any) -> Any:
		settings = self._settings
		keys = (
			(settings.execution.api_key, settings.evaluation.api_key, settings.jev.api_key if settings.jev else None)
			if settings
			else ()
		)
		return redact_secrets(data, *keys)

	def folder(self, run_id):
		if not isinstance(run_id, str) or not re.fullmatch(r'\d{8}_\d{6}_\d{3}(?:_\d+)?', run_id):
			raise ValueError('Invalid run ID')
		folder = (self.root / run_id).resolve()
		if not folder.is_relative_to(self.root):
			raise ValueError('Invalid run path')
		return folder

	def recover(self):
		if not self.root.exists():
			return
		for path in sorted(self.root.glob('*/summary.json')):
			try:
				folder = self.folder(path.parent.name)
				batch = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
				if batch['status'] not in TERMINAL:
					batch.update(status='interrupted', reason='Server restarted; not resumed automatically')
					for row in batch['attempts']:
						if row['status'] in {'queued', 'running'}:
							row.update(status='interrupted', reason=batch['reason'])
							if row.get('execution_status') in {'queued', 'running'}:
								row.update(execution_status='interrupted', execution_reason=batch['reason'])
					self.write_summary(batch)
				self.batch = batch
			except (ValueError, OSError, KeyError, TypeError):
				continue

	def runs(self):
		rows = []
		for path in sorted(self.root.glob('*/summary.json'), reverse=True):
			try:
				batch = self.read_run(path.parent.name, include_tokens=False)
				rows.append({**{k: batch[k] for k in ('id', 'status', 'created_at')}, 'name': batch.get('name', '')})
			except (ValueError, OSError, KeyError):
				continue
		return rows

	def state(self):
		with self.lock:
			state = {
				'tasks': deepcopy(self.tasks),
				'active': self.active,
				'batch': deepcopy(self.batch),
				'confirmed_success': deepcopy(CONFIRMED_SUCCESS),
			}
		state['runs'] = self.runs()
		if state['batch']:
			self.add_token_usage(state['batch'])
			state['batch']['outcome_statistics'] = outcome_statistics(state['batch']['attempts'])
		return self.clean(state)

	def add_token_usage(self, batch):
		"""Backfill old summaries for display without rewriting historical evidence."""
		root = self.folder(batch['id'])
		for row in batch['attempts']:
			if 'token_usage' in row:
				continue
			path = (root / row['id'] / 'trace.json').resolve()
			if not path.is_relative_to(root):
				continue
			try:
				stat = path.stat()
				stamp = (stat.st_mtime_ns, stat.st_size)
				cached = self.token_cache.get(path)
				if cached is None or cached[0] != stamp:
					trace = json.loads(path.read_text(encoding='utf-8'))
					if not isinstance(trace, dict) or not isinstance(trace.get('model_calls'), list):
						continue
					cached = (stamp, token_usage(trace['model_calls']))
					self.token_cache[path] = cached
				row['token_usage'] = deepcopy(cached[1])
			except (OSError, ValueError, TypeError):
				continue
		batch['token_usage'] = aggregate_token_usage(batch['attempts'])
		batch['task_token_usage'] = {
			task: aggregate_token_usage([r for r in batch['attempts'] if r['task_id'] == task])
			for task in dict.fromkeys(r['task_id'] for r in batch['attempts'])
		}
		return batch

	def read_run(self, run_id, *, include_tokens=True):
		folder = self.folder(run_id)
		with self.lock:
			if self.batch and self.batch['id'] == run_id:
				batch = deepcopy(self.batch)
				batch['outcome_statistics'] = outcome_statistics(batch['attempts'])
				return self.clean(self.add_token_usage(batch) if include_tokens else batch)
		try:
			batch = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
			return self.add_token_usage(batch) if include_tokens else batch
		except FileNotFoundError:
			raise ValueError('Unknown run') from None

	def attempt_folder(self, run_id, attempt_id):
		batch = self.read_run(run_id, include_tokens=False)
		if not isinstance(attempt_id, str) or attempt_id not in {r['id'] for r in batch['attempts']}:
			raise ValueError('Unknown attempt')
		folder = (self.folder(run_id) / attempt_id).resolve()
		if not folder.is_relative_to(self.folder(run_id)):
			raise ValueError('Invalid attempt path')
		return folder

	def read_attempt(self, run_id, attempt_id):
		"""读取运行证据，缺失原始指令时只从同批次归档补充返回值。"""
		folder = self.attempt_folder(run_id, attempt_id)
		with self.lock:
			snapshot = deepcopy(self.snapshots.get((run_id, attempt_id)))
		if snapshot is None:
			path = folder / 'trace.json'
			snapshot = (
				json.loads(path.read_text(encoding='utf-8'))
				if path.exists()
				else {
					'page': None,
					'history': [],
					'model_calls': [],
					'status': 'queued',
				}
			)
		row = next(r for r in self.read_run(run_id)['attempts'] if r['id'] == attempt_id)
		snapshot['baseline_attempt'] = row
		snapshot['token_usage'] = token_usage(snapshot.get('model_calls', []))
		# 旧 trace 或尚未产生 trace 的任务只能回退到当时的 manifest。
		if not isinstance(snapshot.get('goal'), str) or not snapshot['goal'].strip():
			try:
				manifest = json.loads((self.folder(run_id) / 'manifest.json').read_text(encoding='utf-8'))
			except (OSError, ValueError):
				manifest = {}
			tasks = manifest.get('tasks', []) if isinstance(manifest, dict) else []
			# 损坏的可选归档不应妨碍其他证据展示，也不能用现行任务定义代替。
			if isinstance(tasks, list):
				archived_goal = next(
					(
						task['goal']
						for task in tasks
						if isinstance(task, dict)
						and task.get('id') == row['task_id']
						and isinstance(task.get('goal'), str)
						and task['goal'].strip()
					),
					None,
				)
				# 只补当前返回对象，原文件和内存中的原始快照均保持不变。
				if archived_goal is not None:
					snapshot['goal'] = archived_goal
		snapshot.setdefault('system1_provider', row.get('system1_provider', 'laya'))
		snapshot.setdefault('screenshot_mode', row.get('screenshot_mode', False))
		snapshot.setdefault('screenshot_policy', row.get('screenshot_policy', 'always' if snapshot['screenshot_mode'] else 'off'))
		# 老归档只在返回对象中补默认值，不迁移或覆写原始文件。
		snapshot.setdefault('execution_mode', row.get('execution_mode', 'direct'))
		snapshot.setdefault('model_profile', row.get('model_profile', DEBUG_PROFILE))
		for field in ('text_model', 'text_model_base_url'):
			if field in row:
				snapshot.setdefault(field, row[field])
		frames = snapshot.get('observations', [])
		snapshot['reevaluations'] = []
		for path in sorted((folder / 'reevaluations').glob('*/evaluation.json')):
			try:
				snapshot['reevaluations'].append(json.loads(path.read_text(encoding='utf-8')))
			except (OSError, ValueError):
				continue
		calls = {call['id']: call for call in snapshot.get('model_calls', [])}
		if any('page' not in frame for frame in frames) or any('observation_index' not in call for call in calls.values()):
			# Reconstruct old traces from event order, not repeated step numbers or URLs.
			events = folder / 'events.jsonl'
			index, laya_id = -1, None
			if events.exists():
				with events.open(encoding='utf-8') as stream:
					for line in stream:
						try:
							event = json.loads(line)
						except ValueError:
							continue  # A running worker may still be appending the last line.
						if event.get('event') == 'observed':
							index += 1
							laya_id = None
							if index < len(frames):
								frames[index].setdefault('page', event.get('page'))
						elif event.get('event') == 'model_start':
							call = calls.get(event.get('call', {}).get('id'))
							if call is not None:
								call.setdefault('observation_index', index if 0 <= index < len(frames) else None)
								if call['kind'] in {'laya', 'deepseek', 'jev'}:
									laya_id = call['id']
								else:
									call.setdefault('decision_call_id', laya_id)
									if calls.get(laya_id, {}).get('kind') == 'laya':
										call.setdefault('laya_call_id', laya_id)
		for frame in frames:
			if frame.get('page'):
				frame.setdefault('elements', indexed_elements(frame['page'].get('actions', [])))
		return self.clean(snapshot)

	def frame(self, run_id, attempt_id, name):
		if not isinstance(name, str) or not re.fullmatch(r'\d{6}\.jpg', name):
			raise ValueError('Invalid screenshot name')
		folder = self.attempt_folder(run_id, attempt_id)
		path = (folder / 'screenshots' / name).resolve()
		if not path.is_relative_to(folder) or not path.is_file():
			raise ValueError('Unknown screenshot')
		return path.read_bytes()

	def start(
		self,
		task_ids=None,
		repeats=1,
		system1_provider=None,
		execution_mode=None,
		*,
		screenshot_mode=False,
		screenshot_policy=None,
		model_profile=None,
		before_start=None,
	):
		"""验证配置后串行启动新批次，运行模式与凭据均冻结。"""
		settings = runtime_settings(
			system1_provider, execution_mode, screenshot_mode, screenshot_policy=screenshot_policy, model_profile=model_profile
		)
		config = settings.system1_config()
		profile, text = settings.profile, settings.evaluation
		text_endpoint = text.public()['base_url']
		ids = [t['id'] for t in self.tasks] if task_ids is None else task_ids
		allowed = {t['id'] for t in self.tasks}
		if (
			not isinstance(ids, list)
			or not ids
			or any(not isinstance(i, str) for i in ids)
			or len(set(ids)) != len(ids)
			or not set(ids) <= allowed
		):
			raise ValueError('Choose known task IDs without duplicates')
		if type(repeats) is not int or not 1 <= repeats <= 10:
			raise ValueError('Repeat count must be 1-10')
		tasks = [t for t in self.tasks if t['id'] in ids]
		with self.lock:
			if self.active:
				raise BaselineConflict('A baseline run is already active')
			self.root.mkdir(parents=True, exist_ok=True)
			run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
			candidate, suffix = run_id, 0
			while True:
				try:
					self.folder(candidate).mkdir()
					break
				except FileExistsError:
					suffix += 1
					candidate = f'{run_id}_{suffix}'
			run_id = candidate
			attempts = [
				dict(
					id=f'{t["id"]}/attempt-{n:02}',
					task_id=t['id'],
					name=t['name'],
					repeat=n,
					status='queued',
					actions=0,
					laya_calls=0,
					deepseek_calls=0,
					jev_calls=0,
					decision_calls=0,
					decision_rounds=0,
					system1_provider=config['provider'],
					text_calls=0,
					execution_mode=settings.execution_mode,
					screenshot_mode=False,
					screenshot_policy='off',
					model_profile=profile,
					text_model=text.model,
					text_model_base_url=text_endpoint,
					planner_calls=0,
					replans=0,
					collect_calls=0,
					summarize_calls=0,
					verify_calls=0,
					research_usage={},
					elapsed_ms=0,
					reason='',
					execution_status='queued',
					evaluation_status='not_run',
					evaluation=None,
					token_usage=token_usage([]),
				)
				for t in tasks
				for n in range(1, repeats + 1)
			]
			batch: dict[str, Any] = dict(
				id=run_id,
				status='running',
				browser_mode=self._browser_mode,
				path=str(self.folder(run_id)),
				created_at=now(),
				suite_version=VERSION,
				attempts=attempts,
				system1=config,
				execution_mode=settings.execution_mode,
				planner=settings.execution.public() if settings.jev is not None else None,
				screenshot_mode=False,
				screenshot_policy='off',
				model_profile=profile,
				text_model=text.model,
				text_model_base_url=text_endpoint,
			)
			self.batch = batch
			self.snapshots = {}
			# 私有配置包含密钥，不进入 batch/manifest/trace。
			self._settings = settings
			self.stop_event.clear()
			self.active = True
		try:
			sources = {
				str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
				for p in Path(__file__).parent.rglob('*')
				if p.suffix in {'.py', '.js', '.html', '.css', '.json'} and '__pycache__' not in p.parts
			}
			manifest = {
				'run_id': run_id,
				'created_at': batch['created_at'],
				'suite_version': VERSION,
				'tasks': tasks,
				'repeats': repeats,
				'sources_sha256': sources,
				'limits': {
					'actions': MAX_STEPS,
					'decision_rounds': MAX_STEPS,
					'decision_calls': MAX_STEPS,
					'seconds': self.timeout,
				},
				'execution_mode': settings.execution_mode,
				'screenshot_mode': False,
				'screenshot_policy': 'off',
				'model_profile': profile,
				'system1': config,
				'planner': settings.execution.public() if settings.jev is not None else None,
				'viewport': VIEWPORT,
				'browser_mode': self._browser_mode,
				'shared_browser_profile': self._browser_mode == 'attached_chrome' or bool(settings.user_data_dir),
				'models': {
					'TEXT_MODEL': text.model,
					'TEXT_MODEL_REASONING': (
						'disabled' if profile == FINAL_REPORT_PROFILE else os.getenv('TEXT_MODEL_REASONING', 'disabled')
					),
				},
				'endpoints': {'text': text_endpoint},
			}
			atomic_json(self.folder(run_id) / 'manifest.json', self.clean(manifest))
			self.persist()
			if before_start is not None:
				before_start()
			self.thread = threading.Thread(target=self.work, args=(batch, tasks), daemon=True, name='baseline-worker')
			self.thread.start()
		except Exception:
			with self.lock:
				self.active = False
				batch['status'] = 'error'
			raise
		return self.state()

	def stop(self):
		with self.lock:
			if self.active and self.batch and self.batch['status'] not in TERMINAL:
				self.stop_event.set()
				self.batch['status'] = 'stopping'
		self.persist()
		return self.state()

	def write_summary(self, batch):
		"""汇总分别保留决策、文本、规划调用，避免混淆成本与轮数。"""
		folder = self.folder(batch['id'])
		counts = Counter(r['status'] for r in batch['attempts'])
		batch['counts'] = dict(counts)
		batch['outcome_statistics'] = outcome_statistics(batch['attempts'])
		batch['success_rate'] = batch['outcome_statistics']['success_rate']
		self.add_token_usage(batch)
		atomic_json(folder / 'summary.json', self.clean(batch))
		fields = [
			'task_id',
			'repeat',
			'status',
			'execution_status',
			'execution_reason',
			'evaluation_status',
			'verdict',
			'evaluation_reason',
			'execution_elapsed_ms',
			'execution_mode',
			'system1_provider',
			'model_profile',
			'text_model',
			'screenshot_policy',
			'screenshot_mode',
			'actions',
			'decision_rounds',
			'stale_retries',
			'forced_after_stale',
			'decision_calls',
			'laya_calls',
			'deepseek_calls',
			'jev_calls',
			'text_calls',
			'planner_calls',
			'replans',
			'planner_usage',
			'collect_calls',
			'summarize_calls',
			'verify_calls',
			'research_usage',
			'elapsed_ms',
			'reason',
			*TOKEN_FIELDS,
			'token_usage_complete',
			'missing_usage_calls',
		]
		output = io.StringIO(newline='')
		writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
		writer.writeheader()
		writer.writerows(
			self.clean(
				[
					{
						**row,
						'verdict': (row.get('evaluation') or {}).get('final', {}).get('verdict'),
						'evaluation_reason': (row.get('evaluation') or {}).get('final', {}).get('reason'),
						**{k: v for k, v in row.get('token_usage', {}).items() if k != 'complete'},
						'token_usage_complete': row.get('token_usage', {}).get('complete'),
					}
					for row in batch['attempts']
				]
			)
		)
		(folder / 'summary.csv').write_text(output.getvalue(), encoding='utf-8-sig')
		screenshot_policy = batch.get('screenshot_policy', 'always' if batch.get('screenshot_mode') else 'off')
		stats = batch['outcome_statistics']
		profile = batch.get('model_profile', DEBUG_PROFILE)
		profile_label = '最终报告' if profile == FINAL_REPORT_PROFILE else '调试'
		planner_model = (batch.get('planner') or {}).get('model') or '—'
		text_model = batch.get('text_model') or '—'
		report = [
			'# 基线运行结果',
			'',
			f'批次：{batch["id"]}',
			f'状态：{batch["status"]}',
			f'目标达成：{stats["met"]} / {stats["attempts"]}（分母包含所有计划尝试）',
			'',
			f'总成功率：{batch["success_rate"]:.1%}；环境阻塞：{counts["environment_blocked"]} 次',
			'',
			f'目标未达成：{stats["not_met"]}；证据不足：{stats["unknown"]}；'
			f'评估调用失败：{stats["evaluation_errors"]}；未评估：{stats["not_run"]}；'
			f'旧档案未判定：{stats["legacy_unknown"]}（旧方法共 {stats["legacy"]} 次）',
			'',
			f'执行模式：{batch.get("execution_mode", "direct")}',
			'',
			'本工作台每步只执行 1 个动作（原生 Browser Use 默认最多 5 个），可能增加模型调用和耗时。',
			'',
			f'模型配置：{profile_label} (`{profile}`) · Planner：{planner_model} · Text：{text_model}',
			'',
			f'截图策略：{screenshot_policy}',
			'',
			'| 任务 | 轮次 | 执行状态 / 评估状态 / 模型判决 | 动作 | '
			'决策轮 / System1 / 文本 / Planner / 收集 / 总结 / 复核 '
			'| 交回后再委派 | 耗时(ms) |',
			'|---|---:|---|---:|---:|---:|---:|',
		]
		for row in batch['attempts']:
			final = (row.get('evaluation') or {}).get('final', {})
			result = (
				f'{row["execution_status"]} / {row.get("evaluation_status", "not_run")} / {final.get("verdict", "—")}'
				if 'execution_status' in row
				else f'旧方法：{row["status"]}'
			)
			report.append(
				f'| {row["name"]} | {row["repeat"]} | {result} | {row["actions"]} | '
				f'{row.get("decision_rounds", row.get("decision_calls", row["laya_calls"]))} / '
				f'{row.get("decision_calls", row["laya_calls"])} / {row["text_calls"]} / '
				f'{row.get("planner_calls", 0)} / {row.get("collect_calls", 0)} / '
				f'{row.get("summarize_calls", 0)} / {row.get("verify_calls", 0)} | {row.get("replans", 0)} | '
				f'{row["elapsed_ms"]} |'
			)
		report += [
			'',
			'## Token 消耗（排除 Verify）',
			'',
			'| 任务 | 轮次 | Jev 输入/总和 | Planner 输入/输出/合计 | 其他 输入/输出/合计 | LLM 总和 | 完整 |',
			'|---|---:|---:|---|---|---:|---|',
		]
		for row in batch['attempts']:
			usage = row.get('token_usage', {})

			def group(name):
				return '/'.join(str(usage.get(f'{name}_{k}_tokens', '—')) for k in ('input', 'output', 'total'))

			report.append(
				f'| {row["name"]} | {row["repeat"]} | {usage.get("jev_total_tokens", "—")} | '
				f'{group("planner")} | {group("other")} | {usage.get("llm_total_tokens", "—")} | '
				f'{"是" if usage.get("complete") else "否：仅已知用量"} |'
			)
		totals = batch['token_usage']
		report += [
			'',
			f'批次 Jev 总和：{totals["jev_total_tokens"]}；LLM 总和：{totals["llm_total_tokens"]}'
			+ ('（仅已知用量）' if not totals['complete'] else ''),
			'其他包含 Text、Summarize、Collect 及启用时的 DeepSeek System1；Verify 和 Laya 不计入。',
			'',
			'状态与证据详见各任务的 evaluation.json；模型宣告 DONE 不等于独立验收通过。',
			'文件可能包含网页内容和个人信息，分享前请检查。',
		]
		(folder / 'report.zh.md').write_text('\n'.join(report), encoding='utf-8')

	def persist(self):
		with self.io_lock:
			with self.lock:
				batch = deepcopy(self.batch)
			if batch:
				self.write_summary(batch)

	def save(self, run_id=None):
		with self.lock:
			current = self.batch['id'] if self.batch else None
		run_id = run_id or current
		if run_id is None:
			raise ValueError('No run to save')
		if run_id == current:
			self.persist()
		batch = self.read_run(run_id)
		return {**self.state(), 'saved_path': batch['path']}

	def open_folder(self, run_id):
		"""只打开已知批次的实际目录，不接受客户端传入任意文件路径。"""
		with self.io_lock:
			self.read_run(run_id)
			folder = self.folder(run_id)
			# 批次可能被外部删除，不能将不存在的路径交给系统处理。
			if not folder.is_dir():
				raise ValueError('Result folder no longer exists')
			try:
				# Windows 直接打开目录，不通过 shell 拼接命令。
				if sys.platform == 'win32':
					os.startfile(str(folder))
				# 其他桌面系统也使用独立参数传递经过校验的绝对目录。
				else:
					opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
					subprocess.run(
						[opener, str(folder)], check=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
					)
			except (OSError, subprocess.SubprocessError) as error:
				raise RuntimeError(f'Could not open result folder: {folder}') from error
		return {**self.state(), 'opened_path': str(folder)}

	def rename(self, run_id, name):
		if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80 or any(ord(c) < 32 for c in name):
			raise ValueError('Batch name must contain 1-80 characters without control characters')
		name = name.strip()
		with self.io_lock, self.lock:
			folder = self.folder(run_id)
			if (
				folder.parent != self.root
				or folder.name != run_id
				or (self.root / run_id).is_symlink()
				or (self.root / run_id).is_junction()
			):
				raise ValueError('Invalid run path')
			if self.batch and self.batch['id'] == run_id and self.active:
				raise BaselineConflict('Cannot rename a running batch')
			try:
				batch = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
			except FileNotFoundError:
				raise ValueError('Unknown run') from None
			if batch.get('status') not in TERMINAL:
				raise BaselineConflict('Cannot rename an unfinished batch')
			batch['name'] = name
			atomic_json(folder / 'summary.json', self.clean(batch))
			if self.batch and self.batch['id'] == run_id:
				self.batch = batch
		return {**self.state(), 'renamed_batch': self.clean(batch)}

	def delete(self, run_id):
		with self.io_lock, self.lock:
			folder = self.folder(run_id)
			entry = self.root / run_id
			if folder.parent != self.root or folder.name != run_id or entry.is_symlink() or entry.is_junction():
				raise ValueError('Invalid run path; linked directories cannot be deleted')
			current = self.batch and self.batch['id'] == run_id
			if current and (self.active or (self.thread and self.thread.is_alive())):
				raise BaselineConflict('Stop the running batch before deleting it')
			if not folder.is_dir():
				raise ValueError('Unknown run')
			batch = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
			if batch.get('status') not in TERMINAL:
				raise BaselineConflict('Cannot delete an unfinished batch')
			# Refuse reparse points anywhere in the tree before recursive removal.
			for parent, directories, files in os.walk(folder, followlinks=False):
				for name in directories + files:
					path = Path(parent) / name
					if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(folder):
						raise ValueError('Run contains a linked path; deletion refused')
			try:
				shutil.rmtree(folder)
			except OSError as error:
				raise RuntimeError('Could not delete all batch files; check file permissions and retry') from error
			self.snapshots = {key: value for key, value in self.snapshots.items() if key[0] != run_id}
			if current:
				self.batch = None
				remaining = self.runs()
				if remaining:
					self.batch = self.read_run(remaining[0]['id'])
		return self.state()

	def work(self, batch: dict[str, Any], tasks: list[dict[str, Any]]):
		run_id = batch['id']
		by_id = {task['id']: task for task in tasks}
		fatal = ''
		try:
			for row in batch['attempts']:
				with self.lock:
					if self.stop_event.is_set() or fatal:
						row.update(status='cancelled', execution_status='cancelled', reason=fatal or 'Stopped by user')
						continue
					row.update(status='running', execution_status='running', started_at=now())
				fatal = self.run_attempt(run_id, row, by_id[row['task_id']])
				self.persist()
		except Exception as error:
			fatal = self.clean(str(error))
		finally:
			with self.lock:
				for row in batch['attempts']:
					if row['status'] in {'running', 'queued'}:
						row.update(
							status='error' if fatal else 'cancelled',
							execution_status='error' if fatal else 'cancelled',
							reason=fatal or 'Stopped',
						)
				batch.update(
					status='error' if fatal else 'cancelled' if self.stop_event.is_set() else 'completed',
					ended_at=now(),
					reason=fatal,
				)
			try:
				self.persist()
			finally:
				with self.lock:
					self.active = False

	def run_attempt(self, run_id, row, task):
		settings = self._settings
		assert settings is not None
		folder = self.attempt_folder(run_id, row['id'])
		(folder / 'screenshots').mkdir(parents=True)
		started = time.monotonic()
		evaluation_started_at = now()
		observations, evaluations, execution_events = [], [], []
		agent = None
		fatal = ''

		def cancelled():
			if self.stop_event.is_set():
				return 'cancelled'
			return 'timeout' if time.monotonic() - started >= self.timeout else None

		def append(event, payload):
			with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
				stream.write(json.dumps(self.clean({'time': now(), 'event': event, **payload}), ensure_ascii=False) + '\n')

		def publish(snapshot: dict[str, Any]):
			"""Persist the same state that the page can read while the attempt is running."""
			snapshot = deepcopy(snapshot)
			snapshot['page'].pop('screenshot', None)
			snapshot['observations'] = observations
			if observations:
				snapshot['page']['screenshot_url'] = observations[-1].get('screenshot_url')
			snapshot = self.clean(snapshot)
			atomic_json(folder / 'trace.json', snapshot)
			result = snapshot.get('result')
			atomic_json(folder / 'result.json', result)
			(folder / 'result.md').write_text((result or {}).get('markdown', ''), encoding='utf-8')
			if partial := snapshot.get('partial_result'):
				(folder / 'partial-result.md').write_text(partial['markdown'], encoding='utf-8')
			calls = snapshot.get('model_calls', [])
			verify_calls = [call for call in calls if call['kind'] == 'verify']
			verify_usage = Counter()
			for call in verify_calls:
				for key, value in (call.get('response') or {}).get('usage', {}).items():
					if type(value) in (int, float):
						verify_usage[key] += value
			with self.lock:
				self.snapshots[(run_id, row['id'])] = snapshot
				row.update(
					actions=len(snapshot['history']),
					decision_calls=sum(call['kind'] == PROVIDER for call in calls),
					decision_rounds=snapshot['decision_rounds'],
					stale_retries=snapshot['stale_retries'],
					forced_after_stale=snapshot['forced_after_stale'],
					verify_calls=len(verify_calls),
					research_usage={'verify': dict(verify_usage)},
					token_usage=snapshot['token_usage'],
					elapsed_ms=round((time.monotonic() - started) * 1000),
				)
				if roles := snapshot.get('jev_usage_by_role'):
					row.update(
						jev_usage_by_role=roles,
						jev_trace_path=snapshot['jev_trace_path'],
						jev_calls=roles['system1']['calls'],
						decision_calls=roles['system1']['calls'],
						planner_calls=roles['planner']['calls'],
						replans=snapshot.get('replans', 0),
						text_calls=roles['text']['calls'],
						collect_calls=roles['summary']['calls'],
						summarize_calls=roles['answer']['calls'],
						risk_calls=roles['risk']['calls'],
					)

		def event(record: RunEvent):
			name, payload, snapshot = record.name, record.payload, record.snapshot
			append(name, payload)
			if name in {'action_started', 'action_executed', 'action_result', 'observation_error'}:
				execution_events.append({'event': name, 'time': now(), **deepcopy(payload)})
			if record.preview is not None:
				page, preview = snapshot['page'], record.preview
				step = len(snapshot['history'])
				phase = payload['phase']
				evaluations.append(
					{'step': step, 'time': now(), 'phase': phase, 'event_seq': payload['event_seq'], **preview.sample}
				)
				frame = {
					'step': step,
					'phase': phase,
					**({'stale_reason': payload['stale_reason']} if 'stale_reason' in payload else {}),
					'url': page['url'],
					'title': page['title'],
					'time': now(),
					'event_seq': payload['event_seq'],
					'elements': snapshot['elements'],
					'page': page,
				}
				try:
					if preview.image is None:
						frame['screenshot_error'] = preview.error
					else:
						filename = f'{len(observations):06}.jpg'
						(folder / 'screenshots' / filename).write_bytes(base64.b64decode(preview.image, validate=True))
						frame['screenshot_url'] = '/api/baseline/frame?' + urlencode(
							{'run_id': run_id, 'attempt_id': row['id'], 'name': filename}
						)
				except Exception as error:
					frame['screenshot_error'] = str(error)
				observations.append(frame)
				append('observed', {'page': page, 'evaluation': preview.sample, 'frame': frame})
				atomic_json(
					folder / 'evaluation.json',
					self.clean(
						{'observations': evaluations, 'metrics_status': 'not_computed', 'ever_met': None, 'first_met_step': None}
					),
				)
			publish(snapshot)

		try:
			append('attempt_start', {'task': task, 'repeat': row['repeat']})
			agent = self._run_factory(
				task['url'],
				task['goal'],
				settings,
				on_event=event,
				should_cancel=cancelled,
				probe=CaseProbe(task),
				**({'jev_artifact_dir': folder} if settings.jev is not None else {}),
			)
			state = agent.snapshot()
			while state['status'] not in {'done', 'blocked'}:
				state = agent.command('tick')
			row.update(execution_status=state['status'], execution_reason='')

		except RunCancelled as error:
			with self.lock:
				row.update(
					status=str(error),
					execution_status=str(error),
					reason='Stopped by user' if str(error) == 'cancelled' else 'Run budget exceeded',
				)
		except EnvironmentBlocked as error:
			if isinstance(error, BrowserUnavailable):
				fatal = str(error)
			with self.lock:
				row.update(status='environment_blocked', execution_status='environment_blocked', reason=str(error))
		except Exception as error:
			message = str(error) or type(error).__name__
			status = 'timeout' if isinstance(error, TimeoutError) or 'budget' in message.lower() else 'error'
			if (
				'Model connection failed' in message
				or 'Model provider returned HTTP' in message
				or 'TEXT_MODEL_API_KEY' in message
				or 'PLANNER_API_KEY' in message
			):
				fatal = message
			with self.lock:
				row.update(status=status, execution_status=status, reason=message)
			append('error', {'error': message})
		finally:
			try:
				if agent:
					try:
						if settings.jev is not None:
							agent.finish_execution()
						# Final sampling is read-only, including on timeout/error/cancellation.
						agent.observe('final')
					except Exception as error:
						append('final_observation_error', {'error': str(error)})
						evaluations.append(
							{'phase': 'final', 'time': now(), 'sample_status': 'error', 'evidence': {}, 'reason': str(error)}
						)
					if row.get('execution_status') == 'timeout':
						agent.record_timeout_evidence()
				execution_ended_at = now()
				execution_status = row.get('execution_status', 'error')
				execution_reason = row.get('execution_reason', row.get('reason', ''))
				row.update(
					execution_reason=execution_reason,
					execution_ended_at=execution_ended_at,
					execution_elapsed_ms=round((time.monotonic() - started) * 1000),
				)
				state = deepcopy(agent.snapshot()) if agent else {'goal': task['goal'], 'history': []}
				state['observations'] = deepcopy(observations)
				context = self.clean(
					build_context(
						state,
						evaluations,
						task_started_at=evaluation_started_at,
						task_ended_at=execution_ended_at,
						execution_status=execution_status,
						execution_reason=execution_reason,
						execution_events=execution_events,
					)
				)
				frozen = freeze_context(context)
				# Durable before the first request. Retries/replays never recapture or execute the task.
				atomic_json(folder / 'evaluation-context.json', frozen)
				append('evaluation_context_frozen', {k: frozen[k] for k in ('context_id', 'schema_version')})
				self.persist()  # A restart during verification must retain the already-ended execution state.
				trusted = any(sample.get('sample_status') == 'completed' and sample.get('evidence') for sample in evaluations)
				if agent and trusted and execution_status != 'cancelled':
					final = evaluate_frozen(
						frozen,
						settings.evaluation,
						agent.model_call,
						should_cancel=self.stop_event.is_set,
					)
				else:
					final = {
						'method': 'llm',
						'evaluation_status': 'not_run',
						'context_id': frozen['context_id'],
						'reason': '用户取消，未追加评估调用'
						if execution_status == 'cancelled'
						else '没有可用的归档页面证据，未调用评估模型',
						'attempts': 0,
					}
				final.update(
					phase='final',
					time=now(),
					step=len(state.get('history', [])),
					evidence=deepcopy(evaluations[-1].get('evidence', {})) if evaluations else {},
				)
				evaluation = {
					'metrics_status': 'not_computed',
					'ever_met': None,
					'first_met_step': None,
					'extra_actions': None,
					'early_stop': execution_status == 'done' and final.get('verdict') == 'not_met',
					'execution_status': execution_status,
					'evaluation_status': final['evaluation_status'],
					'final': final,
				}
				with self.lock:
					row.update(
						status=attempt_status(execution_status, final),
						evaluation=evaluation,
						evaluation_status=final['evaluation_status'],
						ended_at=now(),
						elapsed_ms=round((time.monotonic() - started) * 1000),
					)
					row['reason'] = self.clean(execution_reason or final['reason'])
				if agent:
					snapshot = agent.snapshot()
					snapshot.update(
						evaluation=deepcopy(final),
						execution_status=execution_status,
						execution_reason=execution_reason,
						evaluation_status=final['evaluation_status'],
					)
					publish(snapshot)
				atomic_json(folder / 'evaluation.json', self.clean({**evaluation, 'observations': evaluations}))
				append('attempt_end', {'result': row})
			finally:
				if agent:
					try:
						agent.close()
					except BrowserUnavailable as error:
						fatal = fatal or self.clean(str(error))
						append('browser_cleanup_error', {'error': str(error)})
					except Exception:
						pass  # Never mask an artifact or execution error while releasing the owned tab.
		return fatal
