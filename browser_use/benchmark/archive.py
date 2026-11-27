"""Atomically write benchmark JSON snapshots and evaluation results."""

import json
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

_MEMORY_PUBLISH_INTERVAL = 1.0
_DISK_PUBLISH_INTERVAL = 10.0


class AttemptPublisher:
	"""Publish live snapshots at most once a second and persist them every ten seconds."""

	def __init__(
		self,
		publish_memory: Callable[[dict[str, Any]], None],
		write_disk: Callable[[dict[str, Any]], None],
		*,
		prepare: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
	):
		self._publish_memory = publish_memory
		self._write_disk = write_disk
		self._prepare = prepare
		self._lock = threading.Lock()
		self._last_memory = float('-inf')
		self._last_disk = float('-inf')
		self.memory_publishes = 0
		self.disk_writes = 0

	def publish(self, snapshot: dict[str, Any], *, force: bool = False) -> None:
		with self._lock:
			current = time.monotonic()
			memory_due = force or current - self._last_memory >= _MEMORY_PUBLISH_INTERVAL
			disk_due = force or current - self._last_disk >= _DISK_PUBLISH_INTERVAL
			if not (memory_due or disk_due):
				return
			if self._prepare is not None:
				snapshot = self._prepare(snapshot)
			if memory_due:
				self._publish_memory(snapshot)
				self._last_memory = current
				self.memory_publishes += 1
			if disk_due:
				self._write_disk(snapshot)
				self._last_disk = current
				self.disk_writes += 1


def atomic_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
	"""提交完整快照；短暂文件占用只重试替换，不重放产生快照的任何工作。"""
	payload = json.dumps(data, ensure_ascii=False, indent=indent)
	temporary = None
	try:
		# 每个写入者独占同目录的临时文件，并在替换前关闭句柄。
		with tempfile.NamedTemporaryFile(
			mode='w', encoding='utf-8', dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False
		) as stream:
			temporary = Path(stream.name)
			stream.write(payload)
		for attempt in range(8):
			try:
				temporary.replace(path)
				return
			except OSError as error:
				# Windows 读句柄可能同时表现为访问拒绝、共享冲突或锁冲突。
				if getattr(error, 'winerror', None) not in {5, 32, 33} or attempt == 7:
					raise
				time.sleep(0.01 * 2**attempt)
	finally:
		if temporary is not None:
			# 清理失败不能覆盖原始写入异常；绝不删除仍然有效的目标文件。
			with suppress(OSError):
				temporary.unlink(missing_ok=True)
