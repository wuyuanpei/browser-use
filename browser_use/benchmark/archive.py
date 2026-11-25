"""Atomically write benchmark JSON snapshots and evaluation results."""

import json
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any


def atomic_json(path: Path, data: Any) -> None:
	"""提交完整快照；短暂文件占用只重试替换，不重放产生快照的任何工作。"""
	payload = json.dumps(data, ensure_ascii=False, indent=2)
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
