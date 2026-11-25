"""Append-only UTF-8 Jev decision trace."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _public(value: Any) -> Any:
	if isinstance(value, dict):
		return {
			key: _public(item)
			for key, item in value.items()
			if 'apikey' not in str(key).lower().replace('_', '').replace('-', '') and str(key).lower() != 'authorization'
		}
	if isinstance(value, (list, tuple)):
		return [_public(item) for item in value]
	if isinstance(value, float) and not math.isfinite(value):
		return None
	return value


class JevTrace:
	def __init__(self, path: Path, agent_id: str):
		self.path = Path(path)
		self.agent_id = agent_id
		self.path.parent.mkdir(parents=True, exist_ok=True)

	def write(self, event: str, step: int | None = None, **fields: Any) -> None:
		record = {
			**_public(fields),
			'event': event,
			'ts': datetime.now(UTC).isoformat(),
			'agent_id': self.agent_id,
			'step': step,
		}
		with self.path.open('a', encoding='utf-8') as stream:
			stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + '\n')
