"""Store original viewport captures before asking the summary model."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from browser_use.agent.jev.rules import content_lines, document_key, observed_link_pairs
from browser_use.agent.jev.views import Observation, SubgoalRun


class Capture(BaseModel):
	model_config = ConfigDict(extra='forbid')

	id: str
	agent_id: str
	run_id: str | None
	step: int
	url: str
	title: str
	observed_at: str
	text: str
	links: list[dict[str, str]]
	summary: str | None = None
	summary_status: Literal['pending', 'ok', 'failed'] = 'pending'
	summary_error: str | None = None


class CaptureStore:
	def __init__(self, directory: Path, agent_id: str):
		self.directory = directory
		self.agent_id = agent_id
		self.records: list[Capture] = []
		self._saved_text: dict[str, list[str]] = {}

	def is_saved(self, url: str, text: str) -> bool:
		previous = self._saved_text.get(document_key(url), [])
		return bool(previous) and all(any(line in body for body in previous) for line in content_lines(text))

	def available(self, obs: Observation) -> bool:
		return (
			urlsplit(obs.url).scheme in {'http', 'https'}
			and bool(content_lines(obs.visible_text))
			and not self.is_saved(obs.url, obs.visible_text)
		)

	def path(self, record: Capture) -> Path:
		return self.directory / f'{self.agent_id}_p{int(record.id[1:]):04d}.json'

	def write(self, record: Capture) -> None:
		self.directory.mkdir(parents=True, exist_ok=True)
		path = self.path(record)
		temp = path.with_suffix('.tmp')
		temp.write_text(json.dumps(record.model_dump(), ensure_ascii=False, allow_nan=False), encoding='utf-8')
		temp.replace(path)

	def add(self, run_id: str | None, step: int, page: dict) -> Capture:
		record = Capture(
			id=f'p{len(self.records) + 1}',
			agent_id=self.agent_id,
			run_id=run_id,
			step=step,
			observed_at=datetime.now(UTC).isoformat(),
			url=page['url'],
			title=page['title'],
			text=page['text'],
			links=observed_link_pairs(page.get('links', [])),
		)
		self.write(record)
		self.records.append(record)
		self._saved_text.setdefault(document_key(record.url), []).append(' '.join(content_lines(record.text)))
		return record

	def research_state(self, obs: Observation) -> dict | None:
		available = self.available(obs)
		if not available and not self.records:
			return None
		return {
			'collect_available': available,
			'current_page_saved': self.is_saved(obs.url, obs.visible_text),
			'collected_pages': [r.model_dump(include={'id', 'url', 'title', 'summary'}) for r in self.records],
		}

	def planner_state(self, mode: str, backend: str, run: SubgoalRun | None, page: tuple[str, str] | None) -> str:
		last = 'Last subgoal: none.'
		if run and run.handback:
			last = f'Last subgoal: {run.id} "{run.params.objective}" - handed back ({run.handback.reason}) after {len(run.executed_actions)} of {run.max_actions} actions.'
		during = sum(r.run_id == run.id for r in self.records) if run else 0
		saved = bool(page and self.is_saved(*page))
		available = bool(page and urlsplit(page[0]).scheme in {'http', 'https'} and content_lines(page[1]) and not saved)
		lines = [
			'<jev_state>',
			f'Mode: {mode}. System1 backend: {backend}.',
			last,
			f'Saved pages: {len(self.records)} total; {len(self.records) - during} before the last subgoal, {during} during it. ',
		]
		if page is not None:
			lines.append(f'Current page saved: {"yes" if saved else "no"}. COLLECT available: {"yes" if available else "no"}.')
		if available:
			lines.append(
				'COLLECT saves the current viewport text, while browser_state can include other items offscreen. '
				'For multi-item reading, if the viewport contains an unsaved requested body, make saving it the '
				'next stage before scrolling to another item. One capture can cover items visible together. '
				'Associate an expand control with its own item using its container or surrounding text. '
				'For a truncated body, expand that item and save the newly revealed text.'
			)
		for record in self.records:
			lines.append(f'- {record.id} | {record.url} | {record.title} | {record.summary or "(summary unavailable)"}')
			if record.links:
				lines.append('  links: ' + '; '.join(f'{link["text"]} -> {link["url"]}' for link in record.links[:10]))
		return '\n'.join([*lines, '</jev_state>'])
