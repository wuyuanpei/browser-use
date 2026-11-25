"""Model helpers for field values, risky actions, capture summaries, and final answers."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from browser_use.agent.jev import prompts
from browser_use.agent.jev.views import (
	ANSWER_LIMIT,
	PAGE_TEXT_LIMIT,
	RISK_PAGE_TEXT_LIMIT,
	SUMMARY_LIMIT,
	TEXT_VALUE_LIMIT,
	ActionRecord,
	Candidate,
	JevSettings,
	Observation,
	SubgoalRun,
)
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import SystemMessage, UserMessage


class TextReply(BaseModel):
	model_config = ConfigDict(extra='forbid')
	text: str | None = Field(min_length=1, max_length=TEXT_VALUE_LIMIT)


class SummaryReply(BaseModel):
	model_config = ConfigDict(extra='forbid')
	summary: str = Field(min_length=1)


class AnswerReply(BaseModel):
	model_config = ConfigDict(extra='forbid')
	text: str = Field(min_length=1, max_length=ANSWER_LIMIT)


class RiskAuthorization(BaseModel):
	model_config = ConfigDict(extra='forbid')
	authorized: bool
	reason: str


M = TypeVar('M', bound=BaseModel)


def field_info(field: Candidate) -> dict:
	return field.model_dump(include={'key', 'role', 'name', 'value', 'placeholder', 'context'})


def action_info(record: ActionRecord) -> dict:
	status = {'executed_ok': 'ok', 'executed_failed': f'error: {(record.error or "")[:120]}'}.get(record.status, record.status)
	return {
		'run': record.run_id,
		'operation': record.operation,
		'target': record.target_label,
		'text': record.text,
		'result': status,
		'page_changed': record.page_changed,
	}


class ModelHelpers:
	def __init__(self, settings: JevSettings, planner: BaseChatModel, invoke: Callable[..., Awaitable[Any]], timeout: float):
		self.settings = settings
		self.invoke = invoke
		self.timeout = timeout
		default = settings.helper_llm or planner
		self.models = {
			'text': settings.text_llm or default,
			'summary': settings.summary_llm or default,
			'answer': settings.answer_llm or default,
			'risk': planner,
		}

	async def _call(self, role: str, prompt: str, payload: dict, schema: type[M], attempt: int = 1) -> M:
		messages = [SystemMessage(content=prompt), UserMessage(content=json.dumps(payload, ensure_ascii=False, allow_nan=False))]
		response = await self.invoke(role, self.models[role], messages, output_format=schema, attempt=attempt)
		if isinstance(response.completion, str):
			return schema.model_validate_json(response.completion)
		return schema.model_validate(response.completion)

	async def text(
		self, goal: str, run: SubgoalRun, obs: Observation, field: Candidate, actions: list[ActionRecord]
	) -> str | None:
		payload = {
			'original_goal': goal,
			'subgoal': run.params.model_dump(include={'objective', 'completion', 'inputs'}),
			'field': field_info(field),
			'page': {
				'url': obs.url,
				'title': obs.title,
				'text': obs.visible_text[:PAGE_TEXT_LIMIT],
				'fields': [field_info(c) for c in obs.candidates.get('TYPE_TEXT', [])],
			},
			'recent_actions': [action_info(a) for a in actions[-6:]],
		}
		response = await asyncio.wait_for(
			self._call('text', self.settings.text_prompt or prompts.TEXT_PROMPT, payload, TextReply), self.timeout
		)
		return response.text if response.text and response.text.strip() else None

	async def summary(self, page: dict) -> str:
		response = await asyncio.wait_for(
			self._call('summary', self.settings.summary_prompt or prompts.SUMMARY_PROMPT, {'page': page}, SummaryReply),
			self.timeout,
		)
		if not response.summary.strip():
			raise ValueError('Empty page summary')
		return response.summary[:SUMMARY_LIMIT]

	async def authorize(
		self, goal: str, run: SubgoalRun, obs: Observation, proposed: dict, actions: list[ActionRecord]
	) -> RiskAuthorization:
		payload = {
			'original_goal': goal,
			'subgoal': run.params.model_dump(include={'objective', 'completion'}),
			'proposed': proposed,
			'page': {'url': obs.url, 'title': obs.title, 'text': obs.visible_text[:RISK_PAGE_TEXT_LIMIT]},
			'recent_actions': [action_info(a) for a in actions[-8:]],
		}
		if proposed['effect'] in {'send', 'publish', 'share'}:
			payload['visible_fields'] = [field_info(c) for c in obs.candidates.get('TYPE_TEXT', [])[:12]]
		return await asyncio.wait_for(self._call('risk', prompts.RISK_PROMPT, payload, RiskAuthorization), self.timeout)

	async def answer(self, payload: dict) -> str:
		async with asyncio.timeout(self.settings.answer_timeout):
			prompt = self.settings.answer_prompt or prompts.ANSWER_PROMPT
			for attempt in (1, 2):
				try:
					response = await self._call('answer', prompt, payload, AnswerReply, attempt)
					if not response.text.strip():
						raise ValueError('Empty final answer')
					return response.text
				except (ValidationError, ValueError):
					if attempt == 2:
						raise
					prompt += '\n\n' + prompts.OUTPUT_FORMAT_RETRY
		raise RuntimeError('Final answer was not produced')
