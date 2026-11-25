import asyncio
import json

import pytest

from browser_use.agent.jev.helpers import ModelHelpers
from browser_use.agent.jev.views import JevSettings
from tests.ci.conftest import create_mock_llm


def helpers(replies, **settings):
	model = create_mock_llm(replies)
	calls = []

	async def invoke(role, llm, messages, *, attempt=1, **kwargs):
		calls.append((role, attempt, messages))
		return await llm.ainvoke(messages, **kwargs)

	return ModelHelpers(JevSettings(**settings), model, invoke, 1), calls


async def test_answer_format_retry_reuses_evidence():
	h, calls = helpers(['{"text":""}', '{"text":"Confirmed"}'], answer_prompt='Custom answer instructions.')
	assert await h.answer({'original_goal': 'Task', 'last_page': {'url': 'https://example.test'}}) == 'Confirmed'
	assert [(role, n) for role, n, _ in calls] == [('answer', 1), ('answer', 2)]
	assert calls[0][2][0].content == 'Custom answer instructions.'
	assert 'formatting retry' in calls[1][2][0].content
	assert calls[0][2][1].content == calls[1][2][1].content


async def test_answer_retry_is_bounded():
	h, calls = helpers(['{"invalid":true}', json.dumps({'text': 'a' * 12001})])
	with pytest.raises(ValueError):
		await h.answer({})
	assert len(calls) == 2


async def test_timeout_and_summary_limit():
	h, _ = helpers([json.dumps({'summary': 'a' * 600})])
	assert await h.summary({'text': 'original'}) == 'a' * 400

	async def slow(*args, **kwargs):
		await asyncio.sleep(2)

	h = ModelHelpers(JevSettings(answer_timeout=0.03), create_mock_llm(), slow, 1)
	with pytest.raises(TimeoutError):
		await h.answer({})
