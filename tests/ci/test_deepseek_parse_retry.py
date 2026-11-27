"""DeepSeek exposes one physical response; the Jev planner owns corrective retries."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import APIConnectionError, RateLimitError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from browser_use import Agent
from browser_use.agent.jev import JevSettings
from browser_use.agent.jev.trace import JevTrace
from browser_use.agent.jev.usage import UsageLedger
from browser_use.llm.deepseek.chat import ChatDeepSeek
from browser_use.llm.exceptions import ModelOutputFormatError, ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import UserMessage
from browser_use.tokens.service import TokenCost


class Answer(BaseModel):
	value: int


def response(raw, *, json_mode=False, reason=None, tokens: int | None = 10):
	message = {'role': 'assistant', 'content': raw if json_mode else None}
	if not json_mode and raw is not None:
		message['tool_calls'] = [{'id': '1', 'type': 'function', 'function': {'name': 'Answer', 'arguments': raw}}]
	return ChatCompletion.model_validate(
		{
			'id': 'local',
			'created': 0,
			'object': 'chat.completion',
			'model': 'deepseek-flash',
			'choices': [{'index': 0, 'finish_reason': reason or ('stop' if json_mode else 'tool_calls'), 'message': message}],
			'usage': {'prompt_tokens': tokens, 'completion_tokens': 2, 'total_tokens': tokens + 2, 'prompt_cache_hit_tokens': 3}
			if tokens is not None
			else None,
		}
	)


def model_with(monkeypatch, replies, **settings):
	create = AsyncMock(side_effect=replies)
	client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
	model = ChatDeepSeek(model='deepseek-flash', temperature=0.2, max_tokens=512, **settings)
	monkeypatch.setattr(model, '_client', lambda: client)
	return model, create


@pytest.mark.parametrize(
	('raw', 'reason', 'message'),
	[
		('{"value":', 'length', 'Expecting value'),
		('{"value":1} trailing', 'tool_calls', 'Extra data'),
		('{"value":"bad"}', 'tool_calls', 'int_parsing'),
		('', 'tool_calls', 'Expecting value'),
		pytest.param('x' * 20_001, 'length', 'Expecting value', id='long-invalid-reply'),
	],
)
async def test_tool_format_error_contains_original_without_retry(monkeypatch, raw, reason, message):
	model, create = model_with(monkeypatch, [response(raw, reason=reason)])
	with pytest.raises(ModelOutputFormatError) as caught:
		await model.ainvoke([UserMessage(content='Choose')], output_format=Answer)
	create.assert_awaited_once()
	(call,) = create.await_args_list
	request = call.kwargs
	assert request['max_tokens'] == 512 and request['temperature'] == 0.2
	assert request['tools'][0]['function']['name'] == 'Answer' and 'response_format' not in request
	error = caught.value
	assert error.raw_text == raw and error.finish_reason == reason and message in error.parse_error
	assert error.usage and (
		error.usage.prompt_tokens,
		error.usage.completion_tokens,
		error.usage.total_tokens,
		error.usage.prompt_cached_tokens,
	) == (10, 2, 12, 3)
	record = error.trace_record(role='planner', model=model.model)
	assert record['raw_text'] == raw[:20_000] and record['raw_chars'] == len(raw)
	assert record['raw_truncated'] == (len(raw) > 20_000) and record['planner_feedback_retry'] is False
	assert set(record).isdisjoint({'attempts', 'retried', 'same_as_previous'})


@pytest.mark.parametrize('raw', ['{"value":', '{"value":"bad"}'])
async def test_explicit_json_errors_stay_provider_errors(monkeypatch, raw):
	model, create = model_with(monkeypatch, [response(raw, json_mode=True)])
	with pytest.raises(ValidationError) as baseline:
		Answer.model_validate_json(raw)
	with pytest.raises(ModelProviderError) as caught:
		await model.ainvoke([], output_format=Answer, response_format={'type': 'json_object'})
	create.assert_awaited_once()
	(call,) = create.await_args_list
	assert call.kwargs['response_format'] == {'type': 'json_object'} and 'tools' not in call.kwargs
	assert type(caught.value) is ModelProviderError and str(caught.value) == str(baseline.value)
	assert caught.value.status_code == 502 and caught.value.model == model.model


@pytest.mark.parametrize(
	'explicit_json,tools,expect_json', [(True, None, True), (True, [{'type': 'function'}], False), (False, None, False)]
)
async def test_structured_routing(monkeypatch, explicit_json, tools, expect_json):
	model, create = model_with(monkeypatch, [response('{"value":7}', json_mode=expect_json)])
	reply = await model.ainvoke(
		[],
		output_format=Answer,
		tools=tools,
		response_format={'type': 'json_object'} if explicit_json else None,
	)
	assert reply.completion.value == 7 and reply.usage and reply.usage.total_tokens == 12
	create.assert_awaited_once()
	(call,) = create.await_args_list
	request = call.kwargs
	if expect_json:
		assert request['response_format'] == {'type': 'json_object'} and 'tools' not in request
	else:
		assert request['tools'][0]['function']['name'] == 'Answer' and 'response_format' not in request


@pytest.mark.parametrize('case', ['empty_content', 'missing_tools', 'rate_limit', 'connection'])
async def test_non_parse_failures_keep_existing_retry_policy(monkeypatch, case):
	req = httpx2.Request('POST', 'https://example.invalid/chat/completions')
	expected = ModelProviderError
	if case == 'empty_content':
		reply = response('', json_mode=True)
		message = 'Empty JSON content in DeepSeek response'
	elif case == 'missing_tools':
		reply = response(None)
		message = 'Expected tool_calls in response but got none'
	elif case == 'rate_limit':
		reply = RateLimitError('limited', response=httpx2.Response(429, request=req), body=None)
		message = str(reply)
		expected = ModelRateLimitError
	else:
		reply = APIConnectionError(request=req)
		message = str(reply)
	model, create = model_with(monkeypatch, [reply])
	with pytest.raises(expected) as caught:
		await model.ainvoke(
			[], output_format=Answer, response_format={'type': 'json_object'} if case == 'empty_content' else None
		)
	assert type(caught.value) is expected and str(caught.value) == message
	create.assert_awaited_once()


@pytest.mark.parametrize('role', ['planner', 'text', 'summary', 'answer', 'risk'])
async def test_roles_record_one_failed_physical_response(monkeypatch, tmp_path, role):
	model, create = model_with(monkeypatch, [response('bad response')])
	cost = TokenCost(include_cost=False)
	cost.register_llm(model)
	trace = JevTrace(tmp_path / 'trace.jsonl', 'test')
	ledger = UsageLedger(cost, trace)
	with pytest.raises(ModelOutputFormatError):
		await ledger.invoke(2, role, model, [], output_format=Answer)
	create.assert_awaited_once()
	await ledger.sync_other(3)
	events = [json.loads(line) for line in trace.path.read_text(encoding='utf-8').splitlines()]
	failed, call = events
	assert failed['event'] == 'llm_output_parse_error' and failed['planner_feedback_retry'] is False
	assert (failed['role'], failed['model'], failed['step'], failed['raw_text']) == (role, model.model, 2, 'bad response')
	assert call['event'] == 'llm_call' and call['prompt_tokens'] == 10 and not call['ok']
	row = ledger.summary()[role]
	assert (row['calls'], row['prompt_tokens'], row['completion_tokens'], row['reported_usage_calls']) == (1, 10, 2, 1)
	assert ledger.summary()['other']['calls'] == 0 and [e.usage.prompt_tokens for e in cost.usage_history] == [10]


@pytest.mark.parametrize('corrected', [True, False])
async def test_native_planner_feedback_after_invalid_json(monkeypatch, tmp_path, corrected):
	action = {'finish': {'success': True, 'reason': 'The observed result completes the task.'}}
	final = json.dumps({'memory': 'Review complete', 'action': [action]}) if corrected else 'broken again'
	model, create = model_with(monkeypatch, [response('broken'), response(final, tokens=30 if corrected else None)])
	monkeypatch.setattr(TokenCost, '_log_usage', AsyncMock())
	path = tmp_path / 'jev_trace.jsonl'
	agent = Agent(
		task='Review the current result',
		llm=model,
		jev=JevSettings(jev_api_key='local-test', trace_path=path),
		use_judge=False,
		use_vision=False,
		enable_signal_handler=False,
		file_system_path=str(tmp_path / 'files'),
	)
	assert agent._jev is not None
	try:
		output = await agent._get_model_output_with_retry([UserMessage(content='Review evidence')])
		assert create.await_count == 2
		first, second = [call.kwargs['messages'] for call in create.await_args_list]
		assert second[:-1] == first and 'Invalid JSON' in second[-1]['content']
		result = output.action[0].model_dump(exclude_unset=True)
		assert result['finish']['success'] is corrected
		if corrected:
			assert result == action
		else:
			assert 'Invalid JSON' in result['finish']['reason']
		ledger = agent._jev.usage
		await ledger.sync_other(4)
		row = ledger.summary()['planner']
		assert (row['calls'], row['prompt_tokens'], row['reported_usage_calls']) == (
			2,
			40 if corrected else 10,
			2 if corrected else 1,
		)
		assert ledger.summary()['other']['calls'] == 0
		assert [e.usage.prompt_tokens for e in agent.token_cost_service.usage_history] == ([10, 30] if corrected else [10])
	finally:
		await agent.eventbus.stop(clear=True, timeout=3)
		await agent._jev.aclose()
	rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
	failures = [r for r in rows if r['event'] == 'llm_output_parse_error' and r['role'] == 'planner']
	assert [r['raw_text'] for r in failures] == (['broken'] if corrected else ['broken', 'broken again'])
	assert [r['planner_feedback_retry'] for r in failures] == ([True] if corrected else [True, False])
	assert [r['prompt_tokens'] for r in rows if r['event'] == 'llm_call'] == [10, 30 if corrected else None]
