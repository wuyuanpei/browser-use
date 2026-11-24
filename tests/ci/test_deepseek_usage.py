import pytest
from pydantic import BaseModel
from pytest_httpserver import HTTPServer

from browser_use.llm.deepseek.chat import ChatDeepSeek
from browser_use.llm.messages import UserMessage


class Answer(BaseModel):
	answer: str


@pytest.mark.parametrize('cache_field', ['prompt_cache_hit_tokens', 'prompt_tokens_details', None])
async def test_deepseek_tool_call_usage(httpserver: HTTPServer, cache_field: str | None):
	response = {
		'id': 'local-response',
		'created': 1,
		'object': 'chat.completion',
		'model': 'deepseek-flash',
		'choices': [
			{
				'index': 0,
				'finish_reason': 'tool_calls',
				'message': {
					'role': 'assistant',
					'content': None,
					'tool_calls': [
						{'id': 'local-call', 'type': 'function', 'function': {'name': 'Answer', 'arguments': '{"answer":"ok"}'}}
					],
				},
			}
		],
	}
	if cache_field:
		response['usage'] = {
			'prompt_tokens': 1200,
			'completion_tokens': 80,
			'total_tokens': 1280,
			**(
				{'prompt_cache_hit_tokens': 1000, 'prompt_cache_miss_tokens': 200}
				if cache_field == 'prompt_cache_hit_tokens'
				else {'prompt_tokens_details': {'cached_tokens': 1000}}
			),
		}
	httpserver.expect_oneshot_request('/chat/completions', method='POST').respond_with_json(response)
	model = ChatDeepSeek(base_url=httpserver.url_for('/'), api_key='local-test')
	reply = await model.ainvoke([UserMessage(content='Answer the question')], output_format=Answer)
	assert reply.completion.answer == 'ok'
	if cache_field is None:
		assert reply.usage is None
	else:
		assert reply.usage is not None
		assert reply.usage.prompt_tokens == 1200
		assert reply.usage.completion_tokens == 80
		assert reply.usage.total_tokens == 1280
		assert reply.usage.prompt_cached_tokens == 1000
