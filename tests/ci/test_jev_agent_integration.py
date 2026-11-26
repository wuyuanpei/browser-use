"""Agent entry points, planner output schemas and end-to-end effects on real local pages."""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError
from pytest_httpserver import HTTPServer

from browser_use import Agent, BrowserProfile, BrowserSession, ChatDeepSeek, Tools
from browser_use.agent.jev import JevSettings
from browser_use.agent.jev.state import read_visible_text
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import AgentHistoryList, AgentStepInfo
from browser_use.llm.exceptions import ModelProviderError
from tests.ci.conftest import create_mock_llm
from tests.ci.jev_support import FakeJev, serve_page, serve_two_page_site
from tests.ci.jev_support import jev_browser as _jev_browser_fixture  # noqa: F401
from tests.ci.jev_support import jev_httpserver as _threaded_server  # noqa: F401


def planner_action(name, **params):
	return json.dumps({'memory': 'Evaluate the visible evidence', 'action': [{name: params}]})


def subgoal(objective, completion, **kw):
	return planner_action('delegate_subgoal', objective=objective, completion=completion, reason='Proceed with this stage', **kw)


def finish(success=True):
	return planner_action('finish', success=success, reason='The requested evidence was reviewed.')


def isolated_browser():
	return BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			keep_alive=False,
			user_data_dir=None,
			executable_path=os.environ.get('JEV_TEST_CHROME_PATH'),
			enable_default_extensions=False,
		)
	)


def make_agent(task, outputs, settings, **kwargs):
	return Agent(
		task=task,
		llm=create_mock_llm(outputs),
		jev=settings,
		browser_session=kwargs.pop('browser_session', None) or isolated_browser(),
		use_judge=False,
		use_vision=False,
		enable_signal_handler=False,
		message_compaction=False,
		**kwargs,
	)


def action_names(output):
	schema = output.model_json_schema()

	def resolve(node):
		return schema['$defs'][node['$ref'].split('/')[-1]] if '$ref' in node else node

	item = resolve(schema['properties']['action']['items'])
	return {name for option in item.get('anyOf', [item]) for name in resolve(option)['properties']}


@pytest.mark.parametrize('flash,thinking', [(False, True), (False, False), (True, True)])
def test_planner_templates_and_agent_schema(flash, thinking, tmp_path):
	settings = JevSettings(jev_api_key='local-test', trace_path=tmp_path / 'trace.jsonl')
	a = make_agent('Check formats', [], settings, flash_mode=flash, use_thinking=thinking)
	assert {'navigate', 'click', 'finish', 'delegate_subgoal', 'collect_page'} <= action_names(a.AgentOutput)
	assert action_names(a._planner_output_format()) == {'delegate_subgoal', 'finish'}
	assert action_names(a.PlannerDoneAgentOutput) == {'finish'}
	assert not a.settings.enable_planning and a.settings.max_actions_per_step == 1
	for is_anthropic, is_bu in [(True, False), (False, True), (False, False)]:
		msg = (
			SystemPrompt(
				flash_mode=flash,
				use_thinking=thinking,
				is_anthropic=is_anthropic,
				is_browser_use_model=is_bu,
				jev_mode='isolated',
			)
			.get_system_message()
			.content
		)
		assert all(s in msg for s in ('delegate_subgoal', 'finish', 'review_note', 'STEP_DONE'))


def test_shared_tools_and_sensitive_data(tmp_path, monkeypatch):
	class Report(BaseModel):
		result: str

	opts = JevSettings(jev_api_key='local-test', trace_path=tmp_path / 'trace.jsonl')
	tools = Tools()
	first = make_agent('First', [], opts, tools=tools)
	second = make_agent('Second', [], opts, tools=tools)
	assert (
		first.tools.registry.registry.actions['delegate_subgoal'].function
		is not second.tools.registry.registry.actions['delegate_subgoal'].function
	)
	off = make_agent('Native', [], None, tools=first.tools)
	assert 'delegate_subgoal' not in action_names(off.AgentOutput)
	assert off._planner_output_format() is off.AgentOutput
	with pytest.raises(ValueError, match='isolated'):
		make_agent('Sensitive', [], opts, sensitive_data={'user': 'test'})
	with pytest.raises(ValueError, match='isolated'):
		make_agent('Schema', [], opts, output_model_schema=Report)
	collab = make_agent('Delegate', [], opts.model_copy(update={'mode': 'collaborative'}))
	assert {'click', 'delegate_subgoal', 'done'} <= action_names(collab._planner_output_format())
	assert not {'finish', 'collect_page'} & action_names(collab._planner_output_format())
	sensitive = make_agent('Sensitive', [], opts.model_copy(update={'mode': 'collaborative'}), sensitive_data={'user': 'test'})
	assert sensitive._jev is None
	monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
	with pytest.raises(ValueError, match='(?i)key'):
		make_agent('No key', [], JevSettings())


async def test_forced_finish_and_planner_retries(tmp_path, caplog):
	caplog.set_level('INFO')
	empty = json.dumps({'memory': 'No action', 'action': []})
	a = make_agent(
		'Review the current page',
		[empty, empty],
		JevSettings(
			jev_api_key='local-test',
			trace_path=tmp_path / 'trace.jsonl',
			answer_llm=create_mock_llm([json.dumps({'text': 'Unable to continue'})]),
		),
	)
	info = AgentStepInfo(step_number=3, max_steps=4)
	await a._force_done_after_last_step(info)
	assert a._jev is not None
	assert a._jev.finish_only and action_names(a._planner_output_format()) == {'finish'}
	assert 'click' in action_names(a.AgentOutput)
	await a._update_action_models_for_page('about:blank')
	assert not a._jev.finish_only
	# BU attaches its console handler to its own logger and disables propagation to root.
	agent_logs = logging.getLogger('browser_use')
	agent_logs.addHandler(caplog.handler)
	try:
		history = await a.run(max_steps=2)
	finally:
		agent_logs.removeHandler(caplog.handler)
	assert history.is_done() and history.is_successful() is False
	assert history.final_result() == 'Unable to continue'
	final_logs = [record.getMessage() for record in caplog.records if 'Final Result:' in record.getMessage()]
	assert len(final_logs) == 1 and 'Unable to continue' in final_logs[0]
	usage = a._jev.usage_by_role()['planner']
	assert (usage['decisions'], usage['calls'], usage['missing_usage_calls']) == (1, 2, 2)


@pytest.mark.parametrize('invalid', [subgoal('Read the page', 'Page visible', review_note_placeholder='invalid'), '{"action":'])
@pytest.mark.parametrize('recover', [True, False])
async def test_planner_repairs_malformed_deepseek_output(httpserver, tmp_path, monkeypatch, invalid, recover):
	serve_page(httpserver, '/repair', '<main>Visible evidence</main>')
	second = finish() if recover else invalid
	responses = [
		SimpleNamespace(
			usage=None,
			choices=[
				SimpleNamespace(message=SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments=raw))]))
			],
		)
		for raw in (invalid, second)
	]
	create = AsyncMock(side_effect=responses)
	monkeypatch.setattr(
		ChatDeepSeek, '_client', lambda self: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
	)
	agent = Agent(
		task='Read the page',
		llm=ChatDeepSeek(api_key='local-test'),
		jev=JevSettings(jev_api_key='local-test', answer_llm=create_mock_llm([json.dumps({'text': 'Reviewed'})])),
		browser_session=isolated_browser(),
		use_judge=False,
		enable_signal_handler=False,
		initial_actions=[{'navigate': {'url': httpserver.url_for('/repair')}}],
	)
	history = await agent.run(max_steps=2)
	assert create.await_count == 2
	request = create.await_args
	assert request is not None
	messages = json.dumps(request.kwargs['messages'])
	expected_diagnostic = 'review_note_placeholder' if 'review_note_placeholder' in invalid else 'Invalid JSON'
	assert expected_diagnostic in messages
	assert history.is_done() and history.is_successful() is recover
	assert not any('delegate_subgoal' in action for action in history.model_actions())


@pytest.mark.parametrize(
	'error', [TimeoutError('request timed out'), ModelProviderError('authentication failed', status_code=401)]
)
async def test_planner_does_not_reprompt_for_transport_or_auth_errors(monkeypatch, error):
	agent = make_agent('Read the page', [], JevSettings(jev_api_key='local-test'))
	request = AsyncMock(side_effect=error)
	monkeypatch.setattr(agent.llm, 'ainvoke', request)

	history = await agent.run(max_steps=1)

	assert request.await_count == 1
	assert not history.is_done()
	expected = 'timed out' if isinstance(error, TimeoutError) else 'authentication failed'
	assert any(expected in item for item in history.errors() if item)


@pytest.mark.parametrize('flash,thinking', [(False, True), (False, False), (True, True)])
async def test_public_run_save_and_replay(httpserver: HTTPServer, tmp_path: Path, flash, thinking):
	site = serve_two_page_site(httpserver)
	fake = FakeJev(
		httpserver,
		[
			{'operation': 'CLICK', 'target': 'Go to second page'},
			{'operation': 'CLICK', 'target': 'Save'},
			{'operation': 'STEP_DONE'},
		],
	)
	trace = tmp_path / 'trace.jsonl'
	answer = 'The browser shows Saved.'
	settings = JevSettings(
		jev_base_url=httpserver.url_for('/'),
		jev_api_key='local-test',
		trace_path=trace,
		answer_llm=create_mock_llm([json.dumps({'text': answer})]),
	)
	a = make_agent(
		f'Open {site.first} and save on the second page.',
		[subgoal('Save on the second page', 'Saved is visible', may_submit=True, requires_action=True), finish()],
		settings,
		flash_mode=flash,
		use_thinking=thinking,
	)
	visible = []

	async def capture_final(agent):
		if agent.history.is_done():
			visible.append((await read_visible_text(agent.browser_session))['text'])
			assert agent.history.final_result() == answer

	history = await a.run(max_steps=9, on_step_end=capture_final)
	assert not fake.errors and len(fake.requests) == 3
	assert len(site.save_requests) == 1 and len(visible) == 1 and 'Saved' in visible[0]
	assert history.final_result() == answer
	assert [h.metadata.decision_source for h in history.history if h.metadata and h.metadata.step_number > 0] == [
		'llm',
		'system1',
		'system1',
		'llm',
	]
	initial_output = history.history[0].model_output
	assert initial_output and initial_output.action[0].model_dump(exclude_none=True).get('navigate')
	events = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
	assert [e['status'] for e in events if e['event'] == 'action_result'] == ['executed_ok', 'executed_ok']
	assert events[-1]['event'] == 'usage_summary'
	if flash:
		history_file = tmp_path / 'history.json'
		history.save_to_file(history_file)
		assert len(AgentHistoryList.load_from_file(history_file, a.AgentOutput).history) == len(history.history)
		with pytest.raises(ValidationError):
			AgentHistoryList.load_from_file(history_file, a.PlannerAgentOutput)
		replay = make_agent(
			'Replay local save', [], settings.model_copy(update={'trace_path': tmp_path / 'replay.jsonl'}), flash_mode=True
		)
		await replay.load_and_rerun(history_file, summary_llm=create_mock_llm(), delay_between_actions=0, max_step_interval=0)
		assert len(site.save_requests) == 2


@pytest.mark.parametrize('entry_point', ['step', 'take_step'])
async def test_single_step_driver_has_final_answer(entry_point, httpserver, tmp_path):
	url = serve_page(httpserver, '/done', '<p>Verified on this page</p>')
	helper = create_mock_llm(['{"text":"Answer from the finishing page"}'])
	agent = make_agent(
		'Review the current page',
		[finish()],
		JevSettings(jev_api_key='test', answer_llm=helper, trace_path=tmp_path / 'trace.jsonl'),
	)
	assert agent._jev is not None
	try:
		await agent.browser_session.start()
		await agent.browser_session.navigate_to(url)
		await getattr(agent, entry_point)(AgentStepInfo(step_number=0, max_steps=4))
		assert agent.state.last_result and agent.state.last_result[-1].extracted_content == 'Answer from the finishing page'
		assert agent.history.final_result() == 'Answer from the finishing page'
		await agent._jev.finalize_if_done(agent)
		assert agent._jev.usage_by_role()['answer']['calls'] == 1
	finally:
		await agent.eventbus.stop(clear=True, timeout=3)
		await agent.close()


@pytest.mark.parametrize('mode', ['record', 'off'])
async def test_record_and_off_keep_reachable_business_effect(httpserver: HTTPServer, tmp_path: Path, mode):
	site = serve_two_page_site(httpserver)
	fake = FakeJev(
		httpserver,
		[
			{'operation': 'STEP_DONE'},
			{'operation': 'CLICK', 'target': 'Go to second page', 'p': 0.65, 'confidence': 0.99},
			{'operation': 'CLICK', 'target': 'Save'},
			{'operation': 'STEP_DONE'},
		],
	)
	work = 'Save on the second page'
	condition = 'Saved is visible'
	trace = tmp_path / 'trace.jsonl'
	agent = make_agent(
		f'Open {site.first} and save.',
		[
			subgoal(work, condition, requires_action=True, may_submit=True),
			subgoal(work, condition, requires_action=True, may_submit=True, review_note='The Save button was never pressed.'),
			finish(),
		],
		JevSettings(
			jev_api_key='test',
			jev_base_url=httpserver.url_for('/'),
			checks=mode,
			trace_path=trace,
			answer_llm=create_mock_llm(['{"text":"Saved"}']),
		),
	)
	observation = []

	async def verify(at_step):
		if at_step.history.is_done():
			observation.append((await read_visible_text(at_step.browser_session))['text'])

	history = await agent.run(max_steps=10, on_step_end=verify)
	assert not fake.errors and len(fake.requests) == 4
	assert len(site.save_requests) == 1 and observation and 'Saved' in observation[0]
	assert history.final_result() == 'Saved'
	events = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
	checks = [e for e in events if e['event'] == 'check']
	if mode == 'record':
		assert {'confidence', 'completion'} <= {e['name'] for e in checks if e['outcome'] == 'would_block'}
	else:
		assert checks == []


async def test_off_uses_native_done_without_jev_trace(httpserver: HTTPServer, tmp_path: Path, jev_browser):
	url = serve_page(httpserver, '/native', '<p>Native hello</p>')
	path = tmp_path / 'disabled.jsonl'
	agent = make_agent(
		f'Open {url} and report the greeting',
		[planner_action('done', text='Native hello', success=True)],
		JevSettings(mode='off', trace_path=path),
		browser_session=jev_browser,
	)
	history = await agent.run(max_steps=3)
	assert history.is_successful() and history.final_result() == 'Native hello'
	assert all(h.metadata.decision_source == 'llm' for h in history.history if h.metadata)
	assert not path.exists()
