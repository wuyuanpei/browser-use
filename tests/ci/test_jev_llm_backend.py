import json

import pytest

from browser_use.agent.jev import JevSettings
from browser_use.agent.jev.backends import LLMBackend, System1Error, make_backend
from browser_use.agent.jev.questions import LLM_SYSTEM1_INSTRUCTION
from tests.ci.conftest import create_mock_llm

STATE = {'original_goal': '打开详情'}
QUESTIONS = {
	'operation': {'type': 'choice', 'criteria': {'CLICK': 'Click', 'STEP_DONE': 'Review'}, 'instructions': 'Choose'},
	'click_target': {
		'type': 'choice',
		'criteria': {'4': {'element': 'details'}, '5': {'element': 'help'}},
		'instructions': 'Choose',
	},
}


@pytest.mark.parametrize(
	('two_stage', 'script', 'batches'),
	[
		(False, [{'operation': 'CLICK', 'click_target': '4'}], [['operation', 'click_target']]),
		(True, [{'operation': 'CLICK'}, {'click_target': '4'}], [['operation'], ['click_target']]),
		(True, [{'operation': 'STEP_DONE'}], [['operation']]),
	],
)
async def test_calls_only_required_questions(two_stage, script, batches):
	llm = create_mock_llm([json.dumps(entry) for entry in script])
	captured = []

	async def invoke(model, messages, **kwargs):
		assert model is llm and messages[0].content == LLM_SYSTEM1_INSTRUCTION
		captured.append(json.loads(messages[1].content))
		return await model.ainvoke(messages, **kwargs)

	settings = JevSettings(system1='llm', system1_llm=llm, system1_llm_two_stage=two_stage)
	client = make_backend(settings, None, llm, invoke=invoke)
	reply = await client.ask(STATE, QUESTIONS)
	assert client.name == 'llm' and not reply.has_probabilities
	assert reply.answers == {k: {'choice': v} for step in script for k, v in step.items()}
	assert len(reply.attempts) == len(batches)
	assert captured == [{'state': STATE, 'questions': {name: QUESTIONS[name] for name in batch}} for batch in batches]
	await client.aclose()


@pytest.mark.parametrize('action', ['{"operation":"CLICK","click_target":"not offered"}', '{}', 'not JSON'])
async def test_invalid_structured_choice(action):
	llm = create_mock_llm([action])
	client = LLMBackend(JevSettings(system1='llm', system1_llm=llm))
	with pytest.raises(System1Error) as exc:
		await client.ask(STATE, QUESTIONS)
	assert exc.value.attempts[0]['error_type']
