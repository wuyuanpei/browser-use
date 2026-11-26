import json

import pytest

from browser_use.agent.jev import JevChecks
from browser_use.agent.jev import decision as d
from browser_use.agent.jev import questions as prompts
from browser_use.agent.jev.views import ActionRecord, Candidate, DelegateSubgoalParams, Observation, SubgoalRun, System1Reply
from browser_use.tools.service import Tools
from browser_use.tools.views import NoParamsAction


@pytest.fixture
def obs():
	field = Candidate(
		key='2',
		kind='element',
		index=2,
		role='textbox',
		name='Town',
		tag='input',
		input_type='text',
		placeholder='Where?',
		editable=True,
		value='',
		context={'form': 'Search'},
	)
	return Observation(
		step=1,
		url='https://example.org/',
		title='Search',
		scroll={'y': 0},
		visible_text='hello',
		links=[],
		candidates={
			'CLICK': [
				Candidate(key='1', kind='element', index=1, role='button', name='Find'),
				Candidate(key='tab:1', kind='tab', tab_id='B123', role='tab', name='Second', href='https://example.org/b'),
			],
			'TYPE_TEXT': [field],
			'SELECT': [
				Candidate(key='3:2', kind='option', index=3, role='option', name='Other', option_text='Other', match_text='2')
			],
			'SCROLL_UP': [Candidate(key='9', kind='scroll', index=9, role='region', name='Results')],
			'SCROLL_DOWN': [Candidate(key='page', kind='scroll', role='region', name='Page')],
		},
		tabs=[{'key': 'tab:1', 'tab_id': 'B123', 'title': 'Second', 'url': 'https://example.org/b'}],
		can_go_back=True,
		is_pdf=False,
		state_error=None,
		fingerprint='a',
	)


@pytest.fixture
def run():
	return SubgoalRun(
		id='sg2',
		params=DelegateSubgoalParams(
			objective='Find town',
			completion='Results visible',
			reason='Business goal',
			inputs='Town=Zürich',
			review_note='Suggestions still missing',
		),
		identity='Find town\nResults visible',
		max_actions=15,
		started_step=1,
		start_url='https://example.org',
		start_site='example.org',
	)


def record(n, source='sg2', status='executed_ok'):
	return ActionRecord.model_validate(
		dict(
			n=n,
			step=n,
			run_id=source,
			operation='TYPE_TEXT',
			url_before='https://example.org',
			fingerprint_before='before',
			bu_action={'input': {'index': 2, 'text': 'Zürich'}},
			text='Zürich',
			target_label='[2] textbox "Town"',
			status=status,
			page_changed=True,
		)
	)


def test_operation_availability_and_registry_filtering(obs, run):
	modes = JevChecks()
	all_tools = d.SYSTEM1_ACTIONS
	ops = d.offered_operations(obs, run, all_tools, modes)
	assert set(ops) == {op for op in d.ENABLED_OPERATIONS if op not in ('COLLECT', 'PRESS_ENTER')}
	field = obs.candidates['TYPE_TEXT'][0]
	assert set(d.offered_operations(obs, run, all_tools, modes, collect_available=True, enter_field=field)) == set(
		d.ENABLED_OPERATIONS
	)
	assert 'PRESS_ENTER' not in d.offered_operations(obs, run, ['input'], modes, enter_field=field)
	assert 'PRESS_ENTER' not in d.offered_operations(obs, run, all_tools, modes)
	assert d.offered_operations(obs, run, [], modes) == ['STEP_DONE', 'UNABLE']
	ops = d.offered_operations(obs, run, {'switch'}, modes)
	assert list(d.build_questions(obs, run, ops, registered_actions={'switch'})['click_target']['criteria']) == ['tab:1']
	ops = d.offered_operations(obs, run, {'click'}, modes)
	assert list(d.build_questions(obs, run, ops, registered_actions={'click'})['click_target']['criteria']) == ['1']
	obs.can_go_back = False
	obs.candidates.clear()
	assert set(d.offered_operations(obs, run, all_tools, modes)) == {'WAIT', 'STEP_DONE', 'UNABLE'}


@pytest.mark.parametrize('trigger', ['requires_action', 'rejected_fingerprint'])
def test_completion_rules(trigger, obs, run):
	if trigger == 'requires_action':
		run.params.requires_action = True
	else:
		run.rejected_fingerprint = obs.fingerprint
	assert d.completion_blocked(obs, run)
	for mode in ('enforce', 'record', 'off'):
		gate = JevChecks.model_validate({'default': mode})
		assert ('STEP_DONE' in d.offered_operations(obs, run, d.SYSTEM1_ACTIONS, gate)) == (mode != 'enforce')
	run.actions.append(record(1, status='discarded'))
	assert d.completion_blocked(obs, run)
	run.actions.append(record(2))
	run.applied = True
	assert not d.completion_blocked(obs, run)


def test_state_visible_evidence_and_history(obs, run):
	obs.visible_text = 'first row\n' + ('x' * 6000)
	run.params.requires_action = True
	run.actions = [record(1, status='discarded'), record(2)]
	all_actions = [record(n, 'sg1') for n in range(1, 12)] + [record(12, 'sg1', 'discarded')] + run.actions
	research = {'collect_available': False, 'collected_pages': [{'id': 'p1', 'summary': 'first'}]}
	s = d.build_state(obs, run, 'full goal', all_actions, research)
	assert list(s) == ['research', 'page', 'original_goal', 'current_subgoal', 'tabs', 'segment_actions', 'task_actions']
	assert s['page']['text'] == 'first row\n[truncated]' and s['original_goal'] == 'full goal'
	assert s['current_subgoal']['pending_apply'] and s['current_subgoal']['actions_used'] == 1
	assert s['segment_actions'][0]['n'] == 2 and s['segment_actions'][0]['result'] == 'ok'
	assert len(s['task_actions']) == 10 and all(a['run'] == 'sg1' for a in s['task_actions'])
	assert 'tab_id' not in s['tabs'][0]
	run.applied = True
	run.actions.clear()
	obs.tabs.clear()
	s = d.build_state(obs, run, 'goal', [], {})
	assert list(s) == ['page', 'original_goal', 'current_subgoal'] and 'pending_apply' not in s['current_subgoal']


def test_questions_and_chosen_evidence(obs, run):
	run.params.requires_action = True
	offered = d.offered_operations(obs, run, d.SYSTEM1_ACTIONS, JevChecks(default='record'), collect_available=True)
	q = d.build_questions(obs, run, offered)
	assert set(q) == {'operation', 'type_text_target', 'click_target', 'select_target', 'scroll_up_target', 'scroll_down_target'}
	instructions = q['operation']['instructions']
	lines = instructions.splitlines()
	assert lines[:5] == [
		'Current objective: Find town',
		'Completion: Results visible',
		'Inputs: Town=Zürich',
		'Review note: Suggestions still missing',
		prompts.PENDING_APPLY,
	]
	assert (
		prompts.TAB_RULE in instructions
		and prompts.COLLECTION_RULES in instructions
		and prompts.PRESS_ENTER_RULE not in instructions
	)
	assert prompts.COLLECTION_BOUNDARY in q['click_target']['instructions']
	assert prompts.TYPE_TEXT_TARGET_RULE in q['type_text_target']['instructions']
	assert q['operation']['criteria']['CLICK']['current_targets'] == [
		'[1] button "Find"',
		'[tab:1] browser tab "Second" https://example.org/b',
	]
	assert q['type_text_target']['criteria']['2'] == {
		'element': '[2] textbox "Town" type=text placeholder="Where?"',
		'role': 'textbox',
		'current_value': '',
		'context': {'form': 'Search'},
	}
	state = {'page': '非 ASCII'}
	assert d.request_size(state, q) == len(json.dumps({'state': state, 'questions': q}, ensure_ascii=False))
	assert d.request_size(state, q, 'jev-latest') == len(
		json.dumps({'model': 'jev-latest', 'state': state, 'questions': q}, ensure_ascii=False)
	)


def test_operation_questions_distinguish_focus_from_editing(obs, run):
	field = obs.candidates['TYPE_TEXT'][0]
	field.states['focused'] = True
	obs.candidates['CLICK'].append(field)
	questions = d.build_questions(obs, run, ['CLICK', 'TYPE_TEXT'])
	for operation in ('CLICK', 'TYPE_TEXT'):
		assert questions['operation']['criteria'][operation]['current_targets'][-1] == (
			f'{field.element()} (editable and already focused)'
		)
		assert questions[f'{operation.lower()}_target']['criteria'][field.key]['states']['focused'] is True


@pytest.mark.parametrize(
	'answer',
	[
		None,
		{},
		{'choice': None},
		{'choice': 'Z'},
		{'choice': 'A'},
		{'choice': 'A', 'confidence': 1, 'probabilities': {'A': 1}},
		{'choice': 'A', 'confidence': 1, 'probabilities': {'A': True, 'B': False}},
		{'choice': 'A', 'confidence': False, 'probabilities': {'A': 1, 'B': 0}},
		{'choice': 'A', 'confidence': float('inf'), 'probabilities': {'A': 1, 'B': 0}},
		{'choice': 'A', 'confidence': 1, 'probabilities': {'A': float('nan'), 'B': 0}},
		{'choice': 'A', 'confidence': 1, 'probabilities': {'A': 1.01, 'B': -0.01}},
		{'choice': 'A', 'confidence': 0.3, 'probabilities': {'A': 0.45, 'B': 0.45}},
		{'choice': 'A', 'confidence': 0.9, 'probabilities': {'A': 0.45, 'B': 0.55}},
	],
)
def test_invalid_chosen_answers(answer):
	with pytest.raises(d.InvalidAnswer):
		d.validate_answer(answer, ['A', 'B'], True)


def test_combined_decision_uses_selected_probabilities(obs, run):
	q = d.build_questions(obs, run, ['CLICK', 'SELECT', 'UNABLE'])
	answers = {
		'operation': {'choice': 'CLICK', 'probabilities': {'CLICK': 0.75, 'SELECT': 0, 'UNABLE': 0.25}, 'confidence': 0.95},
		'click_target': {'choice': '1', 'probabilities': {'1': 0.5, 'tab:1': 0.5}, 'confidence': 0.78},
		'select_target': {'wrong': 'ignore me'},
	}
	reply = System1Reply(answers=answers, latency_ms=1, attempts=[], has_probabilities=True)
	chosen = d.combine(reply, q, obs, unable_threshold=0.25)
	assert chosen.target is not None
	assert chosen.choice_is_argmax == {'operation': True, 'click_target': True, 'select_target': False}
	assert chosen.target.key == '1' and (chosen.operation_probability, chosen.target_probability, chosen.unable_probability) == (
		0.75,
		0.5,
		0.25,
	)
	reply.answers['operation']['probabilities'].update(CLICK=0.74, UNABLE=0.26)
	reply.answers['click_target'] = {'choice': 'invalid'}
	reply.answers['select_target'] = None
	assert d.combine(reply, q, obs, unable_threshold=0.25).target is None
	with pytest.raises(d.InvalidAnswer):
		d.combine(reply, q, obs)
	llmreply = System1Reply(
		answers={'operation': {'choice': 'CLICK'}, 'click_target': {'choice': '1'}},
		latency_ms=1,
		attempts=[],
		has_probabilities=False,
	)
	chosen = d.combine(llmreply, q, obs)
	assert chosen.target is not None
	assert chosen.target.key == '1' and chosen.operation_probability is None and chosen.choice_is_argmax is None


def test_native_mapping_and_memory(obs):
	tools = Tools()

	@tools.registry.action('Save current page text', param_model=NoParamsAction)
	async def collect_page(params: NoParamsAction):
		return 'saved'

	model = tools.registry.create_action_model(include_actions=d.SYSTEM1_ACTIONS)
	cases = [
		('CLICK', obs.candidates['CLICK'][0], None, {'click': {'index': 1}}),
		('CLICK', obs.candidates['CLICK'][1], None, {'switch': {'tab_id': 'B123'}}),
		('TYPE_TEXT', obs.candidates['TYPE_TEXT'][0], 'Zurich', {'input': {'index': 2, 'clear': True, 'text': 'Zurich'}}),
		('SELECT', obs.candidates['SELECT'][0], None, {'select_dropdown': {'index': 3, 'text': '2'}}),
		('SCROLL_DOWN', obs.candidates['SCROLL_DOWN'][0], None, {'scroll': {'down': True, 'pages': 0.8}}),
		('SCROLL_UP', obs.candidates['SCROLL_UP'][0], None, {'scroll': {'down': False, 'pages': 0.8, 'index': 9}}),
		('WAIT', None, None, {'wait': {'seconds': 2}}),
		('GO_BACK', None, None, {'go_back': {}}),
		('PRESS_ENTER', None, None, {'send_keys': {'keys': 'Enter'}}),
		('COLLECT', None, None, {'collect_page': {}}),
	]
	for op, candidate, text, expected in cases:
		action = d.to_bu_action(op, candidate, text)
		assert action == expected
		assert model.model_validate(action).model_dump(exclude_unset=True) == expected
	for op in ('STEP_DONE', 'UNABLE'):
		with pytest.raises(ValueError):
			d.to_bu_action(op)
	target = Candidate(key='1530', kind='element', index=1530, role='option', name='Zürich, Switzerland (ZRH)')
	assert (
		d.memory_line('sg1', 'CLICK', target, None, 0.96, 0.91)
		== 'System1 sg1: CLICK [1530] option "Zürich, Switzerland (ZRH)" (operation p=0.96, target p=0.91)'
	)
	assert d.memory_line('sg2', 'WAIT', None, None, 0.81, None) == 'System1 sg2: WAIT (operation p=0.81)'
	assert (
		d.memory_line('sg2', 'SCROLL_DOWN', obs.candidates['SCROLL_DOWN'][0], None, None, None)
		== 'System1 sg2: SCROLL_DOWN [page] the whole page (operation p=n/a, target p=n/a)'
	)
