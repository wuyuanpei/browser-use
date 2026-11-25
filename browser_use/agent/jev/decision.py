"""Prepare the System1 observation and questions, then interpret its selected operation."""

import json
import math
import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Protocol, cast, get_args

from browser_use.agent.jev import questions as wording
from browser_use.agent.jev.views import (
	MAX_CHOICE_OPTIONS,
	PAGE_TEXT_LIMIT,
	SCROLL_PAGES,
	SEGMENT_ACTIONS_LIMIT,
	TASK_ACTIONS_LIMIT,
	WAIT_SECONDS,
	ActionRecord,
	Candidate,
	CheckMode,
	CheckName,
	JevModel,
	Observation,
	Operation,
	SubgoalRun,
	System1Reply,
)

ENABLED_OPERATIONS: tuple[Operation, ...] = get_args(Operation)
SYSTEM1_ACTIONS = ['click', 'input', 'select_dropdown', 'scroll', 'wait', 'go_back', 'send_keys', 'switch', 'collect_page']
TARGET_OPERATIONS = frozenset({'CLICK', 'TYPE_TEXT', 'SELECT', 'SCROLL_UP', 'SCROLL_DOWN'})
_READ_OPERATIONS = frozenset({'COLLECT', 'SCROLL_UP', 'SCROLL_DOWN'})
_NATIVE_ACTIONS = {
	'TYPE_TEXT': 'input',
	'SELECT': 'select_dropdown',
	'SCROLL_UP': 'scroll',
	'SCROLL_DOWN': 'scroll',
	'WAIT': 'wait',
	'GO_BACK': 'go_back',
	'PRESS_ENTER': 'send_keys',
	'COLLECT': 'collect_page',
}


class _Gate(Protocol):
	def mode(self, name: CheckName) -> CheckMode: ...


class InvalidAnswer(ValueError):
	"""A selected answer violates the backend's choice contract."""


class Decision(JevModel):
	operation: Operation
	operation_probability: float | None
	read_operation_probability: float | None = None
	target: Candidate | None = None
	target_probability: float | None = None
	unable_probability: float | None = None
	choice_is_argmax: dict[str, bool] | None = None


def completion_blocked(obs: Observation, run: SubgoalRun) -> bool:
	return (run.params.requires_action and not run.applied) or (
		run.rejected_fingerprint is not None and run.rejected_fingerprint == obs.fingerprint and not run.executed_actions
	)


def _targets(obs: Observation, op: str, registered_actions: Collection[str] | None) -> list[Candidate]:
	targets = obs.candidates.get(op, [])
	if op == 'CLICK' and registered_actions is not None:
		return [c for c in targets if ('switch' if c.kind == 'tab' else 'click') in registered_actions]
	return targets


def offered_operations(
	obs: Observation, run: SubgoalRun, registered_actions: Collection[str], gate: _Gate, *, collect_available: bool = False
) -> list[Operation]:
	executed = run.executed_actions
	availability = {
		'GO_BACK': obs.can_go_back,
		'PRESS_ENTER': bool(executed and executed[-1].operation == 'TYPE_TEXT' and executed[-1].status == 'executed_ok'),
		'COLLECT': collect_available,
		'STEP_DONE': gate.mode('completion') != 'enforce' or not completion_blocked(obs, run),
	}
	offered = []
	for op in ENABLED_OPERATIONS:
		native = _NATIVE_ACTIONS.get(op)
		if native and native not in registered_actions:
			continue
		if op in TARGET_OPERATIONS and not _targets(obs, op, registered_actions):
			continue
		if availability.get(op, True):
			offered.append(op)
	return offered


def page_text(text: str, limit: int = PAGE_TEXT_LIMIT) -> str:
	if len(text) <= limit:
		return text
	marker = '\n[truncated]'
	prefix = text[: limit - len(marker)]
	# A single exceptionally long line has no useful boundary to preserve.
	return (prefix.rsplit('\n', 1)[0] if '\n' in prefix else prefix) + marker


def action_state(record: ActionRecord, *, include_run: bool = False) -> dict[str, Any]:
	row: dict[str, Any] = {'run': record.run_id} if include_run else {'n': record.n}
	row['operation'] = record.operation
	if record.target_label:
		row['target'] = record.target_label
	if record.text is not None:
		row['text'] = record.text
	row['result'] = (
		'ok'
		if record.status == 'executed_ok'
		else f'error: {(record.error or "")[:120]}'
		if record.status == 'executed_failed'
		else record.status
	)
	if record.page_changed is not None:
		row['page_changed'] = record.page_changed
	return row


def build_state(
	obs: Observation, run: SubgoalRun, task: str, task_actions: list[ActionRecord], research: dict[str, Any] | None = None
) -> dict[str, Any]:
	state: dict[str, Any] = {}
	if research and (research.get('collect_available') or research.get('collected_pages')):
		state['research'] = research
	state['page'] = {'url': obs.url, 'title': obs.title, 'scroll': obs.scroll, 'text': page_text(obs.visible_text)}
	state['original_goal'] = task
	params = run.params
	subgoal: dict[str, Any] = {'id': run.id, 'objective': params.objective, 'completion': params.completion}
	if params.inputs:
		subgoal['inputs'] = params.inputs
	subgoal.update(may_submit=params.may_submit, requires_action=params.requires_action, text_mode=params.text_mode)
	if params.requires_action and not run.applied:
		subgoal['pending_apply'] = True
	if params.review_note:
		subgoal['review_note'] = params.review_note
	executed = run.executed_actions
	subgoal.update(actions_used=len(executed), max_actions=run.max_actions)
	state['current_subgoal'] = subgoal
	if obs.tabs:
		state['tabs'] = [{k: tab[k] for k in ('key', 'title', 'url')} for tab in obs.tabs]
	if executed:
		state['segment_actions'] = [action_state(r) for r in executed[-SEGMENT_ACTIONS_LIMIT:]]
	prior = [r for r in task_actions if r.executed and r.run_id != run.id]
	if prior:
		state['task_actions'] = [action_state(r, include_run=True) for r in prior[-TASK_ACTIONS_LIMIT:]]
	return state


def _candidate_terms(text: str) -> set[str]:
	text = unicodedata.normalize('NFKC', text).casefold()
	terms = set(re.findall(r'[^\W\d_\u3400-\u9fff]{2,}|\d+', text))
	for run in re.findall(r'[\u3400-\u9fff]+', text):
		terms.update(run[i : i + size] for size in (2, 3, 4) for i in range(len(run) - size + 1))
	return terms


def _limit_targets(targets: list[Candidate], maximum: int, goal_terms: set[str]) -> list[Candidate]:
	if len(targets) <= maximum:
		return targets

	def priority(index: int) -> tuple[bool, int, bool]:
		candidate = targets[index]
		label = _candidate_terms(f'{candidate.name} {candidate.placeholder or ""}')
		scope = _candidate_terms(' '.join(candidate.context.values()))
		return (
			candidate.key == 'page',
			2 * sum(len(term) for term in label & goal_terms) + len(scope & goal_terms),
			candidate.editable or candidate.kind == 'tab',
		)

	selected = sorted(range(len(targets)), key=priority, reverse=True)[:maximum]
	return [targets[index] for index in sorted(selected)]


def build_questions(
	obs: Observation,
	run: SubgoalRun,
	offered: list[Operation],
	*,
	research: dict[str, Any] | None = None,
	registered_actions: Collection[str] | None = None,
	max_candidates: int = MAX_CHOICE_OPTIONS,
) -> dict[str, dict[str, Any]]:
	params = run.params
	goal_terms = _candidate_terms(f'{params.objective} {params.completion}')
	context = [f'Current objective: {params.objective}', f'Completion: {params.completion}']
	if params.text_mode == 'append':
		context.append('Append only the requested new text to the current field.')
	if params.inputs:
		context.append(f'Inputs: {params.inputs}')
	if params.review_note:
		context.append(f'Review note: {params.review_note}')
	if params.requires_action and not run.applied:
		context.append(wording.PENDING_APPLY)
	research_active = 'COLLECT' in offered or bool(
		research and (research.get('collect_available') or research.get('collected_pages'))
	)
	rules = [wording.OPERATION_RULES]
	for ops, rule in (
		(('TYPE_TEXT', 'SELECT'), wording.INPUT_RULES),
		(('SCROLL_UP', 'SCROLL_DOWN'), wording.SCROLL_RULES),
		(('WAIT',), wording.WAIT_RULE),
		(('GO_BACK',), wording.GO_BACK_RULE),
		(('PRESS_ENTER',), wording.PRESS_ENTER_RULE),
	):
		if any(op in offered for op in ops):
			rules.append(rule)
	if research_active:
		rules.append(wording.COLLECTION_RULES)
	if 'CLICK' in offered and any(c.kind == 'tab' for c in _targets(obs, 'CLICK', registered_actions)):
		rules.append(wording.TAB_RULE)
	criteria: dict[str, Any] = {}
	questions: dict[str, dict[str, Any]] = {
		'operation': {'type': 'choice', 'criteria': criteria, 'instructions': '\n'.join(context + rules)}
	}
	for op in offered:
		description = wording.OPERATION_DESCRIPTIONS[op]
		criteria[op] = description
		if op not in TARGET_OPERATIONS:
			continue
		targets = _limit_targets(_targets(obs, op, registered_actions), max_candidates, goal_terms)
		if op in ('CLICK', 'TYPE_TEXT', 'SELECT'):
			criteria[op] = {'description': description, 'current_targets': [c.element() for c in targets]}
		target_rules = [f'Operation: {op}', wording.TARGET_RULES]
		if op == 'TYPE_TEXT':
			target_rules.append(wording.TYPE_TEXT_TARGET_RULE)
		if op == 'CLICK' and research_active:
			target_rules.append(wording.COLLECTION_BOUNDARY)
		questions[f'{op.lower()}_target'] = {
			'type': 'choice',
			'criteria': {c.key: c.criterion() for c in targets},
			'instructions': '\n'.join(context + target_rules),
		}
	return questions


def request_size(state: dict[str, Any], questions: dict[str, dict[str, Any]], model: str | None = None) -> int:
	body = (
		{'model': model, 'state': state, 'questions': questions}
		if model is not None
		else {'state': state, 'questions': questions}
	)
	return len(json.dumps(body, ensure_ascii=False, allow_nan=False))


def _is_probability(value: Any) -> bool:
	return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def validate_answer(answer: Any, keys: Iterable[str], has_probabilities: bool) -> tuple[str, float | None]:
	allowed = set(keys)
	if not isinstance(answer, dict) or not isinstance(choice := answer.get('choice'), str) or choice not in allowed:
		raise InvalidAnswer('Choice must be one of the offered option keys')
	if not has_probabilities:
		return choice, None
	probs = answer.get('probabilities')
	if not isinstance(probs, dict) or set(probs) != allowed:
		raise InvalidAnswer('Probability keys do not match the offered options')
	if not all(_is_probability(p) for p in probs.values()) or not _is_probability(answer.get('confidence')):
		raise InvalidAnswer('Probabilities and confidence must be finite numbers in [0, 1]')
	if abs(math.fsum(probs.values()) - 1) >= 0.02:
		raise InvalidAnswer('Probabilities must sum to one')
	if probs[choice] + 1e-6 < max(probs.values()):
		raise InvalidAnswer('The selected option must have the highest probability')
	return choice, float(probs[choice])


def argmax_flags(reply: System1Reply) -> dict[str, bool] | None:
	"""Describe each answer for the trace, including unused questions, without applying a gate."""
	if not reply.has_probabilities:
		return None
	flags = {}
	for name, answer in reply.answers.items():
		choice = answer.get('choice') if isinstance(answer, dict) else None
		probs = answer.get('probabilities') if isinstance(answer, dict) else None
		flags[name] = bool(
			isinstance(choice, str)
			and isinstance(probs, dict)
			and choice in probs
			and all(_is_probability(p) for p in probs.values())
			and probs[choice] + 1e-6 >= max(probs.values())
		)
	return flags


def combine(
	reply: System1Reply, questions: Mapping[str, dict[str, Any]], obs: Observation, *, unable_threshold: float | None = None
) -> Decision:
	op_answer = reply.answers.get('operation')
	choice, probability = validate_answer(op_answer, questions['operation']['criteria'], reply.has_probabilities)
	op = cast(Operation, choice)
	probabilities = op_answer['probabilities'] if reply.has_probabilities and isinstance(op_answer, dict) else {}
	unable_p = probabilities.get('UNABLE')
	result = Decision(
		operation=op,
		operation_probability=probability,
		read_operation_probability=math.fsum(probabilities.get(name, 0) for name in _READ_OPERATIONS)
		if probabilities and op in _READ_OPERATIONS
		else None,
		unable_probability=unable_p,
		choice_is_argmax=argmax_flags(reply),
	)
	if op not in TARGET_OPERATIONS or (unable_threshold is not None and unable_p is not None and unable_p > unable_threshold):
		return result
	name = f'{op.lower()}_target'
	if name not in questions:
		raise InvalidAnswer('The selected operation has no target question')
	target, result.target_probability = validate_answer(
		reply.answers.get(name), questions[name]['criteria'], reply.has_probabilities
	)
	result.target = next((c for c in obs.candidates.get(op, []) if c.key == target), None)
	if result.target is None:
		raise InvalidAnswer('The chosen target is missing from the observation')
	return result


def to_bu_action(
	op: Operation, candidate: Candidate | None = None, text: str | None = None, *, append: bool = False
) -> dict[str, dict[str, Any]]:
	if op in TARGET_OPERATIONS and candidate is None:
		raise ValueError(f'{op} requires a target')
	if op == 'CLICK' and candidate is not None:
		if candidate.kind == 'tab' and candidate.tab_id:
			return {'switch': {'tab_id': candidate.tab_id}}
		if candidate.kind == 'element' and candidate.index is not None:
			return {'click': {'index': candidate.index}}
	if op == 'TYPE_TEXT' and candidate is not None and candidate.index is not None and text is not None:
		return {'input': {'index': candidate.index, 'text': text, 'clear': not append}}
	if op == 'SELECT' and candidate is not None and candidate.index is not None and candidate.match_text is not None:
		return {'select_dropdown': {'index': candidate.index, 'text': candidate.match_text}}
	if op in ('SCROLL_UP', 'SCROLL_DOWN') and candidate is not None:
		params = {'down': op == 'SCROLL_DOWN', 'pages': SCROLL_PAGES}
		if candidate.index is not None:
			params['index'] = candidate.index
		return {'scroll': params}
	if op == 'WAIT':
		return {'wait': {'seconds': WAIT_SECONDS}}
	if op == 'GO_BACK':
		return {'go_back': {}}
	if op == 'PRESS_ENTER':
		return {'send_keys': {'keys': 'Enter'}}
	if op == 'COLLECT':
		return {'collect_page': {}}
	raise ValueError(f'{op} cannot be executed with this target or text')


def memory_line(
	run_id: str, op: Operation, candidate: Candidate | None, text: str | None, op_p: float | None, target_p: float | None
) -> str:
	def probability(value: float | None) -> str:
		return 'n/a' if value is None else f'{value:.2f}'

	parts = ['System1', f'{run_id}:', op]
	if candidate is not None:
		parts.append(candidate.element())
	if text is not None:
		parts.append(f'text={json.dumps(text, ensure_ascii=False)}')
	probs = f'operation p={probability(op_p)}'
	if candidate is not None:
		probs += f', target p={probability(target_p)}'
	return (' '.join(parts) + f' ({probs})')[:300]
