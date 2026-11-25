"""Explicit offline-evidence reevaluation: python -m browser_use.benchmark.reevaluate ATTEMPT_FOLDER."""

import argparse
import json
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from .archive import atomic_json
from .evaluation import build_context, evaluate_frozen, freeze_context, validate_frozen
from .models import EvaluationModel, load_environment, redact_secrets, text_config


def load_context(folder):
	"""Use the exact frozen input, or reconstruct only from the same legacy archive."""
	frozen_path = folder / 'evaluation-context.json'
	if frozen_path.exists():
		frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
		validate_frozen(frozen)
		return frozen
	state = json.loads((folder / 'trace.json').read_text(encoding='utf-8'))
	evaluation = json.loads((folder / 'evaluation.json').read_text(encoding='utf-8'))
	# Task definitions may have changed. Only this archive may supply a missing original goal.
	manifest_path = folder.parent.parent / 'manifest.json'
	manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
	if not state.get('goal'):
		task = next((task for task in manifest.get('tasks', []) if task.get('id') == folder.parent.name), {})
		state['goal'] = task.get('goal')
	if not isinstance(state.get('goal'), str) or not state['goal'].strip():
		raise ValueError('Archive has no original goal; cannot reconstruct evaluation context')
	samples = evaluation.get('observations', [])
	if evaluation.get('final') and evaluation['final'] not in samples:
		samples = [*samples, evaluation['final']]
	events = []
	events_path = folder / 'events.jsonl'
	if events_path.exists():
		for line in events_path.read_text(encoding='utf-8').splitlines():
			try:
				event = json.loads(line)
			except ValueError:
				continue
			if event.get('event') in {'action_started', 'action_executed', 'action_result', 'action_rejected'}:
				events.append(event)
	context = build_context(
		state,
		samples,
		task_started_at=state.get('task_started_at') or manifest.get('created_at'),
		task_ended_at=evaluation.get('final', {}).get('time'),
		execution_status=state.get('execution_status', state.get('status', 'unknown')),
		execution_reason=state.get('error', ''),
		execution_events=events,
	)
	return freeze_context(context, reconstructed=True)


def reevaluate(folder, config: EvaluationModel, *, frozen=None):
	"""Append a separate audit record. Original trace/result/evaluation/summary files are untouched."""
	folder = Path(folder).resolve()
	frozen = load_context(folder) if frozen is None else deepcopy(frozen)
	validate_frozen(frozen)
	# An unfinished attempt can still be writing evidence; require an explicit completed archive.
	summary_path = folder.parent.parent / 'summary.json'
	if summary_path.exists():
		summary = json.loads(summary_path.read_text(encoding='utf-8'))
		if summary.get('status') not in {'completed', 'cancelled', 'error', 'interrupted'}:
			raise ValueError('Cannot reevaluate an unfinished batch')
	output = folder / 'reevaluations' / datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
	output.mkdir(parents=True)
	atomic_json(output / 'evaluation-context.json', frozen)
	calls = []

	@contextmanager
	def model_call(kind, **metadata):
		record = {'id': len(calls) + 1, 'kind': kind, 'status': 'pending', **metadata}
		calls.append(record)
		started = time.perf_counter()
		try:
			yield record
			record['status'] = 'success'
		except Exception as error:
			record.update(status='error', error=str(error))
			raise
		finally:
			record['latency_ms'] = round((time.perf_counter() - started) * 1000)
			atomic_json(output / 'model-calls.json', redact_secrets(calls, config.api_key))

	result = evaluate_frozen(frozen, config, model_call)
	result.update(reevaluated_at=result['ended_at'], original_result_unchanged=True)
	atomic_json(output / 'evaluation.json', redact_secrets(result, config.api_key))
	return {
		'path': str(output),
		'context_id': frozen['context_id'],
		'evaluation_status': result['evaluation_status'],
		'verdict': result.get('verdict'),
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('attempt_folder', type=Path)
	parser.add_argument('--prepare-only', action='store_true', help='Write reconstructed context without a model call')
	parser.add_argument('--output', type=Path, help='Destination context for --prepare-only')
	args = parser.parse_args()
	if args.prepare_only:
		if args.output is None:
			parser.error('--prepare-only requires --output')
		frozen = load_context(args.attempt_folder)
		args.output.parent.mkdir(parents=True, exist_ok=True)
		atomic_json(args.output, frozen)
		print(json.dumps({'path': str(args.output), 'context_id': frozen['context_id']}))
	else:
		load_environment()
		print(json.dumps(reevaluate(args.attempt_folder, text_config()), ensure_ascii=False))


if __name__ == '__main__':
	main()
