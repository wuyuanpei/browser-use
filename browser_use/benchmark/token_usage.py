"""Production token accounting from actual responses, excluding independent verification."""

TOKEN_FIELDS = (
	'jev_input_tokens',
	'jev_output_tokens',
	'planner_input_tokens',
	'planner_output_tokens',
	'planner_total_tokens',
	'other_input_tokens',
	'other_cached_input_tokens',
	'other_output_tokens',
	'other_total_tokens',
	'jev_total_tokens',
	'llm_total_tokens',
)


def token_usage(calls):
	result: dict[str, int] = dict.fromkeys(TOKEN_FIELDS, 0)
	missing = 0
	for call in calls:
		kind = call.get('kind')
		group = (
			'jev'
			if kind == 'jev'
			else 'planner'
			if kind == 'planner'
			else 'other'
			if kind in {'text', 'summarize', 'collect', 'summary', 'answer', 'risk', 'other', 'deepseek', 'browser_use'}
			else None
		)
		if group is None:
			continue
		response = call.get('response')
		usage = response.get('usage') if isinstance(response, dict) else None
		if not isinstance(usage, dict):
			usage = {}
		if group == 'other':
			for key in ('prompt_cached_tokens', 'prompt_cache_hit_tokens'):
				cached = usage.get(key)
				if type(cached) is int and cached >= 0:
					result['other_cached_input_tokens'] += cached
					break
		incomplete = False
		if group == 'jev' and call.get('role') == 'system1':
			# One inspected decision can include multiple billable HTTP attempts.
			missing += max(0, len(call.get('attempts') or []) - 1)
		for direction, aliases in (
			('input', ('input_tokens', 'prompt_tokens')),
			('output', ('output_tokens', 'completion_tokens')),
		):
			if group == 'jev' and direction == 'output' and 'output_tokens' not in usage and call.get('role') != 'system1':
				continue
			value = next((usage[k] for k in aliases if type(usage.get(k)) is int and usage[k] >= 0), None)
			if value is None:
				incomplete = True
			else:
				result[f'{group}_{direction}_tokens'] += value
		missing += incomplete
	result['jev_total_tokens'] = result['jev_input_tokens'] + result['jev_output_tokens']
	for group in ('planner', 'other'):
		result[f'{group}_total_tokens'] = result[f'{group}_input_tokens'] + result[f'{group}_output_tokens']
	result['llm_total_tokens'] = result['planner_total_tokens'] + result['other_total_tokens']
	return {**result, 'missing_usage_calls': missing, 'complete': missing == 0}


def aggregate_token_usage(attempts):
	result = dict.fromkeys((*TOKEN_FIELDS, 'missing_usage_calls', 'unavailable_attempts'), 0)
	for row in attempts:
		usage = row.get('token_usage')
		if usage is None:
			result['unavailable_attempts'] += 1
			continue
		for field in (*TOKEN_FIELDS, 'missing_usage_calls'):
			result[field] += usage.get(field, 0) if field in ('other_cached_input_tokens', 'jev_output_tokens') else usage[field]
	return {**result, 'complete': not (result['missing_usage_calls'] or result['unavailable_attempts'])}
