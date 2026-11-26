"""Role attribution counts physical requests and reports incomplete billing coverage."""

import json
from typing import cast
from unittest.mock import AsyncMock

from browser_use.agent.jev.trace import JevTrace
from browser_use.agent.jev.usage import UsageLedger
from browser_use.agent.jev.views import System1Reply
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from browser_use.tokens.service import TokenCost
from tests.ci.conftest import create_mock_llm


def test_escalations_are_independent_of_request_counts(tmp_path):
	ledger = UsageLedger(TokenCost(include_cost=False), JevTrace(tmp_path / 'trace.jsonl', 'test'))
	for reason in ('low_confidence', 'no_effect', 'low_confidence'):
		ledger.record_escalation(reason)
	ledger.escalations_executed += 1
	ledger.record_jev(None, 1, vision=True)
	ledger.record_jev(None, 1)
	row = ledger.summary()['system1']
	assert (row['escalations'], row['escalations_executed'], row['vision_requests'], row['text_requests']) == (3, 1, 1, 1)
	assert row['escalations_by_reason'] == {'low_confidence': 2, 'no_effect': 1}


async def test_shared_model_retry_and_missing_usage(tmp_path):
	trace = JevTrace(tmp_path / 'usage.jsonl', 'test')
	cost = TokenCost(include_cost=False)
	ledger = UsageLedger(cost, trace)
	llm = create_mock_llm()
	usage = ChatInvokeUsage(
		prompt_tokens=7,
		prompt_cached_tokens=2,
		prompt_cache_creation_tokens=None,
		prompt_image_tokens=None,
		completion_tokens=3,
		total_tokens=10,
	)
	cast(AsyncMock, llm.ainvoke).side_effect = [
		ChatInvokeCompletion(completion='empty', usage=usage),
		ChatInvokeCompletion(completion='valid', usage=None),
		ChatInvokeCompletion(completion='helper', usage=usage),
		ChatInvokeCompletion(completion='compaction', usage=usage),
	]
	cost.register_llm(llm)
	ledger.roles['planner'].decisions += 1
	await ledger.invoke(3, 'planner', llm, [])
	await ledger.invoke(3, 'planner', llm, [], attempt=2)
	await ledger.invoke(3, 'answer', llm, [])
	await llm.ainvoke([])
	await ledger.sync_other(3)
	summary = ledger.summary()
	assert summary['planner']['decisions'] == 1 and summary['planner']['calls'] == 2
	assert summary['planner']['missing_usage_calls'] == 1 and summary['planner']['reported_usage_calls'] == 1
	assert summary['planner']['prompt_tokens'] == 7
	assert summary['planner']['cost_usd'] is None
	assert summary['answer']['calls'] == summary['other']['calls'] == 1
	assert summary['answer']['prompt_tokens'] == summary['other']['prompt_tokens'] == 7
	assert summary['other']['call_coverage'] == 'reported_usage_only'
	assert summary['other']['latency_coverage'] == 'unavailable'
	records = [json.loads(x) for x in trace.path.read_text(encoding='utf-8').splitlines()]
	assert [r['role'] for r in records] == ['planner', 'planner', 'answer', 'other']
	assert records[1]['prompt_tokens'] is None


def test_jev_transport_retries_have_independent_coverage(tmp_path):
	ledger = UsageLedger(TokenCost(include_cost=False), JevTrace(tmp_path / 'trace.jsonl', 'test'))
	reply = System1Reply(
		answers={},
		model='test',
		input_tokens=10,
		output_tokens=4,
		latency_ms=350,
		attempts=[{'status': 503}, {'status': 200}],
		has_probabilities=True,
	)
	ledger.record_jev(reply, 350)
	row = ledger.summary()['system1']
	assert row['calls'] == 2 and row['reported_usage_calls'] == row['missing_usage_calls'] == 1
	assert row['latency_ms'] == 350 and row['prompt_tokens'] == 10 and row['completion_tokens'] == 4
	assert row['cost_usd'] is None
