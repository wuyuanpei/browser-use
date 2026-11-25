"""Apply each configured policy and record its decision at the point of use."""

from browser_use.agent.jev.trace import JevTrace
from browser_use.agent.jev.views import CheckMode, CheckName, JevChecks


class CheckGate:
	def __init__(self, checks: JevChecks, trace: JevTrace):
		self.checks = checks
		self.trace = trace

	def mode(self, name: CheckName) -> CheckMode:
		return self.checks.mode(name)

	def is_off(self, name: CheckName) -> bool:
		return self.mode(name) == 'off'

	def evaluate(self, name: CheckName, triggered: bool, detail: str, step: int | None = None) -> bool:
		mode = self.mode(name)
		if not triggered or mode == 'off':
			return False
		block = mode == 'enforce'
		self.trace.write('check', step=step, name=name, mode=mode, outcome='blocked' if block else 'would_block', detail=detail)
		return block
