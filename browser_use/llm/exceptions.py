from typing import Any

from browser_use.llm.views import ChatInvokeUsage


class ModelError(Exception):
	pass


class ModelProviderError(ModelError):
	"""Exception raised when a model provider returns an error."""

	def __init__(
		self,
		message: str,
		status_code: int = 502,
		model: str | None = None,
	):
		super().__init__(message)
		self.message = message
		self.status_code = status_code
		self.model = model


class ModelOutputFormatError(ModelProviderError):
	"""The provider responded, but its structured tool arguments could not be decoded or validated."""

	def __init__(
		self,
		message: str,
		status_code: int = 502,
		model: str | None = None,
		*,
		raw_text: str = '',
		finish_reason: str | None = None,
		usage: ChatInvokeUsage | None = None,
	):
		super().__init__(message, status_code, model)
		self.raw_text = raw_text
		self.finish_reason = finish_reason
		self.parse_error = message
		self.usage = usage
		# The workbench holds this record while the planner decides whether to send corrective feedback.
		self._trace_record: dict[str, Any] = {
			'raw_text': raw_text[:20_000],
			'raw_chars': len(raw_text),
			'raw_truncated': len(raw_text) > 20_000,
			'finish_reason': finish_reason,
			'parse_error': message,
			'usage': usage.model_dump(mode='json') if usage is not None else None,
			'planner_feedback_retry': False,
		}

	def trace_record(self, *, role: str, model: str, planner_feedback_retry: bool = False) -> dict[str, Any]:
		self._trace_record.update(role=role, model=model, planner_feedback_retry=planner_feedback_retry)
		return self._trace_record


class ModelRateLimitError(ModelProviderError):
	"""Exception raised when a model provider returns a rate limit error."""

	def __init__(
		self,
		message: str,
		status_code: int = 429,
		model: str | None = None,
	):
		super().__init__(message, status_code, model)


class ModelOutputTruncatedError(ModelProviderError):
	"""Output was cut off at an output-token limit (finish_reason='length' / stop_reason='max_tokens').

	Status 400 keeps it out of same-provider retry loops; the agent's fallback switch treats it as recoverable.
	"""

	def __init__(
		self,
		message: str,
		model: str | None = None,
	):
		super().__init__(message, status_code=400, model=model)
