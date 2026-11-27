"""Load, validate and freeze the execution and evaluation profiles for a run."""

import math
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from browser_use.agent.jev.views import (
	CandidatePipeline,
	EffectiveCandidatePipeline,
	ignore_legacy_vision_escalation,
	resolve_candidate_pipeline,
)

PROVIDER = 'browser_use'
_LEGACY_SETTING_UNSET = object()
ExecutionMode = Literal['direct', 'isolated', 'collaborative']
DEBUG_PROFILE = 'debug'
FINAL_REPORT_PROFILE = 'final_report'
DEEPSEEK_URL = 'https://api.deepseek.com/v1'
REPORT_MODEL = {'model': 'deepseek-flash', 'base_url': DEEPSEEK_URL}
_SECRET_NAMES = (
	'PLANNER_API_KEY',
	'TEXT_MODEL_API_KEY',
	'SYSTEM1_DEEPSEEK_API_KEY',
	'REPORT_TEXT_MODEL_API_KEY',
	'LAYA_API_KEY',
	'TYPESAFE_API_KEY',
)


class ModelEndpoint(BaseModel):
	model_config = ConfigDict(frozen=True)

	model: str
	base_url: str
	api_key: str = Field(repr=False, exclude=True)

	def public(self) -> dict[str, str]:
		address = urlsplit(self.base_url)
		host = address.hostname or ''
		if ':' in host:
			host = f'[{host}]'
		if address.port:
			host += f':{address.port}'
		return {'model': self.model, 'base_url': urlunsplit((address.scheme, host, address.path, '', ''))}


class ExecutionModel(ModelEndpoint):
	timeout: float
	max_tokens: int


class EvaluationModel(ModelEndpoint):
	reasoning: dict[str, Any]


class RunSettings(BaseModel):
	model_config = ConfigDict(frozen=True)

	profile: Literal['debug', 'final_report']
	execution_mode: ExecutionMode = 'direct'
	screenshot_policy: Literal['off', 'always', 'smart'] = 'off'
	candidate_pipeline: CandidatePipeline | None = None
	min_read_operation_probability: float = Field(default=0.6, ge=0, le=1)
	jev: ModelEndpoint | None = None
	execution: ExecutionModel
	evaluation: EvaluationModel
	headless: bool
	user_data_dir: str | None

	@model_validator(mode='before')
	@classmethod
	def _ignore_legacy_settings(cls, data: Any) -> Any:
		return ignore_legacy_vision_escalation(data)

	@computed_field
	@property
	def effective_candidate_pipeline(self) -> EffectiveCandidatePipeline | None:
		if self.jev is None:
			return None
		return resolve_candidate_pipeline(self.candidate_pipeline, self.screenshot_policy)

	def system1_config(self) -> dict[str, str]:
		if self.jev is not None:
			return {'provider': 'jev', **self.jev.public()}
		return {'provider': PROVIDER, **self.execution.public()}


def load_environment():
	"""Load optional local settings without overriding the process environment."""
	for key, value in dotenv_values(Path.cwd() / '.env', encoding='utf-8-sig', interpolate=False).items():
		if value is not None:
			os.environ.setdefault(key, value)


def load_profile_environment(path: Path):
	"""Import the model fields accepted by the workbench from a profile file."""
	if not path.is_file():
		raise ValueError('Benchmark model profile configuration file does not exist')
	prefixes = ('PLANNER_', 'TEXT_MODEL_', 'REPORT_TEXT_MODEL_', 'SYSTEM1_DEEPSEEK_')
	for key, value in dotenv_values(path, encoding='utf-8-sig', interpolate=False).items():
		if value is not None and (key in {'TEXT_MODEL', 'REPORT_TEXT_MODEL', 'TYPESAFE_API_KEY'} or key.startswith(prefixes)):
			os.environ[key] = value


def _debug_endpoint() -> ModelEndpoint:
	# Keep the original profile variable names for existing local installations.
	base = (os.getenv('PLANNER_BASE_URL') or os.getenv('SYSTEM1_DEEPSEEK_BASE_URL') or DEEPSEEK_URL).rstrip('/')
	model = os.getenv('PLANNER_MODEL') or os.getenv('SYSTEM1_DEEPSEEK_MODEL') or 'deepseek-flash'
	key = os.getenv('PLANNER_API_KEY', '').strip()
	if not key and urlsplit(base).hostname == 'api.deepseek.com':
		for prefix in ('SYSTEM1_DEEPSEEK', 'TEXT_MODEL'):
			if urlsplit(os.getenv(f'{prefix}_BASE_URL') or DEEPSEEK_URL).hostname == 'api.deepseek.com':
				key = os.getenv(f'{prefix}_API_KEY', '').strip()
				if key:
					break
	return ModelEndpoint(model=model, base_url=base, api_key=key)


def text_config() -> EvaluationModel:
	base = os.getenv('TEXT_MODEL_BASE_URL', DEEPSEEK_URL).rstrip('/')
	reasoning: dict[str, Any] = (
		{'thinking': {'type': 'disabled'}} if 'api.deepseek.com/' in base else {'reasoning': {'effort': 'low'}}
	)
	if os.getenv('TEXT_MODEL_REASONING') == 'none':
		reasoning = {'reasoning': {'enabled': False}}
	return EvaluationModel(
		model=os.getenv('TEXT_MODEL', 'deepseek-chat'),
		base_url=base,
		api_key=os.getenv('TEXT_MODEL_API_KEY', ''),
		reasoning=reasoning,
	)


def _report_evaluation() -> EvaluationModel:
	if (os.getenv('REPORT_TEXT_MODEL_BASE_URL') or DEEPSEEK_URL).rstrip('/') != DEEPSEEK_URL:
		raise ValueError(f'REPORT_TEXT_MODEL_BASE_URL must be {DEEPSEEK_URL}')
	if (os.getenv('REPORT_TEXT_MODEL') or REPORT_MODEL['model']) != REPORT_MODEL['model']:
		raise ValueError(f'REPORT_TEXT_MODEL must be {REPORT_MODEL["model"]}')
	if (os.getenv('REPORT_TEXT_MODEL_REASONING') or 'disabled') != 'disabled':
		raise ValueError('REPORT_TEXT_MODEL_REASONING must be disabled')
	key = os.getenv('REPORT_TEXT_MODEL_API_KEY', '').strip()
	if not key:
		raise ValueError('Final report needs REPORT_TEXT_MODEL_API_KEY')
	return EvaluationModel(**REPORT_MODEL, api_key=key, reasoning={'thinking': {'type': 'disabled'}})


def model_profiles() -> list[dict[str, Any]]:
	debug_execution, debug_text = _debug_endpoint(), text_config()
	try:
		_report_evaluation()
		report_ready = True
	except ValueError:
		report_ready = False
	return [
		{
			'id': DEBUG_PROFILE,
			'label': '调试',
			'configured': bool(debug_execution.api_key and debug_text.api_key),
			# The inspector already reads the execution endpoint from this field.
			'planner': debug_execution.public(),
			'text': debug_text.public(),
		},
		{
			'id': FINAL_REPORT_PROFILE,
			'label': '最终报告',
			'configured': report_ready,
			'planner': REPORT_MODEL,
			'text': REPORT_MODEL,
		},
	]


def runtime_settings(
	provider=None,
	mode=None,
	screenshot_mode=False,
	*,
	screenshot_policy=None,
	vision_escalation=_LEGACY_SETTING_UNSET,
	candidate_pipeline=None,
	model_profile=None,
) -> RunSettings:
	if vision_escalation is not _LEGACY_SETTING_UNSET:
		ignore_legacy_vision_escalation({'vision_escalation': vision_escalation})
	mode = 'direct' if mode is None else mode
	if mode not in ('direct', 'isolated', 'collaborative'):
		raise ValueError('execution_mode must be direct, isolated, or collaborative')
	if provider not in (None, '', PROVIDER) and not (provider == 'jev' and mode != 'direct'):
		raise ValueError('测试台使用 Browser Use 执行。')
	policy = screenshot_policy if screenshot_policy is not None else 'always' if screenshot_mode else 'off'
	if policy not in ('off', 'always', 'smart'):
		raise ValueError('screenshot_policy must be off, always, or smart')
	if mode == 'direct' and policy != 'off':
		raise ValueError('BU 直接执行模式只支持关闭模型截图。')
	jev = None
	if mode != 'direct':
		from browser_use.agent.jev import JevSettings

		key = os.getenv('TYPESAFE_API_KEY', '').strip()
		if not key:
			raise ValueError('Jev 执行需要服务器配置 TYPESAFE_API_KEY。')
		defaults = JevSettings()
		jev = ModelEndpoint(model=defaults.jev_model, base_url=defaults.jev_base_url, api_key=key)
	profile = DEBUG_PROFILE if model_profile is None else model_profile
	if profile not in (DEBUG_PROFILE, FINAL_REPORT_PROFILE):
		raise ValueError('model_profile must be debug or final_report')
	if profile == FINAL_REPORT_PROFILE:
		evaluation = _report_evaluation()
		endpoint = evaluation
	else:
		endpoint, evaluation = _debug_endpoint(), text_config()
		if not endpoint.api_key:
			raise ValueError('调试执行需要 PLANNER_API_KEY 或同一官方 DeepSeek 服务的凭据。')
		if not evaluation.api_key:
			raise ValueError('调试终验需要 TEXT_MODEL_API_KEY。')

	timeout = float(os.getenv('PLANNER_TIMEOUT_SECONDS') or '60')
	if not math.isfinite(timeout) or timeout <= 0:
		raise ValueError('PLANNER_TIMEOUT_SECONDS must be positive and finite')
	try:
		max_tokens = int(os.getenv('PLANNER_MAX_TOKENS') or '4096')
		if max_tokens <= 0:
			raise ValueError()
	except ValueError:
		raise ValueError('PLANNER_MAX_TOKENS must be a positive integer') from None

	return RunSettings(
		profile=profile,
		execution_mode=mode,
		screenshot_policy=policy,
		candidate_pipeline=candidate_pipeline,
		jev=jev,
		execution=ExecutionModel(
			model=endpoint.model, base_url=endpoint.base_url, api_key=endpoint.api_key, timeout=timeout, max_tokens=max_tokens
		),
		evaluation=evaluation,
		headless=os.getenv('BROWSER_USE_BENCHMARK_HEADLESS', 'true').lower() not in {'0', 'false'},
		user_data_dir=os.getenv('BROWSER_USE_BENCHMARK_USER_DATA_DIR'),
	)


def redact_secrets(data: Any, *keys: str | None) -> Any:
	"""Remove current credentials from public responses, records and diagnostics."""
	secrets = {key for key in (*keys, *(os.getenv(name) for name in _SECRET_NAMES)) if key}
	if not secrets:
		return data

	def clean(value: Any) -> Any:
		if isinstance(value, str):
			for secret in secrets:
				value = value.replace(secret, '[REDACTED]')
		elif isinstance(value, dict):
			return {clean(k): clean(v) for k, v in value.items()}
		elif isinstance(value, list):
			return [clean(item) for item in value]
		return value

	return clean(data)
