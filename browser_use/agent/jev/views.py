"""Public Jev configuration and shared decision, observation, and execution records."""

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from browser_use.llm.base import BaseChatModel

PAGE_TEXT_LIMIT = 6_000
VISIBLE_TEXT_LIMIT = 50_000
MAX_REQUEST_CHARS = 120_000
ELEMENT_TEXT_LIMIT = 300
SEGMENT_ACTIONS_LIMIT = 30
TASK_ACTIONS_LIMIT = 10
REPEAT_WINDOW = 8
REPEAT_LIMIT = 3
NO_EFFECT_LIMIT = 2
NO_ACTION_HANDBACK_LIMIT = 2
SCROLL_PAGES = 0.8
WAIT_SECONDS = 2
JEV_RETRY_DELAYS = (0.5, 1.0)
JEV_INPUT_USD_PER_MTOK = 0.042
MAX_CHOICE_OPTIONS = 255
TEXT_VALUE_LIMIT = 2_000
SUMMARY_LIMIT = 400
ANSWER_LIMIT = 12_000
LINKS_LIMIT_CHARS = 6_000
RISK_PAGE_TEXT_LIMIT = 1_500
SUBGOAL_FIELD_LIMITS = {
	'objective': 500,
	'completion': 500,
	'reason': 1_000,
	'inputs': 1_000,
	'review_note': 300,
	'screenshot_reason': 300,
}
DEFAULT_SCREENSHOT_REASON = 'Planner requested the image without a specific reason.'


class JevModel(BaseModel):
	model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True, hide_input_in_errors=True)


CheckMode = Literal['enforce', 'record', 'off']
CheckName = Literal['risky_action', 'submit', 'completion', 'confidence', 'scope']


class JevChecks(JevModel):
	default: CheckMode = 'enforce'
	risky_action: CheckMode | None = None
	submit: CheckMode | None = None
	completion: CheckMode | None = None
	confidence: CheckMode | None = None
	scope: CheckMode | None = None

	def mode(self, name: CheckName) -> CheckMode:
		value = getattr(self, name)
		return self.default if value is None else value


def _parse_checks(value: Any) -> Any:
	return {'default': value} if isinstance(value, str) else value


CandidatePipeline = Literal['pr5', 'extended']
EffectiveCandidatePipeline = Literal['pr5', 'extended', 'pr5_with_regions']


def resolve_candidate_pipeline(requested: CandidatePipeline | None, screenshot_policy: str) -> EffectiveCandidatePipeline:
	if requested is not None:
		return requested
	return 'pr5_with_regions' if screenshot_policy == 'off' else 'extended'


class JevSettings(JevModel):
	mode: Literal['isolated', 'collaborative', 'off'] = 'isolated'
	system1: Literal['jev', 'llm'] = 'jev'
	# Jev backend
	jev_base_url: str = 'https://api.typesafe.ai'
	jev_model: str = 'jev-latest'
	jev_api_key: str | None = Field(default=None, repr=False, exclude=True)
	jev_timeout: float = Field(default=10, gt=0, le=120)
	screenshot_policy: Literal['off', 'always', 'smart'] = 'off'
	candidate_pipeline: CandidatePipeline | None = None
	pruning_budget_ms: float = Field(default=300, ge=0, allow_inf_nan=False)
	vision_escalation: bool = True
	vision_max_side: int = Field(default=1600, ge=320, le=4096)
	vision_jpeg_quality: int = Field(default=72, ge=30, le=95)
	# LLM backend
	system1_llm: BaseChatModel | None = None
	system1_llm_two_stage: bool = False
	# helper LLMs; each role falls back to helper_llm, then to Agent.llm
	helper_llm: BaseChatModel | None = None
	text_llm: BaseChatModel | None = None
	text_prompt: str | None = None
	summary_llm: BaseChatModel | None = None
	summary_prompt: str | None = None
	answer_llm: BaseChatModel | None = None
	answer_prompt: str | None = None
	answer_timeout: float = Field(default=120, gt=0)
	# thresholds and limits
	min_read_operation_probability: float = Field(default=0.6, ge=0, le=1)
	min_operation_probability: float = Field(default=0.8, ge=0, le=1)
	min_target_probability: float = Field(default=0.8, ge=0, le=1)
	unable_probability_handback: float = Field(default=0.25, ge=0, le=1)
	default_max_actions: int = Field(default=15, ge=1, le=30)
	max_candidates: int = Field(default=200, ge=1, le=255)
	trace_path: str | Path | None = None
	capture_dir: str | Path | None = None
	# programmatic checks
	checks: Annotated[JevChecks, BeforeValidator(_parse_checks)] = Field(default_factory=JevChecks)
	risky_action_labels: list[str] | None = None

	@property
	def effective_candidate_pipeline(self) -> EffectiveCandidatePipeline:
		return resolve_candidate_pipeline(self.candidate_pipeline, self.screenshot_policy)

	@model_validator(mode='before')
	@classmethod
	def _disabled_settings(cls, data: Any) -> Any:
		if isinstance(data, dict) and data.get('mode') == 'off':
			return {'mode': 'off'}
		return data

	@field_validator('jev_base_url')
	@classmethod
	def _http_endpoint(cls, value: str) -> str:
		url = urlsplit(value)
		if url.scheme not in ('http', 'https') or not url.hostname or url.query or url.fragment or url.username is not None:
			raise ValueError('jev_base_url must be an HTTP(S) base URL without credentials, query, or fragment')
		return value.rstrip('/')

	@model_validator(mode='after')
	def _configured_backend(self) -> Self:
		if self.screenshot_policy != 'off' and self.system1 != 'jev':
			raise ValueError('Screenshot policy requires the Jev backend')
		if self.system1 == 'llm' and self.system1_llm is None:
			raise ValueError('system1_llm is required for the LLM backend')
		if self.system1_llm_two_stage and self.system1 != 'llm':
			raise ValueError('system1_llm_two_stage requires the LLM backend')
		return self


class DelegateSubgoalParams(JevModel):
	objective: str = Field(
		description='What System1 should achieve in this stage only. State the result, not clicks, element indices, selectors or coordinates.'
	)
	completion: str = Field(description='A condition directly visible on the page when this stage is done.')
	reason: str = Field(
		description='Why this subgoal now. After a handback, first state your verification of the previous stage: what was achieved, the evidence, and what is missing. Use the language of the task.'
	)
	inputs: str | None = Field(
		default=None, description='Values System1 must enter, e.g. "From=Zurich; To=London". Omit if nothing needs typing.'
	)
	text_mode: Literal['replace', 'append'] = Field(
		default='replace',
		description='How to type into an existing field. Use append for text that must preserve and follow its current contents. Use a separate stage when another field needs replacement.',
	)
	may_submit: bool = Field(
		default=False,
		description='Allow submits (search, filter, next, pressing Enter) and irreversible or account-changing clicks (send, pay, delete) in this stage. Irreversible clicks are still confirmed with you one by one.',
	)
	requires_action: bool = Field(
		default=False,
		description='True if this stage must actually change the page (type, select, or click a control) before System1 may request completion. False for observation or navigation stages and when the page already shows the required state.',
	)
	max_actions: int | None = Field(default=None, description='Action budget for this stage, 1-30. Default 15.')
	start_url: str | None = Field(
		default=None, description='Open this http(s) URL in the current tab before System1 starts. May include query parameters.'
	)
	start_tab: str | None = Field(
		default=None,
		description='Switch to this open tab (4-character tab id) before System1 starts. Do not combine with start_url.',
	)
	review_note: str | None = Field(
		default=None,
		description='Required when you re-delegate the same objective and completion after rejecting STEP_DONE: what is still missing and why the stage is not done.',
	)


class DelegateSubgoalParamsSmart(DelegateSubgoalParams):
	screenshot_required: bool | None = Field(
		default=None,
		description=(
			'Whether System1 sees the current viewport image in this stage. REQUIRED (true or false) when you delegate after '
			'a handback other than an accepted STEP_DONE, including a re-delegation after you rejected STEP_DONE. '
			'True when System1 chose wrong or ambiguous targets, missed visible targets or evidence, or could not interpret '
			'the layout (rows/cards/icons/overlays/visual-only state). False when text is enough. '
			'May be omitted for the first stage and after an accepted STEP_DONE; omitted means text-only. '
			'Images never add targets or permit coordinates.'
		),
	)
	screenshot_reason: str | None = Field(
		default=None,
		description=(
			'Required whenever you set screenshot_required: the specific visual difficulty System1 should resolve with the '
			'image, or why text is enough.'
		),
	)


class FinishParams(JevModel):
	success: bool = Field(description='True only if every requirement of the task is supported by evidence.')
	reason: str = Field(description='Concrete observations supporting this decision. Use the language of the task.')


class ScrollRegionParams(JevModel):
	key: str
	down: bool


class ScrollRegion(JevModel):
	key: str
	backend_node_id: int
	target_id: str
	document: float
	scroll_top: int
	scroll_height: int
	client_height: int


class ScrollObservation(JevModel):
	target_id: str
	document: float
	root_y: float
	root_overflow_y: str | None = None
	center_id: int | None = None
	center_path: str | None = None
	center_y: float | None = None
	center_embedded: bool = False
	page_suppressed: bool = False
	# Selector index -> (target, frame, backend node, scrollTop).
	indexed: dict[int, tuple[str, str | None, int, float]] = Field(default_factory=dict)


class ScrollEffect(JevModel):
	target_key: str | None
	root_delta_y: float | None = None
	center_delta_y: float | None = None
	target_delta_y: float | None = None
	url_changed: bool | None = None
	measurable: bool = False

	@property
	def page_changed(self) -> bool | None:
		if self.url_changed or any((self.root_delta_y, self.center_delta_y, self.target_delta_y)):
			return True
		return False if self.measurable else None


class CollectPageParams(BaseModel):
	model_config = ConfigDict(extra='ignore')


HandbackReason = Literal[
	'step_done',
	'unable',
	'unable_probability',
	'low_confidence',
	'no_effect',
	'repeated_action',
	'wait_limit',
	'action_error',
	'step_error',
	'system1_error',
	'invalid_answer',
	'text_unavailable',
	'out_of_scope',
	'action_budget',
	'too_many_candidates',
	'state_too_large',
	'submit_not_allowed',
	'risk_not_allowed',
	'risk_denied',
	'unsupported_page',
	'step_limit',
	'verification_stalled',
	'observation_changed',
]

Operation = Literal[
	'CLICK', 'TYPE_TEXT', 'SELECT', 'SCROLL_UP', 'SCROLL_DOWN', 'WAIT', 'GO_BACK', 'PRESS_ENTER', 'COLLECT', 'STEP_DONE', 'UNABLE'
]
ActionStatus = Literal['predicted', 'dispatched', 'executed_ok', 'executed_failed', 'discarded', 'unknown']
EXECUTED_STATUSES = frozenset({'executed_ok', 'executed_failed', 'unknown'})  # D27


class ActionRecord(JevModel):
	n: int  # 1-based position within the run, counting every record including discarded ones
	step: int  # BU step number (agent.state.n_steps) of the step that produced it
	run_id: str
	operation: Operation
	target_key: str | None = None
	target_label: str | None = None  # the candidate's "element" string, at most 120 characters
	text: str | None = None  # TYPE_TEXT value
	bu_action: dict[str, dict[str, Any]]  # e.g. {'click': {'index': 1530}}
	fingerprint_before: str
	url_before: str
	status: ActionStatus = 'predicted'
	error: str | None = None
	verified: bool | None = None  # SELECT only: the chosen option was selected in the next observation (5.8)
	page_changed: bool | None = None
	scroll_effect: ScrollEffect | None = None
	url_after: str | None = None
	applied: bool = False  # counts as an actual change (D15), set when status becomes executed_ok

	@property
	def executed(self) -> bool:
		return self.status in EXECUTED_STATUSES


class Handback(JevModel):
	reason: HandbackReason  # 5.9
	detail: str
	step: int
	operation: Operation | None = None
	operation_probability: float | None = None
	target_label: str | None = None
	target_probability: float | None = None
	fingerprint: str | None = None  # observation at handback time, if one was made
	evidence_key: str | None = None


class SubgoalRun(JevModel):
	id: str  # 'sg1', 'sg2', ...
	params: DelegateSubgoalParams  # after truncation; max_actions resolved
	identity: str  # normalized objective + '\n' + normalized completion (D18)
	max_actions: int
	started_step: int
	start_url: str  # page URL after start_url / start_tab was applied
	start_site: str | None  # registrable_domain(start_url), None for non-http pages
	actions: list[ActionRecord] = Field(default_factory=list)
	applied: bool = False  # any applied action in this run
	no_effect_streak: int = 0
	new_evidence: bool = False
	decisions: int = 0
	vision_decisions: int = 0
	vision: bool = False
	vision_at_start: bool = Field(default_factory=lambda data: data['vision'])
	no_effect_escalated: bool = False
	escalations: list[dict[str, Any]] = Field(default_factory=list)
	vision_source: Literal['policy', 'planner', 'escalation'] | None = None
	vision_reason: str | None = None
	review_delegation: bool = False
	screenshot_declared: bool | None = None
	rejected_fingerprint: str | None = None  # set when this run re-delegates a rejected STEP_DONE
	status: Literal['active', 'handed_back'] = 'active'
	handback: Handback | None = None

	@property
	def executed_actions(self) -> list[ActionRecord]:
		return [a for a in self.actions if a.executed]


class Candidate(JevModel):
	key: str  # '1530' | '1412:2' (select option) | 'tab:1' | 'page'
	kind: Literal['element', 'option', 'scroll', 'tab']
	index: int | None = None  # selector_map key; for options, the <select> index
	role: str
	name: str
	tag: str | None = None
	input_type: str | None = None
	value: str | None = None
	states: dict[str, bool] = Field(default_factory=dict)
	placeholder: str | None = None
	href: str | None = None
	context: dict[str, str] = Field(default_factory=dict)
	editable: bool = False
	is_submit: bool = False
	risky_effect: str | None = None  # appendix D, filled from T14 on
	tab_id: str | None = None  # 4-character BU tab id, for kind='tab'
	option_text: str | None = None  # for kind='option'
	match_text: str | None = None  # for kind='option': the select_dropdown text that hits this option first (D30)
	form_key: str | None = None  # backend node id of the nearest form ancestor, as str (D31)
	dialog_key: str | None = None  # backend node id of the nearest dialog ancestor, as str (D31)
	rect: dict[str, int] | None = None  # clipped CSS viewport coordinates

	def element(self) -> str:
		if self.key == 'page' and self.kind == 'scroll':
			return '[page] the whole page'
		if self.kind == 'tab':
			label = f'[{self.key}] browser tab {json.dumps(self.name, ensure_ascii=False)} {self.href or ""}'
		else:
			label = f'[{self.key}] {self.role} {json.dumps(self.name, ensure_ascii=False)}'
			if self.tag == 'input' and self.input_type:
				label += f' type={self.input_type}'
			if self.placeholder:
				label += f' placeholder={json.dumps(self.placeholder, ensure_ascii=False)}'
			if self.href:
				label += f' href={self.href}'
		return label.rstrip()[:ELEMENT_TEXT_LIMIT]

	def criterion(self, include_rect: bool = False) -> dict[str, Any]:
		data: dict[str, Any] = {'element': self.element(), 'role': self.role}
		if self.value is not None:
			data['current_value'] = self.value
		if self.states:
			data['states'] = self.states
		if self.context:
			data['context'] = self.context
		if include_rect and self.rect is not None:
			data['rect'] = self.rect
		return data


class Observation(JevModel):
	step: int  # agent.state.n_steps when observed
	url: str
	title: str
	scroll: dict[str, int]  # y, viewport_height, page_height, pixels_above, pixels_below
	scroll_regions: list[ScrollRegion] = Field(default_factory=list)
	scroll_observation: ScrollObservation | None = Field(default=None, exclude=True)
	viewport_width: int = 0
	viewport_height: int = 0
	visible_text: str  # full visible text, appendix F, at most 50,000 characters
	links: list[dict[str, str]]  # visible http(s) links: {'text', 'url'}
	candidates: dict[str, list[Candidate]]  # keyed by CLICK, TYPE_TEXT, SELECT, SCROLL_UP, SCROLL_DOWN
	tabs: list[dict[str, str]]  # other open tabs: {'key', 'tab_id', 'title', 'url'}
	can_go_back: bool
	is_pdf: bool
	state_error: str | None
	unselectable_options: int = 0  # options left out because no unique select_dropdown text exists (D30)
	fingerprint: str  # 5.9


class FinalBrowserObservation(JevModel):
	url: str
	step: int
	target_id: str
	source_document: float
	dom_text: str


class System1Reply(JevModel):
	# Only the operation and the chosen operation's target influence execution.
	answers: dict[str, Any]
	model: str | None = None
	input_tokens: int | None = None
	output_tokens: int | None = None
	latency_ms: int
	attempts: list[dict[str, Any]]  # [{'attempt': 1, 'status': 429 | None, 'error_type': str | None, 'latency_ms': int}]
	has_probabilities: bool  # True for Jev, False for the LLM backend
	image_count: int = 0
	image_bytes: int = 0
