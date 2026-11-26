"""Versioned baseline tasks and read-only fact sampling, never agent guidance."""

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from .observation import document_key, fingerprint

VERSION = '2026-11-26.1'
XIAOYI_TASKS = json.loads(Path(__file__).with_name('xiaoyi_selected_20.json').read_text(encoding='utf-8'))
XIAOYI_TASK_IDS = frozenset(task['id'] for task in XIAOYI_TASKS)
ARTICLE_GOAL = 'Open the article about using finite choices to control browser agents.'
TASKS = [
	{
		'id': '01_article_open',
		'name': '从列表打开文章',
		'url': '/fixture.html?scenario=research',
		'goal': ARTICLE_GOAL,
		'capability': '语义匹配、终止判断',
	},
	{
		'id': '02_article_already_open',
		'name': '已完成时直接停止',
		'url': '/fixture.html?scenario=research',
		'goal': ARTICLE_GOAL,
		'capability': '初始完成、零动作',
		'setup': 'choices',
	},
	{
		'id': '03_article_recover',
		'name': '从错误文章恢复',
		'url': '/fixture.html?scenario=research',
		'goal': ARTICLE_GOAL,
		'capability': '错误状态识别、恢复导航',
		'setup': 'latency',
	},
	{
		'id': '04_wikipedia',
		'name': '查找英文维基词条',
		'url': 'https://www.wikipedia.org/',
		'goal': 'Find and open the English Wikipedia article about Alan Turing.',
		'capability': '文本生成、搜索提交、详情识别',
	},
	{
		'id': '05_baidu',
		'name': '百度搜索后停止',
		'url': 'https://www.baidu.com/',
		'goal': '在百度搜索“华为官网”，停留在搜索结果页，不登录、不打开任何结果。',
		'capability': '中文搜索、终点约束',
	},
	{
		'id': '06_pagination',
		'name': '滚动并进入第二页',
		'url': 'https://books.toscrape.com/',
		'goal': 'Go to page 2 of the book catalogue and leave that catalogue page open.',
		'capability': '滚动、新候选发现、翻页',
	},
	{
		'id': '07_checkboxes',
		'name': '修改并保持勾选状态',
		'url': 'https://the-internet.herokuapp.com/checkboxes',
		'goal': 'Check the first checkbox. Leave the second checkbox checked without changing it.',
		'capability': '布尔状态、避免重复操作',
	},
	{
		'id': '08_travel_filters',
		'name': '筛选住宿并打开详情',
		'url': '/fixture.html',
		'goal': ('Search for stays in Lisbon, set the category to Design, enable Free cancellation, then open Casa Flora.'),
		'capability': '提交、下拉选择、多约束保持',
	},
	{
		'id': '09_dynamic_loading',
		'name': '等待异步结果',
		'url': 'https://the-internet.herokuapp.com/dynamic_loading/2',
		'goal': 'Start loading and stop when “Hello World!” is visible.',
		'capability': '等待、加载完成判断',
	},
	{
		'id': '10_ctrip',
		'name': '携程多条件机票查询',
		'url': 'https://flights.ctrip.com/online/channel/domestic',
		'goal': (
			'在携程查询 2026 年 12 月 20 日从北京到上海的单程经济舱直飞航班。'
			'看到符合条件的航班列表后停止。已登录或未登录均可，不主动登录或退出，不选购或预订航班。'
		),
		'capability': '自动补全、日期、多条件、禁止预订',
		'date': '2026-12-20',
	},
	{
		'id': '11_xiaohongshu_outfits',
		'name': '收集小红书穿搭内容',
		'url': 'https://www.xiaohongshu.com/',
		'goal': '帮我收集小红书的穿搭博主发布的穿搭内容及对应链接，至少3条内容，详细一些',
		'capability': '跨页面资料收集、来源引用、详细报告',
	},
	{
		'id': '12_law_advanced_form',
		'name': '法律法规库高级搜索（原 case 10）',
		'url': 'https://flk.npc.gov.cn/',
		'goal': (
			'从头操控：帮我打开国家数据法律法规数据库网站首页进行高级搜索，填写搜索表单。'
			'公布日期写2026年1月1号到2026年2月1号。法律法规分类选地方法规-法规性决定。'
			'制定机关选地方人大及其常委会-甘肃。只填写，不要确定/提交。没提到的字段不要填。'
		),
		'capability': '日期区间、层级分类、未指定字段保持空白、禁止提交',
		'source_case_id': '10',
	},
	{
		'id': '13_tencent_questionnaire',
		'name': '腾讯问卷前十题（原 case 7）',
		'url': 'https://wj.qq.com/s2/26481632/fc56/',
		'goal': '从头操控：帮我填写问卷https://wj.qq.com/s2/26481632/fc56/。只填1-10题，尽量回答正确，填写就好，不要提交。',
		'capability': '题目识别、连续填写、前十题范围、禁止提交',
		'source_case_id': '7',
	},
	{
		'id': '14_baidu_scholar_form',
		'name': '百度学术高级搜索（原 case 6）',
		'url': 'https://xueshu.baidu.com/',
		'goal': (
			'从头操控：帮我打开百度学术搜索页进行高级搜索填写表单，包含全部检索词填浏览器操控，'
			'包含精确搜索词填操控，至少包含一个词填浏览器，不包含检索词填AI，作者填小王，'
			'机构填青海大学，出版物选会议，填AAAI，发表时间填2010-2026年，语言检索范围选中文。'
			'帮我填就好，不要提交搜索。'
		),
		'capability': '多字段填写、出版物选择、年份区间、禁止提交',
		'source_case_id': '6',
	},
]
TASKS.extend(XIAOYI_TASKS)
TASKS.extend(
	[
		{
			'id': '112_visual_cards',
			'name': '按插图选择房源',
			'url': '/fixture.html?scenario=visual-cards',
			'goal': 'Open the stay whose illustration is green.',
			'capability': '按颜色区分同名卡片',
		},
		{
			'id': '113_visual_icons',
			'name': '按图标标记条目',
			'url': '/fixture.html?scenario=visual-icons',
			'goal': 'Mark this item with the star button.',
			'capability': '按形状区分无文字按钮',
		},
		{
			'id': '114_visual_cards_plain',
			'name': '按房源名打开同名卡片详情',
			'url': '/fixture.html?scenario=visual-cards-plain',
			'goal': 'Open the Sage House stay.',
			'capability': '从文本选择升级到视觉识别',
		},
		{
			'id': '115_scroll_panel',
			'name': '滚动侧栏后勾选筛选项',
			'url': '/fixture.html?scenario=scroll-panel',
			'goal': 'Check the Rooftop terrace option in Filters and leave it checked.',
			'capability': '区分独立滚动区、按标签修改筛选状态',
		},
		{
			'id': '116_dense',
			'name': '从密集控件打开档案总账',
			'url': '/fixture.html?scenario=dense',
			'goal': 'Open the archive ledger and leave it open.',
			'capability': '裁剪和遮挡过滤、保留低词重合按钮',
		},
	]
)
PROBE = Path(__file__).with_name('baseline_probe.js').read_text(encoding='utf-8')
INSPECT_PAGE = (
	'(() => { const page=window.__benchmarkObserver?.readState?.(); const facts=' + PROBE + '; return {page,facts}; })()'
)


def task_catalog(origin):
	"""Resolve only the built-in fixture URLs; callers cannot supply executable setup."""
	return [{**deepcopy(task), 'version': VERSION, 'url': urljoin(origin, task['url'])} for task in TASKS]


class EnvironmentBlocked(RuntimeError):
	"""A built-in fixture could not prepare the specified initial state."""


class CaseProbe:
	"""Prepare built-in fixtures and collect source-qualified facts for one attempt."""

	def __init__(self, task: dict[str, Any]):
		self._task = task
		self._form_descriptions: dict[str, Any] = {}

	def prepare(self, evaluate: Callable[[str], Any]) -> None:
		article = self._task.get('setup')
		if article:
			if article not in {'choices', 'latency'}:
				raise EnvironmentBlocked('Unknown baseline article setup')
			expected = urlsplit(self._task['url'])
			current = urlsplit(evaluate('location.href') or '')
			if (current.netloc, current.path) != (expected.netloc, expected.path):
				raise EnvironmentBlocked('Local fixture did not load for baseline setup')
			result = evaluate(
				'(() => { const a=document.querySelector(\'[data-article="' + article + '"]\');'
				" if(!a) return false; a.click(); return !!document.querySelector('.article-content'); })()"
			)
			if not result:
				raise EnvironmentBlocked('Article setup was not confirmed')
		if self._task['id'] == '07_checkboxes':
			checked = evaluate("[...document.querySelectorAll('#checkboxes input[type=checkbox]')].map(e=>e.checked)")
			if checked != [False, True] or any(type(value) is not bool for value in checked):
				raise EnvironmentBlocked('Checkbox initial state is not [false, true]')

	def inspect(self, evaluate: Callable[[str], Any], page: dict[str, Any]) -> dict[str, Any]:
		expected_document = document_key(page)
		target = page.get('target_id')
		try:
			# The reader is bound to the CDP target that supplied page.
			sample = evaluate(INSPECT_PAGE)
			if not isinstance(sample, dict) or not isinstance(sample.get('facts'), dict):
				raise ValueError('No independent DOM facts')
			facts = sample['facts']
			facts_document = (facts.get('url'), facts.get('document_id'))
			sampled_page = sample.get('page')
			if not sampled_page:
				if expected_document[1] is not None and facts_document != expected_document:
					raise ValueError('Document changed during independent evaluation')
				raise ValueError('No independent DOM facts')
			sampled_document = document_key(sampled_page)
			sampled_fingerprint = fingerprint(sampled_page)
			expected_fingerprint = page.get('dom_fingerprint', page.get('fingerprint'))
			if expected_document[1] is not None:
				if sampled_document != expected_document:
					raise ValueError('Document changed during independent evaluation')
				if facts_document != sampled_document:
					raise ValueError('Independent facts do not match the sampled document')
			elif sampled_fingerprint != expected_fingerprint:
				raise ValueError('Page changed during independent evaluation')
			evidence = {
				'url': facts.get('url'),
				'facts': facts,
				'sampled_page': {
					**{key: deepcopy(sampled_page[key]) for key in ('url', 'title', 'text', 'content') if key in sampled_page},
					'observed_at': facts.get('sampled_at'),
					'source_document': sampled_document[1],
					'target_id': target,
					'dom_fingerprint': sampled_fingerprint,
					'agent_dom_fingerprint': expected_fingerprint,
					'changed_since_observation': sampled_fingerprint != expected_fingerprint,
				},
				'observed_controls': deepcopy(sampled_page.get('actions', [])),
			}
			form_state = _form_state_evidence(sampled_page, facts, self._form_descriptions)
			if form_state is not None:
				evidence['form_state'] = form_state
			return {'sample_status': 'completed', 'evidence': evidence}
		except Exception as error:
			return {'sample_status': 'error', 'reason': str(error), 'evidence': {'error': str(error)}}


def _form_state_evidence(page, facts, memory):
	"""Join independent DOM descriptions to independently read final state, never to expected answers."""
	controls = facts.get('form_controls')
	if not isinstance(controls, list):
		controls = (facts.get('questionnaire') or {}).get('controls')  # Legacy independent archives.
	if not isinstance(controls, list):
		return None
	source = document_key(page)
	if (facts.get('document_id') is not None and facts['document_id'] != source[1]) or (
		facts.get('url') is not None and facts['url'] != source[0]
	):
		memory.clear()
		return {
			'source_url': facts.get('url'),
			'source_document': facts.get('document_id'),
			'controls': deepcopy(controls),
			'description_status': 'source_mismatch',
		}
	if source[1] is None or memory.get('source') != source:
		memory.clear()
		memory.update(source=source, descriptions={})
	descriptions = memory['descriptions']
	native = {c['node']: c for c in controls if type(c.get('node')) is int}
	descriptions = {node: entry for node, entry in descriptions.items() if node in native}

	def identity(control):
		keys = ['tag', 'type', 'name', 'form_owner']
		if control.get('type') in {'checkbox', 'radio'}:
			keys.append('value')
		return {key: control.get(key) for key in keys}

	described = set()
	for action in page.get('actions', []):
		node = action.get('node')
		if node not in native or node in described:
			continue
		described.add(node)
		entry = {
			'identity': identity(native[node]),
			'description': {
				'label': action.get('label', '').split(' → ')[0][:200],
				'role': action.get('role'),
				'context': deepcopy(action.get('context')),
				'source_url': source[0],
				'source_document': source[1],
				'observed_at': facts.get('sampled_at') or page.get('observed_at'),
			},
		}
		previous = descriptions.get(node, {})
		history = previous.get('history', [])
		# 相同描述只刷新来源；滚出表头后的较短描述不能抹掉之前有出处的完整行列信息。
		same = history and all(
			history[-1]['description'].get(key) == entry['description'].get(key) for key in ('label', 'role', 'context')
		)
		updated = history[:-1] + [entry] if same else history + [entry]
		omitted = previous.get('omitted_descriptions', 0) + max(0, len(updated) - 3)
		descriptions[node] = {**entry, 'history': updated[-3:], 'omitted_descriptions': omitted}
	memory['descriptions'] = descriptions
	joined = []
	for control in controls:
		entry = deepcopy(control)
		previous = descriptions.get(control.get('node'))
		if previous and previous['identity'] == identity(control):
			entry['observed_description'] = deepcopy(previous['description'])
		if previous:
			entry['description_history'] = [
				{
					**deepcopy(item['description']),
					'native_identity': deepcopy(item['identity']),
					'native_identity_matches': item['identity'] == identity(control),
				}
				for item in previous['history']
			]
			if previous.get('omitted_descriptions'):
				entry['omitted_descriptions'] = previous['omitted_descriptions']
		joined.append(entry)
	return {
		'source_url': source[0],
		'source_document': source[1],
		'sampled_at': facts.get('sampled_at') or page.get('observed_at'),
		'controls': joined,
	}
