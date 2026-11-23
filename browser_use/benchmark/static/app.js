const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="demo-token"]').content;
let state = null,
  busy = false,
  automatic = false;
const goals = {
  flights: 'Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. Stop when matching flight options are visible. Do not select or book a flight.',
  travel: 'Find a Design stay in Lisbon with Free cancellation and open Casa Flora.',
  research:
    "Open the article about using finite choices to control browser agents.",
};
const websites = {
  flights: "https://www.google.com/travel/flights?hl=en",
  travel: new URL("/fixture.html?scenario=travel", window.location.origin).href,
  research: new URL("/fixture.html?scenario=research", window.location.origin).href,
};
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const percent = (value) => `${(value * 100).toFixed(value < 0.01 ? 1 : 0)}%`;
let selectedCallId = null;
let renderedQuestionContexts = null;
const browserUseProvider = 'browser_use';
const executionChoice = 'direct';
const screenshotChoice = 'off';
let modelProfiles = [], modelProfileChoice = 'debug';
const modelProfileLabel = profile => ({ debug: '调试', final_report: '最终报告' })[profile] || '配置未记录';
const finalReportConfigured = () => modelProfiles.some(profile => profile.id === 'final_report' && profile.configured);
function renderModelProfile() {
  $('model-profile').value = modelProfileChoice;
  $('model-profile-final').disabled = !finalReportConfigured();
  const next = modelProfiles.find(p => p.id === modelProfileChoice);
  const describe = service => `${service.model}${service.base_url ? `（${service.base_url}）` : ''}`;
  $('model-profile-hint').textContent = next?.configured && next.planner && next.text
    ? `下次运行 · 执行 ${describe(next.planner)} · 终验 ${describe(next.text)}。最终报告选择仅使用一次。`
    : next ? '所选运行配置尚未就绪。' : '正在读取运行配置。';
  const current = view === 'baseline' ? (state?.model_profile || baseline.batch?.model_profile) : state?.model_profile;
  $('model-profile-current').textContent = current ? `当前记录：${modelProfileLabel(current)}` : '';
}
function consumeModelProfile() {
  modelProfileChoice = 'debug';
  renderModelProfile();
}
const screenshotPolicy = record => record?.screenshot_policy || (record?.screenshot_mode ? 'always' : 'off');
const screenshotLabel = policy => ({ off: '始终关闭', always: '始终开启', smart: '智能截图' })[policy] || '始终关闭';
const modeLabel = mode => mode === 'planned' ? 'Planner 子任务' : '直接执行';
const handoffLabel = trigger => ({ initial: '初始规划', STEP_DONE: '完成检查', UNABLE: '无法推进',
  subgoal_action_budget: '子任务动作上限', subgoal_round_budget: '子任务决策上限',
  no_progress: '持续无进展', action_cycle: '重复操作', stale_observation: '观察失效' })[trigger] || trigger;
const providerLabel = p => ({ laya: 'Laya', deepseek: 'DeepSeek', jev: 'Jev', fixture: '本地 fixture', browser_use: 'Browser Use' })[p] || 'Laya';
const nativeProvider = provider => provider === 'fixture' || provider === 'browser_use';
const callLabel = c => ({collect:'Collect helper · 页面摘要', summarize:'Summarize helper · 任务输出', verify:'Verify helper · 独立复核'})[c?.kind] || (c?.kind === 'planner' ? `Planner / ${handoffLabel(c.trigger)}` : c?.kind === 'text' ? 'Text helper' : c?.kind === 'deepseek' ? `DeepSeek (System1)${c.stage ? ` / 第 ${c.decision_round} 轮 / ${c.stage === 'operation' ? '操作选择' : '目标选择'}` : ''}` : jevScreenshot(c) ? 'Jev（截图）' : providerLabel(c?.kind));
const decisionCount = attempt => attempt?.decision_calls ?? ((attempt?.laya_calls || 0) + (attempt?.deepseek_calls || 0) + (attempt?.jev_calls || 0));
const callCounts = a => `${decisionCount(a)} / ${a.text_calls || 0} / ${a.planner_calls || 0}<small>收集 ${a.collect_calls || 0} · 总结 ${a.summarize_calls || 0}</small><small>${a.decision_rounds ?? decisionCount(a)} 决策轮 · ${a.replans || 0} 次改计划</small>`;
function tokenPanel(usage, title = 'Token 消耗', panelKey = 'tokens') {
  if (!usage) return '<p class="token-note">Token 用量暂无记录</p>';
  const n = key => Number.isFinite(usage[key]) ? escape(usage[key].toLocaleString('en-US')) : '—';
  return `<section class="token-panel" aria-label="${escape(title)}"><div class="token-totals">
    <div><span>Jev 总和</span><strong>${n('jev_total_tokens')}</strong></div>
    <div><span>LLM 总和</span><strong>${n('llm_total_tokens')}</strong></div></div>
    <details data-panel="${escape(panelKey)}"><summary>${escape(title)} · 明细</summary><table><thead><tr><th>类型</th><th>输入</th><th>输出</th><th>合计</th></tr></thead>
    <tbody><tr><th>Jev</th><td>${n('jev_input_tokens')}</td><td>不计</td><td>${n('jev_total_tokens')}</td></tr>
    ${['planner', 'other'].map(g => `<tr><th>${g === 'planner' ? 'Planner' : '其他'}</th><td>${n(g+'_input_tokens')}</td><td>${n(g+'_output_tokens')}</td><td>${n(g+'_total_tokens')}</td></tr>`).join('')}</tbody></table>
    <p class="token-note">其他：Text / Summarize / Collect，及启用时的 DeepSeek System1。排除 Verify 和 Laya。</p></details>
    ${usage.complete ? '' : `<p class="token-note">仅已知用量：${n('missing_usage_calls')} 次调用缺少 usage${usage.unavailable_attempts ? `，${n('unavailable_attempts')} 次运行缺少记录` : ''}</p>`}</section>`;
}
const evaluationCount = a => a?.verify_calls ? `<small>独立评估 ${escape(a.verify_calls)} 次</small>` : '';
const executionCalls = () => (state?.model_calls || []).filter(c => c.kind !== 'verify');
let displayedTargetIndex = null, hoveredElementIndex = null, overlayContextKey = null;
let view = 'free', freeState = null;
let baseline = { tasks: [], active: false, batch: null, runs: [] };
const baselineGroups = {
  all: { label: '全部用例', action: '运行全部', help: '切换分组可查看各组用例。' },
  success: { label: '已成功', action: '复测已成功组', help: '最近一次确认达成目标的用例。' },
  partial: { label: '部分成功', action: '复测部分成功组', help: '已交付报告，部分任务要求仍未达成。' },
  other: { label: '其他用例', action: '复测其他用例', help: '包含失败、未运行或按本轮范围跳过的用例。' },
};
let baselineGroup = 'success', baselineSearch = '';
try {
  const saved = window.localStorage?.getItem('jev-baseline-group');
  if (Object.hasOwn(baselineGroups, saved)) baselineGroup = saved;
} catch { /* Storage may be disabled. */ }
let selectedRunId = null, selectedAttemptId = null, selectedFrame = null;
let pinnedRun = false, pinnedAttempt = false;
let baselineBusy = false, baselinePolling = false, baselineEpoch = 0;
let promptRecordKey = null;
const baselineLabels = {
  queued: '等待运行', pending: '等待运行', preparing: '准备中', running: '运行中',
  stopping: '正在停止', completed: '已结束', finished: '已结束', done: '已停止',
  passed: '通过', success: '通过', failed: '策略失败', policy_failure: '策略失败',
  environment_blocked: '环境阻塞', blocked: '环境阻塞', timeout: '超时',
  execution_error: '执行错误', error: '执行错误', unknown: '评估未知',
  evaluation_unknown: '评估未知', cancelled: '已取消', interrupted: '已中断',
  met: '目标达成', not_met: '目标未达成',
  evidence_insufficient: '证据不足', evaluation_error: '评估调用失败', not_evaluated: '未评估',
  transport_error: '评估调用失败', invalid_response: '评估响应无效', not_run: '未评估',
};
const baselineLabel = value => baselineLabels[value] || value || '未运行';
const duration = ms => ms == null ? '—' : `${(ms / 1000).toFixed(1)} s`;
const resultName = attempt => {
  const result = attempt?.evaluation;
  if (!result) return '';
  if (typeof result === 'string') return baselineLabel(result);
  const final = result.final || result;
  if (final.evaluation_status === 'completed') return final.verdict === 'unknown' ? '证据不足' : baselineLabel(final.verdict);
  if (final.evaluation_status) return baselineLabel(final.evaluation_status);
  if (result.final?.outcome) return baselineLabel(result.final.outcome);
  if (result.outcome || result.status || result.result) return baselineLabel(result.outcome || result.status || result.result);
  return result.passed === true ? '通过' : result.passed === false ? '未通过' : '';
};
const executionName = attempt => attempt?.execution_status
  ? `执行：${baselineLabel(attempt.execution_status)}` : baselineLabel(attempt?.status);
function formatPayload(payload) {
  return JSON.stringify(payload, (key, value) => {
    if (key === "content" && typeof value === "string") {
      try { return JSON.parse(value); } catch { /* Plain-text message. */ }
    }
    return value;
  }, 2);
}
function requestImages(call) {
  return Object.entries(call?.request_images || {}).map(([part, data]) =>
    typeof data === 'string' && /^[A-Za-z0-9+/]+={0,2}$/.test(data)
      ? `<figure class="model-image"><figcaption>提交给 Jev 的页面截图 · ${escape(part)}</figcaption><img src="data:image/jpeg;base64,${data}" alt="本次 Jev 请求的页面截图" /></figure>` : ''
  ).join('');
}
function observationIndex() {
  return selectedFrame === null ? (state?.observations?.length || 0) - 1 : Number(selectedFrame);
}
function selectedModelCall() {
  const calls = executionCalls();
  if (view !== 'baseline' && selectedFrame === null) return calls.find(c => c.id === selectedCallId) || calls.at(-1);
  if (observationIndex() === -1) return calls.find(c => c.id === selectedCallId);
  const linked = calls.filter(c => Number.isInteger(c.observation_index) && c.observation_index === observationIndex());
  return linked.find(c => c.id === selectedCallId) || linked.find(c => c.kind !== 'text') || linked[0];
}
function historicalDecision(call) {
  // Planner 的结论不是浏览器操作，不能借用上一轮概率。
  if (['planner', 'collect', 'summarize', 'verify'].includes(call?.kind)) return { decision: null, elements: [] };
  if (call?.kind === 'text') call = (state.model_calls || []).find(c => c.id === (call.decision_call_id ?? call.laya_call_id));
  if (call?.stage) {
    const group = state.model_calls.filter(c => c.kind === 'deepseek' && c.decision_round === call.decision_round);
    call = group.find(c => c.stage !== 'operation' && c.status === 'success') || group.find(c => c.stage === 'operation');
  }
  if (!call) return { decision: null, elements: [] };
  const elements = call.indexed_elements || call.decision_input?.state?.elements || call.request?.state?.elements || [];
  if (call.decision && call.status === 'success') return { elements, decision: call.decision };
  const answers = call.response?.answers;
  const operation = call.operation;
  if (call.status !== 'success' || !operation || !answers?.operation?.probabilities) {
    return { decision: null, elements };
  }
  const target = answers[operation.toLowerCase() + '_target'];
  return { elements, decision: {
    operation, target: call.target, choice: operation, latency_ms: call.latency_ms,
    operation_probabilities: answers.operation.probabilities,
    target_probabilities: target?.probabilities || {}, target_confidence: target?.confidence,
  } };
}
const renderedHTML = new WeakMap();
function stableHTML(id, html, key) {
  const node = $(id);
  const previous = renderedHTML.get(node);
  // User toggles change serialized innerHTML. Compare render inputs so polling keeps the actual DOM nodes.
  if (previous?.html === html && previous.key === key) return;
  const old = node.dataset?.renderKey === key ? Array.from(node.querySelectorAll?.('[data-panel]') || []).map(el => ({
    name: el.dataset.panel, top: el.scrollTop, left: el.scrollLeft, open: el.open,
  })) : [];
  node.innerHTML = html;
  renderedHTML.set(node, { html, key });
  if (node.dataset) node.dataset.renderKey = key;
  for (const el of node.querySelectorAll?.('[data-panel]') || []) {
    const saved = old.find(s => s.name === el.dataset.panel);
    if (saved) { el.scrollTop = saved.top; el.scrollLeft = saved.left; if (saved.open !== undefined) el.open = saved.open; }
  }
}
function stagePanels(selected) {
  if (selected?.kind !== 'deepseek' || !selected.stage) return null;
  const calls = state.model_calls.filter(c => c.kind === 'deepseek' && c.decision_round === selected.decision_round);
  const op = calls.find(c => c.stage === 'operation');
  const target = calls.find(c => c.stage !== 'operation');
  const missing = op?.status === 'success' && !['CLICK', 'TYPE_TEXT', 'SELECT'].includes(op.operation)
    ? '该操作无需目标调用' : '目标调用尚未发出';
  const panels = [{ title: '操作选择', call: op }, { title: '目标选择', call: target }];
  const html = kind => panels.map(({ title, call }, i) => {
    const payload = kind === 'input' ? call?.request?.messages : kind === 'request' ? call?.request : call?.response;
    const content = kind === 'input' && call?.laya_context && payload
      ? payload.map(m => `${m.role}:\n${m.content}`).join('\n\n')
      : payload ? formatPayload(payload) : call ? '等待记录' : missing;
    const metrics = kind === 'input' && call?.laya_context
      ? `<p class="context-metrics">Laya tokenizer: ${call.laya_context.input_tokens} / ${call.laya_context.max_tokens} tokens</p>` : '';
    return `<details class="question-context" data-panel="${kind}-${i}" open><summary>${title} · ${escape(call?.status || missing)}</summary>${metrics}<pre data-panel="${kind}-${i}-body">${escape(content)}</pre>${call?.error ? `<p class="muted">${escape(call.error)}</p>` : ''}</details>`;
  }).join('');
  return { input: html('input'), request: html('request'), output: html('output'),
    key: `${selectedRunId}/${selectedAttemptId}/${selected.decision_round}` };
}
function decisionTrail(calls) {
  const rows = [], rounds = new Map();
  for (const call of calls) {
    if (call.kind === 'verify') continue;
    const grouped = call.kind === 'deepseek' && call.stage && Number.isInteger(call.decision_round);
    let row = grouped ? rounds.get(call.decision_round) : null;
    if (!row) {
      row = { call, members: [] };
      rows.push(row);
      if (grouped) rounds.set(call.decision_round, row);
    }
    row.members.push(call);
  }
  return rows.map(({ call, members }) => {
    const grouped = call.kind === 'deepseek' && call.stage && Number.isInteger(call.decision_round);
    const operation = members.find(c => c.stage === 'operation') || call;
    const target = members.find(c => c.stage && c.stage !== 'operation');
    const status = members.some(c => c.status === 'error' || c.application_status === 'rejected') ? 'error'
      : members.some(c => c.status === 'pending') ? 'pending'
      : grouped && ['CLICK', 'TYPE_TEXT', 'SELECT'].includes(operation.operation) && !target
        ? '目标未调用' : members.at(-1).status;
    return { ...call, members, status,
      operation: operation.operation, target: target?.target || call.target,
      latency_ms: members.reduce((total, c) => total + (c.latency_ms || 0), 0),
      label: grouped ? `DeepSeek (System1) / 第 ${call.decision_round} 轮` : callLabel(call),
    };
  });
}
function jevScreenshot(call) {
  return call?.kind === 'jev' && (call.request_format === 'multipart/form-data'
    || Object.keys(call.request_images || {}).length > 0);
}
function traceCard(call, index, active) {
  const label = ({planner:'Planner', text:'Text helper', collect:'Collect', summarize:'Summarize', verify:'Verify'})[call.kind]
    || (call.kind === 'jev' ? jevScreenshot(call) ? 'Jev（截图）' : 'Jev' : call.kind === 'deepseek' ? 'DeepSeek' : call.label);
  const operation = call.operation === 'UNABLE' && call.decision?.forced_unable === true
    ? 'UNABLE(强制)' : call.operation;
  const action = call.planner_result?.decision || operation || call.field
    || ({collect:'页面摘要', summarize:'生成任务输出', verify:'核验报告', text:'输入文本'})[call.kind] || '等待结果';
  const status = ({success:'成功', error:'失败', pending:'进行中'})[call.status] || call.status;
  const description = [`${index + 1}. ${call.label}`, action, call.model, call.target ? `target ${call.target}` : '',
    call.subgoal_id, `${call.latency_ms ?? 0} ms`, status].filter(Boolean).join(' · ');
  return `<button type="button" class="trace-row ${active ? 'active' : ''}" data-call-id="${call.id}" aria-pressed="${active}" aria-label="${escape(description)}" title="${escape(description)}"><span class="trace-top"><span class="number">${String(index + 1).padStart(2, '0')}</span><span class="call-status ${call.status === 'error' ? 'failed' : ''}">${escape(status)}</span></span><span class="trace-main"><span class="trace-kind">${escape(label)}</span><b>${escape(action)}</b></span><span class="trace-bottom"><span class="time">${call.latency_ms ?? 0} ms</span><small>${call.target ? '#'+escape(call.target) : ''}</small></span></button>`;
}
function renderModelCalls() {
  const calls = executionCalls();
  const selected = selectedModelCall();
  const exactPayload = ['collect', 'summarize', 'verify'].includes(selected?.kind)
    ? value => JSON.stringify(value, null, 2) : formatPayload;
  const rows = decisionTrail(calls);
  const selectedRow = rows.find(row => row.members.some(c => c.id === selected?.id));
  const historyHTML = rows.length ? rows.map((call, index) => traceCard(call, index, call === selectedRow)).join('')
    : '<p class="muted">No model calls yet.</p>';
  stableHTML('history', historyHTML, `${view}/${selectedRunId}/${selectedAttemptId}/${state.started_at ?? state.goal ?? ''}`);
  $("step-count").textContent = `${state.decision_rounds ?? state.decisions?.length ?? 0} rounds / ${calls.length} execution calls / ${state.history?.length || 0} actions`;
  $("selected-call").textContent = selected
    ? `${selectedRow?.label || callLabel(selected)} / Call ${selectedRow?.members.map(c => c.id).join(', ') || selected.id} / ${selected.model || selected.status}`
    : 'No call selected';
  const requestText = selected?.request
    ? exactPayload(selected.request) : 'No request recorded.';
  if ($('model-state').textContent !== requestText) $('model-state').textContent = requestText;
  const contexts = selected?.response?.question_contexts;
  const hasContexts = Array.isArray(contexts) && contexts.length > 0;
  const stages = stagePanels(selected);
  const contextHTML = stages ? stages.input : hasContexts
    ? `<p class="context-total">${contexts.length} sequences / ${contexts.reduce((n, c) => n + c.input_tokens, 0)} input tokens total</p>` + contexts.map(c =>
      `<section class="question-context"><h3>${escape(c.question)}</h3><p class="context-metrics">Pass ${escape(c.pass)} / <strong>${escape(c.input_tokens)} / ${escape(c.max_tokens)} tokens</strong></p><pre>${escape(c.context)}</pre></section>`
    ).join('')
    : nativeProvider(selected?.kind) || ['deepseek', 'planner', 'text', 'collect', 'summarize', 'verify'].includes(selected?.kind) ? `<pre data-panel="messages">${escape(JSON.stringify(selected.request?.messages || [], null, 2))}</pre>`
    : selected?.kind === 'jev' ? `${requestImages(selected)}<pre>${escape(formatPayload({ state: selected.request?.state, questions: selected.request?.questions }))}</pre><p class="muted">Jev 服务端未提供最终编码上下文；此处为实际提交内容。${selected.request_format === 'multipart/form-data' ? ' 图片以 multipart 文件提交，HTTP 请求面板显示 JSON 部分。' : ''}</p>`
    : selected?.kind === 'laya' ? '<p class="muted">Per-question context unavailable in this response.</p>' : '';
  if (renderedQuestionContexts !== contextHTML) {
    stableHTML('question-contexts', contextHTML, stages?.key || String(selected?.id));
    renderedQuestionContexts = contextHTML;
  }
  $('stage-requests').hidden = $('stage-outputs').hidden = !stages;
  $('model-state').hidden = $('model-output').hidden = !!stages;
  if (stages) {
    stableHTML('stage-requests', stages.request, stages.key);
    stableHTML('stage-outputs', stages.output, stages.key);
  }
  const output = ['laya', 'jev'].includes(selected?.kind) && selected.response != null
    ? Object.fromEntries(Object.entries(selected.response).filter(([key]) => key !== 'question_contexts'))
    : selected?.response;
  const outputText = selected?.error
    ? exactPayload({ response: output ?? null, error: selected.error })
    : output != null ? exactPayload(output) : 'No response recorded.';
  if ($('model-output').textContent !== outputText) $('model-output').textContent = outputText;
}
async function call(name, body = {}) {
  const response = await fetch(`/api/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Demo-Token": token },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || "Request failed");
  state = data;
  freeState = data;
  if (name === "reset") { selectedCallId = null; selectedFrame = null; consumeModelProfile(); }
  render();
  return data;
}
function controls() {
  const live = state?.page && !["done", "blocked"].includes(state.status);
  const locked = busy || baseline.active || view !== 'free';
  $('execution-mode').disabled = busy || automatic || baseline.active || baselineBusy;
  $('model-profile').disabled = busy || automatic || baseline.active || baselineBusy;
  $('screenshot-off').disabled = busy || automatic || baseline.active || baselineBusy;
  renderModelProfile();
  $('choose').textContent = state?.planning?.pending ? '请求云端规划' : 'Choose next';
  $('baseline-tab').disabled = busy;
  $("start").disabled = locked;
  $("scenario").disabled = locked;
  $("website-url").disabled = locked;
  $("goal").disabled = locked;
  $("choose").disabled = locked || !live;
  $("execute").disabled = locked || !state?.decision || !live;
  $("auto").disabled = locked || !live;
  $("auto").hidden = automatic;
  $("stop").hidden = !automatic;
  $("download").disabled = !state?.model_calls?.length && !state?.history?.length;
}
async function perform(fn, label) {
  if (busy || baseline.active || view !== 'free') return;
  busy = true;
  $("error").hidden = true;
  controls();
  $("status").textContent = label;
  try {
    await fn();
  } catch (error) {
    automatic = false;
    try {
      state = await fetch("/api/state").then((r) => r.json());
      freeState = state;
      render();
    } catch {
      /* Preserve the original failure if the server disconnected. */
    }
    $("error").textContent = error.message;
    $("error").hidden = false;
    $("status").textContent = "Paused · needs attention";
  } finally {
    busy = false;
    controls();
  }
}
// 原始指令只来自运行快照，不读取尚未提交的输入框或当前子任务。
function renderOriginalPrompt() {
  const current = view === 'baseline'
    ? Boolean(selectedRunId && selectedAttemptId && state?.baseline_attempt &&
      promptRecordKey === `${selectedRunId}/${selectedAttemptId}/${baselineEpoch}`)
    : Boolean(state && (typeof state.goal === 'string' || state.page || state.model_calls?.length));
  const text = current ? (typeof state.goal === 'string' && state.goal.trim() ? state.goal : '未记录原始任务') : '';
  $('original-prompt-panel').hidden = !current;
  // 相同轮询结果不重写文本节点，保留用户正在进行的文本选择。
  if ($('original-prompt').textContent !== text) $('original-prompt').textContent = text;
}
// 根据选中的调用恢复当时计划；未选择历史时显示实时累计状态。
function renderPlanning() {
  const call = selectedModelCall();
  const historical = view === 'baseline' || selectedCallId !== null;
  const saved = historical ? call?.planning_after || call?.planning_snapshot : null;
  const plan = saved || state.planning;
  const planned = state.execution_mode === 'planned';
  $('planning-panel').hidden = !planned;
  $('plan').hidden = planned;
  $('planner-endpoint').hidden = !planned && executionChoice !== 'planned';
  $('planner-address').textContent = (state.planner || state.planner_default || freeState?.planner_default)?.base_url || 'Unavailable';
  $('planner-name').textContent = `Planner · ${(state.planner || state.planner_default || freeState?.planner_default)?.model || 'Unavailable'}`;
  // 旧 trace 不包含结构化计划，避免以新运行的计划填充历史。
  if (!planned || !plan) return;
  const actions = saved ? saved.actions : Math.max(0, (state.history?.length || 0) - plan.action_start);
  const rounds = saved ? saved.rounds : Math.max(0, (state.decision_rounds || 0) - plan.round_start);
  const limits = plan.limits || {};
  $('planning-counts').textContent = `本段 ${actions} / ${limits.subgoal_actions ?? 10} 动作 · ${rounds} / ${limits.subgoal_rounds ?? 16} 轮 · Planner ${plan.calls} / ${limits.planner_calls ?? 12} · 重规划 ${plan.replans} 次`;
  const stopReason = saved ? saved.stop_reason : state.stop_reason;
  const goals = plan.subgoals || [];
  const completed = goals.filter((goal, i) => goal.status === 'completed' || i < plan.index).length;
  $('planning-status').textContent = `已完成 ${completed} 项 · 版本 ${plan.revision} · ${plan.pending ? '待交接：'+handoffLabel(plan.pending) : stopReason || '执行子任务'}`;
  const html = goals.map((goal, i) => {
    const status = goal.status === 'completed' || i < plan.index ? 'completed'
      : goal.status === 'blocked' || goal.status === 'failed' ? 'blocked'
      : i === plan.index ? 'current' : 'pending';
    const label = { completed: '已完成', current: '执行中', blocked: '已停止', pending: '待执行' }[status];
    const attempts = goal.attempts || [];
    const historyLabel = attempts.some(attempt => attempt.status === 'unverified') ? '重规划历史' : '失败历史';
    const history = attempts.length ? `<details class="plan-attempts" data-panel="attempts-${escape(goal.stage_id || goal.id || i)}"><summary>${historyLabel}（${attempts.length} 次）</summary><ol class="plan-attempt-list">${attempts.map((attempt, index) => {
      const uncertain = attempt.status === 'unverified';
      const trigger = attempt.review_trigger || attempt.failure_trigger;
      return `<li><div class="plan-attempt-title">方案 ${index + 1} · ${escape(attempt.title)}</div><p>${escape(attempt.objective)}</p><small>完成条件：${escape(attempt.completion)}</small><p class="plan-failure-reason">${uncertain ? '待核实' : '失败'}原因：${escape(attempt.review_reason || attempt.failure_reason || '未满足完成条件')}</p>${trigger ? `<small>交接原因：${escape(handoffLabel(trigger))}</small>` : ''}</li>`;
    }).join('')}</ol></details>` : '';
    return `<li class="${status}"><span class="plan-goal-status">${i + 1}<b>${label}</b></span><div class="plan-goal-body"><strong>${escape(goal.title)}</strong><p>${escape(goal.objective)}</p><small>完成条件：${escape(goal.completion)}</small>${goal.failure_reason ? `<p class="plan-failure-reason">${escape(goal.failure_reason)}</p>` : ''}${history}</div></li>`;
  }).join('');
  stableHTML('planning-goals', html || '<li class="plan-empty">等待生成第一个子任务计划。</li>',
    `${view}/${selectedRunId}/${selectedAttemptId}/${state.started_at ?? state.goal ?? ''}/${selectedCallId ?? 'live'}`);
}
function researchVersion() {
  const research = state?.research;
  if (!research?.versions) return null;
  const call = selectedCallId !== null ? selectedModelCall() : null;
  const revision = call ? call.research_after ?? call.research_before
    : selectedFrame !== null ? state.observations?.[observationIndex()]?.research_revision : research.revision;
  return research.versions.find(v => v.revision === revision) || null;
}
function sourceLink(url, label = '原文链接') {
  try {
    const parsed = new URL(url);
    if (['https:', 'http:'].includes(parsed.protocol)) return `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)}</a>`;
  } catch { /* Untrusted source text is never executable markup. */ }
  return '<span>未获取链接</span>';
}
function taskResult() {
  // New task output belongs to the task, independently of the selected notebook step.
  return Array.isArray(state?.research?.pages) ? state?.result : researchVersion()?.result;
}
function renderResearch() {
  const research = state?.research, version = researchVersion();
  const call = selectedCallId !== null ? selectedModelCall() : null;
  const historical = selectedCallId !== null || selectedFrame !== null;
  const pages = version?.pages, modern = Array.isArray(research?.pages);
  $('research-status').textContent = version
    ? `${historical ? '此步完成后 · ' : ''}${pages ? pages.length + ' 次采集（非条目数）' : (version.entries || []).length + ' 条资料'}` : '未记录';
  const helper = ['collect', 'summarize'].includes(call?.kind);
  const failed = helper && (call.application_status === 'rejected' || call.status === 'error');
  const reason = helper ? call.application_error || call.error || '' : '';
  const summaryFailed = modern && call?.kind === 'collect' && (call.application_status === 'summary_failed' || failed);
  const notice = summaryFailed ? `网页原文已保存，摘要生成失败：${reason}`
    : modern && call?.kind === 'summarize' && failed ? `任务输出生成失败：${reason}`
    : failed ? `未保存资料：${reason}` : reason;
  if ($('research-notice').textContent !== notice) $('research-notice').textContent = notice;
  const entries = pages ? pages.map(page => `<article class="research-entry"><h3>${escape(page.title || '已收集页面')}</h3><p>${sourceLink(page.url)}</p><p class="research-entry-text">${escape(page.summary || (page.summary_status === 'pending' ? '原文已保存，正在生成摘要…' : '原文已保存，摘要暂不可用。'))}</p><details data-panel="page-${escape(page.id)}"><summary>查看网页原文</summary><pre data-panel="page-${escape(page.id)}-text">${escape(page.text)}</pre></details></article>`).join('')
    : (version?.entries || []).map(entry => `<article class="research-entry"><h3>${escape(entry.title)}</h3><p class="research-entry-text">${escape(entry.text)}</p><blockquote><strong>原文引用</strong><p>${escape(entry.quote)}</p></blockquote><p>${sourceLink(entry.url)}</p></article>`).join('');
  stableHTML('research-entries', entries || '<p class="muted">尚未保存资料。</p>', `${version?.revision}`);
  const sources = (research?.sources || []).filter(s => version?.source_ids?.includes(s.id));
  $('research-source-panel').hidden = !sources.length;
  stableHTML('research-sources', sources.map(s => `<details data-panel="source-${escape(s.id)}"><summary>${escape(s.title)} · ${escape(s.observed_at)}</summary><p>${sourceLink(s.url)} <button type="button" data-source-observation="${s.observation_index}">查看当时网页</button></p><pre data-panel="source-${escape(s.id)}-text">${escape(s.text)}</pre></details>`).join(''), `${version?.revision}`);
  const result = taskResult();
  $('research-report').hidden = !result;
  const markdown = result?.text || result?.markdown || '';
  if ($('report-content').textContent !== markdown) $('report-content').textContent = markdown;
  $('report-copy').disabled = $('report-download').disabled = !markdown;
}
function render() {
  renderOriginalPrompt();
  renderResearch();
  if (!state) return;
  const provider = state.system1_provider || browserUseProvider;
  const native = nativeProvider(provider);
  const config = provider === browserUseProvider ? modelProfiles.find(p => p.id === 'debug')?.planner : null;
  const policy = state.page ? screenshotPolicy(state) : screenshotChoice;
  const modelName = state.system1_model || config?.model;
  $('system1-name').textContent = `${native ? '执行 LLM' : 'System1'} · ${providerLabel(provider)}${modelName ? ' · '+modelName : ''}${policy !== 'off' ? ' · '+screenshotLabel(policy) : ''}`;
  $("laya-address").textContent = state.system1_base_url || config?.base_url || state.laya_base_url || freeState?.laya_base_url || "Unavailable";
  $("text-model-address").textContent = state.text_model_base_url || freeState?.text_model_base_url || "Unavailable";
  $('text-model-name').textContent = `${native ? '终验模型' : 'Text model'} · ${state.text_model || freeState?.text_model || 'Unavailable'}`;
  $("helper").textContent = `${native ? '终验模型' : 'Text helper'} · ${state.text_model || freeState?.text_model || 'Unavailable'}`;
  renderPlanning();
  $("plan").innerHTML = (state.plan || [])
    .map(
      (goal, i) =>
        `<div class="plan-step ${i === state.plan_index ? "current" : ""}"><span>${i < state.plan_index ? "✓" : i + 1}</span>${escape(goal)}</div>`,
    )
    .join("");
  const inspecting = view === 'baseline' || selectedFrame !== null || (selectedCallId !== null && Number.isInteger(selectedModelCall()?.observation_index));
  const frame = state.observations?.[observationIndex()];
  const historical = inspecting ? historicalDecision(selectedModelCall()) : null;
  const page = inspecting ? frame?.page || (frame ? { url: frame.url, title: frame.title, actions: [] } : null) : state.page;
  $('viewport').style?.setProperty('aspect-ratio', page?.w > 0 && page?.h > 0 ? `${page.w} / ${page.h}` : '1120 / 780');
  const d = inspecting ? historical.decision : state.decision || (state.status === 'done' && selectedModelCall()?.kind !== 'planner' ? state.decisions?.at(-1) : null);
  const observedElements = inspecting ? (historical.elements.length ? historical.elements : frame?.elements || []) : state.elements || [];
  displayedTargetIndex = d?.target?.split(':')[0] || null;
  const overlayKey = inspecting ? `${selectedRunId}/${selectedAttemptId}/${observationIndex()}/${selectedModelCall()?.id}` : page?.fingerprint;
  if (overlayContextKey !== overlayKey) hoveredElementIndex = null;
  overlayContextKey = overlayKey;
  const labels = {
    idle: "Ready to explore",
    ready: "Page observed · ready for a decision",
    predicted: "Choice ready · inspect or execute",
    collecting: "网页原文已保存 · 正在生成页面摘要",
    finalizing: "正在生成任务输出",
    done: `${providerLabel(provider)} reports complete · inspect the page`,
    blocked: "Stopped · no supported next action",
    planning: "待云端交接 · 无浏览器动作",
  };
  $("status").textContent = labels[state.status] || state.status;
  renderModelCalls();
  if (!page) {
    $("screenshot").hidden = true;
    $("empty").hidden = false;
    $('empty-title').textContent = view === 'baseline' ? '尚无页面观察' : 'Watch the decision happen.';
    $('empty-description').textContent = view === 'baseline' ? '' : 'Start a task to see the browser and its actions.';
    $("targets").innerHTML = '';
    $("choices").innerHTML = '';
    $("operation-choices").innerHTML = '';
    $("url").textContent = 'No page observed';
    $("page-title").textContent = '';
    $("action-count").textContent = '0 elements';
    $("choice-title").textContent = 'Waiting for a page';
    for (const id of ['latency', 'confidence', 'completion']) $(id).textContent = '—';
    renderObservations();
    controls();
    return;
  }
  $("empty").hidden = true;
  const screenshot = page.screenshot_url || (page.screenshot ? `data:image/jpeg;base64,${page.screenshot}` : '');
  $("screenshot").hidden = !screenshot;
  $("screenshot").src = screenshot;
  $("empty").hidden = Boolean(screenshot);
  $('empty-title').textContent = view === 'baseline' ? '截图不可用' : 'Screenshot unavailable';
  $('empty-description').textContent = page.screenshot_error || '';
  $("url").textContent = page.url;
  $("page-title").textContent = page.title;
  $("action-count").textContent = `${observedElements.length} elements`;
  const chosen = (page.actions || []).find((a) => a.id === d?.choice);
  const chosenElement = observedElements.find(e => e.index === d?.target?.split(':')[0]);
  $("choice-title").textContent = d
    ? chosen?.label || chosenElement?.options?.find(o => o.index === d.target)?.label || chosenElement?.label || d.choice
    : inspecting ? '该次观察没有可用决策' : "Choose an action";
  $("latency").textContent = d ? `${d.latency_ms} ms` : "—";
  $("confidence").textContent = d?.target_confidence != null ? percent(d.target_confidence) : "—";
  $("completion").textContent = d ? d.operation : "—";
  $("ranking-note").textContent = d?.provider === 'deepseek' || nativeProvider(d?.provider)
    ? `${providerLabel(d.provider)} · 概率未提供` : d ? `Ranked by ${providerLabel(d.provider)}` : "Unranked";
  const op = Object.entries(d?.operation_probabilities || {}).sort((a,b)=>b[1]-a[1]);
  $("operation-choices").innerHTML = op.map(([name,p]) =>
    `<span class="operation-choice ${name === d.operation ? 'best' : ''}">${escape(name)} <b>${percent(p)}</b></span>`).join('');
  const probability = e => d?.target_probabilities?.[e.index] ??
    Math.max(-1, ...(e.options || []).map(o=>d?.target_probabilities?.[o.index] ?? -1));
  const selectedIndex = d?.target?.split(':')[0];
  const elements = [...observedElements];
  if (d) elements.sort((a,b)=>probability(b)-probability(a));
  $("choices").innerHTML = elements.map(e => {
    const p = probability(e);
    return `<div class="choice ${selectedIndex === e.index ? 'best' : ''}" data-action="${escape(e.index)}"><span class="choice-id">[${escape(e.index)}]</span><div class="choice-label">${escape(e.label)}<small>${escape(e.role)} · ${escape(e.operations.join(' / '))}${e.value ? ' · '+escape(e.value) : ''}${e.checked !== undefined ? ' · checked '+escape(e.checked) : ''}</small>${p >= 0 ? `<div class="bar" style="--probability:${p*100}%"></div>` : ''}</div><span class="probability">${p >= 0 ? percent(p) : '—'}</span></div>`;
  }).join('');
  const targets = new Map();
  for (const a of page.actions || []) {
    if (['click', 'fill', 'select'].includes(a.kind) && a.node != null && !targets.has(a.node)) targets.set(a.node, a);
  }
  const observedIndices = new Set(observedElements.map(e => e.index));
  $("targets").innerHTML = [...targets.values()].map((a,i) => {
    const index=String(i+1);
    const r = a.rect;
    if (!observedIndices.has(index) || !r || ![r.x, r.y, r.w, r.h, page.w, page.h].every(Number.isFinite)
      || r.w <= 0 || r.h <= 0 || page.w <= 0 || page.h <= 0) return '';
    return `<div class="target ${index === selectedIndex || index === hoveredElementIndex ? 'selected' : ''}" data-action="${index}" style="left:${100*r.x/page.w}%;top:${100*r.y/page.h}%;width:${100*r.w/page.w}%;height:${100*r.h/page.h}%"><span>${index}</span></div>`;
  }).join('');
  renderObservations();
  updateOverlayVisibility();
  controls();
}
function renderObservations() {
  const frames = state?.observations || [];
  const inspecting = view === 'baseline' || selectedFrame !== null;
  $('browser-mode').textContent = inspecting ? 'RECORD' : 'LIVE';
  $('observation-toolbar').hidden = !frames.length;
  if (!frames.length) return;
  const index = selectedFrame === null ? frames.length - 1 : Number(selectedFrame);
  const frame = frames[index];
  const frameOptions = frames.map((f, i) => `<option value="${i}">${i + 1} · Step ${escape(f.step)}${f.phase ? ' / ' + escape(f.phase) : ''} · ${escape(f.title || f.page?.title || '观察')}</option>`).join('');
  if ($('observation-step').innerHTML !== frameOptions) $('observation-step').innerHTML = frameOptions;
  if (!frame) {
    $('observation-step').value = '';
    $('observation-status').textContent = '该调用没有关联的观察记录';
    return;
  }
  $('observation-step').value = String(index);
  if (!inspecting) return;
  const src = frame.screenshot_url || frame.page?.screenshot_url || (frame.page?.screenshot ? `data:image/jpeg;base64,${frame.page.screenshot}` : '');
  $('screenshot').src = src;
  $('screenshot').hidden = !src;
  $('empty').hidden = Boolean(src);
  $('empty-title').textContent = '截图不可用';
  $('empty-description').textContent = frame.screenshot_error || '该次观察未保存截图';
  $('url').textContent = frame.url || frame.page?.url || state.page?.url || '';
  $('page-title').textContent = frame.title || frame.page?.title || state.page?.title || '';
  $('observation-status').textContent = frame.screenshot_error || (!src ? '该次观察未保存截图' : `第 ${frames.indexOf(frame) + 1} / ${frames.length} 次观察`);
}
function renderBaselineEvaluation(selected) {
  const current = view === 'baseline' && selected && state?.baseline_attempt?.id === selected.id
    && promptRecordKey === `${selectedRunId}/${selected.id}/${baselineEpoch}`;
  const calls = current ? (state.model_calls || []).filter(c => c.kind === 'verify') : [];
  const evaluation = selected?.evaluation || (current ? state.evaluation : null);
  const final = evaluation?.final || evaluation;
  const count = Math.max(calls.length, selected?.verify_calls || 0);
  $('baseline-evaluation-panel').hidden = !evaluation && !count;
  const last = calls.at(-1);
  const status = last?.status === 'pending' ? '正在评估'
    : resultName({evaluation}) || '尚无评估结论';
  const summary = `${selected?.execution_status ? `${executionName(selected)} · ` : ''}${status}${count ? ` · 独立评估 ${count} 次` : ''}`;
  const reason = [selected?.execution_reason, final?.reason || final?.semantic_review?.reason || last?.error].filter(Boolean).join('；');
  if ($('baseline-evaluation-status').textContent !== summary) $('baseline-evaluation-status').textContent = summary;
  if ($('baseline-evaluation-reason').textContent !== reason) $('baseline-evaluation-reason').textContent = reason;
  const record = evaluation ? formatPayload(evaluation) : '等待评估结果';
  if ($('baseline-evaluation').textContent !== record) $('baseline-evaluation').textContent = record;
  const html = calls.map((call, i) => {
    const status = ({pending:'进行中', success:'成功', error:'失败'})[call.status] || call.status;
    const output = call.error ? { response: call.response ?? null, error: call.error } : call.response;
    const inputText = call.request ? JSON.stringify(call.request, null, 2) : '等待请求记录';
    const outputText = output == null ? '等待模型返回' : JSON.stringify(output, null, 2);
    return `<details class="question-context" data-panel="verify-${escape(call.id)}"><summary>评估调用 ${i + 1} · ${escape(status)} · ${escape(call.model || '')} · ${call.latency_ms ?? 0} ms</summary><details data-panel="verify-${escape(call.id)}-input"><summary>评估模型输入</summary><pre data-panel="verify-${escape(call.id)}-input-text">${escape(inputText)}</pre></details><details data-panel="verify-${escape(call.id)}-output"><summary>评估模型输出</summary><pre data-panel="verify-${escape(call.id)}-output-text">${escape(outputText)}</pre></details></details>`;
  }).join('') + (current ? (state.reevaluations || []).map((review, i) =>
    `<details class="question-context" data-panel="reevaluation-${i}"><summary>归档重评 · ${escape(review.reevaluated_at || '')} · ${escape(resultName({evaluation:review}))}</summary><pre>${escape(formatPayload(review))}</pre></details>`).join('') : '');
  stableHTML('baseline-verification-calls', html, `${selectedRunId}/${selected?.id || ''}`);
}
function baselineGroupTasks(group = baselineGroup) {
  if (group === 'all') return baseline.tasks;
  const confirmed = new Set(baseline.confirmed_success?.task_ids || []);
  const partial = new Set(baseline.confirmed_success?.partial_task_ids || []);
  return baseline.tasks.filter(task => {
    const taskGroup = confirmed.has(task.id) ? 'success' : partial.has(task.id) ? 'partial' : 'other';
    return taskGroup === group;
  });
}
function visibleBaselineTasks() {
  const query = baselineSearch.trim().toLowerCase().replace(/^case\s*/, '');
  return baselineGroupTasks().filter(task => !query || (/^\d{1,3}$/.test(query)
    ? task.id.split('_', 1)[0] === query.padStart(2, '0')
    : `${task.id} ${task.name} ${task.capability || ''}`.toLowerCase().includes(query)));
}
function selectBaselineGroup(group) {
  if (!Object.hasOwn(baselineGroups, group)) return;
  baselineGroup = group;
  baselineSearch = '';
  $('baseline-case-search').value = '';
  try { window.localStorage?.setItem('jev-baseline-group', group); } catch { /* Storage may be disabled. */ }
  renderBaseline();
}
function renderBaseline() {
  const batch = baseline.batch;
  const attempts = batch?.attempts || [];
  const locked = baselineBusy || baseline.active || busy;
  const groupTasks = baselineGroupTasks();
  const visibleTasks = visibleBaselineTasks();
  const currentGroup = baselineGroups[baselineGroup];
  $('baseline-run-all').disabled = locked || !groupTasks.length;
  $('baseline-run-all').textContent = `▶ ${currentGroup.action}（${groupTasks.length}）`;
  for (const [group, { label }] of Object.entries(baselineGroups)) {
    const button = $(`baseline-group-${group}`);
    button.setAttribute('aria-pressed', String(group === baselineGroup));
    button.innerHTML = `${label} <span class="group-count">${baselineGroupTasks(group).length}</span>`;
  }
  $('baseline-group-title').textContent = `${currentGroup.label} · ${groupTasks.length}`;
  $('baseline-group-help').textContent = currentGroup.help;
  const countText = `当前显示 ${visibleTasks.length} / ${groupTasks.length} 个用例${baselineSearch.trim() ? ' · 搜索仅筛选显示，整组复测仍运行本组全部用例' : ''}`;
  if ($('baseline-group-count').textContent !== countText) $('baseline-group-count').textContent = countText;
  $('baseline-stop').disabled = baselineBusy || !baseline.active;
  $('baseline-open-folder').disabled = baselineBusy || !batch;
  $('baseline-rename').disabled = baselineBusy || !batch || !['completed', 'cancelled', 'error', 'interrupted'].includes(batch.status);
  $('baseline-delete').disabled = baselineBusy || !batch || !['completed', 'cancelled', 'error', 'interrupted'].includes(batch.status);
  $('baseline-repeats').disabled = locked;
  const runs = [...baseline.runs];
  if (batch && !runs.some(r => r.id === batch.id)) runs.unshift(batch);
  const runOptions = runs.length
    ? runs.map(r => `<option value="${escape(r.id)}">${r.name ? `${escape(r.name)} · ` : ''}${escape(r.id)} · ${escape(baselineLabel(r.status))}</option>`).join('')
    : '<option value="">尚无批次</option>';
  if ($('baseline-runs').innerHTML !== runOptions) $('baseline-runs').innerHTML = runOptions;
  $('baseline-runs').value = selectedRunId || batch?.id || '';
  $('baseline-runs').disabled = baselineBusy || !runs.length;
  $('baseline-status').textContent = batch ? `${baselineLabel(batch.status)}${baseline.active && batch.status !== 'running' ? ' · 有批次运行中' : ''}` : '尚未运行';
  $('baseline-task-heading').textContent = `${baseline.tasks.length} 项网页基线`;
  $('baseline-path').textContent = batch?.path || '';
  stableHTML('baseline-tasks', visibleTasks.map(task => {
    const matching = attempts.filter(a => a.task_id === task.id);
    const attempt = matching.find(a => ['running', 'preparing'].includes(a.status)) || matching.at(-1);
    const result = resultName(attempt);
    return `<article class="baseline-task-card" data-status="${escape(attempt?.status || 'idle')}" aria-labelledby="task-title-${escape(task.id)}">
      <div class="baseline-task-header"><span class="task-number">${escape(task.id.split('_', 1)[0])}</span><h3 id="task-title-${escape(task.id)}">${escape(task.name)}</h3><button type="button" class="baseline-task-run" data-run-task="${escape(task.id)}" title="运行${escape(task.name)}" aria-label="运行${escape(task.name)}" ${locked ? 'disabled' : ''}>▶</button></div>
      <p class="baseline-task-capability" title="${escape(task.capability)}">${escape(task.capability)}</p>
      <div class="baseline-task-result"><span class="baseline-task-status">${escape(executionName(attempt))}</span>${result ? `<span>${escape(result)}</span>` : ''}${evaluationCount(attempt)}</div>
      <div class="baseline-task-metrics"><span>${escape(attempt?.actions ?? '—')} 动作</span><span>${duration(attempt?.elapsed_ms)}</span></div>
      <div class="baseline-task-calls"><span>System1 <b>${attempt ? escape(decisionCount(attempt)) : '—'}</b></span><span>文本 <b>${attempt ? escape(attempt.text_calls || 0) : '—'}</b></span><span>规划 <b>${attempt ? escape(attempt.planner_calls || 0) : '—'}</b></span></div>
      ${matching.length ? tokenPanel(batch?.task_token_usage?.[task.id], '用例累计 Token', task.id) : ''}
    </article>`;
  }).join('') || '<p class="muted baseline-attempt-empty">没有匹配的用例。</p>', `${selectedRunId}/${baselineGroup}`);
  stableHTML('baseline-attempts', attempts.length ? attempts.map(a => {
    const task = baseline.tasks.find(t => t.id === a.task_id);
    const result = resultName(a);
    const reason = a.reason || (typeof a.evaluation === 'object' && a.evaluation?.reason) || '';
    return `<article class="baseline-task-card baseline-attempt-card${a.id === selectedAttemptId ? ' selected-attempt' : ''}" data-status="${escape(a.status)}">
      <div class="baseline-attempt-header"><h4>${escape(task?.name || a.task_id)}</h4><button type="button" data-attempt-id="${escape(a.id)}" aria-pressed="${a.id === selectedAttemptId}" ${['queued', 'pending'].includes(a.status) ? 'disabled' : ''}>查看记录</button></div>
      <p class="baseline-attempt-meta">第 ${escape(a.repeat)} 次 · ${providerLabel(a.system1_provider)}${screenshotPolicy(a) !== 'off' ? ' · '+screenshotLabel(screenshotPolicy(a)) : ''} · ${modeLabel(a.execution_mode)} · ${modelProfileLabel(a.model_profile)}</p>
      <div class="baseline-task-result"><span class="baseline-task-status">${escape(executionName(a))}</span>${result ? `<span class="${['通过', '目标达成'].includes(result) ? 'result-passed' : 'result-other'}">${escape(result)}</span>` : ''}${evaluationCount(a)}</div>
      <div class="baseline-task-metrics"><span>${escape(a.actions || 0)} 动作</span><span>${duration(a.elapsed_ms)}</span></div>
      ${reason ? `<p class="attempt-reason" title="${escape(reason)}">${escape(reason)}</p>` : ''}
      <div class="baseline-attempt-calls"><span>System1 / 文本 / 规划</span><div>${callCounts(a)}</div></div>
      ${tokenPanel(a.token_usage, '本次运行 Token', a.id)}
    </article>`;
  }).join('') : '<p class="muted baseline-attempt-empty">尚无运行记录</p>', selectedRunId);
  const passed = attempts.filter(a => a.status === 'passed' || resultName(a) === '通过').length;
  const blocked = attempts.filter(a => resultName(a) === '环境阻塞' || a.status === 'environment_blocked').length;
  const stats = batch?.outcome_statistics;
  $('baseline-summary').textContent = stats
    ? `${stats.met} / ${stats.attempts} 目标达成 · ${stats.not_met} 未达成 · ${stats.unknown} 证据不足 · ${stats.evaluation_errors} 评估调用失败 · ${stats.not_run} 未评估 · ${stats.legacy_unknown} 旧档案未判定 · 未判决 ${(100 * (stats.unjudged_rate || 0)).toFixed(1)}%`
    : attempts.length ? `${passed} / ${attempts.length} 通过 · ${blocked} 环境阻塞（旧方法）` : '';
  stableHTML('baseline-token-summary', batch ? tokenPanel(batch.token_usage, '批次累计 Token') : '', selectedRunId);
  const selected = attempts.find(a => a.id === selectedAttemptId);
  $('baseline-selected').textContent = selected ? `${baseline.tasks.find(t => t.id === selected.task_id)?.name || selected.task_id} · 第 ${selected.repeat} 次 · ${executionName(selected)}${resultName(selected) ? ' · ' + resultName(selected) : ''}` : '未选择运行记录';
  renderBaselineEvaluation(selected);
  controls();
}
async function baselineFetch(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Demo-Token': token }, body: JSON.stringify(body),
  };
  const response = await fetch(`/api/baseline${path}`, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '基线请求失败');
  return data;
}
function clearFreeTask() {
  // 新批次拥有浏览器时清除旧的实时计划，历史文件仍然保留。
  freeState = { ...freeState, page: null, status: 'idle', history: [], model_calls: [], decision: null, elements: [], plan: [], planning: null, execution_mode: 'direct' };
  if (view === 'free') { state = freeState; render(); }
}
function acceptBaseline(data) {
  const old = baseline.batch;
  if (data.active && !baseline.active) clearFreeTask();
  baseline = { ...baseline, ...data };
  if (pinnedRun && selectedRunId && data.batch?.id !== selectedRunId) baseline.batch = old;
  else {
    const id = data.batch?.id || null;
    if (id !== selectedRunId) {
      selectedAttemptId = null;
      pinnedAttempt = false;
      selectedCallId = null;
      selectedFrame = null;
      if (view === 'baseline') {
        state = { status: 'idle', model_calls: [], history: [], elements: [] };
        render();
      }
    }
    selectedRunId = id;
  }
}
function chooseBaselineAttempt() {
  const attempts = baseline.batch?.attempts || [];
  if (pinnedAttempt && attempts.some(a => a.id === selectedAttemptId)) return selectedAttemptId;
  const recorded = attempts.filter(a => !['queued', 'pending'].includes(a.status)
    && (a.status !== 'cancelled' || a.started_at || decisionCount(a) || a.text_calls || a.actions || a.evaluation));
  return (attempts.find(a => ['running', 'preparing'].includes(a.status)) || recorded.at(-1))?.id || null;
}
async function loadBaselineAttempt(id, epoch = baselineEpoch) {
  if (!id || !selectedRunId) return;
  const run = selectedRunId;
  // 自动跟随下一任务时也要立即清空上一条指令，而不只处理手动切换。
  if (selectedAttemptId !== id) promptRecordKey = null;
  renderOriginalPrompt();
  const snapshot = await baselineFetch(`/attempt?${new URLSearchParams({ run_id: run, attempt_id: id })}`);
  if (epoch !== baselineEpoch || view !== 'baseline' || selectedRunId !== run) return;
  if (selectedAttemptId !== id) {
      selectedCallId = null;
    selectedFrame = null;
  }
  selectedAttemptId = id;
  state = snapshot;
  promptRecordKey = `${run}/${id}/${epoch}`;
  render();
  renderBaseline();
}
async function refreshBaseline() {
  if (baselinePolling || baselineBusy) return;
  baselinePolling = true;
  const epoch = baselineEpoch;
  try {
    const data = await baselineFetch('');
    if (epoch !== baselineEpoch) return;
    acceptBaseline(data);
    if (pinnedRun && selectedRunId && data.batch?.id !== selectedRunId) {
      const batch = await baselineFetch(`/run?${new URLSearchParams({ run_id: selectedRunId })}`);
      if (epoch !== baselineEpoch) return;
      baseline.batch = batch;
    }
    renderBaseline();
    if (view === 'baseline') await loadBaselineAttempt(chooseBaselineAttempt(), epoch);
  } catch (error) {
    if (epoch === baselineEpoch && view === 'baseline') showBaselineError(error);
  } finally {
    baselinePolling = false;
  }
}
function showBaselineError(error) {
  $('baseline-error').textContent = error.message;
  $('baseline-error').hidden = false;
}
async function baselineAction(name, body = {}) {
  if (baselineBusy || busy) return;
  baselineBusy = true;
  const epoch = ++baselineEpoch;
  $('baseline-error').hidden = true;
  renderBaseline();
  try {
    const data = await baselineFetch(`/${name}`, body);
    if (name === 'start') { pinnedRun = false; pinnedAttempt = false; clearFreeTask(); consumeModelProfile(); }
    if (name === 'delete') {
      pinnedRun = false; pinnedAttempt = false;
      selectedRunId = null; selectedAttemptId = null; selectedFrame = null; selectedCallId = null;
      if (view === 'baseline') state = { status: 'idle', page: null, model_calls: [], history: [], elements: [] };
    }
    acceptBaseline(data);
    if (name === 'rename' && data.renamed_batch?.id === selectedRunId) baseline.batch = data.renamed_batch;
    if (name === 'delete' && view === 'baseline') render();
    renderBaseline();
    // 打开的是所选批次的实际目录，不依赖浏览器下载位置。
    if (data.opened_path) $('baseline-path').textContent = data.opened_path;
    if (view === 'baseline') await loadBaselineAttempt(chooseBaselineAttempt(), epoch);
  } catch (error) {
    showBaselineError(error);
  } finally {
    baselineBusy = false;
    renderBaseline();
  }
}
function startBaseline(taskIds) {
  if (baseline.active || busy || baselineBusy) return;
  const repeats = Number($('baseline-repeats').value);
  if (!Number.isInteger(repeats) || repeats < 1 || repeats > 10) {
    showBaselineError(Error('重复次数必须为 1–10 的整数。'));
    return;
  }
  baselineAction('start', { repeats, system1_provider: browserUseProvider, execution_mode: executionChoice, screenshot_policy: screenshotChoice, model_profile: modelProfileChoice, ...(taskIds ? { task_ids: taskIds } : {}) });
}
async function selectView(next) {
  selectedFrame = null;
  if (busy) return;
  view = next;
  try { window.localStorage?.setItem('jev-task-view', next); } catch { /* Storage may be disabled. */ }
  baselineEpoch++;
  for (const name of ['free', 'baseline']) {
    const selected = next === name;
    $(`${name}-tab`).setAttribute('aria-selected', String(selected));
    $(`${name}-tab`).tabIndex = selected ? 0 : -1;
    $(`${name}-panel`).hidden = !selected;
  }
  $('free-controls').hidden = next !== 'free';
  $('error').hidden = true;
  selectedCallId = null;
  if (next === 'free') {
    state = freeState;
    $('observation-toolbar').hidden = true;
    render();
  } else {
    state = { status: 'idle', model_calls: [], history: [], elements: [] };
    render();
    renderBaseline();
    await refreshBaseline();
  }
  controls();
}
$('free-tab').addEventListener('click', () => selectView('free'));
$('baseline-tab').addEventListener('click', () => selectView('baseline'));
for (const name of ['free', 'baseline']) $(`${name}-tab`).addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const next = event.key === 'Home' ? 'free' : event.key === 'End' ? 'baseline' : name === 'free' ? 'baseline' : 'free';
  $(`${next}-tab`).focus();
  selectView(next);
});
$('baseline-run-all').addEventListener('click', () => {
  const ids = baselineGroupTasks().map(task => task.id);
  if (ids.length) startBaseline(ids);
});
for (const group of Object.keys(baselineGroups)) {
  $(`baseline-group-${group}`).addEventListener('click', () => selectBaselineGroup(group));
}
$('baseline-case-search').addEventListener('input', () => {
  baselineSearch = $('baseline-case-search').value;
  renderBaseline();
});
$('baseline-stop').addEventListener('click', () => baselineAction('stop'));
$('baseline-open-folder').addEventListener('click', () => baselineAction('open-folder', { run_id: selectedRunId }));
$('baseline-rename').addEventListener('click', () => {
  if ($('baseline-rename').disabled) return;
  const batch = baseline.batch;
  if (!batch) return;
  const name = window.prompt(`重命名批次 ${batch.id}（最多 80 字）`, batch.name || batch.id);
  if (name === null) return;
  if (!name.trim() || name.trim().length > 80) {
    showBaselineError(Error('批次名称必须为 1–80 个字符。'));
    return;
  }
  baselineAction('rename', { run_id: batch.id, name: name.trim() });
});
$('baseline-delete').addEventListener('click', () => {
  if ($('baseline-delete').disabled) return;
  const batch = baseline.batch;
  if (!batch || !window.confirm(`永久删除批次 ${batch.id}？\n${batch.path || ''}\n所有 trace、截图和评估文件都会从磁盘删除，无法撤销。`)) return;
  baselineAction('delete', { run_id: batch.id });
});
$('baseline-tasks').addEventListener('click', event => {
  const button = event.target.closest('[data-run-task]');
  if (button && !button.disabled) startBaseline([button.dataset.runTask]);
});
$('baseline-attempts').addEventListener('click', event => {
  const button = event.target.closest('[data-attempt-id]');
  if (!button || button.disabled) return;
  pinnedAttempt = true;
  const epoch = ++baselineEpoch;
  selectedAttemptId = button.dataset.attemptId;
  selectedCallId = null;
  selectedFrame = null;
  renderOriginalPrompt();
  renderBaseline();
  loadBaselineAttempt(button.dataset.attemptId, epoch).catch(showBaselineError);
});
$('baseline-runs').addEventListener('change', async () => {
  const id = $('baseline-runs').value;
  if (!id) return;
  const epoch = ++baselineEpoch;
  // 请求历史批次期间隐藏旧指令，失败或过期响应也不能将其重新显示。
  pinnedRun = true;
  pinnedAttempt = false;
  selectedRunId = id;
  selectedAttemptId = null;
  selectedCallId = null;
  selectedFrame = null;
  baseline.batch = null;
  state = { status: 'idle', model_calls: [], history: [], elements: [] };
  render();
  try {
    const batch = await baselineFetch(`/run?${new URLSearchParams({ run_id: id })}`);
    if (epoch !== baselineEpoch) return;
    baseline.batch = batch;
    render();
    renderBaseline();
    await loadBaselineAttempt(chooseBaselineAttempt(), epoch);
  } catch (error) { showBaselineError(error); }
});
$('observation-step').addEventListener('change', () => {
  selectedFrame = $('observation-step').value;
  selectedCallId = null;
  render();
  $("model-output-panel").open = true;
});
$('research-latest').addEventListener('click', () => {
  selectedCallId = null; selectedFrame = null;
  render();
});
$('research-sources').addEventListener('click', event => {
  const target = event.target.closest('[data-source-observation]');
  if (!target) return;
  const index = Number(target.dataset.sourceObservation);
  if (!Number.isInteger(index) || !state.observations?.[index]) return;
  selectedFrame = String(index); selectedCallId = null; render();
});
$('report-copy').addEventListener('click', async () => {
  const markdown = taskResult()?.text || taskResult()?.markdown;
  if (!markdown) return;
  try { await navigator.clipboard.writeText(markdown); }
  catch { $('research-status').textContent = '复制失败，请选择任务输出文字复制。'; }
});
$('report-download').addEventListener('click', () => {
  const markdown = taskResult()?.text || taskResult()?.markdown;
  if (!markdown) return;
  const url = URL.createObjectURL(new Blob([markdown], {type:'text/markdown;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = 'result.md'; link.click();
  URL.revokeObjectURL(url);
});
$("history").addEventListener("click", (event) => {
  const row = event.target.closest("[data-call-id]");
  if (!row) return;
  const call = executionCalls().find(c => c.id === Number(row.dataset.callId));
  if (!call) return;
  selectedCallId = call.id;
  if (view === 'baseline' || Number.isInteger(call?.observation_index)) {
    selectedFrame = Number.isInteger(call?.observation_index) ? String(call.observation_index) : '-1';
    render();
  } else renderModelCalls();
  // 即使旧记录没有观察关联，仍切换其保存的计划版本。
  renderPlanning();
  $("model-output-panel").open = true;
});
$("task-form").addEventListener("submit", (event) => {
  event.preventDefault();
  automatic = false;
  perform(
    () =>
      call("reset", {
        scenario: $("scenario").value,
        system1_provider: browserUseProvider,
        execution_mode: executionChoice,
        screenshot_policy: screenshotChoice,
        model_profile: modelProfileChoice,
        url: $("website-url").value.trim(),
        goal: $("goal").value,
      }),
    "Opening a fresh browser…",
  );
});
$("scenario").addEventListener("change", () => {
  const scenario = $("scenario").value;
  if (websites[scenario]) {
    $("website-url").value = websites[scenario];
    $("goal").value = goals[scenario];
  } else {
    $("website-url").focus();
  }
});
$("website-url").addEventListener("input", () => {
  const url = $("website-url").value.trim();
  $("scenario").value = Object.keys(websites).find(key => websites[key] === url) || "custom";
});
$("choose").addEventListener("click", () =>
  perform(() => call("predict"), `${providerLabel(state?.system1_provider || browserUseProvider)} is comparing the actions…`),
);
$("execute").addEventListener("click", () =>
  perform(
    () => call("act", { fingerprint: state.page.fingerprint }),
    "Executing the choice…",
  ),
);
$("auto").addEventListener("click", () =>
  perform(async () => {
    automatic = true;
    controls();
    // 云端交接不是 System1 轮次，预算由后端累计检查；暂停在请求边界生效。
    while (automatic && !['done', 'blocked'].includes(state.status)) {
      $("status").textContent = "Running…";
      if ($("pace").checked) {
        await call("predict");
        await new Promise(resolve => setTimeout(resolve, 450));
        if (!automatic) break;
        // 初始规划或复核只更新计划，不会产生可执行 decision。
        if (state.decision) await call("act", {fingerprint: state.page.fingerprint});
      } else {
        await call("tick");
      }
      if (["done", "blocked"].includes(state.status)) break;
    }
    automatic = false;
  }, "Running the browser…"),
);
$("stop").addEventListener("click", () => {
  automatic = false;
  // 独立请求通知后端；云端返回后不得继续目标调用或浏览器变更。
  fetch('/api/pause', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Demo-Token': token }, body: '{}' })
    .catch(() => { $('status').textContent = '暂停请求未送达；当前请求结束后停止'; });
  $("status").textContent = "Pausing after the current request…";
  controls();
});
function updateOverlayVisibility() {
  $('targets').hidden = !$('overlays').checked || $('screenshot').hidden || !$('targets').innerHTML;
}
$("overlays").addEventListener("change", updateOverlayVisibility);
$("choices").addEventListener("pointerover", (event) => {
  const id = event.target.closest("[data-action]")?.dataset.action;
  hoveredElementIndex = id || null;
  document
    .querySelectorAll(".target")
    .forEach((t) =>
      t.classList.toggle(
        "selected",
        t.dataset.action === id || t.dataset.action === displayedTargetIndex,
      ),
    );
});
$("choices").addEventListener("pointerleave", () => {
  hoveredElementIndex = null;
  document
    .querySelectorAll(".target")
    .forEach((t) =>
      t.classList.toggle(
        "selected",
        t.dataset.action === displayedTargetIndex,
      ),
    );
});
$("download").addEventListener("click", () => {
  const { page, ...rest } = state;
  const blob = new Blob(
    [
      JSON.stringify(
        { ...rest, page: { ...page, screenshot: undefined } },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "typesafe-browser-trace.json";
  a.click();
  URL.revokeObjectURL(url);
});
$('model-profile').addEventListener('change', () => {
  const choice = $('model-profile').value;
  modelProfileChoice = choice === 'final_report' && finalReportConfigured()
    ? 'final_report' : 'debug';
  renderModelProfile();
});
$('execution-mode').value = executionChoice;
fetch("/api/state")
  .then((r) => r.json())
  .then((s) => {
    freeState = s;
    modelProfiles = s.model_profiles || [];
    renderModelProfile();
    if (view === 'free') { state = s; render(); }
  })
  .catch(() => {
    $("status").textContent = "Cannot reach local demo server";
    $("laya-address").textContent = "Unavailable";
    $("text-model-address").textContent = "Unavailable";
  });
let storedView;
try { storedView = window.localStorage?.getItem('jev-task-view'); } catch { /* Storage may be disabled. */ }
if (window.location.hash === '#baseline' || (!window.location.hash && storedView === 'baseline')) selectView('baseline');
else refreshBaseline();
setInterval(refreshBaseline, 1000);
