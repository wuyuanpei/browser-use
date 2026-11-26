(() => {
  const rendered = e => !!e && !e.closest('[hidden],[aria-hidden="true"],[inert]') &&
    e.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}) &&
    e.getBoundingClientRect().width > 0 && e.getBoundingClientRect().height > 0;
  const onscreen = e => {
    if (!rendered(e)) return false;
    const r = e.getBoundingClientRect();
    return r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
  };
  const text = e => rendered(e) ? e.innerText.trim() : '';
  const read = selector => text(document.querySelector(selector));
  const cache = window.__benchmarkObserver;
  const node = e => cache?.ids.get(e) ?? null;
  const truncation = [];
  const bounded = (value, size, field) => {
    if (value.length > size) truncation.push({field, original_characters: value.length, retained_characters: size});
    return value.slice(0, size);
  };
  const cleanLabel = value => bounded(value.replace(/\p{Co}/gu, ' ').replace(/\s+/g, ' ').trim(),
    200, 'form_control_label');
  // 独立验收读取全页当前标签，不把可变的 input name 当作选项含义或稳定身份。
  // 只沿显式标签关系读取渲染文本；不从相邻问题、隐藏内容或输入值猜测标签。
  const labelText = e => {
    if (!rendered(e) || e.matches('input,textarea,select,button,script,style,svg')) return '';
    return [...e.childNodes].map(n => n.nodeType === 3 ? n.textContent :
      n.nodeType === 1 ? labelText(n) : '').join(' ');
  };
  const explicitLabel = e => {
    const referenced = (e.getAttribute('aria-labelledby') || '').split(/\s+/)
      .map(id => labelText(document.getElementById(id))).join(' ');
    const sources = [['aria-labelledby', referenced], ['aria-label', e.getAttribute('aria-label') || ''],
      ['native_labels', [...(e.labels || [])].map(labelText).join(' ')],
      ['title', e.getAttribute('title') || ''], ['placeholder', e.getAttribute('placeholder') || '']];
    const match = sources.map(([source, value]) => [source, cleanLabel(value)]).find(([, value]) => value);
    return match ? {label: match[1], label_source: match[0]} : null;
  };
  const formDescription = e => {
    const description = explicitLabel(e);
    if (!description) return null;
    const group = e.closest('fieldset,[role="group"],[role="radiogroup"]');
    const legend = group?.matches('fieldset') ? [...group.children].find(c => c.matches('legend')) : null;
    const groupLabel = group && (explicitLabel(group)?.label || cleanLabel(labelText(legend)));
    return {...description, ...(groupLabel ? {context: {group_label: groupLabel}} : {})};
  };
  // 全页原生表单状态仅供独立验收；不会进入执行模型的候选或观察。
  const formControls = [...document.querySelectorAll('input,textarea,select')]
    .filter(e => !['password','hidden','file'].includes(e.type) && (rendered(e) ||
      (['checkbox','radio'].includes(e.type) && !e.closest('[hidden],[aria-hidden="true"],[inert]') &&
        [...e.labels].some(label => label.control === e && rendered(label)))))
    .map(e => {
      const description = formDescription(e);
      return {node: node(e), tag: e.tagName, type: e.type, name: e.name, value: e.value,
        form_owner: e.form ? [...document.forms].indexOf(e.form) : null,
        checked: ['checkbox','radio'].includes(e.type) ? e.checked : null,
        ...(description ? {current_description: description} : {}),
        ...(e.tagName === 'SELECT' ? {selected_options: [...e.selectedOptions].map(o => ({value:o.value,label:o.label}))} : {})};
    });
  const richText = [...document.querySelectorAll('[contenteditable="true"]')]
    .filter(e => rendered(e) && !e.parentElement?.isContentEditable).slice(0, 8).map(editor => {
      const format = new Map();
      const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT);
      let current;
      for (let i = 0; i < 250 && (current = walker.nextNode()); i++) {
        const parent = current.parentElement;
        if (!current.textContent.trim() || !rendered(parent)) continue;
        const style = getComputedStyle(parent);
        const bold = Number(style.fontWeight) >= 600, italic = /^(italic|oblique)/.test(style.fontStyle);
        if (!bold && !italic) continue;
        const key = [bold, italic].join(',');
        if (!format.has(key)) format.set(key, {bold, italic, runs: []});
        const runs = format.get(key).runs;
        if (runs.length < 20) runs.push(bounded(current.textContent, 300, 'rich_text.formatted_text'));
      }
      return {node: node(editor), label: explicitLabel(editor)?.label || null,
        text: bounded(text(editor), 5000, 'rich_text.text'), format: [...format.values()],
        unordered_lists: [...editor.querySelectorAll('ul')].filter(rendered).slice(0, 12)
          .map(list => [...list.children].filter(item => item.matches('li') && rendered(item)).slice(0, 30)
            .map(item => bounded(text(item), 300, 'rich_text.list_item')))};
    });
  const article = document.querySelector('.article-content');
  const wiki = document.querySelector('#mw-content-text .mw-parser-output');
  const stay = document.querySelector('.detail-copy');
  const results = [...document.querySelectorAll('#content_left .result, #content_left .result-op, #content_left .c-container')]
    .filter(e => onscreen(e) && e.querySelector('h3 a'));
  const finish = document.querySelector('#finish h4');
  const visible = page?.text || '';
  const dialogs = [...document.querySelectorAll('[role="dialog"],[aria-modal="true"]')].filter(onscreen);
  const challenge = /请完成安全验证|请拖动滑块|访问过于频繁|异常访问|verify you are human|checking your browser|access denied/i;
  let environmentBlock = '';
  if (location.protocol === 'chrome-error:' || /ERR_(NAME_NOT_RESOLVED|CONNECTION|TIMED_OUT|CERT)/.test(visible))
    environmentBlock = 'Browser network error';
  else if (challenge.test(visible)) environmentBlock = 'Visible access or CAPTCHA challenge';
  else if (dialogs.some(e => /登录|log\s*in|sign\s*in/i.test(text(e)) &&
      e.querySelector('input[type="password"],input[autocomplete="one-time-code"]')))
    environmentBlock = 'Login required';
  let travel = null;
  if (location.pathname === '/fixture.html' && typeof searched === 'boolean' && typeof query === 'string') {
    travel = {searched, query, category, free};
  }
  const ctrip = /(^|\.)ctrip\.com$/.test(location.hostname);
  const rowSelector = '.flight-item,.flight-box,.flight-card,[class*="flight-item"],[class*="flightItem"],[class*="flight-card"]';
  const candidates = ctrip ? [...document.querySelectorAll(rowSelector)].filter(e => onscreen(e) &&
    /\d{1,2}:\d{2}/.test(text(e)) && /[¥￥]|\d+起/.test(text(e)) && /[A-Z0-9]{2}\s?\d{3,4}/.test(text(e))) : [];
  // 外层 flight-item 和内层 flight-box 会同时命中，保留最内层的完整航班行。
  const rows = candidates.filter(e => !candidates.some(other => other !== e && e.contains(other)));
  const selectedDirect = ctrip && [...document.querySelectorAll('input:checked,[aria-checked="true"],[aria-selected="true"]')]
    .some(e => {
      const label = text(e.closest('label') || e.parentElement).replace(/\p{Co}|\s/gu, '');
      return rendered(e) && /仅直飞|只看直飞|^直飞$/.test(label) && !/经停|中转|转机/.test(label);
    });
  const flightFact = e => {
    const value = text(e), base = {text: bounded(value, 1500, 'flight_row.text')};
    if (/经停|中转|转机/.test(value))
      return {...base, direct: false, direct_evidence: {source: 'connection_text'}};
    // 当前国内列表用“空的经停信息槽 + 单航段起抵结构”表示直飞，不显示“直飞”字样。
    // 只接受完整的已知结构；不能把任意一行没有“中转”文字当成直飞。
    const details = [...e.querySelectorAll('.flight-detail')].filter(rendered);
    if (details.length > 1)
      return {...base, direct: false, direct_evidence: {source: 'multiple_legs', legs: details.length}};
    if (/直飞/.test(value)) return {...base, direct: true, direct_evidence: {source: 'direct_text'}};
    const detail = details[0], planes = [...e.querySelectorAll('.plane-No')].filter(rendered);
    const departures = detail ? [...detail.querySelectorAll('.depart-box')].filter(rendered) : [];
    const arrivals = detail ? [...detail.querySelectorAll('.arrive-box')].filter(rendered) : [];
    const transfers = detail ? [...detail.querySelectorAll('[id^="transfer-text-"]')] : [];
    const groups = detail ? [...detail.querySelectorAll('.transfer-info-group')] : [];
    const departure = departures[0], arrival = arrivals[0];
    const routeEnd = node => ({time: text(node?.querySelector('.time')), airport: text(node?.querySelector('.airport .name'))});
    const from = routeEnd(departure), to = routeEnd(arrival);
    const emptyTransfer = node => ![node.textContent, node.getAttribute('title'), node.getAttribute('aria-label')]
      .some(value => value?.trim());
    if (details.length === 1 && planes.length === 1 && /[A-Z0-9]{2}\s?\d{3,4}/.test(text(planes[0])) &&
        departures.length === 1 && arrivals.length === 1 && /^\d{1,2}:\d{2}/.test(from.time) &&
        /^\d{1,2}:\d{2}/.test(to.time) && from.airport && to.airport &&
        detail.querySelectorAll('.arrow-oneway').length === 1 && transfers.length === 1 && groups.length === 1 &&
        emptyTransfer(transfers[0]) && emptyTransfer(groups[0])) {
      return {...base, direct: true, direct_evidence: {source: 'single_leg_dom',
        flight: text(planes[0]), departure: from, arrival: to, transfer_text: '', transfer_info: ''}};
    }
    return {...base, direct: null, direct_evidence: {source: 'unrecognized_dom'}};
  };
  const formValues = ctrip ? [...document.querySelectorAll('input:not([type="password"]),select')]
    .filter(onscreen).map(e => e.value).join('\n') : '';
  // Separate read-only detail evidence. Unknown page structures never pass from model claims alone.
  const xhs = /^(www\.)?xiaohongshu\.com$/.test(location.hostname);
  const detail = xhs ? document.querySelector('.note-detail-mask .note-container,.note-detail,.note-container') : null;
  const visibleText = root => {
    if (!root) return '';
    const parts = [], walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), range = document.createRange();
    let n;
    while ((n = walker.nextNode())) {
      const parent = n.parentElement;
      if (!n.textContent.trim() || !rendered(parent) || parent.closest('script,style,noscript')) continue;
      range.selectNodeContents(n);
      const rect = range.getBoundingClientRect();
      let l=Math.max(0,rect.left), r=Math.min(innerWidth,rect.right);
      let t=Math.max(0,rect.top), b=Math.min(innerHeight,rect.bottom);
      for (let p=parent.parentElement;p && p!==document.body;p=p.parentElement) {
        const style=getComputedStyle(p), box=p.getBoundingClientRect();
        if (/auto|scroll|hidden|clip/.test(style.overflowX)) {l=Math.max(l,box.left);r=Math.min(r,box.right);}
        if (/auto|scroll|hidden|clip/.test(style.overflowY)) {t=Math.max(t,box.top);b=Math.min(b,box.bottom);}
      }
      if (r>l && b>t && parent.contains(document.elementFromPoint((l+r)/2,(t+b)/2))) parts.push(n.textContent.trim());
    }
    return bounded(parts.join('\n'), 12000, 'research_note.text');
  };
  const noteText = visibleText(detail);
  const field = selector => visibleText(detail?.querySelector(selector));
  const note = detail && noteText ? {url:location.href, observed_at:new Date().toISOString(),
    title:field('#detail-title,.title'), author:field('.author-container .username,.author .name'),
    published:field('.date,.publish-date'), text:noteText,
    links:[...detail.querySelectorAll('a[href]')].filter(onscreen).map(e=>e.href)} : null;
  // 仅供独立验收。新版页面的 wrapper 和 input 重复使用 ID，必须限定到 input。
  let scholar = null;
  if (location.hostname === 'xueshu.baidu.com') {
    const panel = document.querySelector('.advanced-search-wrap');
    const input = key => panel?.querySelector('input[id="advanced-search-' + key + '"]');
    const keys = ['all', 'precise', 'or', 'not', 'author', 'affs', 'publication', 'year-start', 'year-end'];
    const publication = input('publication')?.closest('.select-input')?.querySelector('.publish-select .value-text');
    const languages = panel ? [...panel.querySelectorAll('input[type="radio"]')]
      .filter(e => rendered(e.closest('label')) && e.checked) : [];
    scholar = {
      visible: rendered(panel),
      fields: Object.fromEntries(keys.map(key => {
        const e = input(key);
        return [key, rendered(e) ? e.value : null];
      })),
      publication_type: rendered(publication) ? text(publication) : null,
      language: languages.length === 1 ? text(languages[0].closest('label')) : null,
      submit_nodes: panel ? [...panel.querySelectorAll('button,input[type="submit"]')]
        .filter(e => rendered(e) && /^高级检索$/.test(text(e)))
        .map(node).filter(n => n !== null) : [],
    };
  }
  let law = null, questionnaire = null;
  if (location.hostname === 'flk.npc.gov.cn') {
    const items = [...document.querySelectorAll('.item')].filter(e => e.querySelector('.label'));
    const fields = Object.fromEntries(items.map(e => [
      e.querySelector('.label').textContent.replace(/[：:\s]/g, ''),
      {values: [...e.querySelectorAll('input.el-input__inner,input.el-range-input')]
        .map(i => i.value), checked: [...e.querySelectorAll('input[type="checkbox"]')].map(i => i.checked),
      tags: [...e.querySelectorAll('.el-cascader__tags .el-tag')].map(e => e.textContent.trim())}
    ]));
    const submit = [...document.querySelectorAll('button')].filter(e => rendered(e) &&
      e.textContent.replace(/\s/g, '') === '确定' && !e.closest('.el-picker-panel'));
    law = {form_visible: items.some(e => rendered(e) && /公布日期/.test(e.textContent)), fields,
      selections: [...document.querySelectorAll('.input-item-content .el-select__placeholder')].map(text).filter(Boolean),
      submit_nodes: submit.map(node).filter(n => n !== null)};
  }
  if (location.hostname === 'wj.qq.com') {
    const login = location.pathname === '/r/login.html' && /登录/.test(document.title + visible);
    if (login) environmentBlock = 'Tencent questionnaire requires login; questions are not visible';
    questionnaire = {login_required: login, controls: formControls,
      submit_nodes: [...document.querySelectorAll('button,[role="button"],a')]
        .filter(e => rendered(e) && /^提交/.test(text(e))).map(node).filter(n => n !== null)};
  }
  return {
    url: location.href, document_id: performance.timeOrigin, sampled_at: new Date().toISOString(),
    form_controls: formControls, environment_block: environmentBlock,
    article_visible: onscreen(article), article_heading: read('.article-content h1'), article_text: bounded(text(article), 5000, 'article_text'),
    wiki_visible: onscreen(wiki), wiki_heading: read('#firstHeading'), wiki_text: bounded(text(wiki), 5000, 'wiki_text'),
    baidu_results: results.length,
    baidu_result_nodes: results.flatMap(e => [...e.querySelectorAll('h3 a')].map(node)).filter(n => n !== null),
    pagination: read('.pager .current'), books_visible: [...document.querySelectorAll('article.product_pod')].filter(onscreen).length,
    checkboxes: [...document.querySelectorAll('#checkboxes input[type="checkbox"]')].filter(rendered)
      .map(e => ({checked: e.checked, node: node(e)})),
    travel, stay_visible: onscreen(stay), stay_heading: read('.detail-copy h1'), stay_text: bounded(text(stay), 2500, 'stay_text'),
    loading_finish_visible: onscreen(finish), loading_finish_text: text(finish),
    ctrip_controls: ctrip ? (bounded(visible, 20000, 'ctrip_controls') + '\n' + formValues) : '',
    ctrip_direct_selected: selectedDirect,
    flight_rows: rows.map(flightFact),
    baidu_scholar: scholar, law_form: law, questionnaire,
    research_note: note, ...(richText.length ? {rich_text: richText} : {}), truncation,
  };
})()
