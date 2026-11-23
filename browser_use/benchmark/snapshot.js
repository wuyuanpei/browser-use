(function readState() {
  if (!document.body) return null;
  const cache = window.__benchmarkObserver ||= { ids: new WeakMap(), nodes: new Map(), next: 1 };
  cache.readState = readState;
  cache.headingVersions ||= new WeakMap();
  cache.fields ||= new WeakSet();
  const identity = e => {
    if (!cache.ids.has(e)) cache.ids.set(e, cache.next++);
    const id = cache.ids.get(e); cache.nodes.set(id, e); return id;
  };
  for (const [id, e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);
  const safe = e => !['password', 'file', 'hidden'].includes(e.type);
  const rendered = e => e.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true });
  const visible = e => !e.closest('[aria-hidden="true"],[inert]') && rendered(e);
  const nativeChoice = e => e?.tagName === 'INPUT' && ['checkbox', 'radio'].includes(e.type);
  const nativeText = e => e?.tagName === 'TEXTAREA' ||
    (e?.tagName === 'INPUT' && ['text', 'search', 'email', 'url', 'tel', 'number'].includes(e.type));
  // 语义身份始终属于控件；关联 label 只提供鼠标命中区域。
  // 采集和执行共享此判断，避免把 CSS 裁剪的原生控件当作可直点目标。
  cache.clickPoint = e => {
    if (!e?.isConnected || e.closest('[hidden],[aria-hidden="true"],[inert],:disabled,[aria-disabled="true"]')) return null;
    const native = e.matches(selector), custom = !native && customClick(e), owner = e.getBoundingClientRect();
    const ownerPress = !native && ownPress(e);
    function* surfaces() {
      yield e;
      if (nativeChoice(e)) yield* e.labels || [];
      // Only borrow exposed descendants after the existing centers fail; identity stays on e.
      if (custom) for (const child of [...e.querySelectorAll('*')].slice(0, 16)) {
        if ([...child.childNodes].some(node => node.nodeType === 3 && node.textContent.trim()) ||
            (child.matches('img,svg') && !child.children.length)) yield child;
      }
    }
    let decorativePoint = null;
    for (const surface of surfaces()) {
      if (!visible(surface) || surface.closest(':disabled,[aria-disabled="true"]')) continue;
      const r = surface.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) continue;
      if (surface !== e && surface.control !== e && (!custom || !e.contains(surface) ||
          r.left < owner.left || r.right > owner.right || r.top < owner.top || r.bottom > owner.bottom)) continue;
      const points = [[0.5, 0.5]];
      for (const [fx, fy] of points) {
        const x = r.x + r.width * fx, y = r.y + r.height * fy;
        const hit = document.elementFromPoint(x, y);
        const hitLabel = hit?.closest('label'), labelledControl = hitLabel?.control;
        let hitSurface = surface;
        if (!surface.contains(hit)) {
          // Overlapping native labels can focus their field at the field's own center.
          if (surface !== e || labelledControl !== e || !nativeText(e) || !editable(e) ||
              !visible(hitLabel) || hitLabel.closest('[hidden],:disabled,[aria-disabled="true"]') ||
              hitLabel.matches(selector) || ownPress(hitLabel)) continue;
          hitSurface = hitLabel;
        }
        // Keep independent controls inside labels and handler-backed owners.
        if (labelledControl && labelledControl !== e && surface.contains(labelledControl) && surface !== hitLabel)
          continue;
        const boundary = ownerPress ? e : hitSurface;
        let nested = hit;
        while (nested && nested !== boundary && nested !== e && !nested.matches(selector) && !ownPress(nested))
          nested = nested.parentElement;
        if (nested !== boundary && nested !== e) {
          if (nested && points.length === 1 && surface === e && e.matches('a[href]') && !nested.matches(selector))
            points.push([0.05, 0.5], [0.95, 0.5], [0.5, 0.05], [0.5, 0.95]);
          continue;
        }
        const point = { x, y, surface: hitSurface, rect: { x: r.x, y: r.y, w: r.width, h: r.height } };
        if (!custom) return point;
        if (!controlBranch(e, surface)) continue;
        const ownerCenter = surface === e && ownerPress;
        if (controlBranch(e, hit, ownerCenter)) return point;
        // Keep existing click surfaces ahead of the owner's center on a decorative SVG.
        if (surface !== e || !(hit instanceof SVGElement)) continue;
        const svg = hit.closest('svg[aria-hidden="true"]');
        if (svg && e.contains(svg) && controlBranch(e, hit, ownerCenter, svg)) decorativePoint = point;
      }
    }
    return decorativePoint;
  };
  // Text and links must survive viewport clipping, overflow clipping, and occlusion.
  const clipped = (e, rect) => {
    let left = Math.max(0, rect.left), right = Math.min(innerWidth, rect.right);
    let top = Math.max(0, rect.top), bottom = Math.min(innerHeight, rect.bottom);
    for (let p = e.parentElement; p && p !== document.body; p = p.parentElement) {
      const style = getComputedStyle(p), r = p.getBoundingClientRect();
      if (/(auto|scroll|hidden|clip)/.test(style.overflowX)) { left = Math.max(left, r.left); right = Math.min(right, r.right); }
      if (/(auto|scroll|hidden|clip)/.test(style.overflowY)) { top = Math.max(top, r.top); bottom = Math.min(bottom, r.bottom); }
    }
    return right > left && bottom > top ? {left, right, top, bottom} : null;
  };
  const exposed = (e, rect) => {
    const r = clipped(e, rect);
    return r && e.contains(document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2));
  };
  const scrollable = e => e && e !== document.body && e !== document.documentElement && visible(e) &&
    /auto|scroll/.test(getComputedStyle(e).overflowY) && e.scrollHeight > e.clientHeight + 2;
  cache.scrollPoint = e => {
    if (!e?.isConnected || !visible(e)) return null;
    const r = clipped(e, e.getBoundingClientRect());
    if (!r) return null;
    for (const [fx, fy] of [[0.5, 0.5], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]) {
      const x = r.left + (r.right - r.left) * fx, y = r.top + (r.bottom - r.top) * fy;
      let hit = document.elementFromPoint(x, y);
      if (!e.contains(hit) || hit.matches('iframe') || hit.closest('dialog:modal') !== e.closest('dialog:modal')) continue;
      while (hit && hit !== e && hit !== document.documentElement && !scrollable(hit)) hit = hit.parentElement;
      if (hit === e) return {x, y};
    }
    return null;
  };
  const viewport = () => [performance.timeOrigin, location.href, scrollX, scrollY, innerWidth, innerHeight];
  const regions = [...document.body.querySelectorAll('*')].filter(e => scrollable(e) && cache.scrollPoint(e)).slice(0, 12);
  // 图标字体的私用区字符没有通用文本含义，还会挤占候选的 token 前缀预算。
  // 只清理控件名称；输入值及用于独立验收的原始正文保持原样。
  const cleanName = text => text.replace(/\p{Co}/gu, ' ').replace(/\s+/g, ' ').trim();
  const textPrefix = (text, limit) => {
    // Keep the UTF-16 budget and end before a pair that crosses the cutoff.
    const last = text.charCodeAt(limit - 1), next = text.charCodeAt(limit);
    const splitsPair = last >= 0xd800 && last <= 0xdbff && next >= 0xdc00 && next <= 0xdfff;
    return text.slice(0, splitsPair ? limit - 1 : limit);
  };
  const explicitName = (e, seen = new Set([e]), textNodes) => {
    const referenced = (e.getAttribute('aria-labelledby') || '').split(/\s+/)
      .map(id => name(document.getElementById(id), seen, null, textNodes)).filter(Boolean).join(' ');
    return referenced || e.getAttribute('aria-label') ||
      [...(e.labels || [])].map(l => name(l, seen, null, textNodes)).filter(Boolean).join(' ') ||
      (['button', 'submit', 'reset'].includes(e.type) ? e.value : '') || e.getAttribute('alt') || '';
  };
  const name = (e, seen = new Set(), included = null, textNodes) => {
    if (!e || seen.has(e) || (included && !included(e))) return '';
    seen.add(e);
    return explicitName(e, seen, textNodes) || implicitName(e, seen, included, textNodes);
  };
  const implicitName = (e, seen, included, textNodes) => {
    const content = e.tagName === 'INPUT' ? '' : [...e.childNodes].map(node => {
      if (node.nodeType === 1 && node.getAttribute('aria-hidden') !== 'true')
        return name(node, seen, included, textNodes);
      if (node.nodeType !== 3 || (included && !included(node))) return '';
      textNodes?.push(node);
      return node.textContent;
    }).join(' ').trim();
    return content || e.getAttribute('title') || e.getAttribute('placeholder') || '';
  };
  const roles = ['button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem', 'menuitemradio',
    'option', 'gridcell', 'combobox', 'textbox', 'searchbox', 'spinbutton'];
  const selector = 'a[href],button,input,textarea,select,summary,[contenteditable="true"],' +
    roles.map(role => '[role="' + role + '"]').join(',');
  const pressEvents = ['click', 'mousedown', 'mouseup', 'pointerdown', 'pointerup'];
  // React 将事件委托给根节点，DOM onclick/onmousedown 不一定暴露处理器。
  // 只读取元素自身的事件属性，不调用网页回调，也不借用祖先的事件作为子节点的点击证据。
  const reactProps = e => {
    const key = Object.keys(e).find(k => k.startsWith('__reactProps$') || k.startsWith('__reactEventHandlers$'));
    return key ? e[key] : null;
  };
  const ownPress = e => {
    if (pressEvents.some(event => typeof e['on' + event] === 'function' || e.hasAttribute('on' + event))) return true;
    const props = reactProps(e);
    if (props && ['Click', 'MouseDown', 'MouseUp', 'PointerDown', 'PointerUp']
      .some(event => typeof props['on' + event] === 'function' || typeof props['on' + event + 'Capture'] === 'function'))
      return true;
    // Vue stores addEventListener invokers on the element, without assigning DOM on* properties.
    const invokerKey = Object.getOwnPropertySymbols(e).find(key => key.description === '_vei');
    const invokers = invokerKey ? e[invokerKey] : null;
    return invokers && Object.keys(invokers).some(event =>
      /^on(?:Click|Mousedown|Mouseup|Pointerdown|Pointerup)(?:Once|Passive|Capture)*$/.test(event) &&
      typeof invokers[event] === 'function');
  };
  const customDisabled = e => {
    for (let parent = e; parent && parent !== document.body; parent = parent.parentElement) {
      if ([...parent.classList].some(c => /(?:^|[-_])disabled?$/i.test(c)) || reactProps(parent)?.disabled === true)
        return true;
    }
    return false;
  };
  const pointer = e => e && getComputedStyle(e).cursor === 'pointer';
  const independentControl = (owner, element, allowInnerPointer = false) =>
    element.matches(selector) || ownPress(element) ||
    (!allowInnerPointer && pointer(element) && !pointer(element.parentElement)) ||
    (element.matches('label') && element.control && element.control !== owner);
  // Borrowed surfaces and their implicit descriptions share independent-control boundaries.
  const controlBranch = (owner, child, ownerCenter = false, decorativeSvg = null) => {
    if (!owner.contains(child)) return false;
    for (let current = child; current && current !== owner; current = current.parentElement) {
      const isRenderedDecoration = decorativeSvg && current.closest('[aria-hidden="true"]') === decorativeSvg &&
        !current.closest('[hidden],[inert]') && rendered(current);
      if (independentControl(owner, current, ownerCenter) ||
          !(visible(current) || isRenderedDecoration) || customDisabled(current) ||
          current.closest(':disabled,[aria-disabled="true"]')) return false;
    }
    return true;
  };
  const customClick = e => {
    const label = e.closest('label'), control = label?.control;
    if (nativeChoice(control) && (visible(control) || control.matches(':disabled') || cache.clickPoint(control))) return false;
    const handler = ownPress(e);
    if (!handler && (!pointer(e) || pointer(e.parentElement))) return false;
    if (!handler && nativeText(control) && cache.clickPoint(control)?.surface === label) return false;
    if (customDisabled(e)) return false;
    // 不重复原生控件及其内部文字，也不把包含控件的布局容器提升为按钮。
    if (e.closest(selector)) return false;
    if (handler) return true;
    if ([...e.querySelectorAll('*')].some(child =>
      (child.matches(selector) || ownPress(child)) && visible(child))) return false;
    // React 等委托事件不会暴露在 onclick 上，使用指针区域的边界作为补充信号。
    // cursor 会被子节点继承，不能将同一个按钮的图标和文字再次编号。
    for (let parent = e.parentElement; parent && parent !== document.body; parent = parent.parentElement) {
      if (ownPress(parent)) return false;
    }
    return true;
  };
  const role = e => {
    if (!e.matches(selector)) return customClick(e) ? 'button' : null;
    const explicit = e.getAttribute('role');
    if (roles.includes(explicit)) return explicit;
    if (e.tagName === 'BUTTON' || e.tagName === 'SUMMARY') return 'button';
    if (e.tagName === 'A' && e.hasAttribute('href')) return 'link';
    if (e.tagName === 'SELECT') return 'combobox';
    if (e.tagName === 'TEXTAREA' || e.isContentEditable) return 'textbox';
    if (e.tagName === 'INPUT') {
      if (['checkbox', 'radio'].includes(e.type)) return e.type;
      if (['button', 'submit', 'reset', 'image'].includes(e.type)) return 'button';
      if (e.type === 'search') return 'searchbox';
      if (e.type === 'number') return 'spinbutton';
      if (['text', 'email', 'url', 'tel'].includes(e.type)) return 'textbox';
    }
    return null;
  };
  const editable = (e, rname = role(e)) => {
    const nativeField = ['INPUT', 'TEXTAREA'].includes(e.tagName);
    return !e.readOnly && e.getAttribute('aria-readonly') !== 'true' &&
      ((['textbox', 'searchbox', 'spinbutton'].includes(rname) && (nativeField || e.isContentEditable)) ||
        (rname === 'combobox' && nativeField));
  };
  const fieldSelector = 'input,textarea,select,[contenteditable="true"],' +
    ['textbox', 'searchbox', 'combobox', 'spinbutton'].map(r => '[role="' + r + '"]').join(',');
  const headingSelector = 'h1,h2,h3,h4,h5,h6,[role="heading"],legend';
  const sectionSelector = 'form,fieldset,section,article,dialog,[role="dialog"],[role="region"],[role="form"]';
  const nearbyRange = document.createRange();
  const shortText = (node, allowNumeric = false) => {
    const element = node.nodeType === 1 ? node : node.parentElement;
    if (!element || !visible(element)) return '';
    nearbyRange.selectNodeContents(node);
    const r = nearbyRange.getBoundingClientRect();
    if (!r.width || !r.height || r.bottom <= 0 || r.top >= innerHeight || r.right <= 0 || r.left >= innerWidth) return '';
    const text = cleanName(node.nodeType === 3 ? node.textContent : node.innerText || '');
    return text.length <= 80 && (allowNumeric ? /[\p{L}\p{N}]/u.test(text) : /\p{L}/u.test(text)) ? text : '';
  };
  const choiceOwner = e => {
    if (!nativeChoice(e)) return null;
    const ownerSelector = '[role="menuitem"],[role="option"],[role="treeitem"]';
    const owner = e.closest(ownerSelector);
    if (!owner || [...owner.querySelectorAll('input')].filter(c =>
      nativeChoice(c) && c.closest(ownerSelector) === owner).length !== 1) return null;
    const texts = [], walker = document.createTreeWalker(owner, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const parent = node.parentElement;
      if (parent.closest(ownerSelector) !== owner || parent.closest(selector) !== owner ||
          parent.closest('script,style,noscript,template')) continue;
      const text = shortText(node, true);
      const rect = nearbyRange.getBoundingClientRect(), r = text ? clipped(parent, rect) : null;
      const hit = r && document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
      const label = hit?.closest('label');
      // A label associated with this same choice can cover its owner's sibling text.
      const labelled = label?.control === e && owner.contains(label) &&
        [...(e.labels || [])].includes(label) && controlBranch(label, hit);
      if (text && (exposed(parent, rect) || labelled)) texts.push(text);
    }
    const label = cleanName(texts.join(' '));
    return label && label.length <= 80 ? {label, node_ref: identity(owner), role: owner.getAttribute('role'),
      source: 'semantic_owner_text'} : null;
  };
  const visibleFields = scope => [...scope.querySelectorAll(fieldSelector)].filter(visible);
  const nearbyLabel = e => {
    let branch = e;
    for (let depth = 0, scope = e.parentElement; scope && scope !== document.body && depth < 6;
      depth++, branch = scope, scope = scope.parentElement) {
      // 沿输入框所在分支向前找短标签，不跨过另一个字段、模块标题或显式表单边界。
      for (let sibling = branch.previousSibling; sibling; sibling = sibling.previousSibling) {
        if (sibling.nodeType === 1) {
          if (!visible(sibling)) continue;
          if (sibling.matches(headingSelector) || sibling.querySelector(headingSelector)) break;
          // A carousel/list of suggestions is interactive content, not this field's label.
          // Borrowing its text makes a stable field appear renamed on every rotation.
          if (!sibling.matches('label') && [sibling, ...sibling.querySelectorAll('*')]
            .some(node => node.matches(selector) || ownPress(node) ||
              (pointer(node) && !pointer(node.parentElement)))) return '';
          if (sibling.matches(fieldSelector) || visibleFields(sibling).length) return '';
          if (sibling.matches('label[for]') && sibling.control !== e) return '';
        }
        if (![1, 3].includes(sibling.nodeType)) continue;
        const text = shortText(sibling);
        if (text) return text;
      }
      if (scope.matches(sectionSelector) || visibleFields(scope).some(f => f !== e && !f.contains(e))) break;
    }
    return '';
  };
  const fieldSection = (e, visibleOnly = false) => {
    for (let depth = 0, scope = e.parentElement; scope && scope !== document.body && depth < 8;
      depth++, scope = scope.parentElement) {
      if (scope.matches(sectionSelector)) {
        const labelled = (scope.getAttribute('aria-labelledby') || '').split(/\s+/)
          .map(id => {
            const label = document.getElementById(id);
            return visibleOnly ? (label ? shortText(label) : '') : name(label);
          }).filter(Boolean).join(' ');
        const title = labelled || scope.getAttribute('aria-label');
        if (title && cleanName(title)) return textPrefix(cleanName(title), 80);
      }
      for (const heading of [...scope.querySelectorAll(headingSelector)].reverse()) {
        if (!(heading.compareDocumentPosition(e) & Node.DOCUMENT_POSITION_FOLLOWING)) continue;
        let branch = heading;
        while (branch.parentElement !== scope) branch = branch.parentElement;
        // 不借用旁边另一个表单/模块内部的标题。
        if (branch.contains(e) || visibleFields(branch).length) continue;
        const text = shortText(heading);
        if (text) return text;
      }
      if (scope.matches(sectionSelector)) break;
    }
    return '';
  };
  const choiceGroupSelector = 'fieldset,[role="group"],[role="radiogroup"],tr,[role="row"],form,[role="form"]';
  const cellSelector = 'td,th,[role="cell"],[role="gridcell"],[role="rowheader"],[role="columnheader"]';
  const rowSelector = 'tr,[role="row"]';
  const tableSelector = 'table,[role="table"],[role="grid"],[role="treegrid"]';
  const tableContext = e => {
    const cell = e.closest(cellSelector), row = cell?.closest(rowSelector), table = row?.closest(tableSelector);
    if (!table || cell.closest(tableSelector) !== table) return null;
    const rowCells = new Map();
    const ownCells = r => {
      if (!rowCells.has(r)) rowCells.set(r, [...r.querySelectorAll(cellSelector)].filter(c =>
        c.closest(rowSelector) === r && c.closest(tableSelector) === table));
      return rowCells.get(r);
    };
    const rows = [...table.querySelectorAll(rowSelector)].filter(r => r.closest(tableSelector) === table);
    const cells = ownCells(row);
    const headerText = nodes => {
      const texts = [...new Set(nodes.filter(c => visible(c) && exposed(c, c.getBoundingClientRect()))
        .map(c => shortText(c, true)).filter(Boolean))];
      return texts.length === 1 ? texts[0] : '';
    };
    const isRowHeader = c => c.matches('[role="rowheader"],th[scope="row"]');
    const isColumnHeader = c => c.matches('[role="columnheader"],th:not([scope]),th[scope="col"]');
    // 显式 headers 关系优先；不读取隐藏标题，也不跨嵌套表格借用标题。
    const referenced = (cell.getAttribute('headers') || '').split(/\s+/).map(id => document.getElementById(id))
      .filter(c => c && c.closest(tableSelector) === table);
    let rowLabel = headerText(referenced.filter(isRowHeader)) || headerText(cells.filter(isRowHeader));
    let columnLabel = headerText(referenced.filter(isColumnHeader));
    // 没有显式列关系时，只处理无合并单元格、列数一致且列号明确的规则网格。
    // 复杂跨度、缺失单元格或多个标题保留为未知，由原始 DOM 顺序提供补充证据。
    const regular = rows.every(r => ownCells(r).length === cells.length && ownCells(r).every(c =>
      Number(c.getAttribute('colspan') || c.getAttribute('aria-colspan') || 1) === 1 &&
      Number(c.getAttribute('rowspan') || c.getAttribute('aria-rowspan') || 1) === 1));
    if (!columnLabel && regular) {
      const explicitColumn = cell.getAttribute('aria-colindex');
      const column = cells.indexOf(cell);
      const headers = rows.filter(r => r !== row && (r.compareDocumentPosition(row) & Node.DOCUMENT_POSITION_FOLLOWING))
        .flatMap(ownCells).filter(c => isColumnHeader(c) &&
          (explicitColumn ? c.getAttribute('aria-colindex') === explicitColumn : ownCells(c.closest(rowSelector)).indexOf(c) === column));
      columnLabel = headerText(headers);
    }
    // 规则表格也常用普通首列单元格显示行名，未声明 scope=row。
    // 只有首列为静态文本且后面所有单元格均含控件时才采用该结构；不可见列标题仍为空。
    let rowSource = '';
    if (!rowLabel && regular && cells.indexOf(cell) > 0 &&
        !row.closest('thead') && !cells[0].matches(selector) && !cells[0].querySelector(selector) &&
        !cells[0].querySelector(tableSelector) && cells.slice(1).every(c => c.querySelector(selector))) {
      rowLabel = headerText([cells[0]]);
      if (rowLabel) rowSource = 'leading_static_cell';
    }
    // Native rows/cells describe a complete DOM grid only without spans, nested tables or virtual indices.
    const position = table.matches('table') && regular && rows.length && cells.length &&
      rows.includes(row) && cells.includes(cell) && !table.parentElement.closest(tableSelector) &&
      !table.querySelector(tableSelector) && !table.matches('[aria-rowcount],[aria-colcount]') &&
      rows.every(r => r.tagName === 'TR' && !r.hasAttribute('aria-rowindex') && ownCells(r).every(c =>
        c.matches('td,th') && c.parentElement === r && !c.matches('[aria-rowindex],[aria-colindex]')))
      ? {node_ref: identity(table), row_index: rows.indexOf(row) + 1, column_index: cells.indexOf(cell) + 1,
        row_count: rows.length, column_count: cells.length} : null;
    if (!rowLabel && !columnLabel && !position) return null;
    return {row_label: rowLabel, column_label: columnLabel,
      ...position,
      ...(rowSource ? {row_label_source: rowSource} : {}),
      source: table.matches('table') ? 'html_table' : 'aria_table'};
  };
  const choiceContext = (e, accessible) => {
    let group = e.closest(choiceGroupSelector);
    // Native radio grouping is defined by name and form owner, regardless of layout classes.
    // Its common DOM ancestor also distinguishes repeated option labels in separate rows.
    if (e.type === 'radio' && e.name) {
      const peers = [...document.querySelectorAll('input[type="radio"]')]
        .filter(other => other.name === e.name && other.form === e.form);
      if (peers.length > 1) {
        let common = e.parentElement;
        while (common && !peers.every(other => common.contains(other))) common = common.parentElement;
        if (common && common !== document.body && common !== document.documentElement &&
            (!group || group.contains(common))) group = common;
      }
    }
    if (!group) return null;
    const explicitLabel = scope => {
      const labelled = (scope.getAttribute('aria-labelledby') || '').split(/\s+/).map(id => {
        const label = document.getElementById(id);
        return label ? shortText(label) : '';
      }).filter(Boolean).join(' ');
      const aria = cleanName(scope.getAttribute('aria-label') || '');
      if (labelled || aria) return textPrefix(labelled || aria, 80);
      if (scope.matches('fieldset')) {
        const legend = [...scope.children].find(child => child.matches('legend'));
        const text = legend ? shortText(legend) : '';
        if (text) return text;
      }
      if (scope.matches('tr,[role="row"]')) {
        for (const header of scope.querySelectorAll('th[scope="row"],[role="rowheader"]')) {
          if (header.closest('tr,[role="row"]') !== scope) continue;
          const text = shortText(header);
          if (text) return text;
        }
      }
      return '';
    };
    let labelScope = group, groupLabel = explicitLabel(group) || nearbyLabel(group);
    // 原生同名组的公共容器有时直接包含行标签。只取第一个控件前的静态短文本，
    // 不把其他组、字段值或整块正文当作标签，也不按站点类名猜测含义。
    const choices = [...group.querySelectorAll('input,[role="radio"],[role="checkbox"]')];
    if (!groupLabel && !group.matches('form,[role="form"]') && choices.length && choices.every(c =>
      e.type === 'radio' && e.name ? c.type === 'radio' && c.name === e.name && c.form === e.form
        : c.closest(choiceGroupSelector) === group)) {
      const leading = [];
      for (const child of group.childNodes) {
        if (child.nodeType === 1 && (child.matches(selector) || child.querySelector(selector))) break;
        if (![1, 3].includes(child.nodeType)) continue;
        const text = shortText(child);
        if (text) leading.push(text);
      }
      if (leading.length === 1) groupLabel = leading[0];
    }
    // A short heading adjacent to a native group is usable only while its containing
    // branch has no other fields. Do not borrow another row's or module's heading.
    for (let depth = 0, parent = group.parentElement; !groupLabel && parent && parent !== document.body && depth < 3;
      depth++, parent = parent.parentElement) {
      if (visibleFields(parent).some(field => !group.contains(field))) break;
      labelScope = parent;
      groupLabel = explicitLabel(parent);
      if (!groupLabel) {
        for (const heading of [...parent.querySelectorAll(headingSelector)].reverse()) {
          if (group.contains(heading) || !(heading.compareDocumentPosition(group) & Node.DOCUMENT_POSITION_FOLLOWING)) continue;
          const text = shortText(heading);
          if (text) { groupLabel = text; break; }
        }
      }
      if (parent.matches(sectionSelector)) break;
    }
    return {accessible_name: accessible, group_ref: identity(group), group_label: groupLabel,
      section: fieldSection(labelScope, true)};
  };
  const sectionHeadings = new Map();
  const invalidateHeading = heading => {
    const previous = cache.headingVersions.get(heading);
    if (previous && previous.signature !== null)
      cache.headingVersions.set(heading, {signature: null, revision: previous.revision});
  };
  const sectionChoiceContext = e => {
    if (!e.matches('input[type="checkbox"]')) return null;
    if (e.closest('[role="menuitem"],[role="option"],[role="treeitem"]')) return null;
    const section = e.closest('section');
    if (!section) return null;
    const headings = [...section.querySelectorAll(headingSelector)]
      .filter(node => node.closest('section') === section);
    if (section.children.length !== 2 || section.querySelector('section') || headings.length !== 1) {
      headings.forEach(invalidateHeading);
      return null;
    }
    const heading = headings[0];
    if (!visible(heading) || heading.closest('[hidden],[aria-hidden="true"],[inert]')) {
      invalidateHeading(heading);
      return null;
    }
    const [head, body] = [...section.children];
    const nestedGroup = choiceGroupSelector + ',section,article,dialog,[role="dialog"],'
      + '[role="region"],[role="menuitem"],[role="option"],[role="treeitem"],'
      + 'table,[role="table"],[role="grid"]';
    if (!head.contains(heading) || !body.contains(e) ||
        body.matches(nestedGroup) || body.querySelector(nestedGroup)) {
      invalidateHeading(heading);
      return null;
    }
    const controls = [...section.querySelectorAll(fieldSelector + ',button,a[href],'
      + '[role="checkbox"],[role="radio"],[role="button"]')];
    if (controls.length < 2 || controls.some(node => !body.contains(node) ||
        !node.matches('input[type="checkbox"]'))) {
      invalidateHeading(heading);
      return null;
    }
    const references = [];
    for (const attribute of ['aria-labelledby', 'aria-describedby']) {
      const ids = (heading.getAttribute(attribute) || '').split(/\s+/).filter(Boolean);
      for (const id of ids) {
        const node = document.getElementById(id);
        if (!node || !visible(node) || node.closest('[hidden],[aria-hidden="true"],[inert]')) {
          invalidateHeading(heading);
          return null;
        }
        references.push([attribute, id, identity(node), name(node)]);
      }
    }
    const signature = JSON.stringify([heading.getAttribute('aria-label'),
      heading.getAttribute('aria-labelledby'), heading.getAttribute('aria-describedby'),
      references, name(heading), heading.textContent]);
    const previous = cache.headingVersions.get(heading);
    const version = previous && previous.signature === signature ? previous.revision :
      (previous?.revision || 0) + 1;
    cache.headingVersions.set(heading, {signature, revision: version});
    const context = {group_ref: identity(section), heading_ref: identity(heading),
      heading_revision: version};
    sectionHeadings.set(heading, context);
    return context;
  };
  const markupText = value => value.replace(/[&<>"]/g, c =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[c]);
  const resourceName = source => {
    if (!source) return '';
    if (/^data:/i.test(source)) {
      const comma = source.indexOf(',');
      return comma >= 0 ? source.slice(0, comma) : 'data:';
    }
    try {
      const url = new URL(source, document.baseURI);
      // 只提供已观察资源的文件名和片段，不发送 URL 参数或内联图片数据。
      if (!['http:', 'https:', 'file:'].includes(url.protocol)) return url.protocol;
      const filename = source.startsWith('#') ? url.hash : (url.pathname.split('/').pop() || '') + url.hash;
      try { return decodeURIComponent(filename); } catch { return filename; }
    } catch { return ''; }
  };
  const iconSelector = 'svg,img,input[type="image"]';
  const maxIconField = 120, maxIconMarkup = 512;
  const metadataKeys = ['id', 'class', 'name', 'aria-label', 'title', 'data-icon', 'data-name', 'data-testid'];
  const hintText = value => {
    if (value.length > maxIconField * 4) return '';
    const text = cleanName(value);
    return text.length <= maxIconField ? text : '';
  };
  const attributes = (e, keys) => keys.map(key => {
    const value = hintText(e.getAttribute(key) || '');
    return value ? ' ' + key + '="' + markupText(value) + '"' : '';
  }).join('');
  const iconDescription = (icon, owns) => {
    if (icon.matches('svg')) {
      const title = hintText([...icon.querySelectorAll('title')].find(owns)?.textContent || '');
      let markup = '<svg' + attributes(icon, metadataKeys) + '>' +
        (title ? '<title>' + markupText(title) + '</title>' : '');
      for (const ref of icon.querySelectorAll('use,image')) {
        if (!owns(ref) || !rendered(ref)) continue;
        const href = hintText(resourceName(ref.getAttribute('href') || ref.getAttribute('xlink:href')));
        const reference = '<' + ref.localName + (href ? ' href="' + markupText(href) + '"' : '') +
          attributes(ref, metadataKeys) + '></' + ref.localName + '>';
        if (markup.length + reference.length + '</svg>'.length > maxIconMarkup) break;
        markup += reference;
      }
      return markup + '</svg>';
    }
    const source = hintText(resourceName(icon.currentSrc || icon.getAttribute('src')));
    return '<' + icon.localName + (icon.matches('input') ? ' type="image"' : '') +
      (source ? ' src="' + markupText(source) + '"' : '') + attributes(icon, ['alt', ...metadataKeys]) + '>';
  };
  const iconMarkup = (surface, owns, included) => {
    const walker = document.createTreeWalker(surface, NodeFilter.SHOW_ELEMENT, {
      acceptNode(element) {
        if (!owns(element)) return NodeFilter.FILTER_REJECT;
        return element.matches(iconSelector) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
      },
    });
    let markup = '';
    for (let icon = surface; icon; icon = walker.nextNode()) {
      if (!icon.matches(iconSelector) || !rendered(icon) || (included && !included(icon))) continue;
      const description = iconDescription(icon, owns);
      if (description.length > maxIconMarkup) continue;
      const separator = markup ? ' ' : '';
      if (markup.length + separator.length + description.length > maxIconMarkup) break;
      markup += separator + description;
    }
    return markup;
  };
  const nearbyDescription = e => {
    const parent = e.parentElement;
    if (!parent || parent === document.body) return '';
    const walker = document.createTreeWalker(parent, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (node.nodeType === 3) return NodeFilter.FILTER_ACCEPT;
        if (node === e || independentControl(parent, node) ||
            node.matches('script,style,noscript,template,[hidden],[aria-hidden="true"],[inert]') ||
            (!rendered(node) && getComputedStyle(node).display !== 'contents')) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_SKIP;
      },
    });
    let nearby = '';
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = cleanName(node.textContent);
      if (!text) continue;
      nearby += (nearby ? ' ' : '') + text;
      if (nearby.length > 80) return '';
    }
    return nearby;
  };
  const customName = (e, accessible, surface, icons) => {
    // 图标只提供原始 DOM 信息，不按名称猜测关闭、搜索、方向等操作含义。
    let label = [accessible, icons].filter(Boolean).join(' · ');
    if (!label && ['button', 'link'].includes(role(e))) {
      label = '<' + surface.localName + attributes(surface, metadataKeys) + '>';
      const nearby = surface === e ? nearbyDescription(e) : '';
      if (nearby) label += ' · ' + nearby;
    }
    if (icons && !accessible && !surface.matches(iconSelector) && surface.hasAttribute('class'))
      label += ' · <' + surface.localName + attributes(surface, ['class']) + '>';
    // 同一面板可能展示多个月，使用元素自身已有的日期属性区分同名的“20”。
    // 属性仅补充名称；日期字符串本身不能使静态文本成为点击候选。
    for (const key of ['datetime', 'data-date', 'data-value', 'data-testid']) {
      const match = (e.getAttribute(key) || '').match(/(?:^|\D)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)/);
      if (!match) continue;
      const [, year, month, day] = match, y = Number(year), m = Number(month), d = Number(day);
      const days = [31, y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0) ? 29 : 28,
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
      if (m < 1 || m > 12 || d < 1 || d > days[m - 1]) continue;
      const date = year + '-' + month.padStart(2, '0') + '-' + day.padStart(2, '0');
      return label.includes(date) ? label : date + (label ? ' · ' + label : '');
    }
    return label;
  };
  const tableName = (label, table) => table
    ? [table.row_label, table.column_label, table.node_ref ?
      `table#${table.node_ref} ${table.row_count}×${table.column_count} row ${table.row_index} col ${table.column_index}` : '',
    label].filter((text, i, parts) => text && parts.indexOf(text) === i).join(' · ')
    : label;
  const readonlyInput = icon => {
    for (let depth = 0, scope = icon.parentElement; scope && scope !== document.body && depth < 4;
      depth++, scope = scope.parentElement) {
      if (scope.matches(sectionSelector)) return null;
      const walker = document.createTreeWalker(scope, NodeFilter.SHOW_ELEMENT);
      let input = null, count = 0;
      for (let node = scope; node; node = walker.nextNode()) {
        // A partial subtree cannot prove uniqueness. Stop at the first ambiguous boundary.
        if (++count > 200) return null;
        if (node.matches(fieldSelector)) {
          if (input || icon.contains(node) || node.tagName !== 'INPUT' || !node.readOnly || !safe(node) || !visible(node) ||
              !['textbox', 'searchbox', 'spinbutton', 'combobox'].includes(role(node)) ||
              node.closest(':disabled,[aria-disabled="true"]') || !clipped(node, node.getBoundingClientRect())) return null;
          input = node;
        } else if (node !== icon && visible(node) && (node.matches(selector) || ownPress(node) || customClick(node))) {
          return null;
        }
      }
      if (input) return input;
    }
    return null;
  };
  const describe = (e, point, naming) => {
    const textNodes = naming?.nodes;
    const borrowed = point && point.surface !== e && !e.matches(selector) && customClick(e);
    const surface = borrowed ? point.surface : e;
    // Pointer styling alone cannot create a separate target under a native or handler owner.
    const allowInnerPointer = !borrowed && (e.matches(selector) || ownPress(e));
    const owns = element => {
      if (!surface.contains(element) || element.closest('script,style,noscript,template')) return false;
      for (let current = element; current !== surface; current = current.parentElement) {
        if (independentControl(e, current, allowInnerPointer)) return false;
      }
      return true;
    };
    const included = node => {
      const element = node.nodeType === 3 ? node.parentElement : node;
      if (!owns(element) || element.closest('[hidden],[aria-hidden="true"],[inert]')) return false;
      // Own captions may be visually clipped for screen readers.
      if (!borrowed) {
        const svgTitle = element.localName === 'title' && element.parentElement?.localName === 'svg';
        return rendered(svgTitle ? element.parentElement : element) || getComputedStyle(element).display === 'contents';
      }
      if (!controlBranch(e, element)) return false;
      let rect = element.getBoundingClientRect();
      if (node.nodeType === 3) { nearbyRange.selectNodeContents(node); rect = nearbyRange.getBoundingClientRect(); }
      return exposed(element, rect);
    };
    // Explicit owner names may reference hidden labels; only implicit fallback is surface-local.
    const seen = new Set([e]);
    const accessible = cleanName(explicitName(e, seen, textNodes) || (borrowed ?
      e.getAttribute('title') || e.getAttribute('placeholder') || name(surface, new Set(), included, textNodes) :
      implicitName(e, seen, included, textNodes)));
    if (naming) naming.name = accessible;
    const icons = iconMarkup(surface, owns, borrowed ? included : null);
    const label = accessible => customName(e, accessible, surface, icons);
    const table = tableContext(e);
    const input = role(e) === 'button' && !accessible && !e.matches(fieldSelector) ? readonlyInput(e) : null;
    if (input) {
      const field = describe(input, null, naming);
      return {label: [field.label, cleanName(input.name || ''), label(accessible)]
        .filter((text, i, parts) => text && parts.indexOf(text) === i).join(' · '),
      value: input.value, context: field.context, readonly_input: identity(input)};
    }
    if (['checkbox', 'radio'].includes(role(e))) {
      const owner = !accessible ? choiceOwner(e) : null;
      const group = choiceContext(e, accessible);
      const context = group || table || owner ? {...group, ...(table ? {table} : {}),
        ...(owner ? {accessible_name: accessible, choice_owner: owner} : {})} :
        sectionChoiceContext(e);
      return {label: tableName(label(accessible || owner?.label || ''), table), ...(context ? {context} : {})};
    }
    if (!e.matches(fieldSelector) || ['checkbox', 'radio', 'button', 'submit', 'reset', 'image'].includes(e.type))
      return { label: tableName(label(accessible), table), ...(table ? {context: {table}} : {}) };
    const context = { accessible_name: accessible, nearby_label: nearbyLabel(e), section: fieldSection(e),
      ...(table ? {table} : {}) };
    return {
      label: tableName([context.section, context.nearby_label, accessible]
        .filter((text, i, parts) => text && parts.indexOf(text) === i).join(' · '), table),
      context
    };
  };
  // Keep offscreen state for controls that supplied evidence in this document.
  cache.pageKey = () => [...viewport(),
  [...document.querySelectorAll('input,textarea,select')]
    .filter(e => safe(e) && (cache.fields.has(e) || role(e) && cache.clickPoint(e)))
    .map(e => [identity(e), e.value, e.checked, e.selectedIndex, e.disabled, e.readOnly]),
  regions.map(e => [identity(e), e.scrollTop, e.scrollHeight, e.clientHeight])];
  const actions = [], readonlyIcons = new Set(), descriptionText = new Map();
  for (const e of document.body.querySelectorAll('*')) {
    if (!safe(e) || (!visible(e) && !nativeChoice(e)) || e.closest(':disabled,[aria-disabled="true"]')) continue;
    const rname = role(e);
    if (!rname) continue;
    const point = cache.clickPoint(e);
    if (!point) continue;
    // 纯图标或无文字的自定义控件也保留；名称可由原始 DOM 标记补充。
    const naming = {nodes: []};
    const {readonly_input, ...description} = describe(e, point, naming), label = description.label;
    if (readonly_input) {
      readonlyIcons.add(e);
      cache.fields.add(cache.nodes.get(readonly_input));
    }
    if (rname === 'gridcell' && e.querySelector('button,[role="button"]')) continue;
    if (e.matches('input,textarea,select')) cache.fields.add(e);
    if (naming.nodes.length) descriptionText.set(e, naming);
    const base = {
      node: identity(e), role: rname, ...description, label: label || rname,
      rect: point.rect
    };
    for (const key of ['checked', 'selected', 'expanded']) {
      const value = e.getAttribute('aria-' + key);
      if (value !== null) base[key] = value;
    }
    if (['checkbox', 'radio'].includes(e.type)) base.checked = String(e.checked);
    if (e.tagName === 'SELECT') {
      for (const o of e.options) if (!o.selected && !o.disabled && !o.closest('optgroup[disabled]'))
        actions.push({
          ...base, kind: 'select', value: o.value,
          current_value: [...e.selectedOptions].map(o => o.label).join(', '), label: base.label + ' → ' + cleanName(o.label)
        });
    } else {
      const canFill = editable(e, rname);
      const value = description.value ?? ('value' in e ? String(e.value) :
        e.isContentEditable || rname === 'combobox' ? e.innerText.trim() : '');
      actions.push({ ...base, kind: canFill ? 'fill' : 'click', value });
      if (canFill) actions.push({ ...base, kind: 'click', value, label: 'Open ' + base.label });
    }
  }
  if (regions.length) {
    for (const e of [...regions, document.scrollingElement]) {
      if (!cache.scrollPoint(e)) continue;
      const r = e.getBoundingClientRect(), root = e === document.scrollingElement;
      const label = root ? 'Page' : textPrefix(
        (e.getAttribute('aria-label') || e.querySelector('h1,h2,h3')?.textContent || 'Scrollable region').trim(), 80);
      const base = {node: identity(e), role: 'region', label, rect: {x:r.x,y:r.y,w:r.width,h:r.height}};
      const top = root ? scrollY : e.scrollTop, height = root ? innerHeight : e.clientHeight;
      if (top > 2) actions.push({...base, id:'region_up_' + base.node, kind:'scroll_region_up', delta:-560});
      if (top + height < e.scrollHeight - 2) actions.push({...base, id:'region_down_' + base.node, kind:'scroll_region_down', delta:560});
    }
  }
  const links = [...document.querySelectorAll('a[href]')].filter(e => visible(e) &&
    exposed(e, e.getBoundingClientRect()) && /^https?:$/.test(new URL(e.href, location.href).protocol))
    .slice(0, 250).map((e, i) => ({id:'l' + (i + 1), node:identity(e), url:e.href, text:textPrefix(cleanName(name(e)), 300)}));
  // 标题可把完整可访问名称放在父级，文字子节点则对 AX 隐藏以免重复朗读。
  const namedText = new Map();
  for (const e of document.querySelectorAll('h1,h2,h3,h4,h5,h6,legend,[role="heading"]')) {
    if (!visible(e) || !(e.hasAttribute('aria-label') || e.hasAttribute('aria-labelledby'))) continue;
    const r = e.getBoundingClientRect(), label = name(e);
    if (label && r.width > 0 && r.height > 0 && exposed(e, r)) namedText.set(e, label);
  }
  // 标题可能先于选项进入视口；仅记录符合条件且当前可见的标题来源。
  for (const heading of document.querySelectorAll(headingSelector)) {
    if (!visible(heading) || !exposed(heading, heading.getBoundingClientRect()) ||
        sectionHeadings.has(heading)) continue;
    const section = heading.closest('section');
    const choice = section && [...section.querySelectorAll('input[type="checkbox"]')]
      .find(input => input.closest('section') === section);
    if (choice) {
      const accessible = name(choice);
      if (!choiceContext(choice, accessible) && !tableContext(choice))
        sectionChoiceContext(choice);
    }
  }
  const inNamedText = e => {
    for (let parent = e; parent; parent = parent.parentElement) if (namedText.has(parent)) return true;
    return false;
  };
  const words = [], walker = document.createTreeWalker(
    document.body, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT
  );
  const range = document.createRange(); let node, length = 0;
  while ((node = walker.nextNode()) && length < 6000) {
    if (node.nodeType === 1) {
      if (namedText.has(node)) { words.push(namedText.get(node)); length += namedText.get(node).length; }
      continue;
    }
    const value = node.textContent.trim(), parent = node.parentElement;
    if (!value || !parent || parent.closest('script,style,noscript,template') || !visible(parent) ||
        inNamedText(parent)) continue;
    range.selectNodeContents(node); const r = range.getBoundingClientRect();
    if (r.width > 0 && r.height > 0 && exposed(parent, r)) {
      words.push(value); length += value.length;
    }
  }
  const text = textPrefix(words.join('\n'), 6000), height = document.documentElement.scrollHeight;
  const page_key = cache.pageKey();
  // Compare meaning and identity. Geometry is always resolved and hit-tested just before input.
  const semantics = actions.map(({ rect, ...action }) => action);
  const marker = [performance.timeOrigin, location.href, scrollX, scrollY, innerWidth, innerHeight,
  document.title, text, semantics, page_key[6], page_key[7], links];
  const omitted_actions = Math.max(0, actions.length - 250);
  actions.splice(250);
  actions.forEach((a, i) => a.id = 'e' + (i + 1));
  // 按首次出现的节点编号，与 Python action_space 的元素去重顺序保持一致。
  const indexed = new Map();
  for (const action of actions) {
    const element = cache.nodes.get(action.node);
    // 一个输入框可以同时支持输入和点击，但正文只输出一次。
    if (indexed.has(element)) continue;
    const fields = [String(indexed.size + 1), action.role];
    // 关联图标展示所属 readonly input 的值；checkbox 的 value="on" 并不是勾选状态。
    if (action.kind === 'fill' || action.kind === 'select' ||
        ['textbox', 'searchbox', 'spinbutton'].includes(action.role) || readonlyIcons.has(element)) {
      const value = action.current_value ?? action.value ?? '';
      fields.push('value=' + JSON.stringify(textPrefix(String(value), 80)));
    }
    // 保留 false 和 mixed 等显式状态，不按真假过滤。
    for (const key of ['checked', 'selected', 'expanded']) {
      if (key in action) fields.push(key + '=' + action[key]);
    }
    const label = action.label.split(' → ')[0].replace(/\s+/g, ' ').trim();
    const caption = label === action.role ? '' : textPrefix(label, 80);
    // A complete name segment must fit. A table or section prefix is not evidence that it was shown.
    const sources = new Set();
    const naming = descriptionText.get(element);
    const nameOffset = naming?.name ? (' · ' + label + ' · ').indexOf(' · ' + naming.name + ' · ') : -1;
    if (nameOffset >= 0 && nameOffset + naming.name.length <= caption.length) {
      const labels = [...(element.labels || [])];
      // Independent ARIA references such as headings keep their own position in the document.
      for (const node of naming.nodes) {
        if (element.contains(node) || labels.some(label => label.contains(node))) sources.add(node);
      }
    }
    indexed.set(element, {line: '[' + fields.join(' ') + ']' + (caption ? ' ' + caption : ''), sources});
  }
  // 同一次观察内按 DOM 顺序插入状态，不把控件表与正文分成两个区域。
  const contentParts = [], contentWalker = document.createTreeWalker(
    document.body, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT
  );
  const shownText = new Set(), textParts = new Map();
  let contentNode, contentLength = 0;
  while ((contentNode = contentWalker.nextNode()) && contentLength < 6000) {
    let part = '';
    // 控件标记仅来自已观察到且仍在候选表中的元素。
    if (contentNode.nodeType === 1) {
      const group = sectionHeadings.get(contentNode);
      if (group && visible(contentNode) && exposed(contentNode, contentNode.getBoundingClientRect())) {
        part = `[group_ref=${group.group_ref} heading_ref=${group.heading_ref} ` +
          `heading_revision=${group.heading_revision}]` +
          (namedText.has(contentNode) ? ' ' + namedText.get(contentNode) : '');
      } else {
        const control = indexed.get(contentNode);
        part = control?.line || namedText.get(contentNode) || '';
        if (control?.sources.size) {
          const priorParts = [...control.sources].filter(node => textParts.has(node));
          const repeatedLength = priorParts.reduce(
            (length, node) => length + contentParts[textParts.get(node)].length + 1, 0);
          // A label may precede its control. Remove it only when the entire marker will be emitted.
          if (contentLength - repeatedLength + part.length + 1 <= 6000) {
            for (const node of priorParts) {
              contentParts[textParts.get(node)] = '';
              textParts.delete(node);
            }
            contentLength -= repeatedLength;
            for (const node of control.sources) shownText.add(node);
          }
        }
      }
    } else {
      const parent = contentNode.parentElement;
      // 保持原正文的可见性边界，排除脚本、隐藏文本及敏感输入。
      if (!parent || parent.closest('script,style,noscript,template') || !visible(parent) ||
          shownText.has(contentNode) || inNamedText(parent)) continue;
      range.selectNodeContents(contentNode);
      const r = range.getBoundingClientRect();
      // 仅收集与当前视口相交的正文，不读取下方尚未观察的内容。
      if (r.width > 0 && r.height > 0 && exposed(parent, r)) part = contentNode.textContent.trim();
    }
    // 空节点不占用上下文；总长度仍受原观察预算约束。
    if (part) {
      if (contentNode.nodeType === 3) textParts.set(contentNode, contentParts.length);
      contentParts.push(part);
      contentLength += part.length + 1;
    }
  }
  const content = contentParts.filter(Boolean).join('\n');
  // DOM 顺序也属于模型观察，页面重排后不得复用过期决策。
  marker.push(content);
  // 仅在视口下方仍有页面空间时提供下滑选项，向模型明确可探索的方向。
  const scrollRoot = document.scrollingElement;
  if (scrollRoot && !regions.length && scrollY + innerHeight < height - 2) actions.push({
    id: 'scroll_down', kind: 'scroll',
    label: 'More page content is below the viewport. Scroll down to reveal it when the needed control is not visible.',
    delta: 560
  });
  if (scrollRoot && !regions.length && scrollY > 0)
    actions.push({ id: 'scroll_up', kind: 'scroll', label: 'Scroll up', delta: -560 });
  actions.push({ id: 'wait', kind: 'wait', label: 'Wait for the page to update' });
  return {
    url: location.href, title: document.title, w: innerWidth, h: innerHeight, text, content, links,
    scroll: { y: scrollY, height }, actions, marker, page_key, omitted_actions
  };
})()
