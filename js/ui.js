import { SCHEMA, GROUPS, PRESETS, defaults } from './settings.js';

const OPEN_BY_DEFAULT = new Set(['mode', 'phase', 'solver']);

export function buildPanel(root, s, onChange, extras) {
  root.innerHTML = '';
  const rows = [];

  // presets + actions
  const top = el('div', 'panel-top');
  const preset = el('select', 'preset');
  preset.append(new Option('Presets…', ''));
  for (const name of Object.keys(PRESETS)) preset.append(new Option(name, name));
  preset.onchange = () => {
    if (!preset.value) return;
    const keep = { size: s.size, source: s.source, facing: s.facing, mirror: s.mirror, rotate: s.rotate, zoom: s.zoom };
    Object.assign(s, defaults(), keep, PRESETS[preset.value]);
    preset.value = '';
    refresh();
    onChange(null);
  };
  const share = button('Copy link', () => extras.share());
  const reset = button('Defaults', () => {
    const keepSize = s.size, keepSource = s.source, keepFacing = s.facing;
    Object.assign(s, defaults(), { size: keepSize, source: keepSource, facing: keepFacing });
    refresh();
    onChange(null);
  });
  const helpT = button('Help', () => root.classList.toggle('show-help'));
  top.append(preset, share, reset, helpT);
  root.append(top);

  for (const [gid, gname] of GROUPS) {
    const det = el('details', 'group');
    det.open = OPEN_BY_DEFAULT.has(gid);
    const sum = el('summary');
    sum.textContent = gname;
    det.append(sum);
    if (gid === 'source') {
      const pick = button('Choose image…', () => extras.pickImage());
      pick.classList.add('wide');
      const r = el('div', 'row');
      r.append(pick);
      rows.push({ c: { show: (st) => st.source === 'image' }, row: r, sync: () => {} });
      det.append(r);
    }
    for (const c of SCHEMA.filter((x) => x.group === gid)) {
      const r = control(c, s, () => { refresh(); onChange(c.key); });
      det.append(r.row);
      rows.push(r);
    }
    root.append(det);
  }

  function refresh() {
    for (const r of rows) {
      r.sync();
      r.row.hidden = r.c.show ? !r.c.show(s) : false;
    }
  }
  refresh();
  return { refresh };
}

function control(c, s, changed) {
  const row = el('div', 'row');
  const lab = el('label');
  const name = el('span', 'name');
  name.textContent = c.label;
  lab.append(name);
  let input, out, sync;
  const fmt = c.fmt || ((v) => (Number.isInteger(c.step) && c.step >= 1 ? String(v) : (+v).toFixed(c.step < 0.01 ? 3 : 2)));

  if (c.type === 'range') {
    out = el('output');
    lab.append(out);
    input = el('input');
    input.type = 'range';
    input.min = c.min; input.max = c.max; input.step = c.step;
    input.oninput = () => { s[c.key] = Number(input.value); out.textContent = fmt(s[c.key]); changed(); };
    sync = () => { input.value = s[c.key]; out.textContent = fmt(s[c.key]); };
    // double-click / double-tap the label to reset
    name.title = 'Double-click to reset';
    name.ondblclick = () => { s[c.key] = c.def; sync(); changed(); };
    row.append(lab, input);
  } else if (c.type === 'select') {
    input = el('select');
    for (const o of c.options) input.append(new Option(o.l, String(o.v)));
    input.onchange = () => { s[c.key] = c.num ? Number(input.value) : input.value; changed(); };
    sync = () => { input.value = String(s[c.key]); };
    row.append(lab, input);
  } else if (c.type === 'check') {
    row.classList.add('check');
    input = el('input');
    input.type = 'checkbox';
    input.onchange = () => { s[c.key] = input.checked; changed(); };
    sync = () => { input.checked = !!s[c.key]; };
    lab.prepend(input);
    row.append(lab);
  }
  if (c.help) {
    const h = el('small', 'help');
    h.textContent = c.help;
    row.append(h);
  }
  return { c, row, sync };
}

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function button(text, fn) {
  const b = el('button');
  b.type = 'button';
  b.textContent = text;
  b.onclick = fn;
  return b;
}
