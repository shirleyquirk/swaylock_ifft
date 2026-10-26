// Every decision the inverse makes is a control here. The UI is generated from this list.

const coarse = typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches;

const opt = (...pairs) => pairs.map(([v, l]) => ({ v, l }));
const pow10 = (x) => {
  const v = Math.pow(10, x);
  return v >= 100 || v < 0.01 ? v.toExponential(1) : v.toFixed(v < 1 ? 3 : 2);
};

export const GROUPS = [
  ['source', 'Source'],
  ['mode', 'Mode'],
  ['c', 'C constants'],
  ['decode', 'Decode (undo the forward pass)'],
  ['phase', 'Phase'],
  ['solver', 'Phase retrieval'],
  ['output', 'Output'],
];

export const SCHEMA = [
  // ---------------------------------------------------------------- source
  { key: 'source', group: 'source', type: 'select', label: 'Source', def: 'camera',
    options: opt(['camera', 'Camera'], ['pattern', 'Test pattern'], ['image', 'Image file']),
    help: 'Camera needs HTTPS (github.io is fine) and a permission prompt.' },
  { key: 'facing', group: 'source', type: 'select', label: 'Camera', def: 'user',
    options: opt(['user', 'Front'], ['environment', 'Back']), show: (s) => s.source === 'camera' },
  { key: 'mirror', group: 'source', type: 'check', label: 'Mirror', def: true },
  { key: 'rotate', group: 'source', type: 'range', label: 'Rotate (quarter turns)', def: 0, min: 0, max: 3, step: 1 },
  { key: 'zoom', group: 'source', type: 'range', label: 'Zoom', def: 1, min: 1, max: 4, step: 0.05 },
  { key: 'size', group: 'source', type: 'select', label: 'FFT size', def: coarse ? 256 : 512, num: true,
    options: opt([128, '128'], [256, '256'], [512, '512'], [1024, '1024 (desktop GPU)']),
    help: 'Square, power of two. Cost per solver iteration is two full 2-D FFTs.' },

  // ---------------------------------------------------------------- mode
  { key: 'mode', group: 'mode', type: 'select', label: 'Pipeline', def: 'inverse',
    options: opt(['inverse', 'Inverse: camera image is the spectrum'], ['roundtrip', 'Round trip: C effect, then inverse']),
    help: 'Round trip applies the swaylock effect exactly as the C does (8-bit truncation included), then decodes it. Ground truth for the inverse.' },
  { key: 'view', group: 'mode', type: 'select', label: 'View', def: 'output',
    options: opt(['output', 'Output'], ['split', 'Split (source | output)'], ['input', 'Decoder input'],
      ['encoded', 'Forward effect (C)'], ['magnitude', 'Final spectrum (log |F|)'])},
  { key: 'shift', group: 'mode', type: 'check', label: 'fftshift (DC at centre)', def: false,
    help: 'The C writes DC at the top-left pixel. On: the image centre is read as DC, so the middle of the frame controls large structures.' },

  // ---------------------------------------------------------------- C constants
  { key: 'gH', group: 'c', type: 'range', label: 'Hue gain  (C: 50)', def: 50, min: 1, max: 200, step: 0.5,
    help: 'h = fmod(gH·log(1+|H|), 360). Lower = fewer wraps, less branch ambiguity.' },
  { key: 'gS', group: 'c', type: 'range', label: 'Sat gain  (C: 25)', def: 25, min: 1, max: 100, step: 0.5,
    help: 's = min(1, gS·log(1+|S|)/255). Clips once |S| > e^(255/gS).' },
  { key: 'gV', group: 'c', type: 'range', label: 'Value gain  (C: 50)', def: 50, min: 1, max: 200, step: 0.5,
    help: 'v = min(1, gV·log(1+|V|)/255). At 50 every bin above |V|≈163 clips — most of the low frequencies.' },

  // ---------------------------------------------------------------- decode
  { key: 'bias', group: 'decode', type: 'range', label: 'Truncation bias', def: 0.5, min: 0, max: 1, step: 0.01,
    help: '(uint8_t) truncates, so the true value lies in [q, q+1)/255. 0.5 = midpoint, 0 = the C\'s literal value.' },
  { key: 'herm', group: 'decode', type: 'select', label: 'Conjugate symmetry', def: 'intersect',
    options: opt(['intersect', 'Intersect ±k intervals'], ['average', 'Average ±k'], ['left', 'Left half is truth (C)'],
      ['right', 'Right half is truth'], ['none', 'Ignore (complex output)']),
    help: 'A real image has |F(k)| = |F(−k)|. A camera frame isn\'t symmetric, so the two halves disagree; this picks how to reconcile them.' },
  { key: 'hueBranch', group: 'decode', type: 'select', label: 'Hue branch (decode)', def: 'vmatch',
    options: opt(['fixed', 'Fixed k'], ['prior', 'Nearest radial 1/f prior'], ['vmatch', 'Nearest ratio·|V| (cross-channel)']),
    help: 'Hue only gives log(1+|H|) mod 360/gH. Each branch k multiplies |H| by e^(360/gH).' },
  { key: 'hueK', group: 'decode', type: 'range', label: 'Branch k', def: 0, min: 0, max: 8, step: 1,
    show: (s) => s.hueBranch === 'fixed' },
  { key: 'hueRatio', group: 'decode', type: 'range', label: '|H|, |S| : |V| ratio', def: 0, min: -3, max: 1, step: 0.05,
    fmt: pow10, show: (s) => s.hueBranch === 'vmatch' || s.unkFill === 'vmatch' },
  { key: 'unkFill', group: 'decode', type: 'select', label: 'H / S where the pixel is black or grey', def: 'vmatch',
    options: opt(['vmatch', 'ratio·|V| (cross-channel)'], ['prior', 'Radial 1/f prior']),
    help: 'A black or grey encoded pixel says nothing about |H| (and black says nothing about |S|). Borrowing V\'s magnitude keeps flat regions flat.' },
  { key: 'priorLevel', group: 'decode', type: 'range', label: '1/f prior level', def: -1, min: -7, max: 0, step: 0.05,
    fmt: pow10, help: 'Fill for clipped / unknown bins: |F| ≈ level·N²·r^−α, never below the clip threshold.' },
  { key: 'priorAlpha', group: 'decode', type: 'range', label: '1/f prior exponent α', def: 1.5, min: 0, max: 3, step: 0.05 },
  { key: 'magScale', group: 'decode', type: 'range', label: 'Magnitude scale', def: 0, min: -3, max: 3, step: 0.05,
    fmt: pow10, help: 'Raw magnitudes scale with pixel count. 1 = exact for the round trip at this FFT size.' },
  { key: 'dcOverride', group: 'decode', type: 'check', label: 'Override DC (means)', def: false,
    help: 'DC is just the mean of each channel. Set it directly instead of decoding a clipped pixel.' },
  { key: 'dcH', group: 'decode', type: 'range', label: 'Mean hue', def: 0.5, min: 0, max: 1, step: 0.01, show: (s) => s.dcOverride },
  { key: 'dcS', group: 'decode', type: 'range', label: 'Mean saturation', def: 0.5, min: 0, max: 1, step: 0.01, show: (s) => s.dcOverride },
  { key: 'dcV', group: 'decode', type: 'range', label: 'Mean value', def: 0.4, min: 0, max: 1, step: 0.01, show: (s) => s.dcOverride },

  // ---------------------------------------------------------------- phase
  { key: 'phase', group: 'phase', type: 'select', label: 'Phase source', def: 'camera',
    options: opt(['camera', 'Camera FFT (phase swap)'], ['warm', 'Previous frame (warm start)'],
      ['random', 'Random (Hermitian)'], ['zero', 'Zero']),
    help: 'The forward pass discards phase. Camera: magnitudes from the decode, phase from the camera\'s own FFT. Warm: keep the solver\'s estimate across frames.' },
  { key: 'lockPhase', group: 'phase', type: 'check', label: 'Use V phase for H and S', def: false,
    show: (s) => s.phase !== 'warm' },
  { key: 'jitter', group: 'phase', type: 'range', label: 'Phase jitter', def: 0, min: 0, max: 1, step: 0.01,
    show: (s) => s.phase !== 'warm' },
  { key: 'drift', group: 'phase', type: 'range', label: 'Phase drift speed', def: 0, min: 0, max: 1, step: 0.005,
    help: 'Each bin rotates at its own odd-symmetric rate, so the output stays real.', show: (s) => s.phase !== 'warm' },
  { key: 'seed', group: 'phase', type: 'range', label: 'Seed', def: 1, min: 0, max: 64, step: 1 },

  // ---------------------------------------------------------------- solver
  { key: 'iters', group: 'solver', type: 'range', label: 'Iterations / frame', def: 0, min: 0, max: 20, step: 1,
    help: 'Each = FFT, magnitude projection, IFFT, object constraint. Warm start accumulates them across frames.' },
  { key: 'algo', group: 'solver', type: 'select', label: 'Algorithm', def: 'hio',
    options: opt(['hio', 'HIO (Fienup)'], ['er', 'Error reduction']) },
  { key: 'beta', group: 'solver', type: 'range', label: 'HIO β', def: 0.9, min: 0, max: 2, step: 0.01, show: (s) => s.algo === 'hio' },
  { key: 'erEvery', group: 'solver', type: 'range', label: 'ER every n iters (0 = never)', def: 0, min: 0, max: 10, step: 1,
    show: (s) => s.algo === 'hio' },
  { key: 'strength', group: 'solver', type: 'range', label: 'Magnitude constraint strength', def: 1, min: 0, max: 1, step: 0.01 },
  { key: 'slack', group: 'solver', type: 'range', label: 'Interval slack', def: 1, min: 0, max: 4, step: 0.05,
    help: 'Each 8-bit value only pins |F| to an interval. 0 = force the bias point, 1 = the exact quantisation interval, >1 looser.' },
  { key: 'clipMode', group: 'solver', type: 'select', label: 'Clipped bins', def: 'free',
    options: opt(['free', 'Free above threshold'], ['fill', 'Fixed to prior']),
    help: 'A clipped pixel only says |F| ≥ threshold. Free lets the solver choose anything above it.' },
  { key: 'hueSolve', group: 'solver', type: 'check', label: 'Solver picks hue branch', def: true,
    help: 'Per bin, snap |H| to whichever 360° branch is nearest the current estimate instead of the decode-time guess.' },
  { key: 'hueBound', group: 'solver', type: 'select', label: 'Hue domain', def: 'wrap',
    options: opt(['wrap', 'Circular (wrap)'], ['clamp', 'Clamp to [0,1]']) },
  { key: 'support', group: 'solver', type: 'select', label: 'Support', def: 'full',
    options: opt(['full', 'Whole frame'], ['circle', 'Disc'], ['square', 'Square']),
    help: 'Classic phase-retrieval constraint: image is zero outside the support. Pins translation and helps convergence.' },
  { key: 'supportR', group: 'solver', type: 'range', label: 'Support size', def: 0.7, min: 0.1, max: 1, step: 0.01,
    show: (s) => s.support !== 'full' },

  // ---------------------------------------------------------------- output
  { key: 'space', group: 'output', type: 'select', label: 'Interpret channels as', def: 'hsv',
    options: opt(['hsv', 'HSV (as the C)'], ['rgb', 'RGB']) },
  { key: 'part', group: 'output', type: 'select', label: 'Complex → real', def: 'real',
    options: opt(['real', 'Real part'], ['abs', 'Magnitude'])},
  { key: 'hueShift', group: 'output', type: 'range', label: 'Hue shift', def: 0, min: 0, max: 1, step: 0.005 },
  { key: 'hueScale', group: 'output', type: 'range', label: 'Hue scale', def: 1, min: -4, max: 4, step: 0.05 },
  { key: 'satGain', group: 'output', type: 'range', label: 'Saturation gain', def: 1, min: 0, max: 4, step: 0.01 },
  { key: 'autoExpo', group: 'output', type: 'check', label: 'Auto exposure', def: true },
  { key: 'valGain', group: 'output', type: 'range', label: 'Value gain', def: 0, min: -3, max: 3, step: 0.02, fmt: pow10,
    help: 'Multiplies on top of auto exposure when that is on.' },
  { key: 'gamma', group: 'output', type: 'range', label: 'Gamma', def: 1, min: 0.2, max: 3, step: 0.01 },
];

export const PRESETS = {
  'Phase swap': { mode: 'inverse', phase: 'camera', iters: 0, view: 'output' },
  'Warm HIO': { mode: 'inverse', phase: 'warm', iters: 3, algo: 'hio', beta: 0.9, view: 'output' },
  'Round trip — oracle phase': { mode: 'roundtrip', phase: 'camera', iters: 0, view: 'split', shift: false },
  'Round trip — blind HIO': { mode: 'roundtrip', phase: 'warm', iters: 4, algo: 'hio', view: 'split', support: 'full' },
  'Noise bloom': { mode: 'inverse', phase: 'random', drift: 0.15, iters: 0, shift: true, view: 'output' },
  'C effect only': { mode: 'roundtrip', view: 'encoded' },
};

const KEY = 'fft-invert.settings.v1';
const NOPERSIST = new Set();

export function defaults() {
  return Object.fromEntries(SCHEMA.map((c) => [c.key, c.def]));
}

function sanitize(obj) {
  const out = {};
  for (const c of SCHEMA) {
    if (!(c.key in obj)) continue;
    let v = obj[c.key];
    if (c.type === 'check') v = !!v;
    else if (c.type === 'range') { v = Number(v); if (!Number.isFinite(v)) continue; v = Math.min(c.max, Math.max(c.min, v)); }
    else if (c.type === 'select') {
      if (c.num) v = Number(v);
      if (!c.options.some((o) => o.v === v)) continue;
    }
    out[c.key] = v;
  }
  return out;
}

export function load() {
  let s = defaults();
  try { s = { ...s, ...sanitize(JSON.parse(localStorage.getItem(KEY) || '{}')) }; } catch (e) { /* storage blocked */ }
  const h = location.hash.match(/[#&]c=([^&]+)/);
  if (h) {
    try {
      const json = atob(h[1].replace(/-/g, '+').replace(/_/g, '/'));
      s = { ...s, ...sanitize(JSON.parse(json)) };
    } catch (e) { /* bad link */ }
  }
  return s;
}

export function save(s) {
  const d = defaults();
  const diff = {};
  for (const [k, v] of Object.entries(s)) if (!NOPERSIST.has(k) && v !== d[k]) diff[k] = v;
  try { localStorage.setItem(KEY, JSON.stringify(diff)); } catch (e) { /* storage blocked */ }
  return diff;
}

export function shareURL(s) {
  const diff = save(s);
  const b64 = btoa(JSON.stringify(diff)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  return `${location.origin}${location.pathname}#c=${b64}`;
}
