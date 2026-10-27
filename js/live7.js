import { processFrame, decodeThumb, CoefBuffer, camPoint } from './live7core.js';

// ---------------------------------------------------------------- settings (every decision is a knob)
const SCHEMA = [
  ['Screen (the locked one)', [
    { k: 'screenW', l: 'Screen width (px)', t: 'num', d: 1920, min: 64, max: 8192,
      h: 'Physical pixels of the locked output: sets the scale of every coefficient.' },
    { k: 'screenH', l: 'Screen height (px)', t: 'num', d: 1080, min: 64, max: 8192 },
    { k: 'rot', l: 'Screen top-left corner is at camera…', t: 'sel', d: 0, num: true,
      o: [[0, 'top-left'], [90, 'top-right'], [180, 'bottom-right'], [270, 'bottom-left']] },
    { k: 'mirror', l: 'Mirror (front camera)', t: 'chk', d: false },
  ]],
  ['Camera', [
    { k: 'res', l: 'Capture resolution', t: 'sel', d: '3840x2160', o: [['1280x720', '720p'], ['1920x1080', '1080p'], ['2560x1440', '1440p'], ['3840x2160', '4K (best)']],
      h: 'More camera pixels per screen pixel is the whole game. Ask for 4K; the browser gives what it can.' },
    { k: 'zoom', l: 'Zoom', t: 'rng', d: 1, min: 1, max: 10, s: 0.1, h: 'Only if the browser exposes camera zoom.' },
  ]],
  ['Grid lock', [
    { k: 'latticeSignal', l: 'Lattice signal', t: 'sel', d: 'auto', o: [['auto', 'auto'], ['lum', 'brightness'], ['edge', 'edge energy']],
      h: 'Brightness sees pixel gaps / subpixel stripes; edge energy sees pixel boundaries and survives blur.' },
    { k: 'minPitch', l: 'Min camera px per screen px', t: 'rng', d: 2.5, min: 2, max: 10, s: 0.1 },
    { k: 'maxPitch', l: 'Max camera px per screen px', t: 'rng', d: 40, min: 8, max: 80, s: 1 },
    { k: 'regridEvery', l: 'Re-find lattice every n frames', t: 'rng', d: 15, min: 1, max: 60, s: 1 },
    { k: 'subpixel', l: 'Per-channel subpixel windows', t: 'chk', d: true,
      h: 'Measures where R, G, B emitters sit inside a pixel and samples each there.' },
    { k: 'qScale', l: 'Window size (× emitter width)', t: 'rng', d: 0.25, min: 0.1, max: 0.6, s: 0.01 },
    { k: 'q', l: 'Window half-width (no subpixel)', t: 'rng', d: 0.3, min: 0.05, max: 0.48, s: 0.01 },
    { k: 'phaseA', l: 'Nudge window x (pixels)', t: 'rng', d: 0, min: -0.5, max: 0.5, s: 0.01 },
    { k: 'phaseB', l: 'Nudge window y (pixels)', t: 'rng', d: 0, min: -0.5, max: 0.5, s: 0.01 },
    { k: 'autoOrigin', l: 'Find pixel (0,0) from the bezel', t: 'chk', d: true },
    { k: 'nudgeX', l: 'Origin nudge x (pixels)', t: 'rng', d: 0, min: -8, max: 8, s: 1 },
    { k: 'nudgeY', l: 'Origin nudge y (pixels)', t: 'rng', d: 0, min: -8, max: 8, s: 1 },
  ]],
  ['Calibration (camera → bytes)', [
    { k: 'calib', l: 'Tone curve', t: 'sel', d: 'symmetric', o: [['symmetric', 'symmetric (quantile pairing)'], ['linear', 'linear']],
      h: 'Coefficient signs are 50/50, so each channel is symmetric about 127.5: pairing quantiles undoes the camera curve.' },
    { k: 'autoCentre', l: 'Auto centre (median → 127.5)', t: 'chk', d: true },
    { k: 'autoSlope', l: 'Auto scale from pixel (0,0)', t: 'chk', d: true, h: "Pixel (0,0) is Y's DC: about 244 for any ordinary screen." },
    { k: 'dcTarget', l: 'DC byte', t: 'rng', d: 244, min: 225, max: 254, s: 0.5 },
    { k: 'gain', l: 'Gain', t: 'rng', d: 1, min: 0.3, max: 3, s: 0.01 },
    { k: 'gR', l: 'Gain R (Y)', t: 'rng', d: 1, min: 0.5, max: 1.5, s: 0.01 },
    { k: 'gG', l: 'Gain G (Cb)', t: 'rng', d: 1, min: 0.5, max: 1.5, s: 0.01 },
    { k: 'gB', l: 'Gain B (Cr)', t: 'rng', d: 1, min: 0.5, max: 1.5, s: 0.01 },
    { k: 'gamma', l: 'Gamma on |e|', t: 'rng', d: 1, min: 0.5, max: 2, s: 0.01 },
    { k: 'crosstalk', l: 'Colour crosstalk', t: 'rng', d: 0, min: 0, max: 0.25, s: 0.005 },
  ]],
  ['Decode', [
    { k: 'logScale', l: 'Log softening κ', t: 'rng', d: 0.75, min: 0.4, max: 1, s: 0.01,
      h: 'One byte level is a 12.5% coefficient change; κ<1 turns an error factor F into F^κ. 1 = exact.' },
    { k: 'lowTrust', l: 'Low-frequency trust', t: 'rng', d: 1, min: 0, max: 1, s: 0.01 },
    { k: 'lowN', l: 'Low-frequency block', t: 'rng', d: 8, min: 1, max: 32, s: 1 },
    { k: 'taper', l: 'Taper (ringing)', t: 'rng', d: 0.3, min: 0, max: 1, s: 0.01 },
    { k: 'maxK', l: 'Max coefficients per axis', t: 'rng', d: 384, min: 16, max: 512, s: 4 },
    { k: 'accumulate', l: 'Accumulate across frames', t: 'chk', d: true, h: 'Pixels are indexed on the screen, so the camera can move.' },
    { k: 'alpha', l: 'Forgetting rate', t: 'rng', d: 0.05, min: 0.01, max: 1, s: 0.01 },
    { k: 'autoLevels', l: 'Auto levels', t: 'chk', d: true },
    { k: 'chroma', l: 'Colour', t: 'rng', d: 1, min: 0, max: 3, s: 0.01 },
    { k: 'pixelated', l: 'Show blocks (no smoothing)', t: 'chk', d: false },
  ]],
];
const ALL = SCHEMA.flatMap(([, g]) => g);
const KEY = 'swaylock-ifft.live7.v1';
const s = Object.fromEntries(ALL.map((c) => [c.k, c.d]));
try { Object.assign(s, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) { /* storage blocked */ }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) { /* storage blocked */ } };

const $ = (id) => document.getElementById(id);
const video = $('video'), view = $('view'), thumb = $('thumb'), status = $('status'), msg = $('msg');
const vctx = view.getContext('2d');
const grab = document.createElement('canvas');
const gctx = grab.getContext('2d', { willReadFrequently: true });
const tctx = thumb.getContext('2d');

const state = { buf: new CoefBuffer(512), frame: 0 };
let stream = null, track = null, frozen = false, still = null, busy = false, last = null, lastFrame = null;
let fps = 0, tPrev = performance.now();

const coreOpt = () => ({
  ...s, chGain: [s.gR, s.gG, s.gB], q: s.q,
});

// ---------------------------------------------------------------- panel
function buildPanel() {
  const root = $('panel');
  root.innerHTML = '';
  const top = document.createElement('div');
  top.className = 'panel-top';
  const mk = (t, f) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.onclick = f; return b; };
  top.append(
    mk('Defaults', () => { for (const c of ALL) s[c.k] = c.d; save(); buildPanel(); changed('all'); }),
    mk('Help', () => root.classList.toggle('show-help')),
  );
  root.append(top);
  SCHEMA.forEach(([name, ctrls], gi) => {
    const det = document.createElement('details');
    det.className = 'group';
    det.open = gi === 0;
    const sum = document.createElement('summary');
    sum.textContent = name;
    det.append(sum);
    for (const c of ctrls) {
      const row = document.createElement('div');
      row.className = 'row' + (c.t === 'chk' ? ' check' : '');
      const lab = document.createElement('label');
      const nm = document.createElement('span');
      nm.textContent = c.l;
      lab.append(nm);
      let input;
      if (c.t === 'rng') {
        const out = document.createElement('output');
        out.textContent = s[c.k];
        lab.append(out);
        input = document.createElement('input');
        Object.assign(input, { type: 'range', min: c.min, max: c.max, step: c.s, value: s[c.k] });
        input.oninput = () => { s[c.k] = +input.value; out.textContent = input.value; save(); changed(c.k); };
        nm.ondblclick = () => { s[c.k] = c.d; input.value = c.d; out.textContent = c.d; save(); changed(c.k); };
        row.append(lab, input);
      } else if (c.t === 'num') {
        input = document.createElement('input');
        Object.assign(input, { type: 'number', min: c.min, max: c.max, value: s[c.k], inputMode: 'numeric' });
        input.className = 'num';
        input.onchange = () => { s[c.k] = Math.max(c.min, Math.min(c.max, Math.round(+input.value) || c.d)); input.value = s[c.k]; save(); changed(c.k); };
        row.append(lab, input);
      } else if (c.t === 'sel') {
        input = document.createElement('select');
        for (const [v, t] of c.o) input.append(new Option(t, String(v)));
        input.value = String(s[c.k]);
        input.onchange = () => { s[c.k] = c.num ? +input.value : input.value; save(); changed(c.k); };
        row.append(lab, input);
      } else {
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = !!s[c.k];
        input.onchange = () => { s[c.k] = input.checked; save(); changed(c.k); };
        lab.prepend(input);
        row.append(lab);
      }
      if (c.h) { const hp = document.createElement('small'); hp.className = 'help'; hp.textContent = c.h; row.append(hp); }
      det.append(row);
    }
    root.append(det);
  });
}

const REPROCESS = new Set(['latticeSignal', 'minPitch', 'maxPitch', 'subpixel', 'qScale', 'q', 'phaseA', 'phaseB', 'autoOrigin',
  'nudgeX', 'nudgeY', 'rot', 'mirror', 'all']);
function changed(k) {
  if (k === 'res' && stream) startCamera();
  if (k === 'zoom') applyZoom();
  if (k === 'screenW' || k === 'screenH' || k === 'all') setAspect();
  if (k === 'pixelated' || k === 'all') thumb.style.imageRendering = s.pixelated ? 'pixelated' : 'auto';
  // grid/geometry knobs change which pixel each sample lands in: re-read (a still or frozen frame)
  if (REPROCESS.has(k)) { state.buf.reset(); state.coarse = null; if (lastFrame && (frozen || still)) runFrame(lastFrame); }
  decodeAndShow();
}
function setAspect() { thumb.style.aspectRatio = `${s.screenW} / ${s.screenH}`; }

// ---------------------------------------------------------------- camera
async function startCamera() {
  stopCamera();
  still = null;
  if (!navigator.mediaDevices?.getUserMedia) { say('No camera API: needs HTTPS (github.io is fine) or localhost.'); return; }
  const [w, h] = s.res.split('x').map(Number);
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: w }, height: { ideal: h } },
    });
    track = stream.getVideoTracks()[0];
    video.srcObject = stream;
    await video.play();
    applyZoom();
    $('overlay').hidden = true;
    const st = track.getSettings?.() || {};
    say(`Camera ${st.width || video.videoWidth}×${st.height || video.videoHeight}. Aim at the screen's top-left corner, close enough to see single pixels.`);
    loop();
  } catch (e) { say(`Camera: ${e.name || ''} ${e.message || e}`); }
}
function stopCamera() { if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; track = null; }
async function applyZoom() {
  const cap = track?.getCapabilities?.();
  if (!cap?.zoom) return;
  const z = Math.max(cap.zoom.min, Math.min(cap.zoom.max, s.zoom));
  try { await track.applyConstraints({ advanced: [{ zoom: z }] }); } catch (e) { /* not supported */ }
}

function loop() {
  if (!stream) return;
  const next = () => (video.requestVideoFrameCallback ? video.requestVideoFrameCallback(loop) : requestAnimationFrame(loop));
  if (frozen || busy || video.readyState < 2 || !video.videoWidth) { drawPreview(); next(); return; }
  const w = video.videoWidth, h = video.videoHeight;
  if (grab.width !== w || grab.height !== h) { grab.width = w; grab.height = h; }
  gctx.drawImage(video, 0, 0, w, h);
  lastFrame = { data: gctx.getImageData(0, 0, w, h).data, w, h, src: video };
  runFrame(lastFrame);
  next();
}

function runFrame(fr) {
  busy = true;
  try {
    last = processFrame(fr.data, fr.w, fr.h, state, coreOpt());
    decodeAndShow();
    drawPreview();
  } catch (e) { console.error(e); say(String(e.message || e)); }
  busy = false;
  const now = performance.now();
  fps = 0.8 * fps + 0.2 * (1000 / Math.max(1, now - tPrev));
  tPrev = now;
  showStatus();
}

let lastDecode = null;
function decodeAndShow() {
  const [ka] = state.buf.extent();
  if (!ka) { tctx.clearRect(0, 0, thumb.width, thumb.height); return; }
  lastDecode = decodeThumb(state.buf, coreOpt());
  const { rgba, K, Kh } = lastDecode;
  if (thumb.width !== K || thumb.height !== Kh) { thumb.width = K; thumb.height = Kh; }
  tctx.putImageData(new ImageData(rgba, K, Kh), 0, 0);
}

// ---------------------------------------------------------------- preview + overlay
function drawPreview() {
  const src = still ? still.bitmap : video;
  const w = still ? still.w : video.videoWidth, h = still ? still.h : video.videoHeight;
  if (!w) return;
  const box = view.parentElement.getBoundingClientRect();
  const sc = Math.min(box.width / w, box.height / h);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = Math.max(1, Math.round(w * sc * dpr)), ch = Math.max(1, Math.round(h * sc * dpr));
  if (view.width !== cw || view.height !== ch) { view.width = cw; view.height = ch; view.style.width = `${cw / dpr}px`; view.style.height = `${ch / dpr}px`; }
  vctx.drawImage(src, 0, 0, cw, ch);
  if (!last?.ga) return;
  const k = cw / w;
  const P = (u, v) => { const [x, y] = camPoint(last.ga, last.gb, u, v); return [x * k, y * k]; };
  const oa = last.offA?.[1] ?? 0, ob = last.offB?.[1] ?? 0;
  vctx.lineWidth = Math.max(1, dpr);
  // sparse lattice dots
  if (last.acc) {
    vctx.fillStyle = 'rgba(79,209,197,0.8)';
    const { i0, j0, ni, nj } = last.acc, st = Math.max(4, Math.round(ni / 24));
    for (let j = j0; j < j0 + nj; j += st) for (let i = i0; i < i0 + ni; i += st) {
      const [x, y] = P(i + oa, j + ob);
      if (x >= 0 && y >= 0 && x < cw && y < ch) vctx.fillRect(x - 1.5 * dpr, y - 1.5 * dpr, 3 * dpr, 3 * dpr);
    }
  }
  // screen pixel (0,0) and the screen axes
  if (last.ok && last.S) {
    const toCam = (a, b) => {
      let si = a + last.os + last.S.s0, ti = b + last.ot + last.S.t0, i, j;
      switch (s.rot) { case 90: i = -ti; j = si; break; case 180: i = -si; j = -ti; break; case 270: i = ti; j = -si; break; default: i = si; j = ti; }
      if (s.mirror) i = -i;
      return [i, j];
    };
    const [i, j] = toCam(0, 0);
    const corner = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([du, dv]) => P(i + oa + du, j + ob + dv));
    vctx.strokeStyle = '#ff4d4d';
    vctx.lineWidth = 2 * dpr;
    vctx.beginPath(); corner.forEach(([x, y], n) => (n ? vctx.lineTo(x, y) : vctx.moveTo(x, y))); vctx.closePath(); vctx.stroke();
    const arrow = (a, b, col) => {
      const [ii, jj] = toCam(a, b);
      const [x0, y0] = P(i + oa, j + ob), [x1, y1] = P(ii + oa, jj + ob);
      vctx.strokeStyle = col; vctx.beginPath(); vctx.moveTo(x0, y0); vctx.lineTo(x1, y1); vctx.stroke();
    };
    arrow(16, 0, '#ffb547');
    arrow(0, 16, '#4d7cff');
  }
}

function showStatus() {
  if (!last) return;
  const parts = [];
  if (last.pitch) parts.push(`${last.pitch.toFixed(2)} cam px/px`, `${last.angle.toFixed(1)}°`, `fit ${(last.ga.rms * 100).toFixed(1)}%/${(last.gb.rms * 100).toFixed(1)}%`);
  if (lastDecode) parts.push(`${lastDecode.K}×${lastDecode.Kh} coeffs`, `scale ${lastDecode.slope.toFixed(2)}${lastDecode.slopeOk ? '' : '?'}`);
  parts.push(`${fps.toFixed(1)} fps`, `${(last.ms ?? 0).toFixed(0)} ms`);
  status.textContent = parts.join(' · ');
  let m = last.ok ? '' : last.msg || '';
  if (last.pitch && last.pitch < 4) m = `Too far: ${last.pitch.toFixed(1)} camera px per screen px (need ≥ 4.5). Move closer, zoom, or use 4K. ${m}`;
  if (last.ga && (last.ga.rms > 0.05 || last.gb.rms > 0.05)) m = `Grid lock weak (hold steadier / square-on). ${m}`;
  say(m, true);
}
function say(t, quiet) { msg.textContent = t; msg.hidden = !t; if (!quiet && t) console.log(t); }

// ---------------------------------------------------------------- still photo
$('file').onchange = async () => {
  const f = $('file').files?.[0];
  if (!f) return;
  stopCamera();
  try {
    const bitmap = await createImageBitmap(f);
    const w = bitmap.width, h = bitmap.height;
    grab.width = w; grab.height = h;
    gctx.drawImage(bitmap, 0, 0);
    still = { bitmap, w, h };
    lastFrame = { data: gctx.getImageData(0, 0, w, h).data, w, h };
    state.buf.reset(); state.coarse = null; state.frame = 0;
    $('overlay').hidden = true;
    say(`Photo ${w}×${h}.`);
    runFrame(lastFrame);
  } catch (e) { say(`Could not read photo: ${e.message || e}`); }
};

// ---------------------------------------------------------------- buttons
$('btn-camera').onclick = startCamera;
$('btn-photo').onclick = () => $('file').click();
$('btn-freeze').onclick = (e) => { frozen = !frozen; e.currentTarget.classList.toggle('on', frozen); };
$('btn-reset').onclick = () => { state.buf.reset(); state.coarse = null; decodeAndShow(); };
$('btn-save').onclick = () => {
  if (!lastDecode) return;
  // save at the screen's aspect ratio, smoothed, 960 px wide
  const c = document.createElement('canvas');
  c.width = 960; c.height = Math.round((960 * s.screenH) / s.screenW);
  const x = c.getContext('2d');
  x.imageSmoothingEnabled = !s.pixelated;
  x.drawImage(thumb, 0, 0, c.width, c.height);
  c.toBlob((b) => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'live7.png'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); }, 'image/png');
};
$('btn-panel').onclick = () => document.body.classList.toggle('panel-closed');
window.addEventListener('resize', drawPreview);

buildPanel();
setAspect();
thumb.style.imageRendering = s.pixelated ? 'pixelated' : 'auto';
window.live7 = { state, settings: s, get last() { return last; }, get decode() { return lastDecode; }, runFrame };
