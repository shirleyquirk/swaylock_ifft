// tryseven live, the simple way: the camera rectangle you line up IS the tryseven image.
// Sample it into a K x Kh grid (one cell per coefficient), uncompand, inverse DCT on the GPU, show.

// ---------------------------------------------------------------- shaders
const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const float PI = 3.14159265358979;
`;

// camera rectangle -> per-cell bytes (raw) and coefficients (scaled by 1/(4WH))
const SAMPLE = HEAD + `
uniform sampler2D uCam;
uniform vec2 uCamSize, uCenter, uExtent;   // camera px: rectangle centre and full width/height
uniform float uRot;
uniform int uK, uKh, uTaps, uMirror;
uniform vec3 uCentre, uGain;
uniform float uGamma, uKappa, uNorm, uScale, uTaper;
layout(location = 0) out vec4 oCoef;
layout(location = 1) out vec4 oRaw;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);              // texel row 0 = coefficient row 0
  float cs = cos(uRot), sn = sin(uRot);
  vec3 acc = vec3(0.0);
  for (int j = 0; j < uTaps; j++) for (int i = 0; i < uTaps; i++) {
    vec2 g = (vec2(p) + (vec2(i, j) + 0.5) / float(uTaps)) / vec2(uK, uKh);
    if (uMirror == 1) g.x = 1.0 - g.x;
    vec2 d = (g - 0.5) * uExtent;
    vec2 q = uCenter + vec2(cs * d.x - sn * d.y, sn * d.x + cs * d.y);
    acc += texture(uCam, q / uCamSize).rgb;
  }
  vec3 rgb = acc / float(uTaps * uTaps);
  vec3 e = clamp((rgb * 255.0 - uCentre) * uGain / 127.5, -1.0, 1.0);
  e = sign(e) * pow(abs(e), vec3(uGamma));
  vec3 c = sign(e) * (exp(abs(e) * uNorm * uKappa) - 1.0) * uScale;
  float ta = mix(1.0, 0.5 + 0.5 * cos(PI * float(p.x) / float(uK)), uTaper);
  float tb = mix(1.0, 0.5 + 0.5 * cos(PI * float(p.y) / float(uKh)), uTaper);
  oCoef = vec4(c * ta * tb, 1.0);
  oRaw = vec4(rgb, 1.0);
}`;

// DCT-III (FFTW REDFT01 convention: x0 + 2 sum) along one axis, by direct sum
const IDCT = HEAD + `
uniform sampler2D uIn;
uniform int uN, uAxis;      // uAxis 0: along x, 1: along y
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int n = uAxis == 0 ? p.x : p.y;
  int m4 = 4 * uN, step = 2 * n + 1, m = 0;
  vec3 s = vec3(0.0);
  for (int k = 0; k < uN; k++) {
    ivec2 t = uAxis == 0 ? ivec2(k, p.y) : ivec2(p.x, k);
    float w = k == 0 ? 1.0 : 2.0;
    s += w * cos(PI * float(m) / float(2 * uN)) * texelFetch(uIn, t, 0).rgb;
    m += step; if (m >= m4) m -= m4;           // k (2n+1) mod 4N, exact integers
    if (m >= m4) m -= m4;
  }
  o = vec4(s, 1.0);
}`;

const SHOW = HEAD + `
uniform sampler2D uImg, uRaw;
uniform int uKh, uView;
uniform float uSplit, uChroma, uContrast, uBright, uLo, uHi;
out vec4 o;
void main() {
  ivec2 s = ivec2(gl_FragCoord.xy);
  ivec2 p = ivec2(s.x, uKh - 1 - s.y);
  int view = uView;
  if (view == 2) view = float(s.x) < uSplit ? 1 : 0;
  vec3 c;
  if (view == 1) c = texelFetch(uRaw, p, 0).rgb;
  else {
    vec3 ycc = texelFetch(uImg, p, 0).rgb;
    float sc = 1.0 / max(uHi - uLo, 1e-6);
    float y = (ycc.x - uLo) * sc, cb = ycc.y * uChroma * sc, cr = ycc.z * uChroma * sc;
    c = vec3(y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb);
    c = (c - 0.5) * uContrast + 0.5 + uBright;
  }
  if (uView == 2 && abs(float(s.x) - uSplit) < 0.5) c = vec3(1.0);
  o = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

// ---------------------------------------------------------------- settings
const SCHEMA = [
  ['Line up', [
    { k: 'K', l: 'Coefficients across (grid width)', t: 'rng', d: 128, min: 16, max: 512, s: 1,
      h: 'The rectangle is cut into this many cells across; each cell is read as one screen pixel of the corner.' },
    { k: 'cw', l: 'Rectangle width (camera px)', t: 'rng', d: 600, min: 16, max: 4000, s: 1 },
    { k: 'cx', l: 'Rectangle centre x', t: 'rng', d: 640, min: 0, max: 4000, s: 1 },
    { k: 'cy', l: 'Rectangle centre y', t: 'rng', d: 360, min: 0, max: 4000, s: 1 },
    { k: 'rot', l: 'Rotation (°)', t: 'rng', d: 0, min: -45, max: 45, s: 0.1 },
    { k: 'taps', l: 'Samples per cell', t: 'rng', d: 2, min: 1, max: 4, s: 1 },
    { k: 'mirror', l: 'Mirror', t: 'chk', d: false },
    { k: 'grid', l: 'Show grid on camera', t: 'chk', d: true },
  ]],
  ['Screen', [
    { k: 'screenW', l: 'Locked screen width (px)', t: 'num', d: 1920, min: 16, max: 8192, h: 'Sets the log scale: log(1 + 4·W·H).' },
    { k: 'screenH', l: 'Locked screen height (px)', t: 'num', d: 1080, min: 16, max: 8192 },
  ]],
  ['Colour → coefficient', [
    { k: 'autoCentre', l: 'Auto zero level (median of the cells)', t: 'chk', d: true,
      h: 'tryseven bytes are symmetric about 127.5 away from DC, so the median camera value is the zero.' },
    { k: 'centre', l: 'Zero level (byte) / trim when auto', t: 'rng', d: 127.5, min: 60, max: 200, s: 0.5 },
    { k: 'autoGain', l: 'Auto gain (spread)', t: 'chk', d: true,
      h: 'Scales so the typical distance from zero is the spread below (~48 for a 1080p corner).' },
    { k: 'spread', l: 'Target spread (bytes)', t: 'rng', d: 48, min: 5, max: 100, s: 0.5 },
    { k: 'gain', l: 'Gain (× auto)', t: 'rng', d: 1, min: 0.2, max: 4, s: 0.01 },
    { k: 'gamma', l: 'Gamma', t: 'rng', d: 1, min: 0.3, max: 3, s: 0.01 },
    { k: 'kappa', l: 'Log softening κ (1 = exact)', t: 'rng', d: 0.75, min: 0.3, max: 1.2, s: 0.01 },
    { k: 'taper', l: 'Taper', t: 'rng', d: 0, min: 0, max: 1, s: 0.01 },
  ]],
  ['Output', [
    { k: 'view', l: 'View', t: 'sel', d: 2, num: true, o: [[0, 'decoded'], [1, 'sampled'], [2, 'split: sampled | decoded']] },
    { k: 'split', l: 'Split position', t: 'rng', d: 0.5, min: 0, max: 1, s: 0.01 },
    { k: 'autoLevels', l: 'Auto levels', t: 'chk', d: true },
    { k: 'contrast', l: 'Contrast', t: 'rng', d: 1, min: 0.1, max: 20, s: 0.05 },
    { k: 'bright', l: 'Brightness', t: 'rng', d: 0, min: -1, max: 1, s: 0.01 },
    { k: 'chroma', l: 'Colour', t: 'rng', d: 1, min: 0, max: 4, s: 0.01 },
    { k: 'smooth', l: 'Smooth', t: 'chk', d: true },
  ]],
  ['Camera', [
    { k: 'facing', l: 'Camera', t: 'sel', d: 'environment', o: [['environment', 'back'], ['user', 'front']] },
    { k: 'res', l: 'Resolution', t: 'sel', d: '1920x1080', o: [['1280x720', '720p'], ['1920x1080', '1080p'], ['3840x2160', '4K']] },
    { k: 'zoom', l: 'Camera zoom (if supported)', t: 'rng', d: 1, min: 1, max: 10, s: 0.1 },
  ]],
];
const ALL = SCHEMA.flatMap(([, g]) => g);
const KEY = 'swaylock-ifft.live7.simple.v1';
const s = Object.fromEntries(ALL.map((c) => [c.k, c.d]));
try { Object.assign(s, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) { /* storage blocked */ }
let saveT;
const save = () => { clearTimeout(saveT); saveT = setTimeout(() => { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) { /* blocked */ } }, 200); };
const inputs = {};

// ---------------------------------------------------------------- GL
const $ = (id) => document.getElementById(id);
const out = $('out'), video = $('video'), ov = $('ov'), octx = ov.getContext('2d');
const gl = out.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: true });
if (!gl) { $('msg').textContent = 'WebGL2 is not available in this browser.'; throw new Error('no webgl2'); }
const fmt = gl.getExtension('EXT_color_buffer_float') ? { i: gl.RGBA32F, t: gl.FLOAT } : (gl.getExtension('EXT_color_buffer_half_float'), { i: gl.RGBA16F, t: gl.HALF_FLOAT });
function prog(fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, VERT], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    gl.attachShader(p, sh);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = { l: gl.getUniformLocation(p, a.name), t: a.type }; }
  return { p, u };
}
const P = { sample: prog(SAMPLE), idct: prog(IDCT), show: prog(SHOW) };
const vao = gl.createVertexArray();
function tex(w, h, internal, type, linear) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, gl.RGBA, type, null);
  const f = linear ? gl.LINEAR : gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}
function fbo(texs) {
  const f = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, f);
  texs.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
  gl.drawBuffers(texs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
  return f;
}
function draw(pr, target, w, h, uni) {
  gl.useProgram(pr.p);
  let unit = 0;
  for (const [k, v] of Object.entries(uni)) {
    const u = pr.u[k];
    if (!u) continue;
    if (u.t === gl.SAMPLER_2D) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, v); gl.uniform1i(u.l, unit++); }
    else if (u.t === gl.INT) gl.uniform1i(u.l, v | 0);
    else if (u.t === gl.FLOAT) gl.uniform1f(u.l, v);
    else if (u.t === gl.FLOAT_VEC2) gl.uniform2fv(u.l, v);
    else if (u.t === gl.FLOAT_VEC3) gl.uniform3fv(u.l, v);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, target);
  gl.viewport(0, 0, w, h);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

const camTex = tex(1, 1, gl.RGBA8, gl.UNSIGNED_BYTE, true);
let srcW = 0, srcH = 0, grid = null;
function ensureGrid(K, Kh) {
  if (grid && grid.K === K && grid.Kh === Kh) return;
  if (grid) { [grid.coef, grid.raw, grid.rows, grid.img].forEach((t) => gl.deleteTexture(t)); [grid.f1, grid.f2, grid.f3].forEach((f) => gl.deleteFramebuffer(f)); }
  const coef = tex(K, Kh, fmt.i, fmt.t), raw = tex(K, Kh, gl.RGBA8, gl.UNSIGNED_BYTE), rows = tex(K, Kh, fmt.i, fmt.t), img = tex(K, Kh, fmt.i, fmt.t);
  grid = { K, Kh, coef, raw, rows, img, f1: fbo([coef, raw]), f2: fbo([rows]), f3: fbo([img]) };
  out.width = K; out.height = Kh;
}

// ---------------------------------------------------------------- source
let stream = null, track = null, still = null, frozen = false;
async function startCamera() {
  stop(); still = null;
  if (!navigator.mediaDevices?.getUserMedia) { say('No camera API here: needs HTTPS or localhost.'); return; }
  const [w, h] = s.res.split('x').map(Number);
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: s.facing }, width: { ideal: w }, height: { ideal: h } } });
    track = stream.getVideoTracks()[0];
    video.hidden = false; $('stillview').hidden = true;
    if (!s.autoCentre && !s.autoGain && still === null && startCamera.fromImage) { setVal('autoCentre', true); setVal('autoGain', true); }
    video.srcObject = stream;
    await video.play();
    applyZoom();
    $('start').hidden = true;
    say('');
  } catch (e) { say(`Camera: ${e.name || ''} ${e.message || e}`); }
}
function stop() { if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; track = null; }
async function applyZoom() {
  const cap = track?.getCapabilities?.();
  if (!cap?.zoom) return;
  try { await track.applyConstraints({ advanced: [{ zoom: Math.max(cap.zoom.min, Math.min(cap.zoom.max, s.zoom)) }] }); } catch (e) { /* unsupported */ }
}
async function loadImage(f) {
  stop();
  startCamera.fromImage = true;
  const bmp = await createImageBitmap(f, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  still = bmp;
  video.srcObject = null;
  video.hidden = true;
  const sv = $('stillview');
  sv.hidden = false; sv.dataset.src = '';
  upload(bmp, bmp.width, bmp.height);
  $('start').hidden = true;
  if (frame.exactFit) fitExact(bmp.width, bmp.height);
}
// frame an image exactly: one cell per image pixel from its top-left corner (a tryseven file decodes exactly)
function fitExact(w, h) {
  setVal('screenW', w); setVal('screenH', h);
  setVal('K', Math.min(512, w));
  const Kh = Math.max(4, Math.round((s.K * s.screenH) / s.screenW));
  setVal('cw', s.K); setVal('cx', s.K / 2); setVal('cy', Kh / 2); setVal('rot', 0);
  setVal('autoCentre', false); setVal('autoGain', false); setVal('centre', 127.5); setVal('gain', 1); setVal('taps', 1); setVal('kappa', 1);
}
function upload(el, w, h) {
  gl.bindTexture(gl.TEXTURE_2D, camTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, el);
  if (w !== srcW || h !== srcH) {
    const first = !srcW;
    srcW = w; srcH = h;
    for (const k of ['cw', 'cx', 'cy']) { inputs[k].max = Math.max(w, h); }
    if (first && !still) { setVal('cx', Math.min(s.cx, w)); setVal('cy', Math.min(s.cy, h)); }
  }
}

// ---------------------------------------------------------------- frame
let fps = 0, tPrev = performance.now();
function frame() {
  requestAnimationFrame(frame);
  if (!still && !(video.readyState >= 2 && video.videoWidth)) return;
  if (!still && !frozen) upload(video, video.videoWidth, video.videoHeight);
  const K = s.K | 0, Kh = Math.max(4, Math.round((K * s.screenH) / s.screenW));
  ensureGrid(K, Kh);
  const extent = [s.cw, (s.cw * Kh) / K];
  const cal = calibrate(K, Kh);
  draw(P.sample, grid.f1, K, Kh, {
    uCam: camTex, uCamSize: [srcW, srcH], uCenter: [s.cx, s.cy], uExtent: extent, uRot: (s.rot * Math.PI) / 180,
    uK: K, uKh: Kh, uTaps: s.taps, uMirror: s.mirror ? 1 : 0,
    uCentre: cal.centre, uGain: cal.gain, uGamma: s.gamma, uKappa: s.kappa,
    uNorm: Math.log1p(4 * s.screenW * s.screenH), uScale: 1 / (4 * s.screenW * s.screenH), uTaper: s.taper,
  });
  draw(P.idct, grid.f2, K, Kh, { uIn: grid.coef, uN: K, uAxis: 0 });
  draw(P.idct, grid.f3, K, Kh, { uIn: grid.rows, uN: Kh, uAxis: 1 });
  const lv = levels(K, Kh);
  draw(P.show, null, K, Kh, { uImg: grid.img, uRaw: grid.raw, uKh: Kh, uView: s.view, uSplit: s.split * K, uChroma: s.chroma, uContrast: s.contrast, uBright: s.bright, uLo: lv[0], uHi: lv[1] });
  overlay(Kh);
  const now = performance.now();
  fps = 0.9 * fps + 0.1 * (1000 / Math.max(1, now - tPrev)); tPrev = now;
  $('status').textContent = `${fps.toFixed(0)} fps · ${K}×${Kh} coefficients · ${(s.cw / K).toFixed(1)} camera px per cell · zero ${cal.centre.map((v) => v.toFixed(0)).join('/')} gain ${cal.gain[0].toFixed(2)} · source ${srcW}×${srcH}`;
}

// auto zero / gain from the previous frame's sampled cells (tiny readback, every 3rd frame, smoothed)
const auto = { c: [127.5, 127.5, 127.5], g: 1, n: 0 };
function calibrate(K, Kh) {
  if ((s.autoCentre || s.autoGain) && grid && auto.n++ % 3 === 0) {
    const px = new Uint8Array(K * Kh * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, grid.f1);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, K, Kh, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    const ch = [[], [], []];
    for (let b = 0; b < Kh; b++) for (let a = 0; a < K; a++) if (a + b >= 3) { const o = 4 * (b * K + a); ch[0].push(px[o]); ch[1].push(px[o + 1]); ch[2].push(px[o + 2]); }
    const med = (v) => { v.sort((x, y) => x - y); return v[v.length >> 1]; };
    const c = ch.map((v) => med(v.slice()));
    const spreads = ch.map((v, i) => med(v.map((x) => Math.abs(x - c[i]))));
    const sp = (spreads[0] + spreads[1] + spreads[2]) / 3;
    const g = sp > 0.5 ? s.spread / sp : auto.g;
    const k = auto.n < 4 ? 1 : 0.3;
    auto.c = auto.c.map((v, i) => v + k * (c[i] - v));
    auto.g += k * (g - auto.g);
  }
  const centre = s.autoCentre ? auto.c.map((v) => v + (s.centre - 127.5)) : [s.centre, s.centre, s.centre];
  const gain = (s.autoGain ? auto.g : 1) * s.gain;
  return { centre, gain: [gain, gain, gain] };
}

// auto levels: 2nd..98th percentile of decoded luma, read back every 3rd frame, smoothed
const lev = { lo: 0, hi: 1, n: 0 };
function levels(K, Kh) {
  if (!s.autoLevels) return [0, 1];
  if (lev.n++ % 3 === 0 && fmt.t === gl.FLOAT) {
    const px = new Float32Array(K * Kh * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, grid.f3);
    gl.readPixels(0, 0, K, Kh, gl.RGBA, gl.FLOAT, px);
    const y = new Float32Array(K * Kh);
    for (let i = 0; i < K * Kh; i++) y[i] = px[4 * i];
    y.sort();
    const lo = y[Math.floor(K * Kh * 0.02)], hi = y[Math.floor(K * Kh * 0.98)];
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
      const k = lev.n < 4 ? 1 : 0.3;
      lev.lo += k * (lo - lev.lo); lev.hi += k * (hi - lev.hi);
    }
  }
  return [lev.lo, lev.hi];
}

// rectangle + grid over the camera preview
function shownRect() {
  const el = still ? $('stillview') : video;
  const r = el.getBoundingClientRect(), pr = ov.parentElement.getBoundingClientRect();
  const sc = Math.min(r.width / srcW, r.height / srcH);
  return { sc, x0: r.left - pr.left + (r.width - srcW * sc) / 2, y0: r.top - pr.top + (r.height - srcH * sc) / 2 };
}
function overlay(Kh) {
  const pr = ov.parentElement.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
  if (ov.width !== Math.round(pr.width * dpr) || ov.height !== Math.round(pr.height * dpr)) { ov.width = Math.round(pr.width * dpr); ov.height = Math.round(pr.height * dpr); }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, pr.width, pr.height);
  if (!srcW) return;
  if (still) { const sv = $('stillview'); if (sv.dataset.src !== 'set') { sv.width = srcW; sv.height = srcH; sv.getContext('2d').drawImage(still, 0, 0); sv.dataset.src = 'set'; } }
  const { sc, x0, y0 } = shownRect();
  const K = s.K, cs = Math.cos((s.rot * Math.PI) / 180), sn = Math.sin((s.rot * Math.PI) / 180);
  const W = s.cw, H = (s.cw * Kh) / K;
  const pt = (gx, gy) => { if (s.mirror) gx = 1 - gx; const dx = (gx - 0.5) * W, dy = (gy - 0.5) * H; return [x0 + (s.cx + cs * dx - sn * dy) * sc, y0 + (s.cy + sn * dx + cs * dy) * sc]; };
  const line = (a, b) => { octx.beginPath(); octx.moveTo(...pt(...a)); octx.lineTo(...pt(...b)); octx.stroke(); };
  if (s.grid) {
    octx.strokeStyle = 'rgba(79,209,197,0.35)'; octx.lineWidth = 1;
    const step = Math.max(1, Math.round(K / 16));
    for (let a = step; a < K; a += step) line([a / K, 0], [a / K, 1]);
    for (let b = step; b < Kh; b += step) line([0, b / Kh], [1, b / Kh]);
  }
  octx.strokeStyle = '#ffb547'; octx.lineWidth = 2;
  octx.beginPath(); [[0, 0], [1, 0], [1, 1], [0, 1]].forEach((c, i) => (i ? octx.lineTo(...pt(...c)) : octx.moveTo(...pt(...c)))); octx.closePath(); octx.stroke();
  // cell (0,0) = the screen's top-left pixel goes here
  octx.fillStyle = '#ff4d4d';
  const [cx0, cy0] = pt(0.5 / K, 0.5 / Kh);
  octx.beginPath(); octx.arc(cx0, cy0, 5, 0, 2 * Math.PI); octx.fill();
}

// drag to move, pinch / wheel to resize
const pts = new Map();
let gest = null;
ov.addEventListener('pointerdown', (e) => { ov.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); gest = null; });
ov.addEventListener('pointerup', (e) => { pts.delete(e.pointerId); gest = null; });
ov.addEventListener('pointercancel', (e) => { pts.delete(e.pointerId); gest = null; });
ov.addEventListener('pointermove', (e) => {
  if (!pts.has(e.pointerId) || !srcW) return;
  const prev = pts.get(e.pointerId);
  pts.set(e.pointerId, [e.clientX, e.clientY]);
  const { sc } = shownRect();
  if (pts.size === 1) { setVal('cx', s.cx + (e.clientX - prev[0]) / sc); setVal('cy', s.cy + (e.clientY - prev[1]) / sc); }
  else if (pts.size === 2) {
    const [a, b] = [...pts.values()];
    const d = Math.hypot(a[0] - b[0], a[1] - b[1]), ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    if (gest) { setVal('cw', s.cw * (d / gest.d)); setVal('rot', s.rot + ((ang - gest.a) * 180) / Math.PI); }
    gest = { d, a: ang };
  }
});
ov.addEventListener('wheel', (e) => { e.preventDefault(); setVal('cw', s.cw * Math.exp(e.deltaY * 0.001)); }, { passive: false });

// ---------------------------------------------------------------- panel
function setVal(k, v) {
  const c = ALL.find((x) => x.k === k);
  if (c.t === 'rng' || c.t === 'num') v = Math.max(c.min, Math.min(Number(inputs[k]?.max ?? c.max), v));
  if (c.s >= 1 && c.t === 'rng') v = Math.round(v / c.s) * c.s;
  s[k] = v;
  const inp = inputs[k];
  if (inp) { if (c.t === 'chk') inp.checked = v; else inp.value = v; inp._out && (inp._out.textContent = fmtv(c, v)); }
  save();
}
const fmtv = (c, v) => (c.s >= 1 ? String(Math.round(v)) : (+v).toFixed(c.s < 0.1 ? 2 : 1));
function buildPanel() {
  const root = $('panel');
  root.innerHTML = '';
  const top = document.createElement('div'); top.className = 'panel-top';
  const b = (t, f) => { const x = document.createElement('button'); x.type = 'button'; x.textContent = t; x.onclick = f; return x; };
  top.append(b('Defaults', () => { for (const c of ALL) s[c.k] = c.d; save(); buildPanel(); }), b('Help', () => root.classList.toggle('show-help')));
  root.append(top);
  SCHEMA.forEach(([name, ctrls], gi) => {
    const det = document.createElement('details'); det.className = 'group'; det.open = gi < 3;
    const sum = document.createElement('summary'); sum.textContent = name; det.append(sum);
    for (const c of ctrls) {
      const row = document.createElement('div'); row.className = 'row' + (c.t === 'chk' ? ' check' : '');
      const lab = document.createElement('label'); const nm = document.createElement('span'); nm.textContent = c.l; lab.append(nm);
      let inp;
      if (c.t === 'rng') {
        const o = document.createElement('output'); o.textContent = fmtv(c, s[c.k]); lab.append(o);
        inp = document.createElement('input'); Object.assign(inp, { type: 'range', min: c.min, max: c.max, step: c.s, value: s[c.k] }); inp._out = o;
        inp.oninput = () => { s[c.k] = +inp.value; o.textContent = fmtv(c, s[c.k]); save(); changed(c.k); };
        nm.ondblclick = () => { setVal(c.k, c.d); changed(c.k); };
        row.append(lab, inp);
      } else if (c.t === 'num') {
        inp = document.createElement('input'); Object.assign(inp, { type: 'number', min: c.min, max: c.max, value: s[c.k], inputMode: 'numeric' }); inp.className = 'num';
        inp.onchange = () => { setVal(c.k, Math.round(+inp.value) || c.d); changed(c.k); };
        row.append(lab, inp);
      } else if (c.t === 'sel') {
        inp = document.createElement('select');
        for (const [v, t] of c.o) inp.append(new Option(t, String(v)));
        inp.value = String(s[c.k]);
        inp.onchange = () => { s[c.k] = c.num ? +inp.value : inp.value; save(); changed(c.k); };
        row.append(lab, inp);
      } else {
        inp = document.createElement('input'); inp.type = 'checkbox'; inp.checked = !!s[c.k];
        inp.onchange = () => { s[c.k] = inp.checked; save(); changed(c.k); };
        lab.prepend(inp); row.append(lab);
      }
      inputs[c.k] = inp;
      if (c.h) { const hp = document.createElement('small'); hp.className = 'help'; hp.textContent = c.h; row.append(hp); }
      det.append(row);
    }
    root.append(det);
  });
  if (srcW) for (const k of ['cw', 'cx', 'cy']) inputs[k].max = Math.max(srcW, srcH);
}
function changed(k) {
  if ((k === 'facing' || k === 'res') && stream) startCamera();
  if (k === 'zoom') applyZoom();
  if (k === 'smooth') out.style.imageRendering = s.smooth ? 'auto' : 'pixelated';
}
function say(t) { $('msg').textContent = t; $('msg').hidden = !t; }

$('btn-camera').onclick = startCamera;
$('btn-photo').onclick = () => $('file').click();
$('file').onchange = () => { const f = $('file').files?.[0]; if (f) loadImage(f).catch((e) => say(String(e.message || e))); };
$('btn-freeze').onclick = (e) => { frozen = !frozen; e.currentTarget.classList.toggle('on', frozen); };
$('btn-save').onclick = () => out.toBlob((b) => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'tryseven-live.png'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); }, 'image/png');
$('btn-panel').onclick = () => document.body.classList.toggle('panel-closed');

buildPanel();
out.style.imageRendering = s.smooth ? 'auto' : 'pixelated';
requestAnimationFrame(frame);
frame.exactFit = true;
window.live7 = { settings: s, setVal, gl, get grid() { return grid; }, loadImage, fitExact };
