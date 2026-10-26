// Headless verification: node test/run.mjs   (needs playwright + chromium, and test/cref/cref built)
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = process.env.OUT || path.join(root, 'test', 'out');
fs.mkdirSync(outDir, { recursive: true });

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  const f = p.endsWith('/') ? path.join(p, 'index.html') : p;
  if (!f.startsWith(root) || !fs.existsSync(f)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' });
  fs.createReadStream(f).pipe(res);
}).listen(0);
const port = server.address().port;
const base = `http://localhost:${port}/`;

// deterministic, natural-ish test image
function testImage(N) {
  const b = new Uint8Array(N * N * 4);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const u = x / N, v = y / N;
    const blob = Math.exp(-((u - 0.35) ** 2 + (v - 0.4) ** 2) * 40);
    const ring = Math.exp(-((Math.hypot(u - 0.65, v - 0.6) - 0.18) ** 2) * 600);
    const r = 40 + 150 * blob + 60 * u + 50 * ring;
    const g = 30 + 80 * v + 120 * ring;
    const bl = 60 + 100 * (1 - u) * v + 40 * Math.sin(10 * u);
    const i = 4 * (y * N + x);
    b[i] = Math.min(255, r); b[i + 1] = Math.min(255, g); b[i + 2] = Math.min(255, bl); b[i + 3] = 255;
  }
  return b;
}

const results = [];
const check = (name, ok, detail) => { results.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const browser = await chromium.launch({
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
});
const ctx = await browser.newContext({ viewport: { width: 1200, height: 800 }, permissions: ['camera'] });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

// ---------------------------------------------------------------- numerical tests (no frame loop)
await page.goto(base + '?noloop');
await page.waitForFunction(() => window.app?.engine);
const fmt = await page.evaluate(() => window.app.engine.fmt.name);
console.log('render target format:', fmt);

// 1. GPU FFT vs JS DFT
const fftErr = await page.evaluate(() => {
  const e = window.app.engine, N = 32;
  e.resize(N);
  const a = new Float32Array(N * N * 4), b = new Float32Array(N * N * 4);
  for (let i = 0; i < a.length; i++) { a[i] = Math.sin(i * 0.37) + 0.3 * Math.cos(i * 1.7); b[i] = (i % 4) < 2 ? Math.cos(i * 0.11) : 0; }
  e.writeField(e.F.specA, a, b);
  e.fft(e.F.specA, e.F.cam, false);
  const [A] = e.readField(e.F.cam);
  // reference for signal 0 (A.xy)
  let err = 0, mx = 0;
  for (let ky = 0; ky < N; ky++) for (let kx = 0; kx < N; kx++) {
    let sr = 0, si = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = 4 * (y * N + x), ang = -2 * Math.PI * (kx * x + ky * y) / N;
      const c = Math.cos(ang), s = Math.sin(ang);
      sr += a[i] * c - a[i + 1] * s; si += a[i] * s + a[i + 1] * c;
    }
    const j = 4 * (ky * N + kx);
    err = Math.max(err, Math.abs(A[j] - sr / N), Math.abs(A[j + 1] - si / N));
    mx = Math.max(mx, Math.hypot(sr, si) / N);
  }
  // inverse round trip
  e.fft(e.F.cam, e.F.g, true);
  const [G] = e.readField(e.F.g);
  let rt = 0;
  for (let i = 0; i < a.length; i++) rt = Math.max(rt, Math.abs(G[i] - a[i]));
  return { rel: err / mx, rt };
});
check('FFT matches DFT (N=32)', fftErr.rel < 1e-5, `max rel err ${fftErr.rel.toExponential(2)}`);
check('IFFT(FFT(x)) = x', fftErr.rt < 1e-4, `max abs err ${fftErr.rt.toExponential(2)}`);

// 2. GPU forward effect vs the actual C (compiled against a naive-DFT FFTW shim)
const N = 64;
const img = testImage(N);
const cOut = execFileSync(path.join(root, 'test/cref/cref'), [String(N)], { input: Buffer.from(img), maxBuffer: 1 << 24 });
const encRes = await page.evaluate(({ N, img }) => {
  const { engine: e, settings: s } = window.app;
  Object.assign(s, { mode: 'roundtrip', view: 'output', shift: false, gH: 50, gS: 25, gV: 50, phase: 'camera', iters: 0 });
  e.setSourcePixels(N, new Uint8Array(img));
  e.render(s, 0);
  return Array.from(e.readRGBA8(e.enc));
}, { N, img: Array.from(img) });
let exact = 0, off1 = 0, worse = 0, worst = 0;
for (let i = 0; i < N * N; i++) {
  let d = 0;
  for (let c = 0; c < 3; c++) d = Math.max(d, Math.abs(encRes[4 * i + c] - cOut[4 * i + c]));
  if (d === 0) exact++; else if (d === 1) off1++; else worse++;
  worst = Math.max(worst, d);
}
check('Encode matches the C bit-for-bit (≥99% exact, rest within ±1 LSB float32 vs double)',
  exact / (N * N) >= 0.99 && worse <= 2, `${exact} exact, ${off1} ±1, ${worse} worse (worst ${worst}) of ${N * N}`);

// 2b. decode: does the true magnitude fall inside the decoded interval?
const dec = await page.evaluate(({ N, img }) => {
  const { engine: e, settings: s } = window.app;
  Object.assign(s, { mode: 'roundtrip', herm: 'intersect', hueBranch: 'fixed', hueK: 0, dcOverride: false, magScale: 0, gH: 50, gS: 25, gV: 50 });
  e.setSourcePixels(N, new Uint8Array(img));
  e.render(s, 0);
  const [CA, CB] = e.readField(e.F.cam);
  const gl = e.gl;
  const rd = (t) => { const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    const b = new Float32Array(N * N * 4); gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, b); return b; };
  const lo = rd(e.mag.lo), hi = rd(e.mag.hi);
  const tol = 2e-3; let inV = 0, inS = 0, inH = 0, nH = 0;
  for (let k = 0; k < N * N; k++) {
    const LH = Math.log(1 + Math.hypot(CA[4 * k], CA[4 * k + 1]) * N);
    const LS = Math.log(1 + Math.hypot(CA[4 * k + 2], CA[4 * k + 3]) * N);
    const LV = Math.log(1 + Math.hypot(CB[4 * k], CB[4 * k + 1]) * N);
    const inside = (L, l, h) => L >= l - tol && L <= Math.abs(h) + tol;
    if (inside(LV, lo[4 * k + 2], hi[4 * k + 2])) inV++;
    if (inside(LS, lo[4 * k + 1], hi[4 * k + 1])) inS++;
    if (hi[4 * k + 3] >= 0) {
      nH++;
      const hd = (LH * s.gH) % 360, d = Math.abs(((hd - hi[4 * k + 3] + 540) % 360) - 180);
      if (d <= lo[4 * k + 3] * s.gH + 0.5) inH++;
    }
  }
  return { V: inV / (N * N), S: inS / (N * N), H: inH / Math.max(1, nH) };
}, { N, img: Array.from(img) });
check('Decode intervals contain the true |F|', dec.V > 0.99 && dec.S > 0.97 && dec.H > 0.9,
  `V ${(100 * dec.V).toFixed(1)}%  S ${(100 * dec.S).toFixed(1)}%  H(mod 360) ${(100 * dec.H).toFixed(1)}%`);

// 3. Round trip: C effect -> decode -> inverse, compared channel by channel with the source HSV.
//    Information is genuinely lost (8-bit log magnitudes; S and H are pinned only as well as V is
//    bright), so only V is held to a tight bound. S and H errors are reported.
async function roundTrip(over) {
  return page.evaluate(({ N, img, over }) => {
    const { engine: e, settings: s } = window.app;
    Object.assign(s, {
      mode: 'roundtrip', view: 'output', phase: 'camera', iters: 0, herm: 'intersect', bias: 0.5,
      magScale: 0, dcOverride: false, hueBranch: 'vmatch', hueK: 0, shift: false, hueBound: 'wrap',
      jitter: 0, drift: 0, lockPhase: false, strength: 1, slack: 1, clipMode: 'free', hueSolve: true,
      support: 'full', algo: 'hio', beta: 0.9,
    }, over);
    e.setSourcePixels(N, new Uint8Array(img));
    e.resetSolver();
    for (let i = 0; i < (over.frames || 1); i++) e.render(s, i / 30);
    const out = (s.iters > 0 || s.phase === 'warm') ? e.F.gPrime : e.F.g;
    const [GA, GB] = e.readField(out);
    e.draw('tohsv', e.F.specB, { uTex: e.src.tex });
    const [TA, TB] = e.readField(e.F.specB);
    let eh = 0, es = 0, ev = 0, bad = 0;
    const cl = (x) => Math.min(1, Math.max(0, x));
    for (let i = 0; i < N * N; i++) {
      const vals = [GA[4 * i], GA[4 * i + 2], GB[4 * i]];
      if (!vals.every(Number.isFinite)) { bad++; continue; }
      const dh = Math.abs((((GA[4 * i] - TA[4 * i]) % 1) + 1.5) % 1 - 0.5);
      eh += dh * dh; es += (cl(GA[4 * i + 2]) - TA[4 * i + 2]) ** 2; ev += (cl(GB[4 * i]) - TB[4 * i]) ** 2;
    }
    const n = N * N;
    return { H: Math.sqrt(eh / n), S: Math.sqrt(es / n), V: Math.sqrt(ev / n), bad };
  }, { N, img: Array.from(img), over });
}
const f = (r) => `RMSE  V ${r.V.toFixed(3)}  S ${r.S.toFixed(3)}  H ${r.H.toFixed(3)} (turns)  non-finite ${r.bad}`;
const rtClean = await roundTrip({ gH: 20, gS: 10, gV: 20 });
check('Round trip, oracle phase, gains below clipping: V recovered', rtClean.V < 0.03 && rtClean.bad === 0, f(rtClean));
const rtC = await roundTrip({ gH: 50, gS: 25, gV: 50 });
check('Round trip, oracle phase, C constants: finite', rtC.bad === 0, f(rtC));
const rtCi = await roundTrip({ gH: 50, gS: 25, gV: 50, iters: 10, algo: 'er' });
check('  + 10 ER iterations (interval projection) does not diverge', rtCi.bad === 0 && rtCi.V < rtC.V * 1.5 + 0.02, f(rtCi));
const rtHio = await roundTrip({ gH: 50, gS: 25, gV: 50, phase: 'warm', iters: 5, frames: 20 });
check('Blind HIO, 100 iterations: finite', rtHio.bad === 0, `${f(rtHio)}  (no phase information: not expected to match)`);

// ---------------------------------------------------------------- live app with the fake camera
await page.goto(base);
await page.waitForFunction(() => window.app?.engine);
await page.click('#btn-start');
await page.waitForTimeout(2500);
await page.screenshot({ path: path.join(outDir, 'app-camera-phase-swap.png') });
const presets = ['Warm HIO', 'Round trip — oracle phase', 'Round trip — blind HIO', 'Noise bloom', 'C effect only'];
for (const p of presets) {
  await page.selectOption('.preset', p);
  await page.waitForTimeout(2000);
  await page.screenshot({ path: path.join(outDir, `app-${p.replace(/[^a-z]+/gi, '-').toLowerCase()}.png`) });
}
const statusText = await page.textContent('#status');
console.log('status:', statusText);

// mobile layout
const mob = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, permissions: ['camera'] });
const mp = await mob.newPage();
mp.on('pageerror', (e) => errors.push(String(e)));
await mp.goto(base);
await mp.waitForTimeout(1500);
await mp.screenshot({ path: path.join(outDir, 'mobile-start.png') });
await mp.click('#btn-start');
await mp.waitForTimeout(2500);
await mp.screenshot({ path: path.join(outDir, 'mobile-running.png') });

check('No page errors', errors.length === 0, errors.join(' | ') || 'none');
await browser.close();
server.close();
const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} passed`);
process.exit(failed ? 1 : 0);
