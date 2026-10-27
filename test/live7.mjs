// node test/live7.mjs <screen7.rgba> <screen.rgba> W H <simdir> [optsJSON]
// Runs the live decoder core on synthetic camera frames and scores it against the exact screen bytes.
import fs from 'node:fs';
import { processFrame, decodeThumb, CoefBuffer, gridU } from '../js/live7core.js';

const [enc7, orig, Ws, Hs, dir, extra] = process.argv.slice(2);
const W = +Ws, H = +Hs;
const meta = JSON.parse(fs.readFileSync(`${dir}/meta.json`));
const scr = new Uint8Array(fs.readFileSync(enc7));
const src = new Uint8Array(fs.readFileSync(orig));
const opt = {
  screenW: W, screenH: H, rot: 0, q: 0.3, autoPhase: true, autoOrigin: true, accumulate: true, alpha: 0.5,
  autoCentre: true, autoSlope: true, dcTarget: 244, gain: 1, gamma: 1, crosstalk: 0, taper: 0.3,
  maxK: 384, autoLevels: false, chroma: 1, ...(extra ? JSON.parse(extra) : {}),
};
const state = { buf: new CoefBuffer(512) };
let last;
for (let f = 0; f < meta.frames.length; f++) {
  const fr = new Uint8Array(fs.readFileSync(`${dir}/frame_${String(f).padStart(2, '0')}.rgba`));
  last = processFrame(fr, meta.cw, meta.ch, state, opt);
  console.log(`frame ${f}: ${last.ok ? 'ok' : last.msg}  pitch ${last.pitch?.toFixed(3)} (true ${meta.mag})  angle ${last.angle?.toFixed(2)}° (true ${meta.frames[f].rot.toFixed(2)})  `
    + `rms ${last.ga?.rms.toFixed(3)}/${last.gb?.rms.toFixed(3)}  patches ${last.ga?.patches}/${last.ga?.of}  off ${last.offA?.map((v) => v.toFixed(2))}/${last.offB?.map((v) => v.toFixed(2))} q ${last.qA?.map((v) => v.toFixed(2))}/${last.qB?.map((v) => v.toFixed(2))}  cells ${last.added}  ${last.ms?.toFixed(0)} ms`);
}

// registration: correlate buffer against exact bytes at shifts
const buf = state.buf, M = buf.max;
const [ka, kb] = buf.extent();
function corrAt(dx, dy) {
  let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let b = 0; b < kb; b++) for (let a = 0; a < ka; a++) {
    const k = b * M + a;
    if (!(buf.wt[k] > 0)) continue;
    const ta = a + dx, tb = b + dy;
    if (ta < 0 || tb < 0 || ta >= W || tb >= H) continue;
    const x = buf.val[3 * k], y = scr[4 * (tb * W + ta)];       // R channel = Y coefficient byte
    n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
  }
  return (sxy / n - (sx / n) * (sy / n)) / Math.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2));
}
let best = { c: -2 };
for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const c = corrAt(dx, dy); if (c > best.c) best = { c, dx, dy }; }
console.log(`coefficient window ${ka}x${kb}; correlation with true bytes at origin ${corrAt(0, 0).toFixed(3)}; best shift (${best.dx},${best.dy}) ${best.c.toFixed(3)}`);

// thumbnails: camera vs ideal (exact bytes, same K x Kh)
const cam = decodeThumb(buf, opt);
const ideal = new CoefBuffer(M);
for (let b = 0; b < cam.Kh; b++) for (let a = 0; a < cam.K; a++) {
  const k = b * M + a, p = 4 * (b * W + a);
  ideal.wt[k] = 1; ideal.val[3 * k] = scr[p]; ideal.val[3 * k + 1] = scr[p + 1]; ideal.val[3 * k + 2] = scr[p + 2];
}
const idl = decodeThumb(ideal, { ...opt, autoCentre: false, autoSlope: false, gain: 1, gamma: 1, crosstalk: 0, chGain: [1, 1, 1], autoLevels: false, logScale: 1, lowTrust: 1, calib: "linear" });
// box-downsampled original at the same K x Kh
const K = cam.K, Kh = cam.Kh, box = new Float64Array(K * Kh * 3);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const o = 3 * (Math.min(Kh - 1, Math.floor((y * Kh) / H)) * K + Math.min(K - 1, Math.floor((x * K) / W)));
  for (let c = 0; c < 3; c++) box[o + c] += src[4 * (y * W + x) + c] / ((W / K) * (H / Kh));
}
const psnr = (a, b) => { let se = 0, n = 0; for (let i = 0; i < K * Kh; i++) for (let c = 0; c < 3; c++) { se += (a[4 * i + c] - b(i, c)) ** 2; n++; } return 10 * Math.log10(65025 / (se / n)); };
console.log(`thumbnail ${K}x${Kh}  calib slope ${cam.slope.toFixed(3)}${cam.slopeOk ? '' : ' (fallback)'}  centres ${cam.med.map((v) => v.toFixed(1)).join(',')}  coverage ${(100 * cam.coverage).toFixed(0)}%`);
console.log(`  ideal (exact bytes) vs box-downsampled source : ${psnr(idl.rgba, (i, c) => box[3 * i + c]).toFixed(1)} dB`);
console.log(`  camera decode vs ideal                       : ${psnr(cam.rgba, (i, c) => idl.rgba[4 * i + c]).toFixed(1)} dB`);
console.log(`  camera decode vs box-downsampled source      : ${psnr(cam.rgba, (i, c) => box[3 * i + c]).toFixed(1)} dB`);
if (process.env.OUTPNG) {
  // raw dumps; python turns them into a PNG
  fs.writeFileSync(process.env.OUTPNG + '.cam.rgba', cam.rgba); fs.writeFileSync(process.env.OUTPNG + '.ideal.rgba', idl.rgba);
  fs.writeFileSync(process.env.OUTPNG + '.json', JSON.stringify({ K, Kh }));
}

// oracle calibration: per-channel affine (and 3x3 + offset) least squares against the true bytes
{
  const rows = [];
  for (let b = 0; b < kb; b++) for (let a = 0; a < ka; a++) {
    const k = b * M + a;
    if (!(buf.wt[k] > 0)) continue;
    const p = 4 * (b * W + a);
    rows.push([buf.val[3 * k], buf.val[3 * k + 1], buf.val[3 * k + 2], scr[p], scr[p + 1], scr[p + 2]]);
  }
  const lsq = (X, y) => { // normal equations
    const n = X[0].length, A = Array.from({ length: n }, () => new Float64Array(n)), r = new Float64Array(n);
    X.forEach((x, i) => { for (let a = 0; a < n; a++) { r[a] += x[a] * y[i]; for (let b = 0; b < n; b++) A[a][b] += x[a] * x[b]; } });
    for (let c = 0; c < n; c++) { let pv = c; for (let q = c + 1; q < n; q++) if (Math.abs(A[q][c]) > Math.abs(A[pv][c])) pv = q; [A[c], A[pv]] = [A[pv], A[c]]; [r[c], r[pv]] = [r[pv], r[c]];
      for (let q = 0; q < n; q++) if (q !== c) { const f = A[q][c] / A[c][c]; for (let b = c; b < n; b++) A[q][b] -= f * A[c][b]; r[q] -= f * r[c]; } }
    return Array.from(r, (v, i) => v / A[i][i]);
  };
  const rms = (pred, c) => Math.sqrt(rows.reduce((s, r, i) => s + (pred[i] - r[3 + c]) ** 2, 0) / rows.length);
  const report = [];
  const fits = [0, 1, 2].map((c) => {
    const aff = lsq(rows.map((r) => [r[c], 1]), rows.map((r) => r[3 + c]));
    const full = lsq(rows.map((r) => [r[0], r[1], r[2], 1]), rows.map((r) => r[3 + c]));
    const cub = lsq(rows.map((r) => [r[0], r[1], r[2], r[c] * r[c] / 255, r[c] ** 3 / 65025, 1]), rows.map((r) => r[3 + c]));
    report.push(`ch${c}: affine slope ${aff[0].toFixed(3)} rms ${rms(rows.map((r) => aff[0] * r[c] + aff[1]), c).toFixed(2)} | 3x3 rms ${rms(rows.map((r) => full[0] * r[0] + full[1] * r[1] + full[2] * r[2] + full[3]), c).toFixed(2)} | 3x3+cubic rms ${rms(rows.map((r) => cub[0] * r[0] + cub[1] * r[1] + cub[2] * r[2] + cub[3] * r[c] * r[c] / 255 + cub[4] * r[c] ** 3 / 65025 + cub[5]), c).toFixed(2)}`);
    return cub;
  });
  console.log('oracle calibration (byte error, levels):\n  ' + report.join('\n  '));
  const ob = new CoefBuffer(M);
  for (let b = 0; b < kb; b++) for (let a = 0; a < ka; a++) {
    const k = b * M + a; if (!(buf.wt[k] > 0)) continue;
    const v = [buf.val[3 * k], buf.val[3 * k + 1], buf.val[3 * k + 2]];
    ob.wt[k] = 1;
    for (let c = 0; c < 3; c++) { const f = fits[c]; ob.val[3 * k + c] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[c] * v[c] / 255 + f[4] * v[c] ** 3 / 65025 + f[5]; }
  }
  const od = decodeThumb(ob, { ...opt, autoCentre: false, autoSlope: false, gain: 1, gamma: 1, crosstalk: 0, chGain: [1, 1, 1] });
  console.log(`  camera decode with oracle 3x3+cubic calibration vs ideal: ${psnr(od.rgba, (i, c) => idl.rgba[4 * i + c]).toFixed(1)} dB`);
  if (process.env.OUTPNG) fs.writeFileSync(process.env.OUTPNG + '.oracle.rgba', od.rgba);
}

// transfer curve: camera value vs true byte, per channel
if (process.env.CURVE) {
  for (let c = 0; c < 3; c++) {
    const bins = Array.from({ length: 16 }, () => []);
    for (let b = 0; b < kb; b++) for (let a = 0; a < ka; a++) {
      const k = b * M + a; if (!(buf.wt[k] > 0)) continue;
      bins[scr[4 * (b * W + a) + c] >> 4].push(buf.val[3 * k + c]);
    }
    console.log(`ch${c} byte-bin: ` + bins.map((v, i) => { if (v.length < 5) return `${i * 16}:-`; const m = v.reduce((s, x) => s + x, 0) / v.length; const sd = Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length); return `${i * 16}:${m.toFixed(0)}±${sd.toFixed(0)}(${v.length})`; }).join(' '));
  }
}

// error anatomy: residual after per-channel affine fit, by position, and its neighbour structure
if (process.env.ANATOMY) {
  for (let c = 0; c < 3; c++) {
    const pts = [];
    for (let b = 0; b < kb; b++) for (let a = 0; a < ka; a++) { const k = b * M + a; if (buf.wt[k] > 0) pts.push([a, b, buf.val[3 * k + c], scr[4 * (b * W + a) + c]]); }
    const n = pts.length, mx = pts.reduce((s, p) => s + p[2], 0) / n, my = pts.reduce((s, p) => s + p[3], 0) / n;
    const sl = pts.reduce((s, p) => s + (p[2] - mx) * (p[3] - my), 0) / pts.reduce((s, p) => s + (p[2] - mx) ** 2, 0);
    const res = new Map(); pts.forEach((p) => res.set(p[0] + ',' + p[1], my + sl * (p[2] - mx) - p[3]));
    const rmsOf = (f) => { const v = pts.filter(f).map((p) => res.get(p[0] + ',' + p[1])); return `${Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length).toFixed(2)} (n=${v.length})`; };
    // residual vs neighbour true-byte deviation: does the error look like bleed?
    let sxy = 0, sxx = 0;
    for (const p of pts) {
      const r = res.get(p[0] + ',' + p[1]);
      let nb = 0, nn = 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const a = p[0] + dx, b = p[1] + dy; if (a >= 0 && b >= 0) { nb += scr[4 * (b * W + a) + c] - p[3]; nn++; } }
      if (nn === 4) { sxy += r * nb; sxx += nb * nb; }
    }
    console.log(`ch${c}: residual rms  a=0 col ${rmsOf((p) => p[0] === 0)}  b=0 row ${rmsOf((p) => p[1] === 0)}  a,b<8 ${rmsOf((p) => p[0] < 8 && p[1] < 8)}  rest ${rmsOf((p) => p[0] >= 8 || p[1] >= 8)}  | residual vs (neighbours - self) slope ${(sxy / sxx).toFixed(3)}`);
  }
}

// registration error vs the simulator's true geometry (last frame): fractional u at true pixel centres
if (process.env.REG && meta.persp !== undefined) {
  const fr = meta.frames[meta.frames.length - 1], th = (fr.rot * Math.PI) / 180, m = meta.mag;
  const Hm = [[m * Math.cos(th), -m * Math.sin(th), fr.corner[0]], [m * Math.sin(th), m * Math.cos(th), fr.corner[1]], [meta.persp, meta.persp * 0.5, 1]];
  const toCam = (sx, sy) => { const d = Hm[2][0] * sx + Hm[2][1] * sy + Hm[2][2]; return [(Hm[0][0] * sx + Hm[0][1] * sy + Hm[0][2]) / d, (Hm[1][0] * sx + Hm[1][1] * sy + Hm[1][2]) / d]; };
  const { ga, gb, offA, offB } = last;
  const region = (name, cells) => {
    let su = 0, sv = 0, n = 0, worst = 0;
    for (const [a, b] of cells) {
      const [x, y] = toCam(a + 0.5, b + 0.5);
      if (x < 0 || y < 0 || x >= meta.cw || y >= meta.ch) continue;
      // green subpixel centre is at screen x + 0.5 in the sim; compare against green's window centre
      let du = gridU(ga, x, y) - offA[1]; du -= Math.round(du);
      let dv = gridU(gb, x, y) - offB[1]; dv -= Math.round(dv);
      su += du * du; sv += dv * dv; n++; worst = Math.max(worst, Math.abs(du), Math.abs(dv));
    }
    console.log(`  ${name}: rms du ${Math.sqrt(su / n).toFixed(3)} dv ${Math.sqrt(sv / n).toFixed(3)} worst ${worst.toFixed(3)} (pixels; n=${n})`);
  };
  const cells = (a0, a1, b0, b1) => { const r = []; for (let b = b0; b < b1; b++) for (let a = a0; a < a1; a++) r.push([a, b]); return r; };
  console.log('registration vs truth (fraction of a screen pixel):');
  region('corner a,b<8   ', cells(0, 8, 0, 8));
  region('first row/col  ', [...cells(0, 400, 0, 1), ...cells(0, 1, 0, 220)]);
  region('middle         ', cells(150, 250, 80, 140));
  region('far edge       ', cells(380, 400, 180, 215));
}
if (process.env.CORNER) {
  for (let b = 0; b < 4; b++) {
    let line = `b=${b}: `;
    for (let a = 0; a < 6; a++) { const k = b * M + a, p = 4 * (b * W + a); line += `[${[0, 1, 2].map((c) => `${scr[p + c]}/${buf.val[3 * k + c].toFixed(0)}`).join(' ')}]  `; }
    console.log(line);
  }
}
if (process.env.ORACLE2) {
  // byte_c ~ cam(cell, 3ch) + cam(4 neighbours, 3ch) + squares of own channels + 1
  const feats = (a, b) => {
    const f = [];
    for (const [dx, dy] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const k = (b + dy) * M + a + dx;
      if (a + dx < 0 || b + dy < 0 || !(buf.wt[k] > 0)) return null;
      f.push(buf.val[3 * k] / 255, buf.val[3 * k + 1] / 255, buf.val[3 * k + 2] / 255);
    }
    f.push(f[0] * f[0], f[1] * f[1], f[2] * f[2], f[0] ** 3, f[1] ** 3, f[2] ** 3, 1);
    return f;
  };
  const X = [], Y = [];
  for (let b = 1; b < kb - 1; b++) for (let a = 1; a < ka - 1; a++) { const f = feats(a, b); if (f) { X.push(f); Y.push([0, 1, 2].map((c) => scr[4 * (b * W + a) + c])); } }
  const n = X[0].length;
  for (let c = 0; c < 3; c++) {
    const A = Array.from({ length: n }, () => new Float64Array(n)), r = new Float64Array(n);
    X.forEach((x, i) => { for (let p = 0; p < n; p++) { r[p] += x[p] * Y[i][c]; for (let q = 0; q < n; q++) A[p][q] += x[p] * x[q]; } });
    for (let p = 0; p < n; p++) A[p][p] += 1e-6;
    for (let cc = 0; cc < n; cc++) { let pv = cc; for (let q = cc + 1; q < n; q++) if (Math.abs(A[q][cc]) > Math.abs(A[pv][cc])) pv = q; [A[cc], A[pv]] = [A[pv], A[cc]]; [r[cc], r[pv]] = [r[pv], r[cc]];
      for (let q = 0; q < n; q++) if (q !== cc) { const f = A[q][cc] / A[cc][cc]; for (let t = cc; t < n; t++) A[q][t] -= f * A[cc][t]; r[q] -= f * r[cc]; } }
    const wv = Array.from(r, (v, i) => v / A[i][i]);
    const rms = Math.sqrt(X.reduce((s, x, i) => s + (x.reduce((t, v, j) => t + v * wv[j], 0) - Y[i][c]) ** 2, 0) / X.length);
    console.log(`oracle (cell+neighbours, cubic) ch${c}: rms ${rms.toFixed(2)} levels; own-channel weight ${(wv[c] / 255).toFixed(3)}, other channels ${[0, 1, 2].filter((x) => x !== c).map((x) => (wv[x] / 255).toFixed(3))}, neighbour same-ch ${[1, 2, 3, 4].map((nb) => (wv[nb * 3 + c] / 255).toFixed(3))}`);
  }
}
