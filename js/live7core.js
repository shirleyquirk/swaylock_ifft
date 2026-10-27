// Live tryseven decoder core: camera frame (zoomed on the screen's top-left corner) -> thumbnail.
//
// tryseven puts DCT coefficient (a, b) at screen pixel (a, b). The low frequencies, which are all a
// thumbnail needs, are the top-left corner. With the camera close enough to resolve single screen
// pixels, each pixel's colour is read back as one coefficient:
//   1. lock onto the pixel grid (FFT peak of the lattice, then a smooth phase field per axis);
//   2. average each pixel's interior;
//   3. find pixel (0,0) from the bezel edge;
//   4. map camera values back to bytes (calibration), uncompand, inverse DCT of the K x Kh corner.
// Pure functions, no DOM: runs the same in the page and in node tests.

import { fft2, dct3 } from './seven.js';

const TAU = 2 * Math.PI;
const C0 = 1.0;

export function luminance(rgba, w, h) {
  const L = new Float32Array(w * h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) L[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  return L;
}

// Squared gradient: every screen-pixel boundary is an edge whatever the content, so this has a lattice
// line at the pixel pitch on both axes even when the dark gaps between pixels are blurred away.
export function edgeEnergy(L, w, h) {
  const E = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    const r = y * w;
    for (let x = 1; x < w - 1; x++) {
      const gx = L[r + x + 1] - L[r + x - 1], gy = L[r + x + w] - L[r + x - w];
      E[r + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return E;
}

function hann(n) {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = 0.5 - 0.5 * Math.cos((TAU * (i + 0.5)) / n);
  return a;
}

// ---------------------------------------------------------------- 1. coarse lattice frequency
// Returns the two lattice vectors (cycles per camera px): a ~ along camera x, b ~ along camera y.
export function coarseGrid(L, w, h, opt = {}) {
  const minPitch = opt.minPitch ?? 2.5, maxPitch = opt.maxPitch ?? 40;
  let P = 64;
  while (P * 2 <= Math.min(w, h, opt.patch ?? 512)) P *= 2;
  const cx = opt.cx ?? w / 2, cy = opt.cy ?? h / 2;
  const x0 = Math.max(0, Math.min(w - P, Math.round(cx - P / 2)));
  const y0 = Math.max(0, Math.min(h - P, Math.round(cy - P / 2)));
  const win = hann(P);
  let mean = 0;
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) mean += L[(y0 + y) * w + x0 + x];
  mean /= P * P;
  const re = new Float64Array(P * P), im = new Float64Array(P * P);
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) re[y * P + x] = (L[(y0 + y) * w + x0 + x] - mean) * win[x] * win[y];
  const tr = new Float64Array(P), ti = new Float64Array(P);
  for (let y = 0; y < P; y++) {
    for (let x = 0; x < P; x++) { tr[x] = re[y * P + x]; ti[x] = 0; }
    fft2(tr, ti, false);
    for (let x = 0; x < P; x++) { re[y * P + x] = tr[x]; im[y * P + x] = ti[x]; }
  }
  for (let x = 0; x < P; x++) {
    for (let y = 0; y < P; y++) { tr[y] = re[y * P + x]; ti[y] = im[y * P + x]; }
    fft2(tr, ti, false);
    for (let y = 0; y < P; y++) { re[y * P + x] = tr[y]; im[y * P + x] = ti[y]; }
  }
  const mag = (kx, ky) => {
    const i = (((ky % P) + P) % P) * P + (((kx % P) + P) % P);
    return Math.hypot(re[i], im[i]);
  };
  const rmin = P / maxPitch, rmax = Math.min(P / minPitch, P / 2 - 2);

  // lattice peaks are sharp; image content is broad. "peakiness" = peak / median of a ring around it
  const peakiness = (kx, ky) => {
    const ring = [];
    for (let dy = -5; dy <= 5; dy++) for (let dx = -5; dx <= 5; dx++) {
      const r = Math.hypot(dx, dy);
      if (r >= 3 && r <= 5) ring.push(mag(kx + dx, ky + dy));
    }
    ring.sort((p, q) => p - q);
    return mag(kx, ky) / (ring[ring.length >> 1] + 1e-9);
  };
  const isLocalMax = (kx, ky) => {
    const m = mag(kx, ky);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if ((dx || dy) && mag(kx + dx, ky + dy) > m) return false;
    return true;
  };
  const search = (axis) => {
    let best = null;
    for (let a = 1; a <= rmax; a++) for (let b = -a; b <= a; b++) {
      if (axis === 1 && Math.abs(b) === a) continue;
      const kx = axis === 0 ? a : b, ky = axis === 0 ? b : a;
      const r = Math.hypot(kx, ky);
      if (r < rmin || r > rmax) continue;
      const m = mag(kx, ky);
      if (!best || m * Math.sqrt(r) > best.score) best = { kx, ky, m, score: m * Math.sqrt(r) };
    }
    return best;
  };
  // the strongest lattice line may be a harmonic (RGB subpixel stripes put a big peak at 3/pitch)
  // A true sub-harmonic f/d needs every multiple k f/d (k < d) to be a sharp lattice line too;
  // image content only ever supplies isolated broad bumps.
  const latticeAt = (tx, ty, floor) => {
    let bm = -1, bx = 0, by = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const kx = Math.round(tx) + dx, ky = Math.round(ty) + dy, m = mag(kx, ky);
      if (m > bm) { bm = m; bx = kx; by = ky; }
    }
    return bm >= floor && isLocalMax(bx, by) && peakiness(bx, by) >= 5 ? { kx: bx, ky: by, m: bm } : null;
  };
  const fundamental = (pk) => {
    for (const d of [4, 3, 2]) {
      if (Math.hypot(pk.kx / d, pk.ky / d) < rmin) continue;
      let base = null, ok = true;
      for (let k = 1; k < d && ok; k++) {
        const hit = latticeAt((pk.kx * k) / d, (pk.ky * k) / d, 0.05 * pk.m);
        if (!hit) ok = false; else if (k === 1) base = hit;
      }
      if (ok && base) return { ...base, harmonic: d };
    }
    return { ...pk, harmonic: 1 };
  };
  const refine = (pk) => {
    const l = (kx, ky) => Math.log(mag(kx, ky) + 1e-12);
    const c = l(pk.kx, pk.ky);
    const xm = l(pk.kx - 1, pk.ky), xp = l(pk.kx + 1, pk.ky), ym = l(pk.kx, pk.ky - 1), yp = l(pk.kx, pk.ky + 1);
    const dx = xm - 2 * c + xp < 0 ? (0.5 * (xm - xp)) / (xm - 2 * c + xp) : 0;
    const dy = ym - 2 * c + yp < 0 ? (0.5 * (ym - yp)) / (ym - 2 * c + yp) : 0;
    const cl = (v) => Math.max(-0.5, Math.min(0.5, v));
    return { fx: (pk.kx + cl(dx)) / P, fy: (pk.ky + cl(dy)) / P, peak: peakiness(pk.kx, pk.ky), harmonic: pk.harmonic };
  };
  const ra = search(0), rb = search(1);
  if (!ra || !rb) return null;
  let a = refine(fundamental(ra)), b = refine(fundamental(rb));
  // Pixels are square. Trust the sharper axis; the other must be it rotated by 90 degrees. Keep the
  // weaker axis's own measurement only if it agrees (it carries any perspective), else use the rotation.
  const rotA = { fx: -a.fy, fy: a.fx }, rotB = { fx: b.fy, fy: -b.fx };
  const agree = (f, g) => Math.hypot(f.fx - g.fx, f.fy - g.fy) < 0.08 * Math.hypot(g.fx, g.fy);
  const nearPeak = (g) => {
    const kx = Math.round(g.fx * P), ky = Math.round(g.fy * P);
    let best = null;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const m = mag(kx + dx, ky + dy);
      if (!best || m > best.m) best = { kx: kx + dx, ky: ky + dy, m };
    }
    return isLocalMax(best.kx, best.ky) && peakiness(best.kx, best.ky) >= 3 ? refine({ ...best, harmonic: 1 }) : null;
  };
  if (a.peak >= b.peak) {
    if (!agree(b, rotA)) b = { ...(nearPeak(rotA) || { ...rotA, peak: 0 }), derived: true };
  } else if (!agree(a, rotB)) a = { ...(nearPeak(rotB) || { ...rotB, peak: 0 }), derived: true };
  return { a, b, P };
}

// ---------------------------------------------------------------- 2. phase field per axis
// u(x, y) = f.(x, y) + psi(x, y) / 2pi, psi a smooth (quadratic) field fitted to patch phases.
// Pixel centres sit at integer u (the lattice term peaks mid-pixel, where there is no gap).
export function phaseField(L, w, h, f, opt = {}) {
  const S = Math.min(opt.fine ?? 128, w, h) & ~1;
  const nx = opt.nx ?? Math.max(3, Math.round(w / 200)), ny = opt.ny ?? Math.max(3, Math.round(h / 200));
  const H2 = S >> 1;
  const winH = hann(H2), win = hann(S);
  const wx = TAU * f.fx, wy = TAU * f.fy;
  const cr = Math.cos(wx), ci = -Math.sin(wx);
  const pts = [];
  for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
    const px0 = Math.round((w - S) * (nx === 1 ? 0.5 : ix / (nx - 1)));
    const py0 = Math.round((h - S) * (ny === 1 ? 0.5 : iy / (ny - 1)));
    let mean = 0;
    for (let y = 0; y < S; y++) { const r = (py0 + y) * w + px0; for (let x = 0; x < S; x++) mean += L[r + x]; }
    mean /= S * S;
    // whole patch + four half-patches (for the local phase gradient)
    const z = new Float64Array(10); // [all r,i, left r,i, right r,i, top r,i, bottom r,i]
    for (let y = 0; y < S; y++) {
      const Y = py0 + y;
      const th = -(wx * px0 + wy * Y) % TAU;
      let er = Math.cos(th), ei = Math.sin(th);
      const row = Y * w + px0;
      const top = y < H2, wyh = winH[top ? y : y - H2];
      for (let x = 0; x < S; x++) {
        const v = L[row + x] - mean;
        const va = v * win[x] * win[y];
        z[0] += va * er; z[1] += va * ei;
        const left = x < H2, wxh = winH[left ? x : x - H2];
        const vx = v * wxh * win[y], vy = v * win[x] * wyh;
        if (left) { z[2] += vx * er; z[3] += vx * ei; } else { z[4] += vx * er; z[5] += vx * ei; }
        if (top) { z[6] += vy * er; z[7] += vy * ei; } else { z[8] += vy * er; z[9] += vy * ei; }
        const t = er * cr - ei * ci; ei = er * ci + ei * cr; er = t;
      }
    }
    const dphase = (ar, ai, br, bi) => Math.atan2(ai * br - ar * bi, ar * br + ai * bi); // arg(a conj b)
    pts.push({
      ix, iy, cx: px0 + S / 2, cy: py0 + S / 2,
      z: Math.hypot(z[0], z[1]), psi: Math.atan2(z[1], z[0]),
      gx: dphase(z[4], z[5], z[2], z[3]) / H2, gy: dphase(z[8], z[9], z[6], z[7]) / H2,
    });
  }
  const zmax = Math.max(...pts.map((p) => p.z));
  const use = pts.map((p) => p.z >= (opt.minRel ?? 0.15) * zmax);
  const at = (ix, iy) => iy * nx + ix;
  let start = 0;
  pts.forEach((p, i) => { if (p.z > pts[start].z) start = i; });
  pts[start].uw = pts[start].psi;
  const seen = new Set([start]);
  const queue = [start];
  while (queue.length) {
    const i = queue.shift(), p = pts[i];
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const qx = p.ix + dx, qy = p.iy + dy;
      if (qx < 0 || qy < 0 || qx >= nx || qy >= ny) continue;
      const j = at(qx, qy);
      if (seen.has(j) || !use[j]) continue;
      const q = pts[j];
      const pred = p.uw + 0.5 * ((p.gx + q.gx) * (q.cx - p.cx) + (p.gy + q.gy) * (q.cy - p.cy));
      q.uw = q.psi + TAU * Math.round((pred - q.psi) / TAU);
      seen.add(j);
      queue.push(j);
    }
  }
  let fit = pts.filter((_, i) => seen.has(i));
  let pcoef = new Array(10).fill(0), rms = 0;
  for (let round = 0; round < 3; round++) {
    const nb = fit.length >= 24 ? 10 : fit.length >= 9 ? 6 : fit.length >= 3 ? 3 : 1;
    const basis = (X, Y) => [1, X, Y, X * X, X * Y, Y * Y, X * X * X, X * X * Y, X * Y * Y, Y * Y * Y].slice(0, nb);
    const A = Array.from({ length: nb }, () => new Float64Array(nb)), rhs = new Float64Array(nb);
    for (const p of fit) {
      const X = (p.cx - w / 2) / w, Y = (p.cy - h / 2) / w;
      const bv = basis(X, Y), wt = p.z;
      for (let r = 0; r < nb; r++) { rhs[r] += wt * bv[r] * p.uw; for (let c = 0; c < nb; c++) A[r][c] += wt * bv[r] * bv[c]; }
    }
    pcoef = new Array(10).fill(0);
    solve(A, rhs).forEach((v, i) => { pcoef[i] = v; });
    let se = 0, sw = 0;
    const res = fit.map((p) => {
      const X = (p.cx - w / 2) / w, Y = (p.cy - h / 2) / w;
      const r = (p.uw - psiPoly(pcoef, X, Y)) / TAU;
      se += p.z * r * r; sw += p.z;
      return Math.abs(r);
    });
    rms = Math.sqrt(se / Math.max(sw, 1e-9));
    const keep = fit.filter((_, i) => res[i] < 0.2);
    if (keep.length === fit.length || keep.length < 6) break;
    fit = keep;
  }
  return { fx: f.fx, fy: f.fy, p: pcoef, w, h, rms, patches: fit.length, of: pts.length };
}

function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    if (Math.abs(M[c][c]) < 1e-12) return new Array(n).fill(0);
    for (let r = 0; r < n; r++) if (r !== c) {
      const k = M[r][c] / M[c][c];
      for (let k2 = c; k2 <= n; k2++) M[r][k2] -= k * M[c][k2];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}

function psiPoly(p, X, Y) {
  return p[0] + p[1] * X + p[2] * Y + p[3] * X * X + p[4] * X * Y + p[5] * Y * Y
    + p[6] * X * X * X + p[7] * X * X * Y + p[8] * X * Y * Y + p[9] * Y * Y * Y;
}

export function gridU(g, x, y) {
  const X = (x - g.w / 2) / g.w, Y = (y - g.h / 2) / g.w;
  return g.fx * x + g.fy * y + psiPoly(g.p, X, Y) / TAU;
}

// camera position of grid coordinates (u, v): Newton on the two phase fields
export function camPoint(ga, gb, u, v) {
  const det = ga.fx * gb.fy - ga.fy * gb.fx;
  let x = ga.w / 2, y = ga.h / 2;
  for (let it = 0; it < 6; it++) {
    const du = u - gridU(ga, x, y), dv = v - gridU(gb, x, y);
    x += (gb.fy * du - ga.fy * dv) / det;
    y += (-gb.fx * du + ga.fx * dv) / det;
  }
  return [x, y];
}

// ---------------------------------------------------------------- 3. subpixel profile + per-pixel averages
// Average each channel against the position inside the screen pixel (frac of u, v). Cell contents are
// effectively random, so this is the subpixel layout: where each channel's emitter sits and how wide it is.
export function subpixelProfile(rgba, w, h, ga, gb, opt = {}) {
  const NB = opt.bins ?? 24, step = opt.step ?? 2;
  const pu = new Float64Array(3 * NB), pv = new Float64Array(3 * NB), nu = new Float64Array(NB), nv = new Float64Array(NB);
  const pa = ga.p, pb = gb.p, iw = 1 / w;
  for (let y = 0; y < h; y += step) {
    const Y = (y - h / 2) * iw;
    const Aa = (pa[0] + pa[2] * Y + pa[5] * Y * Y + pa[9] * Y * Y * Y) / TAU, Ba = (pa[1] + pa[4] * Y + pa[8] * Y * Y) / TAU, Ca = (pa[3] + pa[7] * Y) / TAU, Da = pa[6] / TAU;
    const Ab = (pb[0] + pb[2] * Y + pb[5] * Y * Y + pb[9] * Y * Y * Y) / TAU, Bb = (pb[1] + pb[4] * Y + pb[8] * Y * Y) / TAU, Cb = (pb[3] + pb[7] * Y) / TAU, Db = pb[6] / TAU;
    for (let x = 0; x < w; x += step) {
      const X = (x - w / 2) * iw;
      const u = ga.fx * x + ga.fy * y + Aa + X * (Ba + X * (Ca + Da * X));
      const v = gb.fx * x + gb.fy * y + Ab + X * (Bb + X * (Cb + Db * X));
      const bu = Math.floor((u - Math.floor(u)) * NB), bv = Math.floor((v - Math.floor(v)) * NB);
      const p = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) { pu[c * NB + bu] += rgba[p + c]; pv[c * NB + bv] += rgba[p + c]; }
      nu[bu]++; nv[bv]++;
    }
  }
  const analyse = (prof, n) => {
    const out = [];
    for (let c = 0; c < 3; c++) {
      const v = Array.from({ length: NB }, (_, i) => prof[c * NB + i] / Math.max(n[i], 1));
      const lo = Math.min(...v), hi = Math.max(...v);
      let zr = 0, zi = 0, above = 0;
      v.forEach((x, i) => { const wgt = x - lo, ang = (TAU * (i + 0.5)) / NB; zr += wgt * Math.cos(ang); zi += wgt * Math.sin(ang); if (x >= (lo + hi) / 2) above++; });
      let off = Math.atan2(zi, zr) / TAU;
      off -= Math.round(off);                          // in (-0.5, 0.5]
      out.push({ off, width: above / NB, depth: (hi - lo) / Math.max(hi, 1), profile: v });
    }
    return out;
  };
  return { u: analyse(pu, nu), v: analyse(pv, nv) };
}

// opt.offA/offB/qA/qB: numbers, or per-channel arrays [r, g, b]
export function accumulate(rgba, w, h, ga, gb, opt = {}) {
  const arr = (v, d) => (Array.isArray(v) ? v : [v ?? d, v ?? d, v ?? d]);
  const oa = arr(opt.offA, 0), ob = arr(opt.offB, 0), qa = arr(opt.qA ?? opt.q, 0.3), qb = arr(opt.qB ?? opt.q, 0.3);
  const step = opt.step ?? 1;
  let umin = Infinity, umax = -Infinity, vmin = Infinity, vmax = -Infinity;
  for (const [x, y] of [[0, 0], [w, 0], [0, h], [w, h], [w / 2, 0], [w / 2, h], [0, h / 2], [w, h / 2], [w / 2, h / 2]]) {
    const u = gridU(ga, x, y), v = gridU(gb, x, y);
    umin = Math.min(umin, u); umax = Math.max(umax, u); vmin = Math.min(vmin, v); vmax = Math.max(vmax, v);
  }
  const i0 = Math.floor(umin) - 2, j0 = Math.floor(vmin) - 2;
  const ni = Math.ceil(umax) + 3 - i0, nj = Math.ceil(vmax) + 3 - j0;
  if (!(ni > 0 && nj > 0) || ni * nj > 4e6) return null;
  const sum = new Float32Array(ni * nj * 3), cnt = new Float32Array(ni * nj * 3);
  const pa = ga.p, pb = gb.p, iw = 1 / w;
  for (let y = 0; y < h; y += step) {
    const Y = (y - h / 2) * iw;
    const Aa = (pa[0] + pa[2] * Y + pa[5] * Y * Y + pa[9] * Y * Y * Y) / TAU, Ba = (pa[1] + pa[4] * Y + pa[8] * Y * Y) / TAU, Ca = (pa[3] + pa[7] * Y) / TAU, Da = pa[6] / TAU;
    const Ab = (pb[0] + pb[2] * Y + pb[5] * Y * Y + pb[9] * Y * Y * Y) / TAU, Bb = (pb[1] + pb[4] * Y + pb[8] * Y * Y) / TAU, Cb = (pb[3] + pb[7] * Y) / TAU, Db = pb[6] / TAU;
    const ua0 = ga.fy * y, vb0 = gb.fy * y;
    for (let x = 0; x < w; x += step) {
      const X = (x - w / 2) * iw;
      const u = ga.fx * x + ua0 + Aa + X * (Ba + X * (Ca + Da * X));
      const v = gb.fx * x + vb0 + Ab + X * (Bb + X * (Cb + Db * X));
      const p = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        const uc = u - oa[c], iu = Math.round(uc);
        if (Math.abs(uc - iu) > qa[c]) continue;
        const vc = v - ob[c], iv = Math.round(vc);
        if (Math.abs(vc - iv) > qb[c]) continue;
        const k = 3 * ((iv - j0) * ni + (iu - i0)) + c;
        sum[k] += rgba[p + c];
        cnt[k]++;
      }
    }
  }
  const cellArea = 1 / Math.abs(ga.fx * gb.fy - ga.fy * gb.fx);
  const exp = [0, 1, 2].map((c) => (cellArea * 4 * qa[c] * qb[c]) / (step * step));
  const mean = new Float32Array(ni * nj * 3), valid = new Uint8Array(ni * nj);
  for (let k = 0; k < ni * nj; k++) {
    let ok = 1;
    for (let c = 0; c < 3; c++) if (!(cnt[3 * k + c] >= 0.5 * exp[c] && cnt[3 * k + c] > 0)) ok = 0;
    if (!ok) continue;
    valid[k] = 1;
    for (let c = 0; c < 3; c++) mean[3 * k + c] = sum[3 * k + c] / cnt[3 * k + c];
  }
  return { i0, j0, ni, nj, mean, valid, expected: exp[1], cellArea };
}

// spread of per-pixel means: high when sampling pixel interiors, low when straddling boundaries
export function contrast(acc) {
  let s = 0, s2 = 0, n = 0;
  for (let k = 0; k < acc.ni * acc.nj; k++) {
    if (!acc.valid[k]) continue;
    const l = 0.299 * acc.mean[3 * k] + 0.587 * acc.mean[3 * k + 1] + 0.114 * acc.mean[3 * k + 2];
    s += l; s2 += l * l; n++;
  }
  return n > 10 ? s2 / n - (s / n) ** 2 : 0;
}

// sub-pixel phase that maximises the spread (on demand; ~16 subsampled passes)
export function bestOffsets(rgba, w, h, ga, gb, opt = {}) {
  const steps = 8, q = 0.2;
  const run = (oa, ob) => { const a = accumulate(rgba, w, h, ga, gb, { q, offA: oa, offB: ob, step: 2 }); return a ? contrast(a) : 0; };
  let bestA = 0, bv = -1;
  for (let s = 0; s < steps; s++) { const v = run(s / steps, opt.offB ?? 0); if (v > bv) { bv = v; bestA = s / steps; } }
  let bestB = 0; bv = -1;
  for (let s = 0; s < steps; s++) { const v = run(bestA, s / steps); if (v > bv) { bv = v; bestB = s / steps; } }
  const wrap = (v) => (v > 0.5 ? v - 1 : v);
  return { offA: wrap(bestA), offB: wrap(bestB) };
}

// ---------------------------------------------------------------- 4. screen orientation + origin
// rot: where the screen's top-left corner appears in the camera image: 0 top-left, 90 top-right,
// 180 bottom-right, 270 bottom-left.
export function toScreen(acc, rot = 0, mirror = false) {
  const map = (i, j) => {
    if (mirror) i = -i;
    switch (rot) {
      case 90: return [j, -i];
      case 180: return [-i, -j];
      case 270: return [-j, i];
      default: return [i, j];
    }
  };
  const I0 = acc.i0, J0 = acc.j0, I1 = acc.i0 + acc.ni - 1, J1 = acc.j0 + acc.nj - 1;
  const cs = [map(I0, J0), map(I1, J0), map(I0, J1), map(I1, J1)];
  const s0 = Math.min(...cs.map((c) => c[0])), t0 = Math.min(...cs.map((c) => c[1]));
  const ns = Math.max(...cs.map((c) => c[0])) - s0 + 1, nt = Math.max(...cs.map((c) => c[1])) - t0 + 1;
  const mean = new Float32Array(ns * nt * 3), valid = new Uint8Array(ns * nt);
  for (let j = 0; j < acc.nj; j++) for (let i = 0; i < acc.ni; i++) {
    const k = j * acc.ni + i;
    if (!acc.valid[k]) continue;
    const [s, t] = map(i + acc.i0, j + acc.j0);
    const o = (t - t0) * ns + (s - s0);
    valid[o] = 1;
    mean[3 * o] = acc.mean[3 * k]; mean[3 * o + 1] = acc.mean[3 * k + 1]; mean[3 * o + 2] = acc.mean[3 * k + 2];
  }
  return { s0, t0, ns, nt, mean, valid, map };
}

const median = (a) => { if (!a.length) return NaN; const b = Float64Array.from(a).sort(); return b[b.length >> 1]; };

// first column / row where the dark bezel ends and three lit pixels follow
export function findOrigin(S, opt = {}) {
  const frac = opt.frac ?? 0.45;
  const lum = (o) => 0.299 * S.mean[3 * o] + 0.587 * S.mean[3 * o + 1] + 0.114 * S.mean[3 * o + 2];
  const edge = (n, line) => {
    const sc = new Float64Array(n).fill(NaN);
    for (let i = 0; i < n; i++) { const v = line(i); if (v.length >= 3) sc[i] = median(v); }
    const lit = [...sc].filter((v) => !Number.isNaN(v)).sort((a, b) => a - b);
    if (lit.length < 6) return null;
    const thr = frac * lit[Math.floor(lit.length * 0.9)];
    let dark = false;
    for (let i = 0; i < n - 2; i++) {
      if (Number.isNaN(sc[i])) continue;
      if (sc[i] < thr) dark = true;
      else if (dark && sc[i + 1] >= thr && sc[i + 2] >= thr) return i;
    }
    return null;
  };
  const col = (s) => { const v = []; for (let t = 0; t < S.nt; t++) { const o = t * S.ns + s; if (S.valid[o]) v.push(lum(o)); } return v; };
  const sIdx = edge(S.ns, col);
  const row = (t) => {
    const v = [];
    for (let s = sIdx ?? 0; s < S.ns; s++) { const o = t * S.ns + s; if (S.valid[o]) v.push(lum(o)); }
    return v;
  };
  const tIdx = edge(S.nt, row);
  return { s: sIdx, t: tIdx }; // indices into S's arrays (null = edge not in view)
}

// ---------------------------------------------------------------- 5. coefficient buffer (screen coords)
export class CoefBuffer {
  constructor(max = 512) { this.max = max; this.val = new Float32Array(max * max * 3); this.wt = new Float32Array(max * max); }
  reset() { this.val.fill(0); this.wt.fill(0); }
  // S cell (si, ti) = screen pixel (si - os, ti - ot)
  add(S, os, ot, alpha) {
    let n = 0;
    for (let ti = 0; ti < S.nt; ti++) for (let si = 0; si < S.ns; si++) {
      const o = ti * S.ns + si;
      if (!S.valid[o]) continue;
      const a = si - os, b = ti - ot;
      if (a < 0 || b < 0 || a >= this.max || b >= this.max) continue;
      const k = b * this.max + a;
      // running mean until 1/n reaches alpha, then exponential forgetting at rate alpha
      const r = Math.max(1 / (this.wt[k] + 1), alpha);
      for (let c = 0; c < 3; c++) this.val[3 * k + c] += r * (S.mean[3 * o + c] - this.val[3 * k + c]);
      this.wt[k] = Math.min(1e6, this.wt[k] + 1);
      n++;
    }
    return n;
  }
  extent() {
    let ka = 0, kb = 0;
    for (let b = 0; b < this.max; b++) for (let a = 0; a < this.max; a++) if (this.wt[b * this.max + a] > 0) { if (a + 1 > ka) ka = a + 1; if (b + 1 > kb) kb = b + 1; }
    return [ka, kb];
  }
}

// ---------------------------------------------------------------- 6. calibrate + decode
const YCC2RGB = [[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]];

export function decodeThumb(buf, opt) {
  const W = opt.screenW, H = opt.screenH;
  let [K, Kh] = buf.extent();
  K = Math.min(K, opt.maxK ?? 256); Kh = Math.min(Kh, opt.maxK ?? 256);
  K = Math.max(4, K & ~3); Kh = Math.max(4, Kh & ~3);
  const n = K * Kh, M = buf.max;
  const cam = new Float32Array(n * 3), has = new Uint8Array(n);
  const k3 = opt.crosstalk ?? 0, inv = 1 / (1 - 3 * k3);
  for (let b = 0; b < Kh; b++) for (let a = 0; a < K; a++) {
    const s = b * M + a, o = b * K + a;
    if (!(buf.wt[s] > 0)) continue;
    has[o] = 1;
    const r = buf.val[3 * s], g = buf.val[3 * s + 1], bl = buf.val[3 * s + 2], sm = r + g + bl;
    // undo symmetric crosstalk: cam = (1-3k) s + k * sum(s)
    cam[3 * o] = inv * (r - k3 * sm); cam[3 * o + 1] = inv * (g - k3 * sm); cam[3 * o + 2] = inv * (bl - k3 * sm);
  }
  // Symmetric calibration: away from DC each channel's bytes are distributed symmetrically about
  // 127.5 (coefficient signs are 50/50). Pairing the camera's quantile q with 1-q therefore undoes any
  // monotone tone curve up to its odd part: z(x) = (x - Q(1 - F(x))) / 2 is an odd, centred coordinate.
  let zf = null;
  if (opt.calib === 'symmetric') {
    const NQ = 129;
    zf = [0, 1, 2].map((c) => {
      const v = [];
      for (let b = 0; b < Kh; b++) for (let a = 0; a < K; a++) { const o = b * K + a; if (has[o] && a + b >= 3) v.push(cam[3 * o + c]); }
      if (v.length < 200) return null;
      v.sort((p, q) => p - q);
      const Q = Float64Array.from({ length: NQ }, (_, i) => v[Math.min(v.length - 1, Math.round((i / (NQ - 1)) * (v.length - 1)))]);
      const F = (x) => { // inverse of the quantile table, linear interpolation, clamped
        if (x <= Q[0]) return 0;
        if (x >= Q[NQ - 1]) return 1;
        let lo = 0, hi = NQ - 1;
        while (hi - lo > 1) { const m = (lo + hi) >> 1; if (Q[m] <= x) lo = m; else hi = m; }
        const t = Q[hi] > Q[lo] ? (x - Q[lo]) / (Q[hi] - Q[lo]) : 0.5;
        return (lo + t) / (NQ - 1);
      };
      const Qi = (q) => { const f = q * (NQ - 1), i = Math.min(NQ - 2, Math.floor(f)); return Q[i] + (f - i) * (Q[i + 1] - Q[i]); };
      return (x) => (x - Qi(1 - F(x))) / 2;
    });
  }
  // per-channel centre: coefficient bytes away from DC are symmetric about 127.5
  const med = [0, 1, 2].map((c) => {
    if (!opt.autoCentre) return 127.5;
    const v = [];
    for (let b = 0; b < Kh; b++) for (let a = 0; a < K; a++) { const o = b * K + a; if (has[o] && a + b >= 3) v.push(cam[3 * o + c]); }
    const m = median(v);
    return Number.isNaN(m) ? 127.5 : m;
  });
  // slope camera->byte: pixel (0,0) is Y's DC, ~244 for any ordinary screen (mean luma 0.1..0.6 -> 237..251)
  let slope = 1, slopeOk = true;
  const zOf = (c, x) => (zf && zf[c] ? zf[c](x) : x - med[c]);
  if (opt.autoSlope && has[0]) {
    const d = zOf(0, cam[0]);
    slope = d > 3 ? ((opt.dcTarget ?? 244) - 127.5) / d : NaN;
    if (!(slope > 0.2 && slope < 20)) { slope = 1; slopeOk = false; }
  }
  slope *= opt.gain ?? 1;
  const chg = opt.chGain ?? [1, 1, 1], gam = opt.gamma ?? 1;
  const norm = Math.log1p((4 * W * H) / C0);
  const taper = opt.taper ?? 0, kappa = opt.logScale ?? 0.75, lowTrust = opt.lowTrust ?? 1, lowN = opt.lowN ?? 8;
  const planes = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  for (let b = 0; b < Kh; b++) {
    const tb = 1 - taper + taper * 0.5 * (1 + Math.cos((Math.PI * b) / Kh));
    for (let a = 0; a < K; a++) {
      const o = b * K + a;
      if (!has[o]) continue;
      const ta = 1 - taper + taper * 0.5 * (1 + Math.cos((Math.PI * a) / K));
      for (let c = 0; c < 3; c++) {
        const byte = 127.5 + slope * chg[c] * zOf(c, cam[3 * o + c]);
        let e = Math.max(-1, Math.min(1, (byte - 127.5) / 127.5));
        e = Math.sign(e) * Math.abs(e) ** gam;
        // low-frequency trust: the first row/column and the block around DC are read worst (bezel
        // bleed, extreme bytes on the bend of the camera tone curve) and matter most
        const lowT = a === 0 && b === 0 ? 1 : a === 0 || b === 0 || (a < lowN && b < lowN) ? lowTrust : 1;
        planes[c][o] = Math.sign(e) * C0 * Math.expm1(Math.abs(e) * norm * kappa) * ta * tb * lowT;
      }
    }
  }
  // inverse DCT of the K x Kh corner, sampled on a K x Kh grid spanning the whole screen
  const scale = 1 / (4 * W * H);
  const row = new Float64Array(K), rowo = new Float64Array(K), col = new Float64Array(Kh), colo = new Float64Array(Kh);
  for (const P of planes) {
    for (let b = 0; b < Kh; b++) { for (let a = 0; a < K; a++) row[a] = P[b * K + a]; dct3(row, rowo); for (let a = 0; a < K; a++) P[b * K + a] = rowo[a]; }
    for (let a = 0; a < K; a++) { for (let b = 0; b < Kh; b++) col[b] = P[b * K + a]; dct3(col, colo); for (let b = 0; b < Kh; b++) P[b * K + a] = colo[b] * scale; }
  }
  const cg = opt.lumaOnly ? 0 : opt.chroma ?? 1;
  let lo = 0, hi = 1;
  if (opt.autoLevels) {
    const ys = Float64Array.from(planes[0]).sort();
    lo = ys[Math.floor(n * 0.01)]; hi = ys[Math.floor(n * 0.99)];
    if (!(hi > lo)) { lo = 0; hi = 1; }
  }
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const y = (planes[0][i] - lo) / (hi - lo), cb = (planes[1][i] * cg) / (hi - lo), cr = (planes[2][i] * cg) / (hi - lo);
    for (let o = 0; o < 3; o++) rgba[4 * i + o] = 255 * (YCC2RGB[o][0] * y + YCC2RGB[o][1] * cb + YCC2RGB[o][2] * cr);
    rgba[4 * i + 3] = 255;
  }
  return { rgba, K, Kh, slope, slopeOk, med, coverage: has.reduce((s, v) => s + v, 0) / n };
}

// ---------------------------------------------------------------- whole frame
export function processFrame(rgba, w, h, state, opt) {
  const t0 = performance.now();
  // Lattice signal: brightness (pixel gaps, subpixel stripes) or edge energy (every pixel boundary;
  // survives blur that washes out the gaps). Peaks come from brightness; each axis's phase from
  // whichever signal fits more cleanly.
  const L = luminance(rgba, w, h);
  const mode = opt.latticeSignal ?? 'auto';
  let E = null;
  const getE = () => (E ??= edgeEnergy(L, w, h));
  if (!state.coarse || state.frame % (opt.regridEvery ?? 15) === 0 || state.lost) {
    state.coarse = coarseGrid(mode === 'edge' ? getE() : L, w, h, opt);
  }
  state.frame = (state.frame ?? 0) + 1;
  const res = { ok: false };
  if (!state.coarse) { state.lost = true; res.msg = 'no pixel grid found'; return res; }
  const fitAxis = (f) => {
    if (mode === 'lum') return phaseField(L, w, h, f, opt);
    if (mode === 'edge') return phaseField(getE(), w, h, f, opt);
    const g1 = phaseField(L, w, h, f, opt);
    if (g1.rms < 0.03 && g1.patches >= 0.5 * g1.of) return g1;
    const g2 = phaseField(getE(), w, h, f, opt);
    return g2.rms < g1.rms ? g2 : g1;
  };
  const ga = fitAxis(state.coarse.a), gb = fitAxis(state.coarse.b);
  res.ga = ga; res.gb = gb;
  res.pitch = 1 / Math.hypot(ga.fx, ga.fy);
  res.angle = (Math.atan2(ga.fy, ga.fx) * 180) / Math.PI;
  // per-channel sampling windows from the subpixel profile (or one shared window)
  let offA = [0, 0, 0], offB = [0, 0, 0], qA = [opt.q ?? 0.3, opt.q ?? 0.3, opt.q ?? 0.3], qB = [...qA];
  if (opt.subpixel !== false) {
    const prof = subpixelProfile(rgba, w, h, ga, gb);
    res.profile = prof;
    const qs = opt.qScale ?? 0.25;
    offA = prof.u.map((p) => p.off); offB = prof.v.map((p) => p.off);
    qA = prof.u.map((p) => Math.max(0.06, Math.min(0.42, qs * p.width)));
    qB = prof.v.map((p) => Math.max(0.06, Math.min(0.42, qs * p.width)));
  }
  offA = offA.map((v) => v + (opt.phaseA ?? 0)); offB = offB.map((v) => v + (opt.phaseB ?? 0));
  // all channels of one screen pixel must share an index: re-centre offsets around green
  offA = offA.map((v) => v - Math.round(v - offA[1])); offB = offB.map((v) => v - Math.round(v - offB[1]));
  res.offA = offA; res.offB = offB; res.qA = qA; res.qB = qB;
  const acc = accumulate(rgba, w, h, ga, gb, { offA, offB, qA, qB });
  if (!acc) { state.lost = true; res.msg = 'grid fit failed'; return res; }
  res.acc = acc;
  const S = toScreen(acc, opt.rot ?? 0, opt.mirror ?? false);
  res.S = S;
  const org = opt.autoOrigin === false ? { s: null, t: null } : findOrigin(S);
  res.origin = org;
  const os = (org.s ?? state.lastS ?? 0) + (opt.nudgeX ?? 0), ot = (org.t ?? state.lastT ?? 0) + (opt.nudgeY ?? 0);
  if (org.s == null || org.t == null) {
    res.msg = org.s == null && org.t == null ? 'screen corner not in view' : org.s == null ? 'left screen edge not in view' : 'top screen edge not in view';
    state.lost = res.ga.rms > 0.1;
    res.ms = performance.now() - t0;
    return res;
  }
  state.lost = false;
  res.os = os; res.ot = ot;
  if (!opt.accumulate) state.buf.reset();
  res.added = state.buf.add(S, os, ot, opt.alpha ?? 0.3);
  res.ok = true;
  res.ms = performance.now() - t0;
  return res;
}
