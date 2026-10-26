// tryseven / untryseven in plain JS, bit-compatible with seven/tryseven.c (FFTW REDFT10/REDFT01).
// DCTs of any length via Makhoul's reordering + a Bluestein FFT on power-of-two sizes.

const C0 = 1.0;
const YCC = [
  [0.299, 0.587, 0.114],
  [-0.168736, -0.331264, 0.5],
  [0.5, -0.418688, -0.081312],
];
const RGB = [
  [1.0, 0.0, 1.402],
  [1.0, -0.344136, -0.714136],
  [1.0, 1.772, 0.0],
];

// ---------------------------------------------------------------- radix-2 FFT (in place)
const r2cache = new Map();
function radix2(n) {
  let t = r2cache.get(n);
  if (t) return t;
  const rev = new Uint32Array(n);
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos((2 * Math.PI * i) / n); sin[i] = -Math.sin((2 * Math.PI * i) / n); }
  t = { rev, cos, sin };
  r2cache.set(n, t);
  return t;
}
function fft2(re, im, inverse) {
  const n = re.length;
  const { rev, cos, sin } = radix2(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) { let x = re[i]; re[i] = re[j]; re[j] = x; x = im[i]; im[i] = im[j]; im[j] = x; }
  }
  const sgn = inverse ? -1 : 1;
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let s = 0; s < n; s += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sgn * sin[k * step];
        const a = s + k, b = a + half;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

// ---------------------------------------------------------------- any-length FFT (Bluestein)
const bcache = new Map();
function bluestein(n) {
  let t = bcache.get(n);
  if (t) return t;
  let m = 1;
  while (m < 2 * n - 1) m <<= 1;
  const wr = new Float64Array(n), wi = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = (Math.PI * ((k * k) % (2 * n))) / n;   // exact mod keeps the angle small
    wr[k] = Math.cos(a); wi[k] = -Math.sin(a);         // w_k = e^{-i pi k^2 / n}
  }
  const br = new Float64Array(m), bi = new Float64Array(m);
  br[0] = wr[0]; bi[0] = -wi[0];
  for (let k = 1; k < n; k++) { br[k] = br[m - k] = wr[k]; bi[k] = bi[m - k] = -wi[k]; }
  fft2(br, bi, false);
  t = { m, wr, wi, br, bi, ar: new Float64Array(m), ai: new Float64Array(m) };
  bcache.set(n, t);
  return t;
}
// forward DFT (e^{-i}) of length n, in place
function fft(re, im) {
  const n = re.length;
  if ((n & (n - 1)) === 0) { fft2(re, im, false); return; }
  const { m, wr, wi, br, bi, ar, ai } = bluestein(n);
  ar.fill(0); ai.fill(0);
  for (let k = 0; k < n; k++) { ar[k] = re[k] * wr[k] - im[k] * wi[k]; ai[k] = re[k] * wi[k] + im[k] * wr[k]; }
  fft2(ar, ai, false);
  for (let k = 0; k < m; k++) { const x = ar[k] * br[k] - ai[k] * bi[k]; ai[k] = ar[k] * bi[k] + ai[k] * br[k]; ar[k] = x; }
  fft2(ar, ai, true);
  for (let k = 0; k < n; k++) {
    const xr = ar[k] / m, xi = ai[k] / m;
    re[k] = xr * wr[k] - xi * wi[k]; im[k] = xr * wi[k] + xi * wr[k];
  }
}

// ---------------------------------------------------------------- 1-D DCTs, FFTW conventions
// REDFT10 (DCT-II):  X_k = 2 sum_n x_n cos(pi k (2n+1) / 2N)
// REDFT01 (DCT-III): y_n = x_0 + 2 sum_{k>0} x_k cos(pi k (2n+1) / 2N)
const tcache = new Map();
function dctTables(n) {
  let t = tcache.get(n);
  if (t) return t;
  const c = new Float64Array(n), s = new Float64Array(n);
  for (let k = 0; k < n; k++) { c[k] = Math.cos((Math.PI * k) / (2 * n)); s[k] = Math.sin((Math.PI * k) / (2 * n)); }
  t = { c, s, re: new Float64Array(n), im: new Float64Array(n) };
  tcache.set(n, t);
  return t;
}
function dct2(x, out) {
  const n = x.length;
  const { c, s, re, im } = dctTables(n);
  const h = (n + 1) >> 1;
  for (let i = 0; i < h; i++) re[i] = x[2 * i];
  for (let i = 0; i < n >> 1; i++) re[n - 1 - i] = x[2 * i + 1];
  im.fill(0);
  fft(re, im);
  for (let k = 0; k < n; k++) out[k] = 2 * (re[k] * c[k] + im[k] * s[k]);
}
function dct3(x, out) {
  const n = x.length;
  const { c, s, re, im } = dctTables(n);
  // V_k = (x_k - i x_{n-k}) e^{i pi k / 2n}, x_n = 0; v = n * IDFT(V); then un-reorder
  for (let k = 0; k < n; k++) {
    const a = x[k], b = k === 0 ? 0 : x[n - k];
    re[k] = a * c[k] + b * s[k];
    im[k] = a * s[k] - b * c[k];
  }
  // inverse DFT via forward on the conjugate
  for (let k = 0; k < n; k++) im[k] = -im[k];
  fft(re, im);
  const h = (n + 1) >> 1;
  for (let i = 0; i < h; i++) out[2 * i] = re[i];
  for (let i = 0; i < n >> 1; i++) out[2 * i + 1] = re[n - 1 - i];
}

// 2-D separable transform on a row-major W x H plane, in place
function dct2d(plane, W, H, f) {
  const row = new Float64Array(W), rowo = new Float64Array(W);
  for (let y = 0; y < H; y++) {
    const o = y * W;
    for (let x = 0; x < W; x++) row[x] = plane[o + x];
    f(row, rowo);
    for (let x = 0; x < W; x++) plane[o + x] = rowo[x];
  }
  const col = new Float64Array(H), colo = new Float64Array(H);
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) col[y] = plane[y * W + x];
    f(col, colo);
    for (let y = 0; y < H; y++) plane[y * W + x] = colo[y];
  }
}

// ---------------------------------------------------------------- tryseven / untryseven on RGBA bytes
export function tryseven(rgba, W, H) {
  const n = W * H;
  const norm = Math.log1p((4 * W * H) / C0);
  const out = new Uint8ClampedArray(rgba.length);
  const plane = new Float64Array(n);
  for (let ch = 0; ch < 3; ch++) {
    const [a, b, c] = YCC[ch];
    for (let i = 0; i < n; i++) plane[i] = a * (rgba[4 * i] / 255) + b * (rgba[4 * i + 1] / 255) + c * (rgba[4 * i + 2] / 255);
    dct2d(plane, W, H, dct2);
    for (let i = 0; i < n; i++) {
      const v = plane[i];
      const e = (Math.sign(v) * Math.log1p(Math.abs(v) / C0)) / norm;
      out[4 * i + ch] = Math.floor(127.5 + 127.5 * e + 0.5);
    }
  }
  for (let i = 0; i < n; i++) out[4 * i + 3] = rgba[4 * i + 3];
  return out;
}

export function untryseven(rgba, W, H) {
  const n = W * H;
  const norm = Math.log1p((4 * W * H) / C0);
  const scale = 1 / (4 * W * H);
  const ycc = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  for (let ch = 0; ch < 3; ch++) {
    const plane = ycc[ch];
    for (let i = 0; i < n; i++) {
      const e = (rgba[4 * i + ch] - 127.5) / 127.5;
      plane[i] = Math.sign(e) * C0 * Math.expm1(Math.abs(e) * norm);
    }
    dct2d(plane, W, H, dct3);
  }
  const out = new Uint8ClampedArray(rgba.length);
  for (let i = 0; i < n; i++) {
    const y = ycc[0][i] * scale, cb = ycc[1][i] * scale, cr = ycc[2][i] * scale;
    for (let o = 0; o < 3; o++) out[4 * i + o] = Math.floor((RGB[o][0] * y + RGB[o][1] * cb + RGB[o][2] * cr) * 255 + 0.5);
    out[4 * i + 3] = rgba[4 * i + 3];
  }
  return out;
}

export function psnr(a, b) {
  let se = 0, n = 0;
  for (let i = 0; i < a.length; i++) { if ((i & 3) === 3) continue; const d = a[i] - b[i]; se += d * d; n++; }
  return 10 * Math.log10((255 * 255) / (se / n));
}
