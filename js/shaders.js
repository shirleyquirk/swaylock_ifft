// All GLSL lives here. Conventions:
//  - Every N x N internal texture is indexed by integer texel (x, y) with y = image row from the TOP.
//    Only the final screen passes flip.
//  - A "field" is two RGBA float textures written together via MRT:
//      A = (H.re, H.im, S.re, S.im)   B = (V.re, V.im, 0, 0)
//  - Spectra are stored with orthonormal scaling: stored = raw / N, where raw is the
//    unnormalised FFTW-style DFT the C code uses. Each FFT stage multiplies by 1/sqrt(2).
//  - Decoded magnitudes are stored in the log domain L = log(1 + raw) as intervals [lo, hi].
//    Every |F| is physically bounded by sum|x| <= N^2 (all channels live in [0,1]), so every
//    interval is capped at Lmax = log(1 + N^2 / magScale). An interval that is open above
//    (clipped or unknown) is stored with hi NEGATED: hi < 0 means [lo, -hi] with lo a hard floor.

export const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const float PI = 3.14159265358979;
const float TAU = 6.28318530717959;
const float BIG = 1.0e6;
`;

const COMMON = `
uniform int uN;
ivec2 negk(ivec2 k) { return (ivec2(uN) - k) & ivec2(uN - 1); }
vec2 signedk(ivec2 k) {
  return vec2(k.x <= uN / 2 ? k.x : k.x - uN, k.y <= uN / 2 ? k.y : k.y - uN);
}
uint pcg(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v.x ^ v.y ^ v.z;
}
float h01(ivec2 k, int seed) {
  return float(pcg(uvec3(uvec2(k), uint(seed))) >> 8u) / 16777216.0;
}
// odd function of k -> phases built from it keep the inverse transform real
float antisym(ivec2 k, int seed) { return h01(k, seed) - h01(negk(k), seed); }

uniform float uPriorA;   // radial prior amplitude, raw units
uniform float uAlpha;    // radial prior exponent
float Lprior(ivec2 k) {
  float r = max(length(signedk(k)), 0.5);
  return log(1.0 + uPriorA * pow(r, -uAlpha));
}
`;

// C-exact rgb_to_hsv (h in degrees)
const RGB2HSV = `
vec3 rgb2hsv_c(vec3 c) {
  float mx = max(max(c.r, c.g), c.b);
  float mn = min(min(c.r, c.g), c.b);
  float d = mx - mn;
  float h = 0.0, s = 0.0;
  if (mx == 0.0) return vec3(0.0, 0.0, 0.0);
  s = d / mx;
  if (d == 0.0) h = 0.0;
  else if (mx == c.r) h = 60.0 * mod((c.g - c.b) / d, 6.0);
  else if (mx == c.g) h = 60.0 * ((c.b - c.r) / d + 2.0);
  else h = 60.0 * ((c.r - c.g) / d + 4.0);
  if (h < 0.0) h += 360.0;
  return vec3(h, s, mx);
}
`;
// note: C fmod keeps the sign of the dividend, GLSL mod does not. For (g-b)/d in [-1,1]
// C gives a negative value which is then +360'd; GLSL gives value+6 directly. Same result.

const HSV2RGB = `
vec3 hsv2rgb_raw(float h, float s, float v) {
  float c = v * s;
  float x = c * (1.0 - abs(mod(h / 60.0, 2.0) - 1.0));
  float m = v - c;
  vec3 r;
  if (h >= 0.0 && h < 60.0) r = vec3(c, x, 0.0);
  else if (h >= 60.0 && h < 120.0) r = vec3(x, c, 0.0);
  else if (h >= 120.0 && h < 180.0) r = vec3(0.0, c, x);
  else if (h >= 180.0 && h < 240.0) r = vec3(0.0, x, c);
  else if (h >= 240.0 && h < 300.0) r = vec3(x, 0.0, c);
  else r = vec3(c, 0.0, x);
  return r + m;
}
`;

// ---------------------------------------------------------------- FFT
// One radix-2 Stockham stage along x or y, on 4 complex signals at once.
export const FFT = HEAD + `
uniform sampler2D uA, uB;
uniform sampler2D uTw;   // N x 1: (cos, sin)(2 pi j / N), computed in double on the CPU
uniform int uN, uSub, uHoriz;
uniform float uSign, uScale;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int idx = uHoriz == 1 ? p.x : p.y;
  int hs = uSub / 2;
  int e = (idx / uSub) * hs + (idx % hs);
  int o = e + uN / 2;
  ivec2 pe = uHoriz == 1 ? ivec2(e, p.y) : ivec2(p.x, e);
  ivec2 po = uHoriz == 1 ? ivec2(o, p.y) : ivec2(p.x, o);
  // GPU sin/cos is too coarse on some drivers (SwiftShader ~1e-4, many mobile GPUs worse)
  vec2 w = texelFetch(uTw, ivec2((idx % uSub) * (uN / uSub), 0), 0).xy;
  w.y *= uSign;
  vec4 ea = texelFetch(uA, pe, 0), oa = texelFetch(uA, po, 0);
  vec4 eb = texelFetch(uB, pe, 0), ob = texelFetch(uB, po, 0);
  oA = uScale * vec4(ea.xy + cmul(w, oa.xy), ea.zw + cmul(w, oa.zw));
  oB = uScale * vec4(eb.xy + cmul(w, ob.xy), eb.zw + cmul(w, ob.zw));
}`;

// ---------------------------------------------------------------- source crop
export const CROP = HEAD + `
uniform sampler2D uTex;
uniform vec2 uTexSize;
uniform float uZoom;
uniform int uMirror, uRot, uN;
out vec4 o;
void main() {
  vec2 f = gl_FragCoord.xy / float(uN) - 0.5;   // y down (row 0 = top)
  if (uMirror == 1) f.x = -f.x;
  for (int i = 0; i < 3; i++) if (i < uRot) f = vec2(-f.y, f.x);
  float side = min(uTexSize.x, uTexSize.y) / uZoom;
  vec2 px = uTexSize * 0.5 + f * side;
  o = vec4(texture(uTex, px / uTexSize).rgb, 1.0);
}`;

// ---------------------------------------------------------------- RGB -> HSV field (spatial)
export const TOHSV = HEAD + RGB2HSV + `
uniform sampler2D uTex;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
void main() {
  vec3 hsv = rgb2hsv_c(texelFetch(uTex, ivec2(gl_FragCoord.xy), 0).rgb);
  oA = vec4(hsv.x / 360.0, 0.0, hsv.y, 0.0);
  oB = vec4(hsv.z, 0.0, 0.0, 0.0);
}`;

// ---------------------------------------------------------------- forward effect (the C code)
export const ENCODE = HEAD + COMMON + HSV2RGB + `
uniform sampler2D uA, uB;
uniform int uShift;
uniform vec3 uGain;          // 50, 25, 50 in the C
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 k = (p + (uShift == 1 ? ivec2(uN / 2) : ivec2(0))) & ivec2(uN - 1);
  vec4 a = texelFetch(uA, k, 0), b = texelFetch(uB, k, 0);
  float n = float(uN);
  float hm = log(1.0 + length(a.xy) * n) * uGain.x;
  float sm = log(1.0 + length(a.zw) * n) * uGain.y;
  float vm = log(1.0 + length(b.xy) * n) * uGain.z;
  float h = mod(hm, 360.0);
  float s = min(1.0, sm / 255.0);
  float v = min(1.0, vm / 255.0);
  vec3 rgb = floor(hsv2rgb_raw(h, s, v) * 255.0) / 255.0;   // (uint8_t) truncation
  o = vec4(rgb, 1.0);
}`;

// ---------------------------------------------------------------- decode: RGB "spectrum" -> magnitude intervals
export const DECODE = HEAD + COMMON + RGB2HSV + `
uniform sampler2D uIn;
uniform int uShift, uHerm, uBranch, uK, uDC;
uniform vec3 uGain;
uniform float uRatio, uBias, uMagScale;
uniform vec3 uDCmean;
float Lmax;
layout(location = 0) out vec4 oLo;   // lo L for H,S,V ; w = hue half-width in L
layout(location = 1) out vec4 oHi;   // hi L for H,S,V ; w = hue degrees (-1 unknown)

struct Dec { vec3 lo; vec3 hi; float hw; float hd; };

Dec decodeOne(ivec2 q, ivec2 k) {
  Dec d;
  vec3 rgb = texelFetch(uIn, q, 0).rgb;
  vec3 qi = floor(rgb * 255.0 + 0.5);
  float qmax = max(max(qi.r, qi.g), qi.b);
  float qmin = min(min(qi.r, qi.g), qi.b);
  // V: v_final in [qmax, qmax+1)/255, L = 255 v / gV
  // clipping is only possible if the clip threshold is below the physical maximum
  if (qmax >= 255.0) { d.lo.z = 255.0 / uGain.z; d.hi.z = 255.0 / uGain.z < Lmax ? BIG : d.lo.z; }
  else { d.lo.z = qmax / uGain.z; d.hi.z = (qmax + 1.0) / uGain.z; }
  // S: min channel = v(1-s) truncated
  if (qmax <= 0.0) { d.lo.y = 0.0; d.hi.y = BIG; }
  else {
    float slo = clamp(1.0 - (qmin + 1.0) / qmax, 0.0, 1.0);
    float shi = clamp(1.0 - qmin / (qmax + 1.0), 0.0, 1.0);
    d.lo.y = 255.0 * slo / uGain.y;
    d.hi.y = qmin > 0.0 ? 255.0 * shi / uGain.y : (255.0 / uGain.y < Lmax ? BIG : 255.0 / uGain.y);
  }
  // H: degrees, wraps every 360 -> branch k
  float dl = qmax - qmin;
  if (dl <= 0.0) { d.lo.x = 0.0; d.hi.x = BIG; d.hw = 0.0; d.hd = -1.0; }
  else {
    float hd = rgb2hsv_c(qi / 255.0).x;
    float wdeg = 60.0 / dl;
    float kk;
    if (uBranch == 0) kk = float(uK);
    else {
      float Lp;
      if (uBranch == 1) Lp = Lprior(k);
      else {
        float Lv = d.hi.z > BIG * 0.1 ? clamp(Lprior(k), d.lo.z, Lmax) : mix(d.lo.z, d.hi.z, uBias);
        Lp = log(1.0 + uRatio * (exp(min(Lv, 80.0)) - 1.0));
      }
      kk = floor((uGain.x * Lp - hd) / 360.0 + 0.5);
    }
    // k = -1 is allowed only where the hue interval straddles 360 -> 0 (true |H| ~ 0 seen as ~359 deg)
    kk = max(kk, hd + wdeg > 360.0 ? -1.0 : 0.0);
    float c = hd + 360.0 * kk;
    d.lo.x = max(0.0, (c - wdeg) / uGain.x);
    d.hi.x = (c + wdeg) / uGain.x;
    d.hw = wdeg / uGain.x;
    d.hd = hd;
  }
  return d;
}

void main() {
  ivec2 k = ivec2(gl_FragCoord.xy);
  Lmax = log(1.0 + float(uN) * float(uN) / uMagScale);
  ivec2 off = uShift == 1 ? ivec2(uN / 2) : ivec2(0);
  ivec2 mk = negk(k);
  Dec a = decodeOne((k + off) & ivec2(uN - 1), k);
  Dec r = a;
  if (uHerm != 4) {   // 4 = none
    Dec b = decodeOne((mk + off) & ivec2(uN - 1), mk);
    bool leftSide = k.x <= uN / 2;
    // (WebGL forbids ?: on structs)
    if (uHerm == 2) { if (leftSide) r = a; else r = b; }        // C convention: left half is truth
    else if (uHerm == 3) { if (leftSide) r = b; else r = a; }   // right half is truth
    else {
      vec3 alo = 0.5 * (a.lo + b.lo), ahi = 0.5 * (a.hi + b.hi);
      if (uHerm == 0) {                            // intersect, fall back to average
        vec3 ilo = max(a.lo, b.lo), ihi = min(a.hi, b.hi);
        bvec3 ok = lessThanEqual(ilo, ihi);
        r.lo = mix(alo, ilo, vec3(ok));
        r.hi = mix(ahi, ihi, vec3(ok));
      } else { r.lo = alo; r.hi = ahi; }
      if (a.hd < 0.0) { r.hd = b.hd; r.hw = b.hw; }
    }
  }
  if (uDC == 1 && k == ivec2(0)) {
    float n2 = float(uN) * float(uN);
    vec3 L = log(1.0 + n2 * max(uDCmean, vec3(0.0)) / uMagScale);
    r.lo = L; r.hi = L; r.hd = -1.0; r.hw = 0.0;
  }
  for (int c = 0; c < 3; c++) {
    bool open = r.hi[c] > BIG * 0.1;
    float h = min(r.hi[c], max(r.lo[c], Lmax));
    r.hi[c] = open ? -h : h;
  }
  oLo = vec4(r.lo, r.hw);
  oHi = vec4(r.hi, r.hd);
}`;

// shared magnitude-target helpers
const MAGHELP = `
uniform sampler2D uLo, uHi;
uniform float uBias, uMagScale, uMaxStored, uRatio;
uniform int uUnkFill;    // 0 = radial prior, 1 = ratio * |V| for H/S bins with no information
float centreL(int c, vec4 lo, vec4 hi, ivec2 k) {
  float l = lo[c], h = hi[c];
  if (h >= 0.0) return c == 0 ? 0.5 * (l + h) : mix(l, h, uBias);
  // open interval [l, -h]
  if (c < 2 && l <= 0.0 && uUnkFill == 1) {
    // a black (or grey) encoded pixel carries no H / S information; borrow V's magnitude
    float Lv = hi.z >= 0.0 ? mix(lo.z, hi.z, uBias) : clamp(Lprior(k), lo.z, -hi.z);
    return clamp(log(1.0 + uRatio * (exp(min(Lv, 80.0)) - 1.0)), l, -h);
  }
  return clamp(Lprior(k), l, -h);
}
float storedFromL(float L) {
  return min((exp(min(L, 80.0)) - 1.0) * uMagScale / float(uN), uMaxStored);
}
`;

// ---------------------------------------------------------------- initial spectrum = |M| e^{i phi}
export const INIT = HEAD + COMMON + MAGHELP + `
uniform sampler2D uCamA, uCamB;
uniform int uPhase, uLock, uSeed;
uniform float uJitter, uDriftT;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
float basePhase(int c, ivec2 k, vec4 ca, vec4 cb) {
  if (uPhase == 1) return TAU * antisym(k, uSeed * 7 + c);
  if (uPhase == 2) {
    vec2 z = c == 0 ? ca.xy : (c == 1 ? ca.zw : cb.xy);
    return dot(z, z) > 1e-30 ? atan(z.y, z.x) : 0.0;
  }
  return 0.0;
}
void main() {
  ivec2 k = ivec2(gl_FragCoord.xy);
  vec4 lo = texelFetch(uLo, k, 0), hi = texelFetch(uHi, k, 0);
  vec4 ca = texelFetch(uCamA, k, 0), cb = texelFetch(uCamB, k, 0);
  vec2 z[3];
  float pv = basePhase(2, k, ca, cb);
  for (int c = 0; c < 3; c++) {
    float ph = (uLock == 1) ? pv : basePhase(c, k, ca, cb);
    ph += uJitter * TAU * antisym(k, uSeed * 7 + c + 101);
    ph += uDriftT * TAU * antisym(k, uSeed * 7 + c + 211);
    float m = storedFromL(centreL(c, lo, hi, k));
    z[c] = m * vec2(cos(ph), sin(ph));
  }
  oA = vec4(z[0], z[1]);
  oB = vec4(z[2], 0.0, 0.0);
}`;

// ---------------------------------------------------------------- Fourier-domain projection
export const PROJECT = HEAD + COMMON + MAGHELP + `
uniform sampler2D uA, uB;
uniform int uClipFree, uHueFree;
uniform float uSlack, uStrength;
uniform vec3 uGain;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
vec2 proj(vec2 G, int c, vec4 lo, vec4 hi, ivec2 k) {
  float mag = length(G);
  float L = log(1.0 + mag * float(uN) / uMagScale);
  float l = lo[c], h = hi[c], Lt;
  if (c == 0 && uHueFree == 1 && hi.w >= 0.0) {
    // let the solver pick the 360-degree branch nearest its current estimate
    float kk = floor((L * uGain.x - hi.w) / 360.0 + 0.5);
    kk = max(kk, hi.w + lo.w * uGain.x > 360.0 ? -1.0 : 0.0);
    float cen = (hi.w + 360.0 * kk) / uGain.x;
    Lt = clamp(L, max(0.0, cen - uSlack * lo.w), cen + uSlack * lo.w);
  } else if (h < 0.0) {
    Lt = uClipFree == 1 ? clamp(L, l, -h) : centreL(c, lo, hi, k);
  } else {
    float cen = c == 0 ? 0.5 * (l + h) : mix(l, h, uBias);
    Lt = clamp(L, cen - uSlack * (cen - l), cen + uSlack * (h - cen));
  }
  Lt = mix(L, Lt, uStrength);
  float nm = storedFromL(Lt);
  return mag > 1e-30 ? G * (nm / mag) : vec2(nm, 0.0);
}
void main() {
  ivec2 k = ivec2(gl_FragCoord.xy);
  vec4 lo = texelFetch(uLo, k, 0), hi = texelFetch(uHi, k, 0);
  vec4 a = texelFetch(uA, k, 0), b = texelFetch(uB, k, 0);
  oA = vec4(proj(a.xy, 0, lo, hi, k), proj(a.zw, 1, lo, hi, k));
  oB = vec4(proj(b.xy, 2, lo, hi, k), 0.0, 0.0);
}`;

// ---------------------------------------------------------------- object-domain constraint (ER / HIO)
const OBJHELP = `
uniform int uSupport, uHueBound;
uniform float uSupportR;
bool inSupport(ivec2 p) {
  if (uSupport == 0) return true;
  vec2 f = (vec2(p) + 0.5) / float(uN) * 2.0 - 1.0;
  if (uSupport == 1) return length(f) <= uSupportR;
  return max(abs(f.x), abs(f.y)) <= uSupportR;
}
// projection P onto the object constraint set, componentwise
vec4 projA(vec4 a, bool sup) {
  if (!sup) return vec4(0.0);
  float h = uHueBound == 1 ? clamp(a.x, 0.0, 1.0) : a.x;
  return vec4(h, 0.0, clamp(a.z, 0.0, 1.0), 0.0);
}
vec4 projB(vec4 b, bool sup) {
  if (!sup) return vec4(0.0);
  return vec4(clamp(b.x, 0.0, 1.0), 0.0, 0.0, 0.0);
}
`;

export const UPDATE = HEAD + COMMON + OBJHELP + `
uniform sampler2D uGA, uGB, uPA, uPB;   // g (previous input), g' (after Fourier projection)
uniform int uAlgo;                     // 0 = ER, 1 = HIO
uniform float uBeta;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
vec4 fix(vec4 v) { return vec4(isnan(v.x) ? 0.0 : v.x, isnan(v.y) ? 0.0 : v.y, isnan(v.z) ? 0.0 : v.z, isnan(v.w) ? 0.0 : v.w); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  bool sup = inSupport(p);
  vec4 ga = texelFetch(uGA, p, 0), gb = texelFetch(uGB, p, 0);
  vec4 pa = texelFetch(uPA, p, 0), pb = texelFetch(uPB, p, 0);
  vec4 Pa = projA(pa, sup), Pb = projB(pb, sup);
  if (uAlgo == 0) { oA = Pa; oB = Pb; return; }
  vec4 da = pa - Pa, db = pb - Pb;
  vec4 eps = vec4(1e-7);
  oA = fix(mix(ga - uBeta * da, pa, vec4(lessThan(abs(da), eps))));
  oB = fix(mix(gb - uBeta * db, pb, vec4(lessThan(abs(db), eps))));
}`;

export const COPY = HEAD + `
uniform sampler2D uA, uB;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
void main() { ivec2 p = ivec2(gl_FragCoord.xy); oA = texelFetch(uA, p, 0); oB = texelFetch(uB, p, 0); }`;

// ---------------------------------------------------------------- screen passes
const OUTHELP = HSV2RGB + `
uniform sampler2D uA, uB;
uniform int uPart, uSpace, uHueBoundOut;
uniform float uHueShift, uHueScale, uSatGain, uValGain, uGamma;
vec3 outColour(ivec2 p) {
  vec4 a = texelFetch(uA, p, 0), b = texelFetch(uB, p, 0);
  vec3 x = uPart == 1 ? vec3(length(a.xy), length(a.zw), length(b.xy)) : vec3(a.x, a.z, b.x);
  float h = uHueBoundOut == 1 ? clamp(x.x, 0.0, 1.0) : x.x;
  h = fract(h * uHueScale + uHueShift);
  float s = clamp(x.y * uSatGain, 0.0, 1.0);
  float v = pow(clamp(x.z * uValGain, 0.0, 1.0), uGamma);
  if (uSpace == 1) return vec3(h, s, v);
  return hsv2rgb_raw(h * 360.0, s, v);
}
`;

// screen: flip y; uMode 0 output, 1 blit tex, 2 magnitude view; uSplit draws a wipe
export const SCREEN = HEAD + COMMON + OUTHELP + `
uniform sampler2D uTex;      // RGBA8 to blit
uniform sampler2D uSA, uSB;  // spectrum for magnitude view
uniform sampler2D uHi;
uniform int uMode, uShift;
uniform float uSplit, uLref;
out vec4 o;
void main() {
  ivec2 s = ivec2(gl_FragCoord.xy);
  ivec2 p = ivec2(s.x, uN - 1 - s.y);
  int mode = uMode;
  if (uSplit > 0.0 && float(s.x) < uSplit * float(uN)) mode = 1;
  vec3 c;
  if (mode == 1) c = texelFetch(uTex, p, 0).rgb;
  else if (mode == 2) {
    ivec2 k = (p + (uShift == 1 ? ivec2(uN / 2) : ivec2(0))) & ivec2(uN - 1);
    vec4 a = texelFetch(uSA, k, 0), b = texelFetch(uSB, k, 0);
    float n = float(uN);
    vec3 L = log(1.0 + vec3(length(a.xy), length(a.zw), length(b.xy)) * n) / uLref;
    c = clamp(L, 0.0, 1.0);
    vec4 hi = texelFetch(uHi, k, 0);
    if (hi.z < 0.0) c = mix(c, vec3(1.0, 0.0, 1.0), 0.35);
  } else c = outColour(p);
  if (uSplit > 0.0 && abs(float(s.x) - uSplit * float(uN)) < 1.0) c = vec3(1.0);
  o = vec4(c, 1.0);
}`;

// auto-exposure probe: sample V (pre-gain, after object projection) on a 32x32 grid
export const PROBE = HEAD + `
uniform sampler2D uA, uB;
uniform int uN, uPart;
out vec4 o;
void main() {
  ivec2 g = ivec2(gl_FragCoord.xy);
  ivec2 p = (g * uN) / 32 + uN / 64;
  vec4 a = texelFetch(uA, p, 0), b = texelFetch(uB, p, 0);
  float v = uPart == 1 ? length(b.xy) : b.x;
  float s = uPart == 1 ? length(a.zw) : a.z;
  o = vec4(max(v, 0.0), max(s, 0.0), 0.0, 1.0);
}`;
