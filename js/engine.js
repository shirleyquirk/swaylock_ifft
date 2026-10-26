import * as S from './shaders.js';

const ENUM = {
  herm: { intersect: 0, average: 1, left: 2, right: 3, none: 4 },
  hueBranch: { fixed: 0, prior: 1, vmatch: 2 },
  phase: { zero: 0, random: 1, camera: 2, warm: 3 },
  support: { full: 0, circle: 1, square: 2 },
};

export class Engine {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      antialias: false, alpha: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;

    // float render targets: prefer 32-bit, fall back to 16-bit
    if (gl.getExtension('EXT_color_buffer_float')) {
      this.fmt = { internal: gl.RGBA32F, type: gl.FLOAT, name: 'RGBA32F' };
    } else if (gl.getExtension('EXT_color_buffer_half_float')) {
      this.fmt = { internal: gl.RGBA16F, type: gl.HALF_FLOAT, name: 'RGBA16F' };
    } else {
      throw new Error('This GPU cannot render to float textures (EXT_color_buffer_float missing).');
    }
    this.maxStored = this.fmt.type === gl.FLOAT ? 1e30 : 6e4;

    this.vao = gl.createVertexArray();
    this.progs = {};
    for (const [name, src] of Object.entries({
      fft: S.FFT, crop: S.CROP, tohsv: S.TOHSV, encode: S.ENCODE, decode: S.DECODE,
      init: S.INIT, project: S.PROJECT, update: S.UPDATE, copy: S.COPY,
      screen: S.SCREEN, probe: S.PROBE,
    })) this.progs[name] = this.program(src, name);

    this.N = 0;
    this.srcTex = this.texture(1, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, true);
    this.srcSize = [1, 1];
    this.gValid = false;
    this.valGainAuto = 1;
    this.frameNo = 0;
    this.stats = { passes: 0 };
  }

  // ------------------------------------------------------------ GL plumbing
  program(fsSrc, name) {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s);
        throw new Error(`shader ${name}: ${log}`);
      }
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, S.VERT));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`link ${name}: ${gl.getProgramInfoLog(p)}`);
    const uniforms = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      uniforms[info.name] = { loc: gl.getUniformLocation(p, info.name), type: info.type };
    }
    return { p, uniforms, name };
  }

  texture(w, h, internal, format, type, linear = false) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    const f = linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  target(texs, w, h) {
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    texs.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
    gl.drawBuffers(texs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer incomplete (0x${st.toString(16)}) for ${this.fmt.name}`);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, w, h, texs };
  }

  field(N) {
    const gl = this.gl;
    const a = this.texture(N, N, this.fmt.internal, gl.RGBA, this.fmt.type);
    const b = this.texture(N, N, this.fmt.internal, gl.RGBA, this.fmt.type);
    const t = this.target([a, b], N, N);
    return { ...t, a, b };
  }

  rgba8(N) {
    const gl = this.gl;
    const t = this.texture(N, N, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    return { ...this.target([t], N, N), tex: t };
  }

  draw(prog, target, uniforms, viewport) {
    const gl = this.gl;
    const P = this.progs[prog];
    gl.useProgram(P.p);
    let unit = 0;
    for (const [name, val] of Object.entries(uniforms)) {
      const u = P.uniforms[name];
      if (!u) continue; // optimised out
      switch (u.type) {
        case gl.SAMPLER_2D:
          gl.activeTexture(gl.TEXTURE0 + unit);
          gl.bindTexture(gl.TEXTURE_2D, val);
          gl.uniform1i(u.loc, unit++);
          break;
        case gl.INT: case gl.BOOL: gl.uniform1i(u.loc, val | 0); break;
        case gl.FLOAT: gl.uniform1f(u.loc, val); break;
        case gl.FLOAT_VEC2: gl.uniform2fv(u.loc, val); break;
        case gl.FLOAT_VEC3: gl.uniform3fv(u.loc, val); break;
        default: throw new Error(`uniform type ${u.type} for ${name}`);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    const [w, h] = viewport || (target ? [target.w, target.h] : [this.canvas.width, this.canvas.height]);
    gl.viewport(0, 0, w, h);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.stats.passes++;
  }

  // ------------------------------------------------------------ resources
  resize(N) {
    if (N === this.N) return;
    const gl = this.gl;
    this.free();
    this.N = N;
    this.F = {};
    for (const k of ['ping', 'pong', 'cam', 'specA', 'specB', 'g', 'gNew', 'gPrime']) this.F[k] = this.field(N);
    this.src = this.rgba8(N);
    this.enc = this.rgba8(N);
    const lo = this.texture(N, N, this.fmt.internal, gl.RGBA, this.fmt.type);
    const hi = this.texture(N, N, this.fmt.internal, gl.RGBA, this.fmt.type);
    this.mag = { ...this.target([lo, hi], N, N), lo, hi };
    const tw = new Float32Array(N * 4);
    for (let j = 0; j < N; j++) { tw[4 * j] = Math.cos(2 * Math.PI * j / N); tw[4 * j + 1] = Math.sin(2 * Math.PI * j / N); }
    this.twiddle = this.texture(N, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, 1, gl.RGBA, gl.FLOAT, tw);
    const pt = this.texture(32, 32, this.fmt.internal, gl.RGBA, this.fmt.type);
    this.probeT = { ...this.target([pt], 32, 32), tex: pt };
    this.canvas.width = N;
    this.canvas.height = N;
    this.gValid = false;
    this.srcDirty = true;
  }

  free() {
    const gl = this.gl;
    const kill = (t) => { if (!t) return; gl.deleteFramebuffer(t.fbo); t.texs.forEach((x) => gl.deleteTexture(x)); };
    if (this.F) Object.values(this.F).forEach(kill);
    [this.src, this.enc, this.mag, this.probeT].forEach(kill);
    if (this.twiddle) gl.deleteTexture(this.twiddle);
  }

  // upload a video / image / canvas into the source texture
  setSource(el, w, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, el);
    this.srcSize = [w, h];
    this.rawSource = false;
    this.srcDirty = true;
  }

  // raw RGBA bytes, already N x N, row 0 = top (used by tests)
  setSourcePixels(N, bytes) {
    const gl = this.gl;
    this.resize(N);
    gl.bindTexture(gl.TEXTURE_2D, this.src.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    this.rawSource = true;
    this.srcDirty = true;
  }

  // ------------------------------------------------------------ FFT
  fft(src, dst, inverse) {
    const N = this.N;
    const { ping, pong } = this.F;
    const stages = [];
    for (const horiz of [1, 0]) for (let sub = 2; sub <= N; sub *= 2) stages.push([horiz, sub]);
    let input = src;
    stages.forEach(([horiz, sub], i) => {
      const out = i === stages.length - 1 ? dst : (i % 2 === 0 ? ping : pong);
      this.draw('fft', out, {
        uA: input.a, uB: input.b, uTw: this.twiddle, uN: N, uSub: sub, uHoriz: horiz,
        uSign: inverse ? 1 : -1, uScale: Math.SQRT1_2,
      });
      input = out;
    });
  }

  // ------------------------------------------------------------ one frame
  render(s, tSec) {
    const N = this.N;
    const F = this.F;
    this.stats.passes = 0;
    const gains = [s.gH, s.gS, s.gV];
    const magScale = Math.pow(10, s.magScale);
    const priorA = N * N * Math.pow(10, s.priorLevel);
    const shift = s.shift ? 1 : 0;
    const prior = { uN: N, uPriorA: priorA, uAlpha: s.priorAlpha };

    // 1. source -> N x N RGBA8
    if (!this.rawSource) {
      this.draw('crop', this.src, {
        uTex: this.srcTex, uTexSize: this.srcSize, uZoom: s.zoom,
        uMirror: s.mirror ? 1 : 0, uRot: s.rotate | 0, uN: N,
      });
    }

    // 2. forward FFT of the source (needed for round-trip, camera phase, encoded view)
    const roundtrip = s.mode === 'roundtrip';
    const needCam = roundtrip || s.phase === 'camera' || s.view === 'encoded';
    if (needCam) {
      this.draw('tohsv', F.specA, { uTex: this.src.tex });
      this.fft(F.specA, F.cam, false);
    }
    if (roundtrip || s.view === 'encoded') {
      this.draw('encode', this.enc, { uA: F.cam.a, uB: F.cam.b, uN: N, uShift: shift, uGain: gains });
    }
    const decIn = roundtrip ? this.enc : this.src;

    // 3. decode the "spectrum image" into magnitude intervals
    this.draw('decode', this.mag, {
      uIn: decIn.tex, ...prior, uShift: shift, uHerm: ENUM.herm[s.herm],
      uBranch: ENUM.hueBranch[s.hueBranch], uK: s.hueK, uDC: s.dcOverride ? 1 : 0,
      uGain: gains, uRatio: Math.pow(10, s.hueRatio), uBias: s.bias, uMagScale: magScale,
      uDCmean: [s.dcH, s.dcS, s.dcV],
    });

    const magU = {
      uLo: this.mag.lo, uHi: this.mag.hi, uBias: s.bias, uMagScale: magScale, uMaxStored: this.maxStored,
      uRatio: Math.pow(10, s.hueRatio), uUnkFill: s.unkFill === 'vmatch' ? 1 : 0,
    };

    // 4. initial estimate
    const warm = s.phase === 'warm' && this.gValid;
    let iters = s.iters | 0;
    if (!warm) {
      const ph = s.phase === 'warm' ? ENUM.phase.random : ENUM.phase[s.phase];
      this.draw('init', F.specA, {
        ...prior, ...magU, uCamA: F.cam.a, uCamB: F.cam.b,
        uPhase: ph, uLock: s.lockPhase ? 1 : 0, uSeed: s.seed | 0,
        uJitter: s.jitter, uDriftT: s.drift * tSec,
      });
      this.fft(F.specA, F.g, true);
      this.lastSpec = F.specA;
    } else if (iters === 0) iters = 1;

    // 5. iterate: g -> FFT -> project magnitudes -> IFFT -> object constraint
    const obj = {
      uN: N, uSupport: ENUM.support[s.support], uSupportR: s.supportR,
      uHueBound: s.hueBound === 'clamp' ? 1 : 0,
    };
    for (let i = 0; i < iters; i++) {
      this.fft(F.g, F.specA, false);
      this.draw('project', F.specB, {
        ...prior, ...magU, uA: F.specA.a, uB: F.specA.b,
        uClipFree: s.clipMode === 'free' ? 1 : 0, uHueFree: s.hueSolve ? 1 : 0,
        uSlack: s.slack, uStrength: s.strength, uGain: gains,
      });
      this.lastSpec = F.specB;
      this.fft(F.specB, F.gPrime, true);
      const er = s.algo === 'er' || (s.erEvery > 0 && (i + 1) % s.erEvery === 0);
      this.draw('update', F.gNew, {
        ...obj, uGA: F.g.a, uGB: F.g.b, uPA: F.gPrime.a, uPB: F.gPrime.b,
        uAlgo: er ? 0 : 1, uBeta: s.beta,
      });
      [F.g, F.gNew] = [F.gNew, F.g];
    }
    this.gValid = true;
    const out = iters > 0 ? F.gPrime : F.g;

    // 6. auto exposure
    const part = s.part === 'abs' ? 1 : 0;
    if (s.autoExpo && this.frameNo % 6 === 0) this.expose(out, part);
    this.frameNo++;
    const valGain = s.autoExpo ? this.valGainAuto * Math.pow(10, s.valGain) : Math.pow(10, s.valGain);

    // 7. screen
    const mode = { output: 0, split: 0, input: 1, encoded: 1, magnitude: 2 }[s.view];
    const blit = s.view === 'encoded' ? this.enc.tex : (s.view === 'input' ? decIn.tex : this.src.tex);
    this.draw('screen', null, {
      uN: N, uTex: blit, uSA: this.lastSpec.a, uSB: this.lastSpec.b, uHi: this.mag.hi,
      uMode: mode, uShift: shift, uSplit: s.view === 'split' ? 0.5 : 0,
      uLref: Math.log(1 + N * N * 0.5),
      uA: out.a, uB: out.b, uPart: part, uSpace: s.space === 'rgb' ? 1 : 0,
      uHueBoundOut: s.hueBound === 'clamp' ? 1 : 0,
      uHueShift: s.hueShift, uHueScale: s.hueScale, uSatGain: s.satGain,
      uValGain: valGain, uGamma: s.gamma,
    }, [N, N]);
    this.srcDirty = false;
  }

  expose(out, part) {
    const gl = this.gl;
    this.draw('probe', this.probeT, { uA: out.a, uB: out.b, uN: this.N, uPart: part });
    const buf = new Float32Array(32 * 32 * 4);
    try {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.probeT.fbo);
      gl.readPixels(0, 0, 32, 32, gl.RGBA, gl.FLOAT, buf);
    } catch (e) { return; }
    const v = [];
    for (let i = 0; i < 1024; i++) if (Number.isFinite(buf[i * 4])) v.push(buf[i * 4]);
    if (!v.length) return;
    v.sort((a, b) => a - b);
    const p98 = v[Math.floor(v.length * 0.98)];
    const target = p98 > 1e-9 ? 0.95 / p98 : this.valGainAuto;
    this.valGainAuto = Math.exp(0.7 * Math.log(this.valGainAuto) + 0.3 * Math.log(target));
    if (!Number.isFinite(this.valGainAuto)) this.valGainAuto = 1;
  }

  resetSolver() { this.gValid = false; }

  // ------------------------------------------------------------ test hooks
  readField(field) {
    const gl = this.gl;
    const N = this.N;
    const out = [];
    for (const t of [field.a, field.b]) {
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      const buf = new Float32Array(N * N * 4);
      gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, buf);
      gl.deleteFramebuffer(fb);
      out.push(buf);
    }
    return out;
  }

  readRGBA8(t) {
    const gl = this.gl;
    const N = this.N;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    const buf = new Uint8Array(N * N * 4);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    return buf;
  }

  writeField(field, a, b) {
    const gl = this.gl;
    const N = this.N;
    gl.bindTexture(gl.TEXTURE_2D, field.a);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.FLOAT, a);
    gl.bindTexture(gl.TEXTURE_2D, field.b);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.FLOAT, b);
  }
}
