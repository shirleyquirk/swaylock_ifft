import { Engine } from './engine.js';
import * as Settings from './settings.js';
import { buildPanel } from './ui.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const video = $('video');
const status = $('status');
const overlay = $('overlay');
const toast = $('toast');

const s = Settings.load();
let engine;
let panel;
let frozen = false;
let snapshot = false;
let stream = null;
let cameraState = 'idle'; // idle | starting | live | error
let image = null;

function say(msg, ms = 2200) {
  toast.textContent = msg;
  toast.hidden = false;
  clearTimeout(say.t);
  say.t = setTimeout(() => { toast.hidden = true; }, ms);
}

function fatal(msg) {
  overlay.hidden = false;
  overlay.querySelector('.msg').textContent = msg;
  overlay.querySelector('.actions').hidden = true;
}

try {
  engine = new Engine(canvas);
} catch (e) {
  fatal(String(e.message || e));
  throw e;
}
window.app = { engine, settings: s };

canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();
  fatal('The GPU context was lost (memory pressure?). Reload the page; a smaller FFT size helps.');
});

// ---------------------------------------------------------------- camera
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    say('No camera API here. Camera needs HTTPS (or localhost).', 4000);
    cameraState = 'error';
    return;
  }
  stopCamera();
  cameraState = 'starting';
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: s.facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    video.srcObject = stream;
    video.muted = true;
    video.setAttribute('playsinline', '');
    await video.play();
    cameraState = 'live';
    overlay.hidden = true;
  } catch (e) {
    cameraState = 'error';
    say(`Camera: ${e.name || ''} ${e.message || e}`, 5000);
  }
}

function stopCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  cameraState = 'idle';
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible' || s.source !== 'camera') return;
  const ended = !stream || stream.getVideoTracks().every((t) => t.readyState === 'ended');
  if (ended && cameraState !== 'idle') startCamera();
});

// ---------------------------------------------------------------- test pattern
const pat = document.createElement('canvas');
pat.width = pat.height = 512;
const pctx = pat.getContext('2d');
function drawPattern(t) {
  const g = pctx.createLinearGradient(0, 0, 512, 512);
  g.addColorStop(0, '#16213a');
  g.addColorStop(1, '#3a1630');
  pctx.fillStyle = g;
  pctx.fillRect(0, 0, 512, 512);
  pctx.strokeStyle = 'rgba(255,255,255,0.15)';
  pctx.lineWidth = 2;
  for (let i = 32; i < 512; i += 32) {
    pctx.beginPath(); pctx.moveTo(i, 0); pctx.lineTo(i, 512); pctx.stroke();
    pctx.beginPath(); pctx.moveTo(0, i); pctx.lineTo(512, i); pctx.stroke();
  }
  const cols = ['#ff5a36', '#ffd23f', '#3bceac', '#4d7cff', '#e34fd0'];
  cols.forEach((c, i) => {
    const a = t * 0.6 + (i * Math.PI * 2) / cols.length;
    pctx.fillStyle = c;
    pctx.beginPath();
    pctx.arc(256 + Math.cos(a) * 140, 256 + Math.sin(a * 1.3) * 120, 46 + 12 * Math.sin(t + i), 0, Math.PI * 2);
    pctx.fill();
  });
  pctx.fillStyle = '#fff';
  pctx.font = 'bold 120px system-ui, sans-serif';
  pctx.textAlign = 'center';
  pctx.textBaseline = 'middle';
  pctx.fillText('FFT', 256, 260);
}

// ---------------------------------------------------------------- image file
const file = $('file');
file.onchange = async () => {
  const f = file.files?.[0];
  if (!f) return;
  try {
    image = await createImageBitmap(f);
    s.source = 'image';
    panel.refresh();
    engine.setSource(image, image.width, image.height);
    engine.resetSolver();
    overlay.hidden = true;
  } catch (e) { say(`Could not read image: ${e.message || e}`); }
};

// ---------------------------------------------------------------- settings / UI
let saveT;
function onChange(key) {
  clearTimeout(saveT);
  saveT = setTimeout(() => Settings.save(s), 300);
  if (key === 'source' || key === null) applySource();
  if (key === 'facing' && s.source === 'camera' && cameraState !== 'idle') startCamera();
  if (key === 'facing') s.mirror = s.facing === 'user';
  if (key === 'facing') panel.refresh();
  if (key === 'size' || key === 'mode' || key === null) engine.resetSolver();
}

function applySource() {
  if (s.source !== 'camera') stopCamera();
  if (s.source === 'image' && image) engine.setSource(image, image.width, image.height);
  if (s.source === 'image' && !image) say('Choose an image in Source.');
  if (s.source === 'camera' && cameraState === 'idle') overlay.hidden = false;
  else overlay.hidden = true;
}

panel = buildPanel($('panel'), s, onChange, {
  share: async () => {
    const url = Settings.shareURL(s);
    try { await navigator.clipboard.writeText(url); say('Link with these settings copied.'); }
    catch (e) { prompt('Copy this link:', url); }
  },
  pickImage: () => file.click(),
});

// toolbar
$('btn-start').onclick = () => { s.source = 'camera'; panel.refresh(); overlay.hidden = true; startCamera(); };
$('btn-pattern').onclick = () => { s.source = 'pattern'; panel.refresh(); applySource(); };
$('btn-flip').onclick = () => {
  s.facing = s.facing === 'user' ? 'environment' : 'user';
  onChange('facing');
};
$('btn-freeze').onclick = (e) => {
  frozen = !frozen;
  e.currentTarget.classList.toggle('on', frozen);
  say(frozen ? 'Frozen: the solver keeps iterating on this frame.' : 'Live.');
};
$('btn-reset').onclick = () => { engine.resetSolver(); say('Solver reset.'); };
$('btn-snap').onclick = () => { snapshot = true; };
$('btn-panel').onclick = () => document.body.classList.toggle('panel-closed');
$('btn-view').onclick = () => {
  const order = ['output', 'split', 'input', 'encoded', 'magnitude'];
  s.view = order[(order.indexOf(s.view) + 1) % order.length];
  panel.refresh();
  onChange('view');
  say(`View: ${s.view}`, 1000);
};

applySource();

// ---------------------------------------------------------------- frame loop
let last = performance.now();
let fpsAcc = 0, fpsN = 0, fps = 0;
const t0 = performance.now();

function frame(now) {
  requestAnimationFrame(frame);
  const dt = now - last;
  last = now;
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 500) { fps = (1000 * fpsN) / fpsAcc; fpsAcc = 0; fpsN = 0; }

  engine.resize(s.size);
  const t = (now - t0) / 1000;
  if (!frozen) {
    const camReady = s.source === 'camera' && cameraState === 'live' && video.readyState >= 2 && video.videoWidth > 0;
    if (camReady) engine.setSource(video, video.videoWidth, video.videoHeight);
    else if (s.source === 'pattern' || (s.source === 'camera' && !camReady) || (s.source === 'image' && !image)) {
      drawPattern(t);
      engine.setSource(pat, 512, 512);
    }
  }
  try {
    engine.render(s, t);
  } catch (e) {
    console.error(e);
    fatal(String(e.message || e));
    return;
  }
  if (snapshot) {
    snapshot = false;
    canvas.toBlob((b) => {
      if (!b) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = `ifft-${Date.now()}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }, 'image/png');
  }
  status.textContent = `${fps.toFixed(0)} fps · ${s.size}² · ${engine.stats.passes} passes · ${engine.fmt.name}`;
}
// ?noloop leaves the engine idle for automated tests
if (!new URLSearchParams(location.search).has('noloop')) requestAnimationFrame(frame);
