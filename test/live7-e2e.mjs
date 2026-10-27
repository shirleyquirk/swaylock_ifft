// node test/live7-e2e.mjs <tryseven.png> <untryseven-reference.rgba> <outdir> [cam.y4m]
// 1. Loads a tryseven image into live7.html (exact fit) and compares the GPU decode with the C decoder.
// 2. Optionally runs the fake camera on a y4m and reports frame rate.
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const [png, refRgba, out, y4m] = process.argv.slice(2);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = http.createServer((req, res) => {
  let f = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (f.endsWith('/')) f += 'index.html';
  if (!f.startsWith(root) || !fs.existsSync(f)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' });
  fs.createReadStream(f).pipe(res);
}).listen(0);
const base = `http://localhost:${server.address().port}/live7.html`;
const args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'];
if (y4m) args.push(`--use-file-for-fake-video-capture=${y4m}`);
const browser = await chromium.launch({ args });
const errors = [];

const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 }, permissions: ['camera'] });
const page = await ctx.newPage();
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(base);
await page.evaluate(() => localStorage.clear());
await page.reload();
await page.setInputFiles('#file', png);
await page.waitForFunction(() => window.live7.grid, null, { timeout: 30000 });
await page.evaluate(() => {
  const { setVal } = window.live7;
  for (const [k, v] of Object.entries({ view: 0, taps: 1, centre: 127.5, gain: 1, gamma: 1, kappa: 1, taper: 0, contrast: 1, bright: 0, chroma: 1, autoLevels: false, autoCentre: false, autoGain: false })) setVal(k, v);
});
await page.waitForTimeout(800);
const got = await page.evaluate(() => {
  const { gl, grid } = window.live7;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  const px = new Uint8Array(grid.K * grid.Kh * 4);
  gl.readPixels(0, 0, grid.K, grid.Kh, gl.RGBA, gl.UNSIGNED_BYTE, px);
  return { K: grid.K, Kh: grid.Kh, px: Array.from(px) };
});
const ref = new Uint8Array(fs.readFileSync(refRgba));
let se = 0, worst = 0, n = 0;
for (let y = 0; y < got.Kh; y++) for (let x = 0; x < got.K; x++) for (let c = 0; c < 3; c++) {
  const g = got.px[4 * ((got.Kh - 1 - y) * got.K + x) + c], r = ref[4 * (y * got.K + x) + c], d = g - r;
  se += d * d; worst = Math.max(worst, Math.abs(d)); n++;
}
console.log(`exact image: GPU decode ${got.K}x${got.Kh} vs C untryseven: PSNR ${(10 * Math.log10(65025 / (se / n))).toFixed(1)} dB, worst ${worst} levels`);
await page.screenshot({ path: `${out}/live7s-image.png` });

if (y4m) {
  await page.click('#btn-camera');
  await page.waitForTimeout(4000);
  console.log('camera:', await page.textContent('#status'));
  await page.screenshot({ path: `${out}/live7s-camera.png` });
  const ph = await browser.newContext({ viewport: { width: 390, height: 844 }, permissions: ['camera'], isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
  const pp = await ph.newPage();
  pp.on('pageerror', (e) => errors.push('phone: ' + e.message));
  await pp.goto(base);
  await pp.evaluate(() => localStorage.clear());
  await pp.reload();
  await pp.click('#btn-camera');
  await pp.waitForTimeout(4000);
  console.log('phone camera:', await pp.textContent('#status'));
  await pp.screenshot({ path: `${out}/live7s-phone.png` });
}
console.log('page errors:', errors.length ? errors : 'none');
await browser.close();
server.close();
