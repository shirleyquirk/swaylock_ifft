// node test/live7-e2e.mjs <cam.y4m> <screen7.rgba> W H <outdir>
// Opens live7.html with Chromium's fake camera playing the y4m, lets it run, then scores the pixels it read.
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const [y4m, enc7, Ws, Hs, out] = process.argv.slice(2);
const W = +Ws, H = +Hs;
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = http.createServer((req, res) => {
  let f = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (f.endsWith('/')) f += 'index.html';
  if (!f.startsWith(root) || !fs.existsSync(f)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' });
  fs.createReadStream(f).pipe(res);
}).listen(0);
const base = `http://localhost:${server.address().port}/`;
const scr = new Uint8Array(fs.readFileSync(enc7));

const browser = await chromium.launch({
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${y4m}`],
});
const errors = [];
for (const [vw, vh, tag] of [[1280, 860, 'desktop'], [390, 844, 'phone']]) {
  const ctx = await browser.newContext({ viewport: { width: vw, height: vh }, permissions: ['camera'], isMobile: tag === 'phone', hasTouch: tag === 'phone', deviceScaleFactor: tag === 'phone' ? 2 : 1 });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`${tag}: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`${tag}: ${m.text()}`); });
  await page.goto(base + 'live7.html');
  await page.evaluate(([w, h]) => { localStorage.clear(); }, [W, H]);
  await page.reload();
  await page.screenshot({ path: `${out}/live7-${tag}-start.png` });
  await page.click('#btn-camera');
  await page.waitForFunction(() => window.live7?.last?.ok, null, { timeout: 60000 });
  await page.waitForTimeout(tag === 'desktop' ? 15000 : 6000);
  const r = await page.evaluate(() => {
    const { state, last, decode } = window.live7;
    const [ka, kb] = state.buf.extent();
    return { ka, kb, max: state.buf.max, val: Array.from(state.buf.val.subarray(0, state.buf.max * kb * 3)), wt: Array.from(state.buf.wt.subarray(0, state.buf.max * kb)),
      pitch: last.pitch, ms: last.ms, status: document.getElementById('status').textContent, msg: document.getElementById('msg').textContent, K: decode?.K, Kh: decode?.Kh };
  });
  let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let b = 0; b < r.kb; b++) for (let a = 0; a < r.ka; a++) {
    const k = b * r.max + a;
    if (!(r.wt[k] > 0)) continue;
    const x = r.val[3 * k], y = scr[4 * (b * W + a)];
    n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
  }
  const corr = (sxy / n - (sx / n) * (sy / n)) / Math.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2));
  console.log(`${tag}: read ${r.ka}x${r.kb} pixels, Y-byte correlation with truth ${corr.toFixed(3)} | ${r.status} | ${r.msg}`);
  await page.screenshot({ path: `${out}/live7-${tag}-running.png` });
  const thumbPng = await page.evaluate(() => document.getElementById('thumb').toDataURL('image/png'));
  fs.writeFileSync(`${out}/live7-${tag}-thumb.png`, Buffer.from(thumbPng.split(',')[1], 'base64'));
  await ctx.close();
}
console.log('page errors:', errors.length ? errors : 'none');
await browser.close();
server.close();
