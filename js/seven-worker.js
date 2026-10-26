import { tryseven, untryseven } from './seven.js';

self.onmessage = (e) => {
  const { op, data, W, H, id } = e.data;
  const t = performance.now();
  const out = op === 'encode' ? tryseven(data, W, H) : untryseven(data, W, H);
  self.postMessage({ id, out, ms: performance.now() - t }, [out.buffer]);
};
