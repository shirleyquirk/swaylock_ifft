"""Synthetic phone camera looking at the top-left corner of a tryseven'd screen.

python3 test/camsim.py screen7.rgba 1920 1080 out_dir [--frames N] [--mag 4.5] ...
Writes out_dir/frame_XX.rgba (camera RGBA), out_dir/meta.json and, with --y4m, a y4m video for
Chromium's fake camera.

Model (deliberately pessimistic where it is cheap to be):
  * screen: RGB vertical subpixel stripes (fill 85%), 8% dark gap between rows, sRGB -> linear light;
  * geometry: camera px per screen px (mag), rotation, mild perspective, corner position, per-frame hand jitter;
  * optics: Gaussian blur; sensor: colour crosstalk, white balance, shot + read noise;
  * ISP: tone curve (sRGB-ish with an S bend), 8-bit.
"""
import argparse, json, os
import numpy as np
from scipy.ndimage import gaussian_filter

ap = argparse.ArgumentParser()
ap.add_argument('screen'); ap.add_argument('W', type=int); ap.add_argument('H', type=int); ap.add_argument('out')
ap.add_argument('--cw', type=int, default=1920); ap.add_argument('--ch', type=int, default=1080)
ap.add_argument('--mag', type=float, default=4.5)
ap.add_argument('--rot', type=float, default=2.0, help='degrees')
ap.add_argument('--persp', type=float, default=2e-5)
ap.add_argument('--corner', type=float, nargs=2, default=[140, 110], help='camera position of screen pixel (0,0) top-left corner')
ap.add_argument('--blur', type=float, default=0.9)
ap.add_argument('--noise', type=float, default=2.0, help='read noise, 8-bit levels')
ap.add_argument('--crosstalk', type=float, default=0.06)
ap.add_argument('--frames', type=int, default=1)
ap.add_argument('--jitter', type=float, default=6.0, help='camera px of hand shake per frame')
ap.add_argument('--y4m', action='store_true')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
os.makedirs(a.out, exist_ok=True)

scr = np.frombuffer(open(a.screen, 'rb').read(), np.uint8).reshape(a.H, a.W, 4)[..., :3]
lin = (scr / 255.0) ** 2.2                                     # emitted linear light per channel
X = np.array([[1 - 2 * a.crosstalk, a.crosstalk, a.crosstalk],
              [a.crosstalk, 1 - 2 * a.crosstalk, a.crosstalk],
              [a.crosstalk, a.crosstalk, 1 - 2 * a.crosstalk]])
wb = np.array([1.08, 1.0, 0.93])
SS = 4                                                         # supersamples per axis

def tone(x):
    s = np.clip(x, 0, 1) ** (1 / 2.2)
    return np.clip(s + 0.06 * np.sin(2 * np.pi * s) * s * (1 - s) * 4, 0, 1)   # gentle S bend

def render(corner, rot, persp):
    th = np.deg2rad(rot)
    A = a.mag * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Hm = np.array([[A[0, 0], A[0, 1], corner[0]], [A[1, 0], A[1, 1], corner[1]], [persp, persp * 0.5, 1.0]])
    Hi = np.linalg.inv(Hm)
    img = np.zeros((a.ch, a.cw, 3))
    off = (np.arange(SS) + 0.5) / SS
    for y0 in range(0, a.ch, 60):
        ys = np.arange(y0, min(a.ch, y0 + 60))
        yy, xx = np.meshgrid((ys[:, None] + off[None, :]).ravel(), (np.arange(a.cw)[:, None] + off[None, :]).ravel(), indexing='ij')
        d = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
        sx = (Hi[0, 0] * xx + Hi[0, 1] * yy + Hi[0, 2]) / d
        sy = (Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]) / d
        px, py = np.floor(sx).astype(int), np.floor(sy).astype(int)
        fx, fy = sx - px, sy - py
        on = (px >= 0) & (py >= 0) & (px < a.W) & (py < a.H)
        sub = np.minimum((fx * 3).astype(int), 2)
        lit = on & ((fx * 3 - sub) < 0.85) & (fy > 0.04) & (fy < 0.96)
        val = np.zeros(sx.shape + (3,))
        pxc, pyc = np.clip(px, 0, a.W - 1), np.clip(py, 0, a.H - 1)
        e = lin[pyc, pxc, sub] * lit
        val[np.arange(sx.shape[0])[:, None], np.arange(sx.shape[1])[None, :], sub] = e
        val[~on] = 0.004                                          # bezel: almost black
        img[ys] = val.reshape(len(ys), SS, a.cw, SS, 3).mean((1, 3))
    img = img @ X.T * wb * 2.2                                      # sensor colour + exposure
    img = np.stack([gaussian_filter(img[..., c], a.blur) for c in range(3)], -1)
    img = img + rng.normal(0, 1, img.shape) * np.sqrt(np.maximum(img, 0) / 2000)   # shot noise
    out = tone(img) * 255 + rng.normal(0, a.noise, img.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

frames = []
for f in range(a.frames):
    c = np.array(a.corner) + (rng.normal(0, a.jitter, 2) if f else 0)
    r = a.rot + (rng.normal(0, 0.3) if f else 0)
    im = render(c, r, a.persp)
    rgba = np.concatenate([im, np.full(im.shape[:2] + (1,), 255, np.uint8)], -1)
    open(f'{a.out}/frame_{f:02d}.rgba', 'wb').write(rgba.tobytes())
    frames.append({'corner': c.tolist(), 'rot': r})
    if f == 0:
        from PIL import Image
        Image.fromarray(im).save(f'{a.out}/frame_00.png')
json.dump({'cw': a.cw, 'ch': a.ch, 'W': a.W, 'H': a.H, 'mag': a.mag, 'persp': a.persp, 'frames': frames}, open(f'{a.out}/meta.json', 'w'))

if a.y4m:
    with open(f'{a.out}/cam.y4m', 'wb') as fo:
        fo.write(f'YUV4MPEG2 W{a.cw} H{a.ch} F30:1 Ip A1:1 C420jpeg\n'.encode())
        for f in range(a.frames):
            im = np.frombuffer(open(f'{a.out}/frame_{f:02d}.rgba', 'rb').read(), np.uint8).reshape(a.ch, a.cw, 4)[..., :3].astype(float)
            Y = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
            Cb = 128 - 0.168736 * im[..., 0] - 0.331264 * im[..., 1] + 0.5 * im[..., 2]
            Cr = 128 + 0.5 * im[..., 0] - 0.418688 * im[..., 1] - 0.081312 * im[..., 2]
            sub = lambda c: c.reshape(a.ch // 2, 2, a.cw // 2, 2).mean((1, 3))
            fo.write(b'FRAME\n')
            for p in (Y, sub(Cb), sub(Cr)):
                fo.write(np.clip(np.round(p), 0, 255).astype(np.uint8).tobytes())
print('wrote', a.frames, 'frames to', a.out)
