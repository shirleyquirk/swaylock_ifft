# ifft·cam

Live camera → inverse of the swaylock-effects FFT effect (`reference/swaylock_effect.c`), in the browser.
WebGL2 only, no build step, no dependencies. Works on iOS Safari 15+, Chrome, Firefox.

## Deploy (GitHub Pages)

```sh
git init && git add . && git commit -m "ifft-cam"
git remote add origin git@github.com:<you>/ifft-cam.git
git push -u origin main
```

Repo → Settings → Pages → *Deploy from a branch* → `main` / `(root)`.
The site lands at `https://<you>.github.io/ifft-cam/`. HTTPS is required for the camera; Pages provides it.
`.nojekyll` is included so nothing gets mangled.

Local: any static server, e.g. `python3 -m http.server`, then `http://localhost:8000` (localhost counts as secure for the camera).

## What the forward effect throws away, and what the inverse does about it

The C does, per HSV channel: `|FFT|` → `log(1+·)·gain` → hue `fmod 360`, S/V `min(1, ·/255)` → HSV→RGB → truncate to 8 bits.

| Lost | Inverse's handling (all switchable in the UI) |
|---|---|
| Phase | Camera's own FFT phase (phase swap), random Hermitian phase, zero, or warm-started Fienup HIO/ER phase retrieval that carries its estimate across frames |
| Hue wraps every 360° | Branch per bin: fixed k, nearest radial 1/f prior, nearest ratio·\|V\| (cross-channel), or let the solver snap each bin to the branch nearest its current estimate |
| S/V clip at 1 | Clipped bins become a *floor*, not a value. Solver: free above it, or pinned to the prior |
| 8-bit truncation | Every pixel decodes to an *interval* of \|F\|, not a point. Solver projects into the interval (slack knob), not onto a midpoint |
| H/S on black/grey pixels | No information; fill from V's magnitude or the prior |
| Camera frame isn't conjugate-symmetric | Intersect ±k intervals, average, take the left half (as the C), right half, or ignore and get a complex output |

Physical bound used throughout: every channel lives in [0,1], so `|F(k)| ≤ N²`. That caps every open interval and decides whether clipping is even possible at a given gain.

## Controls

Everything is in the *Knobs* panel, persisted in `localStorage`. *Copy link* puts the non-default settings in the URL hash. *Help* shows a line per control. Double-click (desktop) a slider label to reset it.

- **Pipeline → Round trip** runs the C effect on the camera, then inverts it: ground truth for the inverse. *View → Split* puts source and output side by side.
- **Freeze** holds the frame while the solver keeps iterating.
- **FFT size** defaults to 256 on touch devices, 512 elsewhere. Each solver iteration costs two full 2-D FFTs (2·log₂N passes each).

## Implementation

- `js/shaders.js`: radix-2 Stockham FFT in fragment shaders (4 complex signals per pass via MRT), twiddles from a CPU-computed table since GPU `sin/cos` is too coarse on some drivers. Orthonormal scaling (1/√2 per stage) keeps magnitudes inside half-float range for the fallback path.
- Float targets: `RGBA32F` via `EXT_color_buffer_float`, falling back to `RGBA16F`.
- `js/engine.js`: pass orchestration. `js/settings.js`: the control schema the UI is generated from.

## Tests

`test/run.mjs` (Playwright, headless Chromium + SwiftShader, fake camera):

- GPU FFT vs naive DFT, and IFFT∘FFT identity.
- GPU forward effect vs **the actual C** compiled against a naive-DFT FFTW shim (`test/cref`): bit-exact on a 64² image.
- Decoded intervals contain the true magnitude for 100% of bins.
- Round-trip reconstruction error per channel, and solver stability.

```sh
gcc -O2 -Itest/cref -o test/cref/cref test/cref/harness.c reference/swaylock_effect.c -lm
npm i -D playwright && npx playwright install chromium
node test/run.mjs
```
