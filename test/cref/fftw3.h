/* Minimal FFTW shim for testing: naive double-precision 2-D r2c DFT, FFTW output layout. */
#pragma once
#include <stdlib.h>
#include <math.h>
typedef double fftw_complex[2];
typedef struct { int h, w; double *in; fftw_complex *out; } *fftw_plan;
#define FFTW_ESTIMATE 0
static inline void *fftw_malloc(size_t n) { return malloc(n); }
static inline void fftw_free(void *p) { free(p); }
static inline fftw_plan fftw_plan_dft_r2c_2d(int h, int w, double *in, fftw_complex *out, unsigned f) {
  (void)f; fftw_plan p = malloc(sizeof(*p)); p->h = h; p->w = w; p->in = in; p->out = out; return p;
}
static inline void fftw_destroy_plan(fftw_plan p) { free(p); }
static inline void fftw_execute(fftw_plan p) {
  int H = p->h, W = p->w, OW = W / 2 + 1;
  /* rows first: full complex row DFT of real input */
  double *tr = malloc(sizeof(double) * H * W), *ti = malloc(sizeof(double) * H * W);
  for (int y = 0; y < H; y++) for (int kx = 0; kx < W; kx++) {
    double sr = 0, si = 0;
    for (int x = 0; x < W; x++) { double a = -2 * M_PI * (double)kx * x / W; sr += p->in[y * W + x] * cos(a); si += p->in[y * W + x] * sin(a); }
    tr[y * W + kx] = sr; ti[y * W + kx] = si;
  }
  for (int kx = 0; kx < OW; kx++) for (int ky = 0; ky < H; ky++) {
    double sr = 0, si = 0;
    for (int y = 0; y < H; y++) { double a = -2 * M_PI * (double)ky * y / H, c = cos(a), s = sin(a);
      sr += tr[y * W + kx] * c - ti[y * W + kx] * s; si += tr[y * W + kx] * s + ti[y * W + kx] * c; }
    p->out[ky * OW + kx][0] = sr; p->out[ky * OW + kx][1] = si;
  }
  free(tr); free(ti);
}
