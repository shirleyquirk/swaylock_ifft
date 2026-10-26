/* Run the effect offline on raw RGBA:
 *   cc -O2 -o trysix_cli test/cref/harness.c reference/swaylock_effect.c -lfftw3 -lm
 *   magick shot.png -depth 8 rgba:- | ./trysix_cli W H | magick -size WxH -depth 8 rgba:- out.png
 * (tests build it against test/cref/fftw3.h, a naive-DFT shim, instead of real FFTW: add -Itest/cref)
 * One size argument means a square N x N image. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
void swaylock_effect(uint32_t *data, int width, int height);
int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s W [H] < in.rgba > out.rgba\n", argv[0]); return 2; }
  int W = atoi(argv[1]), H = argc > 2 ? atoi(argv[2]) : W;
  size_t n = (size_t)W * H;
  uint8_t *rgba = malloc(n * 4); uint32_t *d = malloc(n * 4);
  if (fread(rgba, 1, n * 4, stdin) != n * 4) { fprintf(stderr, "short read: need %zu bytes\n", n * 4); return 1; }
  for (size_t i = 0; i < n; i++) d[i] = 0xFF000000u | (rgba[4*i] << 16) | (rgba[4*i+1] << 8) | rgba[4*i+2];
  swaylock_effect(d, W, H);
  for (size_t i = 0; i < n; i++) { rgba[4*i] = (d[i] >> 16) & 255; rgba[4*i+1] = (d[i] >> 8) & 255; rgba[4*i+2] = d[i] & 255; rgba[4*i+3] = 255; }
  fwrite(rgba, 1, n * 4, stdout);
  return 0;
}
