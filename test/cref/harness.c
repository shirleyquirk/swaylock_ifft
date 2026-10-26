#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
void swaylock_effect(uint32_t *data, int width, int height);
int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  uint8_t *rgba = malloc(N * N * 4); uint32_t *d = malloc(N * N * 4);
  fread(rgba, 1, N * N * 4, stdin);
  for (int i = 0; i < N * N; i++) d[i] = 0xFF000000u | (rgba[4*i] << 16) | (rgba[4*i+1] << 8) | rgba[4*i+2];
  swaylock_effect(d, N, N);
  for (int i = 0; i < N * N; i++) { rgba[4*i] = (d[i] >> 16) & 255; rgba[4*i+1] = (d[i] >> 8) & 255; rgba[4*i+2] = d[i] & 255; rgba[4*i+3] = 255; }
  fwrite(rgba, 1, N * N * 4, stdout);
  return 0;
}
