/* tryseven: an invertible relative of trysix.
 *
 * trysix keeps |FFT| of H, S, V and throws the phase away, which cannot be undone.
 * tryseven takes the 2-D DCT-II of Y, Cb, Cr instead. The DCT of a real image is real:
 * there is no phase to lose, and three channels give exactly three real coefficients per
 * pixel, which is exactly what an RGB pixel holds.
 *
 * Each coefficient c is companded with the same log(1 + |x|) trysix uses, keeping its sign:
 *     e    = sign(c) * log(1 + |c|) / log(1 + 4*W*H)          (in [-1, 1])
 *     byte = round(127.5 + 127.5 * e)
 * R carries Y's coefficient, G carries Cb's, B carries Cr's. Zero is mid grey, so the result
 * is grey static with a bright corner at DC (top left).
 *
 * Inverse: untryseven.c, or the browser decoder (seven.html). A desktop screenshot round-trips
 * at roughly 40 dB PSNR.
 *
 * Same entry point as swaylock-effects' --effect-custom. Build:
 *     cc -O2 -shared -fPIC -o tryseven.so tryseven.c -lfftw3 -lm
 */
#include <stdint.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include "seven.h"

void swaylock_effect(uint32_t *data, int width, int height) {
    size_t size = (size_t)width * height;
    double *in = fftw_malloc(sizeof(double) * size);
    double *out = fftw_malloc(sizeof(double) * size);
    uint8_t *bytes = malloc(size * 3);   /* the original pixels are needed for all three channels */
    if (!in || !out || !bytes) goto done;

    fftw_plan plan = fftw_plan_r2r_2d(height, width, in, out, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    const double norm = log1p(4.0 * (double)width * (double)height / SEVEN_C0);

    for (int ch = 0; ch < 3; ch++) {
        for (size_t i = 0; i < size; i++) {
            double r = ((data[i] >> 16) & 0xFF) / 255.0;
            double g = ((data[i] >> 8) & 0xFF) / 255.0;
            double b = (data[i] & 0xFF) / 255.0;
            in[i] = SEVEN_YCC[ch][0] * r + SEVEN_YCC[ch][1] * g + SEVEN_YCC[ch][2] * b;
        }
        fftw_execute(plan);
        for (size_t i = 0; i < size; i++) {
            double e = copysign(log1p(fabs(out[i]) / SEVEN_C0) / norm, out[i]);
            double q = floor(127.5 + 127.5 * e + 0.5);
            bytes[ch * size + i] = (uint8_t)(q < 0 ? 0 : (q > 255 ? 255 : q));
        }
    }
    for (size_t i = 0; i < size; i++)
        data[i] = (data[i] & 0xFF000000u) | ((uint32_t)bytes[i] << 16)
                | ((uint32_t)bytes[size + i] << 8) | bytes[2 * size + i];

    fftw_destroy_plan(plan);
done:
    free(bytes);
    if (in) fftw_free(in);
    if (out) fftw_free(out);
}
