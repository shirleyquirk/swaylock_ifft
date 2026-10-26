/* untryseven: inverse of tryseven.c. Same signature, so it can also be loaded as a
 * swaylock effect (apply it to a tryseven'd screen and you get the screen back).
 *     cc -O2 -shared -fPIC -o untryseven.so untryseven.c -lfftw3 -lm
 */
#include <stdint.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include "seven.h"

void swaylock_effect(uint32_t *data, int width, int height) {
    size_t size = (size_t)width * height;
    double *in = fftw_malloc(sizeof(double) * size);
    double *ycc = malloc(sizeof(double) * size * 3);
    if (!in || !ycc) goto done;

    /* REDFT01 is DCT-III: the inverse of REDFT10 up to a factor 2N per dimension */
    fftw_plan plan = fftw_plan_r2r_2d(height, width, in, in, FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
    const double norm = log1p(4.0 * (double)width * (double)height / SEVEN_C0);
    const double scale = 1.0 / (4.0 * (double)width * (double)height);
    const int shift[3] = { 16, 8, 0 };

    for (int ch = 0; ch < 3; ch++) {
        for (size_t i = 0; i < size; i++) {
            double e = (((data[i] >> shift[ch]) & 0xFF) - 127.5) / 127.5;
            in[i] = copysign(SEVEN_C0 * expm1(fabs(e) * norm), e);
        }
        fftw_execute(plan);
        for (size_t i = 0; i < size; i++) ycc[ch * size + i] = in[i] * scale;
    }
    for (size_t i = 0; i < size; i++) {
        uint32_t px = data[i] & 0xFF000000u;
        for (int o = 0; o < 3; o++) {
            double v = 0;
            for (int ch = 0; ch < 3; ch++) v += SEVEN_RGB[o][ch] * ycc[ch * size + i];
            double q = floor(v * 255.0 + 0.5);
            px |= (uint32_t)(q < 0 ? 0 : (q > 255 ? 255 : q)) << shift[o];
        }
        data[i] = px;
    }

    fftw_destroy_plan(plan);
done:
    free(ycc);
    if (in) fftw_free(in);
}
