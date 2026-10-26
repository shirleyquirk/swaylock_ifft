/* Just enough of FFTW 3.3's public API to build the effects against the runtime library
 * (libfftw3.so.3) on a machine without the -dev headers. Values match fftw3.h. */
#pragma once
#include <stddef.h>
typedef double fftw_complex[2];
typedef struct fftw_plan_s *fftw_plan;
typedef enum {
    FFTW_R2HC = 0, FFTW_HC2R = 1, FFTW_DHT = 2,
    FFTW_REDFT00 = 3, FFTW_REDFT01 = 4, FFTW_REDFT10 = 5, FFTW_REDFT11 = 6,
    FFTW_RODFT00 = 7, FFTW_RODFT01 = 8, FFTW_RODFT10 = 9, FFTW_RODFT11 = 10
} fftw_r2r_kind;
#define FFTW_ESTIMATE (1U << 6)
void *fftw_malloc(size_t n);
void fftw_free(void *p);
fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_r2r_2d(int n0, int n1, double *in, double *out, fftw_r2r_kind k0, fftw_r2r_kind k1, unsigned flags);
void fftw_execute(const fftw_plan p);
void fftw_destroy_plan(fftw_plan p);
