/* Constants shared by tryseven.c (encoder) and untryseven.c (decoder). seven.js mirrors them. */
#pragma once

/* compander knee: e = sign(c) * log(1 + |c|/C0) / log(1 + 4WH/C0). Measured best of 0.1..300. */
#define SEVEN_C0 1.0

/* full-range BT.601: rows give Y, Cb, Cr from R, G, B in [0, 1] */
static const double SEVEN_YCC[3][3] = {
    {  0.299,     0.587,     0.114    },
    { -0.168736, -0.331264,  0.5      },
    {  0.5,      -0.418688, -0.081312 },
};
/* inverse: R, G, B from Y, Cb, Cr */
static const double SEVEN_RGB[3][3] = {
    { 1.0,  0.0,       1.402    },
    { 1.0, -0.344136, -0.714136 },
    { 1.0,  1.772,     0.0      },
};
