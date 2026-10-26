/* Reference: the forward effect this app inverts (swaylock-effects custom effect).
 * Kept verbatim apart from this header. Build with -lfftw3 -lm. */
#include <stdint.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>

// Convert RGB to HSV
void rgb_to_hsv(double r, double g, double b, double *h, double *s, double *v) {
    double max = fmax(fmax(r, g), b);
    double min = fmin(fmin(r, g), b);
    double delta = max - min;
    
    *v = max;
    
    if (max == 0) {
        *s = 0;
        *h = 0;
        return;
    }
    
    *s = delta / max;
    
    if (delta == 0) {
        *h = 0;
    } else if (max == r) {
        *h = 60.0 * fmod((g - b) / delta, 6.0);
    } else if (max == g) {
        *h = 60.0 * ((b - r) / delta + 2.0);
    } else {
        *h = 60.0 * ((r - g) / delta + 4.0);
    }
    
    if (*h < 0) *h += 360.0;
}

// Convert HSV to RGB
void hsv_to_rgb(double h, double s, double v, uint8_t *r, uint8_t *g, uint8_t *b) {
    double c = v * s;
    double x = c * (1.0 - fabs(fmod(h / 60.0, 2.0) - 1.0));
    double m = v - c;
    
    double r1, g1, b1;
    
    if (h >= 0 && h < 60) {
        r1 = c; g1 = x; b1 = 0;
    } else if (h >= 60 && h < 120) {
        r1 = x; g1 = c; b1 = 0;
    } else if (h >= 120 && h < 180) {
        r1 = 0; g1 = c; b1 = x;
    } else if (h >= 180 && h < 240) {
        r1 = 0; g1 = x; b1 = c;
    } else if (h >= 240 && h < 300) {
        r1 = x; g1 = 0; b1 = c;
    } else {
        r1 = c; g1 = 0; b1 = x;
    }
    
    *r = (uint8_t)((r1 + m) * 255);
    *g = (uint8_t)((g1 + m) * 255);
    *b = (uint8_t)((b1 + m) * 255);
}

void swaylock_effect(uint32_t *data, int width, int height) {
    int size = width * height;
    
    // Allocate FFTW arrays for each HSV channel
    double *h_in = (double*) fftw_malloc(sizeof(double) * size);
    double *s_in = (double*) fftw_malloc(sizeof(double) * size);
    double *v_in = (double*) fftw_malloc(sizeof(double) * size);
    
    fftw_complex *h_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    fftw_complex *s_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    fftw_complex *v_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    
    if (!h_in || !s_in || !v_in || !h_out || !s_out || !v_out) {
        if (h_in) fftw_free(h_in);
        if (s_in) fftw_free(s_in);
        if (v_in) fftw_free(v_in);
        if (h_out) fftw_free(h_out);
        if (s_out) fftw_free(s_out);
        if (v_out) fftw_free(v_out);
        return;
    }
    
    // Create FFTW plans for 2D real-to-complex transforms
    fftw_plan h_plan = fftw_plan_dft_r2c_2d(height, width, h_in, h_out, FFTW_ESTIMATE);
    fftw_plan s_plan = fftw_plan_dft_r2c_2d(height, width, s_in, s_out, FFTW_ESTIMATE);
    fftw_plan v_plan = fftw_plan_dft_r2c_2d(height, width, v_in, v_out, FFTW_ESTIMATE);
    
    // Convert RGB to HSV and normalize
    for (int i = 0; i < size; i++) {
        double r = ((data[i] >> 16) & 0xFF) / 255.0;
        double g = ((data[i] >> 8) & 0xFF) / 255.0;
        double b = (data[i] & 0xFF) / 255.0;
        
        double h, s, v;
        rgb_to_hsv(r, g, b, &h, &s, &v);
        
        // Normalize HSV values (0.0 to 1.0)
        h_in[i] = h / 360.0;  // Hue: 0-360° → 0-1
        s_in[i] = s;          // Saturation: already 0-1
        v_in[i] = v;          // Value: already 0-1
    }
    
    // Execute 2D FFTs
    fftw_execute(h_plan);
    fftw_execute(s_plan);
    fftw_execute(v_plan);
    
    // Convert FFT output back to pixel data
    int out_width = width / 2 + 1;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double h_mag, s_mag, v_mag;
            
            if (x < out_width) {
                // Direct FFT output
                int idx = y * out_width + x;
                h_mag = sqrt(h_out[idx][0] * h_out[idx][0] + h_out[idx][1] * h_out[idx][1]);
                s_mag = sqrt(s_out[idx][0] * s_out[idx][0] + s_out[idx][1] * s_out[idx][1]);
                v_mag = sqrt(v_out[idx][0] * v_out[idx][0] + v_out[idx][1] * v_out[idx][1]);
            } else {
                // Mirror the complex conjugate for the right half
                int mirror_x = width - x;
                int mirror_y = (y == 0) ? 0 : height - y;
                
                if (mirror_x < out_width) {
                    int idx = mirror_y * out_width + mirror_x;
                    h_mag = sqrt(h_out[idx][0] * h_out[idx][0] + h_out[idx][1] * h_out[idx][1]);
                    s_mag = sqrt(s_out[idx][0] * s_out[idx][0] + s_out[idx][1] * s_out[idx][1]);
                    v_mag = sqrt(v_out[idx][0] * v_out[idx][0] + v_out[idx][1] * v_out[idx][1]);
                } else {
                    h_mag = s_mag = v_mag = 0;
                }
            }
            
            // Scale magnitudes to visible range
            h_mag = log(1.0 + h_mag) * 50.0;
            s_mag = log(1.0 + s_mag) * 25.0;
            v_mag = log(1.0 + v_mag) * 50.0;
            
            // Normalize back to HSV ranges
            double h_final = fmod(h_mag, 360.0);  // Hue wraps around
            double s_final = fmin(1.0, s_mag / 255.0);  // Saturation 0-1
            double v_final = fmin(1.0, v_mag / 255.0);  // Value 0-1
            
            // Convert back to RGB
            uint8_t r_val, g_val, b_val;
            hsv_to_rgb(h_final, s_final, v_final, &r_val, &g_val, &b_val);
            
            // Preserve alpha and set new RGB values
            int pixel_idx = y * width + x;
            uint32_t alpha = data[pixel_idx] & 0xFF000000;
            data[pixel_idx] = alpha | (r_val << 16) | (g_val << 8) | b_val;
        }
    }
    
    // Cleanup
    fftw_destroy_plan(h_plan);
    fftw_destroy_plan(s_plan);
    fftw_destroy_plan(v_plan);
    
    fftw_free(h_in);
    fftw_free(s_in);
    fftw_free(v_in);
    fftw_free(h_out);
    fftw_free(s_out);
    fftw_free(v_out);
}
