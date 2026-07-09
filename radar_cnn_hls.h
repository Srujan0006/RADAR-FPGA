#ifndef RADAR_CNN_HLS_H
#define RADAR_CNN_HLS_H

#include <stdint.h>

// Fixed-point types.
// Input spectrogram values are expected in approximately [0, 1].
// Intermediate feature maps may need signed range after folded BN.
#ifdef RADAR_CNN_SOFTWARE_FLOAT
// Host-only mode for numerical comparison without a Vitis installation.
typedef float data_t;
typedef float acc_t;
#else
#include <ap_fixed.h>
typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<32, 12> acc_t;
#endif

#define IMG_H 100
#define IMG_W 100
#define NUM_CLASSES 4

#define C_STEM 16
#define C_B1 32
#define C_B2 64
#define C_B3 96
#define C_B4 128

#define S0 100
#define S1 50
#define S2 25
#define S3 12

// Top-level HLS accelerator.
// Input:  flattened spectrogram, shape [1][100][100] in row-major order.
// Output: 4 logits, one per class: Drone, Bird, Human, CornerReflector.
extern "C" {
void radar_cnn_accel(const data_t *input, data_t *logits);
}

#endif // RADAR_CNN_HLS_H
