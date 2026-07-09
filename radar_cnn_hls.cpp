#include "radar_cnn_hls.h"
#include "weights.h"

static data_t relu(data_t x) {
#pragma HLS INLINE
    return x > (data_t)0 ? x : (data_t)0;
}

// Stem: regular 3x3 convolution, 1 -> 16 channels, same padding.
static void stem_conv_relu(
    data_t in[IMG_H][IMG_W],
    data_t out[C_STEM][S0][S0]
) {
#pragma HLS INLINE off
    for (int oc = 0; oc < C_STEM; oc++) {
        for (int y = 0; y < S0; y++) {
            for (int x = 0; x < S0; x++) {
#pragma HLS PIPELINE II=1

                acc_t acc = B_STEM[oc];
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int iy = y + ky - 1;
                        int ix = x + kx - 1;
                        if (iy >= 0 && iy < S0 && ix >= 0 && ix < S0) {
                            acc += (acc_t)in[iy][ix] *
                                   (acc_t)W_STEM[oc][0][ky][kx];
                        }
                    }
                }
                out[oc][y][x] = relu((data_t)acc);
            }
        }
    }
}

template<int C, int S>
static void depthwise_conv_relu(
    data_t in[C][S][S],
    data_t out[C][S][S],
    const data_t weights[C][1][3][3],
    const data_t biases[C]
) {
#pragma HLS INLINE off
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < S; y++) {
            for (int x = 0; x < S; x++) {
#pragma HLS PIPELINE II=1
                acc_t acc = biases[c];
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int iy = y + ky - 1;
                        int ix = x + kx - 1;
                        if (iy >= 0 && iy < S && ix >= 0 && ix < S) {
                            acc += (acc_t)in[c][iy][ix] *
                                   (acc_t)weights[c][0][ky][kx];
                        }
                    }
                }
                out[c][y][x] = relu((data_t)acc);
            }
        }
    }
}

template<int IN_C, int OUT_C, int S>
static void pointwise_conv_relu(
    data_t in[IN_C][S][S],
    data_t out[OUT_C][S][S],
    const data_t weights[OUT_C][IN_C],
    const data_t biases[OUT_C]
) {
#pragma HLS INLINE off
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int y = 0; y < S; y++) {
            for (int x = 0; x < S; x++) {
#pragma HLS PIPELINE II=1
                acc_t acc = biases[oc];
                for (int ic = 0; ic < IN_C; ic++) {
                    acc += (acc_t)in[ic][y][x] * (acc_t)weights[oc][ic];
                }
                out[oc][y][x] = relu((data_t)acc);
            }
        }
    }
}

template<int C, int IN_S, int OUT_S>
static void maxpool_2x2(
    data_t in[C][IN_S][IN_S],
    data_t out[C][OUT_S][OUT_S]
) {
#pragma HLS INLINE off
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < OUT_S; y++) {
            for (int x = 0; x < OUT_S; x++) {
#pragma HLS PIPELINE II=1
                data_t m = in[c][2 * y][2 * x];
                data_t v1 = in[c][2 * y][2 * x + 1];
                data_t v2 = in[c][2 * y + 1][2 * x];
                data_t v3 = in[c][2 * y + 1][2 * x + 1];
                if (v1 > m) m = v1;
                if (v2 > m) m = v2;
                if (v3 > m) m = v3;
                out[c][y][x] = m;
            }
        }
    }
}

static void global_avgpool(
    data_t in[C_B4][S3][S3],
    data_t out[C_B4]
) {
#pragma HLS INLINE off
    for (int c = 0; c < C_B4; c++) {
#pragma HLS PIPELINE II=1
        acc_t acc = 0;
        for (int y = 0; y < S3; y++) {
            for (int x = 0; x < S3; x++) {
                acc += in[c][y][x];
            }
        }
        out[c] = (data_t)(acc / (acc_t)(S3 * S3));
    }
}

static void classifier_1x1(
    data_t in[C_B4],
    data_t out[NUM_CLASSES]
) {
#pragma HLS INLINE off
    for (int oc = 0; oc < NUM_CLASSES; oc++) {
#pragma HLS PIPELINE II=1
        acc_t acc = B_CLS[oc];
        for (int ic = 0; ic < C_B4; ic++) {
            acc += (acc_t)in[ic] * (acc_t)W_CLS[oc][ic];
        }
        out[oc] = (data_t)acc;
    }
}

extern "C" {
void radar_cnn_accel(const data_t *input, data_t *logits) {
#pragma HLS INTERFACE m_axi     port=input  offset=slave bundle=gmem0 depth=10000
#pragma HLS INTERFACE m_axi     port=logits offset=slave bundle=gmem1 depth=4
#pragma HLS INTERFACE s_axilite port=input  bundle=control
#pragma HLS INTERFACE s_axilite port=logits bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static data_t img[IMG_H][IMG_W];
    static data_t stem[C_STEM][S0][S0];
    static data_t dw1[C_STEM][S0][S0];
    static data_t pw1[C_B1][S0][S0];
    static data_t p1[C_B1][S1][S1];
    static data_t dw2[C_B1][S1][S1];
    static data_t pw2[C_B2][S1][S1];
    static data_t p2[C_B2][S2][S2];
    static data_t dw3[C_B2][S2][S2];
    static data_t pw3[C_B3][S2][S2];
    static data_t p3[C_B3][S3][S3];
    static data_t dw4[C_B3][S3][S3];
    static data_t pw4[C_B4][S3][S3];
    static data_t gap[C_B4];
    static data_t out[NUM_CLASSES];

// #pragma HLS BIND_STORAGE variable=stem type=ram_t2p impl=uram
// #pragma HLS BIND_STORAGE variable=dw1 type=ram_t2p impl=uram
// #pragma HLS BIND_STORAGE variable=pw1 type=ram_t2p impl=uram
// #pragma HLS BIND_STORAGE variable=pw2 type=ram_t2p impl=uram
// #pragma HLS BIND_STORAGE variable=pw3 type=ram_t2p impl=uram
// #pragma HLS BIND_STORAGE variable=pw4 type=ram_t2p impl=bram

    for (int i = 0; i < IMG_H * IMG_W; i++) {
#pragma HLS PIPELINE II=1
        img[i / IMG_W][i % IMG_W] = input[i];
    }

    stem_conv_relu(img, stem);
    depthwise_conv_relu<C_STEM, S0>(stem, dw1, W_DW1, B_DW1);
    pointwise_conv_relu<C_STEM, C_B1, S0>(dw1, pw1, W_PW1, B_PW1);
    maxpool_2x2<C_B1, S0, S1>(pw1, p1);

    depthwise_conv_relu<C_B1, S1>(p1, dw2, W_DW2, B_DW2);
    pointwise_conv_relu<C_B1, C_B2, S1>(dw2, pw2, W_PW2, B_PW2);
    maxpool_2x2<C_B2, S1, S2>(pw2, p2);

    depthwise_conv_relu<C_B2, S2>(p2, dw3, W_DW3, B_DW3);
    pointwise_conv_relu<C_B2, C_B3, S2>(dw3, pw3, W_PW3, B_PW3);
    maxpool_2x2<C_B3, S2, S3>(pw3, p3);

    depthwise_conv_relu<C_B3, S3>(p3, dw4, W_DW4, B_DW4);
    pointwise_conv_relu<C_B3, C_B4, S3>(dw4, pw4, W_PW4, B_PW4);
    global_avgpool(pw4, gap);
    classifier_1x1(gap, out);

    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS PIPELINE II=1
        logits[i] = out[i];
    }
}
}
