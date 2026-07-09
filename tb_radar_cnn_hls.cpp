#include "radar_cnn_hls.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    static data_t input[IMG_H * IMG_W];
    static data_t logits[NUM_CLASSES];

    if (argc == 2) {
        std::ifstream stream(argv[1]);
        if (!stream) {
            std::cerr << "Unable to open input file: " << argv[1] << std::endl;
            return 2;
        }
        for (int i = 0; i < IMG_H * IMG_W; i++) {
            float value = 0.0f;
            if (!(stream >> value)) {
                std::cerr << "Expected 10000 input values; stopped at " << i << std::endl;
                return 3;
            }
            input[i] = (data_t)value;
        }
    } else {
        // Standalone deterministic test pattern in [0, 1].
        for (int y = 0; y < IMG_H; y++) {
            for (int x = 0; x < IMG_W; x++) {
                float value = 0.5f + 0.5f * std::sin(0.05f * y) * std::cos(0.05f * x);
                input[y * IMG_W + x] = (data_t)value;
            }
        }
    }

    radar_cnn_accel(input, logits);

    std::cout << std::setprecision(10);
    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << "LOGIT " << i << " " << (float)logits[i] << std::endl;
    }

    return 0;
}
