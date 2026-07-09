#include "radar_cnn_hls.h"
#include <fstream>
#include <iomanip>
#include <iostream>

int main()
{
    const char* INPUT_FILE =
    "C:\\Users\\LENOVO\\Desktop\\KV260_Radar_2D_CNN_MVP\\hardware\\hls\\radar_cnn_hls_accel\\reports\\fixed_point_validation\\input_samples.txt";

    std::ifstream stream(INPUT_FILE);

    if (!stream) {
        std::cerr << "Unable to open input file: "
                  << INPUT_FILE << std::endl;
        return 3;
    }
    int num_samples = 0;

    if (!(stream >> num_samples) || num_samples <= 0) {
        std::cerr << "Bad input file." << std::endl;
        return 4;
    }

    static data_t input[IMG_H * IMG_W];
    static data_t logits[NUM_CLASSES];

    std::cout << std::setprecision(10);

    for (int sample = 0; sample < num_samples; sample++) {

        int cache_index;
        int label;

        stream >> cache_index >> label;

        for (int i = 0; i < IMG_H * IMG_W; i++) {
            float value;
            stream >> value;
            input[i] = value;
        }

        radar_cnn_accel(input, logits);

        int pred = 0;
        float best = logits[0];

        for (int i = 1; i < NUM_CLASSES; i++) {
            if ((float)logits[i] > best) {
                best = logits[i];
                pred = i;
            }
        }

        std::cout
            << "Sample " << sample
            << " True=" << label
            << " Pred=" << pred
            << std::endl;
    }

    return 0;
}