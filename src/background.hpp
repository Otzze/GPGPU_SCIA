#pragma once

#include "Image.hpp"

#define RESERVOIR_K 16
#define RGB_DIFF_THRESHOLD 1
#define MAX_WEIGHTS 50

struct Reservoir {
    rgb8 rgb;
    uint8_t w;
};

void background_removal_cpp(ImageView<rgb8> in);
// void background_removal_cu(ImageView<rgb8> in);
