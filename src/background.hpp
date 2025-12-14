#pragma once

#include "Image.hpp"

#define RESERVOIR_K 20
#define RGB_DIFF_THRESHOLD 1
#define MAX_WEIGHTS 20.f

struct Reservoir {
    rgb8 rgb;
    float w;
};

void background_removal_cpp(ImageView<rgb8> in);
// void background_removal_cu(ImageView<rgb8> in);
