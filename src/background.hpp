#pragma once

#include "Image.hpp"

#define K 20
#define RGB_DIFF_THRESHOLD 40
#define MAX_WEIGHTS 100.f

struct Reservoir {
    rgb8 rgb;
    float w;
};

void background_removal_cpp(ImageView<rgb8> in);
void background_removal_cu(ImageView<rgb8> in);
