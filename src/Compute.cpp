#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);

/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

int get_neighborhood(const ImageView<int>& img, int x, int y, bool find_max, int field = 3) {
    int extremum = find_max ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
    int half = field / 2;

    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            int nx = x + i;
            int ny = y + j;

            if (nx >= 0 && nx < img.width && ny >= 0 && ny < img.height) {
                int* line = (int*)((std::byte*)img.buffer + ny * img.stride);
                int val = line[nx];

                if (find_max) {
                    if (val > extremum)
                        extremum = val;
                } else {
                    if (val < extremum)
                        extremum = val;
                }
            }
        }
    }
    return extremum;
}

void process_morphology(ImageView<int> map, int field, bool find_max) {
    static std::vector<int> scratch_buffer;
    size_t required_size = map.height * map.stride / sizeof(int);
    
    if (scratch_buffer.size() < required_size) {
        scratch_buffer.resize(required_size);
    }

    int* copy_buffer = scratch_buffer.data();
    std::memcpy(copy_buffer, map.buffer, map.height * map.stride);

    ImageView<int> copy_view = map;
    copy_view.buffer = copy_buffer;

    for (int y = 0; y < map.height; y++) {
        int* line = (int*)((std::byte*)map.buffer + y * map.stride);
        for (int x = 0; x < map.width; x++) {
            line[x] = get_neighborhood(copy_view, x, y, find_max, field);
        }
    }
}

void erosion_cpp(ImageView<int> map, int field = 3) {
    process_morphology(map, field, false);
}

void dilatation_cpp(ImageView<int> map, int field = 3) {
    process_morphology(map, field, true);
}

void noise_cancel_cpp(ImageView<int> map, int field = 3) {
    erosion_cpp(map, field);
    dilatation_cpp(map, field);
}

void hysteresis_threshold(ImageView<int> map, int low, int high, std::vector<std::pair<int, int>>& strong_pixels) {
    strong_pixels.clear();
    strong_pixels.reserve(map.width * map.height / 10);
    for (int y = 0; y < map.height; y++) {
        int* line = (int*)((std::byte*)map.buffer + y * map.stride);
        for (int x = 0; x < map.width; x++) {
            if (line[x] >= high) {
                line[x] = 255;
                strong_pixels.push_back({x, y});
            } else if (line[x] >= low) {
                line[x] = 128;
            } else {
                line[x] = 0;
            }
        }
    }
}

void hysteresis_propagate(ImageView<int> map, std::vector<std::pair<int, int>>& strong_pixels) {
    while (!strong_pixels.empty()) {
        auto [cx, cy] = strong_pixels.back();
        strong_pixels.pop_back();

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0)
                    continue;

                int nx = cx + dx;
                int ny = cy + dy;

                if (nx >= 0 && nx < map.width && ny >= 0 && ny < map.height) {
                    int* line = (int*)((std::byte*)map.buffer + ny * map.stride);
                    if (line[nx] == 128) {
                        line[nx] = 255;
                        strong_pixels.push_back({nx, ny});
                    }
                }
            }
        }
    }
}

void hysteresis_cleanup(ImageView<int> map) {
    for (int y = 0; y < map.height; y++) {
        int* line = (int*)((std::byte*)map.buffer + y * map.stride);
        for (int x = 0; x < map.width; x++) {
            if (line[x] == 128) {
                line[x] = 0;
            }
        }
    }
}

void hesteresys_cpp(ImageView<int> map, int th_low = 4, int th_high = 30) {
    static std::vector<std::pair<int, int>> strong_pixels;

    hysteresis_threshold(map, th_low, th_high, strong_pixels);

    hysteresis_propagate(map, strong_pixels);

    hysteresis_cleanup(map);
}

void maskage_cpp(ImageView<rgb8> in, ImageView<int> mask) {
    for (int y = 0; y < in.height; y++) {
        rgb8* line = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        int* mask_line = (int*)((std::byte*)mask.buffer + y * mask.stride);
        for (int x = 0; x < in.width; x++) {
            if (mask_line[x]) {
                // input + 0.5 * red * masque
                int val = line[x].r + 0.5 * 255;
                line[x].r = (val > 255) ? 255 : val;
            }
        }
    }
}

void maskage_process_cpp(ImageView<rgb8> in, ImageView<int> mask) {
    noise_cancel_cpp(mask);
    hesteresys_cpp(mask);
    for (int y = 0; y < mask.height; y++) {
        int* mask_line = (int*)((std::byte*)mask.buffer + y * mask.stride);
        for (int x = 0; x < mask.width; x++) {
            if (mask_line[x] > 0)
                mask_line[x] = 1;
        }
    }
    maskage_cpp(in, mask);
}

bool alerting_process(const ImageView<int>& mask, int threshold_count) {
    int count = 0;
    for (int y = 0; y < mask.height; y++) {
        int* line = (int*)((std::byte*)mask.buffer + y * mask.stride);
        for (int x = 0; x < mask.width; x++) {
            if (line[x] > 0)
                count++;
        }
    }
    return count > threshold_count;
}

#include <iostream>
#include "background.hpp"

/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in) {
    static Image<rgb8> background;

    if (background.width != in.width || background.height != in.height)
        background = Image<rgb8>(in.width, in.height);

    for (int y = 0; y < in.height; ++y) {
        std::memcpy((char*)background.buffer + y * background.stride, (char*)in.buffer + y * in.stride,
                    in.width * sizeof(rgb8));
    }
    background_removal_cpp(background);

    // static std::vector<rgb8> background;
    // static int bg_width = 0;
    // static int bg_height = 0;

    // 1. Init Background (First frame only - No BEP update)
    // if (background.empty() || bg_width != in.width || bg_height != in.height)
    // {
    //     bg_width = in.width;
    //     bg_height = in.height;
    //     background.resize(in.width * in.height);
    //
    //     for (int y = 0; y < in.height; ++y)
    //     {
    //         rgb8* src_line = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    //         rgb8* dst_line = &background[y * in.width];
    //         std::memcpy(dst_line, src_line, in.width * sizeof(rgb8));
    //     }
    // }

    // 2. Compute Difference (Change mask at t)
    static std::vector<int> diff_buffer;
    if (diff_buffer.size() < (size_t)(in.width * in.height)) {
        diff_buffer.resize(in.width * in.height);
    }
    ImageView<int> diff_map;
    diff_map.buffer = diff_buffer.data();
    diff_map.width = in.width;
    diff_map.height = in.height;
    diff_map.stride = in.width * sizeof(int);

    for (int y = 0; y < in.height; ++y) {
        rgb8* img_line = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        // rgb8* bg_line = &background[y * in.width];
        auto bg_line = (rgb8*)((std::byte*)background.buffer + y * in.stride);
        int* diff_line = (int*)((std::byte*)diff_map.buffer + y * diff_map.stride);

        for (int x = 0; x < in.width; ++x) {
            int diff = std::abs((int)img_line[x].r - (int)bg_line[x].r) +
                       std::abs((int)img_line[x].g - (int)bg_line[x].g) +
                       std::abs((int)img_line[x].b - (int)bg_line[x].b);
            diff_line[x] = diff;
        }
    }

    // 3. Clean change mask (Process)
    noise_cancel_cpp(diff_map, 7);  // Rayon 3 -> diameter 7
    hesteresys_cpp(diff_map, 4, 30);

    // Binarize
    for (int y = 0; y < diff_map.height; y++) {
        int* line = (int*)((std::byte*)diff_map.buffer + y * diff_map.stride);
        for (int x = 0; x < diff_map.width; x++) {
            if (line[x] > 0)
                line[x] = 1;
        }
    }

    // 4. Alerting process
    bool alert = alerting_process(diff_map, 500);  // Threshold 500 pixels

    // 5. Visualization
    maskage_cpp(in, diff_map);

    // Alert indicator (Red border)
    if (alert) {
        for (int x = 0; x < in.width; ++x) {
            ((rgb8*)in.buffer)[x] = {255, 0, 0};
            ((rgb8*)((std::byte*)in.buffer + (in.height - 1) * in.stride))[x] = {255, 0, 0};
        }
        for (int y = 0; y < in.height; ++y) {
            ((rgb8*)((std::byte*)in.buffer + y * in.stride))[0] = {255, 0, 0};
            ((rgb8*)((std::byte*)in.buffer + y * in.stride))[in.width - 1] = {255, 0, 0};
        }
    }
}

extern "C" {

static Parameters g_params;

void cpt_init(Parameters* params) {
    g_params = *params;
}

void cpt_process_frame(uint8_t* buffer, int width, int height, int stride) {
    auto img = ImageView<rgb8>{(rgb8*)buffer, width, height, stride};
    if (g_params.device == e_device_t::CPU)
        compute_cpp(img);
    else if (g_params.device == e_device_t::GPU)
        compute_cu(img);
}
}
