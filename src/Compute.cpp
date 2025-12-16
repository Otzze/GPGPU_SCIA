#include "Compute.hpp"
#include "Image.hpp"
#include "background.hpp"

#include <cmath>
#include <cstring>
#include <limits>
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

void find_thershold(const ImageView<int>& map, int &t_high, int &t_low) {
    float sum = 0, sum2 = 0, n = 0;
    for (int y = 0; y < map.height; y++) {
        int* line = (int*)((std::byte*)map.buffer + y * map.stride);
        for (int x = 0; x < map.width; x++) {
            if (line[x] > 0) {
                float v = (float)line[x] / 255;
                sum += v;
                sum2 += v * v;
                n++;
            }
        }
    }
    if (n < 0.005 * map.width * map.height) {
        t_high = t_low = 256;
        return;
    }
    float m = sum / n;
    float std = std::sqrt(sum2 / n - m*m);
    t_high = 255 * std::max(m + 3.f * std, 0.15f);
    t_low = t_high * 0.25f;
}

void compute_cpp(ImageView<rgb8> in) {
    static Image<rgb8> background;

    if (background.width != in.width || background.height != in.height)
        background = Image<rgb8>(in.width, in.height);

    for (int y = 0; y < in.height; ++y) {
        std::memcpy((char*)background.buffer + y * background.stride, (char*)in.buffer + y * in.stride,
                    in.width * sizeof(rgb8));
    }
    background_removal_cpp(background);

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
        auto bg_line = (rgb8*)((std::byte*)background.buffer + y * in.stride);
        int* diff_line = (int*)((std::byte*)diff_map.buffer + y * diff_map.stride);

        for (int x = 0; x < in.width; ++x) {
            int diff = std::abs((int)img_line[x].r - (int)bg_line[x].r) +
                       std::abs((int)img_line[x].g - (int)bg_line[x].g) +
                       std::abs((int)img_line[x].b - (int)bg_line[x].b);
            diff_line[x] = diff;
        }
    }

    noise_cancel_cpp(diff_map, 4);
    int t_high, t_low;
    find_thershold(diff_map, t_high, t_low);
    hesteresys_cpp(diff_map, t_low, t_high);

    for (int y = 0; y < diff_map.height; y++) {
        int* line = (int*)((std::byte*)diff_map.buffer + y * diff_map.stride);
        for (int x = 0; x < diff_map.width; x++) {
            if (line[x] > 0)
                line[x] = 1;
        }
    }

    bool alert = alerting_process(diff_map, 500);

    maskage_cpp(in, diff_map);

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
