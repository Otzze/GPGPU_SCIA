#include "background.hpp"
#include <iostream>
#include <random>

static Reservoir* reservoirs = nullptr;
static int g_width = 0;
static int g_height = 0;

static std::mt19937 randState;

int find_matching_rgb(rgb8 p, Reservoir* rs) {
    int m_idx = -1;
    for (int i = 0; i < RESERVOIR_K; i++) {
        if (rs[i].w > 0) {
            if (abs((int)p.r - (int)rs[i].rgb.r) < RGB_DIFF_THRESHOLD &&
                abs((int)p.g - (int)rs[i].rgb.g) < RGB_DIFF_THRESHOLD &&
                abs((int)p.b - (int)rs[i].rgb.b) < RGB_DIFF_THRESHOLD) {
                m_idx = i;
                return m_idx;
            }
        } else {
            m_idx = i;
        }
    }
    return m_idx;
}

void background_removal_cpp(ImageView<rgb8> in) {
    if (reservoirs == nullptr) {
        g_width = in.width;
        g_height = in.height;
        reservoirs = new Reservoir[g_width * g_height * RESERVOIR_K];
        for (int i = 0; i < g_width * g_height * RESERVOIR_K; ++i) {
            reservoirs[i].w = 0;
        }
        std::random_device rd;
        randState = std::mt19937(rd());
    }

    std::uniform_real_distribution<float> randfloat(0.0, 1.0);

    for (int y = 0; y < in.height; ++y) {
        rgb8* lineptr = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        for (int x = 0; x < in.width; ++x) {
            Reservoir* pixel_reservoirs = &reservoirs[(y * g_width + x) * RESERVOIR_K];
            rgb8 current_pixel_rgb = lineptr[x];

            int m_idx = find_matching_rgb(current_pixel_rgb, pixel_reservoirs);

            if (m_idx != -1 && pixel_reservoirs[m_idx].w > 0) {  // matching
                pixel_reservoirs[m_idx].w += 1;

                pixel_reservoirs[m_idx].rgb.r =
                    (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.r + current_pixel_rgb.r) /
                              pixel_reservoirs[m_idx].w);
                pixel_reservoirs[m_idx].rgb.g =
                    (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.g + current_pixel_rgb.g) /
                              pixel_reservoirs[m_idx].w);
                pixel_reservoirs[m_idx].rgb.b =
                    (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.b + current_pixel_rgb.b) /
                              pixel_reservoirs[m_idx].w);
            } else if (m_idx != -1 && pixel_reservoirs[m_idx].w == 0) {  // empty slot
                pixel_reservoirs[m_idx].rgb = current_pixel_rgb;
                pixel_reservoirs[m_idx].w = 1;
            } else {  // no match and no empty slot, perform weighted reservoir replacement
                int min_idx = 0;
                float min_weight = pixel_reservoirs[0].w;
                float total_weights = pixel_reservoirs[0].w;
                for (int i = 1; i < RESERVOIR_K; ++i) {
                    if (pixel_reservoirs[i].w < min_weight) {
                        min_weight = pixel_reservoirs[i].w;
                        min_idx = i;
                    }
                    total_weights += pixel_reservoirs[i].w;
                }

                if (randfloat(randState) * total_weights >= min_weight) {
                    pixel_reservoirs[min_idx].rgb = current_pixel_rgb;
                    pixel_reservoirs[min_idx].w = 1;
                }
            }

            float max_w = 0;
            int max_idx = 0;
            for (int i = 0; i < RESERVOIR_K; ++i) {
                if (pixel_reservoirs[i].w > MAX_WEIGHTS) {
                    pixel_reservoirs[i].w = MAX_WEIGHTS;
                }
                if (pixel_reservoirs[i].w > max_w) {
                    max_w = pixel_reservoirs[i].w;
                    max_idx = i;
                }
            }

            lineptr[x] = pixel_reservoirs[max_idx].rgb;
        }
    }
}
