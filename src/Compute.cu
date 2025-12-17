#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Compute.hpp"
#include "Image.hpp"
#include "background.hpp"

#define BLOCK_SIZE 128

__device__ int find_matching_reservoir_shared(rgb8 p, uint8_t* s_weights, uchar4* s_colors, int tid) {
    int m_idx = -1;
    for (int i = 0; i < RESERVOIR_K; i++) {
        int idx = i * BLOCK_SIZE + tid;

        if (s_weights[idx] > 0) {
            uchar4 res_rgb = s_colors[idx];
            if (abs((int)p.r - (int)res_rgb.x) < RGB_DIFF_THRESHOLD &&
                abs((int)p.g - (int)res_rgb.y) < RGB_DIFF_THRESHOLD &&
                abs((int)p.b - (int)res_rgb.z) < RGB_DIFF_THRESHOLD) {
                return i;
            }
        } else {
            m_idx = i;
        }
    }
    return m_idx;
}

__device__ void update_match(int shared_idx, rgb8 current_pixel, uint8_t* s_weights, uchar4* s_colors) {
    uint8_t w = s_weights[shared_idx];

    if (w > 0) {  // matching
        int new_w_int = (int)w + 1;
        if (new_w_int > MAX_WEIGHTS)
            new_w_int = MAX_WEIGHTS;
        w = (uint8_t)new_w_int;
        s_weights[shared_idx] = w;

        uchar4 res_rgb = s_colors[shared_idx];
        res_rgb.x = (uint8_t)(((int)(w - 1) * res_rgb.x + (int)current_pixel.r) / (int)w);
        res_rgb.y = (uint8_t)(((int)(w - 1) * res_rgb.y + (int)current_pixel.g) / (int)w);
        res_rgb.z = (uint8_t)(((int)(w - 1) * res_rgb.z + (int)current_pixel.b) / (int)w);
        s_colors[shared_idx] = res_rgb;
    } else {  // empty slot
        s_weights[shared_idx] = 1;
        s_colors[shared_idx] = {current_pixel.r, current_pixel.g, current_pixel.b, 0};
    }
}

__device__ void replace_reservoir(int pixel_idx,
                                  int tid,
                                  rgb8 current_pixel,
                                  uint8_t* s_weights,
                                  uchar4* s_colors,
                                  curandState* randStates) {
    int min_idx = 0;
    unsigned int total_weights = 0;
    uint8_t min_w = 255;

    for (int i = 0; i < RESERVOIR_K; ++i) {
        int shared_idx = i * BLOCK_SIZE + tid;
        uint8_t w = s_weights[shared_idx];
        if (i == 0)
            min_w = w;

        if (w < min_w) {
            min_w = w;
            min_idx = i;
        }
        total_weights += w;
    }

    uint8_t min_weight = min_w;
    curandState localRandState = randStates[pixel_idx];
    if (curand_uniform(&localRandState) * (float)total_weights >= (float)min_weight) {
        int shared_min_idx = min_idx * BLOCK_SIZE + tid;
        s_weights[shared_min_idx] = 1;
        s_colors[shared_min_idx] = {current_pixel.r, current_pixel.g, current_pixel.b, 0};
    }
}

__device__ uchar4 get_background_color(int tid, uint8_t* s_weights, uchar4* s_colors) {
    uint8_t max_w = 0;
    int max_idx = 0;
    for (int i = 0; i < RESERVOIR_K; ++i) {
        int shared_idx = i * BLOCK_SIZE + tid;
        uint8_t w = s_weights[shared_idx];
        if (w > max_w) {
            max_w = w;
            max_idx = i;
        }
    }
    return s_colors[max_idx * BLOCK_SIZE + tid];
}

__global__ void init_rand_states(int n, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curand_init(clock64(), idx, 0, &states[idx]);
    }
}

__global__ void background_removal_kernel(ImageView<rgb8> in,
                                          uint8_t* weights,
                                          uchar4* colors,
                                          curandState* randStates) {
    __shared__ uint8_t s_weights[RESERVOIR_K * BLOCK_SIZE];
    __shared__ uchar4 s_colors[RESERVOIR_K * BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    if (x >= in.width || y >= in.height)
        return;

    int pixel_idx = 0;
    int image_size = 0;

    pixel_idx = y * in.width + x;
    image_size = in.width * in.height;
    for (int i = 0; i < RESERVOIR_K; ++i) {
        int global_idx = i * image_size + pixel_idx;
        int shared_idx = i * BLOCK_SIZE + tid;
        s_weights[shared_idx] = weights[global_idx];
        s_colors[shared_idx] = colors[global_idx];
    }

    rgb8* lineptr = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    rgb8 current_pixel = lineptr[x];

    int m_idx = find_matching_reservoir_shared(current_pixel, s_weights, s_colors, tid);

    if (m_idx != -1) {
        update_match(m_idx * BLOCK_SIZE + tid, current_pixel, s_weights, s_colors);
    } else {
        replace_reservoir(pixel_idx, tid, current_pixel, s_weights, s_colors, randStates);
    }

    uchar4 bg = get_background_color(tid, s_weights, s_colors);
    lineptr[x] = {bg.x, bg.y, bg.z};

    for (int i = 0; i < RESERVOIR_K; ++i) {
        int global_idx = i * image_size + pixel_idx;
        int shared_idx = i * BLOCK_SIZE + tid;
        weights[global_idx] = s_weights[shared_idx];
        colors[global_idx] = s_colors[shared_idx];
    }
}

void background_removal_cu(ImageView<rgb8> in) {
    static bool g_initialized = false;
    static uint8_t* d_res_weights = nullptr;
    static uchar4* d_res_colors = nullptr;
    static curandState* randStates_d = nullptr;

    if (!g_initialized) {
        int num_pixels = in.width * in.height;
        size_t weights_size = num_pixels * RESERVOIR_K * sizeof(uint8_t);
        size_t colors_size = num_pixels * RESERVOIR_K * sizeof(uchar4);

        cudaMalloc(&d_res_weights, weights_size);
        cudaMemset(d_res_weights, 0, weights_size);

        cudaMalloc(&d_res_colors, colors_size);

        cudaMalloc(&randStates_d, num_pixels * sizeof(curandState));

        int block_size = 256;
        int grid_size = (num_pixels + block_size - 1) / block_size;
        init_rand_states<<<grid_size, block_size>>>(num_pixels, randStates_d);

        g_initialized = true;
    }

    dim3 block(16, 8);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

    background_removal_kernel<<<grid, block>>>(in, d_res_weights, d_res_colors, randStates_d);
}

__global__ void reduce_stats_kernel(ImageView<int> map, Stats* partial_stats) {
    float local_sum = 0;
    float local_sum2 = 0;
    int local_n = 0;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < map.width && y < map.height) {
        int* line = (int*)((char*)map.buffer + y * map.stride);
        int val = line[x];
        if (val > 0) {
            float v = (float)val / 255.0f;
            local_sum = v;
            local_sum2 = v * v;
            local_n = 1;
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_sum2 += __shfl_down_sync(0xFFFFFFFF, local_sum2, offset);
        local_n += __shfl_down_sync(0xFFFFFFFF, local_n, offset);
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    __shared__ Stats s_data[32];

    if (lane == 0) {
        s_data[warp_id] = {local_sum, local_sum2, local_n};
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < (blockDim.x * blockDim.y) / 32) ? s_data[lane].sum : 0;
        local_sum2 = (lane < (blockDim.x * blockDim.y) / 32) ? s_data[lane].sum2 : 0;
        local_n = (lane < (blockDim.x * blockDim.y) / 32) ? s_data[lane].n : 0;

        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
            local_sum2 += __shfl_down_sync(0xFFFFFFFF, local_sum2, offset);
            local_n += __shfl_down_sync(0xFFFFFFFF, local_n, offset);
        }

        if (lane == 0) {
            int bid = blockIdx.y * gridDim.x + blockIdx.x;
            partial_stats[bid] = {local_sum, local_sum2, local_n};
        }
    }
}

__global__ void compute_difference_kernel(ImageView<rgb8> in, ImageView<rgb8> bg, ImageView<int> diff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.width && y < in.height) {
        rgb8* in_line = (rgb8*)((char*)in.buffer + y * in.stride);
        rgb8* bg_line = (rgb8*)((char*)bg.buffer + y * bg.stride);
        int* diff_line = (int*)((char*)diff.buffer + y * diff.stride);

        int dr = abs((int)in_line[x].r - (int)bg_line[x].r);
        int dg = abs((int)in_line[x].g - (int)bg_line[x].g);
        int db = abs((int)in_line[x].b - (int)bg_line[x].b);

        diff_line[x] = dr + dg + db;
    }
}

#define TILE_W 16
#define TILE_H 16
#define RADIUS 2
#define SMEM_W (TILE_W + 2 * RADIUS)
#define SMEM_H (TILE_H + 2 * RADIUS)

__device__ void morphology_shared_kernel(ImageView<int> src, ImageView<int> dst, bool isDilation) {
    __shared__ int s_tile[SMEM_H][SMEM_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int x = bx + tx;
    int y = by + ty;

    int tid = ty * blockDim.x + tx;
    int block_size = blockDim.x * blockDim.y;

    int total_pixels = SMEM_H * SMEM_W;

    for (int i = tid; i < total_pixels; i += block_size) {
        int smem_y = i / SMEM_W;
        int smem_x = i % SMEM_W;

        int global_x = bx + smem_x - RADIUS;
        int global_y = by + smem_y - RADIUS;

        if (global_x < 0)
            global_x = 0;
        if (global_x >= src.width)
            global_x = src.width - 1;
        if (global_y < 0)
            global_y = 0;
        if (global_y >= src.height)
            global_y = src.height - 1;

        int* src_line = (int*)((char*)src.buffer + global_y * src.stride);
        s_tile[smem_y][smem_x] = src_line[global_x];
    }

    __syncthreads();

    if (x < src.width && y < src.height) {
        int val = isDilation ? -2147483648 : 2147483647;

#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++) {
#pragma unroll
            for (int i = -RADIUS; i <= RADIUS; i++) {
                int neighbor_val = s_tile[ty + RADIUS + j][tx + RADIUS + i];

                if (isDilation) {
                    if (neighbor_val > val)
                        val = neighbor_val;
                } else {
                    if (neighbor_val < val)
                        val = neighbor_val;
                }
            }
        }

        int* dst_line = (int*)((char*)dst.buffer + y * dst.stride);
        dst_line[x] = val;
    }
}

__global__ void dilation_kernel(ImageView<int> src, ImageView<int> dst) {
    morphology_shared_kernel(src, dst, true);
}

__global__ void erosion_kernel(ImageView<int> src, ImageView<int> dst) {
    morphology_shared_kernel(src, dst, false);
}

__global__ void calculate_thresholds_kernel(Stats* partials,
                                            int num_blocks,
                                            int total_pixels,
                                            int* d_thresholds,
                                            float* d_smoothed) {
    if (threadIdx.x == 0) {
        float total_sum = 0;
        float total_sum2 = 0;
        int total_n = 0;

        for (int i = 0; i < num_blocks; ++i) {
            total_sum += partials[i].sum;
            total_sum2 += partials[i].sum2;
            total_n += partials[i].n;
        }

        float t_high = 255.0f;
        float t_low = 255.0f;

        if (total_n >= 0.005f * total_pixels) {
            float m = total_sum / total_n;
            float std = sqrtf(total_sum2 / total_n - m * m);
            float current_high = 255.0f * (m + 2.0f * std);

            if (current_high < 0.08f * 255.0f)
                current_high = 0.08f * 255.0f;

            float current_low = current_high * 0.5f;

            float smoothed_high = d_smoothed[0];
            float smoothed_low = d_smoothed[1];

            if (smoothed_high < 0.0f) {
                smoothed_high = current_high;
                smoothed_low = current_low;
            } else {
                const float alpha = 0.9f;
                smoothed_high = alpha * smoothed_high + (1.0f - alpha) * current_high;
                smoothed_low = alpha * smoothed_low + (1.0f - alpha) * current_low;
            }

            d_smoothed[0] = smoothed_high;
            d_smoothed[1] = smoothed_low;

            t_high = smoothed_high;
            t_low = smoothed_low;
        }

        d_thresholds[0] = (int)t_low;
        d_thresholds[1] = (int)t_high;
    }
}

__global__ void hysteresis_threshold_kernel(ImageView<int> map, int* d_thresholds) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int low = d_thresholds[0];
    int high = d_thresholds[1];

    if (x < map.width && y < map.height) {
        int* line = (int*)((char*)map.buffer + y * map.stride);
        if (line[x] >= high) {
            line[x] = 255;
        } else if (line[x] >= low) {
            line[x] = 128;
        } else {
            line[x] = 0;
        }
    }
}

__global__ void hysteresis_propagate_kernel(ImageView<int> map) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < map.width && y < map.height) {
        int* line = (int*)((char*)map.buffer + y * map.stride);
        if (line[x] == 128) {
            bool has_strong_neighbor = false;
#pragma unroll
            for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0)
                        continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < map.width && ny >= 0 && ny < map.height) {
                        int* n_line = (int*)((char*)map.buffer + ny * map.stride);
                        if (n_line[nx] == 255) {
                            has_strong_neighbor = true;
                            break;
                        }
                    }
                }
                if (has_strong_neighbor)
                    break;
            }
            if (has_strong_neighbor) {
                line[x] = 255;
            }
        }
    }
}

__global__ void hysteresis_cleanup_kernel(ImageView<int> map) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < map.width && y < map.height) {
        int* line = (int*)((char*)map.buffer + y * map.stride);
        if (line[x] == 128) {
            line[x] = 0;
        }
    }
}

__global__ void binarize_kernel(ImageView<int> map) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < map.width && y < map.height) {
        int* line = (int*)((char*)map.buffer + y * map.stride);
        if (line[x] > 0)
            line[x] = 1;
    }
}

__global__ void count_pixels_kernel(ImageView<int> map, int* count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < map.width && y < map.height) {
        int* line = (int*)((char*)map.buffer + y * map.stride);
        if (line[x] > 0) {
            atomicAdd(count, 1);
        }
    }
}

__global__ void maskage_kernel(ImageView<rgb8> in, ImageView<int> mask) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.width && y < in.height) {
        rgb8* in_line = (rgb8*)((char*)in.buffer + y * in.stride);
        int* mask_line = (int*)((char*)mask.buffer + y * mask.stride);

        if (mask_line[x]) {
            int val = in_line[x].r + 0.5 * 255;
            in_line[x].r = (val > 255) ? 255 : val;
        }
    }
}

__global__ void draw_border_kernel(ImageView<rgb8> in, int* d_count, int th) {
    if (*d_count <= th)
        return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < in.width) {
        rgb8* top_line = (rgb8*)((char*)in.buffer);
        rgb8* bot_line = (rgb8*)((char*)in.buffer + (in.height - 1) * in.stride);
        top_line[idx] = {255, 0, 0};
        bot_line[idx] = {255, 0, 0};
    }

    if (idx < in.height) {
        rgb8* line = (rgb8*)((char*)in.buffer + idx * in.stride);
        line[0] = {255, 0, 0};
        line[in.width - 1] = {255, 0, 0};
    }
}

void compute_cu(ImageView<rgb8> in) {
    static Image<rgb8> device_in;
    static Image<rgb8> device_bg;
    static Image<int> device_diff_map;
    static Image<int> device_scratch_map;

    if (device_in.width != in.width || device_in.height != in.height) {
        device_in = Image<rgb8>(in.width, in.height, true);
        device_bg = Image<rgb8>(in.width, in.height, true);
        device_diff_map = Image<int>(in.width, in.height, true);
        device_scratch_map = Image<int>(in.width, in.height, true);
    }

    cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height,
                 cudaMemcpyHostToDevice);

    cudaMemcpy2D(device_bg.buffer, device_bg.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8),
                 in.height, cudaMemcpyDeviceToDevice);
    background_removal_cu(device_bg);

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

    compute_difference_kernel<<<grid, block>>>(device_in, device_bg, device_diff_map);

    erosion_kernel<<<grid, block>>>(device_diff_map, device_scratch_map);
    dilation_kernel<<<grid, block>>>(device_scratch_map, device_diff_map);

    static Stats* d_partials = nullptr;
    static Stats* h_partials = nullptr;
    int num_blocks = grid.x * grid.y;

    static int cached_num_blocks = 0;
    if (num_blocks > cached_num_blocks) {
        if (d_partials)
            cudaFree(d_partials);
        if (h_partials)
            cudaFreeHost(h_partials);
        cudaMalloc(&d_partials, num_blocks * sizeof(Stats));
        cudaMallocHost(&h_partials, num_blocks * sizeof(Stats));
        cached_num_blocks = num_blocks;
    }

    reduce_stats_kernel<<<grid, block>>>(device_diff_map, d_partials);
    static int* d_thresholds = nullptr;
    static float* d_smoothed = nullptr;
    if (d_thresholds == nullptr) {
        cudaMalloc(&d_thresholds, 2 * sizeof(int));
        cudaMalloc(&d_smoothed, 2 * sizeof(float));
    }
    calculate_thresholds_kernel<<<1, 1>>>(d_partials, num_blocks, in.width * in.height, d_thresholds, d_smoothed);

    hysteresis_threshold_kernel<<<grid, block>>>(device_diff_map, d_thresholds);

    for (int i = 0; i < 30; i++) {
        hysteresis_propagate_kernel<<<grid, block>>>(device_diff_map);
    }

    hysteresis_cleanup_kernel<<<grid, block>>>(device_diff_map);

    binarize_kernel<<<grid, block>>>(device_diff_map);

    static int* d_count = nullptr;
    if (!d_count)
        cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    count_pixels_kernel<<<grid, block>>>(device_diff_map, d_count);

    maskage_kernel<<<grid, block>>>(device_in, device_diff_map);

    int max_dim = (in.width > in.height) ? in.width : in.height;
    dim3 border_grid((max_dim + 255) / 256);
    draw_border_kernel<<<border_grid, 256>>>(device_in, d_count, 500);

    cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height,
                 cudaMemcpyDeviceToHost);
}
