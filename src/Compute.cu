#include "Compute.hpp"
#include "background.hpp"
#include "Image.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
__device__ Reservoir* reservoirs_d = nullptr;
__device__ int g_width_d = 0;
__device__ int g_height_d = 0;
__device__ curandState* randStates_d = nullptr;


// find matching reservoir
__device__ int find_matching_reservoir_d(rgb8 p, Reservoir* rs)
{
    int m_idx = -1;
    for (int i = 0; i < RESERVOIR_K; i++)
    {
        if (rs[i].w > 0)
        {
            if (abs((int)p.r - (int)rs[i].rgb.r) < RGB_DIFF_THRESHOLD &&
                abs((int)p.g - (int)rs[i].rgb.g) < RGB_DIFF_THRESHOLD &&
                abs((int)p.b - (int)rs[i].rgb.b) < RGB_DIFF_THRESHOLD)
            {
                m_idx = i;
                return m_idx;
            }
        }
        else
        {
            m_idx = i;
        }
    }
    return m_idx;
}

__global__ void background_removal_kernel(ImageView<rgb8> in)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.width && y < in.height)
    {
        Reservoir* pixel_reservoirs = &reservoirs_d[(y * g_width_d + x) * RESERVOIR_K];
        
        rgb8* lineptr = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        rgb8 current_pixel = lineptr[x];

        int m_idx = find_matching_reservoir_d(current_pixel, pixel_reservoirs);

        if (m_idx != -1 && pixel_reservoirs[m_idx].w > 0) { // matching
            pixel_reservoirs[m_idx].w += 1;
            pixel_reservoirs[m_idx].rgb.r = (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.r + current_pixel.r) / pixel_reservoirs[m_idx].w);
            pixel_reservoirs[m_idx].rgb.g = (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.g + current_pixel.g) / pixel_reservoirs[m_idx].w);
            pixel_reservoirs[m_idx].rgb.b = (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.b + current_pixel.b) / pixel_reservoirs[m_idx].w);
        } else if (m_idx != -1 && pixel_reservoirs[m_idx].w == 0) { // empty slot
            pixel_reservoirs[m_idx].rgb = current_pixel;
            pixel_reservoirs[m_idx].w = 1;
        } else { // no match and no empty slot, perform weighted reservoir replacement
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
            
            curandState* localRandState = &randStates_d[y * g_width_d + x];
            if (curand_uniform(localRandState) * total_weights >= min_weight) {
                pixel_reservoirs[min_idx].rgb = current_pixel;
                pixel_reservoirs[min_idx].w = 1;
            }
        }

        // cap weights to MAX_WEIGHTS
        float max_w = 0;
        int max_idx = 0;
        for(int i = 0; i < RESERVOIR_K; ++i) {
            if (pixel_reservoirs[i].w > MAX_WEIGHTS) {
                pixel_reservoirs[i].w = MAX_WEIGHTS;
            }
            if (pixel_reservoirs[i].w > max_w) {
                max_w = pixel_reservoirs[i].w;
                max_idx = i;
            }
        }

        // set the background
        lineptr[x] = pixel_reservoirs[max_idx].rgb;
    }
}

__global__ void init_rand_states(int width, int height, curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        curand_init(clock64(), y * width + x, 0, &states[y * width + x]);
    }
}


void background_removal_cu(ImageView<rgb8> in) {
    static bool g_initialized = false;

    if (!g_initialized) {
        Reservoir* d_res_ptr = nullptr;
        cudaMalloc((void**)&d_res_ptr, in.width * in.height * RESERVOIR_K * sizeof(Reservoir));
        cudaMemset(d_res_ptr, 0, in.width * in.height * RESERVOIR_K * sizeof(Reservoir));
        cudaMemcpyToSymbol(reservoirs_d, &d_res_ptr, sizeof(Reservoir*));
        
        curandState* d_rand_ptr = nullptr;
        cudaMalloc((void**)&d_rand_ptr, in.width * in.height * sizeof(curandState));
        cudaMemcpyToSymbol(randStates_d, &d_rand_ptr, sizeof(curandState*));
        
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        init_rand_states<<<grid, block>>>(in.width, in.height, d_rand_ptr);
        
        cudaMemcpyToSymbol(g_width_d, &in.width, sizeof(int));
        cudaMemcpyToSymbol(g_height_d, &in.height, sizeof(int));

        g_initialized = true;
    }

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

    background_removal_kernel<<<grid, block>>>(in);
}

__global__ void compute_difference_kernel(ImageView<rgb8> in, ImageView<rgb8> bg, ImageView<int> diff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.width && y < in.height) {
        rgb8* in_line = (rgb8*)((char*)in.buffer + y * in.stride);
        rgb8* bg_line = (rgb8*)((char*)bg.buffer + y * bg.stride);
        int* diff_line = (int*)((char*)diff.buffer + y * diff.stride);

        int d = abs((int)in_line[x].r - (int)bg_line[x].r) +
                abs((int)in_line[x].g - (int)bg_line[x].g) +
                abs((int)in_line[x].b - (int)bg_line[x].b);
        diff_line[x] = d;
    }
}

__device__ int get_neighborhood_cu(const ImageView<int>& img, int x, int y, bool find_max, int field) {
    int extremum = find_max ? -2147483648 : 2147483647;
    int half = field / 2;

    for (int j = -half; j <= half; j++) {
        for (int i = -half; i <= half; i++) {
            int nx = x + i;
            int ny = y + j;

            if (nx >= 0 && nx < img.width && ny >= 0 && ny < img.height) {
                int* line = (int*)((char*)img.buffer + ny * img.stride);
                int val = line[nx];
                if (find_max) {
                    if (val > extremum) extremum = val;
                } else {
                    if (val < extremum) extremum = val;
                }
            }
        }
    }
    return extremum;
}

__global__ void erosion_kernel(ImageView<int> src, ImageView<int> dst, int field) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < src.width && y < src.height) {
        int val = get_neighborhood_cu(src, x, y, false, field);
        int* dst_line = (int*)((char*)dst.buffer + y * dst.stride);
        dst_line[x] = val;
    }
}

__global__ void dilatation_kernel(ImageView<int> src, ImageView<int> dst, int field) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < src.width && y < src.height) {
        int val = get_neighborhood_cu(src, x, y, true, field);
        int* dst_line = (int*)((char*)dst.buffer + y * dst.stride);
        dst_line[x] = val;
    }
}

__global__ void hysteresis_threshold_kernel(ImageView<int> map, int low, int high, int* change_flag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < map.width && y < map.height) {
        int* line = (int*)((char*)map.buffer + y * map.stride);
        if (line[x] >= high) {
            line[x] = 255;
            // No flag needed here, strong pixels are set
        } else if (line[x] >= low) {
            line[x] = 128;
        } else {
            line[x] = 0;
        }
    }
}

__global__ void hysteresis_propagate_kernel(ImageView<int> map, int* change_flag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < map.width && y < map.height) {
        int* line = (int*)((char*)map.buffer + y * map.stride);
        if (line[x] == 128) {
            bool has_strong_neighbor = false;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
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
                if (has_strong_neighbor) break;
            }
            if (has_strong_neighbor) {
                line[x] = 255;
                *change_flag = 1;
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
        if (line[x] > 0) line[x] = 1;
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

__global__ void draw_border_kernel(ImageView<rgb8> in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Top and Bottom
    if (idx < in.width) {
        rgb8* top_line = (rgb8*)((char*)in.buffer);
        rgb8* bot_line = (rgb8*)((char*)in.buffer + (in.height - 1) * in.stride);
        top_line[idx] = {255, 0, 0};
        bot_line[idx] = {255, 0, 0};
    }
    
    // Left and Right
    if (idx < in.height) {
        rgb8* line = (rgb8*)((char*)in.buffer + idx * in.stride);
        line[0] = {255, 0, 0};
        line[in.width - 1] = {255, 0, 0};
    }
}


void compute_cu(ImageView<rgb8> in)
{
    // Static buffers to persist on GPU
    static Image<rgb8> device_in;
    static Image<rgb8> device_bg;
    static Image<int> device_diff_map;
    static Image<int> device_scratch_map; // For morphology double buffering
    
    // Resize buffers if needed
    if (device_in.width != in.width || device_in.height != in.height) {
        device_in = Image<rgb8>(in.width, in.height, true);
        device_bg = Image<rgb8>(in.width, in.height, true);
        device_diff_map = Image<int>(in.width, in.height, true);
        device_scratch_map = Image<int>(in.width, in.height, true);
    }

    // 1. Copy Input to Device
    cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);

    // 2. Initialize Background (First frame only)
    static bool first_frame = true;
    if (first_frame) {
        cudaMemcpy2D(device_bg.buffer, device_bg.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToDevice);
        first_frame = false;
    }

    // 3. Update Background (in place)
    // Note: The original logic updates background first, then computes diff. 
    // Here we need 'device_bg' to be updated.
    // Assuming background_removal_cu takes the 'current' frame and updates internal state, 
    // but for 'device_bg' used in diff calculation, we usually need the background image.
    // The provided 'background_removal_cu' likely works on 'device_in' to update 'reservoirs' 
    // and potentially outputs the background image? 
    // Wait, looking at cpp: background_removal_cpp(background); 
    // It updates the BACKGROUND image with new pixels.
    // So we should pass 'device_bg' to background_removal_cu?
    // But 'background_removal_cpp' takes 'in' (the background image buffer) and updates it using internal reservoirs.
    // Wait, the cpp logic was: 
    // static Image background; memcpy(background, in); background_removal(background);
    // This implies 'background' passed to removal is actually the CURRENT FRAME at first?
    // Actually, in the optimized CPP version provided earlier:
    // memcpy(bg, in); background_removal(bg);
    // So 'bg' becomes the updated background.
    
    // Let's replicate the CPP logic:
    // Copy IN to BG buffer
    cudaMemcpy2D(device_bg.buffer, device_bg.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToDevice);
    // Run background removal on BG buffer (it uses internal static state to update BG buffer)
    background_removal_cu(device_bg);


    // 4. Compute Difference
    dim3 block(32, 32);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

    compute_difference_kernel<<<grid, block>>>(device_in, device_bg, device_diff_map);

    // 5. Noise Cancellation (Erosion -> Dilation)
    // Erosion
    erosion_kernel<<<grid, block>>>(device_diff_map, device_scratch_map, 7); // 3 radius -> 7 field? CPP used 7.
    // Dilation (result back to diff_map)
    dilatation_kernel<<<grid, block>>>(device_scratch_map, device_diff_map, 7);

    // 6. Hysteresis
    static int* d_change_flag = nullptr;
    if (!d_change_flag) cudaMalloc(&d_change_flag, sizeof(int));
    
    hysteresis_threshold_kernel<<<grid, block>>>(device_diff_map, 4, 30, d_change_flag);
    
    int h_change_flag = 1;
    while(h_change_flag) {
        h_change_flag = 0;
        cudaMemcpy(d_change_flag, &h_change_flag, sizeof(int), cudaMemcpyHostToDevice);
        
        hysteresis_propagate_kernel<<<grid, block>>>(device_diff_map, d_change_flag);
        
        cudaMemcpy(&h_change_flag, d_change_flag, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    hysteresis_cleanup_kernel<<<grid, block>>>(device_diff_map);

    // 7. Binarize & Count
    binarize_kernel<<<grid, block>>>(device_diff_map);
    
    static int* d_count = nullptr;
    if (!d_count) cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    
    count_pixels_kernel<<<grid, block>>>(device_diff_map, d_count);
    
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    bool alert = h_count > 500;

    // 8. Visualization (Maskage)
    // Need to do noise cancel and hysteresis on mask? 
    // In CPP: maskage_process_cpp calls noise_cancel and hesteresys on the diff_map AGAIN?
    // No, CPP compute_cpp calls:
    // noise_cancel_cpp(diff_map)
    // hesteresys_cpp(diff_map)
    // maskage_cpp(in, diff_map)
    // So we already did the processing on 'device_diff_map'.
    
    maskage_kernel<<<grid, block>>>(device_in, device_diff_map);

    if (alert) {
        int max_dim = (in.width > in.height) ? in.width : in.height;
        dim3 border_grid((max_dim + 255) / 256);
        draw_border_kernel<<<border_grid, 256>>>(device_in);
    }

    // 9. Copy Result Back to Host
    cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
}
