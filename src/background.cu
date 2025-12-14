// // #include "background.hpp"
// #include <cuda_runtime.h>
// #include <curand_kernel.h>
//
// __device__ Reservoir* reservoirs_d = nullptr;
// __device__ int g_width_d = 0;
// __device__ int g_height_d = 0;
// __device__ curandState* randStates_d = nullptr;
//
//
// find matching reservoir
// __device__ int find_matching_reservoir_d(rgb8 p, Reservoir* rs)
// {
//     int m_idx = -1;
//     for (int i = 0; i < RESERVOIR_K; i++)
//     {
//         if (rs[i].w > 0)
//         {
//             if (abs((int)p.r - (int)rs[i].rgb.r) < RGB_DIFF_THRESHOLD &&
//                 abs((int)p.g - (int)rs[i].rgb.g) < RGB_DIFF_THRESHOLD &&
//                 abs((int)p.b - (int)rs[i].rgb.b) < RGB_DIFF_THRESHOLD)
//             {
//                 m_idx = i;
//                 return m_idx;
//             }
//         }
//         else
//         {
//             m_idx = i;
//         }
//     }
//     return m_idx;
// }
//
// __global__ void background_removal_kernel(ImageView<rgb8> in)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//     if (x < in.width && y < in.height)
//     {
//         Reservoir* pixel_reservoirs = &reservoirs_d[(y * g_width_d + x) * RESERVOIR_K];
//         
//         rgb8* lineptr = (rgb8*)((std::byte*)in.buffer + y * in.stride);
//         rgb8 current_pixel = lineptr[x];
//
//         int m_idx = find_matching_reservoir_d(current_pixel, pixel_reservoirs);
//
//         if (m_idx != -1 && pixel_reservoirs[m_idx].w > 0) { // matching
//             pixel_reservoirs[m_idx].w += 1;
//             pixel_reservoirs[m_idx].rgb.r = (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.r + current_pixel.r) / pixel_reservoirs[m_idx].w);
//             pixel_reservoirs[m_idx].rgb.g = (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.g + current_pixel.g) / pixel_reservoirs[m_idx].w);
//             pixel_reservoirs[m_idx].rgb.b = (uint8_t)(((pixel_reservoirs[m_idx].w - 1) * pixel_reservoirs[m_idx].rgb.b + current_pixel.b) / pixel_reservoirs[m_idx].w);
//         } else if (m_idx != -1 && pixel_reservoirs[m_idx].w == 0) { // empty slot
//             pixel_reservoirs[m_idx].rgb = current_pixel;
//             pixel_reservoirs[m_idx].w = 1;
//         } else { // no match and no empty slot, perform weighted reservoir replacement
//             int min_idx = 0;
//             float min_weight = pixel_reservoirs[0].w;
//             float total_weights = pixel_reservoirs[0].w;
//             for (int i = 1; i < RESERVOIR_K; ++i) {
//                 if (pixel_reservoirs[i].w < min_weight) {
//                     min_weight = pixel_reservoirs[i].w;
//                     min_idx = i;
//                 }
//                 total_weights += pixel_reservoirs[i].w;
//             }
//             
//             curandState* localRandState = &randStates_d[y * g_width_d + x];
//             if (curand_uniform(localRandState) * total_weights >= min_weight) {
//                 pixel_reservoirs[min_idx].rgb = current_pixel;
//                 pixel_reservoirs[min_idx].w = 1;
//             }
//         }
//
//         // cap weights to MAX_WEIGHTS
//         float max_w = 0;
//         int max_idx = 0;
//         for(int i = 0; i < RESERVOIR_K; ++i) {
//             if (pixel_reservoirs[i].w > MAX_WEIGHTS) {
//                 pixel_reservoirs[i].w = MAX_WEIGHTS;
//             }
//             if (pixel_reservoirs[i].w > max_w) {
//                 max_w = pixel_reservoirs[i].w;
//                 max_idx = i;
//             }
//         }
//
//         // set the background
//         lineptr[x] = pixel_reservoirs[max_idx].rgb;
//     }
// }
//
// __global__ void init_rand_states(int width, int height, curandState* states) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//     if (x < width && y < height) {
//         curand_init(clock64(), y * width + x, 0, &states[y * width + x]);
//     }
// }
//
//
// void background_removal_cu(ImageView<rgb8> in) {
//     static bool g_initialized = false;
//
//     if (!g_initialized) {
//         Reservoir* d_res_ptr = nullptr;
//         cudaMalloc((void**)&d_res_ptr, in.width * in.height * RESERVOIR_K * sizeof(Reservoir));
//         cudaMemset(d_res_ptr, 0, in.width * in.height * RESERVOIR_K * sizeof(Reservoir));
//         cudaMemcpyToSymbol(reservoirs_d, &d_res_ptr, sizeof(Reservoir*));
//         
//         curandState* d_rand_ptr = nullptr;
//         cudaMalloc((void**)&d_rand_ptr, in.width * in.height * sizeof(curandState));
//         cudaMemcpyToSymbol(randStates_d, &d_rand_ptr, sizeof(curandState*));
//         
//         dim3 block(16, 16);
//         dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
//         init_rand_states<<<grid, block>>>(in.width, in.height, d_rand_ptr);
//         
//         cudaMemcpyToSymbol(g_width_d, &in.width, sizeof(int));
//         cudaMemcpyToSymbol(g_height_d, &in.height, sizeof(int));
//
//         g_initialized = true;
//     }
//
//     dim3 block(16, 16);
//     dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
//
//     background_removal_kernel<<<grid, block>>>(in);
// }
