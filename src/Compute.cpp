#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"

#include <chrono>
#include <thread>
#include <limits>
#include <algorithm>
#include <cstring>
#include <vector>


/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);


/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);


int get_neighborhood(const ImageView<int>& img, int x, int y, bool find_max, int field=3)
{
    int extremum = find_max ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
    int half = field / 2;

    for (int i = -half; i <= half; i++)
    {
        for (int j = -half; j <= half; j++)
        {
            int nx = x + i;
            int ny = y + j;

            if (nx >= 0 && nx < img.width && ny >= 0 && ny < img.height)
            {
                int* line = (int*)((std::byte*)img.buffer + ny * img.stride);
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

void process_morphology(ImageView<int> map, int field, bool find_max)
{
    int* copy_buffer = (int*)malloc(map.height * map.stride);
    std::memcpy(copy_buffer, map.buffer, map.height * map.stride);

    ImageView<int> copy_view = map;
    copy_view.buffer = copy_buffer;

    for (int y = 0; y < map.height; y++)
    {
        int* line = (int*)((std::byte*)map.buffer + y * map.stride);
        for (int x = 0; x < map.width; x++)
        {
            line[x] = get_neighborhood(copy_view, x, y, field, find_max);
        }
    }

    free(copy_buffer);
}

void erosion_cpp(ImageView<int> map, int field=3)
{
    process_morphology(map, field, false);
}

void dilatation_cpp(ImageView<int> map, int field=3)
{
    process_morphology(map, field, true);
}

void noise_cancel_cpp(ImageView<int> map, int field=3)
{
  erosion_cpp(map, field);
  dilatation_cpp(map, field);
}

void hysteresis_threshold(ImageView<int> map, int low, int high, std::vector<std::pair<int, int>>& strong_pixels)
{
    strong_pixels.reserve(map.width * map.height / 10);
    for (int y = 0; y < map.height; y++)
    {
        int* line = (int*)((std::byte*)map.buffer + y * map.stride);
        for (int x = 0; x < map.width; x++)
        {
            if (line[x] >= high)
            {
                line[x] = 255;
                strong_pixels.push_back({x, y});
            }
            else if (line[x] >= low)
            {
                line[x] = 128;
            }
            else
            {
                line[x] = 0;
            }
        }
    }
}

void hysteresis_propagate(ImageView<int> map, std::vector<std::pair<int, int>>& strong_pixels)
{
    while (!strong_pixels.empty())
    {
        auto [cx, cy] = strong_pixels.back();
        strong_pixels.pop_back();

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0) continue;

                int nx = cx + dx;
                int ny = cy + dy;

                if (nx >= 0 && nx < map.width && ny >= 0 && ny < map.height)
                {
                    int* line = (int*)((std::byte*)map.buffer + ny * map.stride);
                    if (line[nx] == 128)
                    {
                        line[nx] = 255;
                        strong_pixels.push_back({nx, ny});
                    }
                }
            }
        }
    }
}

void hysteresis_cleanup(ImageView<int> map)
{
    for (int y = 0; y < map.height; y++)
    {
        int* line = (int*)((std::byte*)map.buffer + y * map.stride);
        for (int x = 0; x < map.width; x++)
        {
            if (line[x] == 128)
            {
                line[x] = 0;
            }
        }
    }
}

void hesteresys_cpp(ImageView<int> map, int th_low=4, int th_high=30)
{
    std::vector<std::pair<int, int>> strong_pixels;
    
    hysteresis_threshold(map, th_low, th_high, strong_pixels);

    hysteresis_propagate(map, strong_pixels);

    hysteresis_cleanup(map);
}

void maskage_cpp(ImageView<rgb8> in, ImageView<int> mask)
{
    for (int y = 0; y < in.height; y++)
    {
        rgb8* line = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        int* mask_line = (int*)((std::byte*)mask.buffer + y * mask.stride);
        for (int x = 0; x < in.width; x++)
        {
            line[x].r = uint8_t(line[x].r + 0.5 * line[x].r * mask_line[x]);
        }
    }
}

void maskage_process_cpp(ImageView<rgb8> in, ImageView<int> mask)
{
    noise_cancel_cpp(mask);
    hesteresys_cpp(mask);
    for (int y = 0; y < mask.height; y++)
    {
        int* mask_line = (int*)((std::byte*)mask.buffer + y * mask.stride);
        for (int x = 0; x < mask.width; x++)
        {
            if (mask_line[x] > 0) mask_line[x] = 1;
        }
    }
    maskage_cpp(in, mask);
}


/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
  for (int y = 0; y < in.height; ++y)
  {
    rgb8* lineptr = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    for (int x = 0; x < in.width; ++x)
    {
      lineptr[x].r = 0; // Back out red component

      if (x < logo_width && y < logo_height)
      {
        float alpha  = logo_data[y * logo_width + x] / 255.f;
        lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1 - alpha) * 255);
        lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1 - alpha) * 255);
      }
    }
  }

  // You can fake a long-time process with sleep
   {
     using namespace std::chrono_literals;
     std::this_thread::sleep_for(50ms);
   }
}


extern "C" {

  static Parameters g_params;

  void cpt_init(Parameters* params)
  {
    g_params = *params;
  }

  void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
  {
    auto img = ImageView<rgb8>{(rgb8*)buffer, width, height, stride};
    if (g_params.device == e_device_t::CPU)
      compute_cpp(img);
    else if (g_params.device == e_device_t::GPU)
      compute_cu(img);
  }

}