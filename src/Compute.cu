#include "Compute.hpp"
#include "Image.hpp"
#include "background.hpp"


void compute_cu(ImageView<rgb8> in)
{
    // Copy the input image to the device
    Image<rgb8> device_in(in.width, in.height, true);
    cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
    
    background_removal_cu(device_in);

    // Copy the result back to the host
    cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
}