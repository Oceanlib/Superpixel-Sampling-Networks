#include <stdio.h>
#include <iostream>
#include "pair_wise_distance.cuh"
#define CUDA_NUM_THREADS 256

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>


using at::Half;

// forward-------------------------------------------------------------------------------
template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__   pixel_features,
    const scalar_t* __restrict__   spixel_features,
    const scalar_t* __restrict__   spixel_indices,
    scalar_t*       __restrict__   dist_matrix,

    int batchsize, int channels,
    int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h

)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9)
        return;
    int cp = channels * num_pixels;
    int cs = channels * num_spixels;
    int b = index % batchsize;
    int spixel_offset = (index / batchsize) % 9;
    int p = (index / (batchsize * 9)) % num_pixels;
    int init_spix_index = spixel_indices[b * num_pixels + p];
    int x_index = init_spix_index % num_spixels_w;
    int spixel_offset_x = (spixel_offset % 3 - 1);
    int y_index = init_spix_index / num_spixels_w;
    int spixel_offset_y = (spixel_offset / 3 - 1);
    if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w)
    {
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = 1e16;
    }
    else if (y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h)
    {
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = 1e16;
    }
    else
    {
        int query_spixel_index = init_spix_index + spixel_offset_x + num_spixels_w * spixel_offset_y;
        scalar_t sum_squared_diff = 0;
        for (int c = 0; c < channels; c++)
        {
            sum_squared_diff += pow(pixel_features[b * cp + c * num_pixels + p] -
                                        spixel_features[b * cs + c * num_spixels + query_spixel_index],
                                    2);
        }
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = sum_squared_diff;
    }
    return ;
}


int forward_cuda(
    cudaStream_t stream,
    const at::Tensor& pixel_features,
    const at::Tensor& spixel_features,
    const at::Tensor& spixel_indices,
    
    at::Tensor& dist_matrix,

    int num_spixels_w,
    int num_spixels_h
)
{
    int error = 1;
    int batchsize = pixel_features.size(0);
    int channels = pixel_features.size(1);
    int num_pixels = pixel_features.size(2);
    int num_spixels = spixel_features.size(2);
    
    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    
    AT_DISPATCH_FLOATING_TYPES(
        dist_matrix.type(),
        "forward_kernel",
        ([&]{forward_kernel<scalar_t><<<block, CUDA_NUM_THREADS, 0, stream>>>(
               pixel_features.data<scalar_t>(),
               spixel_features.data<scalar_t>(),
               spixel_indices.data<scalar_t>(),
               dist_matrix.data<scalar_t>(),
               batchsize, channels,
               num_pixels, num_spixels,
               num_spixels_w, num_spixels_h
               ); 
               }
        )
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("gpu error in forward_cuda: %s\n", cudaGetErrorString(err));
        return error;
    }
    error = 0;
    return error;
}



// backward-------------------------------------------------------------------------------
template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ dist_matrix_grad,
    const scalar_t* __restrict__ pixel_features,
    const scalar_t* __restrict__ spixel_features,
    const scalar_t* __restrict__ spixel_indices,
    scalar_t* __restrict__ pixel_feature_grad,
    scalar_t* __restrict__ spixel_feature_grad,
    int batchsize, int channels, 
    int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return;
    int cp = channels * num_pixels;
    int cs = channels * num_spixels;
    int b = index % batchsize;
    int spixel_offset = (index / batchsize) % 9;
    int p = (index / (batchsize * 9)) % num_pixels;
    int init_spix_index = spixel_indices[b * num_pixels + p];
    int x_index = init_spix_index % num_spixels_w;
    int spixel_offset_x = (spixel_offset % 3 - 1);
    int y_index = init_spix_index / num_spixels_w;
    int spixel_offset_y = (spixel_offset / 3 - 1);
    if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w) return;
    else if (y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h) return;
    else {
        int query_spixel_index = init_spix_index + spixel_offset_x + num_spixels_w * spixel_offset_y;
        scalar_t dist_matrix_grad_val = dist_matrix_grad[b * (9 * num_pixels) + spixel_offset * num_pixels + p];
        for (int c=0; c<channels; c++)
        {
            scalar_t pix_value = pixel_features[b * cp + c * num_pixels + p];
            scalar_t spix_value = spixel_features[b * cs + c * num_spixels + query_spixel_index];
            scalar_t diff = (pix_value - spix_value) * dist_matrix_grad_val;
            atomicAdd(&pixel_feature_grad[b * cp + c * num_pixels + p], 2 * diff);
            atomicAdd(&spixel_feature_grad[b * cs + c * num_spixels + query_spixel_index], -2 * diff);
        }
    }
}


int backward_cuda(
    cudaStream_t stream,
    const at::Tensor& dist_matrix_grad,
    const at::Tensor& pixel_features,
    const at::Tensor& spixel_features,
    const at::Tensor& spixel_indices,
    
    at::Tensor& pixel_features_grad,
    at::Tensor& spixel_features_grad,
    
    int num_spixels_w, int num_spixels_h
){
    int error = 1;
    int batchsize = pixel_features.size(0);
    int channels = pixel_features.size(1);
    int num_pixels = pixel_features.size(2);
    int num_spixels = spixel_features.size(2);

    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
    
    AT_DISPATCH_FLOATING_TYPES(
        pixel_features_grad.type(), 
        "backward_kernel", 
        ([&]{backward_kernel<scalar_t><<<block, CUDA_NUM_THREADS, 0, stream>>>(
            dist_matrix_grad.data<scalar_t>(),
            pixel_features.data<scalar_t>(),
            spixel_features.data<scalar_t>(),
            spixel_indices.data<scalar_t>(),
            pixel_features_grad.data<scalar_t>(),
            spixel_features_grad.data<scalar_t>(),
            batchsize, channels, 
            num_pixels, num_spixels, 
            num_spixels_w, num_spixels_h);
            }
        )
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("gpu error in backward_cuda: %s\n", cudaGetErrorString(err));
        return error;
    }

    error = 0;
    return error;
}