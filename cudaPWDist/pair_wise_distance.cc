#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h> //works for 1.0.0
#include "pair_wise_distance.cuh"

int forward(
    at::Tensor& pixel_features,
    at::Tensor& spixel_features,
    at::Tensor& spixel_indices,

    at::Tensor& dist_matrix,

    int num_spixels_w,
    int num_spixels_h
)
{
    int error = 1;

    error = forward_cuda(
        at::cuda::getCurrentCUDAStream(),
        pixel_features,
        spixel_features,
        spixel_indices,
        dist_matrix,
        num_spixels_w,
        num_spixels_h);

    if (error)
    {
        AT_ERROR("CUDA forward call failed");
    }

    return error;
}


int backward(
    at::Tensor& dist_matrix_grad,
    at::Tensor& pixel_features,
    at::Tensor& spixel_features,
    at::Tensor& spixel_indices,
    
    at::Tensor& pixel_features_grad,
    at::Tensor& spixel_features_grad,
    
    int num_spixels_w, 
    int num_spixels_h
)

{
    int error = 1;

    error = backward_cuda(
        at::cuda::getCurrentCUDAStream(),
        dist_matrix_grad,
        pixel_features,
        spixel_features,
        spixel_indices,
        pixel_features_grad,
        spixel_features_grad,
        num_spixels_w,
        num_spixels_h);

    if (error)
    {
        AT_ERROR("CUDA backwarp call failed");
    }

    return error;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &forward, "pair_wise_distance forward");
    m.def("backward", &backward, "pair_wise_distance backward");
}