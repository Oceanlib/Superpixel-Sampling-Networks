#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int forward_cuda(
    cudaStream_t stream,
    const at::Tensor& pixel_features,
    const at::Tensor& spixel_features,
    const at::Tensor& spixel_indices,

    at::Tensor& dist_matrix,

    int num_spixels_w,
    int num_spixels_h);


int backward_cuda(
    cudaStream_t stream,
    const at::Tensor& dist_matrix_grad,
    const at::Tensor& pixel_features,
    const at::Tensor& spixel_features,
    const at::Tensor& spixel_indices,
    
    at::Tensor& pixel_features_grad,
    at::Tensor& spixel_features_grad,
    
    int num_spixels_w, 
    int num_spixels_h);