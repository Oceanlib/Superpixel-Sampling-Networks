#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = ['-gencode', 'arch=compute_86,code=sm_86',
             '-gencode', 'arch=compute_50,code=sm_50',
             '-gencode', 'arch=compute_52,code=sm_52',
             '-gencode', 'arch=compute_60,code=sm_60',
             '-gencode', 'arch=compute_61,code=sm_61']

setup(
    name='pair_wise_distance_cuda',
    ext_modules=[
        CUDAExtension('pair_wise_distance_cuda', [
            'pair_wise_distance.cc',
            'pair_wise_distance_cuda.cu'
            ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
