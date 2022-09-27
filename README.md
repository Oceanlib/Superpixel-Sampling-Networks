# Superpixel-Sampling-Networks
we build a pytorch-cuda extension layer of pair_wise_distance for Superpixel Sampling Networks, which is modified from [[SSN-Pytorch]](https://github.com/perrying/ssn-pytorch/tree/master/lib/ssn).

## How to Install
1)Modified cxx_args and nvcc_args in setup.py according to your GPU version.
3)Run "python setup.py install"

## Example
An example for usage can be found in "check()" of pairWiseDistance.py
