#ifndef REDUCTION_CUH
#define REDUCTION_CUH
#include "../include/utils.cuh"

__device__ real warp_reduce(real val);

__global__ void reduce(const real *x, real *res, const natural n);

#endif