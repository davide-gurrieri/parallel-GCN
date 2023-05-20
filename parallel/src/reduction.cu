#include "../include/reduction.cuh"

__device__ real warp_reduce(real val)
{
    const natural warp_size = 32;
    for (natural offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce(const real *x, real *res, const natural n)
{
    const natural warp_size = 32;
    real sum = static_cast<real>(0); // Initialize partial sum for this thread;
    for (natural i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += x[i]; // Each thread accumulates N / total_threads values;
    sum = warp_reduce(sum);
    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)
        // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(res, sum);
    // The first thread in the warp updates the output;
}