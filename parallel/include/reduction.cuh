#include "../include/utils.cuh"
/*
__inline__ __device__ void warpReduce(volatile real *input, int threadId)
// we need volatile flag here, otherwise the compiler might introduce some optimizations in the "input" variable
// and place it in registers instead of shared memory!
{
    input[threadId] += input[threadId + 32];
    input[threadId] += input[threadId + 16];
    input[threadId] += input[threadId + 8];
    input[threadId] += input[threadId + 4];
    input[threadId] += input[threadId + 2];
    input[threadId] += input[threadId + 1];
}

__global__ void p_sum_gpu(real *input);

__global__ void collect_res_gpu(real *input, int numOfBlocks, natural size);

real reduce_gpu(real *dev_input, natural size);

// ##########################################################################

#include <cooperative_groups.h>

// using namespace cooperative_groups;
using cooperative_groups::thread_group;

__device__ float reduce_sum(thread_group g, float *temp, float val);

__device__ int thread_sum(float *input, natural n);

__global__ void sum_reduction(float *sum, float *input, natural n);

real reduce_gpu_2(real *dev_input, natural size);
*/

__global__ void reduce(real *x, real *res, natural n);

__global__ void reduce(real *x, real *res, natural n);