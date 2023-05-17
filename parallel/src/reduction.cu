#include "../include/reduction.cuh"

/*
__global__ void p_sum_gpu(real *input) // compute a parallel redution for each gpu block using multiple threads
{
    natural threadId = threadIdx.x;
    natural index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ real localVars[512]; // we don't need more than this, since we can have 512 threads at most

    if (threadId < blockDim.x / 2)
    {
        localVars[threadId] = input[index] + input[index + blockDim.x / 2];
    }
    __syncthreads();

    for (natural i = blockDim.x / 4; i > 32; i >>= 1)
    {
        if (threadId < i)
        {
            localVars[threadId] += localVars[threadId + i];
        }
        __syncthreads();
    }
    if (threadId < 32)
        warpReduce(localVars, threadId);
    __syncthreads();

    if (threadId == 0)
        input[index] = localVars[threadId];
    __syncthreads();
}

__global__ void collect_res_gpu(real *input, int numOfBlocks, natural size) // compute the final reduction
{
    natural threadId = threadIdx.x;
    natural i;
    __shared__ real localVars[1024];
    localVars[threadId] = 0;
    __syncthreads();
    for (i = 0; i < numOfBlocks; i += blockDim.x) // collect the result of the various blocks
    {
        if ((threadId + i) * blockDim.x < size)
        {
            localVars[threadId] += input[(threadId + i) * blockDim.x];
        }
        __syncthreads();
    }

    for (i = blockDim.x / 2; i > 32; i >>= 1) // compute the parallel reduction for the collected data
    {
        if (threadId < i)
        {
            localVars[threadId] += localVars[threadId + i];
        }
        __syncthreads();
    }
    if (threadId < 32)
        warpReduce(localVars, threadId);
    __syncthreads();

    if (threadId == 0)
        input[threadId] = localVars[threadId];
    __syncthreads();
}

real reduce_gpu(real *dev_input, natural size)
{
    real ret;
    dim3 blocksPerGrid(CEIL(size, N_THREADS));
    dim3 threadsPerBlock(N_THREADS);
    p_sum_gpu<<<blocksPerGrid, threadsPerBlock>>>(dev_input); // call the reduction for all the blocks
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    collect_res_gpu<<<1, threadsPerBlock>>>(dev_input, blocksPerGrid.x, size); // finish the results collection using a single block
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(&ret, dev_input, sizeof(real), cudaMemcpyDeviceToHost)); // retrieve the results from the GPU
    return ret;
}

// #####################################################



// Reduces a thread group to a single element
__device__ float reduce_sum(thread_group g, float *temp, float val)
{
    natural lane = g.thread_rank();

    // Each thread adds its partial sum[i] to sum[lane+i]
    for (natural i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        // wait for all threads to store
        g.sync();
        if (lane < i)
        {
            val += temp[lane + i];
        }
        // wait for all threads to load
        g.sync();
    }
    // note: only thread 0 will return full sum
    return val;
}

// Creates partials sums from the original array
__device__ int thread_sum(float *input, natural n)
{
    float sum = 0;
    natural tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (natural i = tid; i < n / 4; i += blockDim.x * gridDim.x)
    {
        // Cast as int4
        float4 in = ((float4 *)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

__global__ void sum_reduction(float *sum, float *input, natural n)
{
    // Create partial sums from the array
    float my_sum = thread_sum(input, n);

    // Dynamic shared memory allocation
    extern __shared__ float temp[];

    // Identifier for a TB
    auto g = this_thread_block();

    // Reudce each TB
    float block_sum = reduce_sum(g, temp, my_sum);

    // Collect the partial result from each TB
    if (g.thread_rank() == 0)
    {
        atomicAdd(sum, block_sum);
    }
}

// Grid Size (cut in half)
// int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE;

// Call kernel with dynamic shared memory (Could decrease this to fit larger data)
// sum_reduction <<<GRID_SIZE, TB_SIZE, n * sizeof(int)>>> (sum, data, n);

real reduce_gpu_2(real *dev_input, natural size)
{
    dev_shared_ptr<real> sum = dev_shared_ptr<real>(1);
    sum.set_zero();
    dim3 n_blocks(CEIL(size, N_THREADS));
    dim3 n_threads(N_THREADS);
    sum_reduction<<<n_blocks, n_threads, size * sizeof(float)>>>(sum.get(), dev_input, size);
    float ret = 0;
    sum.copy_to_host(&ret);
    return ret;
}
*/

// reduction
__device__ real warp_reduce(real val)
{
    natural warp_size = 32;
    for (natural offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce(real *x, real *res, natural n)
{
    natural warp_size = 32;
    real sum = static_cast<real>(0); // Initialize partial sum for this thread;
    for (natural i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        sum += x[i]; // Each thread accumulates N / total_threads values;
    }
    sum = warp_reduce(sum);
    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)
        // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(res, sum);
    // The first thread in the warp updates the output;
}