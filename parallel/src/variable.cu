#include "../include/variable.cuh"

Variable::Variable(natural size_, bool requires_grad, curandState *dev_rand_states_) : size(size_), dev_rand_states(dev_rand_states_)
{
    size = size_;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_data, size * sizeof(real)));
    if (requires_grad)
        CHECK_CUDA_ERROR(cudaMalloc(&dev_grad, size * sizeof(real)));
    else
        dev_grad = nullptr;
}

Variable::~Variable()
{
    CHECK_CUDA_ERROR(cudaFree(dev_data));
    if (dev_grad)
        CHECK_CUDA_ERROR(cudaFree(dev_grad));
}

__global__ void glorot_kernel(real *data, natural size, real scale, curandState *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        data[i] = (curand_uniform(&state[i % N_THREADS]) - 0.5) * scale;
}

void Variable::glorot(natural in_size, natural out_size)
{
    real range = sqrtf(6.0f / (in_size + out_size));
    real scale = range * 2;
    dim3 n_blocks(CEIL(size, N_THREADS));
    dim3 n_threads(N_THREADS);
    glorot_kernel<<<n_blocks, n_threads>>>(dev_data, size, scale, dev_rand_states);
}
