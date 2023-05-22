#include "../include/variable.cuh"

// ##################################################################################

Variable::Variable(const natural size_, const bool requires_grad, const dev_shared_ptr<randState> dev_rand_states_) : size(size_)
{
    dev_data = dev_shared_ptr<real>(size);
    if (requires_grad)
        dev_grad = dev_shared_ptr<real>(size);
    else
        dev_grad = dev_shared_ptr<real>();

    if (dev_rand_states_.get())
        dev_rand_states = dev_rand_states_;
    else
        dev_rand_states = dev_shared_ptr<randState>();

    // std::cout << "Variable created with size " << size << std::endl;
}

// ##################################################################################

__global__ void glorot_kernel(real *data, const natural size, const real scale, randState *state)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
        data[i] = (curand_uniform(&state[i % N_THREADS]) - 0.5) * scale;
}

void Variable::glorot(const natural in_size, const natural out_size) const
{
    const real range = sqrtf(6.0f / (in_size + out_size));
    const real scale = range * 2;
    const natural n_blocks = std::min(CEIL(size, N_THREADS), N_BLOCKS);
    glorot_kernel<<<n_blocks, N_THREADS>>>(dev_data.get(), size, scale, dev_rand_states.get());
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    cudaStreamSynchronize(streams[0].get());
}

// ##################################################################################

void Variable::zero(smart_stream stream) const
{
    dev_data.set_zero(stream);
}

void Variable::zero_grad(smart_stream stream) const
{
    dev_grad.set_zero(stream);
}

// ##################################################################################

void Variable::print(const std::string &what, natural col) const
{

    real *data = new real[size];
    if (what == "data")
        dev_data.copy_to_host(data);
    else if (what == "grad")
        dev_grad.copy_to_host(data);
    else
    {
        delete[] data;
        std::cerr << "Variable::print: what must be either 'data' or 'grad'" << std::endl;
        exit(EXIT_FAILURE);
    }
    int count = 0;
    for (natural i = 0; i < 20 * col && i < size; i++)
    {
        printf("%.4f ", data[i]);
        count++;
        if (count % col == 0)
            printf("\n");
    }
    delete[] data;
}