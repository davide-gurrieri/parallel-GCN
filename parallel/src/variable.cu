#include "../include/variable.cuh"

// ##################################################################################

Variable::Variable(natural size_, bool requires_grad, dev_shared_ptr<randState> dev_rand_states_) : size(size_)
{
    dev_data = dev_shared_ptr<real>(size);
    if (requires_grad)
        dev_grad = dev_shared_ptr<real>(size);
    else
        dev_grad = dev_shared_ptr<real>();

    if (dev_rand_states_.get())
    {
        dev_rand_states = dev_rand_states_;
    }
    else
    {
        dev_rand_states = dev_shared_ptr<randState>();
    }

    std::cout << "Variable created with size " << size << std::endl;
}

// ##################################################################################

__global__ void glorot_kernel(real *data, natural size, real scale, randState *state)
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
    glorot_kernel<<<n_blocks, n_threads>>>(dev_data.get(), size, scale, dev_rand_states.get());
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ##################################################################################
/*
__global__ void zero_kernel(real *data, natural size)
{
    natural i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        data[i] = static_cast<real>(0.);
}
*/

void Variable::zero()
{
    dev_data.set_zero();
}

void Variable::zero_grad()
{
    dev_grad.set_zero();
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