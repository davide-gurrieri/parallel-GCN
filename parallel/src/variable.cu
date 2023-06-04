#include "../include/variable.cuh"

// ##################################################################################

__global__ void initialize_var_random_kernel(randState *dev_rand_states, const natural seed, const natural size)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
        curand_init(seed * i, i, 0, &dev_rand_states[i]); // curand_init(seed, sequence, offset, &state);
}

/*
Variable::Variable(const natural size_, const bool requires_grad, const bool weights) : size(size_)
{
    dev_data = dev_shared_ptr<real>(size);
    if (requires_grad)
        dev_grad = dev_shared_ptr<real>(size);
    else
        dev_grad = dev_shared_ptr<real>();

    if (weights)
    {
        dev_rand_states = dev_shared_ptr<randState>(size);

        const natural n_blocks = std::min(CEIL(size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
        initialize_var_random_kernel<<<n_blocks, CudaParams::N_THREADS>>>(dev_rand_states.get(), CudaParams::SEED, size);
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    }
    std::cout << "Variable: " << size << std::endl;
}
*/

Variable::Variable(const natural size_, const bool requires_grad, const bool weights, const natural rows_, const natural cols_) : size(size_), rows(rows_), cols(cols_)
{
    dev_data = dev_shared_ptr<real>(size);
    if (requires_grad)
        dev_grad = dev_shared_ptr<real>(size);
    else
        dev_grad = dev_shared_ptr<real>();

    if (weights)
        sizes.push_back(size);

    std::cout << "Variable: " << size << std::endl;
}

// ##################################################################################

__global__ void glorot_kernel(real *data, const natural size, const natural kernel_size, const double scale, randState *state)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
    natural j;
#pragma unroll
    for (natural i = id; i < kernel_size; i += blockDim.x * gridDim.x)
    {
        const float4 randoms = curand_uniform4(&state[i]);
        j = i * 4;
        data[j] = (randoms.x - 0.5) * scale;
        if (++j < size)
            data[j] = (randoms.y - 0.5) * scale;
        if (++j < size)
            data[j] = (randoms.z - 0.5) * scale;
        if (++j < size)
            data[j] = (randoms.w - 0.5) * scale;
    }
}

void Variable::glorot() const
{
    if (!dev_rand_states.get())
    {
        std::cerr << "Variable::glorot: Variable must be initialized with weights = true" << std::endl;
        exit(EXIT_FAILURE);
    }
    const double range = sqrtf(6.0f / (rows + cols));
    const double scale = range * 2;
    const natural kernel_size = CEIL(size, 4);
    const natural n_blocks = std::min(CEIL(kernel_size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    glorot_kernel<<<n_blocks, CudaParams::N_THREADS>>>(dev_data.get(), size, kernel_size, scale, dev_rand_states.get());
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
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

__global__ void set_value_kernel(real *data, const real value, const natural size)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
        data[i] = value;
}

void Variable::set_value(const real value, smart_stream stream) const
{
    const natural n_blocks = std::min(CEIL(size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    set_value_kernel<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(dev_data.get(), value, size);
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

void Variable::save(const std::string &file_name, const std::string &what, natural col) const
{

    std::vector<real> data(size);
    if (what == "data")
        dev_data.copy_to_host(data.data());
    else if (what == "grad")
        dev_grad.copy_to_host(data.data());
    else
    {
        std::cerr << "Variable::print: what must be either 'data' or 'grad'" << std::endl;
        exit(EXIT_FAILURE);
    }
    int count = 0;
    std::ofstream file(file_name);
    if (file.is_open())
    {
        for (const auto &element : data)
        {
            file << element << " ";
            count++;
            if (count % col == 0)
                file << "\n";
        }
        file.close();
        std::cout << "Vector saved to file: " << file_name << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << file_name << std::endl;
    }
}

void Variable::initialize_random()
{
    natural max_size = 0;
    for (const natural &size : sizes)
        max_size = std::max(max_size, size);
    const natural state_size = CEIL(max_size, 4);
    dev_rand_states = dev_shared_ptr<randState>(state_size);

    const natural n_blocks = std::min(CEIL(state_size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    initialize_var_random_kernel<<<n_blocks, CudaParams::N_THREADS>>>(dev_rand_states.get(), CudaParams::SEED, state_size);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    std::cout << "random state: " << state_size << std::endl;
}