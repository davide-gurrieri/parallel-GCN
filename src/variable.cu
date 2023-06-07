#include "../include/variable.cuh"

// ##################################################################################

void Variable::initialize_random()
{
    natural max_size = 0;
    for (const natural &size : sizes)
        max_size = std::max(max_size, size);

    dev_rand_states = dev_shared_ptr<RandState>(max_size);
    std::vector<RandState> states(max_size);
    for (auto &state : states)
    {
        integer x = 0, y = 0;
        while (x == 0 || y == 0)
        {
            x = rand();
            y = rand();
        }
        state.a = x;
        state.b = y;
    }
    dev_rand_states.copy_to_device(states.data());
}

// ##################################################################################

Variable::Variable(const natural size_, const bool requires_grad, const bool rand, const natural rows_, const natural cols_) : size(size_), rows(rows_), cols(cols_)
{
    dev_data = dev_shared_ptr<real>(size);
    if (requires_grad)
        dev_grad = dev_shared_ptr<real>(size);
    else
        dev_grad = dev_shared_ptr<real>();

    if (rand)
        sizes.push_back(size);
}

// ##################################################################################

__global__ void glorot_kernel(real *data, const natural size, const double scale, RandState *states)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
        data[i] = (unif(&states[i]) - 0.5) * scale;
}

void Variable::glorot() const
{
    if (!dev_rand_states.get())
    {
        std::cerr << "Variable::glorot: Variable must be initialized with rand = true" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (rows == 0 || cols == 0)
    {
        std::cerr << "Variable::glorot: rows and cols must be set" << std::endl;
        exit(EXIT_FAILURE);
    }
    const real range = sqrtf(6.0f / (rows + cols));
    const real scale = range * 2;
    const natural n_blocks = std::min(CEIL(size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    glorot_kernel<<<n_blocks, CudaParams::N_THREADS>>>(dev_data.get(), size, scale, dev_rand_states.get());
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

// ##################################################################################

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

// ##################################################################################