#include "../include/optim.cuh"
// #include <cmath>
// #include <cstdlib>

AdamParams AdamParams::get_default() { return {0.001, 0.9, 0.999, 1e-8, 0.0}; }

// ##################################################################################

AdamVariable::AdamVariable(shared_ptr<Variable> var, bool decay_) : dev_data(var->dev_data), dev_grad(var->dev_grad), size(var->size), decay(decay_)
{
    dev_m = dev_shared_ptr<real>(size);
    dev_v = dev_shared_ptr<real>(size);
}

// ##################################################################################

Adam::Adam(std::vector<std::pair<shared_ptr<Variable>, bool>> vars_, AdamParams params_) : params(params_)
{
    step_count = 0;
    for (auto v : vars_)
        vars.emplace_back(v.first, v.second);
    dev_params = dev_shared_ptr<AdamParams>(1);
    dev_params.copy_to_device(&params);
}

__global__ void adam_step_kernel(real *dev_data, real *dev_grad, real *dev_m, real *dev_v, natural size, AdamParams *params, bool decay, real step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float grad = dev_grad[i];
        if (decay) // never used, weight decay set to zero
            grad += params->weight_decay * dev_data[i];
        dev_m[i] = params->beta1 * dev_m[i] + (1.0 - params->beta1) * grad;
        dev_v[i] = params->beta2 * dev_v[i] + (1.0 - params->beta2) * grad * grad;
        dev_data[i] -= step_size * dev_m[i] / (sqrtf(dev_v[i]) + params->eps);
    }
}

void Adam::step()
{
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) /
                      (1 - std::pow(params.beta1, step_count));

    for (auto &var : vars)
    {
        dim3 n_blocks(CEIL(var.size, N_THREADS));
        dim3 n_threads(N_THREADS);
        adam_step_kernel<<<n_blocks, n_threads>>>(var.dev_data.get(), var.dev_grad.get(), var.dev_m.get(), var.dev_v.get(), var.size, dev_params.get(), var.decay, step_size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

// ##################################################################################