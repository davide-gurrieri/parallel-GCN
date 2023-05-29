#include "../include/optim.cuh"
// #include <cmath>
// #include <cstdlib>

// ##################################################################################

AdamVariable::AdamVariable(shared_ptr<Variable> var, bool decay_) : dev_data(var->dev_data), dev_grad(var->dev_grad), size(var->size), decay(decay_)
{
    dev_m = dev_shared_ptr<real>(size);
    dev_v = dev_shared_ptr<real>(size);
    dev_m.set_zero(streams[0]);
    dev_v.set_zero(streams[0]);
    cudaStreamSynchronize(streams[0].get());
}

// ##################################################################################

Adam::Adam(const std::vector<std::pair<shared_ptr<Variable>, bool>> &vars_, AdamParams const *params_) : params(params_), step_count(0)
{
    for (const auto &v : vars_)
        vars.emplace_back(v.first, v.second);

    weight_decay = dev_shared_ptr<real>(1);
    beta1 = dev_shared_ptr<real>(1);
    beta2 = dev_shared_ptr<real>(1);
    eps = dev_shared_ptr<real>(1);
    weight_decay.copy_to_device_async(&(params->weight_decay), streams[0]);
    beta1.copy_to_device_async(&(params->beta1), streams[0]);
    beta2.copy_to_device_async(&(params->beta2), streams[0]);
    eps.copy_to_device_async(&(params->eps), streams[0]);
    cudaStreamSynchronize(streams[0].get());
}

// ##################################################################################

__global__ void adam_step_kernel(real *dev_data, const real *dev_grad, real *dev_m, real *dev_v, const natural size, const real *weight_decay, const real *beta1, const real *beta2, const real *eps, const bool decay, const real step_size)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
    {
        real grad = dev_grad[i];
        if (decay)
            grad += (*weight_decay) * dev_data[i];
        dev_m[i] = (*beta1) * dev_m[i] + (1.0 - (*beta1)) * grad;
        dev_v[i] = (*beta2) * dev_v[i] + (1.0 - (*beta2)) * grad * grad;
        dev_data[i] -= step_size * dev_m[i] / (sqrtf(dev_v[i]) + (*eps));
    }
}

void Adam::step()
{
    // timer_start(TMR_OPTIMIZER);

    step_count++;
    const real step_size = params->learning_rate * sqrtf(1 - powf(params->beta2, step_count)) / (1 - powf(params->beta1, step_count));

    for (natural i = 0; i < vars.size(); i++)
    {
        const natural n_blocks = std::min(CEIL(vars[i].size, N_THREADS), static_cast<natural>(N_BLOCKS));
        adam_step_kernel<<<n_blocks, N_THREADS, 0, streams[i + 1].get()>>>(vars[i].dev_data.get(), vars[i].dev_grad.get(), vars[i].dev_m.get(), vars[i].dev_v.get(), vars[i].size, weight_decay.get(), beta1.get(), beta2.get(), eps.get(), vars[i].decay, step_size);
        cudaEventRecord(events[i].get(), streams[i + 1].get());
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    }

    // timer_stop(TMR_OPTIMIZER);
}

// ##################################################################################