#include "../include/optim.cuh"
// #include <cmath>
// #include <cstdlib>

// ##################################################################################

AdamVariable::AdamVariable(shared_ptr<Variable> var, bool decay_) : dev_data(var->dev_data), dev_grad(var->dev_grad), size(var->size), decay(decay_)
{
    dev_m = dev_shared_ptr<real>(size);
    dev_v = dev_shared_ptr<real>(size);
    dev_m.set_zero();
    dev_v.set_zero();
}

// ##################################################################################

Adam::Adam(const std::vector<shared_ptr<Variable>> &weights, const std::vector<bool> &decays, AdamParams const *params_, const std::vector<smart_stream> &backward_streams_, const std::vector<smart_event> &start_matmul_forward_) : params(params_), step_count(0), backward_streams(backward_streams_), start_matmul_forward(start_matmul_forward_)
{

    if (weights.size() != decays.size())
    {
        std::cout << "Error in Adam constructor: weights and decays must have the same size" << std::endl;
        exit(1);
    }

    for (natural i = 0; i < weights.size(); i++)
        vars.emplace_back(weights[i], decays[i]);

    weight_decay = dev_shared_ptr<real>(1);
    beta1 = dev_shared_ptr<real>(1);
    beta2 = dev_shared_ptr<real>(1);
    eps = dev_shared_ptr<real>(1);

    weight_decay.copy_to_device(&(params->weight_decay));
    beta1.copy_to_device(&(params->beta1));
    beta2.copy_to_device(&(params->beta2));
    eps.copy_to_device(&(params->eps));
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
    // cudaStreamSynchronize(.get());
    // cudaStreamSynchronize(streams[1].get());
    /*
    for (const auto &var : vars)
    {
        const natural n_blocks = std::min(CEIL(var.size, N_THREADS), static_cast<natural>(N_BLOCKS));
        adam_step_kernel<<<n_blocks, N_THREADS, 0, .get()>>>(var.dev_data.get(), var.dev_grad.get(), var.dev_m.get(), var.dev_v.get(), var.size, weight_decay.get(), beta1.get(), beta2.get(), eps.get(), var.decay, step_size);
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    }
    */

    for (natural i = 0; i < vars.size(); i++)
    {
        const natural n_blocks = std::min(CEIL(vars[i].size, N_THREADS), static_cast<natural>(N_BLOCKS));
        adam_step_kernel<<<n_blocks, N_THREADS, 0, backward_streams[i].get()>>>(vars[i].dev_data.get(), vars[i].dev_grad.get(), vars[i].dev_m.get(), vars[i].dev_v.get(), vars[i].size, weight_decay.get(), beta1.get(), beta2.get(), eps.get(), vars[i].decay, step_size);
        cudaEventRecord(start_matmul_forward[i].get(), backward_streams[i].get());
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    }

    // timer_stop(TMR_OPTIMIZER);
}

// ##################################################################################