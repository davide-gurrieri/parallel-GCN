#include "../include/optim.cuh"
// #include <cmath>
// #include <cstdlib>

AdamParams AdamParams::get_default() { return {0.001, 0.9, 0.999, 1e-8, 0.0}; }

// ##################################################################################

AdamVariable::AdamVariable(shared_ptr<Variable> var, bool decay_) : dev_data(var->dev_data), dev_grad(var->dev_grad), size(var->size), decay(decay_)
{
    count++;
    std::cout << "constructor: count=" << count << std::endl;
    dev_m = dev_shared_ptr<real>(size);
    dev_v = dev_shared_ptr<real>(size);
}
/*
AdamVariable::AdamVariable(const AdamVariable &adam_var)
{
    count++;
    std::cout << "copy_constructor: count=" << count << std::endl;
    dev_data = nullptr;
    dev_grad = nullptr;
    dev_m = nullptr;
    dev_v = nullptr;
    size = adam_var.size;
    decay = adam_var.decay;
}
*/
/*
AdamVariable &AdamVariable::operator=(const AdamVariable &adam_var)
{

    dev_data = adam_var.dev_data;
    dev_grad = adam_var.dev_grad;
    dev_m = adam_var.dev_m;
    dev_v = adam_var.dev_v;
    size = adam_var.size;
    decay = adam_var.decay;
    return *this;
}
*/
/*
AdamVariable::~AdamVariable()
{
    count++;
    std::cout << "destructor: count=" << count << std::endl;
    CHECK_CUDA_ERROR(cudaFree(dev_m));
    CHECK_CUDA_ERROR(cudaFree(dev_v));
}
*/

// ##################################################################################

Adam::Adam(std::vector<std::pair<shared_ptr<Variable>, bool>> vars_, AdamParams params_) : params(params_)
{
    step_count = 0;
    std::cout << vars_.size() << std::endl;
    for (auto v : vars_)
        vars.emplace_back(v.first, v.second);
}
/*
__global__ void adam_step_kernel(real *dev_data, real *dev_grad, real *dev_m, real *dev_v, natural size, bool decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {

    }
}

void Adam::step()
{
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) /
                      (1 - powf(params.beta1, step_count));
    for (auto &var : vars)
    {
        for (int i = 0; i < var.size(); i++)
        {
            float grad = (*var.grad)[i];
            if (var.decay) // never used, weight decay set to zero
                grad += params.weight_decay * (*var.data)[i];
            var.m[i] = params.beta1 * var.m[i] + (1.0 - params.beta1) * grad;
            var.v[i] = params.beta2 * var.v[i] + (1.0 - params.beta2) * grad * grad;
            (*var.data)[i] -= step_size * var.m[i] / (sqrtf(var.v[i]) + params.eps);
        }
    }
}
*/

// ##################################################################################