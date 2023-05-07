#ifndef VARIABLE_H
#define VARIABLE_H

#include "../include/utils.cuh"
#include <vector>
#include <curand_kernel.h>

class Variable
{
public:
    real *dev_data;
    real *dev_grad;
    natural size;
    curandState *dev_rand_states;

    Variable(natural size_, bool requires_grad = true, curandState *dev_rand_states_ = nullptr);
    void zero();
    void zero_grad();
    void glorot(natural in_size, natural out_size);
    ~Variable();
};

#endif