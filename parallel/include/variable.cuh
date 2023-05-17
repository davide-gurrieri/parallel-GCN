#ifndef VARIABLE_H
#define VARIABLE_H

#include "../include/utils.cuh"
#include "../include/shared_ptr.cuh"
#include <vector>
#include <curand_kernel.h>

class Variable
{
public:
    dev_shared_ptr<real> dev_data;
    dev_shared_ptr<real> dev_grad;
    natural size;
    dev_shared_ptr<randState> dev_rand_states;
    void print(natural col) const;

    Variable(natural size_, bool requires_grad = true, dev_shared_ptr<randState> dev_rand_states_ = dev_shared_ptr<randState>());
    void zero();
    void zero_grad();
    void glorot(natural in_size, natural out_size);
};

#endif