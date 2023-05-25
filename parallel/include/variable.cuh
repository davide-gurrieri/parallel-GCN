#ifndef VARIABLE_CUH
#define VARIABLE_CUH

#include "../include/utils.cuh"
#include "../include/shared_ptr.cuh"
#include <vector>
#include <curand_kernel.h>
#include "../include/smart_object.cuh"

class Variable
{
public:
    dev_shared_ptr<real> dev_data;
    dev_shared_ptr<real> dev_grad;
    natural size;
    dev_shared_ptr<randState> dev_rand_states;

    Variable(const natural size_, const bool requires_grad = true, const dev_shared_ptr<randState> dev_rand_states_ = dev_shared_ptr<randState>());
    Variable() = default;
    void print(const std::string &what, natural col) const;
    void zero(smart_stream stream) const;
    void zero_grad(smart_stream stream) const;
    void glorot(const natural in_size, const natural out_size) const;
};

#endif