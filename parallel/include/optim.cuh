#ifndef OPTIM_CUH
#define OPTIM_CUH
#include "../include/variable.cuh"
#include "../include/utils.cuh"
#include "../include/timer.h"
#include "../include/shared_ptr.cuh"
#include "../include/smart_object.cuh"
#include <utility>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
using std::shared_ptr;

// ##################################################################################

struct AdamParams
{
    real learning_rate{0.01}, beta1{0.9}, beta2{0.999}, eps{1e-8}, weight_decay{5e-4};
};

// ##################################################################################

struct AdamVariable
{
public:
    dev_shared_ptr<real> dev_data, dev_grad, dev_m, dev_v;
    natural size;
    bool decay;
    AdamVariable(shared_ptr<Variable>, bool);
};

// ##################################################################################

class Adam
{
    const AdamParams *params;
    dev_shared_ptr<real> weight_decay, beta1, beta2, eps;
    natural step_count;
    std::vector<AdamVariable> vars;
    std::vector<smart_stream> backward_streams;
    std::vector<smart_event> start_matmul_forward;

public:
    Adam(){};
    Adam(const std::vector<shared_ptr<Variable>> &weights, const std::vector<bool> &decays, AdamParams const *params_,
         const std::vector<smart_stream> &backward_streams_, const std::vector<smart_event> &start_matmul_forward_);
    void step();
};

// ##################################################################################

#endif