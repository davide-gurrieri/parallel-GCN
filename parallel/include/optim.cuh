#ifndef OPTIM_CUH
#define OPTIM_CUH
#include "../include/variable.cuh"
#include "../include/utils.cuh"
#include "../include/timer.h"
#include "../include/shared_ptr.cuh"
#include <utility>
#include <vector>
#include <memory>
using std::shared_ptr;

// ##################################################################################

struct AdamParams
{
    real lr, beta1, beta2, eps, weight_decay;
    static AdamParams get_default();
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
    dev_shared_ptr<AdamParams> dev_params;
    AdamParams params;
    natural step_count;
    std::vector<AdamVariable> vars;

public:
    Adam() {}
    Adam(const std::vector<std::pair<shared_ptr<Variable>, bool>> &vars_, AdamParams params_);
    void step();
};

// ##################################################################################

#endif