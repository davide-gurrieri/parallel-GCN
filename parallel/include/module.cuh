#ifndef MODULE_CUH
#define MODULE_CUH

#include <memory>
using std::shared_ptr;
using std::unique_ptr;

#include "../include/utils.cuh"
#include "../include/variable.cuh"
#include "../include/timer.h"
#include "../include/sparse.cuh"

// ##################################################################################

class Module
{
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module(){};
};

// ##################################################################################

class Dropout : public Module
{
    shared_ptr<Variable> in;
    bool *dev_mask;
    real p;
    curandState *dev_rand_states;

public:
    Dropout(shared_ptr<Variable> in_, real p_, curandState *dev_rand_states_);
    ~Dropout();
    void forward(bool);
    void backward();
};

// ##################################################################################

class SparseMatmul : public Module
{
    shared_ptr<Variable> a, b, c;
    DevSparseIndex *sp;
    natural m, n, p;

public:
    SparseMatmul(shared_ptr<Variable> a_,
                 shared_ptr<Variable> b_,
                 shared_ptr<Variable> c_,
                 DevSparseIndex *sp_,
                 natural m_,
                 natural n_,
                 natural p_);
    ~SparseMatmul(){};
    void forward(bool){};
    void backward(){};
};

// ##################################################################################

class GraphSum : public Module
{
    shared_ptr<Variable> in, out;
    DevSparseIndex *graph;
    natural dim;

public:
    GraphSum(shared_ptr<Variable> in_, shared_ptr<Variable> out_, DevSparseIndex *graph_, natural dim_);
    ~GraphSum() {}
    void forward(bool){};
    void backward(){};
};

// ##################################################################################

class ReLU : public Module
{
    shared_ptr<Variable> in;
    bool *mask;

public:
    ReLU(shared_ptr<Variable> in);
    ~ReLU();
    void forward(bool){};
    void backward(){};
};

// ##################################################################################

class Matmul : public Module
{
    shared_ptr<Variable> a, b, c;
    natural m, n, p;

public:
    Matmul(shared_ptr<Variable> a_,
           shared_ptr<Variable> b_,
           shared_ptr<Variable> c_,
           natural m_,
           natural n_,
           natural p_);
    ~Matmul() {}
    void forward(bool){};
    void backward(){};
};

// ##################################################################################

class CrossEntropyLoss : public Module
{
    shared_ptr<Variable> logits;
    integer *dev_truth;
    real *loss;
    natural num_classes;

public:
    CrossEntropyLoss(shared_ptr<Variable> logits_, integer *dev_truth_, real *loss_, natural num_classes_);
    ~CrossEntropyLoss(){};
    void forward(bool){};
    void backward(){};
};

// ##################################################################################

#endif