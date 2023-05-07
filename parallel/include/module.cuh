#ifndef MODULE_CUH
#define MODULE_CUH

#include "../include/utils.cuh"
#include "../include/variable.cuh"
#include "../include/timer.h"
#include "../include/sparse.cuh"

class Module
{
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module(){};
};

// ########################################################

class Dropout : public Module
{
    Variable *in;
    bool *dev_mask;
    real p;
    curandState *dev_rand_states;

public:
    Dropout(Variable *in_, real p_, curandState *dev_rand_states_);
    ~Dropout();
    void forward(bool);
    void backward();
};

// ########################################################

class SparseMatmul : public Module
{
    Variable *a, *b, *c;
    DevSparseIndex *sp;
    natural m, n, p;

public:
    SparseMatmul(Variable *a_,
                 Variable *b_,
                 Variable *c_,
                 DevSparseIndex *sp_,
                 natural m_,
                 natural n_,
                 natural p_);
    ~SparseMatmul(){};
    void forward(bool){};
    void backward(){};
};

// ########################################################

class GraphSum : public Module
{
    Variable *in, *out;
    DevSparseIndex *graph;
    natural dim;

public:
    GraphSum(Variable *in_, Variable *out_, DevSparseIndex *graph_, natural dim_);
    ~GraphSum() {}
    void forward(bool){};
    void backward(){};
};

// ########################################################

class ReLU : public Module
{
    Variable *in;
    bool *mask;

public:
    ReLU(Variable *in);
    ~ReLU();
    void forward(bool){};
    void backward(){};
};

// ########################################################

class Matmul : public Module
{
    Variable *a, *b, *c;
    natural m, n, p;

public:
    Matmul(Variable *a_,
           Variable *b_,
           Variable *c_,
           natural m_,
           natural n_,
           natural p_);
    ~Matmul() {}
    void forward(bool){};
    void backward(){};
};

// ########################################################

class CrossEntropyLoss : public Module
{
    Variable *logits;
    natural *truth;
    real *loss;
    natural num_classes;

public:
    CrossEntropyLoss(Variable *logits_, natural *truth_, real *loss_, natural num_classes_);
    ~CrossEntropyLoss(){};
    void forward(bool){};
    void backward(){};
};

// ########################################################

#endif