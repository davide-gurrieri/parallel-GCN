#ifndef MODULE_CUH
#define MODULE_CUH

#include <memory>
#include <cuda_runtime.h>
using std::shared_ptr;
using std::unique_ptr;

// #include <cuda_runtime.h>

#include "../include/utils.cuh"
#include "../include/variable.cuh"
#include "../include/timer.h"
#include "../include/sparse.cuh"
#include "../include/shared_ptr.cuh"
#include "../include/reduction.cuh" // for CrossEntropyLoss
#include "../include/smart_object.cuh"

// ##################################################################################

class Module
{
public:
    virtual void forward(bool, smart_stream stream) const = 0;
    virtual void backward() const = 0;
    virtual void set_num_samples(natural){};
    virtual natural get_num_samples() const { return 0; };
    virtual ~Module(){};
};

// ##################################################################################

class Dropout : public Module
{
    shared_ptr<Variable> in;
    dev_shared_ptr<bool> dev_mask;
    real p;
    dev_shared_ptr<randState> dev_rand_states;

public:
    Dropout(shared_ptr<Variable> in_, real p_, dev_shared_ptr<randState> dev_rand_states_);
    void forward(bool, smart_stream) const;
    void backward() const;
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
    void forward(bool, smart_stream) const;
    void backward() const;
};

// ##################################################################################

class GraphSum : public Module
{
    shared_ptr<Variable> in, out;
    DevSparseIndex *graph;
    dev_shared_ptr<real> dev_graph_value;
    natural dim;

public:
    GraphSum(shared_ptr<Variable> in_, shared_ptr<Variable> out_, DevSparseIndex *graph_, dev_shared_ptr<real> dev_graph_value_, natural dim_);
    ~GraphSum() {}
    void forward(bool, smart_stream) const;
    void backward() const;
};

// ##################################################################################

class ReLU : public Module
{
    shared_ptr<Variable> in;
    dev_shared_ptr<bool> dev_mask;

public:
    ReLU(shared_ptr<Variable> in_);
    void forward(bool, smart_stream) const;
    void backward() const;
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
    void forward(bool, smart_stream) const;
    void backward() const;
};

// ##################################################################################

class CrossEntropyLoss : public Module
{
    shared_ptr<Variable> logits;
    dev_shared_ptr<integer> dev_truth;
    natural num_classes;
    dev_shared_ptr<real> dev_loss_train;
    dev_shared_ptr<real> dev_loss_eval;

public:
    natural num_samples;
    CrossEntropyLoss(shared_ptr<Variable> logits_, dev_shared_ptr<integer> dev_truth_, natural num_classes_, dev_shared_ptr<real> dev_loss_train_, dev_shared_ptr<real> dev_loss_eval_);
    ~CrossEntropyLoss(){};
    void set_num_samples(natural num_samples_);
    void forward(bool, smart_stream) const;
    void backward() const;
};

// ##################################################################################

#endif