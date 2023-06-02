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
    virtual void forward(bool, smart_stream) const = 0;
    virtual void backward(smart_stream) const = 0;
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
    void backward(smart_stream) const;
};

// ##################################################################################

class SparseMatmul : public Module
{
    shared_ptr<Variable> a, b, c;
    DevSparseIndex *sp;
    natural m, n, p;
    smart_event start_sparse_matmul_forward;
    smart_event start_set_input;

public:
    SparseMatmul(shared_ptr<Variable> a_,
                 shared_ptr<Variable> b_,
                 shared_ptr<Variable> c_,
                 DevSparseIndex *sp_,
                 natural m_,
                 natural n_,
                 natural p_,
                 smart_event event1,
                 smart_event event2);
    ~SparseMatmul(){};
    void forward(bool, smart_stream) const;
    void backward(smart_stream) const;
};

// ##################################################################################

class GraphSum : public Module
{
    shared_ptr<Variable> in, out;
    DevSparseIndex *graph;
    dev_shared_ptr<real> dev_graph_value;
    natural dim;
    smart_event event;

public:
    GraphSum(shared_ptr<Variable> in_, shared_ptr<Variable> out_, DevSparseIndex *graph_, dev_shared_ptr<real> dev_graph_value_, natural dim_, smart_event event_);
    ~GraphSum() {}
    void forward(bool, smart_stream) const;
    void backward(smart_stream) const;
};

// ##################################################################################

class ReLU : public Module
{
    shared_ptr<Variable> in;
    dev_shared_ptr<bool> dev_mask;

public:
    ReLU(shared_ptr<Variable> in_);
    void forward(bool, smart_stream) const;
    void backward(smart_stream) const;
};

// ##################################################################################

class Matmul : public Module
{
    shared_ptr<Variable> a, b, c;
    natural m, n, p;
    smart_event event_forward;
    smart_event event_backward;
    smart_stream my_stream;

public:
    Matmul(shared_ptr<Variable> a_,
           shared_ptr<Variable> b_,
           shared_ptr<Variable> c_,
           natural m_,
           natural n_,
           natural p_,
           smart_event event_forward_,
           smart_event event_backward_,
           smart_stream stream_);
    ~Matmul() {}
    void forward(bool, smart_stream) const;
    void backward(smart_stream) const;
};

// ##################################################################################

class CrossEntropyLoss : public Module
{
    shared_ptr<Variable> logits;
    dev_shared_ptr<integer> dev_truth;
    pinned_host_ptr<real> loss;
    natural num_classes;
    dev_shared_ptr<real> dev_loss_res;
    smart_event start_backward;

public:
    natural num_samples;
    CrossEntropyLoss(shared_ptr<Variable> logits_, dev_shared_ptr<integer> dev_truth_, pinned_host_ptr<real> loss_, natural num_classes_, smart_event event);
    ~CrossEntropyLoss(){};
    void set_num_samples(natural num_samples_);
    natural get_num_samples() const;
    void forward(bool, smart_stream) const;
    void backward(smart_stream) const;
};

// ##################################################################################

#endif