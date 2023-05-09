#include "../include/module.cuh"

// DROPOUT
// ##################################################################################

Dropout::Dropout(shared_ptr<Variable> in_, real p_, dev_shared_ptr<curandState> dev_rand_states_) : in(in_), p(p_), dev_rand_states(dev_rand_states_)
{
    if (in->dev_grad.get())
        dev_mask = dev_shared_ptr<bool>(in->size);
    else
        dev_mask = dev_shared_ptr<bool>();
}

// ##################################################################################

__global__ void dropout_kernel_forward(real *dev_data, bool *dev_mask, curandState *dev_rand_states,
                                       const natural size, const real p, const real scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        real x = curand_uniform(&dev_rand_states[i % N_THREADS]);
        bool keep = x >= p;
        dev_data[i] *= keep ? scale : 0;
        if (dev_mask)
            dev_mask[i] = keep;
    }
}

void Dropout::forward(bool training)
{
    if (!training)
        return;
    timer_start(TMR_DROPOUT_FW);
    real scale = 1 / (1 - p);
    dim3 n_blocks(CEIL(in->size, N_THREADS));
    dim3 n_threads(N_THREADS);
    dropout_kernel_forward<<<n_blocks, n_threads>>>(in->dev_data.get(), dev_mask.get(), dev_rand_states.get(), in->size, p, scale);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer_stop(TMR_DROPOUT_FW);
}

// ##################################################################################

void Dropout::backward() {}

// SPARSEMATMUL
// ##################################################################################

SparseMatmul::SparseMatmul(shared_ptr<Variable> a_, shared_ptr<Variable> b_, shared_ptr<Variable> c_, DevSparseIndex *sp_, natural m_, natural n_, natural p_) : a(a_), b(b_), c(c_), sp(sp_), m(m_), n(n_), p(p_) {}

// GRAPHSUM
// ##################################################################################

GraphSum::GraphSum(shared_ptr<Variable> in_, shared_ptr<Variable> out_, DevSparseIndex *graph_, natural dim_) : in(in_), out(out_), graph(graph_), dim(dim_) {}

// RELU
// ##################################################################################

ReLU::ReLU(shared_ptr<Variable> in_) : in(in_)
{
    dev_mask = dev_shared_ptr<bool>(in->size);
}

// MATMUL
// ##################################################################################

Matmul::Matmul(shared_ptr<Variable> a_, shared_ptr<Variable> b_, shared_ptr<Variable> c_, natural m_, natural n_, natural p_) : a(a_), b(b_), c(c_), m(m_), n(n_), p(p_) {}

// CROSSENTROPYLOSS
// ##################################################################################

CrossEntropyLoss::CrossEntropyLoss(shared_ptr<Variable> logits_, dev_shared_ptr<integer> dev_truth_, real *loss_, natural num_classes_) : logits(logits_), dev_truth(dev_truth_), loss(loss_), num_classes(num_classes_) {}
