#include "../include/module.cuh"

// DROPOUT
// ##################################################################################

Dropout::Dropout(shared_ptr<Variable> in_, real p_) : in(in_), p(p_)
{
    if (in->dev_grad.get())
        dev_mask = dev_shared_ptr<bool>(in->size);
    else
        dev_mask = dev_shared_ptr<bool>();
}

// ##################################################################################

__global__ void dropout_kernel_forward(real *dev_data, bool *dev_mask, randState *dev_rand_states,
                                       const natural size, const natural kernel_size, const real p, const real scale)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
    bool keep;
    natural j;
#pragma unroll
    for (natural i = id; i < kernel_size; i += blockDim.x * gridDim.x)
    {
        const float4 rand = curand_uniform4(&dev_rand_states[i]);
        j = i * 4;

        if (dev_mask)
        {
            keep = rand.x >= p;
            dev_data[j] *= keep ? scale : 0.f;
            dev_mask[j] = keep;
            if (++j < size)
            {
                keep = rand.y >= p;
                dev_data[j] *= keep ? scale : 0.f;
                dev_mask[j] = keep;
            }
            if (++j < size)
            {
                keep = rand.z >= p;
                dev_data[j] *= keep ? scale : 0.f;
                dev_mask[j] = keep;
            }
            if (++j < size)
            {
                keep = rand.w >= p;
                dev_data[j] *= keep ? scale : 0.f;
                dev_mask[j] = keep;
            }
        }
        else
        {
            dev_data[j] *= rand.x >= p ? scale : 0.f;
            if (++j < size)
                dev_data[j] *= rand.y >= p ? scale : 0.f;
            if (++j < size)
                dev_data[j] *= rand.z >= p ? scale : 0.f;
            if (++j < size)
                dev_data[j] *= rand.w >= p ? scale : 0.f;
        }
    }
}

void Dropout::forward(bool training, const smart_stream &stream) const
{
    if (!training)
        return;
    const real scale = 1.0 / (1.0 - p);
    const natural kernel_size = CEIL(in->size, 4);
    const natural n_blocks = std::min(CEIL(kernel_size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    dropout_kernel_forward<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(in->dev_data.get(), dev_mask.get(), Variable::dev_rand_states.get(), in->size, kernel_size, p, scale);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ##################################################################################

__global__ void dropout_kernel_backward(real *dev_grad, const bool *mask, const natural size, const real scale)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
    {
        dev_grad[i] *= mask[i] ? scale : 0.f;
    }
}

void Dropout::backward(const smart_stream &backward_stream) const
{
    if (!dev_mask.get())
        return;
    const real scale = 1.0 / (1.0 - p);
    const natural n_blocks = std::min(CEIL(in->size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    dropout_kernel_backward<<<n_blocks, CudaParams::N_THREADS, 0, backward_stream.get()>>>(in->dev_grad.get(), dev_mask.get(), in->size, scale);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// SPARSEMATMUL
// ##################################################################################

SparseMatmul::SparseMatmul(shared_ptr<Variable> a_, shared_ptr<Variable> b_, shared_ptr<Variable> c_, DevSparseIndex *sp_, natural m_, natural n_, natural p_, smart_event &start_matmul_forward_, smart_event &start_set_input_) : a(a_), b(b_), c(c_), sp(sp_), m(m_), n(n_), p(p_), start_matmul_forward(start_matmul_forward_), start_set_input(start_set_input_) {}

// ##################################################################################

__global__ void sparse_matmul_kernel_forward(const real *a, const real *b, real *c, const natural *indptr, const natural *indices, const natural m, const natural p)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < m * p; i += blockDim.x * gridDim.x)
    {
        const natural row = i / p;
        const natural col = i % p;
        real sum = 0;
#pragma unroll
        for (natural jj = indptr[row]; jj < indptr[row + 1]; jj++)
            sum += a[jj] * b[indices[jj] * p + col];
        c[i] = sum;
    }
}

void SparseMatmul::forward(bool training, const smart_stream &stream) const
{
    const natural n_blocks = std::min(CEIL(m * p, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    cudaStreamWaitEvent(stream.get(), start_matmul_forward.get());
    sparse_matmul_kernel_forward<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(a->dev_data.get(), b->dev_data.get(), c->dev_data.get(), sp->dev_indptr.get(), sp->dev_indices.get(), m, p);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ##################################################################################

__global__ void sparse_matmul_kernel_backward(const real *a, real *b, const real *c, const natural *indptr, const natural *indices, const natural m, const natural p)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < m * p; i += blockDim.x * gridDim.x)
    {
        const natural row = i / p;
        const natural col = i % p;
#pragma unroll
        for (natural jj = indptr[row]; jj < indptr[row + 1]; jj++)
        {
            natural j = indices[jj];
            atomicAdd(&b[j * p + col], a[jj] * c[row * p + col]);
        }
    }
}

void SparseMatmul::backward(const smart_stream &backward_stream) const
{
    b->zero_grad(backward_stream);
    const natural n_blocks = std::min(CEIL(m * p, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    sparse_matmul_kernel_backward<<<n_blocks, CudaParams::N_THREADS, 0, backward_stream.get()>>>(a->dev_data.get(), b->dev_grad.get(), c->dev_grad.get(), sp->dev_indptr.get(), sp->dev_indices.get(), m, p);
    cudaEventRecord(start_set_input.get(), backward_stream.get());
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// GRAPHSUM
// ##################################################################################

GraphSum::GraphSum(shared_ptr<Variable> in_, shared_ptr<Variable> out_, DevSparseIndex *graph_, dev_shared_ptr<real> dev_graph_value_, natural dim_, bool generate_event_, smart_event &start_matmul_backward_) : in(in_), out(out_), graph(graph_), dev_graph_value(dev_graph_value_), dim(dim_), generate_event(generate_event_), start_matmul_backward(start_matmul_backward_) {}

// ##################################################################################

__global__ void graphsum_kernel(const real *a, const real *b, real *c, const natural *indptr, const natural *indices, const natural m, const natural p)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < m * p; i += blockDim.x * gridDim.x)
    {
        const natural row = i / p;
        const natural col = i % p;
        real sum = 0;
#pragma unroll
        for (natural jj = indptr[row]; jj < indptr[row + 1]; jj++)
            sum += a[jj] * b[indices[jj] * p + col];
        c[i] = sum;
    }
}

void GraphSum::forward(bool training, const smart_stream &stream) const
{
    const natural numNodes = graph->indptr_size - 1;
    const natural n_blocks = std::min(CEIL(numNodes * dim, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    graphsum_kernel<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(dev_graph_value.get(), in->dev_data.get(), out->dev_data.get(), graph->dev_indptr.get(), graph->dev_indices.get(), numNodes, dim);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ###############################################################################

void GraphSum::backward(const smart_stream &backward_stream) const
{
    const natural numNodes = graph->indptr_size - 1;
    const natural n_blocks = std::min(CEIL(numNodes * dim, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    graphsum_kernel<<<n_blocks, CudaParams::N_THREADS, 0, backward_stream.get()>>>(dev_graph_value.get(), out->dev_grad.get(), in->dev_grad.get(), graph->dev_indptr.get(), graph->dev_indices.get(), numNodes, dim);
    if (generate_event)
        cudaEventRecord(start_matmul_backward.get(), backward_stream.get());
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// RELU
// ##################################################################################

ReLU::ReLU(shared_ptr<Variable> in_) : in(in_)
{
    dev_mask = dev_shared_ptr<bool>(in->size);
}

// ##################################################################################

__global__ void relu_kernel_forward(real *dev_data, bool *dev_mask, const natural size, const bool training)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
    {
        const bool keep = dev_data[i] > 0;
        if (training)
            dev_mask[i] = keep;
        if (!keep)
            dev_data[i] = 0.f;
    }
}

void ReLU::forward(bool training, const smart_stream &stream) const
{
    const natural n_blocks = std::min(CEIL(in->size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    relu_kernel_forward<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(in->dev_data.get(), dev_mask.get(), in->size, training);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ##################################################################################

__global__ void relu_kernel_backward(real *d_in_grad, const bool *d_mask, const natural size)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
    {
        if (!d_mask[i])
            d_in_grad[i] = 0.f;
    }
}

void ReLU::backward(const smart_stream &backward_stream) const
{
    const natural n_blocks = std::min(CEIL(in->size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    relu_kernel_backward<<<n_blocks, CudaParams::N_THREADS, 0, backward_stream.get()>>>(in->dev_grad.get(), dev_mask.get(), in->size);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// MATMUL
// ##################################################################################

Matmul::Matmul(shared_ptr<Variable> a_, shared_ptr<Variable> b_, shared_ptr<Variable> c_, natural m_, natural n_, natural p_, smart_event &event_forward_, smart_event &event_backward_, const smart_stream &stream_) : a(a_), b(b_), c(c_), m(m_), n(n_), p(p_), event_forward(event_forward_), event_backward(event_backward_), my_stream(stream_) {}

// ##################################################################################

__global__ void matmul_kernel_forward(const real *a, const real *b, real *c, const natural m, const natural n, const natural p)
{
    // shared memory arrays that are used as tiles to store a portion of matrices A and B.
    __shared__ real tile_a[CudaParams::TILE_DIM][CudaParams::TILE_DIM];
    __shared__ real tile_b[CudaParams::TILE_DIM][CudaParams::TILE_DIM];
    natural tx = threadIdx.x;
    natural ty = threadIdx.y;
    natural row = blockIdx.y * blockDim.x + ty;
    natural col = blockIdx.x * CudaParams::TILE_DIM + tx;
    //  number of tile rows/columns needed to cover the matrices A and B
    natural range = CEIL(n, CudaParams::TILE_DIM);
    //  partial sum of the result matrix element computed by the thread
    real res = 0;

#pragma unroll
    // iterates over the tiles needed to compute the result matrix element
    for (natural i = 0; i < range; i++)
    {
        // check if the current thread is within the boundaries of A .
        if (row < m && i * CudaParams::TILE_DIM + tx < n)
            // load a portion of matrix A into the shared memory tiles.
            tile_a[ty][tx] = a[row * n + i * CudaParams::TILE_DIM + tx];
        else
            tile_a[ty][tx] = 0;
        // check if the current thread is within the boundaries of  B.
        if (col < p && i * CudaParams::TILE_DIM + ty < n)
            // load a portion of matrix B into the shared memory tiles.
            tile_b[ty][tx] = b[(i * CudaParams::TILE_DIM + ty) * p + col];

        else
            tile_b[ty][tx] = 0;
        // synchronizes all threads in the block before executing the next set of instructions.
        __syncthreads();
#pragma unroll
        // computes the partial sum of the result matrix element using the shared memory tiles
        for (natural j = 0; j < CudaParams::TILE_DIM; j++)
            res += tile_a[ty][j] * tile_b[j][tx];

        __syncthreads();
    }
    // stores the result of the partial sum in the result matrix if the thread is within the boundaries of the result matrix
    if (row < m && col < p)
        c[row * p + col] = res;
}

void Matmul::forward(bool training, const smart_stream &stream) const
{
    const dim3 n_blocks(CEIL(p, CudaParams::TILE_DIM), CEIL(m, CudaParams::TILE_DIM));
    const dim3 n_threads(CudaParams::TILE_DIM, CudaParams::TILE_DIM);
    cudaStreamWaitEvent(stream.get(), event_forward.get());
    matmul_kernel_forward<<<n_blocks, n_threads, 0, stream.get()>>>(a->dev_data.get(), b->dev_data.get(), c->dev_data.get(), m, n, p);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ##################################################################################

__global__ void matmul_kernel_backward_1(real *a, const real *b, const real *c, const natural m, const natural n, const natural p)
{
    // shared memory arrays that are used as tiles to store a portion of matrices A and B.
    __shared__ real tile_c[CudaParams::TILE_DIM][CudaParams::TILE_DIM];
    __shared__ real tile_b[CudaParams::TILE_DIM][CudaParams::TILE_DIM];
    natural tx = threadIdx.x;
    natural ty = threadIdx.y;
    natural row = blockIdx.y * blockDim.x + ty;
    natural col = blockIdx.x * CudaParams::TILE_DIM + tx;
    //  number of tile rows/columns needed to cover the matrices A and B
    natural range = CEIL(p, CudaParams::TILE_DIM);
    //  partial sum of the result matrix element computed by the thread
    real res = 0;

#pragma unroll
    // iterates over the tiles needed to compute the result matrix element
    for (natural i = 0; i < range; i++)
    {
        // check if the current thread is within the boundaries of C .
        if (row < m && i * CudaParams::TILE_DIM + tx < p)
            // load a portion of matrix A into the shared memory tiles.
            tile_c[ty][tx] = c[row * p + i * CudaParams::TILE_DIM + tx];
        else
            tile_c[ty][tx] = 0;
        // check if the current thread is within the boundaries of  B.
        if (col < n && i * CudaParams::TILE_DIM + ty < p)
            // load a portion of matrix B into the shared memory tiles.
            tile_b[ty][tx] = b[col * p + i * CudaParams::TILE_DIM + ty];
        else
            tile_b[ty][tx] = 0;
        // synchronizes all threads in the block before executing the next set of instructions.
        __syncthreads();
#pragma unroll
        // computes the partial sum of the result matrix element using the shared memory tiles
        for (natural k = 0; k < CudaParams::TILE_DIM; k++)
            res += tile_c[ty][k] * tile_b[k][tx];

        __syncthreads();
    }
    // stores the result of the partial sum in the result matrix if the thread is within the boundaries of the result matrix
    if (row < m && col < n)
        a[row * n + col] = res;
}

// first version with atomicAdd
__global__ void matmul_kernel_backward_2(const real *a, real *b, const real *c, const natural m, const natural n, const natural p)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < m * p; i += blockDim.x * gridDim.x)
    {
        natural row = i / p;
        natural col = i % p;
        // i = row * p + col
        const real c_val = c[i];
#pragma unroll
        for (natural j = 0; j < n; j++)
            atomicAdd(&b[j * p + col], c_val * a[row * n + j]);
    }
}

// second version with long internal loop and shared memory
/*
__global__ void matmul_kernel_backward_2(const real *a, real *b, const real *c, const natural m, const natural n, const natural p)
{
    // shared memory arrays that are used as tiles to store a portion of matrices A and B.
    __shared__ real tile_a[TILE_DIM][TILE_DIM];
    __shared__ real tile_c[TILE_DIM][TILE_DIM];
    natural tx = threadIdx.x;
    natural ty = threadIdx.y;

    natural row = blockIdx.y * TILE_DIM + ty;
    natural col = blockIdx.x * TILE_DIM + tx;
    natural range = CEIL(m, TILE_DIM);
    real res = 0;
    // iterates over the tiles needed to compute the result matrix element
#pragma unroll
    for (natural i = 0; i < range; i++)
    {
        if (row < n && i * TILE_DIM + tx < m)
            tile_a[ty][tx] = a[(i * TILE_DIM + tx) * n + row];
        else
            tile_a[ty][tx] = 0;
        if (col < p && i * TILE_DIM + ty < m)
            tile_c[ty][tx] = c[(i * TILE_DIM + ty) * p + col];
        else
            tile_c[ty][tx] = 0;
        __syncthreads();

#pragma unroll
        for (natural k = 0; k < TILE_DIM; k++)
            res += tile_a[ty][k] * tile_c[k][tx];
        __syncthreads();
    }

    if (row < n && col < p)
        b[row * p + col] = res;
}

void Matmul::backward(const smart_stream &backward_stream) const
{
    // timer_start(TMR_MATMUL_BW);

    // b->zero_grad();
    //  a->zero_grad();
    const natural n_blocks_y_1 = std::min(CEIL(m, CudaParams::TILE_DIM), CudaParams::N_BLOCKS);
    dim3 n_blocks_1(CEIL(n, CudaParams::TILE_DIM), n_blocks_y_1);
    dim3 n_blocks_2(CEIL(p, CudaParams::TILE_DIM), CEIL(n, CudaParams::TILE_DIM));
    dim3 n_threads(CudaParams::TILE_DIM, CudaParams::TILE_DIM);
    matmul_kernel_backward_1<<<n_blocks_1, n_threads, 0, backward_stream.get()>>>(a->dev_grad.get(), b->dev_data.get(), c->dev_grad.get(), m, n, p);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    cudaStreamWaitEvent(my_stream.get(), event_backward.get());
    matmul_kernel_backward_2<<<n_blocks_2, n_threads, 0, my_stream.get()>>>(a->dev_data.get(), b->dev_grad.get(), c->dev_grad.get(), m, n, p);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    // cudaStreamSynchronize(streams[0].get());

    // timer_stop(TMR_MATMUL_BW);
}
*/

void Matmul::backward(const smart_stream &backward_stream) const
{
    const dim3 n_blocks_1(CEIL(n, CudaParams::TILE_DIM), CEIL(m, CudaParams::TILE_DIM));
    const dim3 n_threads(CudaParams::TILE_DIM, CudaParams::TILE_DIM);
    matmul_kernel_backward_1<<<n_blocks_1, n_threads, 0, backward_stream.get()>>>(a->dev_grad.get(), b->dev_data.get(), c->dev_grad.get(), m, n, p);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif

    const natural n_blocks_2 = std::min(CEIL(m * p, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    cudaStreamWaitEvent(my_stream.get(), event_backward.get());
    b->zero_grad(my_stream);
    matmul_kernel_backward_2<<<n_blocks_2, CudaParams::N_THREADS, 0, my_stream.get()>>>(a->dev_data.get(), b->dev_grad.get(), c->dev_grad.get(), m, n, p);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// CROSSENTROPYLOSS
// ##################################################################################

CrossEntropyLoss::CrossEntropyLoss(shared_ptr<Variable> logits_, dev_shared_ptr<integer> dev_truth_, pinned_host_ptr<real> loss_, natural num_classes_, smart_event &event) : logits(logits_), dev_truth(dev_truth_), loss(loss_), num_classes(num_classes_), start_backward(event)
{
    dev_loss_res = dev_shared_ptr<real>(1);
}

// ##################################################################################

__global__ void cross_entropy_loss_kernel(real *dev_data, real *dev_grad, const integer *dev_truth, real *dev_loss_res, const natural num_classes, const natural num_nodes, const natural num_samples, bool training)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
    const natural warp_size = 32;
    real sum = static_cast<real>(0);
#pragma unroll
    for (natural i = id; i < num_nodes; i += blockDim.x * gridDim.x)
    {
        if (dev_truth[i] < 0)
            continue;
        real *logit = &dev_data[i * num_classes];
        real sum_exp = 0.;
        real max_logit = logit[0];
#pragma unroll
        // get the maximum value of each node
        for (natural j = 1; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);
#pragma unroll
        for (natural j = 0; j < num_classes; j++)
        {
            logit[j] -= max_logit; // numerical stability
            sum_exp += expf(logit[j]);
        }
        sum += logf(sum_exp) - logit[dev_truth[i]];

        if (training)
        {
#pragma unroll
            for (natural j = 0; j < num_classes; j++)
            {
                real prob = expf(logit[j]) / sum_exp;
                dev_grad[i * num_classes + j] = prob / num_samples;
            }
            dev_grad[i * num_classes + dev_truth[i]] -= 1.0 / num_samples;
        }
    }
    sum = warp_reduce(sum);
    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)
        atomicAdd(dev_loss_res, sum);
}

void CrossEntropyLoss::forward(bool training, const smart_stream &stream) const
{
    if (training)
        logits->zero_grad(stream);

    dev_loss_res.set_zero(stream);
    const natural DIM = logits->size / num_classes;
    const natural n_blocks = std::min(CEIL(DIM, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    cross_entropy_loss_kernel<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(logits->dev_data.get(), logits->dev_grad.get(), dev_truth.get(), dev_loss_res.get(), num_classes, DIM, num_samples, training);
    if (training)
        cudaEventRecord(start_backward.get(), stream.get());
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    dev_loss_res.copy_to_host_async(loss.get(), stream);
}

// ##################################################################################

void CrossEntropyLoss::backward(const smart_stream &backward_stream) const
{
    cudaStreamWaitEvent(backward_stream.get(), start_backward.get());
}

// ##################################################################################

void CrossEntropyLoss::set_num_samples(natural num_samples_)
{
    num_samples = num_samples_;
}

// ##################################################################################

natural CrossEntropyLoss::get_num_samples() const
{
    return num_samples;
};

// ##################################################################################

#ifdef RESIDUAL_CONNECTIONS
ResidualConnection::ResidualConnection(shared_ptr<Variable> prev_, shared_ptr<Variable> current_) : prev(prev_), current(current_)
{
    if (prev->size != current->size)
    {
        std::cerr << "ResidualConnection: prev and current must have the same size" << std::endl;
        exit(1);
    }
    size = prev->size;
}

// ##################################################################################

__global__ void residual_connection_kernel(const real *prev, real *current, const natural size)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
        current[i] += prev[i];
}

void ResidualConnection::forward(bool training, const smart_stream &stream) const
{
    const natural n_blocks = std::min(CEIL(size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    residual_connection_kernel<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(prev->dev_data.get(), current->dev_data.get(), size);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ##################################################################################
#endif