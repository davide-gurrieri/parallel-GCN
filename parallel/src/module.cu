#include "../include/module.cuh"

// DROPOUT
// ##################################################################################

Dropout::Dropout(shared_ptr<Variable> in_, real p_, dev_shared_ptr<randState> dev_rand_states_) : in(in_), p(p_), dev_rand_states(dev_rand_states_)
{
    if (in->dev_grad.get())
        dev_mask = dev_shared_ptr<bool>(in->size);
    else
        dev_mask = dev_shared_ptr<bool>();
}

// ##################################################################################
/*
__global__ void dropout_kernel_forward(real *dev_data, bool *dev_mask, randState *dev_rand_states,
                                       const natural size, const real p, const real scale)
{
    __shared__ randState s_rand_states[N_THREADS_DROPOUT];
    s_rand_states[threadIdx.x] = dev_rand_states[threadIdx.x];
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
    {
        bool keep = curand_uniform(&s_rand_states[threadIdx.x]) >= p;
        dev_data[i] *= keep ? scale : 0.f;
        if (dev_mask)
            dev_mask[i] = keep;
    }
}
*/
// needs curandStatePhilox4_32_10_t
__global__ void dropout_kernel_forward(real *dev_data, bool *dev_mask, randState *dev_rand_states,
                                       const natural size, const real p, const real scale)
{
    __shared__ randState s_rand_states[N_THREADS_DROPOUT];
    s_rand_states[threadIdx.x] = dev_rand_states[threadIdx.x];
    natural id = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

#pragma unroll
    for (natural i = id; i < size; i += 4 * blockDim.x * gridDim.x)
    {
        float4 rand = curand_uniform4(&s_rand_states[threadIdx.x]);
        dev_data[i] *= rand.x >= p ? scale : 0;
        dev_data[i + 1] *= rand.y >= p ? scale : 0;
        dev_data[i + 2] *= rand.z >= p ? scale : 0;
        dev_data[i + 3] *= rand.w >= p ? scale : 0;
        if (dev_mask)
        {
            dev_mask[i] = rand.x >= p;
            dev_mask[i + 1] = rand.y >= p;
            dev_mask[i + 2] = rand.z >= p;
            dev_mask[i + 3] = rand.w >= p;
        }
    }
}

void Dropout::forward(bool training)
{
    if (!training)
        return;
    timer_start(TMR_DROPOUT_FW);
    real scale = 1.0 / (1.0 - p);
    natural n_blocks = std::min(CEIL(in->size, N_THREADS_DROPOUT), static_cast<natural>(N_BLOCKS));
    // natural n_blocks = CEIL(in->size, N_THREADS_DROPOUT);
    dropout_kernel_forward<<<n_blocks, N_THREADS_DROPOUT>>>(in->dev_data.get(), dev_mask.get(), dev_rand_states.get(), in->size, p, scale);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // std::cout << "size of rand: " << sizeof(curandStatePhilox4_32_10_t) << std::endl;
    // std::cout << "n_blocks: " << n_blocks << std::endl;
    timer_stop(TMR_DROPOUT_FW);
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

void Dropout::backward()
{
    if (!dev_mask.get())
        return;
    timer_start(TMR_DROPOUT_BW);

    const real scale = 1.0 / (1.0 - p);
    natural n_blocks = std::min(CEIL(in->size, N_THREADS), static_cast<natural>(N_BLOCKS));
    dropout_kernel_backward<<<n_blocks, N_THREADS>>>(in->dev_grad.get(), dev_mask.get(), in->size, scale);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_DROPOUT_BW);
}

// ##################################################################################

// SPARSEMATMUL
// ##################################################################################

SparseMatmul::SparseMatmul(shared_ptr<Variable> a_, shared_ptr<Variable> b_, shared_ptr<Variable> c_, DevSparseIndex *sp_, natural m_, natural n_, natural p_) : a(a_), b(b_), c(c_), sp(sp_), m(m_), n(n_), p(p_) {}

// ##################################################################################

__global__ void sparse_matmul_kernel_forward(const real *a, const real *b, real *c, const natural *indptr, const natural *indices, natural m, natural p)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < m * p; i += blockDim.x * gridDim.x)
    {
        natural row = i / p;
        natural col = i % p;
        if (row >= m || col >= p)
            return;
        real sum = 0;
#pragma unroll
        for (natural jj = indptr[row]; jj < indptr[row + 1]; jj++)
        {
            natural j = indices[jj];
            sum += a[jj] * b[j * p + col];
        }
        c[i] = sum;
    }
}

void SparseMatmul::forward(bool training)
{
    timer_start(TMR_SPMATMUL_FW);

    natural n_blocks = std::min(CEIL(m * p, N_THREADS), static_cast<natural>(N_BLOCKS));
    sparse_matmul_kernel_forward<<<n_blocks, N_THREADS>>>(a->dev_data.get(), b->dev_data.get(), c->dev_data.get(), sp->dev_indptr.get(), sp->dev_indices.get(), m, p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_SPMATMUL_FW);
}

// ##################################################################################

__global__ void sparse_matmul_kernel_backward(const real *a, real *b, const real *c, const natural *indptr, const natural *indices, const natural m, const natural p)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < m * p; i += blockDim.x * gridDim.x)
    {
        natural row = i / p;
        natural col = i % p;
        if (row >= m || col >= p)
            return;
#pragma unroll
        for (natural jj = indptr[row]; jj < indptr[row + 1]; jj++)
        {
            natural j = indices[jj];
            atomicAdd(&b[j * p + col], a[jj] * c[row * p + col]);
            // b[j * p + col] += a[jj] * c[row * p + col]; BUG!
        }
    }
}

void SparseMatmul::backward()
{
    timer_start(TMR_SPMATMUL_BW);

    b->zero_grad();
    natural n_blocks = std::min(CEIL(m * p, N_THREADS), static_cast<natural>(N_BLOCKS));
    sparse_matmul_kernel_backward<<<n_blocks, N_THREADS>>>(a->dev_data.get(), b->dev_grad.get(), c->dev_grad.get(), sp->dev_indptr.get(), sp->dev_indices.get(), m, p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_SPMATMUL_BW);
}

// serial version
/*
void SparseMatmul::backward()
{
    timer_start(TMR_SPMATMUL_BW);
    std::vector<real> my_a(a->size);
    std::vector<real> my_c_grad(c->size);
    std::vector<real> my_b_grad(b->size, 0.f);
    std::vector<natural> my_indptr(sp->indptr_size);
    std::vector<natural> my_indices(sp->indices_size);
    a->dev_data.copy_to_host(my_a.data());
    c->dev_grad.copy_to_host(my_c_grad.data());

    sp->dev_indptr.copy_to_host(my_indptr.data());
    sp->dev_indices.copy_to_host(my_indices.data());

    b->zero_grad();
    int row = 0;
    for (int i = 0; i < my_indptr.size() - 1; i++)
        for (int k = 0; k < p; k++)
            for (int jj = my_indptr[i]; jj < my_indptr[i + 1]; jj++)
            {
                int j = my_indices[jj];

                my_b_grad[j * p + k] += my_c_grad[i * p + k] * my_a[jj];
            }
    b->dev_grad.copy_to_device(my_b_grad.data());
    timer_stop(TMR_SPMATMUL_BW);
}
*/
/*
__global__ void cuda_SparseMatmul_backward_kernel(float *a_in, float *b_in, float *c_in, natural *indptr, natural *indices, int p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

#pragma unroll
    for (int jj = indptr[i]; jj < indptr[i + 1]; jj++)
    {
        int j = indices[jj];
        b_in[j * p + k] += c_in[i * p + k] * a_in[jj];
    }
}

void SparseMatmul::backward()
{
    timer_start(TMR_SPMATMUL_BW);

    b->zero_grad();
    // TODO: when p larger than 1024?
    if (sp->indptr_size <= 1)
        return;
    dim3 block(sp->indptr_size - 1, 1, 1);
    dim3 thread_in_block(p, 1, 1);
    cuda_SparseMatmul_backward_kernel<<<block, thread_in_block>>>(a->dev_data.get(), b->dev_grad.get(), c->dev_grad.get(), sp->dev_indptr.get(), sp->dev_indices.get(), p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_SPMATMUL_BW);
}
*/
// GRAPHSUM
// ##################################################################################

GraphSum::GraphSum(shared_ptr<Variable> in_, shared_ptr<Variable> out_, DevSparseIndex *graph_, dev_shared_ptr<real> dev_graph_value_, natural dim_) : in(in_), out(out_), graph(graph_), dev_graph_value(dev_graph_value_), dim(dim_) {}

// ##################################################################################

// n è H mentre m è N
__global__ void graphsum_kernel(const real *dev_graph_value, const real *dev_in, real *dev_out, const natural *indptr, const natural *indices, const natural m, const natural n)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < m * n; i += blockDim.x * gridDim.x)
    {
        natural row = i / n;
        natural col = i % n;
        if (row >= m || col >= n)
            return;
        real sum = 0;
#pragma unroll
        for (natural jj = indptr[row]; jj < indptr[row + 1]; jj++)
        {
            natural j = indices[jj];
            // float coef = 1.0 / sqrtf((indptr[row + 1] - indptr[row]) * (indptr[j + 1] - indptr[j]));
            sum += dev_graph_value[jj] * dev_in[j * n + col];
        }
        dev_out[i] = sum;
    }
}

void GraphSum::forward(bool training)
{

    timer_start(TMR_GRAPHSUM_FW);

    const natural numNodes = graph->indptr_size - 1;
    const natural n_blocks = std::min(CEIL(numNodes * dim, N_THREADS), static_cast<natural>(N_BLOCKS));
    graphsum_kernel<<<n_blocks, N_THREADS>>>(dev_graph_value.get(), in->dev_data.get(), out->dev_data.get(), graph->dev_indptr.get(), graph->dev_indices.get(), numNodes, dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_GRAPHSUM_FW);
}

// ###############################################################################

void GraphSum::backward()
{
    timer_start(TMR_GRAPHSUM_BW);

    const natural numNodes = graph->indptr_size - 1;
    const natural n_blocks = std::min(CEIL(numNodes * dim, N_THREADS), static_cast<natural>(N_BLOCKS));
    graphsum_kernel<<<n_blocks, N_THREADS>>>(dev_graph_value.get(), out->dev_grad.get(), in->dev_grad.get(), graph->dev_indptr.get(), graph->dev_indices.get(), numNodes, dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_GRAPHSUM_BW);
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
        bool keep = dev_data[i] > 0;
        if (training)
            dev_mask[i] = keep;
        if (!keep)
            dev_data[i] = 0.f;
    }
}

void ReLU::forward(bool training)
{
    timer_start(TMR_RELU_FW);

    const natural n_blocks = std::min(CEIL(in->size, N_THREADS), static_cast<natural>(N_BLOCKS));
    relu_kernel_forward<<<n_blocks, N_THREADS>>>(in->dev_data.get(), dev_mask.get(), in->size, training);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_RELU_FW);
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

void ReLU::backward()
{
    timer_start(TMR_RELU_BW);

    const natural n_blocks = std::min(CEIL(in->size, N_THREADS), static_cast<natural>(N_BLOCKS));
    relu_kernel_backward<<<n_blocks, N_THREADS>>>(in->dev_grad.get(), dev_mask.get(), in->size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_RELU_BW);
}

// MATMUL
// ##################################################################################

Matmul::Matmul(shared_ptr<Variable> a_, shared_ptr<Variable> b_, shared_ptr<Variable> c_, natural m_, natural n_, natural p_) : a(a_), b(b_), c(c_), m(m_), n(n_), p(p_) {}

// ##################################################################################

// esperimenti con tiles rettangolari
/*
__global__ void matmul_kernel(const real *a, const real *b, real *c, const natural m, const natural n, const natural p)
{
    // shared memory arrays that are used as tiles to store a portion of matrices A and B.
    __shared__ real tile_a[TILE_DIM_Y][TILE_DIM_X];
    __shared__ real tile_b[TILE_DIM_Y][TILE_DIM_X];
    natural tx = threadIdx.x;
    natural ty = threadIdx.y;
    // row/col index of the current element of the result matrix that is computed by the thread.
    natural row = blockIdx.y * TILE_DIM_Y + ty;
    natural col = blockIdx.x * TILE_DIM_X + tx;
    // number of tile rows/columns needed to cover the matrices A and B
    natural range_x = CEIL(n, TILE_DIM_X);
    //  partial sum of the result matrix element computed by the thread
    real res = 0;

#pragma unroll
    // iterates over the tiles needed to compute the result matrix element
    for (natural i = 0; i < range_x; i++)
    {
        // check if the current thread is within the boundaries of A .
        if (row < m && i * TILE_DIM_X + tx < n)
            // load a portion of matrix A into the shared memory tiles.
            tile_a[ty][tx] = a[row * n + i * TILE_DIM_X + tx];
        else
            tile_a[ty][tx] = 0;
        // check if the current thread is within the boundaries of  B.
        if (col < p && i * TILE_DIM_Y + ty < n)
            // load a portion of matrix B into the shared memory tiles.
            tile_b[ty][tx] = b[(i * TILE_DIM_Y + ty) * p + col];
        else
            tile_b[ty][tx] = 0;
        // synchronizes all threads in the block before executing the next set of instructions.
        __syncthreads();
#pragma unroll
        // computes the partial sum of the result matrix element using the shared memory tiles
        for (natural j = 0; j < TILE_DIM_X; j++)
            res += tile_a[ty][j] * tile_b[j][tx];
        __syncthreads();
    }
    // stores the result of the partial sum in the result matrix if the thread is within the boundaries of the result matrix
    if (row < m && col < p)
        c[row * p + col] = res;
}

void Matmul::forward(bool training)
{
    timer_start(TMR_MATMUL_FW);
    // c->zero();
    dim3 n_blocks(CEIL(p, TILE_DIM_X), CEIL(m, TILE_DIM_Y));
    std::cout << "n_blocks: " << n_blocks.x << " " << n_blocks.y << std::endl;
    dim3 n_threads(TILE_DIM_X, TILE_DIM_Y);

    matmul_kernel<<<n_blocks, n_threads>>>(a->dev_data.get(), b->dev_data.get(), c->dev_data.get(), m, n, p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_MATMUL_FW);
}
*/

__global__ void matmul_kernel_forward(const real *a, const real *b, real *c, const natural m, const natural n, const natural p)
{
    // shared memory arrays that are used as tiles to store a portion of matrices A and B.
    __shared__ real tile_a[TILE_DIM][TILE_DIM];
    __shared__ real tile_b[TILE_DIM][TILE_DIM];
    natural tx = threadIdx.x;
    natural ty = threadIdx.y;
#pragma unroll
    for (natural row = blockIdx.y * TILE_DIM + ty; row < m; row += blockDim.y * gridDim.y)
    {
        natural col = blockIdx.x * TILE_DIM + tx;
        //  number of tile rows/columns needed to cover the matrices A and B
        natural range = CEIL(n, TILE_DIM);
        //  partial sum of the result matrix element computed by the thread
        real res = 0;

#pragma unroll
        // iterates over the tiles needed to compute the result matrix element
        for (natural i = 0; i < range; i++)
        {
            // check if the current thread is within the boundaries of A .
            if (i * TILE_DIM + tx < n)
                // load a portion of matrix A into the shared memory tiles.
                tile_a[ty][tx] = a[row * n + i * TILE_DIM + tx];
            else
                tile_a[ty][tx] = 0;
            // check if the current thread is within the boundaries of  B.
            if (col < p && i * TILE_DIM + ty < n)
                // load a portion of matrix B into the shared memory tiles.
                tile_b[ty][tx] = b[(i * TILE_DIM + ty) * p + col];

            else
                tile_b[ty][tx] = 0;
            // synchronizes all threads in the block before executing the next set of instructions.
            __syncthreads();
#pragma unroll
            // computes the partial sum of the result matrix element using the shared memory tiles
            for (natural j = 0; j < TILE_DIM; j++)
                res += tile_a[ty][j] * tile_b[j][tx];

            __syncthreads();
        }
        // stores the result of the partial sum in the result matrix if the thread is within the boundaries of the result matrix
        if (col < p)
            c[row * p + col] = res;
    }
}

void Matmul::forward(bool training)
{
    timer_start(TMR_MATMUL_FW);
    // c->zero();
    const natural n_blocks_y = std::min(CEIL(m, TILE_DIM), static_cast<natural>(N_BLOCKS));
    dim3 n_blocks(CEIL(p, TILE_DIM), n_blocks_y);
    // std::cout << "n_blocks: " << n_blocks.x << " " << n_blocks.y << std::endl;
    dim3 n_threads(TILE_DIM, TILE_DIM);

    matmul_kernel_forward<<<n_blocks, n_threads>>>(a->dev_data.get(), b->dev_data.get(), c->dev_data.get(), m, n, p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_MATMUL_FW);
}

// ##################################################################################

// ###### versione tutto in un kernel (lenta)
/*
__global__ void matmul_kernel_backward_1(real *a_grad, real *b_grad, const real *a_data, const real *b_data, const real *c_grad, const natural m, const natural n, const natural p)
{
    // shared memory arrays that are used as tiles to store a portion of matrices A and B.
    __shared__ real tile_c_grad[TILE_DIM][TILE_DIM];
    __shared__ real tile_b_data[TILE_DIM][TILE_DIM];
    natural tx = threadIdx.x;
    natural ty = threadIdx.y;
#pragma unroll
    for (natural row = blockIdx.y * TILE_DIM + ty; row < m; row += blockDim.y * gridDim.y)
    {
        natural col = blockIdx.x * TILE_DIM + tx;
        //  number of tile rows/columns needed to cover the matrices A and B
        natural range = CEIL(p, TILE_DIM);
        //  partial sum of the result matrix element computed by the thread
        real res = 0;

#pragma unroll
        // iterates over the tiles needed to compute the result matrix element
        for (natural i = 0; i < range; i++)
        {
            // check if the current thread is within the boundaries of C .
            if (i * TILE_DIM + tx < p)
                // load a portion of matrix A into the shared memory tiles.
                tile_c_grad[ty][tx] = c_grad[row * p + i * TILE_DIM + tx];
            else
                tile_c_grad[ty][tx] = 0;
            // check if the current thread is within the boundaries of  B.
            if (col < n && i * TILE_DIM + ty < p)
                // load a portion of matrix B into the shared memory tiles.
                // tile_b[ty][tx] = b[(i * TILE_DIM + ty) * p + col];
                tile_b_data[ty][tx] = b_data[col * p + i * TILE_DIM + ty];
            else
                tile_b_data[ty][tx] = 0;
            // synchronizes all threads in the block before executing the next set of instructions.
            __syncthreads();
#pragma unroll
            // computes the partial sum of the result matrix element using the shared memory tiles
            for (natural k = 0; k < TILE_DIM; k++)
            {
                // res += tile_c_grad[ty][j] * tile_b[tx][j];
                res += tile_c_grad[ty][k] * tile_b_data[k][tx];
                b_grad[col * p + k] += a_data[row * n + col] * tile_c_grad[ty][k];
            }
            __syncthreads();
        }
        // stores the result of the partial sum in the result matrix if the thread is within the boundaries of the result matrix
        if (col < n)
            a_grad[row * n + col] = res;
    }
}

void Matmul::backward()
{
    timer_start(TMR_MATMUL_BW);

    b->zero_grad();
    const natural n_blocks_y_1 = std::min(CEIL(m, TILE_DIM), static_cast<natural>(N_BLOCKS));
    dim3 n_blocks_1(CEIL(n, TILE_DIM), n_blocks_y_1);

    dim3 n_threads(TILE_DIM, TILE_DIM);

    dim3 n_blocks_2((p - 1) / TILE_DIM + 1, (n - 1) / TILE_DIM + 1, 1);

    // matmul_kernel_backward_1<<<n_blocks_1, n_threads>>>(a->dev_grad.get(), b->dev_data.get(), c->dev_grad.get(), m, n, p);
    matmul_kernel_backward_1<<<n_blocks_1, n_threads>>>(a->dev_grad.get(), b->dev_grad.get(), a->dev_data.get(), b->dev_data.get(), c->dev_grad.get(), m, n, p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // matmul_kernel_backward_2<<<n_blocks_1, n_threads>>>(b->dev_grad.get(), a->dev_data.get(), c->dev_grad.get(), m, n, p);
    // CHECK_CUDA_ERROR(cudaGetLastError());
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_MATMUL_BW);
}
*/

__global__ void matmul_kernel_backward_1(real *a, const real *b, const real *c, const natural m, const natural n, const natural p)
{
    // shared memory arrays that are used as tiles to store a portion of matrices A and B.
    __shared__ real tile_c[TILE_DIM][TILE_DIM];
    __shared__ real tile_b[TILE_DIM][TILE_DIM];
    natural tx = threadIdx.x;
    natural ty = threadIdx.y;
#pragma unroll
    for (natural row = blockIdx.y * TILE_DIM + ty; row < m; row += blockDim.y * gridDim.y)
    {
        natural col = blockIdx.x * TILE_DIM + tx;
        //  number of tile rows/columns needed to cover the matrices A and B
        natural range = CEIL(p, TILE_DIM);
        //  partial sum of the result matrix element computed by the thread
        real res = 0;

#pragma unroll
        // iterates over the tiles needed to compute the result matrix element
        for (natural i = 0; i < range; i++)
        {
            // check if the current thread is within the boundaries of C .
            if (i * TILE_DIM + tx < p)
                // load a portion of matrix A into the shared memory tiles.
                tile_c[ty][tx] = c[row * p + i * TILE_DIM + tx];
            else
                tile_c[ty][tx] = 0;
            // check if the current thread is within the boundaries of  B.
            if (col < n && i * TILE_DIM + ty < p)
                // load a portion of matrix B into the shared memory tiles.
                tile_b[ty][tx] = b[col * p + i * TILE_DIM + ty];
            else
                tile_b[ty][tx] = 0;
            // synchronizes all threads in the block before executing the next set of instructions.
            __syncthreads();
#pragma unroll
            // computes the partial sum of the result matrix element using the shared memory tiles
            for (natural k = 0; k < TILE_DIM; k++)
                res += tile_c[ty][k] * tile_b[k][tx];

            __syncthreads();
        }
        // stores the result of the partial sum in the result matrix if the thread is within the boundaries of the result matrix
        if (col < n)
            a[row * n + col] = res;
    }
}

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

void Matmul::backward()
{
    timer_start(TMR_MATMUL_BW);

    // b->zero_grad();
    // a->zero_grad();
    const natural n_blocks_y_1 = std::min(CEIL(m, TILE_DIM), static_cast<natural>(N_BLOCKS));
    dim3 n_blocks_1(CEIL(n, TILE_DIM), n_blocks_y_1);
    dim3 n_blocks_2(CEIL(p, TILE_DIM), CEIL(n, TILE_DIM));
    dim3 n_threads(TILE_DIM, TILE_DIM);
    matmul_kernel_backward_1<<<n_blocks_1, n_threads>>>(a->dev_grad.get(), b->dev_data.get(), c->dev_grad.get(), m, n, p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    matmul_kernel_backward_2<<<n_blocks_2, n_threads>>>(a->dev_data.get(), b->dev_grad.get(), c->dev_grad.get(), m, n, p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer_stop(TMR_MATMUL_BW);
}

// CROSSENTROPYLOSS
// ##################################################################################

CrossEntropyLoss::CrossEntropyLoss(shared_ptr<Variable> logits_, dev_shared_ptr<integer> dev_truth_, real *loss_, natural num_classes_) : logits(logits_), dev_truth(dev_truth_), loss(loss_), num_classes(num_classes_)
{
    dev_loss = dev_shared_ptr<real>(logits->size / num_classes); // N elements (number of nodes)
    dev_loss_res = dev_shared_ptr<real>(1);
}

// ##################################################################################

__global__ void cross_entropy_loss_kernel(real *dev_data, real *dev_grad, const integer *dev_truth, real *dev_loss, const natural num_classes, const natural num_nodes, const natural num_samples, bool training)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < num_nodes; i += blockDim.x * gridDim.x)
    {
        if (dev_truth[i] < 0)
            return;
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
        dev_loss[i] = logf(sum_exp) - logit[dev_truth[i]];
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
}

void CrossEntropyLoss::forward(bool training)
{

    timer_start(TMR_LOSS_FW);

    if (training)
        logits->zero_grad();

    dev_loss.set_zero();
    const natural DIM = logits->size / num_classes;
    // print_gpu<real>(dev_loss, DIM, DIM);
    // print_gpu<integer>(dev_truth, DIM, DIM);
    const natural n_blocks = std::min(CEIL(DIM, N_THREADS), static_cast<natural>(N_BLOCKS));
    cross_entropy_loss_kernel<<<n_blocks, N_THREADS>>>(logits->dev_data.get(), logits->dev_grad.get(), dev_truth.get(), dev_loss.get(), num_classes, DIM, num_samples, training);
    // cross_entropy_loss_kernel<<<DIM, 1>>>(logits->dev_data.get(), logits->dev_grad.get(), dev_truth.get(), dev_loss.get(), num_classes, num_samples, training);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // print_gpu<real>(dev_loss, DIM, DIM);

    // METODO 1 gpu

    dev_loss_res.set_zero();
    reduce<<<n_blocks, N_THREADS>>>(dev_loss.get(), dev_loss_res.get(), DIM);
    *loss = 0;
    dev_loss_res.copy_to_host(loss);
    *loss /= num_samples;

    // METODO 2 cpu
    /*
    std::vector<real> loss_cpu(DIM);
    dev_loss.copy_to_host(loss_cpu.data());
    // print_cpu(loss_cpu, DIM);
    *loss = std::accumulate(loss_cpu.begin(), loss_cpu.end(), static_cast<real>(0)) / num_samples;
    // std::cout << "loss: " << std::accumulate(loss_cpu.begin(), loss_cpu.end(), static_cast<real>(0)) / num_samples << std::endl;
*/
    // METODO 3 reduce_gpu()
    // *loss = reduce_gpu(dev_loss.get(), DIM) / num_samples;

    timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward()
{
}

void CrossEntropyLoss::set_num_samples(natural num_samples_)
{
    num_samples = num_samples_;
}

natural CrossEntropyLoss::get_num_samples() const
{
    return num_samples;
};
