#include "../include/sparse.cuh"

void SparseIndex::print()
{
    std::cout << "---sparse index info--" << std::endl;

    std::cout << "indptr: ";
    for (auto i : indptr)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "indices: ";
    for (auto i : indices)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

// DevSparseIndex::DevSparseIndex() : dev_indices(nullptr), dev_indptr(nullptr), indices_size(0), indptr_size(0) {}

DevSparseIndex::DevSparseIndex(const SparseIndex &sparse_index)
{
    indices_size = sparse_index.indices.size();
    indptr_size = sparse_index.indptr.size();

    CHECK_CUDA_ERROR(cudaMalloc(&dev_indices, indices_size * sizeof(natural)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_indptr, indptr_size * sizeof(natural)));

    CHECK_CUDA_ERROR(cudaMemcpy(dev_indices, sparse_index.indices.data(), indices_size * sizeof(natural), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_indptr, sparse_index.indptr.data(), indptr_size * sizeof(natural), cudaMemcpyHostToDevice));
}

DevSparseIndex::~DevSparseIndex()
{
    if (dev_indices)
        CHECK_CUDA_ERROR(cudaFree(dev_indices));
    if (dev_indptr)
        CHECK_CUDA_ERROR(cudaFree(dev_indptr));
}
