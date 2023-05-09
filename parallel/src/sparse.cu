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

    dev_indices = dev_shared_ptr<natural>(indices_size);
    dev_indptr = dev_shared_ptr<natural>(indptr_size);

    dev_indices.copy_to_device(sparse_index.indices.data());
    dev_indptr.copy_to_device(sparse_index.indptr.data());
}
