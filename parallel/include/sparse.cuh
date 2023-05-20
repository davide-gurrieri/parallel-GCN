#ifndef SPARSE_CUH
#define SPARSE_CUH

#include "../include/utils.cuh"
#include "../include/shared_ptr.cuh"

#include <vector>

// ##################################################################################

class SparseIndex
{
public:
    std::vector<natural> indices;
    std::vector<natural> indptr;
    void print();
};

// ##################################################################################

class DevSparseIndex
{
public:
    dev_shared_ptr<natural> dev_indices;
    dev_shared_ptr<natural> dev_indptr;
    natural indices_size;
    natural indptr_size;
    DevSparseIndex(const SparseIndex &sparse_index);
};

// ##################################################################################

#endif