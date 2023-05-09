#ifndef SPARSE_CUH
#define SPARSE_CUH

#include <vector>

#include "../include/utils.cuh"
#include "../include/shared_ptr.cuh"

// ##################################################################################
/*
 * Class to store parsed sparse matrices from the host.
 */
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
    // DevSparseIndex();
    DevSparseIndex(const SparseIndex &sparse_index);
};

// ##################################################################################

#endif