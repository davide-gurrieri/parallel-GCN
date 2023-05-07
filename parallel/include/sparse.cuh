#ifndef SPARSE_CUH
#define SPARSE_CUH

#include <iostream>
#include <vector>

#include "../include/utils.cuh"

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

class DevSparseIndex
{
public:
    natural *dev_indices;
    natural *dev_indptr;
    natural indices_size;
    natural indptr_size;
    // DevSparseIndex();
    DevSparseIndex(const SparseIndex &sparse_index);
    ~DevSparseIndex();
};

#endif