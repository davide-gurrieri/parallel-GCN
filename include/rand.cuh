#ifndef RANDOM_CUH
#define RANDOM_CUH

#include "../include/shared_ptr.cuh"
#include "../include/utils.cuh"
#include <cstdint>

struct RandState
{
    uint64_t a;
    uint64_t b;
};

__device__ real unif(RandState *state);

#endif