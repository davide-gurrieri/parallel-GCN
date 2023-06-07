#ifndef RANDOM_CUH
#define RANDOM_CUH

#include "../include/shared_ptr.cuh"
#include "../include/utils.cuh"
#include <cstdint>
#include <random>
#define MY_RAND_MAX 0x7fffffff

struct RandState
{
    uint64_t a;
    uint64_t b;
};

__device__ real unif(RandState *state);

#endif