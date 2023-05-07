#ifndef UTILS_CUH
#define UTILS_CUH
#include <iostream>

// Aliases
// using valueType = float;
// using indexType = size_t;

using natural = unsigned;
using integer = int;
using real = float;

// #define indexType int

// Variables
#define N_THREADS 1024
#define SEED 42

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

/*
 * Do the division M/N and approximate the result to the first greater integer
 */
#define CEIL(M, N) (((M) + (N)-1) / (N))

#endif