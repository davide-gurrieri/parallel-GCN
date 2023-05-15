#ifndef UTILS_CUH
#define UTILS_CUH
#include <iostream>
#include <cmath>
#include <curand_kernel.h>

// * for std::accumulate
#include <numeric>

// Aliases
// using valueType = float;
// using indexType = size_t;

using natural = unsigned;
using integer = int;
using real = float;
using randState = curandStatePhilox4_32_10_t;

// #define indexType int

// Variables
/*
namespace cudaParams
{
    natural N_THREADS = 512;
    natural N_THREADS
    natural N_BLOCKS = 128;
    natural TILE_DIM = 8;
}
*/

#define N_THREADS 1024
#define N_THREADS_DROPOUT 512
#define N_BLOCKS 128 // 8 * 16 with 16 number of SM (multiProcessorCount)
#define TILE_DIM 16
// #define TILE_DIM_Y 32 // 128
// #define TILE_DIM_X 32 // 8
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

#include "../include/shared_ptr.cuh"
template <class T>
void print_gpu(dev_shared_ptr<T> dev_data, natural size, natural col)
{

    T *data = new T[size];
    dev_data.copy_to_host(data);
    int count = 0;
    for (natural i = 0; i < size; i++)
    {
        // printf("%.4f ", data[i]);
        std::cout << data[i] << " ";
        count++;
        if (count % col == 0)
            std::cout << std::endl;
    }
    delete[] data;
}
#include <vector>
inline void print_cpu(const std::vector<real> &v, natural col)
{
    int count = 0;
    for (natural i = 0; i < v.size(); i++)
    {
        printf("%.4f ", v[i]);
        count++;
        if (count % col == 0)
            printf("\n");
    }
}

inline void print_gpu_info()
{
    int dev;
    cudaDeviceProp devProp;                 // C struct
    cudaGetDevice(&dev);                    // Get the id of the currently used device
    cudaGetDeviceProperties(&devProp, dev); // Get the device properties
    std::cout << "GPU INFORMATIONS:" << std::endl;
    std::cout << "multiProcessorCount: " << devProp.multiProcessorCount << std::endl;
    std::cout << "maxBlocksPerMultiProcessor: " << devProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "warpSize: " << devProp.warpSize << std::endl;
    std::cout << "sharedMemPerBlock [KB]: " << devProp.sharedMemPerBlock / 1024 << std::endl;
    std::cout << "sharedMemPerMultiprocessor [KB]: " << devProp.sharedMemPerMultiprocessor / 1024 << std::endl;
    std::cout << "number of floats in shared mem per block. " << devProp.sharedMemPerBlock / sizeof(float) << std::endl;
    std::cout << "totalGlobalMem [MB]: " << devProp.totalGlobalMem / 1048576 << std::endl;
    std::cout << std::endl;
}

#endif