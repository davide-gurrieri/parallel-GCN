#ifndef UTILS_CUH
#define UTILS_CUH
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

using natural = unsigned;
using integer = int;
using real = float;
using randState = curandState_t;
// using randState = curandStatePhilox4_32_10_t;

inline natural N_THREADS;
// constexpr natural N_THREADS_DROPOUT = 1024;
inline natural N_BLOCKS; // 8 * 16 with 16 number of SM (multiProcessorCount)
constexpr natural TILE_DIM = 16;
constexpr natural SEED = 42;

#ifdef DEBUG_CUDA

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
#endif

/*
 * Do the division M/N and approximate the result to the first greater integer
 */
#define CEIL(M, N) (((M) + (N)-1) / (N))

// Debugging functions
/*
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
*/

inline natural print_gpu_info()
{
  int dev;
  cudaDeviceProp devProp;                 // C struct
  cudaGetDevice(&dev);                    // Get the id of the currently used device
  cudaGetDeviceProperties(&devProp, dev); // Get the device properties
#ifndef TUNE
  std::cout << std::endl;
  std::cout << "GPU INFORMATIONS:" << std::endl;
  std::cout << "multiProcessorCount: " << devProp.multiProcessorCount
            << std::endl;
  std::cout << "maxBlocksPerMultiProcessor: "
            << devProp.maxBlocksPerMultiProcessor << std::endl;
  std::cout << "maxThreadsPerMultiProcessor: "
            << devProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock
            << std::endl;
  std::cout << "warpSize: " << devProp.warpSize << std::endl;
  std::cout << "sharedMemPerBlock [KB]: " << devProp.sharedMemPerBlock / 1024
            << std::endl;
  std::cout << "sharedMemPerMultiprocessor [KB]: "
            << devProp.sharedMemPerMultiprocessor / 1024 << std::endl;
  std::cout << "totalGlobalMem [MB]: " << devProp.totalGlobalMem / 1048576
            << std::endl;
  std::cout << std::endl;
#endif
  return devProp.multiProcessorCount;
}

#endif