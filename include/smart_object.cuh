#ifndef SMART_OBJECT_CUH
#define SMART_OBJECT_CUH
#include <cuda_runtime.h>
#include "../include/utils.cuh"

/*
Code to see the priority range of streams
int minPriority, maxPriority;
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);
std::cout << "minPriority: " << minPriority << std::endl;
std::cout << "maxPriority: " << maxPriority << std::endl;
*/
enum StreamPriority
{
    Low = 0,
    High = -5
};

class smart_stream
{
public:
    smart_stream();
    explicit smart_stream(StreamPriority priority);
    smart_stream(const smart_stream &other);
    smart_stream &operator=(const smart_stream &other);
    ~smart_stream();
    cudaStream_t get() const;
    size_t getRefCount() const;

private:
    cudaStream_t object;
    size_t *refCount;
    void IncrementRefCount();
    void DecrementRefCount();
};

class smart_event
{
public:
    smart_event();
    smart_event(const smart_event &other);
    smart_event &operator=(const smart_event &other);
    ~smart_event();
    cudaEvent_t get() const;
    size_t getRefCount() const;

private:
    cudaEvent_t object;
    size_t *refCount;
    void IncrementRefCount();
    void DecrementRefCount();
};

// inline std::vector<smart_stream> streams;
// inline std::vector<smart_event> events;

// events[0] -> layer1_weight update
// events[1] -> layer2_weight update
// events[2] -> set_input start
// events[3] -> end training forward pass
// events[4] -> end first backward graphsum

#endif