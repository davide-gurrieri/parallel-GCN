#ifndef SMART_OBJECT_CUH
#define SMART_OBJECT_CUH

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
    smart_stream() : refCount(new size_t(1))
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaStreamCreate(&object));
#else
        cudaStreamCreate(&object);
#endif
    }

    explicit smart_stream(StreamPriority priority) : refCount(new size_t(1))
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&object, cudaStreamDefault, priority));
#else
        cudaStreamCreateWithPriority(&object, cudaStreamDefault, priority);
#endif
    }

    smart_stream(const smart_stream &other)
        : object(other.object), refCount(other.refCount)
    {
        IncrementRefCount();
    }

    smart_stream &operator=(const smart_stream &other)
    {
        if (this != &other)
        {
            DecrementRefCount();
            object = other.object;
            refCount = other.refCount;
            IncrementRefCount();
        }
        return *this;
    }

    ~smart_stream()
    {
        DecrementRefCount();
    }

    cudaStream_t get() const
    {
        return object;
    }

    size_t getRefCount() const
    {
        return *refCount;
    }

private:
    cudaStream_t object;
    size_t *refCount;

    void IncrementRefCount()
    {
        if (refCount)
            ++(*refCount);
    }

    void DecrementRefCount()
    {
        if (refCount && --(*refCount) == 0)
        {
            delete refCount;
            if (object != nullptr)
#ifdef DEBUG_CUDA
                CHECK_CUDA_ERROR(cudaStreamDestroy(object));
#else
                cudaStreamDestroy(object);
#endif
        }
    }
};

class smart_event
{
public:
    smart_event() : refCount(new size_t(1))
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&object, cudaEventDisableTiming));
#else
        cudaEventCreateWithFlags(&object, cudaEventDisableTiming);
#endif
    }

    smart_event(const smart_event &other)
        : object(other.object), refCount(other.refCount)
    {
        IncrementRefCount();
    }

    smart_event &operator=(const smart_event &other)
    {
        if (this != &other)
        {
            DecrementRefCount();
            object = other.object;
            refCount = other.refCount;
            IncrementRefCount();
        }
        return *this;
    }

    ~smart_event()
    {
        DecrementRefCount();
    }

    cudaEvent_t get() const
    {
        return object;
    }

    size_t getRefCount() const
    {
        return *refCount;
    }

private:
    cudaEvent_t object;
    size_t *refCount;

    void IncrementRefCount()
    {
        if (refCount)
            ++(*refCount);
    }

    void DecrementRefCount()
    {
        if (refCount && --(*refCount) == 0)
        {
            delete refCount;
            if (object != nullptr)
#ifdef DEBUG_CUDA
                CHECK_CUDA_ERROR(cudaEventDestroy(object));
#else
                cudaEventDestroy(object);
#endif
        }
    }
};

// events[0] -> layer1_weight update
// events[1] -> layer2_weight update
// events[2] -> set_input start
// events[3] -> end training forward pass
// events[4] -> end first backward graphsum

#endif