#include "../include/smart_object.cuh"

template <>
smart_object<cudaStream_t>::smart_object() : refCount(new size_t(1))
{
    cudaStreamCreate(&object);
}

template <>
smart_object<cudaEvent_t>::smart_object() : refCount(new size_t(1))
{
    cudaEventCreate(&object);
}

template <>
void smart_object<cudaStream_t>::DecrementRefCount()
{
    if (refCount && --(*refCount) == 0)
    {
        delete refCount;
        if (object != nullptr)
            cudaStreamDestroy(object);
    }
}

template <>
void smart_object<cudaEvent_t>::DecrementRefCount()
{
    if (refCount && --(*refCount) == 0)
    {
        delete refCount;
        if (object != nullptr)
            cudaEventDestroy(object);
    }
}