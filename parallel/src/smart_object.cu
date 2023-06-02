#include "../include/smart_object.cuh"

// ##################################################################################

smart_stream::smart_stream() : refCount(new size_t(1))
{
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaStreamCreate(&object));
#else
    cudaStreamCreate(&object);
#endif
}

// ##################################################################################

smart_stream::smart_stream(StreamPriority priority) : refCount(new size_t(1))
{
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&object, cudaStreamDefault, priority));
#else
    cudaStreamCreateWithPriority(&object, cudaStreamDefault, priority);
#endif
}

// ##################################################################################

smart_stream::smart_stream(const smart_stream &other) : object(other.object), refCount(other.refCount)
{
    IncrementRefCount();
}

// ##################################################################################

smart_stream &smart_stream::operator=(const smart_stream &other)
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

// ##################################################################################

smart_stream::~smart_stream()
{
    DecrementRefCount();
}

// ##################################################################################

cudaStream_t smart_stream::get() const
{
    return object;
}

// ##################################################################################

size_t smart_stream::getRefCount() const
{
    return *refCount;
}

// ##################################################################################

void smart_stream::IncrementRefCount()
{
    if (refCount)
        ++(*refCount);
}

// ##################################################################################

void smart_stream::DecrementRefCount()
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

// ##################################################################################

smart_event::smart_event() : refCount(new size_t(1))
{
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&object, cudaEventDisableTiming));
#else
    cudaEventCreateWithFlags(&object, cudaEventDisableTiming);
#endif
}

// ##################################################################################

smart_event::smart_event(const smart_event &other)
    : object(other.object), refCount(other.refCount)
{
    IncrementRefCount();
}

// ##################################################################################

smart_event &smart_event::operator=(const smart_event &other)
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

// ##################################################################################

smart_event::~smart_event()
{
    DecrementRefCount();
}

// ##################################################################################

cudaEvent_t smart_event::get() const
{
    return object;
}

// ##################################################################################

size_t smart_event::getRefCount() const
{
    return *refCount;
}

// ##################################################################################

void smart_event::IncrementRefCount()
{
    if (refCount)
        ++(*refCount);
}

// ##################################################################################

void smart_event::DecrementRefCount()
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

// ##################################################################################