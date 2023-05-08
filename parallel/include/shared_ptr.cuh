#ifndef SHARED_PTR_CUH
#define SHARED_PTR_CUH
#include <cuda_runtime.h>
#include "../include/utils.cuh"

template <typename T>
class dev_shared_ptr
{
public:
    // Default constructor
    dev_shared_ptr() : ptr(nullptr), refCount(nullptr) {}

    // Constructor that takes a device pointer
    explicit dev_shared_ptr(T *p)
    {
        ptr = p;
        refCount = new int(1);
    }

    // Constructor that allocates device memory using cudaMalloc
    explicit dev_shared_ptr(size_t count)
    {
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed");
        }
        refCount = new int(1);
    }

    // Copy constructor
    dev_shared_ptr(const dev_shared_ptr &other)
    {
        ptr = other.ptr;
        refCount = other.refCount;
        (*refCount)++;
    }

    // Move constructor
    dev_shared_ptr(dev_shared_ptr &&other) noexcept
    {
        ptr = other.ptr;
        refCount = other.refCount;
        other.ptr = nullptr;
        other.refCount = nullptr;
    }

    // Copy assignment operator
    dev_shared_ptr &operator=(const dev_shared_ptr &other)
    {
        if (this != &other)
        {
            decrementRefCount();
            ptr = other.ptr;
            refCount = other.refCount;
            (*refCount)++;
        }
        return *this;
    }

    // Move assignment operator
    dev_shared_ptr &operator=(dev_shared_ptr &&other) noexcept
    {
        if (this != &other)
        {
            decrementRefCount();
            ptr = other.ptr;
            refCount = other.refCount;
            other.ptr = nullptr;
            other.refCount = nullptr;
        }
        return *this;
    }

    // Destructor
    ~dev_shared_ptr()
    {
        decrementRefCount();
    }

    // Get the device pointer
    T *get() const
    {
        return ptr;
    }

    // Check if the device pointer is null
    bool isNull() const
    {
        return ptr == nullptr;
    }

    // Get the number of shared references
    int useCount() const
    {
        return refCount != nullptr ? *refCount : 0;
    }

private:
    T *ptr;
    int *refCount;

    // Decrement the reference count and free the device memory if the count reaches zero
    void decrementRefCount()
    {
        if (refCount != nullptr)
        {
            (*refCount)--;
            if (*refCount == 0)
            {
                cudaFree(ptr);
                delete refCount;
            }
        }
    }
};

#endif