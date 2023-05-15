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

    // Constructor that allocates device memory using cudaMalloc
    explicit dev_shared_ptr(size_t n_elements_)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&ptr, n_elements_ * sizeof(T)));
        refCount = new int(1);
        n_elements = n_elements_;
    }

    // Copy constructor
    dev_shared_ptr(const dev_shared_ptr &other)
    {
        ptr = other.ptr;
        refCount = other.refCount;
        n_elements = other.n_elements;
        (*refCount)++;
    }

    // Move constructor
    dev_shared_ptr(dev_shared_ptr &&other) noexcept
    {
        ptr = other.ptr;
        refCount = other.refCount;
        n_elements = other.n_elements;
        other.ptr = nullptr;
        other.refCount = nullptr;
        other.n_elements = 0;
    }

    // Copy assignment operator
    dev_shared_ptr &operator=(const dev_shared_ptr &other)
    {
        if (this != &other)
        {
            decrementRefCount();
            ptr = other.ptr;
            refCount = other.refCount;
            n_elements = other.n_elements;
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
            n_elements = other.n_elements;
            other.ptr = nullptr;
            other.refCount = nullptr;
            other.n_elements = 0;
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

    // Dereferencing operator
    T &operator*() const
    {
        return *ptr;
    }

    void copy_to_device(const T *source) const
    {
        CHECK_CUDA_ERROR(cudaMemcpy(ptr, source, n_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T *destination) const
    {
        CHECK_CUDA_ERROR(cudaMemcpy(destination, ptr, n_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void set_zero() const
    {
        CHECK_CUDA_ERROR(cudaMemset(ptr, 0, n_elements * sizeof(T)));
    }

    int get_n_elements() const
    {
        return n_elements;
    }

private:
    T *ptr;
    int *refCount;
    int n_elements;

    // Decrement the reference count and free the device memory if the count reaches zero
    void decrementRefCount()
    {
        if (refCount != nullptr)
        {
            (*refCount)--;
            if (*refCount == 0)
            {
                CHECK_CUDA_ERROR(cudaFree(ptr));
                delete refCount;
            }
        }
    }
};

#endif