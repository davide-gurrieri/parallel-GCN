#ifndef SHARED_PTR_CUH
#define SHARED_PTR_CUH
#include <cuda_runtime.h>
#include <vector>
#include "../include/utils.cuh"
#include "../include/smart_object.cuh"

template <typename T>
class dev_shared_ptr
{
public:
    // Default constructor
    dev_shared_ptr() : ptr(nullptr), refCount(nullptr) {}

    // Constructor that allocates device memory using cudaMalloc
    explicit dev_shared_ptr(size_t n_elements_)
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaMalloc(&ptr, n_elements_ * sizeof(T)));
#else
        cudaMalloc(&ptr, n_elements_ * sizeof(T));
#endif
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
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaMemcpy(ptr, source, n_elements * sizeof(T), cudaMemcpyHostToDevice));
#else
        cudaMemcpy(ptr, source, n_elements * sizeof(T), cudaMemcpyHostToDevice);
#endif
    }

    void copy_to_device_async(const T *source, smart_stream stream) const
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaMemcpyAsync(ptr, source, n_elements * sizeof(T), cudaMemcpyHostToDevice, stream.get()));
#else
        cudaMemcpyAsync(ptr, source, n_elements * sizeof(T), cudaMemcpyHostToDevice, stream.get());
#endif
    }

    void copy_to_host(T *destination) const
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaMemcpy(destination, ptr, n_elements * sizeof(T), cudaMemcpyDeviceToHost));
#else
        cudaMemcpy(destination, ptr, n_elements * sizeof(T), cudaMemcpyDeviceToHost);
#endif
    }

    void copy_to_host_async(T *destination, smart_stream stream) const
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaMemcpyAsync(destination, ptr, n_elements * sizeof(T), cudaMemcpyDeviceToHost, stream.get()));
#else
        cudaMemcpyAsync(destination, ptr, n_elements * sizeof(T), cudaMemcpyDeviceToHost, stream.get());
#endif
    }

    void set_zero(smart_stream stream) const
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaMemsetAsync(ptr, 0, n_elements * sizeof(T), stream.get()));
#else
        // cudaMemset(ptr, 0, n_elements * sizeof(T));
        cudaMemsetAsync(ptr, 0, n_elements * sizeof(T), stream.get());
#endif
        // cudaDeviceSynchronize();
    }

    int get_n_elements() const
    {
        return n_elements;
    }

    void print(unsigned col)
    {
        std::vector<T> host(n_elements);
        copy_to_host(host.data());
        for (int i = 0; i < n_elements; i++)
        {
            std::cout << host[i] << " ";
            if ((i + 1) % col == 0)
                std::cout << std::endl;
        }
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
#ifdef DEBUG_CUDA
                CHECK_CUDA_ERROR(cudaFree(ptr));
#else
                cudaFree(ptr);
#endif
                delete refCount;
            }
        }
    }
};

/*
template <typename T>
class pinned_host_ptr
{
public:
    // Default constructor
    pinned_host_ptr() : ptr(nullptr), refCount(nullptr) {}

    // Constructor that allocates device memory using cudaMalloc
    explicit pinned_host_ptr(size_t n_elements_)
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaMallocHost(&ptr, n_elements_ * sizeof(T)));
#else
        cudaMallocHost(&ptr, n_elements_ * sizeof(T));
#endif
        refCount = new int(1);
        n_elements = n_elements_;
    }

    // Copy constructor
    pinned_host_ptr(const pinned_host_ptr &other)
    {
        ptr = other.ptr;
        refCount = other.refCount;
        n_elements = other.n_elements;
        (*refCount)++;
    }

    // Move constructor
    pinned_host_ptr(pinned_host_ptr &&other) noexcept
    {
        ptr = other.ptr;
        refCount = other.refCount;
        n_elements = other.n_elements;
        other.ptr = nullptr;
        other.refCount = nullptr;
        other.n_elements = 0;
    }

    // Copy assignment operator
    pinned_host_ptr &operator=(const pinned_host_ptr &other)
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
    pinned_host_ptr &operator=(pinned_host_ptr &&other) noexcept
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
    ~pinned_host_ptr()
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

    void set_zero(smart_stream stream) const
    {
#ifdef DEBUG_CUDA
        CHECK_CUDA_ERROR(cudaMemsetAsync(ptr, 0, n_elements * sizeof(T), stream.get()));
#else
        cudaMemsetAsync(ptr, 0, n_elements * sizeof(T), stream.get());
#endif
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
#ifdef DEBUG_CUDA
                CHECK_CUDA_ERROR(cudaFreeHost(ptr));
#else
                cudaFreeHost(ptr);
#endif
                delete refCount;
            }
        }
    }
};
*/

#endif