#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// RAII Wrapper for Device Memory
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t size) : size_(size), ptr_(nullptr) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            CUDA_CHECK(cudaFree(ptr_));
        }
    }

    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Enable move
    DeviceBuffer(DeviceBuffer&& other) noexcept : size_(other.size_), ptr_(other.ptr_) {
        other.size_ = 0;
        other.ptr_ = nullptr;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                CUDA_CHECK(cudaFree(ptr_));
            }
            size_ = other.size_;
            ptr_ = other.ptr_;
            other.size_ = 0;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    size_t size_;
    T* ptr_;
};

#endif // CUDA_UTILS_H
