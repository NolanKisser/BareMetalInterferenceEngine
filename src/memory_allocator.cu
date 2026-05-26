#include "memory_allocator.h"
#include <iostream>

MemoryAllocator::MemoryAllocator() {}

MemoryAllocator::~MemoryAllocator() {}

void MemoryAllocator::allocate_and_copy_weights(const std::vector<std::shared_ptr<Tensor>>& host_tensors) {
    size_t total_bytes = 0;

    for (const auto& tensor : host_tensors) {
        if (!tensor->data || tensor->size_bytes == 0) {
            std::cerr << "Warning: Tensor " << tensor->name << " has no data or size 0." << std::endl;
            continue;
        }

        // Allocate device memory via RAII wrapper
        auto d_buffer = std::make_shared<DeviceBuffer<char>>(tensor->size_bytes);
        
        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_buffer->get(), tensor->data, tensor->size_bytes, cudaMemcpyHostToDevice));

        // Store in our map
        device_weights_[tensor->name] = d_buffer;
        total_bytes += tensor->size_bytes;
        
        std::cout << "Mapped tensor " << tensor->name << " to GPU. (" << tensor->size_bytes << " bytes)" << std::endl;
    }

    std::cout << "Successfully allocated and copied " << total_bytes / (1024.0 * 1024.0) << " MB of weights to VRAM." << std::endl;
}

void* MemoryAllocator::get_device_tensor(const std::string& name) const {
    auto it = device_weights_.find(name);
    if (it != device_weights_.end()) {
        return it->second->get();
    }
    return nullptr;
}
