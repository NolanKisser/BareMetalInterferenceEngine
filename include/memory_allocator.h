#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

#include "cuda_utils.h"
#include "weights.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

class MemoryAllocator {
public:
    MemoryAllocator();
    ~MemoryAllocator();

    // Map host tensors to device memory
    void allocate_and_copy_weights(const std::vector<std::shared_ptr<Tensor>>& host_tensors);

    // Retrieve device pointer by tensor name
    void* get_device_tensor(const std::string& name) const;

private:
    // Store device buffers mapped by tensor name
    std::unordered_map<std::string, std::shared_ptr<DeviceBuffer<char>>> device_weights_;
};

#endif // MEMORY_ALLOCATOR_H
