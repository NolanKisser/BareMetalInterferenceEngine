#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include "memory_allocator.h"
#include "kv_cache.h"
#include <vector>
#include <memory>

class ExecutionEngine {
public:
    ExecutionEngine(std::shared_ptr<MemoryAllocator> allocator, std::shared_ptr<KVCache> kv_cache);
    ~ExecutionEngine();

    // Perform a forward pass through the transformer
    // Input: A batch of token sequences
    // Output: Logits (unnormalized log probabilities) for the next token
    std::vector<float> forward(const std::vector<int>& input_tokens);

    // Toggle per-pass logging (disable when timing)
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    std::shared_ptr<MemoryAllocator> allocator_;
    std::shared_ptr<KVCache> kv_cache_;
    
    // Intermediate activation buffers (FP32)
    std::shared_ptr<DeviceBuffer<float>> hidden_states_;
    // Separate GEMM output so C never aliases the A input
    std::shared_ptr<DeviceBuffer<float>> ffn_out_;
    
    // Config parameters (hardcoded for dummy Llama-3 8B for now)
    int dim_ = 4096;
    int num_layers_ = 32;
    int vocab_size_ = 128256;

    bool verbose_ = true;
};

#endif // EXECUTION_ENGINE_H
