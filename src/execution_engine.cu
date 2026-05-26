#include "execution_engine.h"
#include <iostream>

ExecutionEngine::ExecutionEngine(std::shared_ptr<MemoryAllocator> allocator, std::shared_ptr<KVCache> kv_cache)
    : allocator_(allocator), kv_cache_(kv_cache) 
{
    // Allocate a buffer for intermediate hidden states 
    // Max batch size * Max Seq Len * Dim (dummy allocation of 1 token batch for now)
    size_t hidden_size = 1 * 1 * dim_ * sizeof(float);
    hidden_states_ = std::make_shared<DeviceBuffer<float>>(hidden_size);
    std::cout << "Execution Engine Initialized. Intermediate buffer size: " << hidden_size << " bytes." << std::endl;
}

ExecutionEngine::~ExecutionEngine() {}

std::vector<float> ExecutionEngine::forward(const std::vector<int>& input_tokens) {
    int seq_len = input_tokens.size();
    std::cout << "\nStarting Forward Pass for " << seq_len << " tokens." << std::endl;

    // 1. Token Embeddings lookup (CPU to GPU for now, or just simulate)
    std::cout << "  -> Embedding Lookup" << std::endl;
    // In reality, we'd launch a CUDA kernel to pull vectors from allocator_->get_device_tensor("tok_embeddings.weight")
    // into hidden_states_

    // 2. Iterate through transformer layers
    for (int l = 0; l < num_layers_; ++l) {
        // std::cout << "  -> Layer " << l << " Forward" << std::endl;
        
        // a. RMSNorm (Attention)
        // b. Attention (Q, K, V Projections)
        
        // Update KV Cache pointers for this layer
        float* k_cache_ptr = kv_cache_->get_key_cache(l);
        float* v_cache_ptr = kv_cache_->get_value_cache(l);
        
        // c. Attention Math (RoPE, QK^T / sqrt(d), Softmax, *V)
        // d. RMSNorm (FFN)
        // e. FFN (Gate, Up, Down Projections)
    }

    std::cout << "  -> Processed " << num_layers_ << " Transformer Layers." << std::endl;

    // 3. Final RMSNorm
    std::cout << "  -> Final RMSNorm" << std::endl;

    // 4. Classifier (lm_head)
    std::cout << "  -> LM Head Projection" << std::endl;

    // Increment KV Cache Sequence Position
    kv_cache_->increment_seq_pos(seq_len);
    
    // Return dummy logits
    return std::vector<float>(vocab_size_, 0.0f);
}
