#include "execution_engine.h"
#include "kernels.cuh"
#include <iostream>
#include <stdexcept>

ExecutionEngine::ExecutionEngine(std::shared_ptr<MemoryAllocator> allocator, std::shared_ptr<KVCache> kv_cache)
    : allocator_(allocator), kv_cache_(kv_cache) 
{
    // Allocate a buffer for intermediate hidden states 
    // Max batch size * Max Seq Len * Dim (dummy allocation of 1 token batch for now)
    // DeviceBuffer takes an element count, not bytes
    size_t hidden_elems = 1 * 1 * dim_;
    hidden_states_ = std::make_shared<DeviceBuffer<float>>(hidden_elems);
    ffn_out_ = std::make_shared<DeviceBuffer<float>>(hidden_elems);
    CUDA_CHECK(cudaMemset(hidden_states_->get(), 0, hidden_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(ffn_out_->get(), 0, hidden_elems * sizeof(float)));
    std::cout << "Execution Engine Initialized. Intermediate buffer size: " << hidden_elems * sizeof(float) << " bytes." << std::endl;
}

ExecutionEngine::~ExecutionEngine() {}

std::vector<float> ExecutionEngine::forward(const std::vector<int>& input_tokens) {
    int seq_len = input_tokens.size();
    std::cout << "\nStarting Forward Pass for " << seq_len << " tokens." << std::endl;

    // 1. Token Embeddings lookup (CPU to GPU for now, or just simulate)
    std::cout << "  -> Embedding Lookup" << std::endl;

    // 2. Iterate through transformer layers
    for (int l = 0; l < num_layers_; ++l) {
        
        // a. RMSNorm (Attention)
        float* d_weight_norm = (float*)allocator_->get_device_tensor("layers.0.attention_norm.weight");
        if (d_weight_norm) {
            launch_rms_norm(hidden_states_->get(), hidden_states_->get(), d_weight_norm, dim_);
        }
        
        // Update KV Cache pointers for this layer
        float* k_cache_ptr = kv_cache_->get_key_cache(l);
        float* v_cache_ptr = kv_cache_->get_value_cache(l);
        
        // c. Attention Math (RoPE, QK^T / sqrt(d), Softmax, *V)
        launch_rope(hidden_states_->get(), k_cache_ptr, kv_cache_->get_current_seq_pos(), 128, 32, 8);
        
        // e. FFN (Gate, Up, Down Projections)
        // Dummy GEMM launch
        float* d_gate = (float*)allocator_->get_device_tensor("layers.0.ffn.gate.weight");
        if (d_gate) {
            launch_gemm(ffn_out_->get(), hidden_states_->get(), d_gate, 1, dim_, dim_);
        }
    }

    if (verbose_) std::cout << "  -> Processed " << num_layers_ << " Transformer Layers." << std::endl;

    // 3. Final RMSNorm
    if (verbose_) std::cout << "  -> Final RMSNorm" << std::endl;

    // 4. Classifier (lm_head)
    if (verbose_) std::cout << "  -> LM Head Projection" << std::endl;

    // Increment KV Cache Sequence Position
    kv_cache_->increment_seq_pos(seq_len);
    
    // Return dummy logits
    return std::vector<float>(vocab_size_, 0.0f);
}
