#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "cuda_utils.h"
#include <memory>
#include <vector>

class KVCache {
public:
    KVCache(int max_batch_size, int max_seq_len, int num_layers, int num_kv_heads, int head_dim);
    ~KVCache();

    // Returns a pointer to the start of the Key cache for a specific layer
    float* get_key_cache(int layer_idx) const;
    
    // Returns a pointer to the start of the Value cache for a specific layer
    float* get_value_cache(int layer_idx) const;

    // Advances the internal sequence pointer (e.g. after generating a token)
    void increment_seq_pos(int num_tokens = 1);
    
    // Reset cache pointers for a new generation request
    void reset();

    int get_current_seq_pos() const { return current_seq_pos_; }
    int get_max_seq_len() const { return max_seq_len_; }

private:
    int max_batch_size_;
    int max_seq_len_;
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;

    int current_seq_pos_;

    // We store KV caches contiguously in FP32 for simplicity in Phase 2
    // Shape per layer: [max_batch_size, num_kv_heads, max_seq_len, head_dim]
    std::vector<std::shared_ptr<DeviceBuffer<float>>> k_caches_;
    std::vector<std::shared_ptr<DeviceBuffer<float>>> v_caches_;
};

#endif // KV_CACHE_H
