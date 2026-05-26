#include "kv_cache.h"
#include <iostream>

KVCache::KVCache(int max_batch_size, int max_seq_len, int num_layers, int num_kv_heads, int head_dim) 
    : max_batch_size_(max_batch_size), 
      max_seq_len_(max_seq_len), 
      num_layers_(num_layers), 
      num_kv_heads_(num_kv_heads), 
      head_dim_(head_dim),
      current_seq_pos_(0) 
{
    size_t elements_per_layer = (size_t)max_batch_size * num_kv_heads * max_seq_len * head_dim;
    size_t bytes_per_layer = elements_per_layer * sizeof(float);

    for (int i = 0; i < num_layers; ++i) {
        k_caches_.push_back(std::make_shared<DeviceBuffer<float>>(elements_per_layer));
        v_caches_.push_back(std::make_shared<DeviceBuffer<float>>(elements_per_layer));
    }

    std::cout << "KV Cache Initialized." << std::endl;
    std::cout << "  Layers: " << num_layers << std::endl;
    std::cout << "  Max Seq Len: " << max_seq_len << std::endl;
    std::cout << "  Total VRAM Allocated: " << (bytes_per_layer * num_layers * 2) / (1024.0 * 1024.0) << " MB" << std::endl;
}

KVCache::~KVCache() {}

float* KVCache::get_key_cache(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_layers_) return nullptr;
    return k_caches_[layer_idx]->get();
}

float* KVCache::get_value_cache(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_layers_) return nullptr;
    return v_caches_[layer_idx]->get();
}

void KVCache::increment_seq_pos(int num_tokens) {
    current_seq_pos_ += num_tokens;
    if (current_seq_pos_ > max_seq_len_) {
        std::cerr << "Warning: KV Cache exceeded maximum sequence length!" << std::endl;
    }
}

void KVCache::reset() {
    current_seq_pos_ = 0;
}
