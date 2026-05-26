#include <iostream>
#include <memory>
#include "tokenizer.h"
#include "weights.h"
#include "memory_allocator.h"
#include "kv_cache.h"
#include "execution_engine.h"

int main(int argc, char** argv) {
    std::cout << "BareMetal Inference Engine - Phase 1 & 2" << std::endl;

    Tokenizer tokenizer;
    std::string vocab_path = "dummy_vocab.txt";
    tokenizer.load(vocab_path);

    std::string prompt = "Hello, BareMetal!";
    auto tokens = tokenizer.encode(prompt);
    
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Tokens: [ ";
    for (int t : tokens) { std::cout << t << " "; }
    std::cout << "]" << std::endl;

    ModelLoader loader;
    std::string model_path = "dummy_model.bin";
    loader.load_metadata(model_path);
    loader.load_weights(model_path);

    // --- Phase 2: Memory & Execution ---
    auto allocator = std::make_shared<MemoryAllocator>();
    allocator->allocate_and_copy_weights(loader.get_tensors());

    // Dummy Llama-3 8B params: 1 batch, 2048 seq_len, 32 layers, 8 kv_heads, 128 head_dim
    auto kv_cache = std::make_shared<KVCache>(1, 2048, 32, 8, 128);

    ExecutionEngine engine(allocator, kv_cache);
    
    // Perform a dummy forward pass
    engine.forward(tokens);

    std::cout << "\nEngine execution complete." << std::endl;

    return 0;
}
