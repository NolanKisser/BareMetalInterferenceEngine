#include <iostream>
#include <chrono>
#include "tokenizer.h"
#include "weights.h"
#include "memory_allocator.h"
#include "kv_cache.h"
#include "execution_engine.h"

int main() {
    std::cout << "--- BareMetal Inference Engine Benchmark ---" << std::endl;
    
    auto allocator = std::make_shared<MemoryAllocator>();
    auto kv_cache = std::make_shared<KVCache>(1, 2048, 32, 8, 128);
    ExecutionEngine engine(allocator, kv_cache);

    std::vector<int> dummy_tokens(100, 1); // 100 tokens

    auto start_time = std::chrono::high_resolution_clock::now();

    // Run forward pass 100 times to simulate 100 token generations
    int generated_tokens = 100;
    for (int i = 0; i < generated_tokens; ++i) {
        engine.forward({dummy_tokens[i]});
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "\n--- Benchmark Results ---" << std::endl;
    std::cout << "Total Time: " << diff.count() << " s" << std::endl;
    std::cout << "Tokens Generated: " << generated_tokens << std::endl;
    std::cout << "Tokens / Sec: " << generated_tokens / diff.count() << " t/s" << std::endl;

    return 0;
}
