#include <iostream>
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
    engine.set_verbose(false); // keep host I/O out of the timed region

    const int warmup_tokens = 10;
    const int generated_tokens = 100;
    std::vector<int> dummy_tokens(generated_tokens, 1);

    // Warm up: first launches pay for context setup and module loading
    for (int i = 0; i < warmup_tokens; ++i) {
        engine.forward({dummy_tokens[i]});
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    kv_cache->reset();

    // Time on the GPU timeline; kernel launches are async, so a host clock
    // without a sync would stop before the GPU finished the work
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < generated_tokens; ++i) {
        engine.forward({dummy_tokens[i]});
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double elapsed_s = elapsed_ms / 1000.0;
    std::cout << "\n--- Benchmark Results ---" << std::endl;
    std::cout << "Total Time: " << elapsed_s << " s" << std::endl;
    std::cout << "Tokens Generated: " << generated_tokens << std::endl;
    std::cout << "Latency / Token: " << elapsed_ms / generated_tokens << " ms" << std::endl;
    std::cout << "Tokens / Sec: " << generated_tokens / elapsed_s << " t/s" << std::endl;
    std::cout << "Note: no model weights are loaded yet, so this measures kernel launch "
                 "overhead only, not real inference throughput." << std::endl;

    return 0;
}
