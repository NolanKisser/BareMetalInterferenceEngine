#include "server.h"
#include "tokenizer.h"
#include "weights.h"
#include "memory_allocator.h"
#include "kv_cache.h"
#include <memory>
#include <iostream>

int main(int argc, char** argv) {
    int port = 8080;
    std::cout << "--- BareMetal Inference API Server ---" << std::endl;

    // Phase 1/2 initialization
    auto allocator = std::make_shared<MemoryAllocator>();
    auto kv_cache = std::make_shared<KVCache>(1, 2048, 32, 8, 128);
    auto engine = std::make_shared<ExecutionEngine>(allocator, kv_cache);

    // Start Phase 4 Server
    InferenceServer server(engine, port);
    server.start();

    return 0;
}
