cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(BareMetalInferenceEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CUDA
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(include)

# Find sources (excluding main executable entry points)
file(GLOB_RECURSE CORE_SOURCES 
    "src/tokenizer.cpp" 
    "src/weights.cpp" 
    "src/memory_allocator.cu" 
    "src/kv_cache.cu" 
    "src/kernels.cu" 
    "src/execution_engine.cu"
    "src/server.cpp"
)

# Executables
add_executable(inference_engine src/main.cpp ${CORE_SOURCES})
add_executable(benchmark src/benchmark.cpp ${CORE_SOURCES})
add_executable(server src/server_main.cpp ${CORE_SOURCES})


