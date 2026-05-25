cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(BareMetalInferenceEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CUDA
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(include)

# Find sources
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Executable
add_executable(inference_engine ${SOURCES})
