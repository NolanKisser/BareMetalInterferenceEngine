# BareMetal Inference Engine

A dependency-free, high-throughput LLM inference pipeline built entirely from scratch in C++ and CUDA. 

This project was designed as an educational, bare-metal approach to serving Large Language Models (like Llama-3) on NVIDIA GPUs without relying on massive Python ecosystems or heavy third-party C++ libraries (no PyTorch, no cuBLAS, no libTorch, no gRPC).

## Features

- **Custom BPE Tokenizer**: Reads vocabulary maps and performs Byte-Pair Encoding directly in C++.
- **Direct Weight Loading**: Maps raw binary tensor files straight into host memory.
- **VRAM Memory Management**: Custom RAII wrappers for `cudaMalloc` and device pointer lifecycle handling.
- **Static KV Caching**: Pre-allocates Keys and Values in VRAM based on the maximum sequence length to dramatically speed up token generation.
- **Custom CUDA Kernels**: Hand-written block-tiled `__global__` kernels for RMSNorm, RoPE, Matrix Multiplication (GEMM), and SwiGLU activations.
- **Dependency-Free API**: A bare-bones POSIX/Winsock server endpoint to accept inference requests.

## Project Structure

The project builds 3 distinct executable targets:

1. `inference_engine`: The core executable for testing tokenization and weight loading.
2. `benchmark`: A standalone tool that bypasses the network layer to stream 100 tokens as fast as possible, reporting raw `Tokens/Sec`.
3. `server`: A socket-based API endpoint that listens for incoming generation requests and routes them through the continuous batching/execution engine.

## Prerequisites

To compile and run this project, you must have the following installed and available in your system's `PATH`:

1.  **CMake** (version 3.18 or higher)
2.  **C++17 Compiler** (e.g., MSVC `cl.exe` for Windows, or `g++` for Linux)
3.  **NVIDIA CUDA Toolkit** (`nvcc` compiler)

> **Windows Users**: It is highly recommended to run the build commands from inside the **Developer Command Prompt for Visual Studio**, as this automatically configures the environment variables for `cl` and `cmake`.

## Building the Engine

Clone the repository and build using standard CMake commands:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

This will output the three executables into the `build/` (or `build/Release/`) directory.

## Testing and Usage

### 1. Run the Baseline Engine Test
This runs a dummy prompt through the Tokenizer and Execution engine to ensure memory allocates successfully and CUDA kernels launch without crashing.

```bash
./inference_engine
```

### 2. Run the Benchmark Tool
Use the benchmark tool to profile the speed of the custom CUDA kernels. It performs a 100-token generation loop.

```bash
./benchmark
```
*Tip: To analyze the memory bandwidth and SM occupancy of the custom block-tiled GEMM kernels, run this under NVIDIA Nsight Compute:*
`ncu --set full ./benchmark`

### 3. Start the API Server
Launch the lightweight socket server to listen for inference requests.

```bash
./server
```
Once running, the server will block and listen on port `8080`. You can simulate sending a payload to it, and the server will execute the forward pass and print the generation status.
