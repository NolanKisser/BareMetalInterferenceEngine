#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Launch configurations and math kernels

// 1. RMSNorm
void launch_rms_norm(float* out, float* in, float* weight, int size, cudaStream_t stream = 0);

// 2. Rotary Positional Embeddings (RoPE)
void launch_rope(float* q, float* k, int seq_pos, int head_dim, int num_heads, int num_kv_heads, cudaStream_t stream = 0);

// 3. Matrix Multiplication (GEMM) - block tiled
// C = A * B. A is [M x K], B is [K x N], C is [M x N]
void launch_gemm(float* C, float* A, float* B, int M, int N, int K, cudaStream_t stream = 0);

// 4. Softmax
void launch_softmax(float* x, int size, cudaStream_t stream = 0);

// 5. SwiGLU Activation (for FFN)
void launch_swiglu(float* out, float* gate, float* up, int size, cudaStream_t stream = 0);

#endif // KERNELS_CUH
