#include "kernels.cuh"
#include <math.h>

// --- RMSNorm ---
__global__ void rms_norm_kernel(float* out, float* in, float* weight, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block reduction (sum of squares)
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    float val = (idx < size) ? in[idx] : 0.0f;
    float sq = val * val;

    // Atomic add for simplicity in this baseline implementation
    // A tree reduction would be much faster
    atomicAdd(&s_sum, sq);
    __syncthreads();

    if (idx < size) {
        float rms = rsqrtf((s_sum / size) + 1e-5f);
        out[idx] = val * rms * weight[idx];
    }
}

void launch_rms_norm(float* out, float* in, float* weight, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    rms_norm_kernel<<<blocks, threads, 0, stream>>>(out, in, weight, size);
}


// --- GEMM ---
#define TILE_SIZE 16
__global__ void gemm_kernel(float* C, float* A, float* B, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_gemm(float* C, float* A, float* B, int M, int N, int K, cudaStream_t stream) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel<<<blocks, threads, 0, stream>>>(C, A, B, M, N, K);
}


// --- Dummy Implementations for compilation ---
__global__ void dummy_rope(float* q, float* k, int seq_pos, int head_dim) {}
void launch_rope(float* q, float* k, int seq_pos, int head_dim, int num_heads, int num_kv_heads, cudaStream_t stream) {
    dummy_rope<<<1, 1, 0, stream>>>(q, k, seq_pos, head_dim);
}

__global__ void dummy_softmax(float* x, int size) {}
void launch_softmax(float* x, int size, cudaStream_t stream) {
    dummy_softmax<<<1, 256, 0, stream>>>(x, size);
}

__global__ void dummy_swiglu(float* out, float* gate, float* up, int size) {}
void launch_swiglu(float* out, float* gate, float* up, int size, cudaStream_t stream) {
    dummy_swiglu<<<1, 256, 0, stream>>>(out, gate, up, size);
}
