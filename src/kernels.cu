#include "kernels.cuh"
#include "cuda_utils.h"
#include <math.h>

// --- RMSNorm ---
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// A single block normalizes the whole vector so every thread sees the full
// sum of squares. Each thread accumulates a strided partial sum, then a
// warp-shuffle + shared-memory tree reduction produces the block total.
// Safe to run in place (out == in): each element is read and written by the
// same thread, and all reads of `in` finish before the final write loop.
__global__ void rms_norm_kernel(float* out, float* in, float* weight, int size) {
    __shared__ float s_warp_sums[32];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = in[i];
        sum += val * val;
    }

    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    sum = warp_reduce_sum(sum);
    if (lane == 0) s_warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < num_warps) ? s_warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) s_warp_sums[0] = sum;
    }
    __syncthreads();

    float rms = rsqrtf((s_warp_sums[0] / size) + 1e-5f);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out[i] = in[i] * rms * weight[i];
    }
}

void launch_rms_norm(float* out, float* in, float* weight, int size, cudaStream_t stream) {
    int threads = 1024;
    rms_norm_kernel<<<1, threads, 0, stream>>>(out, in, weight, size);
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaGetLastError());
}


// --- Dummy Implementations for compilation ---
__global__ void dummy_rope(float* q, float* k, int seq_pos, int head_dim) {}
void launch_rope(float* q, float* k, int seq_pos, int head_dim, int num_heads, int num_kv_heads, cudaStream_t stream) {
    dummy_rope<<<1, 1, 0, stream>>>(q, k, seq_pos, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void dummy_softmax(float* x, int size) {}
void launch_softmax(float* x, int size, cudaStream_t stream) {
    dummy_softmax<<<1, 256, 0, stream>>>(x, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void dummy_swiglu(float* out, float* gate, float* up, int size) {}
void launch_swiglu(float* out, float* gate, float* up, int size, cudaStream_t stream) {
    dummy_swiglu<<<1, 256, 0, stream>>>(out, gate, up, size);
    CUDA_CHECK(cudaGetLastError());
}
