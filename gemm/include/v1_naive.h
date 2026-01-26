#pragma once
#include "common.h"

// v1: The most naive GEMM kernel possible
// each thread computes one C element based on row and column indices
template <typename T, typename ScaleT, typename AccT>
__global__ void v1_naive_kernel(int M, int N, int K, const ScaleT alpha, const ScaleT beta,
                                const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    // Global index (Column Major: x is row, y is col)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        AccT acc = 0;
        // Naive accumulation loop
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            // Column Major Indexing: A[row, k] -> A[row + k*M], B[k, col] -> B[k + col*K]
            AccT A_val = static_cast<AccT>(A[row + k * M]); // contiguous for threads within a warp, coalesced access
            AccT B_val = static_cast<AccT>(B[k + col * K]); // same address, warp broadcast optimization
            acc += A_val * B_val;
        }
        // Apply Alpha and Beta
        AccT c_val = static_cast<AccT>(C[row + col * M]);
        // Write back result
        C[row + col * M] = static_cast<T>(alpha * acc + beta * c_val);
    }
}

// Generic runner template
// T: Data type (storage)
// ScaleT: Alpha/Beta type
// AccT: Accumulator/Math type
template <typename T, typename ScaleT, typename AccT>
void run_v1(int M, int N, int K, ScaleT alpha, ScaleT beta, const T *A, const T *B, T *C, cudaStream_t stream)
{
    int minGridSize, blockSize1D;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize1D, v1_naive_kernel<T, ScaleT, AccT>);

    // Factor 1D block size into 2D (prefer square-ish blocks)
    // Common factorizations: 1024->32x32, 512->16x32, 256->16x16, 128->8x16
    int blockX = 32;
    int blockY = blockSize1D / blockX;
    if (blockY < 1)
        blockY = 1;
    dim3 block(blockX, blockY);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaFuncSetCacheConfig(v1_naive_kernel<T, ScaleT, AccT>, cudaFuncCachePreferNone);
    v1_naive_kernel<T, ScaleT, AccT><<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Instantiate common template combinations
// FP64: double storage, double scale, double accum
template void run_v1<double, double, double>(int, int, int, double, double, const double *, const double *, double *, cudaStream_t);
// FP32: float storage, float scale, float accum
template void run_v1<float, float, float>(int, int, int, float, float, const float *, const float *, float *, cudaStream_t);
// FP16 (Mixed Precision): half storage, float scale, float accum
template void run_v1<half, float, float>(int, int, int, float, float, const half *, const half *, half *, cudaStream_t);
// FP16 (Pure - Optional): half storage, half scale, half accum
// Prone to overflow, just for comparisons
template void run_v1<half, half, half>(int, int, int, half, half, const half *, const half *, half *, cudaStream_t);
