#pragma once
#include "common.h"

// V2: Shared Memory Tiling (square tiles, each thread computes 1 output)
// Load tiles of A and B into shared memory and compute C tile-by-tile
template <typename T, typename ScaleT, typename AccT, int BLOCK_SIZE>
__global__ void v2_tiled_kernel(int M, int N, int K, ScaleT alpha, ScaleT beta,
                                const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    // Shared memory for the tiles
    __shared__ AccT As[BLOCK_SIZE][BLOCK_SIZE]; // store transposed
    __shared__ AccT Bs[BLOCK_SIZE][BLOCK_SIZE]; // store transposed
    // Thread local indices inside the block (0 to BLOCK_SIZE - 1)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // Global result coordinates (Column Major)
    const int row_base = blockIdx.x * BLOCK_SIZE + tx;
    const int col_base = blockIdx.y * BLOCK_SIZE + ty;
    // Thread-local accumulators (register file)
    AccT acc = 0;

    // Loop over the K-dimension in steps of BLOCK_SIZE
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_SIZE)
    {
        // 1. Loading (Global → Shared)
        // Load A tile: A_sub[ty][tx] = A[row, k_tile + ty] (stored transposed)
        // Guard against out-of-bounds if M or K are not multiples of BLOCK_SIZE
        const int a_row = row_base;
        const int a_col = k_tile + ty;
        if (a_row < M && a_col < K)
            // access to both A and As are coalesced
            // the key is to make consecutive threads access consecutive rows of A
            As[ty][tx] = (AccT)A[a_row + a_col * M];
        else
            As[ty][tx] = 0;

        // Load B tile: B_sub[ty][tx] = B[k_tile + tx, col] (stored transposed)
        const int b_row = k_tile + tx;
        const int b_col = col_base;
        if (b_row < K && b_col < N)
            Bs[ty][tx] = (AccT)B[b_row + b_col * K];
        else
            Bs[ty][tx] = 0;
        // Wait for all threads to finish loading
        __syncthreads();
        // Summary of access patterns:
        // A&B: col fixed, row varies → continuous addresses for consecutive threads → coalesced accesses
        // As&Bs: ty fixed within group of BLOCK_SIZE threads, tx varies → no bank conflicts
        // for global memory accesses, the goal is to coalesce accesses within a warp
        // for shared memory accesses, the goal is to avoid bank conflicts, stride is acceptable

        // 2. Compute on Shared Memory
        // accumulate partial results
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            acc += As[k][tx] * Bs[ty][k];
        // Wait for computation to finish before loading new tiles
        __syncthreads();
        // Summary of access patterns:
        // As: tx varies by 1 → continuous addresses for consecutive threads → No bank conflict
        // Bs: ty fixed within BLOCK_SIZE → Same address → Broadcast in group of BLOCK_SIZE threads
    }

    // 3. Write Result
    if (row_base < M && col_base < N)
    {
        const int idx = row_base + col_base * M;
        C[idx] = static_cast<T>(alpha * acc + beta * static_cast<AccT>(C[idx]));
        // access to C → continuous addresses for consecutive threads → coalesced
    }
}

template <typename T, typename ScaleT, typename AccT>
void run_v2(int M, int N, int K, ScaleT alpha, ScaleT beta, const T *A, const T *B, T *C, cudaStream_t stream)
{
    const int BLOCK_SIZE = 16;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaFuncSetCacheConfig(v2_tiled_kernel<T, ScaleT, AccT, BLOCK_SIZE>, cudaFuncCachePreferNone);
    v2_tiled_kernel<T, ScaleT, AccT, BLOCK_SIZE><<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template void run_v2<double, double, double>(int, int, int, double, double, const double *, const double *, double *, cudaStream_t);
template void run_v2<float, float, float>(int, int, int, float, float, const float *, const float *, float *, cudaStream_t);
template void run_v2<half, float, float>(int, int, int, float, float, const half *, const half *, half *, cudaStream_t);
template void run_v2<half, half, half>(int, int, int, half, half, const half *, const half *, half *, cudaStream_t);
