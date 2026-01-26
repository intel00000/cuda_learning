#pragma once
#include "common.h"

// V3: Multi-Output Tiling
// extends V2 by having each thread compute multiple outputs in the N dimension
template <typename T, typename ScaleT, typename AccT, int BLOCK_SIZE, int OUT_N>
__global__ void v3_multiout_kernel(int M, int N, int K, ScaleT alpha, ScaleT beta,
                                   const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    // Shared memory for the tiles
    __shared__ AccT As[BLOCK_SIZE][BLOCK_SIZE];         // store transposed
    __shared__ AccT Bs[BLOCK_SIZE * OUT_N][BLOCK_SIZE]; // store transposed, wider B tile
    // Thread local indices inside the block (0 to BLOCK_SIZE - 1)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // Global result coordinates (Column Major)
    const int row_base = blockIdx.x * BLOCK_SIZE + tx;
    const int col_base = blockIdx.y * (BLOCK_SIZE * OUT_N) + ty;
    // Thread-local accumulators in registers
    AccT acc[OUT_N] = {AccT(0)};

    // Same loading as before, but now load OUT_N columns of B per thread
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_SIZE)
    {
        // 1. Loading (Global → Shared)
        // Same as v2
        const int a_row = row_base;
        const int a_col = k_tile + ty;
        if (a_row < M && a_col < K)
            As[ty][tx] = (AccT)A[a_row + a_col * M];
        else
            As[ty][tx] = 0;

        // Load B tile, this time load OUT_N columns
        // i.e., we increase the size of shared memory used for B by OUT_N times
        const int b_row = k_tile + tx;
        for (int v = 0; v < OUT_N; ++v)
        {
            const int b_col = col_base + v * BLOCK_SIZE;
            if (b_row < K && b_col < N)
                Bs[ty + v * BLOCK_SIZE][tx] = (AccT)B[b_row + b_col * K];
            else
                Bs[ty + v * BLOCK_SIZE][tx] = 0;
        }
        // Summary of access patterns:
        // B: same as v2, b_col fixed, b_row varies → coalesced accesses
        // Bs: ty fixed within BLOCK_SIZE threads, tx varies → no bank conflicts
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            for (int v = 0; v < OUT_N; ++v)
                acc[v] += As[k][tx] * Bs[ty + v * BLOCK_SIZE][k];
        // Summary of access patterns:
        // As: k fixed, tx varies across threads → continuous addresses for consecutive threads
        // Bs: ty fixed within BLOCK_SIZE → Same address → Broadcast in group of BLOCK_SIZE threads
        __syncthreads();
    }

    // Write VEC_N outputs
    for (int v = 0; v < OUT_N; ++v)
    {
        const int o_col = col_base + v * BLOCK_SIZE;
        if (o_col < N && row_base < M)
        {
            const int idx = row_base + o_col * M;
            const AccT c_val = static_cast<AccT>(C[idx]);
            C[idx] = static_cast<T>(alpha * acc[v] + beta * c_val);
            // access to C → continuous addresses for consecutive threads within BLOCK_SIZE → coalesced
        }
    }
}

template <typename T, typename ScaleT, typename AccT>
void run_v3(int M, int N, int K, ScaleT alpha, ScaleT beta, const T *A, const T *B, T *C, cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 16;
    constexpr int OUT_N = 8;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + (BLOCK_SIZE * OUT_N) - 1) / (BLOCK_SIZE * OUT_N));
    v3_multiout_kernel<T, ScaleT, AccT, BLOCK_SIZE, OUT_N><<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template void run_v3<double, double, double>(int, int, int, double, double, const double *, const double *, double *, cudaStream_t);
template void run_v3<float, float, float>(int, int, int, float, float, const float *, const float *, float *, cudaStream_t);
template void run_v3<half, float, float>(int, int, int, float, float, const half *, const half *, half *, cudaStream_t);
template void run_v3<half, half, half>(int, int, int, half, half, const half *, const half *, half *, cudaStream_t);
