#pragma once
#include "common.h"

// V4: Register Tiling (each thread computes TM x TN outputs)
// Block tile: BM x BN, K tile: BK, Thread tile: TM x TN
// Threads per block: (BM/TM) * (BN/TN)
template <typename T, typename ScaleT, typename AccT, int BM, int BN, int BK, int TM, int TN>
__global__ void v4_reg_tiling_kernel(int M, int N, int K, ScaleT alpha, ScaleT beta,
                                     const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    // Shared memory for the tiles
    __shared__ AccT As[BK][BM];     // transposed for higher l1 cache hit
    __shared__ AccT Bs[BK][BN + 1]; // ! worse performance if store transposed
    // 1D thread indexing
    constexpr int threads_per_row = BN / TN; // threads in x-direction
    constexpr int threads_per_col = BM / TM; // threads in y-direction
    constexpr int num_threads = threads_per_row * threads_per_col;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // Which TM×TN sub-tile of the block this thread computes
    const int thread_row = tid % threads_per_col; // 0, 1, ..., threads_per_col-1
    const int thread_col = tid / threads_per_col; // 0, 1, ..., threads_per_row-1
    // Global starting position of this block's output
    const int block_row_start = blockIdx.x * BM;
    const int block_col_start = blockIdx.y * BN;

    // Accumulators in registers (TM x TN per thread)
    AccT acc[TM][TN] = {{AccT(0)}};
    // Temporary registers for the outer product computation
    AccT a_reg[TM];
    AccT b_reg[TN];

    // How many elements each thread needs to load
    // We use a strided pattern: thread i loads elements i, i+num_threads, i+2*num_threads, ...
    constexpr int As_total = BM * BK;
    constexpr int Bs_total = BK * BN;
    // Ceiling division - some threads may load one extra element
    constexpr int As_per_thread = (As_total + num_threads - 1) / num_threads;
    constexpr int Bs_per_thread = (Bs_total + num_threads - 1) / num_threads;

    // Loop over K dimension in steps of BK
    for (int k_base = 0; k_base < K; k_base += BK)
    {
        // 1. Loading (Global → Shared)
        // Load A tile: A[block_row_start : block_row_start+BM, k_base : k_base+BK]
        // Into:        As[0:BM, 0:BK]
        //
        // Memory layout (column-major): A[row, col] = A[row + col*M]
        // We want coalesced access, so consecutive threads should read consecutive rows
        // (which are consecutive in memory for the same column)
        for (int i = 0; i < As_per_thread; i++)
        {
            int load_idx = tid + i * num_threads;
            if (load_idx < As_total)
            {
                // Convert linear index to 2D shared memory index
                // We iterate column-by-column (as_col changes slower) for coalescing
                int as_row = load_idx % BM;
                int as_col = load_idx / BM;

                // Global coordinates
                int global_row = block_row_start + as_row;
                int global_col = k_base + as_col;

                // Bounds check and load (store transposed)
                if (global_row < M && global_col < K)
                    As[as_col][as_row] = static_cast<AccT>(A[global_row + global_col * M]);
                else
                    As[as_col][as_row] = AccT(0);
            }
        }

        // Load B tile: B[k_base : k_base+BK, block_col_start : block_col_start+BN]
        // Into:        Bs[0:BK, 0:BN]
        //
        // Memory layout (column-major): B[row, col] = B[row + col*K]
        for (int i = 0; i < Bs_per_thread; i++)
        {
            int load_idx = tid + i * num_threads;
            if (load_idx < Bs_total)
            {
                // ! worse performance if iterate column-first
                int bs_col = load_idx % BN;
                int bs_row = load_idx / BN;

                // Global coordinates
                int global_row = k_base + bs_row;
                int global_col = block_col_start + bs_col;

                // Bounds check and load
                if (global_row < K && global_col < N)
                    Bs[bs_row][bs_col] = static_cast<AccT>(B[global_row + global_col * K]);
                else
                    Bs[bs_row][bs_col] = AccT(0);
            }
        }
        __syncthreads();

        // 2. Compute using registers (the outer product)
        // For each k in the tile, load values into registers and compute
#pragma unroll
        for (int k = 0; k < BK; k++)
        {
            // Load TM values from A into registers (transposed: As[k][m])
#pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = As[k][thread_row * TM + i];
#pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = Bs[k][thread_col * TN + j];
            // Outer product: TM × TN FMAs using values in registers
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    // 3. Write Results: Registers -> Global Memory
#pragma unroll
    for (int j = 0; j < TN; j++)
    {
        int global_col = block_col_start + thread_col * TN + j;
#pragma unroll
        for (int i = 0; i < TM; i++)
        {
            int global_row = block_row_start + thread_row * TM + i;
            if (global_row < M && global_col < N)
            {
                int idx = global_row + global_col * M;
                AccT c_val = static_cast<AccT>(C[idx]);
                C[idx] = static_cast<T>(static_cast<AccT>(alpha) * acc[i][j] + static_cast<AccT>(beta) * c_val);
            }
        }
    }
}

template <typename T, typename ScaleT, typename AccT>
void run_v4(int M, int N, int K, ScaleT alpha, ScaleT beta, const T *A, const T *B, T *C, cudaStream_t stream)
{
    // Tile sizes
    constexpr int BM = 128; // Block output rows
    constexpr int BN = 128; // Block output cols
    constexpr int BK = 16;  // K-tile size
    constexpr int TM = 8;   // Thread output rows
    constexpr int TN = 8;   // Thread output cols

    // Thread block dimensions
    constexpr int threads_x = BN / TN;
    constexpr int threads_y = BM / TM;
    dim3 block(threads_x, threads_y);

    // Grid dimensions
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    v4_reg_tiling_kernel<T, ScaleT, AccT, BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Template instantiations
template void run_v4<double, double, double>(int, int, int, double, double, const double *, const double *, double *, cudaStream_t);
template void run_v4<float, float, float>(int, int, int, float, float, const float *, const float *, float *, cudaStream_t);
template void run_v4<half, float, float>(int, int, int, float, float, const half *, const half *, half *, cudaStream_t);
template void run_v4<half, half, half>(int, int, int, half, half, const half *, const half *, half *, cudaStream_t);
