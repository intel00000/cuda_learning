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
    __shared__ AccT As[BK][BM];
    __shared__ AccT Bs[BN][BK + 1];
    // 1D thread indexing
    constexpr int threads_per_row = BN / TN; // threads in x-direction
    constexpr int threads_per_col = BM / TM; // threads in y-direction
    constexpr int num_threads = threads_per_row * threads_per_col;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // Which TM×TN sub-tile of the block this thread computes
    const int thread_row = tid % threads_per_col;
    const int thread_col = tid / threads_per_col;
    // Global starting position of this block's output
    const int block_row_start = blockIdx.x * BM;
    const int block_col_start = blockIdx.y * BN;

    // Accumulators in registers (TM x TN per thread)
    AccT acc[TM][TN] = {{AccT(0)}};
    // Temporary registers for the outer product computation
    AccT a_reg[TM];
    AccT b_reg[TN];

    // Advance pointers to the starting tile for this block
    const T *A_ptr = A + block_row_start;
    const T *B_ptr = B + block_col_start * K;
    T *C_ptr = C + block_row_start + (block_col_start + thread_col * TN) * M;

    // load coordinates within Shared memory tiles
    // A tile (BM x BK, column-major): stride over columns
    // Each thread loads one element per wave, waves stride over columns
    const int as_row = tid % BM;
    const int as_col = tid / BM;
    constexpr int as_stride = num_threads / BM;
    // B tile (BK x BN, column-major): stride over columns
    const int bs_row = tid % BK;
    const int bs_col = tid / BK;
    constexpr int bs_stride = num_threads / BK;

    // whether this is a edge block
    const bool is_edge_block = (block_row_start + BM > M) || (block_col_start + BN > N);
    const int K_aligned = K - (K % BK); // largest multiple of BK ≤ K

    if (!is_edge_block)
    {
        // fast path: all interior block, no need for bounds check when loading tiles
        for (int k_base = 0; k_base < K_aligned; k_base += BK)
        {
            // 1. Loading (Global → Shared)
            // Load A tile: A[block_row_start : block_row_start+BM, k_base : k_base+BK]
            // Into:        As[0:BM, 0:BK]
#pragma unroll
            for (int offset = 0; offset < BK; offset += as_stride)
                As[as_col + offset][as_row] = static_cast<AccT>(A_ptr[as_row + (as_col + offset) * M]);

            // Load B tile: B[k_base : k_base+BK, block_col_start : block_col_start+BN]
            // Into:        Bs[0:BK, 0:BN]
#pragma unroll
            for (int offset = 0; offset < BN; offset += bs_stride)
                Bs[bs_col + offset][bs_row] = static_cast<AccT>(B_ptr[bs_row + (bs_col + offset) * K]);
            __syncthreads();

            // advance A and B pointers to the next tile
            A_ptr += BK * M; // move down BK columns
            B_ptr += BK;     // move down BK rows

            // 2. Compute (the outer product)
#pragma unroll
            for (int k = 0; k < BK; k++)
            {
                // Load TM values from A into registers (transposed: As[k][m])
#pragma unroll
                for (int i = 0; i < TM; i++)
                    a_reg[i] = As[k][thread_row * TM + i];
#pragma unroll
                for (int j = 0; j < TN; j++)
                    b_reg[j] = Bs[thread_col * TN + j][k];
                // Outer product: TM × TN FMAs using values in registers
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++)
                        acc[i][j] += a_reg[i] * b_reg[j];
            }
            __syncthreads();
        }
        if (K_aligned < K) // Partial K-tile
        {
            const int k_remain = K - K_aligned;
#pragma unroll
            for (int offset = 0; offset < BK; offset += as_stride)
            {
                int col = as_col + offset;
                if (col < k_remain)
                    As[col][as_row] = static_cast<AccT>(A_ptr[as_row + col * M]);
                else
                    As[col][as_row] = AccT(0);
            }
#pragma unroll
            for (int offset = 0; offset < BN; offset += bs_stride)
            {
                if (bs_row < k_remain)
                    Bs[bs_col + offset][bs_row] = static_cast<AccT>(B_ptr[bs_row + (bs_col + offset) * K]);
                else
                    Bs[bs_col + offset][bs_row] = AccT(0);
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < BK; k++)
            {
#pragma unroll
                for (int i = 0; i < TM; i++)
                    a_reg[i] = As[k][thread_row * TM + i];
#pragma unroll
                for (int j = 0; j < TN; j++)
                    b_reg[j] = Bs[thread_col * TN + j][k];
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++)
                        acc[i][j] += a_reg[i] * b_reg[j];
            }
            __syncthreads();
        }

        // 3. Epilogue: Write results back to global memory
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int idxC = thread_row * TM + i;
                AccT c_val = static_cast<AccT>(C_ptr[idxC]);
                C_ptr[idxC] = static_cast<T>(static_cast<AccT>(alpha) * acc[i][j] + static_cast<AccT>(beta) * c_val);
            }
            C_ptr += M; // move to the next column of C
        }
    }
    else // slow path: full bounds checks
    {
        for (int k_base = 0; k_base < K; k_base += BK)
        {
#pragma unroll
            for (int offset = 0; offset < BK; offset += as_stride)
            {
                int global_row = block_row_start + as_row;
                int global_col = k_base + as_col + offset;

                if (global_row < M && global_col < K)
                    As[as_col + offset][as_row] = static_cast<AccT>(A_ptr[as_row + (as_col + offset) * M]);
                else
                    As[as_col + offset][as_row] = AccT(0);
            }
#pragma unroll
            for (int offset = 0; offset < BN; offset += bs_stride)
            {
                int global_row = k_base + bs_row;
                int global_col = block_col_start + bs_col + offset;
                if (global_row < K && global_col < N)
                    Bs[bs_col + offset][bs_row] = static_cast<AccT>(B_ptr[bs_row + (bs_col + offset) * K]);
                else
                    Bs[bs_col + offset][bs_row] = AccT(0);
            }
            __syncthreads();
            A_ptr += BK * M;
            B_ptr += BK;
#pragma unroll
            for (int k = 0; k < BK; k++)
            {
#pragma unroll
                for (int i = 0; i < TM; i++)
                    a_reg[i] = As[k][thread_row * TM + i];
#pragma unroll
                for (int j = 0; j < TN; j++)
                    b_reg[j] = Bs[thread_col * TN + j][k];
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++)
                        acc[i][j] += a_reg[i] * b_reg[j];
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
            int global_col = block_col_start + thread_col * TN + j;
            if (global_col < N)
            {
#pragma unroll
                for (int i = 0; i < TM; i++)
                {
                    int global_row = block_row_start + thread_row * TM + i;
                    if (global_row < M)
                    {
                        int row_offset = thread_row * TM + i;
                        AccT c_val = static_cast<AccT>(C_ptr[row_offset]);
                        C_ptr[row_offset] = static_cast<T>(
                            static_cast<AccT>(alpha) * acc[i][j] + static_cast<AccT>(beta) * c_val);
                    }
                }
            }
            C_ptr += M;
        }
    }
}

template <typename T, typename ScaleT, typename AccT>
void run_v4(int M, int N, int K, ScaleT alpha, ScaleT beta, const T *A, const T *B, T *C, cudaStream_t stream)
{
    // Tile sizes
    constexpr int BM = 128;                     // Block output rows
    constexpr int BN = 128;                     // Block output cols
    constexpr int BK = sizeof(T) < 8 ? 32 : 16; // K tile size
    constexpr int TM = 8;                       // Thread output rows
    constexpr int TN = 8;                       // Thread output cols

    constexpr int num_threads = (BM / TM) * (BN / TN);
    static_assert(num_threads % BM == 0, "num_threads must be divisible by BM for A load stride");
    static_assert(num_threads % BN == 0, "num_threads must be divisible by BN for B load stride");
    static_assert(BK % (num_threads / BM) == 0, "BK must be divisible by a_stride");
    static_assert(BK % (num_threads / BN) == 0, "BK must be divisible by b_stride");

    // Thread block dimensions
    dim3 block(BN / TN, BM / TM);
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
