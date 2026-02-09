#pragma once
#include "common.h"

// V5: Register Tiling with Vectorized Global Loads
// Block tile: BM x BN, K tile: BK, Thread tile: TM x TN
// Threads per block: (BM/TM) * (BN/TN)
template <typename T, typename ScaleT, typename AccT, int BM, int BN, int BK, int TM, int TN, int VEC = VecSize<T>()>
__global__ void v5_vec_load_kernel(int M, int N, int K, ScaleT alpha, ScaleT beta,
                                   const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    // Shared memory for the tiles
    // both stored transposed for auto-vectorized loads/stores
    __shared__ AccT As[BK][BM];
    __shared__ AccT Bs[BN][BK];
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
    // Accumulators in registers
    AccT acc[TM][TN] = {{AccT(0)}};
    AccT a_reg[TM];
    AccT b_reg[TN];

    // Advance pointers to the starting tile for this block
    const T *A_ptr = A + block_row_start;
    const T *B_ptr = B + block_col_start * K;
    T *C_ptr = C + block_row_start + (block_col_start + thread_col * TN) * M;

    // Vectorized loading parameters
    using VecT = Vec<T, VEC>;
    // A: load VEC rows at a time, stride over columns
    constexpr int A_vecs_per_col = BM / VEC;
    const int as_vec_row = tid % A_vecs_per_col; // which vector in column
    const int as_col = tid / A_vecs_per_col;     // which column
    constexpr int as_stride = num_threads / A_vecs_per_col;
    const int as_row_base = as_vec_row * VEC; // starting row index
    // B: load VEC k-rows at a time, stride over columns
    constexpr int B_vecs_per_col = BK / VEC;
    constexpr int bs_stride = num_threads / B_vecs_per_col;
    const int bs_vec_row = tid % B_vecs_per_col;
    const int bs_col = tid / B_vecs_per_col;
    const int bs_row_base = bs_vec_row * VEC;

    // whether this is a edge block
    const bool is_edge_block = (block_row_start + BM > M) || (block_col_start + BN > N);
    const int K_aligned = K - (K % BK); // largest multiple of BK ≤ K
    const bool a_aligned = vec_aligned<T, VEC>(A_ptr, as_row_base) && (M % VEC == 0);
    const bool b_aligned = vec_aligned<T, VEC>(B_ptr, bs_row_base) && (K % VEC == 0);

    if (!is_edge_block)
    { // interior block
        if (a_aligned && b_aligned)
        {
            // aligned addresses, vectorized loads
            for (int k_base = 0; k_base < K_aligned; k_base += BK)
            {
                // 1. Vectorized Load (Global → Shared)
                // Load A: Vectorized along M dimension (rows)
#pragma unroll
                for (int offset = 0; offset < BK; offset += as_stride)
                {
                    int col = as_col + offset;
                    VecT tmp = *reinterpret_cast<const VecT *>(&A_ptr[as_row_base + col * M]);
#pragma unroll
                    for (int v = 0; v < VEC; v++)
                        As[col][as_row_base + v] = static_cast<AccT>(tmp.data[v]);
                }
#pragma unroll
                for (int offset = 0; offset < BN; offset += bs_stride)
                {
                    int col = bs_col + offset;
                    VecT tmp = *reinterpret_cast<const VecT *>(&B_ptr[bs_row_base + col * K]);
#pragma unroll
                    for (int v = 0; v < VEC; v++)
                        Bs[col][bs_row_base + v] = static_cast<AccT>(tmp.data[v]);
                }
                __syncthreads();

                // advance A and B pointers to the next tile
                A_ptr += BK * M; // move down BK columns
                B_ptr += BK;     // move down BK rows

                // 2. Compute the outer product
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
            if (K_aligned < K) // Partial K-tile
            {
                const int k_remain = K - K_aligned;
#pragma unroll
                for (int offset = 0; offset < BK; offset += as_stride)
                {
                    int col = as_col + offset;
                    if (col < k_remain)
                    {
                        VecT tmp = *reinterpret_cast<const VecT *>(&A_ptr[as_row_base + col * M]);
#pragma unroll
                        for (int v = 0; v < VEC; v++)
                            As[col][as_row_base + v] = static_cast<AccT>(tmp.data[v]);
                    }
                    else
                    {
#pragma unroll
                        for (int v = 0; v < VEC; v++)
                            As[col][as_row_base + v] = AccT(0);
                    }
                }
#pragma unroll
                for (int offset = 0; offset < BN; offset += bs_stride)
                {
                    int col = bs_col + offset;
                    if (bs_row_base < k_remain)
                    {
                        VecT tmp = *reinterpret_cast<const VecT *>(&B_ptr[bs_row_base + col * K]);
#pragma unroll
                        for (int v = 0; v < VEC; v++)
                            Bs[col][bs_row_base + v] = static_cast<AccT>(tmp.data[v]);
                    }
                    else
                    {
#pragma unroll
                        for (int v = 0; v < VEC; v++)
                            Bs[col][bs_row_base + v] = AccT(0);
                    }
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
        }
        else
        {
            // Scalar path, same idea as v4_reg_tiling
            for (int k_base = 0; k_base < K_aligned; k_base += BK)
            {
#pragma unroll
                for (int offset = 0; offset < BK; offset += as_stride)
                {
                    int col = as_col + offset;
                    // because the as_stride is calculated based on VEC load we need to compensate here
                    // access pattern now stride VEC elements, worse than v4_reg_tiling
#pragma unroll
                    for (int v = 0; v < VEC; v++)
                        As[col][as_row_base + v] = static_cast<AccT>(A_ptr[(as_row_base + v) + col * M]);
                }
#pragma unroll
                for (int offset = 0; offset < BN; offset += bs_stride)
                {
                    int col = bs_col + offset;
#pragma unroll
                    for (int v = 0; v < VEC; v++)
                        Bs[col][bs_row_base + v] = static_cast<AccT>(B_ptr[(bs_row_base + v) + col * K]);
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
            if (K_aligned < K)
            {
                const int k_remain = K - K_aligned;
#pragma unroll
                for (int offset = 0; offset < BK; offset += as_stride)
                {
                    int col = as_col + offset;
#pragma unroll
                    for (int v = 0; v < VEC; v++)
                    {
                        if (col < k_remain)
                            As[col][as_row_base + v] = static_cast<AccT>(A_ptr[(as_row_base + v) + col * M]);
                        else
                            As[col][as_row_base + v] = AccT(0);
                    }
                }
#pragma unroll
                for (int offset = 0; offset < BN; offset += bs_stride)
                {
                    int col = bs_col + offset;
#pragma unroll
                    for (int v = 0; v < VEC; v++)
                    {
                        if (bs_row_base + v < k_remain)
                            Bs[col][bs_row_base + v] = static_cast<AccT>(B_ptr[(bs_row_base + v) + col * K]);
                        else
                            Bs[col][bs_row_base + v] = AccT(0);
                    }
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
        }
        // Epilogue: no M/N bounds checks since we guarantee interior block here
        // but we do have situations where C_ptr is not aligned
        const bool c_aligned = vec_aligned<T, VEC>(C_ptr, thread_row * TM) && (M % VEC == 0);
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
#pragma unroll
            for (int i = 0; i < TM; i += VEC)
            {
                int idxC = thread_row * TM + i;
                if (c_aligned)
                {
                    VecT c_vec = *reinterpret_cast<VecT *>(&C_ptr[idxC]);
#pragma unroll
                    for (int v = 0; v < VEC; v++)
                    {
                        AccT c_val = static_cast<AccT>(c_vec.data[v]);
                        c_vec.data[v] = static_cast<T>(static_cast<AccT>(alpha) * acc[i + v][j] + static_cast<AccT>(beta) * c_val);
                    }
                    *reinterpret_cast<VecT *>(&C_ptr[idxC]) = c_vec;
                }
                else
                {
#pragma unroll
                    for (int v = 0; v < VEC; v++)
                    {
                        AccT c_val = static_cast<AccT>(C_ptr[idxC + v]);
                        C_ptr[idxC + v] = static_cast<T>(static_cast<AccT>(alpha) * acc[i + v][j] + static_cast<AccT>(beta) * c_val);
                    }
                }
            }
            C_ptr += M;
        }
    }
    else // slow path: full bounds checks
    {
        for (int k_base = 0; k_base < K; k_base += BK)
        {
#pragma unroll
            for (int offset = 0; offset < BK; offset += as_stride)
            {
                int inner_col = as_col + offset;
                int global_col = k_base + inner_col;
#pragma unroll
                for (int v = 0; v < VEC; v++)
                {
                    int global_row = block_row_start + as_row_base + v;
                    if (global_row < M && global_col < K)
                        As[inner_col][as_row_base + v] = static_cast<AccT>(A_ptr[(as_row_base + v) + inner_col * M]);
                    else
                        As[inner_col][as_row_base + v] = AccT(0);
                }
            }
#pragma unroll
            for (int offset = 0; offset < BN; offset += bs_stride)
            {
                int inner_col = bs_col + offset;
                int global_col = block_col_start + inner_col;
#pragma unroll
                for (int v = 0; v < VEC; v++)
                {
                    int global_row = k_base + bs_row_base + v;
                    if (global_row < K && global_col < N)
                        Bs[inner_col][bs_row_base + v] = static_cast<AccT>(B_ptr[(bs_row_base + v) + inner_col * K]);
                    else
                        Bs[inner_col][bs_row_base + v] = AccT(0);
                }
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
                        C_ptr[row_offset] = static_cast<T>(static_cast<AccT>(alpha) * acc[i][j] + static_cast<AccT>(beta) * c_val);
                    }
                }
            }
            C_ptr += M;
        }
    }
}

template <typename T, typename ScaleT, typename AccT>
void run_v5(int M, int N, int K, ScaleT alpha, ScaleT beta, const T *A, const T *B, T *C, cudaStream_t stream)
{
    // Tile sizes
    constexpr int BM = 128;                     // Block output rows
    constexpr int BN = 128;                     // Block output cols
    constexpr int BK = sizeof(T) < 8 ? 32 : 16; // K tile size
    constexpr int TM = 8;                       // Thread output rows
    constexpr int TN = 8;                       // Thread output cols

    constexpr int num_threads = (BM / TM) * (BN / TN);
    constexpr int A_vecs_per_col = BM / VecSize<T>();
    constexpr int B_vecs_per_col = BK / VecSize<T>();
    static_assert(num_threads % A_vecs_per_col == 0, "num_threads must be divisible by A_vecs_per_col");
    static_assert(num_threads % B_vecs_per_col == 0, "num_threads must be divisible by B_vecs_per_col");
    static_assert(BK % (num_threads / A_vecs_per_col) == 0, "BK must be divisible by as_stride");
    static_assert(BN % (num_threads / B_vecs_per_col) == 0, "BN must be divisible by bs_stride");

    // Thread block dimensions
    dim3 block(BN / TN, BM / TM);
    // Grid dimensions
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    v5_vec_load_kernel<T, ScaleT, AccT, BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Template instantiations
template void run_v5<double, double, double>(int, int, int, double, double, const double *, const double *, double *, cudaStream_t);
template void run_v5<float, float, float>(int, int, int, float, float, const float *, const float *, float *, cudaStream_t);
template void run_v5<half, float, float>(int, int, int, float, float, const half *, const half *, half *, cudaStream_t);
