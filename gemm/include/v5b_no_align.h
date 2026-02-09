#pragma once
#include "common.h"

// V5b: Vectorized loads with no alignment check
// Block tile: BM x BN, K tile: BK, Thread tile: TM x TN
// Threads per block: (BM/TM) * (BN/TN)
template <typename T, typename ScaleT, typename AccT, int BM, int BN, int BK, int TM, int TN, int VEC = VecSize<T>()>
__global__ void v5b_no_align_kernel(int M, int N, int K, ScaleT alpha, ScaleT beta,
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
    const int thread_row = tid % threads_per_col; // 0, 1, ..., threads_per_col-1
    const int thread_col = tid / threads_per_col; // 0, 1, ..., threads_per_row-1
    // Global starting position of this block's output
    const int block_row_start = blockIdx.x * BM;
    const int block_col_start = blockIdx.y * BN;
    // Accumulators in registers
    AccT acc[TM][TN] = {{AccT(0)}};
    AccT a_reg[TM];
    AccT b_reg[TN];

    // Vectorized loading parameters
    using VecT = Vec<T, VEC>;
    // A: BM rows × BK cols, load along M (rows), store transposed As[k][m]
    constexpr int A_vecs_per_col = BM / VEC;
    constexpr int A_total_vecs = A_vecs_per_col * BK;
    constexpr int A_vecs_per_thread = (A_total_vecs + num_threads - 1) / num_threads;
    // B: BK rows × BN cols, load along K (rows), store Bs[k][n]
    constexpr int B_vecs_per_col = BK / VEC;
    constexpr int B_total_vecs = B_vecs_per_col * BN;
    constexpr int B_vecs_per_thread = (B_total_vecs + num_threads - 1) / num_threads;

    // Loop over K dimension in steps of BK
    for (int k_base = 0; k_base < K; k_base += BK)
    {
        // 1. Vectorized Loading (Global → Shared)
#pragma unroll
        for (int i = 0; i < A_vecs_per_thread; i++)
        {
            int vec_idx = tid + i * num_threads;
            if (vec_idx < A_total_vecs)
            {
                int vec_in_col = vec_idx % A_vecs_per_col;
                int col = vec_idx / A_vecs_per_col;
                int row_base = vec_in_col * VEC;
                int global_row = block_row_start + row_base;
                int global_col = k_base + col;
                // Vectorized load, cast global memory to vector type
                VecT src = *reinterpret_cast<const VecT *>(&A[global_row + global_col * M]);
#pragma unroll
                // Store transposed to shared memory, compiler autovectorizes
                for (int v = 0; v < VEC; v++)
                    As[col][row_base + v] = static_cast<AccT>(src.data[v]);
            }
        }
        // Load B: Vectorized along K dimension (rows)
#pragma unroll
        for (int i = 0; i < B_vecs_per_thread; i++)
        {
            int vec_idx = tid + i * num_threads;
            if (vec_idx < B_total_vecs)
            {
                int vec_in_col = vec_idx % B_vecs_per_col;
                int col = vec_idx / B_vecs_per_col;
                int row_base = vec_in_col * VEC;
                int global_k = k_base + row_base;
                int global_n = block_col_start + col;

                VecT src = *reinterpret_cast<const VecT *>(&B[global_k + global_n * K]);
#pragma unroll
                for (int v = 0; v < VEC; v++)
                    Bs[col][row_base + v] = static_cast<AccT>(src.data[v]);
            }
        }
        __syncthreads();

        // 2. Compute using registers (the outer product)
        // For each k in the tile, load values into registers and compute
#pragma unroll
        for (int k = 0; k < BK; k++)
        {
#pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = As[k][thread_row * TM + i]; // auto-vectorized
#pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = Bs[thread_col * TN + j][k]; // auto-vectorized
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    // 3. Write Results: Registers -> Global Memory
    const AccT a_scale = static_cast<AccT>(alpha);
    const AccT b_scale = static_cast<AccT>(beta);
#pragma unroll
    for (int j = 0; j < TN; j++)
    {
        int global_col = block_col_start + thread_col * TN + j;
#pragma unroll
        for (int i = 0; i < TM; i += VEC)
        {
            int global_row = block_row_start + thread_row * TM + i;
            int idxC = global_row + global_col * M;

            VecT c_vec = *reinterpret_cast<VecT *>(&C[idxC]);
#pragma unroll
            for (int v = 0; v < VEC; v++)
            {
                AccT c_val = static_cast<AccT>(c_vec.data[v]);
                c_vec.data[v] = static_cast<T>(a_scale * acc[i + v][j] + b_scale * c_val);
            }
            *reinterpret_cast<VecT *>(&C[idxC]) = c_vec;
        }
    }
}

template <typename T, typename ScaleT, typename AccT>
void run_v5b(int M, int N, int K, ScaleT alpha, ScaleT beta, const T *A, const T *B, T *C, cudaStream_t stream)
{
    // ban non-aligned sizes to avoid out-of-bounds vectorized accesses
    if (M % VecSize<T>() != 0 || K % VecSize<T>() != 0)
    {
        std::cerr << "Error: M and K must be multiples of " << VecSize<T>() << " for v5b_no_align." << std::endl;
        return;
    }

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
    v5b_no_align_kernel<T, ScaleT, AccT, BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template void run_v5b<double, double, double>(int, int, int, double, double, const double *, const double *, double *, cudaStream_t);
template void run_v5b<float, float, float>(int, int, int, float, float, const float *, const float *, float *, cudaStream_t);
template void run_v5b<half, float, float>(int, int, int, float, float, const half *, const half *, half *, cudaStream_t);
