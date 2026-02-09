#pragma once
#include "common.h"
#include <mma.h>

using namespace nvcuda::wmma;

// V6: Tensor Core FP16 kernel with vectorized loads/stores (ONLY supports aligned sizes)
template <int BM, int BN, int BK, int WM, int WN, int SKEW_A, int SKEW_B, int WMMA_M, int WMMA_N, int WMMA_K, int VEC = VecSize<half>()>
__global__ void v6_wmma_kernel(int M, int N, int K, float alpha, float beta,
                               const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C)
{
    // Shared memory (padding reduces bank conflicts)
    __shared__ half As[BK][BM + SKEW_A]; // A tile stored as [k][m]
    __shared__ half Bs[BN][BK + SKEW_B]; // B tile stored as [n][k]

    // Thread/warp indexing
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    // Warp grid within block
    constexpr int warps_per_row = BN / WN;
    constexpr int warps_per_col = BM / WM;
    constexpr int num_warps = warps_per_row * warps_per_col;
    constexpr int num_threads = num_warps * 32;
    // Which TM×TN sub-tile of the block this warp computes
    const int warp_row = warpId / warps_per_row;
    const int warp_col = warpId % warps_per_row;
    // Global starting position of this block's output
    const int block_row_start = blockIdx.x * BM;
    const int block_col_start = blockIdx.y * BN;
    // Each warp computes WM×WN output tile using multiple WMMA operations
    constexpr int WMMA_TILES_M = WM / WMMA_M;
    constexpr int WMMA_TILES_N = WN / WMMA_N;
    // Accumulators fragments in registers
    // this is shared among the warp, each thread holds multiple fragments
    // hardware determines how fragments are mapped to threads in the warp, cannot be controlled by software
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WMMA_TILES_M][WMMA_TILES_N];
#pragma unroll
    for (int i = 0; i < WMMA_TILES_M; ++i)
#pragma unroll
        for (int j = 0; j < WMMA_TILES_N; ++j)
            fill_fragment(acc[i][j], 0.0f);

    // Vectorized loading parameters
    using VecT = Vec<half, VEC>;
    // A: BM rows × BK cols, load along M (rows), store transposed As[k][m]
    constexpr int A_vecs_per_col = BM / VEC;
    constexpr int A_total_vecs = A_vecs_per_col * BK;
    constexpr int A_vecs_per_thread = (A_total_vecs + num_threads - 1) / num_threads;
    // B: BK rows × BN cols, load along K (rows), store Bs[k][n]
    constexpr int B_vecs_per_col = BK / VEC;
    constexpr int B_total_vecs = B_vecs_per_col * BN;
    constexpr int B_vecs_per_thread = (B_total_vecs + num_threads - 1) / num_threads;

    // 1. Vectorized Loading (Global → Shared)
    auto load_tile = [&](int k_base)
    {
#pragma unroll
        for (int i = 0; i < A_vecs_per_thread; i++)
        {
            int vec_idx = tid + i * num_threads;
            if (vec_idx < A_total_vecs)
            {
                int vec_in_col = vec_idx % A_vecs_per_col; // along M
                int col = vec_idx / A_vecs_per_col;        // along K
                int row_base = vec_in_col * VEC;           // M base in tile
                int global_row = block_row_start + row_base;
                int global_col = k_base + col;
                *reinterpret_cast<VecT *>(&As[col][row_base]) = *reinterpret_cast<const VecT *>(&A[global_row + global_col * M]);
            }
        }
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
                *reinterpret_cast<VecT *>(&Bs[col][row_base]) = *reinterpret_cast<const VecT *>(&B[global_k + global_n * K]);
            }
        }
    };

    // 2. Compute using WMMA
    // For each k in the tile, load values into registers and compute
    // threads in a warp execute these together (warp-collective)
    auto compute_tile = [&]()
    {
        for (int k = 0; k < BK; k += WMMA_K)
        {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag[WMMA_TILES_M];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag[WMMA_TILES_N];
            // Load A fragments for this warp's M tiles
#pragma unroll
            for (int i = 0; i < WMMA_TILES_M; i++)
            {
                int a_row = warp_row * WM + i * WMMA_M;
                load_matrix_sync(a_frag[i], &As[k][a_row], BM + SKEW_A);
            }
            // Load B fragments, Bs[n][k] is row_major from B's perspective: K dimension is contiguous
#pragma unroll
            for (int j = 0; j < WMMA_TILES_N; j++)
            {
                int b_col = warp_col * WN + j * WMMA_N;
                load_matrix_sync(b_frag[j], &Bs[b_col][k], BK + SKEW_B);
            }
            // C += A × B (warp-collective operation)
#pragma unroll
            for (int i = 0; i < WMMA_TILES_M; i++)
#pragma unroll
                for (int j = 0; j < WMMA_TILES_N; j++)
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
    };
    // main loop
    for (int k_base = 0; k_base < K; k_base += BK)
    {
        load_tile(k_base);
        __syncthreads();
        compute_tile();
        __syncthreads();
    }
    // Epilogue: Store results to C with scaling
#pragma unroll
    for (int i = 0; i < WMMA_TILES_M; i++)
    {
#pragma unroll
        for (int j = 0; j < WMMA_TILES_N; j++)
        {
            int c_row = block_row_start + warp_row * WM + i * WMMA_M;
            int c_col = block_col_start + warp_col * WN + j * WMMA_N;
            half *C_ptr = &C[c_row + c_col * M];
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
            load_matrix_sync(c_frag, C_ptr, M, mem_col_major);
#pragma unroll
            for (int t = 0; t < c_frag.num_elements; ++t)
                c_frag.x[t] = __float2half(alpha * acc[i][j].x[t] + beta * __half2float(c_frag.x[t]));
            store_matrix_sync(C_ptr, c_frag, M, mem_col_major);
        }
    }
}

void run_v6(int M, int N, int K, float alpha, float beta, const half *A, const half *B, half *C, cudaStream_t stream)
{
    // Tile sizes
    constexpr int BM = 128;                              // Block output rows
    constexpr int BN = 128;                              // Block output cols
    constexpr int BK = 32;                               // K-tile size
    constexpr int WM = 64;                               // Warp output rows
    constexpr int WN = 64;                               // Warp output cols
    constexpr int SKEW_A = 8, SKEW_B = 8;                // Shared memory padding
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16; // WMMA tile sizes

    // Thread block dimensions
    constexpr int warps_per_block = (BM / WM) * (BN / WN);
    constexpr int threads = warps_per_block * 32;

    dim3 block(threads);
    dim3 grid(M / BM, N / BN);

    v6_wmma_kernel<BM, BN, BK, WM, WN, SKEW_A, SKEW_B, WMMA_M, WMMA_N, WMMA_K>
        <<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);

    CHECK_CUDA_ERROR(cudaGetLastError());
}
