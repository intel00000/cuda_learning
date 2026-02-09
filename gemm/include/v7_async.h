#pragma once
#include "common.h"
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda::wmma;

// Type traits for WMMA configurations
template <typename T>
struct WmmaConfig;
// FP16: 16×16×16
template <>
struct WmmaConfig<half>
{
    static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    static constexpr int BM = 128, BN = 128, BK = 32;
    static constexpr int WM = 64, WN = 64;
    static constexpr int SKEW_A = 8, SKEW_B = 8;
    using InputType = half;
    using AccumType = float;
};
// TF32: 16×16×8
template <>
struct WmmaConfig<float>
{
    static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
    static constexpr int BM = 128, BN = 128, BK = 16;
    static constexpr int WM = 64, WN = 64;
    static constexpr int SKEW_A = 4, SKEW_B = 4;
    using InputType = precision::tf32;
    using AccumType = float;
};
// FP64: 8×8×4
template <>
struct WmmaConfig<double>
{
    static constexpr int WMMA_M = 8, WMMA_N = 8, WMMA_K = 4;
    static constexpr int BM = 64, BN = 64, BK = 16;
    static constexpr int WM = 32, WN = 32;
    static constexpr int SKEW_A = 2, SKEW_B = 2;
    using InputType = double;
    using AccumType = double;
};

// V7: Tensor Core GEMM with async copy and double buffering (Aligned ONLY, no boundary check)
template <typename T, typename ScaleT, typename AccT, int BM, int BN, int BK, int WM, int WN,
          int SKEW_A, int SKEW_B, int WMMA_M, int WMMA_N, int WMMA_K, int COPY_SIZE = 16>
__global__ void v7_async_aligned_kernel(int M, int N, int K, ScaleT alpha, ScaleT beta,
                                        const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    using Config = WmmaConfig<T>;
    using InputType = typename Config::InputType;
    using AccumType = typename Config::AccumType;

    // Double-buffered shared memory with padding
    __shared__ T As[2][BK][BM + SKEW_A];
    __shared__ T Bs[2][BN][BK + SKEW_B];

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
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, AccumType> acc_frag[WMMA_TILES_M][WMMA_TILES_N];
#pragma unroll
    for (int i = 0; i < WMMA_TILES_M; i++)
#pragma unroll
        for (int j = 0; j < WMMA_TILES_N; j++)
            fill_fragment(acc_frag[i][j], AccumType(0));

    // Vectorized loading parameters
    constexpr int elems_per_copy = COPY_SIZE / sizeof(T);
    // A: BM rows × BK cols, load along M (rows), store transposed As[k][m]
    constexpr int A_elems_per_col = BM / elems_per_copy;
    constexpr int A_total_copies = A_elems_per_col * BK;
    constexpr int A_copies_per_thread = (A_total_copies + num_threads - 1) / num_threads;
    // B: BK rows × BN cols, load along K (rows), store Bs[k][n]
    constexpr int B_elems_per_col = BK / elems_per_copy;
    constexpr int B_total_copies = B_elems_per_col * BN;
    constexpr int B_copies_per_thread = (B_total_copies + num_threads - 1) / num_threads;

    // 1. Async Loading (Global → Shared)
    auto load_tile = [&](int buf, int k_base)
    {
    // Load A tile
#pragma unroll
        for (int i = 0; i < A_copies_per_thread; i++)
        {
            int idx = tid + i * num_threads;
            if (idx >= A_total_copies)
                continue;

            int copies_in_col = idx % A_elems_per_col;
            int col = idx / A_elems_per_col;               // k index in tile
            int row_base = copies_in_col * elems_per_copy; // m index base in tile

            int global_row = block_row_start + row_base;
            int global_col = k_base + col;

            __pipeline_memcpy_async(&As[buf][col][row_base], &A[global_row + global_col * M], COPY_SIZE);
        }
        // Load B tile
#pragma unroll
        for (int i = 0; i < B_copies_per_thread; i++)
        {
            int idx = tid + i * num_threads;
            if (idx < B_total_copies)
            {
                int copies_in_col = idx % B_elems_per_col;
                int col = idx / B_elems_per_col;
                int row_base = copies_in_col * elems_per_copy;
                int global_k = k_base + row_base;
                int global_n = block_col_start + col;
                __pipeline_memcpy_async(&Bs[buf][col][row_base], &B[global_k + global_n * K], COPY_SIZE);
            }
        }
        __pipeline_commit();
    };
    // 2. Compute using WMMA
    // For each k in the tile, load values into registers and compute
    // threads in a warp execute these together (warp-collective)
    auto compute_tile = [&](int buf)
    {
#pragma unroll
        for (int k = 0; k < BK; k += WMMA_K)
        {
            // Load A fragments
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, InputType, col_major> a_frag[WMMA_TILES_M];
#pragma unroll
            for (int i = 0; i < WMMA_TILES_M; i++)
            {
                int a_row = warp_row * WM + i * WMMA_M;
                load_matrix_sync(a_frag[i], &As[buf][k][a_row], BM + SKEW_A);
            }
            // Load B fragments
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, InputType, col_major> b_frag[WMMA_TILES_N];
#pragma unroll
            for (int j = 0; j < WMMA_TILES_N; j++)
            {
                int b_col = warp_col * WN + j * WMMA_N;
                load_matrix_sync(b_frag[j], &Bs[buf][b_col][k], BK + SKEW_B);
            }
            // C += A × B (warp-collective operation)
#pragma unroll
            for (int i = 0; i < WMMA_TILES_M; i++)
#pragma unroll
                for (int j = 0; j < WMMA_TILES_N; j++)
                    mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
        }
    };

    // main loop
    int num_tiles = K / BK;
    int buf = 0;
    // Initial load
    load_tile(0, 0);
    __pipeline_wait_prior(0);
    __syncthreads();
    // Loop over K tiles
    for (int t = 0; t < num_tiles - 1; t++)
    {
        // flip buffer
        int next_buf = 1 - buf;
        // Async load next tile into alternate buffer
        load_tile(next_buf, (t + 1) * BK);
        // Compute current tile
        compute_tile(buf);
        // wait for async loads to complete
        __pipeline_wait_prior(0);
        __syncthreads();
        buf = next_buf;
    }
    // Final tile
    compute_tile(buf);

    // write results: Registers -> Global Memory
    AccT alpha_acc = static_cast<AccT>(alpha);
    AccT beta_acc = static_cast<AccT>(beta);
#pragma unroll
    for (int i = 0; i < WMMA_TILES_M; i++)
    {
#pragma unroll
        for (int j = 0; j < WMMA_TILES_N; j++)
        {
            int c_row = block_row_start + warp_row * WM + i * WMMA_M;
            int c_col = block_col_start + warp_col * WN + j * WMMA_N;

            // bounds check if within size of C
            if (c_row + WMMA_M <= M && c_col + WMMA_N <= N)
            {
                // Load existing C matrix
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, T> c_frag;
                load_matrix_sync(c_frag, &C[c_row + c_col * M], M, mem_col_major);
                // Scale: C = alpha * acc + beta * C
#pragma unroll
                for (int t = 0; t < c_frag.num_elements; t++)
                {
                    AccT acc_val = static_cast<AccT>(acc_frag[i][j].x[t]);
                    AccT c_val = static_cast<AccT>(c_frag.x[t]);
                    c_frag.x[t] = static_cast<T>(alpha_acc * acc_val + beta_acc * c_val);
                }
                store_matrix_sync(&C[c_row + c_col * M], c_frag, M, mem_col_major);
            }
        }
    }
}

template <typename T, typename ScaleT, typename AccT>
void run_v7(int M, int N, int K, ScaleT alpha, ScaleT beta, const T *A, const T *B, T *C, cudaStream_t stream)
{
    using Config = WmmaConfig<T>;
    constexpr int BM = Config::BM;         // Block output rows
    constexpr int BN = Config::BN;         // Block output cols
    constexpr int BK = Config::BK;         // K-tile size
    constexpr int WM = Config::WM;         // Warp output rows
    constexpr int WN = Config::WN;         // Warp output cols
    constexpr int SKEW_A = Config::SKEW_A; // Shared memory padding A
    constexpr int SKEW_B = Config::SKEW_B; // Shared memory padding B
    constexpr int WMMA_M = Config::WMMA_M; // WMMA output rows
    constexpr int WMMA_N = Config::WMMA_N; // WMMA output cols
    constexpr int WMMA_K = Config::WMMA_K; // WMMA K size

    // Thread block dimensions
    constexpr int warps_per_block = (BM / WM) * (BN / WN);
    constexpr int threads = warps_per_block * 32;
    dim3 block(threads);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    v7_async_aligned_kernel<T, ScaleT, AccT, BM, BN, BK, WM, WN, SKEW_A, SKEW_B, WMMA_M, WMMA_N, WMMA_K>
        <<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Template instantiations
template void run_v7<half, float, float>(int, int, int, float, float, const half *, const half *, half *, cudaStream_t);
template void run_v7<float, float, float>(int, int, int, float, float, const float *, const float *, float *, cudaStream_t);
template void run_v7<double, double, double>(int, int, int, double, double, const double *, const double *, double *, cudaStream_t);
