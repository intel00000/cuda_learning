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
    const int lane = tid % 32;
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

    // Advance pointers to the starting tile for this block
    const half *A_ptr = A + block_row_start;
    const half *B_ptr = B + block_col_start * K;

    // =============================================================================
    // loading lambdas
    // =============================================================================

    // Vectorized load, no bounds checks (interior block, aligned)
    auto load_vec = [&]()
    {
#pragma unroll
        for (int offset = 0; offset < BK; offset += as_stride)
            *reinterpret_cast<VecT *>(&As[as_col + offset][as_row_base]) =
                *reinterpret_cast<const VecT *>(&A_ptr[as_row_base + (as_col + offset) * M]);
#pragma unroll
        for (int offset = 0; offset < BN; offset += bs_stride)
            *reinterpret_cast<VecT *>(&Bs[bs_col + offset][bs_row_base]) =
                *reinterpret_cast<const VecT *>(&B_ptr[bs_row_base + (bs_col + offset) * K]);
    };
    // Vectorized load, K-remainder (interior block, aligned, partial K tile)
    auto load_vec_k_remain = [&](int k_remain)
    {
#pragma unroll
        for (int offset = 0; offset < BK; offset += as_stride)
        {
            int col = as_col + offset;
            if (col < k_remain)
                *reinterpret_cast<VecT *>(&As[col][as_row_base]) =
                    *reinterpret_cast<const VecT *>(&A_ptr[as_row_base + col * M]);
            else
#pragma unroll
                for (int v = 0; v < VEC; v++)
                    As[col][as_row_base + v] = __float2half(0.0f);
        }
#pragma unroll
        for (int offset = 0; offset < BN; offset += bs_stride)
        {
            int col = bs_col + offset;
            if (bs_row_base < k_remain)
                *reinterpret_cast<VecT *>(&Bs[col][bs_row_base]) =
                    *reinterpret_cast<const VecT *>(&B_ptr[bs_row_base + col * K]);
            else
#pragma unroll
                for (int v = 0; v < VEC; v++)
                    Bs[col][bs_row_base + v] = __float2half(0.0f);
        }
    };
    // Scalar load, no bounds checks (interior block, unaligned M or K)
    auto load_scalar = [&]()
    {
#pragma unroll
        for (int offset = 0; offset < BK; offset += as_stride)
        {
            int col = as_col + offset;
#pragma unroll
            for (int v = 0; v < VEC; v++)
                As[col][as_row_base + v] = A_ptr[(as_row_base + v) + col * M];
        }
#pragma unroll
        for (int offset = 0; offset < BN; offset += bs_stride)
        {
            int col = bs_col + offset;
#pragma unroll
            for (int v = 0; v < VEC; v++)
                Bs[col][bs_row_base + v] = B_ptr[(bs_row_base + v) + col * K];
        }
    };
    // Scalar load, K-remainder (interior block, unaligned, partial K tile)
    auto load_scalar_k_remain = [&](int k_remain)
    {
#pragma unroll
        for (int offset = 0; offset < BK; offset += as_stride)
        {
            int col = as_col + offset;
#pragma unroll
            for (int v = 0; v < VEC; v++)
            {
                if (col < k_remain)
                    As[col][as_row_base + v] = A_ptr[(as_row_base + v) + col * M];
                else
                    As[col][as_row_base + v] = __float2half(0.0f);
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
                    Bs[col][bs_row_base + v] = B_ptr[(bs_row_base + v) + col * K];
                else
                    Bs[col][bs_row_base + v] = __float2half(0.0f);
            }
        }
    };
    // Scalar load, full bounds checks
    auto load_edge = [&](int k_base)
    {
#pragma unroll
        for (int offset = 0; offset < BK; offset += as_stride)
        {
            int col = as_col + offset;
            int global_col = k_base + col;
#pragma unroll
            for (int v = 0; v < VEC; v++)
            {
                int global_row = block_row_start + as_row_base + v;
                if (global_row < M && global_col < K)
                    As[col][as_row_base + v] = A_ptr[(as_row_base + v) + col * M];
                else
                    As[col][as_row_base + v] = __float2half(0.0f);
            }
        }
#pragma unroll
        for (int offset = 0; offset < BN; offset += bs_stride)
        {
            int col = bs_col + offset;
            int global_col = block_col_start + col;
#pragma unroll
            for (int v = 0; v < VEC; v++)
            {
                int global_row = k_base + bs_row_base + v;
                if (global_row < K && global_col < N)
                    Bs[col][bs_row_base + v] = B_ptr[(bs_row_base + v) + col * K];
                else
                    Bs[col][bs_row_base + v] = __float2half(0.0f);
            }
        }
    };

    // =============================================================================
    // Compute lambda - identical for all paths (operates on shared memory)
    // =============================================================================

    // For each k in the tile, load values into registers and compute
    // threads in a warp execute these together (warp-collective)
    auto compute = [&]()
    {
        for (int k = 0; k < BK; k += WMMA_K)
        {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag[WMMA_TILES_M];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag[WMMA_TILES_N];
#pragma unroll
            // Load A fragments for this warp's M tiles
            for (int i = 0; i < WMMA_TILES_M; i++)
                load_matrix_sync(a_frag[i], &As[k][warp_row * WM + i * WMMA_M], BM + SKEW_A);
#pragma unroll
            // Load B fragments, Bs[n][k] is row_major from B's perspective: K dimension is contiguous
            for (int j = 0; j < WMMA_TILES_N; j++)
                load_matrix_sync(b_frag[j], &Bs[warp_col * WN + j * WMMA_N][k], BK + SKEW_B);
            // C += A × B (warp-collective operation)
#pragma unroll
            for (int i = 0; i < WMMA_TILES_M; i++)
#pragma unroll
                for (int j = 0; j < WMMA_TILES_N; j++)
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
    };

    // =============================================================================
    // Store lambda
    // =============================================================================

    // Direct WMMA store (requires M % 8 == 0, interior block)
    auto store_wmma = [&]()
    {
#pragma unroll
        for (int i = 0; i < WMMA_TILES_M; i++)
#pragma unroll
            for (int j = 0; j < WMMA_TILES_N; j++)
            {
                int c_row = block_row_start + warp_row * WM + i * WMMA_M;
                int c_col = block_col_start + warp_col * WN + j * WMMA_N;
                half *C_tile = &C[c_row + c_col * M];
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
                load_matrix_sync(c_frag, C_tile, M, mem_col_major);
#pragma unroll
                for (int t = 0; t < c_frag.num_elements; ++t)
                    c_frag.x[t] = __float2half(alpha * acc[i][j].x[t] + beta * __half2float(c_frag.x[t]));
                store_matrix_sync(C_tile, c_frag, M, mem_col_major);
            }
    };

    // Bounce buffer helper (for edge blocks, or when M is not a multiple of 8 and WMMA store cannot be used)
    auto store_bounce = [&](bool check_bounds)
    {
        // recast SMEM to float for storing accumulators in FP32
        float *bounce = reinterpret_cast<float *>(As);
#pragma unroll
        for (int i = 0; i < WMMA_TILES_M; i++)
#pragma unroll
            for (int j = 0; j < WMMA_TILES_N; j++)
            {
                int c_row = block_row_start + warp_row * WM + i * WMMA_M;
                int c_col = block_col_start + warp_col * WN + j * WMMA_N;
                // store accumulator fragments to shared memory, one warp at a time
                store_matrix_sync(bounce + warpId * WMMA_M * WMMA_N, acc[i][j], WMMA_M, mem_col_major);
                __syncwarp();
                for (int k = lane; k < WMMA_M * WMMA_N; k += 32)
                {
                    int global_row = c_row + k % WMMA_M;
                    int global_col = c_col + k / WMMA_M;
                    if (!check_bounds || (global_row < M && global_col < N))
                    {
                        float acc_val = bounce[warpId * WMMA_M * WMMA_N + k];
                        float c_val = __half2float(C[global_row + global_col * M]);
                        C[global_row + global_col * M] = __float2half(alpha * acc_val + beta * c_val);
                    }
                }
                __syncwarp();
            }
    };

    const bool is_edge_block = (block_row_start + BM > M) || (block_col_start + BN > N);
    const int K_aligned = K - (K % BK);
    // =============================================================================
    // Main loop
    // =============================================================================
    if (!is_edge_block)
    {
        const bool can_vec = (M % VEC == 0) && (K % VEC == 0);

        for (int k_base = 0; k_base < K_aligned; k_base += BK)
        {
            can_vec ? load_vec() : load_scalar();
            __syncthreads();
            A_ptr += BK * M;
            B_ptr += BK;
            compute();
            __syncthreads();
        }
        if (K_aligned < K)
        {
            const int k_remain = K - K_aligned;
            can_vec ? load_vec_k_remain(k_remain) : load_scalar_k_remain(k_remain);
            __syncthreads();
            compute();
            __syncthreads();
        }
        (M % 8 == 0) ? store_wmma() : store_bounce(false);
    }
    else
    {
        for (int k_base = 0; k_base < K; k_base += BK)
        {
            load_edge(k_base);
            __syncthreads();
            A_ptr += BK * M;
            B_ptr += BK;
            compute();
            __syncthreads();
        }
        store_bounce(true);
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
    constexpr int VEC = VecSize<half>();

    // Thread block dimensions
    constexpr int num_threads = (BM / WM) * (BN / WN) * 32; // one warp per WM×WN tile, 32 threads per warp
    constexpr int A_vecs_per_col = BM / VEC;
    constexpr int B_vecs_per_col = BK / VEC;
    static_assert(num_threads % A_vecs_per_col == 0, "num_threads must be divisible by A_vecs_per_col");
    static_assert(num_threads % B_vecs_per_col == 0, "num_threads must be divisible by B_vecs_per_col");
    static_assert(BK % (num_threads / A_vecs_per_col) == 0, "BK must be divisible by as_stride");
    static_assert(BN % (num_threads / B_vecs_per_col) == 0, "BN must be divisible by bs_stride");
    // Bounce buffer size check: num_warps * WMMA_M * WMMA_N floats must fit in As
    static_assert((BM / WM) * (BN / WN) * WMMA_M * WMMA_N * sizeof(float) <= BK * (BM + SKEW_A) * sizeof(half),
                  "Bounce buffer doesn't fit in As shared memory");

    // Thread block dimensions
    dim3 block(num_threads);
    // Grid dimensions
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    v6_wmma_kernel<BM, BN, BK, WM, WN, SKEW_A, SKEW_B, WMMA_M, WMMA_N, WMMA_K>
        <<<grid, block, 0, stream>>>(M, N, K, alpha, beta, A, B, C);

    CHECK_CUDA_ERROR(cudaGetLastError());
}
