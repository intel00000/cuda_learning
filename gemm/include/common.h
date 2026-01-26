#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <functional>
#include <iostream>

// macro to wrap CUDA and CUBLAS calls and check for errors
#define CHECK_CUDA_ERROR(call)                                                           \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err_ = call;                                                         \
        if (err_ != cudaSuccess)                                                         \
        {                                                                                \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err_) << std::endl;                          \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)
#define CHECK_CUBLAS_ERROR(call)                                                           \
    do                                                                                     \
    {                                                                                      \
        cublasStatus_t status_ = call;                                                     \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                              \
        {                                                                                  \
            std::cerr << "CUBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << status_ << std::endl;                                             \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)
#define CHECK_CURAND_ERROR(call)                                                           \
    do                                                                                     \
    {                                                                                      \
        curandStatus_t status_ = call;                                                     \
        if (status_ != CURAND_STATUS_SUCCESS)                                              \
        {                                                                                  \
            std::cerr << "cuRAND error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << status_ << std::endl;                                             \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

static int arg_parse_int(int argc, char **argv, const char *key, int def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0)
            return std::atoi(argv[i + 1]);
    return def;
}

// structure to hold benchmark results
struct BenchResult
{
    const char *method;
    float milliseconds = 0.0f;
    double gflops = 0.0f;
};

// assuming standard gemm operation
// A is M x K, B is K x N, C is M x N
// we have 2 * M * N * K floating point operations
// this is considering the multiply and add as separate operations
// though it can also be thought as fused multiply-add (FMA)
static inline double gflops_from_ms(int M, int K, int N, float ms)
{
    double operations = 2.0 * (double)M * (double)N * (double)K;
    return (operations / 1e9) / ((double)ms / 1e3);
}

// use cudaEvent to measure elapsed time
static inline float time_ms_events(cudaStream_t stream,
                                   int warmup_iters, int timed_iters,
                                   const std::function<void()> &func)
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warm up
    for (int i = 0; i < warmup_iters; ++i)
        func();

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i = 0; i < timed_iters; ++i)
        func();
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return ms / timed_iters;
}

// Helper to create an aligned vector type for T
template <typename T>
struct __align__(sizeof(T) * 4) Vector4
{
    T x, y, z, w;
};

// Universal random fill kernel using Philox + curand_uniform4 (4 values per call)
template <typename T>
__global__ void generate_random_kernel(T *__restrict__ dst, size_t n, unsigned long long seed)
{
    // 1. Calculate thread index
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;

    // 2. Vectorized limit (process chunks of 4)
    size_t n_vec = n / 4;

    // 3. Initialize RNG
    // We initialize the state to the starting offset for this thread: 4 * tid.
    // This ensures dst[0] gets seq(0), dst[1] gets seq(1), etc.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, tid * 4, &state);

    // 4. Grid-Stride Loop
    // Cast dst to our vectorized type for efficient 128-bit stores
    Vector4<T> *dst_vec = reinterpret_cast<Vector4<T> *>(dst);

    for (size_t idx = tid; idx < n_vec; idx += total_threads)
    {
        // Generate 4 floats
        float4 r = curand_uniform4(&state);
        // Store as a single vectorized write to encourage coalescing
        dst_vec[idx] = {T(r.x), T(r.y), T(r.z), T(r.w)};

        // 5. Skipahead
        // We generated 1 chunk (4 numbers). The state counter advanced by 1.
        // We need to jump to the next chunk assigned to this thread.
        skipahead(total_threads - 1, &state);
    }

    // 6. Handle Tail (remaining 1-3 elements)
    // Only the one capable thread needs to do this to avoid divergence
    if (tid == 0 && n_vec * 4 < n)
    {
        size_t tail_start = n_vec * 4;
        curand_init(seed, 0, tail_start, &state);
        for (size_t i = tail_start; i < n; i++)
            dst[i] = T(curand_uniform(&state));
    }
}

template <typename T>
static inline void fill_random_device_array(T *d_array, size_t n, cudaStream_t stream = 0,
                                            unsigned long long seed = 42ULL)
{
    // float/double handled by cuRAND library
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        curandGenerator_t gen;
        CHECK_CURAND_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen, seed));
        CHECK_CURAND_ERROR(curandSetStream(gen, stream));

        if constexpr (std::is_same_v<T, float>)
            CHECK_CURAND_ERROR(curandGenerateUniform(gen, (float *)d_array, n));
        else
            CHECK_CURAND_ERROR(curandGenerateUniformDouble(gen, (double *)d_array, n));

        CHECK_CURAND_ERROR(curandDestroyGenerator(gen));
        return;
    }

    // Determine launch configuration
    // We utilize a fixed grid size to maximize occupancy without overhead
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, generate_random_kernel<T>, 0, 0);

    // Clamp grid size to avoid launching too many blocks for small n
    size_t n_vec = (n + 3) / 4;
    int grid = (n_vec + blockSize - 1) / blockSize;
    if (grid > minGridSize * 2)
        grid = minGridSize * 2; // Cap grid size

    generate_random_kernel<T><<<grid, blockSize, 0, stream>>>(d_array, n, seed);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Comparison results
struct CompareResult
{
    float max_abs_err;
    float max_rel_err;
    size_t num_mismatches;
};

// Compare two arrays using vectorized loads + warp reduction
template <typename T>
__global__ void compare_arrays_kernel(const T *__restrict__ ref, const T *__restrict__ test, size_t n,
                                      float *__restrict__ max_abs_err, float *__restrict__ max_rel_err,
                                      unsigned int *__restrict__ mismatch_count, float rel_tol)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vec_elems = n / 4;

    float local_max_abs = 0.0f;
    float local_max_rel = 0.0f;
    int local_mismatch = 0;

    // Vector path: 4 elements at a time
    if (idx < vec_elems)
    {
        const size_t base = idx * 4;
        Vector4<T> r4 = *reinterpret_cast<const Vector4<T> *>(ref + base);
        Vector4<T> t4 = *reinterpret_cast<const Vector4<T> *>(test + base);

        float r[4] = {static_cast<float>(r4.x), static_cast<float>(r4.y),
                      static_cast<float>(r4.z), static_cast<float>(r4.w)};
        float t[4] = {static_cast<float>(t4.x), static_cast<float>(t4.y),
                      static_cast<float>(t4.z), static_cast<float>(t4.w)};
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float abs_err = fabsf(r[i] - t[i]);
            float rel_err = (fabsf(r[i]) > 1e-6f) ? (abs_err / fabsf(r[i])) : abs_err;
            local_max_abs = fmaxf(local_max_abs, abs_err);
            local_max_rel = fmaxf(local_max_rel, rel_err);
            if (rel_err > rel_tol)
                local_mismatch++;
        }
    }
    // Scalar tail
    const size_t tail_idx = vec_elems * 4 + idx;
    if (tail_idx < n)
    {
        float r = static_cast<float>(ref[tail_idx]);
        float t = static_cast<float>(test[tail_idx]);
        float abs_err = fabsf(r - t);
        float rel_err = (fabsf(r) > 1e-6f) ? (abs_err / fabsf(r)) : abs_err;
        local_max_abs = fmaxf(local_max_abs, abs_err);
        local_max_rel = fmaxf(local_max_rel, rel_err);
        if (rel_err > rel_tol)
            local_mismatch++;
    }

    // Warp reduction + atomic from the first thread in each warp
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    float warp_max_abs = cg::reduce(tile, local_max_abs, cg::greater<float>());
    float warp_max_rel = cg::reduce(tile, local_max_rel, cg::greater<float>());
    int warp_mismatch = cg::reduce(tile, local_mismatch, cg::plus<int>());
    // !ONLY correct if all values are non-negative!
    // use atomicCAS if there are negative values
    if (tile.thread_rank() == 0)
    {
        atomicMax(reinterpret_cast<int *>(max_abs_err), __float_as_int(warp_max_abs));
        atomicMax(reinterpret_cast<int *>(max_rel_err), __float_as_int(warp_max_rel));
        if (warp_mismatch > 0)
            atomicAdd(mismatch_count, warp_mismatch);
    }
}

template <typename T>
CompareResult compare_device_arrays(const T *ref, const T *test, size_t n, cudaStream_t stream = 0, float rel_tol = 1e-3f)
{
    // Unified memory for host/device access
    float *max_abs, *max_rel;
    unsigned int *mismatch;

    CHECK_CUDA_ERROR(cudaMallocManaged(&max_abs, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&max_rel, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&mismatch, sizeof(unsigned int)));

    *max_abs = 0.0f;
    *max_rel = 0.0f;
    *mismatch = 0;

    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, compare_arrays_kernel<T>, 0, 0);
    size_t n_vec = (n + 3) / 4;
    int grid = (n_vec + blockSize - 1) / blockSize;
    if (grid > minGridSize * 2)
        grid = minGridSize * 2;

    compare_arrays_kernel<T><<<grid, blockSize, 0, stream>>>(ref, test, n, max_abs, max_rel, mismatch, rel_tol);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CompareResult result;
    result.max_abs_err = *max_abs;
    result.max_rel_err = *max_rel;
    result.num_mismatches = *mismatch;

    CHECK_CUDA_ERROR(cudaFree(max_abs));
    CHECK_CUDA_ERROR(cudaFree(max_rel));
    CHECK_CUDA_ERROR(cudaFree(mismatch));

    return result;
}
