#include "v3_multiout.h"
#include "baseline.h"

#include <cstring>
#include <iomanip>
#include <iostream>

template <typename T>
void run_v3_benchmark(const char *tag, int M, int N, int K, int warmup, int iters)
{
    std::cout << std::string(50, '-') << "\n";
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    cublasHandle_t cublas_h;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_h));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_h, stream));

    size_t a_elems = (size_t)M * K;
    size_t b_elems = (size_t)K * N;
    size_t c_elems = (size_t)M * N;

    T *dA, *dB, *dC, *dC_ref;
    CHECK_CUDA_ERROR(cudaMalloc(&dA, a_elems * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dB, b_elems * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dC, c_elems * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dC_ref, c_elems * sizeof(T)));

    fill_random_device_array<T>(dA, a_elems, stream, 1);
    fill_random_device_array<T>(dB, b_elems, stream, 2);
    fill_random_device_array<T>(dC, c_elems, stream, 3);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dC_ref, dC, c_elems * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float alpha = 1.25f, beta = 0.75f;

    // Define kernel lambdas
    auto run_func = [&]()
    {
        if constexpr (std::is_same_v<T, double>)
            run_v3<double, double, double>(M, N, K, (double)alpha, (double)beta, dA, dB, dC, stream);
        else if constexpr (std::is_same_v<T, float>)
            run_v3<float, float, float>(M, N, K, alpha, beta, dA, dB, dC, stream);
        else if constexpr (std::is_same_v<T, half>)
            run_v3<half, float, float>(M, N, K, alpha, beta, dA, dB, dC, stream);
    };
    auto run_baseline = [&]()
    {
        if constexpr (std::is_same_v<T, double>)
            cublas_gemm_fp64_cm(cublas_h, M, N, K, (double)alpha, (double)beta, dA, dB, dC_ref);
        else if constexpr (std::is_same_v<T, float>)
            cublas_gemm_fp32_cm(cublas_h, M, N, K, alpha, beta, dA, dB, dC_ref);
        else if constexpr (std::is_same_v<T, half>)
            cublas_gemm_fp16_cm(cublas_h, M, N, K, alpha, beta, dA, dB, dC_ref);
    };

    // Benchmark
    float ms = time_ms_events(stream, warmup, iters, run_func);
    std::cout << tag << " perf: ms=" << ms << " gflops=" << gflops_from_ms(M, K, N, ms) << "\n";
    float ref_ms = time_ms_events(stream, warmup, iters, run_baseline);
    std::cout << tag << " baseline perf: ms=" << ref_ms << " gflops=" << gflops_from_ms(M, K, N, ref_ms)
              << " percent=" << (ref_ms / ms * 100.0f) << "%\n";

    // Verify
    float rel_tol = std::is_same_v<T, half> ? 1e-2f : 1e-4f;
    CompareResult cmp = compare_device_arrays<T>(dC_ref, dC, c_elems, stream, rel_tol);
    std::cout << tag << " verify: max_abs=" << cmp.max_abs_err
              << " max_rel=" << cmp.max_rel_err
              << " mismatches=" << cmp.num_mismatches << "\n";

    CHECK_CUDA_ERROR(cudaFree(dA));
    CHECK_CUDA_ERROR(cudaFree(dB));
    CHECK_CUDA_ERROR(cudaFree(dC));
    CHECK_CUDA_ERROR(cudaFree(dC_ref));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_h));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main(int argc, char **argv)
{
    int M = arg_parse_int(argc, argv, "-M", 8192);
    int N = arg_parse_int(argc, argv, "-N", 2048);
    int K = arg_parse_int(argc, argv, "-K", 8192);
    int warmup = arg_parse_int(argc, argv, "--warmup", 3);
    int iters = arg_parse_int(argc, argv, "--iters", 5);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " warmup=" << warmup << " iters=" << iters << "\n";

    run_v3_benchmark<half>("v3_fp16_mixed", M, N, K, warmup, iters);
    run_v3_benchmark<float>("v3_fp32", M, N, K, warmup, iters);
    run_v3_benchmark<double>("v3_fp64", M, N, K, warmup, iters);

    return 0;
}
