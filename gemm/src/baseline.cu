#include "common.h"
#include "baseline.h"

#include <iomanip>
#include <iostream>

template <typename T>
void run_cublas_benchmark(const char *tag, int M, int N, int K, int warmup, int iters)
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    cublasHandle_t h;
    CHECK_CUBLAS_ERROR(cublasCreate(&h));
    CHECK_CUBLAS_ERROR(cublasSetStream(h, stream));

    const size_t a_elems = (size_t)M * (size_t)K;
    const size_t b_elems = (size_t)K * (size_t)N;
    const size_t c_elems = (size_t)M * (size_t)N;

    T *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&dA, a_elems * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dB, b_elems * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dC, c_elems * sizeof(T)));

    fill_random_device_array<T>(dA, a_elems, stream, 1ULL);
    fill_random_device_array<T>(dB, b_elems, stream, 2ULL);
    fill_random_device_array<T>(dC, c_elems, stream, 3ULL);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float alpha = 1.25f, beta = 0.75f;
    auto bench = [&](const char *name, auto fn)
    {
        fill_random_device_array<T>(dC, c_elems, stream, 3ULL);
        float ms = time_ms_events(stream, warmup, iters, fn);
        std::cout << name << " ms=" << ms << " gflops=" << gflops_from_ms(M, K, N, ms) << "\n";
    };

    if constexpr (std::is_same_v<T, half>)
    {
        bench("cublas_fp16", [&]()
              { cublas_gemm_fp16_cm(h, M, N, K, alpha, beta, dA, dB, dC); });
        half alpha_h = __float2half(alpha), beta_h = __float2half(beta);
        bench("cublas_fp16_fast", [&]()
              { cublas_gemm_fp16_fast_cm(h, M, N, K, alpha_h, beta_h, dA, dB, dC); });
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        bench("cublas_fp32", [&]()
              { cublas_gemm_fp32_cm(h, M, N, K, alpha, beta, dA, dB, dC); });
        bench("cublas_tf32", [&]()
              { cublas_gemm_tf32_cm(h, M, N, K, alpha, beta, dA, dB, dC); });
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        bench("cublas_fp64", [&]()
              { cublas_gemm_fp64_cm(h, M, N, K, (double)alpha, (double)beta, dA, dB, dC); });
    }

    CHECK_CUDA_ERROR(cudaFree(dA));
    CHECK_CUDA_ERROR(cudaFree(dB));
    CHECK_CUDA_ERROR(cudaFree(dC));
    CHECK_CUBLAS_ERROR(cublasDestroy(h));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main(int argc, char **argv)
{
    int M = arg_parse_int(argc, argv, "-M", 8192);
    int N = arg_parse_int(argc, argv, "-N", 8192);
    int K = arg_parse_int(argc, argv, "-K", 8192);
    int warmup = arg_parse_int(argc, argv, "--warmup", 5);
    int iters = arg_parse_int(argc, argv, "--iters", 10);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " warmup=" << warmup << " iters=" << iters << "\n";

    run_cublas_benchmark<half>("fp16", M, N, K, warmup, iters);
    run_cublas_benchmark<float>("fp32", M, N, K, warmup, iters);
    run_cublas_benchmark<double>("fp64", M, N, K, warmup, iters);

    return 0;
}
