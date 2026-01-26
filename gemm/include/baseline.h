#pragma once
#include "common.h"
#include <type_traits>

// Map C++ scalar type -> cudaDataType_t used by cublasGemmEx
template <typename T>
struct CublasScalarTraits;
template <>
struct CublasScalarTraits<float>
{
    static constexpr cudaDataType_t type = CUDA_R_32F;
};
template <>
struct CublasScalarTraits<double>
{
    static constexpr cudaDataType_t type = CUDA_R_64F;
};
template <>
struct CublasScalarTraits<half>
{
    static constexpr cudaDataType_t type = CUDA_R_16F;
};

// Column-major GEMM baseline: C = alpha*A*B + beta*C
// A: MxK, B: KxN, C: MxN (all column-major in memory)
template <typename T, typename ScaleT>
inline void cublas_gemm_cm(
    cublasHandle_t handle,
    int M, int N, int K,
    const ScaleT &alpha, const ScaleT &beta,
    const T *dA, const T *dB, T *dC,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo,
    cublasOperation_t transA = CUBLAS_OP_N,
    cublasOperation_t transB = CUBLAS_OP_N)
{
    // Column-major leading dimensions:
    // A is MxK  => lda = M
    // B is KxN  => ldb = K
    // C is MxN  => ldc = M
    const int lda = (transA == CUBLAS_OP_N) ? M : K;
    const int ldb = (transB == CUBLAS_OP_N) ? K : N;
    const int ldc = M;

    // cublasGemmEx expects alpha/beta "Scale Type" matching (computeType, Ctype)
    // and supports the combinations including:
    // - float with CUBLAS_COMPUTE_32F / _PEDANTIC
    // - float with CUBLAS_COMPUTE_32F_FAST_TF32
    // - double with CUBLAS_COMPUTE_64F / _PEDANTIC
    CHECK_CUBLAS_ERROR(cublasGemmEx(
        handle,
        transA, transB,
        M, N, K,
        &alpha, dA, CublasScalarTraits<T>::type, lda,
        dB, CublasScalarTraits<T>::type, ldb,
        &beta,
        dC, CublasScalarTraits<T>::type, ldc,
        computeType, algo));
}

// Convenience wrappers
inline void cublas_gemm_fp64_cm(cublasHandle_t h, int M, int N, int K,
                                double alpha, double beta,
                                const double *A, const double *B, double *C,
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT)
{
    cublas_gemm_cm<double, double>(h, M, N, K, alpha, beta, A, B, C,
                                   CUBLAS_COMPUTE_64F, algo);
}
inline void cublas_gemm_fp32_cm(cublasHandle_t h, int M, int N, int K,
                                float alpha, float beta,
                                const float *A, const float *B, float *C,
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT)
{
    cublas_gemm_cm<float, float>(h, M, N, K, alpha, beta, A, B, C,
                                 CUBLAS_COMPUTE_32F, algo);
}
inline void cublas_gemm_tf32_cm(cublasHandle_t h, int M, int N, int K,
                                float alpha, float beta,
                                const float *A, const float *B, float *C,
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP)
{
    cublas_gemm_cm<float, float>(h, M, N, K, alpha, beta, A, B, C,
                                 CUBLAS_COMPUTE_32F_FAST_TF32, algo);
}

// FP16 storage, FP32 accumulation
inline void cublas_gemm_fp16_cm(cublasHandle_t h, int M, int N, int K,
                                float alpha, float beta,
                                const half *A, const half *B, half *C,
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT)
{
    cublas_gemm_cm<half, float>(h, M, N, K, alpha, beta, A, B, C,
                                CUBLAS_COMPUTE_32F, algo);
}
// FP16 storage, FP16 accumulation
inline void cublas_gemm_fp16_fast_cm(cublasHandle_t h, int M, int N, int K,
                                     half alpha, half beta,
                                     const half *A, const half *B, half *C,
                                     cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT)
{
    cublas_gemm_cm<half, half>(h, M, N, K, alpha, beta, A, B, C,
                               CUBLAS_COMPUTE_16F, algo);
}
