#pragma once
#include "common.h"
#include <type_traits>

// Map C++ scalar type -> cudaDataType_t
template <typename T>
struct CudaTypeTraits;

template <>
struct CudaTypeTraits<double>
{
    static constexpr cudaDataType_t cuda_type = CUDA_R_64F;
    static constexpr const char *name = "FP64";
};
template <>
struct CudaTypeTraits<float>
{
    static constexpr cudaDataType_t cuda_type = CUDA_R_32F;
    static constexpr const char *name = "FP32";
};
template <>
struct CudaTypeTraits<half>
{
    static constexpr cudaDataType_t cuda_type = CUDA_R_16F;
    static constexpr const char *name = "FP16";
};
template <>
struct CudaTypeTraits<__nv_bfloat16>
{
    static constexpr cudaDataType_t cuda_type = CUDA_R_16BF;
    static constexpr const char *name = "BF16";
};
template <>
struct CudaTypeTraits<__nv_fp8_e4m3>
{
    static constexpr cudaDataType_t cuda_type = CUDA_R_8F_E4M3;
    static constexpr const char *name = "FP8_E4M3";
};
template <>
struct CudaTypeTraits<int8_t>
{
    static constexpr cudaDataType_t cuda_type = CUDA_R_8I;
    static constexpr const char *name = "INT8";
};
template <>
struct CudaTypeTraits<int32_t>
{
    static constexpr cudaDataType_t cuda_type = CUDA_R_32I;
    static constexpr const char *name = "INT32";
};

// Column-major GEMM using cublasLt: D = alpha * op(A)*op(B) + beta * C
template <typename InputT, typename ScaleT, typename OutputT = InputT>
inline void cublaslt_gemm_cm(cublasLtHandle_t handle, cudaStream_t stream,
                             int M, int N, int K,
                             const ScaleT &alpha, const ScaleT &beta,
                             const InputT *dA, const InputT *dB, OutputT *dC,
                             cublasComputeType_t computeType, void *workspace = nullptr, size_t workspaceSize = 0,
                             cublasOperation_t transA = CUBLAS_OP_N, cublasOperation_t transB = CUBLAS_OP_N)
{
    constexpr cudaDataType_t inputType = CudaTypeTraits<InputT>::cuda_type;
    constexpr cudaDataType_t outputType = CudaTypeTraits<OutputT>::cuda_type;
    constexpr cudaDataType_t scaleType = CudaTypeTraits<ScaleT>::cuda_type;

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;

    // Leading dimensions for column-major
    // A is MxK  => lda = M
    // B is KxN  => ldb = K
    // C is MxN  => ldc = M
    const int64_t lda = (transA == CUBLAS_OP_N) ? M : K;
    const int64_t ldb = (transB == CUBLAS_OP_N) ? K : N;
    const int64_t ldc = M;

    // Rows/cols of the matrices as stored (before transpose)
    const uint64_t rowsA = (transA == CUBLAS_OP_N) ? M : K;
    const uint64_t colsA = (transA == CUBLAS_OP_N) ? K : M;
    const uint64_t rowsB = (transB == CUBLAS_OP_N) ? K : N;
    const uint64_t colsB = (transB == CUBLAS_OP_N) ? N : K;

    // Create matmul descriptor
    CHECK_CUBLAS_ERROR(cublasLtMatmulDescCreate(&desc, computeType, scaleType));
    CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Create matrix layouts
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&layoutA, inputType, rowsA, colsA, lda));
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&layoutB, inputType, rowsB, colsB, ldb));
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&layoutC, outputType, M, N, ldc));

    // Create preference and set workspace
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceCreate(&pref));
    if (workspace && workspaceSize > 0)
    {
        CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                                &workspaceSize, sizeof(workspaceSize)));
    }

    // Query for best algorithm
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResultCount = 0;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(handle, desc, layoutA, layoutB, layoutC, layoutC,
                                                      pref, 1, &heuristicResult, &returnedResultCount));

    // Execute
    if (returnedResultCount == 0)
        CHECK_CUBLAS_ERROR(cublasLtMatmul(handle, desc, &alpha, dA, layoutA, dB, layoutB, &beta, dC, layoutC, dC, layoutC,
                                          NULL, workspace, workspaceSize, stream));
    else
        CHECK_CUBLAS_ERROR(cublasLtMatmul(handle, desc, &alpha, dA, layoutA, dB, layoutB, &beta, dC, layoutC, dC, layoutC,
                                          &heuristicResult.algo, workspace, workspaceSize, stream));

    // Cleanup
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(desc);
}

// Convenience wrappers - naming: baseline_<input>_<accum if different>_cm

// FP64 input, FP64 accumulation
inline void baseline_fp64_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                             double alpha, double beta, const double *A, const double *B, double *C,
                             void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<double, double, double>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_64F, ws, wsSize);
}
// FP32 input, FP32 accumulation
inline void baseline_fp32_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                             float alpha, float beta, const float *A, const float *B, float *C,
                             void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<float, float, float>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_32F, ws, wsSize);
}
// FP32 input, TF32 accumulation (uses tensor cores)
inline void baseline_tf32_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                             float alpha, float beta, const float *A, const float *B, float *C,
                             void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<float, float, float>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_32F_FAST_TF32, ws, wsSize);
}
// FP16 input, FP32 accumulation
inline void baseline_fp16_fp32acc_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                                     float alpha, float beta, const half *A, const half *B, half *C,
                                     void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<half, float, half>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_32F, ws, wsSize);
}
// FP16 input, FP16 accumulation
inline void baseline_fp16_fp16acc_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                                     half alpha, half beta, const half *A, const half *B, half *C,
                                     void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<half, half, half>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_16F, ws, wsSize);
}
// BF16 input, FP32 accumulation
inline void baseline_bf16_fp32acc_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                                     float alpha, float beta, const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
                                     void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<__nv_bfloat16, float, __nv_bfloat16>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_32F, ws, wsSize);
}
// FP8 E4M3 input, FP32 output
inline void baseline_fp8e4m3_fp32out_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                                        float alpha, float beta, const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B, float *C,
                                        void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<__nv_fp8_e4m3, float, float>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_32F, ws, wsSize);
}
// FP8 E4M3 input, FP16 output
inline void baseline_fp8e4m3_fp16out_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                                        float alpha, float beta, const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B, half *C,
                                        void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<__nv_fp8_e4m3, float, half>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_32F, ws, wsSize);
}
// FP8 E4M3 input, BF16 output
inline void baseline_fp8e4m3_bf16out_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                                        float alpha, float beta, const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B, __nv_bfloat16 *C,
                                        void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<__nv_fp8_e4m3, float, __nv_bfloat16>(h, s, M, N, K, alpha, beta, A, B, C, CUBLAS_COMPUTE_32F, ws, wsSize);
}
// INT8 input, INT32 output
inline void baseline_int8_int32out_cm(cublasLtHandle_t h, cudaStream_t s, int M, int N, int K,
                                      int32_t alpha, int32_t beta, const int8_t *A, const int8_t *B, int32_t *C,
                                      void *ws = nullptr, size_t wsSize = 0)
{
    cublaslt_gemm_cm<int8_t, int32_t, int32_t>(h, s, M, N, K, alpha, beta, A, B, C,
                                               CUBLAS_COMPUTE_32I, ws, wsSize);
}
