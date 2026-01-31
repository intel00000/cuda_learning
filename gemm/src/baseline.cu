#include "common.h"
#include "baseline.h"

#include <iomanip>
#include <iostream>

template <typename InputT, typename OutputT = InputT>
class BaselineBench
{
public:
    BaselineBench(int M, int N, int K, size_t ws_size) : M_(M), N_(N), K_(K), ws_size_(ws_size)
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasLtCreate(&handle_));

        a_elems_ = (size_t)M * K;
        b_elems_ = (size_t)K * N;
        c_elems_ = (size_t)M * N;

        CHECK_CUDA_ERROR(cudaMalloc(&dA_, a_elems_ * sizeof(InputT)));
        CHECK_CUDA_ERROR(cudaMalloc(&dB_, b_elems_ * sizeof(InputT)));
        CHECK_CUDA_ERROR(cudaMalloc(&dC_, c_elems_ * sizeof(OutputT)));
        CHECK_CUDA_ERROR(cudaMalloc(&ws_, ws_size_));

        fill_random_device_array<InputT>(dA_, a_elems_, stream_, 1ULL);
        fill_random_device_array<InputT>(dB_, b_elems_, stream_, 2ULL);
        fill_random_device_array<OutputT>(dC_, c_elems_, stream_, 3ULL);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    ~BaselineBench()
    {
        cudaFree(dA_);
        cudaFree(dB_);
        cudaFree(dC_);
        cudaFree(ws_);
        cublasLtDestroy(handle_);
        cudaStreamDestroy(stream_);
    }

    template <typename Fn>
    void run(const char *name, int warmup, int iters, Fn fn)
    {
        fill_random_device_array<OutputT>(dC_, c_elems_, stream_, 3ULL);
        float ms = time_ms_events(stream_, warmup, iters, fn);
        std::cout << std::setw(20) << std::left << name
                  << "ms=" << std::setw(10) << std::right << ms
                  << "  TFLOPS=" << std::setw(8) << tflops_from_ms(M_, K_, N_, ms) << "\n";
    }

    cublasLtHandle_t handle() const { return handle_; }
    cudaStream_t stream() const { return stream_; }
    InputT *A() const { return dA_; }
    InputT *B() const { return dB_; }
    OutputT *C() const { return dC_; }
    void *ws() const { return ws_; }
    size_t ws_size() const { return ws_size_; }
    int M() const { return M_; }
    int N() const { return N_; }
    int K() const { return K_; }

private:
    int M_, N_, K_;
    size_t ws_size_, a_elems_, b_elems_, c_elems_;
    cudaStream_t stream_;
    cublasLtHandle_t handle_;
    InputT *dA_ = nullptr;
    InputT *dB_ = nullptr;
    OutputT *dC_ = nullptr;
    void *ws_ = nullptr;
};

void bench_fp64(int M, int N, int K, int warmup, int iters, size_t ws_size)
{
    print_header("FP64");
    BaselineBench<double> b(M, N, K, ws_size);
    b.run("FP64", warmup, iters, [&]()
          { baseline_fp64_cm(b.handle(), b.stream(), M, N, K, 1.25, 0.75, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
}
void bench_fp32(int M, int N, int K, int warmup, int iters, size_t ws_size)
{
    print_header("FP32 / TF32");
    BaselineBench<float> b(M, N, K, ws_size);
    b.run("FP32", warmup, iters, [&]()
          { baseline_fp32_cm(b.handle(), b.stream(), M, N, K, 1.25f, 0.75f, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
    b.run("TF32", warmup, iters, [&]()
          { baseline_tf32_cm(b.handle(), b.stream(), M, N, K, 1.25f, 0.75f, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
}
void bench_fp16(int M, int N, int K, int warmup, int iters, size_t ws_size)
{
    print_header("FP16");
    BaselineBench<half> b(M, N, K, ws_size);
    b.run("FP16 in, FP32 acc", warmup, iters, [&]()
          { baseline_fp16_fp32acc_cm(b.handle(), b.stream(), M, N, K, 1.25f, 0.75f, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
    half ah = __float2half(1.25f), bh = __float2half(0.75f);
    b.run("FP16 in, FP16 acc", warmup, iters, [&]()
          { baseline_fp16_fp16acc_cm(b.handle(), b.stream(), M, N, K, ah, bh, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
}
void bench_bf16(int M, int N, int K, int warmup, int iters, size_t ws_size)
{
    print_header("BF16");
    BaselineBench<__nv_bfloat16> b(M, N, K, ws_size);
    b.run("BF16 in, FP32 acc", warmup, iters, [&]()
          { baseline_bf16_fp32acc_cm(b.handle(), b.stream(), M, N, K, 1.25f, 0.75f, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
}
void bench_fp8_e4m3(int M, int N, int K, int warmup, int iters, size_t ws_size)
{
    print_header("FP8 E4M3");
    {
        BaselineBench<__nv_fp8_e4m3, float> b(M, N, K, ws_size);
        b.run("FP8_E4M3 -> FP32", warmup, iters, [&]()
              { baseline_fp8e4m3_fp32out_cm(b.handle(), b.stream(), M, N, K, 1.0f, 0.0f, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
    }
    {
        BaselineBench<__nv_fp8_e4m3, half> b(M, N, K, ws_size);
        b.run("FP8_E4M3 -> FP16", warmup, iters, [&]()
              { baseline_fp8e4m3_fp16out_cm(b.handle(), b.stream(), M, N, K, 1.0f, 0.0f, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
    }
    {
        BaselineBench<__nv_fp8_e4m3, __nv_bfloat16> b(M, N, K, ws_size);
        b.run("FP8_E4M3 -> BF16", warmup, iters, [&]()
              { baseline_fp8e4m3_bf16out_cm(b.handle(), b.stream(), M, N, K, 1.0f, 0.0f, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
    }
}
void bench_int8(int M, int N, int K, int warmup, int iters, size_t ws_size)
{
    print_header("INT8");
    BaselineBench<int8_t, int32_t> b(M, N, K, ws_size);
    b.run("INT8 -> INT32", warmup, iters, [&]()
          { baseline_int8_int32out_cm(b.handle(), b.stream(), M, N, K, 1, 0, b.A(), b.B(), b.C(), b.ws(), b.ws_size()); });
}

int main(int argc, char **argv)
{
    int M = arg_parse_int(argc, argv, "-M", 8192);
    int N = arg_parse_int(argc, argv, "-N", 8192);
    int K = arg_parse_int(argc, argv, "-K", 8192);
    int warmup = arg_parse_int(argc, argv, "--warmup", 3);
    int iters = arg_parse_int(argc, argv, "--iters", 5);
    int ws_mb = arg_parse_int(argc, argv, "--ws", 64);
    size_t ws_size = (size_t)ws_mb << 20;

    std::cout << std::fixed << std::setprecision(3);
    print_header("cuBLASLt GEMM Baseline");
    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " | warmup=" << warmup << " iters=" << iters << " ws=" << ws_mb << "MB\n";

    bench_fp16(M, N, K, warmup, iters, ws_size);
    bench_bf16(M, N, K, warmup, iters, ws_size);
    bench_fp32(M, N, K, warmup, iters, ws_size);
    bench_fp64(M, N, K, warmup, iters, ws_size);
    bench_fp8_e4m3(M, N, K, warmup, iters, ws_size);
    bench_int8(M, N, K, warmup, iters, ws_size);

    return 0;
}
