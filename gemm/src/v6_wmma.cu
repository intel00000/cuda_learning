#include "v6_wmma.h"
#include "baseline.h"

static void bench_v6_fp16(int M, int N, int K, int warmup, int iters, size_t ws_size)
{
    Bench<half, half> b(M, N, K, ws_size);
    float alpha = 1.25f, beta = 0.75f;
    b.run_pair("v6_fp16", warmup, iters, [&]()
               { run_v6(M, N, K, alpha, beta, b.A(), b.B(), b.C(), b.stream()); }, [&]()
               { baseline_fp16_fp32acc_cm(b.lt(), b.stream(), M, N, K, alpha, beta, b.A(), b.B(), b.C_ref(), b.ws(), b.ws_size()); });
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
    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " | warmup=" << warmup << " iters=" << iters << " ws=" << ws_mb << "MB\n";

    bench_v6_fp16(M, N, K, warmup, iters, ws_size);
    return 0;
}
