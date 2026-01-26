#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16 // Block size for CUDA kernel

#define CHECK_CUDA_ERROR(call)                                               \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

class Timer
{
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}

    void reset() { start = std::chrono::high_resolution_clock::now(); }

    double elapsed() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};

/*
 * CUDA kernel to perform matrix multiplication C = A * B
 * where A is of size M x K, B is of size K x N, and C is of size M x N
 *
 * Parameters:
 *  A - pointer to matrix A in device memory
 *  B - pointer to matrix B in device memory
 *  C - pointer to matrix C in device memory
 * M_ - number of rows in matrix A and C
 * K_ - number of columns in matrix A and rows in matrix B
 * N_ - number of columns in matrix B and C
 */
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int M_, int K_, int N_)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (row < M_ && col < N_)
    { // Ensure within bounds
        float sum = 0;
        for (int i = 0; i < K_; i++)
        {
            sum += A[row * K_ + i] * B[i * N_ + col];
        }
        C[row * N_ + col] = sum;
    }
}

// CPU version of matrix multiplication
void matrixMultiplyCPU(float *A, float *B, float *C, int M_, int K_, int N_)
{
    for (int i = 0; i < M_; i++)
    {
        for (int j = 0; j < N_; j++)
        {
            float sum = 0;
            for (int k = 0; k < K_; k++)
            {
                sum += A[i * K_ + k] * B[k * N_ + j];
            }
            C[i * N_ + j] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    // check input
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <K> <N>" << std::endl;
        return 1;
    }
    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);

    // Reset device
    CHECK_CUDA_ERROR(cudaDeviceReset());

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host matrices
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_GPU = (float *)malloc(size_C);
    float *h_C_CPU = (float *)malloc(size_C);
    // Initialize matrices with values
    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = (float)(i + 1); // Initialize A with values 1 to M*K
    }
    for (int i = 0; i < K * N; i++)
    {
        h_B[i] = (float)(i + 1); // Initialize B with values 1 to K*N
    }

    // std::cout << "Matrix A:" << std::endl;
    // for (int i = 0; i < M; i++)
    // {
    //     for (int j = 0; j < K; j++)
    //     {
    //         std::cout << h_A[i * K + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Matrix B:" << std::endl;
    // for (int i = 0; i < K; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         std::cout << h_B[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // ======= CPU Execution ==========
    Timer cpu_timer;
    cpu_timer.reset();
    matrixMultiplyCPU(h_A, h_B, h_C_CPU, M, K, N);
    double cpu_time_taken = cpu_timer.elapsed();

    // ======= GPU Execution ==========
    // Device matrices
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    // Define execution configuration
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE); // threads per block
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    Timer timer;
    timer.reset();
    matrixMultiplyGPU<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    // Synchronize to ensure kernel execution
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    double gpu_time_taken = timer.elapsed();

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_GPU, d_C, size_C, cudaMemcpyDeviceToHost));

    // compare the results from CPU and GPU
    bool isEqual = true;
    for (int i = 0; i < M * N; i++)
    {
        float diff = h_C_CPU[i] - h_C_GPU[i];
        if (diff < -1e-5 || diff > 1e-5)
        {
            isEqual = false;
            std::cout << "Mismatch at index " << i << ": CPU = " << h_C_CPU[i] << ", GPU = " << h_C_GPU[i] << std::endl;
        }
    }
    std::cout << "Results are " << (isEqual ? "equal" : "not equal") << std::endl;

    // Print a small portion of result matrices for verification
    // std::cout << "Matrix C (CPU result, first 5 elements): ";
    // for (int i = 0; i < std::min(5, M * N); i++)
    //     std::cout << h_C_CPU[i] << " ";
    // std::cout << std::endl;

    // std::cout << "Matrix C (GPU result, first 100 elements): ";
    // for (int i = 0; i < std::min(100, M * N); i++)
    //     std::cout << h_C_GPU[i] << " ";
    // std::cout << std::endl;

    std::cout << "CPU Time taken: " << cpu_time_taken << " seconds" << std::endl;
    std::cout << "GPU Time taken: " << gpu_time_taken << " seconds" << std::endl;

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_GPU);
    free(h_C_CPU);

    return 0;
}
