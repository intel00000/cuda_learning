#include <iostream>

__global__ void kernel3D()
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    int block = blockIdx.x;
    int globalThreadId = block * (blockDim.x * blockDim.y * blockDim.z) + (z * (blockDim.x * blockDim.y)) + (y * blockDim.x) + x;

    printf("Hello from thread (%d, %d, %d) in block %d! Global Thread ID: %d\n", x, y, z, block, globalThreadId);
}

int main()
{
    dim3 threadsPerBlock(2, 2, 2); // 2x2x2 = 8 threads per block

    // Launch kernel with 2 blocks containing a 3D thread layout
    kernel3D<<<2, threadsPerBlock, 0, 0>>>();

    cudaDeviceSynchronize(); // Wait for GPU execution to complete
    return 0;
}
