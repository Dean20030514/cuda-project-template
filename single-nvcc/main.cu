// single-nvcc/main.cu - Simple CUDA example using common header
// single-nvcc/main.cu - 使用公共头文件的简单 CUDA 示例
//
// 注意：本文件刻意保持扁平结构，适合快速原型开发。
// 更模块化的写法请参考 cuda-cmake/src/main.cu。
//
// Note: This file is intentionally kept flat for quick prototyping.
// See cuda-cmake/src/main.cu for a more modular structure.
#include "../common/cuda_helper.h"
#include "../common/cuda_demos.cuh"

//==============================================================================
// Kernel
//==============================================================================

__global__ void add_one(int* __restrict__ a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] += 1;
    }
}

//==============================================================================
// Main
//==============================================================================

int main() {
    printDeviceInfo();

    try {
        measureBandwidth();
        bool ok1 = demoKernel(add_one);
        bool ok2 = demoAsyncStream();

        printf("\n=== All demos completed. ===\n");
        return (ok1 && ok2) ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
