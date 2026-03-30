// single-nvcc/main.cu - Simple CUDA example using common header
// single-nvcc/main.cu - 使用公共头文件的简单 CUDA 示例
//
// 注意：本文件刻意保持扁平结构，适合快速原型开发。
// 更模块化的写法请参考 cuda-cmake/src/main.cu。
//
// Note: This file is intentionally kept flat for quick prototyping.
// See cuda-cmake/src/main.cu for a more modular structure.
#include "../common/cuda_helper.h"
#include <vector>
#include <numeric>
#include <cassert>

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
    measureBandwidth();

    printf("\n--- Kernel Demo ---\n");
    const int N = 1 << 20;

    std::vector<int> h(N);
    std::iota(h.begin(), h.end(), 0);

    CudaDeviceMemory<int> d(N);
    d.copyFromHost(h.data());

    const int block = 256;
    const int grid = calcGridSize(N, block);

    CudaEvent e0, e1;
    e0.record();
    add_one<<<grid, block>>>(d.get(), N);
    e1.record();
    CUDA_CHECK_KERNEL();

    float ms = CudaEvent::elapsedMs(e0, e1);
    d.copyToHost(h.data());

    // Verify results | 验证结果
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h[i] != i + 1) { ok = false; break; }
    }

    for (int i = 0; i < 16 && i < N; ++i) printf("%d ", h[i]);
    printf("...  (N=%d)\n", N);
    printf("  kernel elapsed: %.3f ms\n", ms);
    printf("  verification:   %s\n", ok ? "PASSED" : "FAILED");

    // Demo: 使用 Stream 异步拷贝 | Async copy with Stream
    {
        printf("\n--- Async Stream Demo ---\n");
        const int M = 1 << 18;
        CudaStream stream(cudaStreamNonBlocking);
        CudaPinnedMemory<float> h_in(M), h_out(M);
        CudaDeviceMemory<float> d_buf(M);

        for (size_t i = 0; i < (size_t)M; ++i) h_in[i] = static_cast<float>(i);

        CudaEvent s0, s1;
        s0.record(stream.get());
        d_buf.copyFromHostAsync(h_in.get(), stream.get());
        d_buf.copyToHostAsync(h_out.get(), stream.get());
        s1.record(stream.get());
        stream.synchronize();

        float sms = CudaEvent::elapsedMs(s0, s1);
        bool sok = true;
        for (int i = 0; i < M; ++i) {
            if (h_out[i] != static_cast<float>(i)) { sok = false; break; }
        }
        double sizeMB = (double)M * sizeof(float) / (1024.0 * 1024.0);
        printf("  Async round-trip %.1f MB: %.3f ms -> %s\n", sizeMB, sms, sok ? "PASSED" : "FAILED");
    }

    printf("\n=== All demos completed. ===\n");
    return ok ? 0 : 1;
}
