// cuda-cmake/src/main.cu - CUDA + cuDNN/cuBLAS/cuFFT example using common header
// cuda-cmake/src/main.cu - 使用公共头文件的 CUDA + cuDNN/cuBLAS/cuFFT 示例
#include "../../common/cuda_helper.h"
#include <vector>
#include <numeric>
#include <cassert>
#include <cmath>
#ifdef HAVE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif
// cuBLAS/cuFFT headers are included via cuda_helper.h when HAVE_CUBLAS/HAVE_CUFFT are defined

//==============================================================================
// Kernel
//==============================================================================

__global__ void add_one(int* __restrict__ a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] += 1;
    }
}

//==============================================================================
// Demo: 使用 Stream 的异步拷贝 | Async copy with Stream
//==============================================================================

static void demoAsyncStream() {
    printf("\n--- Async Stream Demo ---\n");
    const int N = 1 << 18;

    CudaStream stream(cudaStreamNonBlocking);
    CudaPinnedMemory<float> h_in(N), h_out(N);
    CudaDeviceMemory<float> d_buf(N);

    for (size_t i = 0; i < (size_t)N; ++i) h_in[i] = static_cast<float>(i);

    CudaEvent start, stop;
    start.record(stream.get());
    d_buf.copyFromHostAsync(h_in.get(), stream.get());
    d_buf.copyToHostAsync(h_out.get(), stream.get());
    stop.record(stream.get());

    stream.synchronize();
    float ms = CudaEvent::elapsedMs(start, stop);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != static_cast<float>(i)) { ok = false; break; }
    }
    double sizeMB = (double)N * sizeof(float) / (1024.0 * 1024.0);
    printf("  Async round-trip %.1f MB: %.3f ms -> %s\n", sizeMB, ms, ok ? "PASSED" : "FAILED");
}

//==============================================================================
// Demo: cuBLAS (可选) | cuBLAS demo (optional)
//==============================================================================

#ifdef HAVE_CUBLAS
static void demoCuBLAS() {
    printf("\n--- cuBLAS Demo ---\n");
    const int N = 1024;

    std::vector<float> h_x(N, 1.0f), h_y(N, 2.0f);
    CudaDeviceMemory<float> d_x(N), d_y(N);
    d_x.copyFromHost(h_x.data());
    d_y.copyFromHost(h_y.data());

    CublasHandle handle;

    // y = alpha * x + y (SAXPY)
    float alpha = 3.0f;
    CUBLAS_CHECK(cublasSaxpy(handle.get(), N, &alpha, d_x.get(), 1, d_y.get(), 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    d_y.copyToHost(h_y.data());
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_y[i] - 5.0f) > 1e-5f) { ok = false; break; }
    }
    printf("  SAXPY (y = 3*x + y): %s (y[0]=%.1f)\n", ok ? "PASSED" : "FAILED", h_y[0]);
}
#endif

//==============================================================================
// Demo: cuFFT (可选) | cuFFT demo (optional)
//==============================================================================

#ifdef HAVE_CUFFT
static void demoCuFFT() {
    printf("\n--- cuFFT Demo ---\n");
    const int N = 256;

    std::vector<cufftComplex> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i].x = static_cast<float>(i);
        h_data[i].y = 0.0f;
    }

    CudaDeviceMemory<cufftComplex> d_data(N);
    d_data.copyFromHost(h_data.data());

    CufftPlan plan;
    plan.plan1d(N, CUFFT_C2C);
    assert(plan.valid());

    CudaEvent start, stop;
    start.record();
    CUFFT_CHECK(cufftExecC2C(plan.get(), reinterpret_cast<cufftComplex*>(d_data.get()),
                             reinterpret_cast<cufftComplex*>(d_data.get()), CUFFT_FORWARD));
    stop.record();
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = CudaEvent::elapsedMs(start, stop);
    printf("  1D C2C FFT (N=%d): %.3f ms\n", N, ms);
}
#endif

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

#ifdef HAVE_NVTX
    nvtx3::scoped_range r1{"add_one kernel"};
#endif

    CudaEvent e0, e1;
    e0.record();
    add_one<<<grid, block>>>(d.get(), N);
    e1.record();
    CUDA_CHECK_KERNEL();

    float ms = CudaEvent::elapsedMs(e0, e1);
    d.copyToHost(h.data());

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h[i] != i + 1) { ok = false; break; }
    }

    for (int i = 0; i < 16 && i < N; ++i) printf("%d ", h[i]);
    printf("...  (N=%d)\n", N);
    printf("  kernel elapsed: %.3f ms\n", ms);
    printf("  verification:   %s\n", ok ? "PASSED" : "FAILED");

    demoAsyncStream();

#ifdef HAVE_CUBLAS
    demoCuBLAS();
#else
    printf("\n(cuBLAS not found at configure time; skipping cuBLAS demo)\n");
#endif

#ifdef HAVE_CUFFT
    demoCuFFT();
#else
    printf("\n(cuFFT not found at configure time; skipping cuFFT demo)\n");
#endif

#ifdef HAVE_CUDNN
    printf("\n--- cuDNN Demo ---\n");
    size_t ver = cudnnGetVersion();
    printf("  cuDNN version: %zu\n", ver);

    CudnnHandle handle;
    CudnnTensorDescriptor xDesc;
    xDesc.set4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4);
    printf("  cuDNN tensor descriptor created successfully.\n");
#else
    printf("\n(cuDNN not found at configure time; skipping cuDNN demo)\n");
#endif

    printf("\n=== All demos completed. ===\n");
    return ok ? 0 : 1;
}
