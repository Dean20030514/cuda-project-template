// cuda-cmake/src/main.cu - CUDA + cuDNN/cuBLAS/cuFFT example using common header
// cuda-cmake/src/main.cu - 使用公共头文件的 CUDA + cuDNN/cuBLAS/cuFFT 示例
#include "../../common/cuda_helper.h"
#include "../../common/cuda_demos.cuh"
#include <vector>
#include <cmath>
#include <cassert>
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
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

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
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    float ms = elapsedMs(start, stop);
    printf("  1D C2C FFT (N=%d): %.3f ms\n", N, ms);
}
#endif

//==============================================================================
// Main
//==============================================================================

int main() {
    printDeviceInfo();

    try {
        measureBandwidth();

#ifdef HAVE_NVTX
        nvtx3::scoped_range r1{"add_one kernel"};
#endif
        bool ok1 = demoKernel(add_one);
        bool ok2 = demoAsyncStream();

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
        return (ok1 && ok2) ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
