#pragma once

#include "cuda_raii.h"
#include <climits>
#include <cstdio>

// ============================================================================
// elapsedMs — free function replacing CudaEvent::elapsedMs()
// ============================================================================

/**
 * @warning 此函数会隐式调用 cudaEventSynchronize() 等待两个事件完成，
 *          因此会阻塞调用线程直到 GPU 执行到相应事件。
 *
 * @warning This function implicitly calls cudaEventSynchronize() on both
 *          events, blocking the calling thread until the GPU reaches each
 *          event. If you need non-blocking behavior, synchronize manually.
 */
[[nodiscard]] inline float elapsedMs(const CudaEvent& start, const CudaEvent& end) {
    CUDA_CHECK_THROW(cudaEventSynchronize(start.get()));
    CUDA_CHECK_THROW(cudaEventSynchronize(end.get()));
    float ms = 0.f;
    CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, start.get(), end.get()));
    return ms;
}

// ============================================================================
// calcGridSize
// ============================================================================

/**
 * @brief 计算合适的 grid 大小 | Calculate appropriate grid size
 *
 * @param n 总元素数量 | Total number of elements
 * @param blockSize 每个 block 的线程数 | Number of threads per block
 * @return 需要的 grid 大小 | Required grid size
 */
inline unsigned int calcGridSize(size_t n, unsigned int blockSize) {
    size_t grid = (n + blockSize - 1) / blockSize;
    assert(grid <= UINT_MAX && "grid size exceeds unsigned int range");
    return static_cast<unsigned int>(grid);
}

// ============================================================================
// printDeviceInfo
// ============================================================================

/**
 * @brief 打印当前 GPU 设备信息 | Print current GPU device information
 *
 * 输出设备名称、算力、显存等关键信息，方便调试和性能分析。
 * 自 CUDA 13.0 起，`cudaDeviceProp` 不再包含 `clockRate` / `memoryClockRate`
 *（12.x 已弃用）；此处通过条件编译自动适配。
 *
 * Prints device name, compute capability, memory, etc. for debugging
 * and performance analysis.
 * Since CUDA 13.0, `cudaDeviceProp` no longer has `clockRate` / `memoryClockRate`
 * (deprecated in 12.x); this function adapts automatically via CUDART_VERSION.
 */
inline void printDeviceInfo(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("=== GPU Device %d: %s ===\n", device, prop.name);
    printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
    printf("  SM count           : %d\n", prop.multiProcessorCount);
    printf("  Global memory      : %.1f GiB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared mem/block   : %.1f KiB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  Max threads/block  : %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size          : %d\n", prop.warpSize);

#if CUDART_VERSION >= 13000
    // clockRate / memoryClockRate removed from cudaDeviceProp in CUDA 13.0; use attributes
    int clockKHz = 0, memClockKHz = 0, eccEnabled = 0, asyncEngines = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&eccEnabled, cudaDevAttrEccEnabled, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&asyncEngines, cudaDevAttrAsyncEngineCount, device));
#else
    // CUDA 12.x: these fields are deprecated but still present in cudaDeviceProp
    int clockKHz = prop.clockRate;
    int memClockKHz = prop.memoryClockRate;
    int eccEnabled = prop.ECCEnabled;
    int asyncEngines = prop.asyncEngineCount;
#endif

    printf("  Clock rate (core)  : %.0f MHz\n", clockKHz / 1000.0);
    printf("  Clock rate (mem)   : %.0f MHz\n", memClockKHz / 1000.0);
    printf("  Memory bus width   : %d-bit\n", prop.memoryBusWidth);
    printf("  L2 cache size      : %d KB\n", prop.l2CacheSize / 1024);
    printf("  ECC enabled        : %s\n", eccEnabled ? "Yes" : "No");
    printf("  Async engines      : %d\n", asyncEngines);
    printf("==============================\n");
}

// ============================================================================
// measureBandwidth
// ============================================================================

/**
 * @brief 测量 Host-Device 带宽 | Measure Host-Device bandwidth
 *
 * 使用锁页内存测量 H2D 和 D2H 传输带宽。包含 warmup 和多次平均。
 *
 * Measures H2D and D2H transfer bandwidth using pinned memory.
 * Includes warmup pass and multi-run averaging for stable results.
 *
 * @param sizeBytes 测试数据大小（字节）| Test data size in bytes
 * @param iterations 测量次数（取平均）| Number of iterations to average
 */
inline void measureBandwidth(size_t sizeBytes = 64 * 1024 * 1024, int iterations = 5) {
    CudaPinnedMemory<char> h(sizeBytes);
    CudaDeviceMemory<char> d(sizeBytes);
    h.get()[0] = 0;

    // 使用独立 stream + cudaMemcpyAsync，这是现代 CUDA 的推荐写法。
    // 在单 stream 上异步操作仍串行执行，带宽结果与同步版本基本一致；
    // 但此写法可自然扩展到多 stream overlap 场景。
    //
    // Uses a dedicated stream + cudaMemcpyAsync (modern best practice).
    // On a single stream, async operations still execute serially, so
    // measured bandwidth is essentially the same as the synchronous version;
    // however this form naturally extends to multi-stream overlap scenarios.
    CudaStream stream;
    CudaEvent start, stop;

    // Warmup: 首次传输可能因驱动初始化/页表建立偏慢
    // Warmup: first transfer may be slow due to driver init / page table setup
    CUDA_CHECK_THROW(cudaMemcpyAsync(d.get(), h.get(), sizeBytes, cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK_THROW(cudaMemcpyAsync(h.get(), d.get(), sizeBytes, cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();

    float h2d_total = 0.f, d2h_total = 0.f;
    for (int i = 0; i < iterations; ++i) {
        // H2D
        start.record(stream.get());
        CUDA_CHECK_THROW(cudaMemcpyAsync(d.get(), h.get(), sizeBytes, cudaMemcpyHostToDevice, stream.get()));
        stop.record(stream.get());
        h2d_total += elapsedMs(start, stop);

        // D2H
        start.record(stream.get());
        CUDA_CHECK_THROW(cudaMemcpyAsync(h.get(), d.get(), sizeBytes, cudaMemcpyDeviceToHost, stream.get()));
        stop.record(stream.get());
        d2h_total += elapsedMs(start, stop);
    }

    float h2d_ms = h2d_total / iterations;
    float d2h_ms = d2h_total / iterations;
    double sizeMB = sizeBytes / (1024.0 * 1024.0);
    // GB/s = bytes / (ms * 1e6) = bytes / ms / 1e6, 使用 SI 定义 1 GB = 1e9 bytes
    // GB/s = bytes / (ms * 1e6), using SI definition 1 GB = 1e9 bytes
    double h2d_gbs = (sizeBytes / (h2d_ms / 1000.0)) / 1e9;
    double d2h_gbs = (sizeBytes / (d2h_ms / 1000.0)) / 1e9;
    printf("=== Bandwidth Test (%.0f MiB, avg of %d runs) ===\n", sizeMB, iterations);
    printf("  H2D : %.2f GB/s (%.2f ms)\n", h2d_gbs, h2d_ms);
    printf("  D2H : %.2f GB/s (%.2f ms)\n", d2h_gbs, d2h_ms);
    printf("==============================\n");
}
