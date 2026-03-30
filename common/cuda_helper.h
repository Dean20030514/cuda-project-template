#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#include <string>

/**
 * @brief CUDA 错误检查宏 | CUDA Error Checking Macro
 *
 * 用于包装 CUDA API 调用并检查返回值。如果发生错误，打印详细的
 * 错误信息（包含文件名、行号、错误代码和描述）后退出程序。
 *
 * Wraps CUDA API calls and checks return values. If an error occurs,
 * prints detailed error information (filename, line number, error code
 * and description) then exits the program.
 *
 * 用法 | Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: [%d] %s\n",                     \
                    __FILE__, __LINE__, static_cast<int>(err),                    \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief CUDA 核函数启动后错误检查宏 | CUDA Kernel Launch Error Checking Macro
 *
 * 在核函数启动后使用，检查启动配置错误和异步执行错误。
 * 同时调用 cudaDeviceSynchronize 以捕获异步执行错误。
 *
 * Use after kernel launch to check for launch configuration errors
 * and asynchronous execution errors. Also calls cudaDeviceSynchronize
 * to catch async execution errors.
 *
 * 用法 | Usage:
 *   my_kernel<<<grid, block>>>(args);
 *   CUDA_CHECK_KERNEL();
 */
#define CUDA_CHECK_KERNEL()                                                       \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: [%d] %s\n",       \
                    __FILE__, __LINE__, static_cast<int>(err),                    \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
        err = cudaDeviceSynchronize();                                            \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA kernel exec error at %s:%d: [%d] %s\n",         \
                    __FILE__, __LINE__, static_cast<int>(err),                    \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief CUDA 异常式错误检查宏 | CUDA Exception-based Error Checking Macro
 *
 * 与 CUDA_CHECK 类似，但抛出 std::runtime_error 而非调用 exit()，
 * 使 RAII 析构函数能正常执行，避免资源泄漏。适用于需要错误恢复的场景。
 *
 * Similar to CUDA_CHECK but throws std::runtime_error instead of exit(),
 * allowing RAII destructors to run properly. Use when error recovery is needed.
 *
 * 用法 | Usage: CUDA_CHECK_THROW(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK_THROW(call)                                                    \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                             \
                std::string("CUDA error at ") + __FILE__ + ":" +                  \
                std::to_string(__LINE__) + ": [" +                                \
                std::to_string(static_cast<int>(err)) + "] " +                    \
                cudaGetErrorString(err));                                          \
        }                                                                         \
    } while (0)

/**
 * @brief CUDA 核函数异常式错误检查宏 | CUDA Kernel Exception-based Error Checking Macro
 *
 * 与 CUDA_CHECK_KERNEL 类似，但抛出异常。
 *
 * Similar to CUDA_CHECK_KERNEL but throws exceptions.
 */
#define CUDA_CHECK_KERNEL_THROW()                                                 \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                             \
                std::string("CUDA kernel launch error at ") + __FILE__ + ":" +    \
                std::to_string(__LINE__) + ": [" +                                \
                std::to_string(static_cast<int>(err)) + "] " +                    \
                cudaGetErrorString(err));                                          \
        }                                                                         \
        err = cudaDeviceSynchronize();                                            \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                             \
                std::string("CUDA kernel exec error at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + ": [" +                                \
                std::to_string(static_cast<int>(err)) + "] " +                    \
                cudaGetErrorString(err));                                          \
        }                                                                         \
    } while (0)

/**
 * @brief RAII 风格的 CUDA Stream 包装类 | RAII-style CUDA Stream Wrapper Class
 *
 * 自动管理 cudaStream_t 的生命周期。支持默认流和非阻塞流。
 *
 * Automatically manages cudaStream_t lifecycle. Supports default and
 * non-blocking streams.
 */
class CudaStream {
public:
    explicit CudaStream(unsigned int flags = cudaStreamDefault) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    ~CudaStream() { if (stream_) cudaStreamDestroy(stream_); }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const { return stream_; }
    explicit operator cudaStream_t() const { return stream_; }

    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

    [[nodiscard]] bool query() const {
        cudaError_t status = cudaStreamQuery(stream_);
        if (status == cudaSuccess) return true;
        if (status == cudaErrorNotReady) return false;
        CUDA_CHECK(status);
        return false;
    }

private:
    cudaStream_t stream_ = nullptr;
};

/**
 * @brief RAII 风格的 CUDA Event 包装类 | RAII-style CUDA Event Wrapper Class
 *
 * 自动管理 cudaEvent_t 的生命周期，确保事件在离开作用域时被销毁，
 * 防止资源泄漏。
 *
 * Automatically manages cudaEvent_t lifecycle, ensuring events are
 * destroyed when leaving scope, preventing resource leaks.
 */
class CudaEvent {
public:
    explicit CudaEvent(unsigned int flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }
    ~CudaEvent() { if (event_) cudaEventDestroy(event_); }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaEvent_t get() const { return event_; }
    explicit operator cudaEvent_t() const { return event_; }

    void record(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(event_, stream)); }
    void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

    [[nodiscard]] static float elapsedMs(const CudaEvent& start, const CudaEvent& end) {
        CUDA_CHECK(cudaEventSynchronize(start.event_));
        CUDA_CHECK(cudaEventSynchronize(end.event_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }

private:
    cudaEvent_t event_ = nullptr;
};

/**
 * @brief RAII 风格的 CUDA 设备内存包装类 | RAII-style CUDA Device Memory Wrapper Class
 *
 * 自动管理设备内存的分配和释放。
 *
 * Automatically manages device memory allocation and deallocation.
 *
 * 用法 | Usage:
 *   CudaDeviceMemory<float> d_data(1024);
 *   d_data.copyFromHost(h_data);
 */
template<typename T>
class CudaDeviceMemory {
    static_assert(std::is_trivially_copyable<T>::value,
                  "CudaDeviceMemory<T> requires T to be trivially copyable");
public:
    explicit CudaDeviceMemory(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    ~CudaDeviceMemory() {
        if (ptr_) cudaFree(ptr_);
    }

    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    [[nodiscard]] T* get() const { return ptr_; }
    [[nodiscard]] size_t count() const { return count_; }
    [[nodiscard]] size_t bytes() const { return count_ * sizeof(T); }

    void copyFromHost(const T* src, size_t n = 0) {
        if (!ptr_) return;
        size_t copyCount = (n == 0) ? count_ : std::min(n, count_);
        CUDA_CHECK(cudaMemcpy(ptr_, src, copyCount * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyToHost(T* dst, size_t n = 0) const {
        if (!ptr_) return;
        size_t copyCount = (n == 0) ? count_ : std::min(n, count_);
        CUDA_CHECK(cudaMemcpy(dst, ptr_, copyCount * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void copyFromHostAsync(const T* src, cudaStream_t stream, size_t n = 0) {
        if (!ptr_) return;
        size_t copyCount = (n == 0) ? count_ : std::min(n, count_);
        CUDA_CHECK(cudaMemcpyAsync(ptr_, src, copyCount * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void copyToHostAsync(T* dst, cudaStream_t stream, size_t n = 0) const {
        if (!ptr_) return;
        size_t copyCount = (n == 0) ? count_ : std::min(n, count_);
        CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, copyCount * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    // 注意：逐字节填充，与 std::memset 相同。对非零值仅 memset(0) 可安全用于所有类型。
    // Note: fills per-byte like std::memset. Only memset(0) is safe for all types.
    void memset(int value = 0) {
        if (!ptr_) return;
        CUDA_CHECK(cudaMemset(ptr_, value, count_ * sizeof(T)));
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

/**
 * @brief RAII 风格的 CUDA 锁页主机内存包装类 | RAII-style CUDA Pinned Host Memory Wrapper
 *
 * 锁页内存可以实现更快的 Host-Device 数据传输，支持异步拷贝。
 *
 * Pinned (page-locked) host memory enables faster Host-Device transfers
 * and is required for truly asynchronous copies.
 *
 * 用法 | Usage:
 *   CudaPinnedMemory<float> pinned(1024);
 *   pinned[0] = 3.14f;
 *   d_mem.copyFromHostAsync(pinned.get(), stream);
 */
template<typename T>
class CudaPinnedMemory {
    static_assert(std::is_trivially_copyable<T>::value,
                  "CudaPinnedMemory<T> requires T to be trivially copyable");
public:
    explicit CudaPinnedMemory(size_t count, unsigned int flags = cudaHostAllocDefault)
        : count_(count) {
        CUDA_CHECK(cudaHostAlloc(&ptr_, count * sizeof(T), flags));
    }
    ~CudaPinnedMemory() {
        if (ptr_) cudaFreeHost(ptr_);
    }

    CudaPinnedMemory(const CudaPinnedMemory&) = delete;
    CudaPinnedMemory& operator=(const CudaPinnedMemory&) = delete;

    CudaPinnedMemory(CudaPinnedMemory&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    CudaPinnedMemory& operator=(CudaPinnedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    [[nodiscard]] T* get() const { return ptr_; }
    [[nodiscard]] size_t count() const { return count_; }
    [[nodiscard]] size_t bytes() const { return count_ * sizeof(T); }

    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

/**
 * @brief 计算合适的 grid 大小 | Calculate appropriate grid size
 *
 * @param n 总元素数量 | Total number of elements
 * @param blockSize 每个 block 的线程数 | Number of threads per block
 * @return 需要的 grid 大小 | Required grid size
 */
inline unsigned int calcGridSize(unsigned int n, unsigned int blockSize) {
    return (n + blockSize - 1) / blockSize;
}

inline int calcGridSize(int n, int blockSize) {
    return (n + blockSize - 1) / blockSize;
}

inline unsigned int calcGridSize(size_t n, unsigned int blockSize) {
    return static_cast<unsigned int>((n + blockSize - 1) / blockSize);
}

/**
 * @brief 打印当前 GPU 设备信息 | Print current GPU device information
 *
 * 输出设备名称、算力、显存等关键信息，方便调试和性能分析。
 *
 * Prints device name, compute capability, memory, etc. for debugging
 * and performance analysis.
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
    printf("  Memory bus width   : %d-bit\n", prop.memoryBusWidth);
    printf("  L2 cache size      : %d KB\n", prop.l2CacheSize / 1024);
    printf("==============================\n");
}

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

    CudaEvent start, stop;

    // Warmup: 首次传输可能因驱动初始化/页表建立偏慢
    // Warmup: first transfer may be slow due to driver init / page table setup
    CUDA_CHECK(cudaMemcpy(d.get(), h.get(), sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h.get(), d.get(), sizeBytes, cudaMemcpyDeviceToHost));

    float h2d_total = 0.f, d2h_total = 0.f;
    for (int i = 0; i < iterations; ++i) {
        // H2D
        start.record();
        CUDA_CHECK(cudaMemcpy(d.get(), h.get(), sizeBytes, cudaMemcpyHostToDevice));
        stop.record();
        h2d_total += CudaEvent::elapsedMs(start, stop);

        // D2H
        start.record();
        CUDA_CHECK(cudaMemcpy(h.get(), d.get(), sizeBytes, cudaMemcpyDeviceToHost));
        stop.record();
        d2h_total += CudaEvent::elapsedMs(start, stop);
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

#ifdef HAVE_CUDNN
#include <cudnn.h>

/**
 * @brief cuDNN 错误检查宏 | cuDNN Error Checking Macro
 */
#define CUDNN_CHECK(call)                                                         \
    do {                                                                          \
        cudnnStatus_t status = (call);                                            \
        if (status != CUDNN_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudnnGetErrorString(status));             \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief RAII 风格的 cuDNN Handle 包装类 | RAII-style cuDNN Handle Wrapper Class
 */
class CudnnHandle {
public:
    CudnnHandle() { CUDNN_CHECK(cudnnCreate(&handle_)); }
    ~CudnnHandle() { if (handle_) cudnnDestroy(handle_); }

    CudnnHandle(const CudnnHandle&) = delete;
    CudnnHandle& operator=(const CudnnHandle&) = delete;

    CudnnHandle(CudnnHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    CudnnHandle& operator=(CudnnHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) cudnnDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudnnHandle_t get() const { return handle_; }
    explicit operator cudnnHandle_t() const { return handle_; }

private:
    cudnnHandle_t handle_ = nullptr;
};

/**
 * @brief RAII 风格的 cuDNN TensorDescriptor 包装类 | RAII-style cuDNN TensorDescriptor Wrapper Class
 */
class CudnnTensorDescriptor {
public:
    CudnnTensorDescriptor() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_)); }
    ~CudnnTensorDescriptor() { if (desc_) cudnnDestroyTensorDescriptor(desc_); }

    CudnnTensorDescriptor(const CudnnTensorDescriptor&) = delete;
    CudnnTensorDescriptor& operator=(const CudnnTensorDescriptor&) = delete;

    CudnnTensorDescriptor(CudnnTensorDescriptor&& other) noexcept : desc_(other.desc_) {
        other.desc_ = nullptr;
    }
    CudnnTensorDescriptor& operator=(CudnnTensorDescriptor&& other) noexcept {
        if (this != &other) {
            if (desc_) cudnnDestroyTensorDescriptor(desc_);
            desc_ = other.desc_;
            other.desc_ = nullptr;
        }
        return *this;
    }

    void set4d(cudnnTensorFormat_t format, cudnnDataType_t dataType,
               int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_, format, dataType, n, c, h, w));
    }

    [[nodiscard]] cudnnTensorDescriptor_t get() const { return desc_; }
    explicit operator cudnnTensorDescriptor_t() const { return desc_; }

private:
    cudnnTensorDescriptor_t desc_ = nullptr;
};

#endif // HAVE_CUDNN

#ifdef HAVE_CUBLAS
#include <cublas_v2.h>

/**
 * @brief cuBLAS 错误检查宏 | cuBLAS Error Checking Macro
 */
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",                  \
                    __FILE__, __LINE__, static_cast<int>(status));                \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief RAII 风格的 cuBLAS Handle 包装类 | RAII-style cuBLAS Handle Wrapper Class
 */
class CublasHandle {
public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
    ~CublasHandle() { if (handle_) cublasDestroy(handle_); }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    CublasHandle(CublasHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    CublasHandle& operator=(CublasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) cublasDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cublasHandle_t get() const { return handle_; }
    explicit operator cublasHandle_t() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

#endif // HAVE_CUBLAS

#ifdef HAVE_CUFFT
#include <cufft.h>

/**
 * @brief cuFFT 错误检查宏 | cuFFT Error Checking Macro
 */
#define CUFFT_CHECK(call)                                                         \
    do {                                                                          \
        cufftResult status = (call);                                              \
        if (status != CUFFT_SUCCESS) {                                            \
            fprintf(stderr, "cuFFT error at %s:%d: status=%d\n",                   \
                    __FILE__, __LINE__, static_cast<int>(status));                \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/**
 * @brief RAII 风格的 cuFFT Plan 包装类 | RAII-style cuFFT Plan Wrapper Class
 */
class CufftPlan {
public:
    CufftPlan() = default;
    ~CufftPlan() { if (plan_) cufftDestroy(plan_); }

    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;

    CufftPlan(CufftPlan&& other) noexcept : plan_(other.plan_) {
        other.plan_ = 0;
    }
    CufftPlan& operator=(CufftPlan&& other) noexcept {
        if (this != &other) {
            if (plan_) cufftDestroy(plan_);
            plan_ = other.plan_;
            other.plan_ = 0;
        }
        return *this;
    }

    void plan1d(int nx, cufftType type, int batch = 1) {
        if (plan_) cufftDestroy(plan_);
        CUFFT_CHECK(cufftPlan1d(&plan_, nx, type, batch));
    }

    void plan2d(int nx, int ny, cufftType type) {
        if (plan_) cufftDestroy(plan_);
        CUFFT_CHECK(cufftPlan2d(&plan_, nx, ny, type));
    }

    [[nodiscard]] bool valid() const { return plan_ != 0; }
    [[nodiscard]] cufftHandle get() const { return plan_; }
    explicit operator cufftHandle() const { return plan_; }

private:
    cufftHandle plan_ = 0;
};

#endif // HAVE_CUFFT
