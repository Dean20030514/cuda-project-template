#pragma once

#include "cuda_check.h"
#include <algorithm>
#include <cassert>
#include <type_traits>

// ============================================================================
// UniqueHandle — generic RAII wrapper for non-copyable GPU handles
// ============================================================================

/**
 * @brief 通用 RAII 句柄包装模板 | Generic RAII handle wrapper template
 *
 * @tparam T      句柄类型 | Handle type (e.g. cudaStream_t, cufftHandle)
 * @tparam Null   空值 | Null sentinel (nullptr for pointers, 0 for int handles)
 * @tparam Deleter 无状态删除器 | Stateless deleter functor
 */
template<typename T, auto Null, typename Deleter>
class UniqueHandle {
public:
    UniqueHandle() noexcept = default;
    explicit UniqueHandle(T handle) noexcept : handle_(handle) {}

    // CUDA destroy functions return cudaSuccess by convention; ignoring
    // the return value in the destructor is safe and keeps it noexcept.
    ~UniqueHandle() noexcept { reset(); }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& o) noexcept : handle_(o.handle_) {
        o.handle_ = Null;
    }
    UniqueHandle& operator=(UniqueHandle&& o) noexcept {
        if (this != &o) {
            reset();
            handle_ = o.handle_;
            o.handle_ = Null;
        }
        return *this;
    }

    [[nodiscard]] T get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != Null; }
    explicit operator T() const noexcept { return handle_; }

    void reset() noexcept {
        if (handle_ != Null) { Deleter{}(handle_); handle_ = Null; }
    }

private:
    T handle_ = Null;
};

// ============================================================================
// CudaStream
// ============================================================================

/**
 * @brief RAII 风格的 CUDA Stream 包装类 | RAII-style CUDA Stream Wrapper Class
 *
 * 自动管理 cudaStream_t 的生命周期。支持默认流和非阻塞流。
 *
 * Automatically manages cudaStream_t lifecycle. Supports default and
 * non-blocking streams.
 */
class CudaStream {
    struct Deleter { void operator()(cudaStream_t s) const noexcept { cudaStreamDestroy(s); } };
public:
    explicit CudaStream(unsigned int flags = cudaStreamDefault) {
        cudaStream_t s = nullptr;
        CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&s, flags));
        handle_ = UniqueHandle<cudaStream_t, nullptr, Deleter>(s);
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return handle_.get(); }
    explicit operator cudaStream_t() const noexcept { return handle_.get(); }

    void synchronize() { CUDA_CHECK_THROW(cudaStreamSynchronize(handle_.get())); }

    [[nodiscard]] bool query() const {
        cudaError_t status = cudaStreamQuery(handle_.get());
        if (status == cudaSuccess) return true;
        if (status == cudaErrorNotReady) return false;
        CUDA_CHECK_THROW(status);
        return false; // unreachable: CUDA_CHECK_THROW always throws for error status
    }

private:
    UniqueHandle<cudaStream_t, nullptr, Deleter> handle_;
};

// ============================================================================
// CudaEvent
// ============================================================================

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
    struct Deleter { void operator()(cudaEvent_t e) const noexcept { cudaEventDestroy(e); } };
public:
    explicit CudaEvent(unsigned int flags = cudaEventDefault) {
        cudaEvent_t e = nullptr;
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&e, flags));
        handle_ = UniqueHandle<cudaEvent_t, nullptr, Deleter>(e);
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return handle_.get(); }
    explicit operator cudaEvent_t() const noexcept { return handle_.get(); }

    void record(cudaStream_t stream = 0) { CUDA_CHECK_THROW(cudaEventRecord(handle_.get(), stream)); }
    void synchronize() { CUDA_CHECK_THROW(cudaEventSynchronize(handle_.get())); }

private:
    UniqueHandle<cudaEvent_t, nullptr, Deleter> handle_;
};

// ============================================================================
// CudaDeviceMemory
// ============================================================================

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
    struct Deleter { void operator()(T* p) const noexcept { cudaFree(p); } };
public:
    explicit CudaDeviceMemory(size_t count) : count_(count) {
        T* p = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&p, count * sizeof(T)));
        handle_ = UniqueHandle<T*, nullptr, Deleter>(p);
    }

    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
        : handle_(std::move(other.handle_)), count_(other.count_) {
        other.count_ = 0;
    }
    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
        if (this != &other) {
            handle_ = std::move(other.handle_);
            count_ = other.count_;
            other.count_ = 0;
        }
        return *this;
    }

    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

    [[nodiscard]] T* get() const noexcept { return handle_.get(); }
    [[nodiscard]] size_t count() const noexcept { return count_; }
    [[nodiscard]] size_t bytes() const noexcept { return count_ * sizeof(T); }

    void copyFromHost(const T* src, size_t n = 0) {
        if (!handle_) return;
        assert(n == 0 || n <= count_);
        size_t copyCount = (n == 0) ? count_ : std::min(n, count_);
        CUDA_CHECK_THROW(cudaMemcpy(handle_.get(), src, copyCount * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyToHost(T* dst, size_t n = 0) const {
        if (!handle_) return;
        assert(n == 0 || n <= count_);
        size_t copyCount = (n == 0) ? count_ : std::min(n, count_);
        CUDA_CHECK_THROW(cudaMemcpy(dst, handle_.get(), copyCount * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void copyFromHostAsync(const T* src, cudaStream_t stream, size_t n = 0) {
        if (!handle_) return;
        assert(n == 0 || n <= count_);
        size_t copyCount = (n == 0) ? count_ : std::min(n, count_);
        CUDA_CHECK_THROW(cudaMemcpyAsync(handle_.get(), src, copyCount * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void copyToHostAsync(T* dst, cudaStream_t stream, size_t n = 0) const {
        if (!handle_) return;
        assert(n == 0 || n <= count_);
        size_t copyCount = (n == 0) ? count_ : std::min(n, count_);
        CUDA_CHECK_THROW(cudaMemcpyAsync(dst, handle_.get(), copyCount * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    // 注意：逐字节填充，与 std::memset 相同。对非零值仅 memset(0) 可安全用于所有类型。
    // Note: fills per-byte like std::memset. Only memset(0) is safe for all types.
    void memset(int value = 0) {
        if (!handle_) return;
        CUDA_CHECK_THROW(cudaMemset(handle_.get(), value, count_ * sizeof(T)));
    }

private:
    UniqueHandle<T*, nullptr, Deleter> handle_;
    size_t count_ = 0;
};

// ============================================================================
// CudaPinnedMemory
// ============================================================================

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
    struct Deleter { void operator()(T* p) const noexcept { cudaFreeHost(p); } };
public:
    explicit CudaPinnedMemory(size_t count, unsigned int flags = cudaHostAllocDefault)
        : count_(count) {
        T* p = nullptr;
        CUDA_CHECK_THROW(cudaHostAlloc(&p, count * sizeof(T), flags));
        handle_ = UniqueHandle<T*, nullptr, Deleter>(p);
    }

    CudaPinnedMemory(CudaPinnedMemory&& other) noexcept
        : handle_(std::move(other.handle_)), count_(other.count_) {
        other.count_ = 0;
    }
    CudaPinnedMemory& operator=(CudaPinnedMemory&& other) noexcept {
        if (this != &other) {
            handle_ = std::move(other.handle_);
            count_ = other.count_;
            other.count_ = 0;
        }
        return *this;
    }

    CudaPinnedMemory(const CudaPinnedMemory&) = delete;
    CudaPinnedMemory& operator=(const CudaPinnedMemory&) = delete;

    [[nodiscard]] T* get() const noexcept { return handle_.get(); }
    [[nodiscard]] size_t count() const noexcept { return count_; }
    [[nodiscard]] size_t bytes() const noexcept { return count_ * sizeof(T); }

    T& operator[](size_t i) { assert(i < count_); return handle_.get()[i]; }
    const T& operator[](size_t i) const { assert(i < count_); return handle_.get()[i]; }

private:
    UniqueHandle<T*, nullptr, Deleter> handle_;
    size_t count_ = 0;
};
