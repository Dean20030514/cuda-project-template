#include "cuda_raii.h"
#include "cuda_utils.h"
#include "gpu_test_helper.h"
#include <gtest/gtest.h>
#include <vector>

// ============================================================================
// CudaStream
// ============================================================================

TEST(CudaStream, CreateAndGet) {
    SKIP_IF_NO_GPU();
    CudaStream s;
    EXPECT_NE(s.get(), nullptr);
}

TEST(CudaStream, MoveTransfersOwnership) {
    SKIP_IF_NO_GPU();
    CudaStream a;
    cudaStream_t raw = a.get();
    CudaStream b(std::move(a));
    EXPECT_EQ(b.get(), raw);
    EXPECT_EQ(a.get(), nullptr);
}

TEST(CudaStream, QueryReturnsTrue) {
    SKIP_IF_NO_GPU();
    CudaStream s;
    // 空 stream 应立即完成
    // Empty stream should complete immediately
    s.synchronize();
    EXPECT_TRUE(s.query());
}

// ============================================================================
// CudaEvent
// ============================================================================

TEST(CudaEvent, CreateAndGet) {
    SKIP_IF_NO_GPU();
    CudaEvent e;
    EXPECT_NE(e.get(), nullptr);
}

TEST(CudaEvent, MoveTransfersOwnership) {
    SKIP_IF_NO_GPU();
    CudaEvent a;
    cudaEvent_t raw = a.get();
    CudaEvent b(std::move(a));
    EXPECT_EQ(b.get(), raw);
    EXPECT_EQ(a.get(), nullptr);
}

TEST(CudaEvent, RecordAndSynchronize) {
    SKIP_IF_NO_GPU();
    CudaEvent e;
    e.record();
    EXPECT_NO_THROW(e.synchronize());
}

// ============================================================================
// elapsedMs
// ============================================================================

TEST(ElapsedMs, NonNegative) {
    SKIP_IF_NO_GPU();
    CudaEvent start, stop;
    start.record();
    stop.record();
    float ms = elapsedMs(start, stop);
    EXPECT_GE(ms, 0.0f);
}

// ============================================================================
// CudaDeviceMemory
// ============================================================================

TEST(CudaDeviceMemory, AllocateAndProperties) {
    SKIP_IF_NO_GPU();
    const size_t N = 1024;
    CudaDeviceMemory<float> d(N);
    EXPECT_NE(d.get(), nullptr);
    EXPECT_EQ(d.count(), N);
    EXPECT_EQ(d.bytes(), N * sizeof(float));
}

TEST(CudaDeviceMemory, MoveTransfersOwnership) {
    SKIP_IF_NO_GPU();
    const size_t N = 512;
    CudaDeviceMemory<int> a(N);
    int* raw = a.get();
    CudaDeviceMemory<int> b(std::move(a));
    EXPECT_EQ(b.get(), raw);
    EXPECT_EQ(b.count(), N);
    // 源对象被清空 | Source object is emptied
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(a.count(), 0u);
}

TEST(CudaDeviceMemory, CopyRoundTrip) {
    SKIP_IF_NO_GPU();
    const size_t N = 256;
    std::vector<int> src(N), dst(N, 0);
    for (size_t i = 0; i < N; ++i) src[i] = static_cast<int>(i);

    CudaDeviceMemory<int> d(N);
    d.copyFromHost(src.data());
    d.copyToHost(dst.data());

    EXPECT_EQ(src, dst);
}

TEST(CudaDeviceMemory, PartialCopy) {
    SKIP_IF_NO_GPU();
    const size_t N = 256;
    const size_t half = N / 2;
    std::vector<float> src(N, 1.0f), dst(N, 0.0f);

    CudaDeviceMemory<float> d(N);
    d.memset(0);
    d.copyFromHost(src.data(), half);
    d.copyToHost(dst.data(), half);

    for (size_t i = 0; i < half; ++i)
        EXPECT_EQ(dst[i], 1.0f) << "index " << i;
}

TEST(CudaDeviceMemory, Memset) {
    SKIP_IF_NO_GPU();
    const size_t N = 128;
    std::vector<char> dst(N, 0xFF);

    CudaDeviceMemory<char> d(N);
    d.memset(0);
    d.copyToHost(dst.data());

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(dst[i], 0) << "index " << i;
}

// ============================================================================
// CudaPinnedMemory
// ============================================================================

TEST(CudaPinnedMemory, AllocateAndProperties) {
    SKIP_IF_NO_GPU();
    const size_t N = 1024;
    CudaPinnedMemory<float> h(N);
    EXPECT_NE(h.get(), nullptr);
    EXPECT_EQ(h.count(), N);
    EXPECT_EQ(h.bytes(), N * sizeof(float));
}

TEST(CudaPinnedMemory, MoveTransfersOwnership) {
    SKIP_IF_NO_GPU();
    const size_t N = 512;
    CudaPinnedMemory<int> a(N);
    int* raw = a.get();
    CudaPinnedMemory<int> b(std::move(a));
    EXPECT_EQ(b.get(), raw);
    EXPECT_EQ(b.count(), N);
    // 源对象被清空 | Source object is emptied
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(a.count(), 0u);
}

TEST(CudaPinnedMemory, OperatorBracket) {
    SKIP_IF_NO_GPU();
    const size_t N = 64;
    CudaPinnedMemory<int> h(N);
    for (size_t i = 0; i < N; ++i) h[i] = static_cast<int>(i * 2);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h[i], static_cast<int>(i * 2)) << "index " << i;
}

// ============================================================================
// Async copy round-trip (pinned + stream)
// ============================================================================

TEST(AsyncCopy, PinnedRoundTrip) {
    SKIP_IF_NO_GPU();
    const size_t N = 1024;
    CudaStream stream;
    CudaPinnedMemory<float> h_in(N), h_out(N);
    CudaDeviceMemory<float> d(N);

    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    d.copyFromHostAsync(h_in.get(), stream.get());
    d.copyToHostAsync(h_out.get(), stream.get());
    stream.synchronize();

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_out[i], static_cast<float>(i)) << "index " << i;
}
