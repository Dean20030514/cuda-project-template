#include "cuda_check.h"
#include <gtest/gtest.h>

// ============================================================================
// CUDA_CHECK_THROW
// ============================================================================

TEST(CudaCheckThrow, SuccessDoesNotThrow) {
    EXPECT_NO_THROW(CUDA_CHECK_THROW(cudaSuccess));
}

TEST(CudaCheckThrow, ErrorThrowsRuntimeError) {
    EXPECT_THROW(CUDA_CHECK_THROW(cudaErrorInvalidValue), std::runtime_error);
}

TEST(CudaCheckThrow, ErrorMessageContainsInfo) {
    try {
        CUDA_CHECK_THROW(cudaErrorMemoryAllocation);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        // 消息应包含文件名和错误描述
        // Message should contain filename and error description
        EXPECT_NE(msg.find("test_error_macros"), std::string::npos)
            << "Message missing filename: " << msg;
        EXPECT_NE(msg.find("CUDA error"), std::string::npos)
            << "Message missing 'CUDA error': " << msg;
    }
}

// ============================================================================
// cublasGetStatusString (if available)
// ============================================================================

#ifdef HAVE_CUBLAS
#include "cuda_cublas.h"

TEST(CublasStatusString, KnownCodes) {
    EXPECT_STREQ(cublasGetStatusString(CUBLAS_STATUS_SUCCESS), "CUBLAS_STATUS_SUCCESS");
    EXPECT_STREQ(cublasGetStatusString(CUBLAS_STATUS_NOT_INITIALIZED), "CUBLAS_STATUS_NOT_INITIALIZED");
    EXPECT_STREQ(cublasGetStatusString(CUBLAS_STATUS_ALLOC_FAILED), "CUBLAS_STATUS_ALLOC_FAILED");
    EXPECT_STREQ(cublasGetStatusString(CUBLAS_STATUS_INVALID_VALUE), "CUBLAS_STATUS_INVALID_VALUE");
    EXPECT_STREQ(cublasGetStatusString(CUBLAS_STATUS_EXECUTION_FAILED), "CUBLAS_STATUS_EXECUTION_FAILED");
    EXPECT_STREQ(cublasGetStatusString(CUBLAS_STATUS_INTERNAL_ERROR), "CUBLAS_STATUS_INTERNAL_ERROR");
    EXPECT_STREQ(cublasGetStatusString(CUBLAS_STATUS_NOT_SUPPORTED), "CUBLAS_STATUS_NOT_SUPPORTED");
}

TEST(CublasStatusString, UnknownCodeReturnsFallback) {
    EXPECT_STREQ(cublasGetStatusString(static_cast<cublasStatus_t>(9999)), "CUBLAS_STATUS_UNKNOWN");
}

TEST(CublasCheckThrow, SuccessDoesNotThrow) {
    EXPECT_NO_THROW(CUBLAS_CHECK_THROW(CUBLAS_STATUS_SUCCESS));
}

TEST(CublasCheckThrow, ErrorThrowsRuntimeError) {
    EXPECT_THROW(CUBLAS_CHECK_THROW(CUBLAS_STATUS_INVALID_VALUE), std::runtime_error);
}
#endif // HAVE_CUBLAS

// ============================================================================
// cufftGetResultString (if available)
// ============================================================================

#ifdef HAVE_CUFFT
#include "cuda_cufft.h"

TEST(CufftResultString, KnownCodes) {
    EXPECT_STREQ(cufftGetResultString(CUFFT_SUCCESS), "CUFFT_SUCCESS");
    EXPECT_STREQ(cufftGetResultString(CUFFT_INVALID_PLAN), "CUFFT_INVALID_PLAN");
    EXPECT_STREQ(cufftGetResultString(CUFFT_ALLOC_FAILED), "CUFFT_ALLOC_FAILED");
    EXPECT_STREQ(cufftGetResultString(CUFFT_INVALID_VALUE), "CUFFT_INVALID_VALUE");
    EXPECT_STREQ(cufftGetResultString(CUFFT_EXEC_FAILED), "CUFFT_EXEC_FAILED");
}

TEST(CufftResultString, UnknownCodeReturnsFallback) {
    EXPECT_STREQ(cufftGetResultString(static_cast<cufftResult>(9999)), "CUFFT_UNKNOWN");
}

TEST(CufftCheckThrow, SuccessDoesNotThrow) {
    EXPECT_NO_THROW(CUFFT_CHECK_THROW(CUFFT_SUCCESS));
}

TEST(CufftCheckThrow, ErrorThrowsRuntimeError) {
    EXPECT_THROW(CUFFT_CHECK_THROW(CUFFT_INVALID_VALUE), std::runtime_error);
}
#endif // HAVE_CUFFT
