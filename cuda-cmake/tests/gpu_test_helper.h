#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

inline bool hasGpuDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_GPU()                          \
    do {                                          \
        if (!hasGpuDevice())                      \
            GTEST_SKIP() << "No CUDA device";     \
    } while (0)
