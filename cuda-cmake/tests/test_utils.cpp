#include "cuda_utils.h"
#include <gtest/gtest.h>

// ============================================================================
// calcGridSize
// ============================================================================

TEST(CalcGridSize, ZeroElements) {
    EXPECT_EQ(calcGridSize(size_t(0), 256u), 0u);
}

TEST(CalcGridSize, SingleElement) {
    EXPECT_EQ(calcGridSize(size_t(1), 256u), 1u);
}

TEST(CalcGridSize, ExactMultiple) {
    EXPECT_EQ(calcGridSize(size_t(256), 256u), 1u);
    EXPECT_EQ(calcGridSize(size_t(512), 256u), 2u);
    EXPECT_EQ(calcGridSize(size_t(1024), 256u), 4u);
}

TEST(CalcGridSize, NonMultipleRoundsUp) {
    EXPECT_EQ(calcGridSize(size_t(1), 256u), 1u);
    EXPECT_EQ(calcGridSize(size_t(255), 256u), 1u);
    EXPECT_EQ(calcGridSize(size_t(257), 256u), 2u);
    EXPECT_EQ(calcGridSize(size_t(511), 256u), 2u);
    EXPECT_EQ(calcGridSize(size_t(513), 256u), 3u);
}

TEST(CalcGridSize, LargeValues) {
    // 1M elements, 256 threads -> 4096 blocks
    EXPECT_EQ(calcGridSize(size_t(1 << 20), 256u), 4096u);
    // 1M + 1 -> 4097
    EXPECT_EQ(calcGridSize(size_t((1 << 20) + 1), 256u), 4097u);
}

TEST(CalcGridSize, BlockSizeOne) {
    EXPECT_EQ(calcGridSize(size_t(100), 1u), 100u);
}

TEST(CalcGridSize, BlockSizeMatchesN) {
    EXPECT_EQ(calcGridSize(size_t(128), 128u), 1u);
}
