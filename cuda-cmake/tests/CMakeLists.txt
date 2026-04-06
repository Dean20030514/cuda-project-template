include(FetchContent)
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.15.2
)
# Windows: 防止 GoogleTest 覆盖父项目的编译器/链接器设置
# Windows: prevent GoogleTest from overriding parent project compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(cuda_tests
	test_utils.cpp
	test_error_macros.cpp
	test_raii.cpp
)

target_include_directories(cuda_tests PRIVATE ${CMAKE_SOURCE_DIR}/../common)
target_link_libraries(cuda_tests PRIVATE GTest::gtest_main CUDA::cudart)

# 传递与主目标相同的库编译宏，使 #ifdef 测试能编译
# Pass same library compile defs as main target so #ifdef tests compile
if (ENABLE_CUBLAS AND TARGET CUDA::cublas)
	target_link_libraries(cuda_tests PRIVATE CUDA::cublas)
	target_compile_definitions(cuda_tests PRIVATE HAVE_CUBLAS=1)
endif()

if (ENABLE_CUFFT AND TARGET CUDA::cufft)
	target_link_libraries(cuda_tests PRIVATE CUDA::cufft)
	target_compile_definitions(cuda_tests PRIVATE HAVE_CUFFT=1)
endif()

if (CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
	target_include_directories(cuda_tests PRIVATE ${CUDNN_INCLUDE_DIR})
	target_link_libraries(cuda_tests PRIVATE ${CUDNN_LIBRARY})
	target_compile_definitions(cuda_tests PRIVATE HAVE_CUDNN=1)
endif()

# DISCOVERY_MODE PRE_TEST: 延迟到 ctest 运行时发现测试，避免在无 GPU 的
# CI 容器中配置阶段执行测试二进制失败。
# PRE_TEST defers test discovery to ctest runtime, avoiding failures when
# the test binary cannot run during cmake configure (e.g. GPU-less CI).
gtest_discover_tests(cuda_tests DISCOVERY_MODE PRE_TEST)
