cmake_minimum_required(VERSION 3.24)
project(CudaDemo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 自动探测本机 GPU 架构（需要 CMake ≥ 3.24）
# Auto-detect native GPU architecture (requires CMake >= 3.24)
set(CMAKE_CUDA_ARCHITECTURES native)

# 导出 compile_commands.json 供 clangd / IntelliSense 使用
# Export compile_commands.json for clangd / IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cudatest src/main.cu)

target_include_directories(cudatest PRIVATE ${CMAKE_SOURCE_DIR}/../common)

## ===== 构建优化选项 | Build Optimization Options =====
option(CUDA_ENABLE_FAST_MATH "Enable --use_fast_math for CUDA files in Release/RelWithDebInfo" ON)
option(MSVC_STATIC_RUNTIME "Link static MSVC runtime (/MT)" OFF)
option(ENABLE_CUBLAS "Enable cuBLAS integration" ON)
option(ENABLE_CUFFT "Enable cuFFT integration" ON)
option(BUILD_TESTING "Build unit tests (requires GoogleTest via FetchContent)" ON)

target_link_libraries(cudatest PRIVATE CUDA::cudart)

target_compile_options(cudatest PRIVATE
	# /utf-8 仅 MSVC 需要；GCC/Clang 默认 UTF-8
	# /utf-8 is MSVC-only; GCC/Clang default to UTF-8
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>

	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>

	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:
		-O3
		$<$<BOOL:${CUDA_ENABLE_FAST_MATH}>:--use_fast_math>
		# 将 -Xptxas 参数与等号连接，防止 CMake/Ninja 依赖文件标志解析冲突
		# Concat -Xptxas with equals sign to prevent CMake/Ninja dep flag parsing conflict
		-Xptxas=-O3
	>
)

if (MSVC AND MSVC_STATIC_RUNTIME)
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ===== cuBLAS 自动检测 | cuBLAS Auto-detection =====
if (ENABLE_CUBLAS AND TARGET CUDA::cublas)
	message(STATUS "cuBLAS found")
	target_link_libraries(cudatest PRIVATE CUDA::cublas)
	target_compile_definitions(cudatest PRIVATE HAVE_CUBLAS=1)
else()
	message(STATUS "cuBLAS not found or disabled; building without cuBLAS demo.")
endif()

# ===== cuFFT 自动检测 | cuFFT Auto-detection =====
if (ENABLE_CUFFT AND TARGET CUDA::cufft)
	message(STATUS "cuFFT found")
	target_link_libraries(cudatest PRIVATE CUDA::cufft)
	target_compile_definitions(cudatest PRIVATE HAVE_CUFFT=1)
else()
	message(STATUS "cuFFT not found or disabled; building without cuFFT demo.")
endif()

# ===== cuDNN 自动检测 | cuDNN Auto-detection =====
# 收集候选搜索目录，支持多种安装布局
# Gather candidate search dirs, supporting various install layouts
set(CUDNN_HINT_DIRS
	"$ENV{CUDA_PATH}"
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
	"$ENV{CUDNN_ROOT}"
	"C:/Program Files/NVIDIA/CUDNN"
)

# 动态收集所有 CUDA_PATH_V* 环境变量指向的路径
# Dynamically collect paths from all CUDA_PATH_V* environment variables
if (DEFINED ENV{CUDA_PATH})
	list(APPEND CUDNN_HINT_DIRS "$ENV{CUDA_PATH}")
endif()

# 匹配版本化子目录 v*（如 CUDNN/v9.14）
# Match versioned subdirs v* (e.g. CUDNN/v9.14)
file(GLOB CUDNN_VERSION_DIRS
	"$ENV{CUDA_PATH}/v*"
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*"
	"$ENV{CUDNN_ROOT}/v*"
	"C:/Program Files/NVIDIA/CUDNN/v*"
)

# 动态查找所有版本化的 cuDNN lib 目录（支持 cuDNN 9.x+ 的 CUDA 版本子目录结构）
# Dynamically find all versioned cuDNN lib dirs (supports cuDNN 9.x+ CUDA version subdir structure)
file(GLOB CUDNN_LIB_VERSIONS
	"$ENV{CUDA_PATH}/lib/*/x64"
	"$ENV{CUDNN_ROOT}/lib/*/x64"
	"C:/Program Files/NVIDIA/CUDNN/lib/*/x64"
)

# cuDNN 9.x+ 新结构: CUDNN/v9.x/lib/CUDA_VER/x64
file(GLOB _CUDNN_V_DIRS "C:/Program Files/NVIDIA/CUDNN/v*")
foreach(_vdir ${_CUDNN_V_DIRS})
	file(GLOB _CUDNN_LIB_SUBDIRS "${_vdir}/lib/*/x64")
	list(APPEND CUDNN_LIB_VERSIONS ${_CUDNN_LIB_SUBDIRS})
endforeach()

find_library(CUDNN_LIBRARY
	NAMES cudnn
	HINTS ${CUDNN_HINT_DIRS} ${CUDNN_VERSION_DIRS} ${CUDNN_LIB_VERSIONS}
	PATH_SUFFIXES
		lib
		lib/x64
)

# 动态查找所有版本化的 cuDNN include 目录
# Dynamically find all versioned cuDNN include dirs
file(GLOB CUDNN_INCLUDE_VERSIONS
	"$ENV{CUDA_PATH}/include/*"
	"$ENV{CUDNN_ROOT}/include/*"
)

foreach(_vdir ${_CUDNN_V_DIRS})
	file(GLOB _CUDNN_INC_SUBDIRS "${_vdir}/include/*")
	list(APPEND CUDNN_INCLUDE_VERSIONS ${_CUDNN_INC_SUBDIRS})
endforeach()

find_path(CUDNN_INCLUDE_DIR
	NAMES cudnn.h
	HINTS ${CUDNN_HINT_DIRS} ${CUDNN_VERSION_DIRS} ${CUDNN_INCLUDE_VERSIONS}
	PATH_SUFFIXES
		include
)

if (CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
	message(STATUS "cuDNN found: lib=${CUDNN_LIBRARY}, include=${CUDNN_INCLUDE_DIR}")
	target_include_directories(cudatest PRIVATE ${CUDNN_INCLUDE_DIR})
	target_link_libraries(cudatest PRIVATE ${CUDNN_LIBRARY})
	target_compile_definitions(cudatest PRIVATE HAVE_CUDNN=1)
else()
	message(WARNING "cuDNN not found; building without cuDNN demo. Set CUDNN_ROOT if needed.")
endif()

# 可选：NVTX 标记，便于 Nsight Systems/Compute 分析
# Optional: NVTX markers for Nsight Systems/Compute profiling
if (TARGET CUDA::nvtx3)
	target_link_libraries(cudatest PRIVATE CUDA::nvtx3)
	target_compile_definitions(cudatest PRIVATE HAVE_NVTX=1)
endif()

# ===== 构建摘要 | Build Summary =====
message(STATUS "")
message(STATUS "=== CUDA Demo Build Configuration ===")
message(STATUS "  CUDA Architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ Standard       : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Fast Math          : ${CUDA_ENABLE_FAST_MATH}")
if (CUDNN_LIBRARY)
	message(STATUS "  cuDNN              : YES")
else()
	message(STATUS "  cuDNN              : NO")
endif()
if (TARGET CUDA::cublas AND ENABLE_CUBLAS)
	message(STATUS "  cuBLAS             : YES")
else()
	message(STATUS "  cuBLAS             : NO")
endif()
if (TARGET CUDA::cufft AND ENABLE_CUFFT)
	message(STATUS "  cuFFT              : YES")
else()
	message(STATUS "  cuFFT              : NO")
endif()
if (TARGET CUDA::nvtx3)
	message(STATUS "  NVTX               : YES")
else()
	message(STATUS "  NVTX               : NO")
endif()
message(STATUS "=====================================")
message(STATUS "")

if (BUILD_TESTING)
	enable_testing()
	add_subdirectory(tests)
endif()

install(TARGETS cudatest RUNTIME DESTINATION bin)
