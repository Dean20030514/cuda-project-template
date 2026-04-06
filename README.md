# CUDA Project Template

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Windows](https://img.shields.io/badge/Platform-Windows-blue.svg)](https://www.microsoft.com/windows)
[![VS Code](https://img.shields.io/badge/IDE-VS%20Code-007ACC.svg)](https://code.visualstudio.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

开箱即用的 CUDA 项目模板，支持 Windows + VS Code 开发环境。

An out-of-the-box CUDA project template supporting Windows + VS Code development environment.

## Features | 特性

- **两种方案 | Two Options**: 单文件 nvcc 编译 / CMake 标准工程 | Single-file nvcc / CMake project
- **自动配置 | Auto Configuration**: 自动探测 GPU 架构，无需手动设置 | Automatic GPU architecture detection
- **一键运行 | One-Click Run**: VS Code 任务或 PowerShell 脚本 | VS Code tasks or PowerShell scripts
- **CUDA 库集成 | CUDA Libraries**: 自动检测 cuDNN / cuBLAS / cuFFT | Auto-detect cuDNN / cuBLAS / cuFFT
- **VS 兼容 | VS Compatible**: 支持 VS 2022/2026，自动处理兼容性 | Supports VS 2022/2026
- **RAII 工具库 | RAII Utilities**: Stream、Event、设备内存、锁页内存、cuDNN handle | Stream, Event, Device/Pinned Memory, cuDNN handles
- **带宽测试 | Bandwidth Test**: 内置 H2D/D2H 带宽测量工具 | Built-in H2D/D2H bandwidth measurement
- **结果验证 | Result Verification**: 自动校验 kernel 计算结果 | Automatic kernel result verification

## Requirements | 环境要求

| 组件 Component | 要求 Requirement |
|----------------|------------------|
| CUDA Toolkit | >= 12.0 |
| CMake | >= 3.24 |
| Visual Studio | 2022 / 2026 Build Tools |
| VS Code 扩展 Extensions | C/C++, CMake Tools |

> **说明 Note**：已在 CUDA 12.8 和 13.0 上测试；通过 `CUDART_VERSION` 条件编译自动适配不同版本。消费级 Blackwell（sm_120 / RTX 50）需 **CUDA 12.8+** 工具链。
>
> **Note**: Tested with CUDA 12.8 and 13.0; automatically adapts via `CUDART_VERSION` conditional compilation. Consumer Blackwell (sm_120 / RTX 50) requires **CUDA 12.8+**.

验证环境 | Verify environment:

```powershell
nvcc --version    # 应显示 CUDA 版本 | Should display CUDA version
nvidia-smi        # 应显示 GPU 信息 | Should display GPU info
```

## Quick Start | 快速开始

### Option A | 方案 A: single-nvcc (单文件，快速上手 | Single file, quick start)

```powershell
# Windows (PowerShell)
cd single-nvcc
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_and_run.ps1
```

```bash
# Linux / WSL
cd single-nvcc
bash scripts/build_and_run.sh
```

### Option B | 方案 B: cuda-cmake (CMake，推荐日常开发 | CMake, recommended)

```powershell
# Windows (PowerShell)
cd cuda-cmake
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\configure_build_run.ps1
```

```bash
# Linux / WSL
cd cuda-cmake
bash scripts/configure_build_run.sh
```

### VS Code

1. 打开仓库目录 | Open the repository directory
2. `Ctrl+Shift+P` -> "Tasks: Run Task" -> 选择任务 | Select a task

## Expected Output | 预期输出

运行成功后，终端输出类似以下内容（具体数值因 GPU 型号而异）：

After a successful run, terminal output looks similar to this (exact values vary by GPU):

```
=== GPU Device 0: NVIDIA GeForce RTX 5070 Laptop GPU ===
  Compute capability : 12.0
  SM count           : 36
  Global memory      : 8.0 GiB
  Shared mem/block   : 48.0 KiB
  Max threads/block  : 1024
  Warp size          : 32
  Clock rate (core)  : 1545 MHz
  Clock rate (mem)   : 12001 MHz
  Memory bus width   : 128-bit
  L2 cache size      : 32768 KB
  ECC enabled        : No
  Async engines      : 1
==============================
=== Bandwidth Test (64 MiB, avg of 5 runs) ===
  H2D : 28.53 GB/s (2.35 ms)
  D2H : 28.78 GB/s (2.33 ms)
==============================

--- Kernel Demo ---
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ...  (N=1048576)
  kernel elapsed: 0.307 ms
  verification:   PASSED

--- Async Stream Demo ---
  Async round-trip 1.0 MB: 0.155 ms -> PASSED

=== All demos completed. ===
```

> 关键指标：所有 `verification` 应显示 **PASSED**，退出码为 0。
>
> Key indicators: all `verification` lines should show **PASSED** and the exit code should be 0.

## Project Structure | 项目结构

```
cuda-project-template/
├── common/
│   ├── cuda_helper.h              # 统一头文件（umbrella）| Umbrella include for backward compat
│   ├── cuda_check.h               # 错误检查宏 | Error checking macros (CUDA_CHECK, _THROW)
│   ├── cuda_raii.h                # UniqueHandle + RAII 类 | UniqueHandle + RAII wrappers
│   ├── cuda_utils.h               # 工具函数 | Utilities (calcGridSize, printDeviceInfo, ...)
│   ├── cuda_demos.cuh             # 共享 demo 函数 | Shared demo functions
│   ├── cuda_cudnn.h               # cuDNN 包装 | cuDNN wrappers (optional)
│   ├── cuda_cublas.h              # cuBLAS 包装 | cuBLAS wrappers (optional)
│   └── cuda_cufft.h               # cuFFT 包装 | cuFFT wrappers (optional)
├── single-nvcc/                   # Option A
│   ├── main.cu
│   └── scripts/
│       ├── build_and_run.ps1      # Windows 构建脚本 | Windows build script
│       └── build_and_run.sh       # Linux 构建脚本 | Linux build script
├── cuda-cmake/                    # Option B
│   ├── CMakeLists.txt
│   ├── src/main.cu
│   └── scripts/
│       ├── configure_build_run.ps1  # Windows 构建脚本 | Windows build script
│       └── configure_build_run.sh   # Linux 构建脚本 | Linux build script
└── scripts/
    ├── common/
    │   ├── VsHelper.psm1                 # VS 开发环境辅助模块 | VS dev env helper module (Windows)
    │   └── cuda_common.sh                # Linux 公共工具函数 | Linux shared utilities
    └── global/                            # 全局环境配置 (Windows) | Global env configuration (Windows)
        ├── enable_cuda_env.ps1            # 临时启用 | Temporary enable
        ├── install_ecuda_alias.ps1        # 安装快捷命令 | Install shortcut command
        ├── install_cuda_env_persistent.ps1  # 持久化安装 | Persistent installation
        └── remove_cuda_env_persistent.ps1   # 卸载持久化 | Remove persistent installation
```

## Build Options | 构建选项

### Configuration Mode | 配置模式

```powershell
# Debug（默认，含调试信息）| Debug (default, with debug info)
-Configuration Debug

# Release（优化编译）| Release (optimized compilation)
-Configuration Release
```

### GPU Architecture | GPU 架构

```powershell
# 自动探测（默认）| Auto detection (default)
# 不指定 -CudaArch/-Sm 参数 | Do not specify -CudaArch/-Sm parameter

# 指定架构 | Specify architecture
# 数据中心 Blackwell (B100/B200, sm_100) | Datacenter Blackwell
-CudaArch "100"   # cuda-cmake
-Sm 100           # single-nvcc
# 消费级 RTX 50 系列 (sm_120，勿与 sm_100 混用) | Consumer RTX 50 (sm_120, not sm_100)
-CudaArch "120"   # cuda-cmake
-Sm 120           # single-nvcc
```

### Performance Optimization | 性能优化

```powershell
# 启用 FastMath（牺牲精度换速度）| Enable FastMath (trade precision for speed)
-FastMath
```

### Complete Examples | 完整示例

```powershell
# Windows (PowerShell)
# cuda-cmake: Release + RTX 50 + FastMath
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\configure_build_run.ps1 -Configuration Release -CudaArch "120" -FastMath

# single-nvcc: Release + RTX 50 + FastMath
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_and_run.ps1 -Configuration Release -Sm 120 -FastMath
```

```bash
# Linux / WSL
# cuda-cmake: Release + RTX 50 + FastMath
bash scripts/configure_build_run.sh -c Release -a 120 -f

# single-nvcc: Release + RTX 50 + FastMath
bash scripts/build_and_run.sh -c Release -s 120 -f
```

## Global Environment | 全局环境配置

### Temporary Enable | 临时启用（当前会话）

```powershell
.\scripts\global\enable_cuda_env.ps1
```

### Install ecuda Shortcut | 安装 ecuda 快捷命令

```powershell
.\scripts\global\install_ecuda_alias.ps1

# 之后新终端直接使用 | Then use directly in new terminal
ecuda
```

### Persistent Installation | 持久化安装

```powershell
# 管理员（所有用户）| Administrator (all users)
.\scripts\global\install_cuda_env_persistent.ps1 -Scope Machine

# 当前用户 | Current user
.\scripts\global\install_cuda_env_persistent.ps1 -Scope User
```

## GPU Architecture Reference | GPU 架构参考

| GPU Series | Architecture | SM | Example Models |
|------------|-------------|-----|----------------|
| GTX 16 / RTX 20 | Turing | 75 | GTX 1660, RTX 2080 |
| RTX 30 | Ampere | 86 | RTX 3060, 3080, 3090 |
| RTX 40 | Ada Lovelace | 89 | RTX 4060, 4080, 4090 |
| H100 | Hopper | 90 | H100 |
| H100 SXM | Hopper (SXM) | 90a | H100 SXM (SM-specific features) |
| B100 / B200 | Blackwell (DC) | 100 | B100, B200, GB200 |
| RTX 50 | Blackwell | 120 | RTX 5070, 5080, 5090 |

> **注意 Note**：`sm_90a` 是 H100 SXM 专用的 SM 变体，包含 SXM 独有的硬件特性（如 TMA）。PCIe 版 H100 使用 `sm_90`。
>
> **Note**: `sm_90a` is an SXM-exclusive SM variant for H100 SXM with SXM-only hardware features (e.g. TMA). PCIe H100 uses `sm_90`.

> 使用 `nvidia-smi` 查看 GPU 型号，对照上表选择 SM 值。脚本默认会自动探测本机 GPU 架构。
>
> Use `nvidia-smi` to check GPU model and select SM value from the table above. Scripts auto-detect native GPU architecture by default.

## cuda_helper.h API

### Macros | 宏

| Macro | Description |
|-------|-------------|
| `CUDA_CHECK(call)` | CUDA API 错误检查（exit）| CUDA API error checking (exit) |
| `CUDA_CHECK_THROW(call)` | CUDA API 错误检查（异常）| CUDA API error checking (exception) |
| `CUDA_CHECK_KERNEL()` | Kernel 错误检查（exit）| Kernel error checking (exit) |
| `CUDA_CHECK_KERNEL_THROW()` | Kernel 错误检查（异常）| Kernel error checking (exception) |
| `CUDNN_CHECK(call)` | cuDNN 错误检查 | cuDNN error checking (requires `HAVE_CUDNN`) |
| `CUBLAS_CHECK(call)` | cuBLAS 错误检查 | cuBLAS error checking (requires `HAVE_CUBLAS`) |
| `CUFFT_CHECK(call)` | cuFFT 错误检查 | cuFFT error checking (requires `HAVE_CUFFT`) |

### RAII Classes | RAII 类

| Class | Description |
|-------|-------------|
| `UniqueHandle<T, Null, Deleter>` | 通用 RAII handle 包装（所有 handle 类的基础）| Generic RAII handle wrapper (base for all handle classes) |
| `CudaStream` | RAII 流包装，支持非阻塞流和查询 | RAII stream wrapper with non-blocking and query support |
| `CudaEvent` | RAII 事件包装，支持计时 | RAII event wrapper with timing |
| `CudaDeviceMemory<T>` | RAII 设备内存，支持同步/异步拷贝 | RAII device memory with sync/async copy |
| `CudaPinnedMemory<T>` | RAII 锁页主机内存，支持高速传输 | RAII pinned host memory for fast transfers |
| `CublasHandle` | RAII cuBLAS handle (requires `HAVE_CUBLAS`) |
| `CufftPlan` | RAII cuFFT plan, 支持 1D/2D (requires `HAVE_CUFFT`) |
| `CudnnHandle` | RAII cuDNN handle (requires `HAVE_CUDNN`) |
| `CudnnTensorDescriptor` | RAII cuDNN tensor descriptor (requires `HAVE_CUDNN`) |

> **错误处理差异 Note on error handling**：核心类（`CudaStream`、`CudaEvent`、`CudaDeviceMemory`、`CudaPinnedMemory`）使用 **throw-style**（`CUDA_CHECK_THROW`），可被 try/catch 捕获。库包装类（`CublasHandle`、`CufftPlan`、`CudnnHandle`、`CudnnTensorDescriptor`）使用 **exit-style**（`CUBLAS_CHECK` / `CUFFT_CHECK` / `CUDNN_CHECK`），初始化失败会直接 `exit(EXIT_FAILURE)`，不会抛出异常。
>
> **Note**: Core classes (`CudaStream`, `CudaEvent`, `CudaDeviceMemory`, `CudaPinnedMemory`) use **throw-style** error handling (`CUDA_CHECK_THROW`) and can be caught with try/catch. Library wrapper classes (`CublasHandle`, `CufftPlan`, `CudnnHandle`, `CudnnTensorDescriptor`) use **exit-style** macros (`CUBLAS_CHECK` / `CUFFT_CHECK` / `CUDNN_CHECK`) — initialization failures call `exit(EXIT_FAILURE)` instead of throwing.

### Utility Functions | 工具函数

| Function | Description |
|----------|-------------|
| `calcGridSize(n, block)` | 计算 grid 大小 | Calculate grid size |
| `printDeviceInfo(device)` | 打印 GPU 设备信息 | Print GPU device info |
| `elapsedMs(start, end)` | 测量两个 `CudaEvent` 间的耗时（ms，阻塞）| Elapsed ms between two `CudaEvent`s (blocking) |
| `measureBandwidth(size)` | 测量 H2D/D2H 传输带宽 | Measure H2D/D2H bandwidth |

## FAQ | 常见问题

<details>
<summary><b>nvcc 不是内部或外部命令 | nvcc is not recognized</b></summary>

将 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` 加入 PATH，重启终端。

Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` to PATH and restart the terminal.

</details>

<details>
<summary><b>cl.exe not found</b></summary>

安装 VS 2022/2026 Build Tools，或使用 Developer Command Prompt 打开 VS Code。

Install VS 2022/2026 Build Tools, or use Developer Command Prompt to open VS Code.

</details>

<details>
<summary><b>算力不匹配 | Compute capability mismatch</b></summary>

修改 `-gencode` / `CMAKE_CUDA_ARCHITECTURES` / `-Sm` 为对应的 SM 值。

Modify `-gencode` / `CMAKE_CUDA_ARCHITECTURES` / `-Sm` to the corresponding SM value.

</details>

<details>
<summary><b>WSL2 支持 | WSL2 Support</b></summary>

1. Windows 安装支持 WSL 的 NVIDIA 驱动 | Install WSL-compatible NVIDIA driver on Windows
2. WSL 中安装 cuda-toolkit | Install cuda-toolkit in WSL
3. 使用 GCC/CMake 构建 | Build with GCC/CMake

</details>

<details>
<summary><b>清理并重新构建 | Clean rebuild</b></summary>

```powershell
# cuda-cmake: 使用 -Clean 参数 | Use -Clean flag
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\configure_build_run.ps1 -Clean

# single-nvcc: 删除 build 目录 | Delete build directory
Remove-Item -Recurse -Force single-nvcc\build
```

</details>

<details>
<summary><b>cuBLAS / cuFFT 未检测到 | cuBLAS / cuFFT not detected</b></summary>

确保 CUDA Toolkit 安装完整（包含 cuBLAS 和 cuFFT 组件）。CMake 会自动检测并链接。

Ensure CUDA Toolkit is fully installed (including cuBLAS and cuFFT components). CMake auto-detects and links them.

</details>

## License

[MIT License](LICENSE)

## Related Links | 相关链接

- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [cuDNN Download](https://developer.nvidia.com/cudnn)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [VS Code CUDA Debugging](https://developer.nvidia.com/nsight-visual-studio-code-edition)
