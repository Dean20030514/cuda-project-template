<#!
Synopsis: Enable CUDA (+ cuDNN) build environment in the current PowerShell session for ANY project.
- Prefers Visual Studio 2022 (stable) Developer environment; falls back to Insiders if needed.
- Imports VsDevCmd environment into THIS PowerShell session (not just a child cmd.exe).
- Ensures CUDA bin folders are on PATH for this session.
- Ensures cuDNN from the CUDA Toolkit install is preferred (sets CUDNN_ROOT=CUDA_PATH and prepends cuDNN bin dirs).
Usage:
  .\enable_cuda_env.ps1
Then in the same terminal:
  nvcc --version
  cl.exe /?
  # build your project...
#>

[CmdletBinding()]
param(
  # 可选：手动指定 CUDA 根目录（例如 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0）
  [string]$CudaPath,
  # 可选：手动指定 cuDNN 根目录（例如 C:\Program Files\NVIDIA\CUDNN）
  [string]$CuDnnRoot,
  # VS 开发者命令行架构（默认 amd64）
  [ValidateSet('amd64','x64','x86','arm64')]
  [string]$VsArch = 'amd64',
  # 安静模式：减少输出
  [switch]$Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# 导入公共 VS 辅助模块
$modulePath = Join-Path $PSScriptRoot '..\common\VsHelper.psm1'
if (Test-Path $modulePath) {
  Import-Module $modulePath -Force
} else {
  throw "VsHelper.psm1 not found at: $modulePath"
}

function Ensure-CudaOnPath {
  param([string]$Preferred)
  $cudaRoot = $Preferred
  if (-not $cudaRoot -or -not (Test-Path $cudaRoot)) {
    if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) { $cudaRoot = $env:CUDA_PATH }
  }
  if (-not $cudaRoot -or -not (Test-Path $cudaRoot)) {
    # Probe default install directory to pick highest version
    $base = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'
    if (Test-Path $base) {
      $versions = Get-ChildItem -Path $base -Directory | Sort-Object Name -Descending
      if ($versions.Count -gt 0) { $cudaRoot = $versions[0].FullName }
    }
  }
  if (-not $cudaRoot) { throw 'CUDA Toolkit not found. Please install CUDA Toolkit and reopen terminal.' }
  $bin = Join-Path $cudaRoot 'bin'
  $bin64 = Join-Path $cudaRoot 'bin\x64'
  foreach ($p in @($bin, $bin64)) {
    if ((Test-Path $p) -and (-not ($env:PATH -split ';' | Where-Object { $_ -ieq $p }))) {
      $env:PATH = "$p;" + $env:PATH
    }
  }
  $env:CUDA_PATH = $cudaRoot
  return $cudaRoot
}

function Ensure-CuDnnOnPath {
  param(
    [string]$Base,
    [string]$Override
  )
  $root = if ($Override -and (Test-Path $Override)) { $Override } else { $Base }
  if (-not $root -or -not (Test-Path $root)) { return }
  $env:CUDNN_ROOT = $root

  $bin = Join-Path $root 'bin'
  # 动态查找所有版本化的 bin 目录（如 bin/13.0, bin/12.9 等）
  $versionedBins = @()
  if (Test-Path $bin) {
    $versionedBins = Get-ChildItem -Path $bin -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match '^\d+\.\d+$' } |
      Sort-Object { [version]$_.Name } -Descending |
      Select-Object -ExpandProperty FullName
  }
  # 优先添加版本化目录，然后是主 bin 目录
  $cands = @($versionedBins) + @($bin)
  foreach ($p in $cands) {
    if ($p -and (Test-Path $p) -and (-not ($env:PATH -split ';' | Where-Object { $_ -ieq $p }))) {
      $env:PATH = "$p;" + $env:PATH
    }
  }
  if (-not $Quiet) {
    try {
      $dll = Get-ChildItem -Path $bin -Recurse -Filter 'cudnn64_*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
      if ($dll) { Write-Host "[enable-cuda-env] cuDNN DLL -> $dll" -ForegroundColor DarkCyan }
    } catch { }
  }
}

function Test-CuDnnRuntime {
  param([string]$Base = $env:CUDA_PATH)
  try {
    # 动态查找 cuDNN DLL
    $binDir = Join-Path $Base 'bin'
    $dllPath = $null
    if (Test-Path $binDir) {
      # 先在版本化子目录中查找
      $versionedDirs = Get-ChildItem -Path $binDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } -Descending
      foreach ($vdir in $versionedDirs) {
        $candidate = Get-ChildItem -Path $vdir.FullName -Filter 'cudnn64_*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($candidate) { $dllPath = $candidate.FullName; break }
      }
      # 如果版本化目录没找到，在主 bin 目录查找
      if (-not $dllPath) {
        $candidate = Get-ChildItem -Path $binDir -Filter 'cudnn64_*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($candidate) { $dllPath = $candidate.FullName }
      }
    }
    if (-not $dllPath) { throw 'cudnn64_9.dll not found under CUDA Toolkit bin.' }
    $dllEscaped = $dllPath -replace '\\','\\\\'
    $src = @"
using System;
using System.Runtime.InteropServices;
public static class CuDnnNativeCheck {
  [DllImport("$dllEscaped", EntryPoint="cudnnGetVersion", CallingConvention=CallingConvention.Cdecl)]
  public static extern long cudnnGetVersion();
}
"@
    Add-Type -TypeDefinition $src -Language CSharp -ErrorAction Stop | Out-Null
    $ver = [CuDnnNativeCheck]::cudnnGetVersion()
    $maj = [int]([math]::Floor($ver / 10000))
    $min = [int]([math]::Floor(($ver % 10000) / 100))
    $pat = [int]($ver % 100)
    Write-Host ("[enable-cuda-env] cuDNN runtime OK: v{0}.{1}.{2} (raw={3}) from {4}" -f $maj,$min,$pat,$ver,$dllPath) -ForegroundColor Green
    return $true
  } catch {
    Write-Host ("[enable-cuda-env] cuDNN runtime check failed: " + $_.Exception.Message) -ForegroundColor Yellow
    return $false
  }
}

function Test-CuDnnDevFiles {
  param([string]$Base = $env:CUDA_PATH)
  $foundH = $false; $foundLib = $false
  if ($Base -and (Test-Path $Base)) {
    # 动态查找 cudnn.h
    $includeDir = Join-Path $Base 'include'
    if (Test-Path $includeDir) {
      # 先查找版本化子目录
      $versionedIncludes = Get-ChildItem -Path $includeDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } -Descending
      foreach ($vdir in $versionedIncludes) {
        $hPath = Join-Path $vdir.FullName 'cudnn.h'
        if (Test-Path $hPath) { $foundH = $true; Write-Host "[enable-cuda-env] cuDNN header -> $hPath" -ForegroundColor DarkCyan; break }
      }
      # 如果版本化目录没找到，查找主 include 目录
      if (-not $foundH) {
        $hPath = Join-Path $includeDir 'cudnn.h'
        if (Test-Path $hPath) { $foundH = $true; Write-Host "[enable-cuda-env] cuDNN header -> $hPath" -ForegroundColor DarkCyan }
      }
    }

    # 动态查找 cudnn.lib
    $libDir = Join-Path $Base 'lib'
    if (Test-Path $libDir) {
      # 先查找版本化子目录
      $versionedLibs = Get-ChildItem -Path $libDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } -Descending
      foreach ($vdir in $versionedLibs) {
        $libPath = Join-Path $vdir.FullName 'x64/cudnn.lib'
        if (Test-Path $libPath) { $foundLib = $true; Write-Host "[enable-cuda-env] cuDNN import lib -> $libPath" -ForegroundColor DarkCyan; break }
      }
      # 如果版本化目录没找到，查找主 lib/x64 目录
      if (-not $foundLib) {
        $libPath = Join-Path $libDir 'x64/cudnn.lib'
        if (Test-Path $libPath) { $foundLib = $true; Write-Host "[enable-cuda-env] cuDNN import lib -> $libPath" -ForegroundColor DarkCyan }
      }
    }
  }
  if ($foundH -and $foundLib) {
    Write-Host "[enable-cuda-env] cuDNN dev files OK (headers + import lib)." -ForegroundColor Green
  } elseif ($foundH -or $foundLib) {
    Write-Host "[enable-cuda-env] cuDNN dev files partially found (headers or lib). Linking may fail." -ForegroundColor Yellow
  } else {
    Write-Host "[enable-cuda-env] cuDNN dev files not found under CUDA Toolkit. Linking will be disabled unless provided elsewhere." -ForegroundColor Yellow
  }
}

# 1) Import VS developer environment (Host x64)
$vsdev = Find-VsDevCmd
if (-not $Quiet) { Write-Host "[enable-cuda-env] Using VsDevCmd: $vsdev" -ForegroundColor Cyan }
Import-CmdEnvironment -CmdLine "call `"$vsdev`" -arch=$VsArch && set"

# 2) Ensure CUDA on PATH for this session (with hints for alternate layouts/WSL)
try {
  $resolvedCuda = Ensure-CudaOnPath -Preferred $CudaPath
  if (-not $Quiet) { Write-Host "[enable-cuda-env] CUDA_PATH = $env:CUDA_PATH" -ForegroundColor Cyan }

  # 2.5) Ensure cuDNN (from CUDA Toolkit) is preferred and on PATH
  Ensure-CuDnnOnPath -Base $env:CUDA_PATH -Override $CuDnnRoot
  if (-not $Quiet) { Write-Host "[enable-cuda-env] CUDNN_ROOT (preferred) = $env:CUDNN_ROOT" -ForegroundColor Cyan }
  # Runtime + dev checks for clearer confirmation
  $rtOk = Test-CuDnnRuntime
  Test-CuDnnDevFiles -Base $env:CUDA_PATH | Out-Null
}
catch {
  Write-Host "[enable-cuda-env] CUDA Toolkit not detected. Common location: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA" -ForegroundColor Yellow
  if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
    Write-Host "[enable-cuda-env] If you build inside WSL, install and use /usr/local/cuda in the Linux distro (e.g., Ubuntu: sudo apt install nvidia-cuda-toolkit) and verify nvcc in WSL." -ForegroundColor Yellow
  }
  Write-Host "[enable-cuda-env] You can also set this session manually: set CUDA_PATH=<your CUDA path>; add <CUDA_PATH>\\bin to PATH, then retry." -ForegroundColor Yellow
  throw
}

# 3) Quick diagnostics
if (-not $Quiet) {
  try { & nvcc --version | Select-Object -First 2 | ForEach-Object { Write-Host $_ } } catch { Write-Host 'nvcc not found on PATH' -ForegroundColor Yellow }
  try { & where.exe cl | Select-Object -First 1 | ForEach-Object { Write-Host "cl.exe -> $_" } } catch { Write-Host 'cl.exe not found (MSVC not on PATH?)' -ForegroundColor Yellow }
  Write-Host "[enable-cuda-env] CUDA build environment is ready for this PowerShell session." -ForegroundColor Green
}
