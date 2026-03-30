param(
    [switch]$BuildOnly,
    [ValidateSet('Debug','Release')]
    [string]$Configuration = 'Debug',
    # 指定 SM 架构（如 89、86、75）。不指定时默认生成通用 fatbin (sm_90/89/86/75 + PTX)。
    [string]$Sm,
    # 启用 --use_fast_math（Release/RelWithDebInfo 建议开启）
    [switch]$FastMath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# 导入公共模块
$modulePath = Join-Path $PSScriptRoot '..\..\scripts\common\VsHelper.psm1'
if (Test-Path $modulePath) {
    Import-Module $modulePath -Force
} else {
    throw "VsHelper.psm1 not found at: $modulePath"
}

$projectDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $projectDir
try {
    if (-not (Test-Path 'build')) { New-Item -ItemType Directory -Path 'build' | Out-Null }

    $vsdev = Find-VsDevCmd
    $cudaAllowFlag = Get-CudaUnsupportedCompilerFlag -VsDevCmdPath $vsdev

    # 生成 -gencode 片段
    if ($Sm) {
        $arch = $Sm.Trim()
        $gencode = "-gencode=arch=compute_${arch},code=sm_${arch} -gencode=arch=compute_${arch},code=compute_${arch}"
    } else {
        $gencode = @(
            '-gencode=arch=compute_100,code=sm_100',
            '-gencode=arch=compute_90,code=sm_90',
            '-gencode=arch=compute_89,code=sm_89',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_100,code=compute_100'
        ) -join ' '
    }

    $common = "nvcc -std=c++17 -Xcompiler /utf-8 ${gencode} main.cu -o build\main.exe"
    if ($Configuration -eq 'Release') {
        $compileCmd = "$common -O3 -DNDEBUG"
        if ($FastMath) { $compileCmd = "$compileCmd --use_fast_math -Xptxas -O3" }
    } else {
        $compileCmd = "$common -g -G"
    }
    if (-not [string]::IsNullOrEmpty($cudaAllowFlag)) { $compileCmd = "$compileCmd $cudaAllowFlag" }
    if ($BuildOnly) {
        $cmd = 'call "{0}" -arch=amd64 && {1}' -f $vsdev, $compileCmd
    } else {
        $cmd = 'call "{0}" -arch=amd64 && {1} && build\main.exe' -f $vsdev, $compileCmd
    }

    Write-Host "[build_and_run] Configuration: $Configuration" -ForegroundColor Yellow
    Write-Host "[build_and_run] Using VsDevCmd: $vsdev" -ForegroundColor Cyan
    if ($Sm) { Write-Host "[build_and_run] Target SM: $Sm" -ForegroundColor Yellow }
    else { Write-Host "[build_and_run] Target: multi-arch fatbin" -ForegroundColor Yellow }
    if ($FastMath) { Write-Host "[build_and_run] FastMath: ON" -ForegroundColor Yellow }
    if (-not [string]::IsNullOrEmpty($cudaAllowFlag)) { Write-Host "[build_and_run] Using CUDA flag: $cudaAllowFlag (VS not 2022)" -ForegroundColor DarkYellow }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    cmd.exe /c $cmd
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    $elapsed = $sw.Elapsed

    if ($exitCode -eq 0) {
        Write-Host ("[build_and_run] Completed in {0:mm\:ss\.ff}" -f $elapsed) -ForegroundColor Green
    } else {
        Write-Host ("[build_and_run] Failed (exit code $exitCode) after {0:mm\:ss\.ff}" -f $elapsed) -ForegroundColor Red
        exit $exitCode
    }
}
finally {
    Pop-Location
}
