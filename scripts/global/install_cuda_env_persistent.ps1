<#!
Synopsis: Persist CUDA (+ cuDNN) environment for all terminals and projects (Windows).
This script sets Machine/User environment variables so that new terminals can use nvcc and cuDNN.

What it does:
- Detects CUDA Toolkit install (highest version under Program Files) unless you pass -CudaPath.
- Sets CUDA_PATH and CUDNN_ROOT (default to CUDA_PATH) at the chosen scope.
- Adds CUDA bin and cuDNN bin subfolders to PATH at the chosen scope (if missing).
- Broadcasts an environment change so some apps pick it up; new terminals will see it.

Notes:
- Run as Administrator when using -Scope Machine (recommended for all users).
- MSVC (cl.exe) developer INCLUDE/LIB vars are not global; still use VS Developer Prompt or VsDevCmd for builds.

Usage examples (PowerShell as Admin):
  # Machine-wide (recommended)
  pwsh -NoProfile -File install_cuda_env_persistent.ps1 -Scope Machine

  # User only (no admin required)
  pwsh -NoProfile -File install_cuda_env_persistent.ps1 -Scope User

  # Specify custom CUDA install path and a separate cuDNN root
  pwsh -NoProfile -File install_cuda_env_persistent.ps1 -CudaPath "C:\\CUDA\\v13.0" -CuDnnRoot "C:\\Program Files\\NVIDIA\\CUDNN"
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [ValidateSet('Machine','User')]
  [string]$Scope = 'Machine',
  [string]$CudaPath,
  [string]$CuDnnRoot,
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-IsAdmin {
  try {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
  } catch { return $false }
}

function Get-HighestCudaPath {
  param([string]$Fallback)
  if ($Fallback -and (Test-Path $Fallback)) { return (Resolve-Path $Fallback).Path }
  if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) { return (Resolve-Path $env:CUDA_PATH).Path }
  $base = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'
  if (-not (Test-Path $base)) { return $null }
  $dirs = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
  if ($dirs.Count -gt 0) { return $dirs[0].FullName }
  return $null
}

function Get-EnvPath([EnvironmentVariableTarget]$target){
  return [Environment]::GetEnvironmentVariable('Path', $target)
}

function Set-EnvVar([string]$name, [string]$value, [EnvironmentVariableTarget]$target){
  if ($DryRun) { Write-Host "[DryRun] Set $name ($target) = $value" -ForegroundColor DarkCyan; return }
  [Environment]::SetEnvironmentVariable($name, $value, $target)
}

function Update-Path([string[]]$toAdd, [EnvironmentVariableTarget]$target){
  $sep = ';'
  $orig = Get-EnvPath $target
  $parts = @()
  if ($orig) { $parts = $orig.Split($sep) }
  $normalized = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach($p in $parts){ if ($p) { $null = $normalized.Add($p.Trim()) } }
  $added = @()
  foreach($p in $toAdd){
    if ($p -and (Test-Path $p)){
      if (-not $normalized.Contains($p)){
        $parts = ,$p + $parts  # prepend for higher priority
        $null = $normalized.Add($p)
        $added += $p
      }
    }
  }
  if ($added.Count -gt 0){
    $newPath = ($parts -join $sep).TrimEnd($sep)
    if ($DryRun) {
      Write-Host "[DryRun] Update Path ($target): will prepend:`n  " -NoNewline
      $added | ForEach-Object { Write-Host $_ }
    } else {
      [Environment]::SetEnvironmentVariable('Path', $newPath, $target)
      Write-Host "[install-cuda-env] Added to PATH ($target):" -ForegroundColor Green
      foreach($a in $added){ Write-Host "  $a" -ForegroundColor Green }
    }
  } else {
    Write-Host "[install-cuda-env] PATH already contains required CUDA/cuDNN entries for $target." -ForegroundColor Yellow
  }
}

function Send-EnvChangeBroadcast {
  if ($DryRun) { Write-Host "[DryRun] Would broadcast WM_SETTINGCHANGE (Environment)"; return }
  $sig = @'
using System;
using System.Runtime.InteropServices;
public static class EnvBroadcast {
  [DllImport("user32.dll", SetLastError=true, CharSet=CharSet.Auto)]
  public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);
}
'@
  Add-Type -TypeDefinition $sig -ErrorAction SilentlyContinue | Out-Null
  $HWND_BROADCAST = [IntPtr]0xffff
  $WM_SETTINGCHANGE = 0x001A
  $res = [UIntPtr]::Zero
  [void][EnvBroadcast]::SendMessageTimeout($HWND_BROADCAST, $WM_SETTINGCHANGE, [UIntPtr]::Zero, 'Environment', 0, 5000, [ref]$res)
}

# Pre-flight checks
$target = if ($Scope -eq 'Machine') { [EnvironmentVariableTarget]::Machine } else { [EnvironmentVariableTarget]::User }
if ($target -eq [EnvironmentVariableTarget]::Machine -and -not (Test-IsAdmin)){
  throw "Need Administrator rights for -Scope Machine. Please re-run in an elevated PowerShell."
}

$cudaRoot = Get-HighestCudaPath -Fallback $CudaPath
if (-not $cudaRoot){ throw "CUDA Toolkit not found. Install CUDA, or pass -CudaPath. Common: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0" }
Write-Host "[install-cuda-env] Using CUDA_PATH = $cudaRoot" -ForegroundColor Cyan

# cuDNN root preference: default to CUDA Toolkit root (CUDA 12.4+ often bundles cuDNN)
$cuDnnRoot = if ($CuDnnRoot) { $CuDnnRoot } else { $cudaRoot }
Write-Host "[install-cuda-env] Using CUDNN_ROOT = $cuDnnRoot" -ForegroundColor Cyan

# Build PATH additions: prefer versioned cuDNN bin folders first, then general bin
$pathsToAdd = @()
$bin     = Join-Path $cudaRoot 'bin'
$bin64   = Join-Path $cudaRoot 'bin\x64'
# 动态发现版本化 bin 子目录（如 bin/13.0, bin/12.9 等），优先高版本
# Dynamically discover versioned bin subdirs (e.g. bin/13.0, bin/12.9), prefer higher versions
if (Test-Path $bin) {
  $versionedBins = Get-ChildItem -Path $bin -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^\d+\.\d+$' } |
    Sort-Object { [version]$_.Name } -Descending |
    Select-Object -ExpandProperty FullName
  foreach ($p in $versionedBins) { $pathsToAdd += $p }
}
if (Test-Path $bin)   { $pathsToAdd += $bin }
if (Test-Path $bin64) { $pathsToAdd += $bin64 }

# Persist variables
Set-EnvVar -name 'CUDA_PATH' -value $cudaRoot -target $target
Set-EnvVar -name 'CUDNN_ROOT' -value $cuDnnRoot -target $target

# Update PATH
Update-Path -toAdd $pathsToAdd -target $target

Send-EnvChangeBroadcast

Write-Host "[install-cuda-env] Done. Open a NEW terminal to see changes. For MSVC builds, still init VS dev env (Developer PowerShell or VsDevCmd)." -ForegroundColor Green
