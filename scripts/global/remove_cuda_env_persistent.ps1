<#!
Synopsis: Remove persistent CUDA/cuDNN environment entries from Machine/User scope.

This script cleans up PATH entries added by install_cuda_env_persistent.ps1, and can optionally
remove CUDA_PATH/CUDNN_ROOT variables.

Usage (PowerShell as Admin for Machine scope):
  pwsh -NoProfile -File remove_cuda_env_persistent.ps1 -Scope Machine -AlsoRemoveCudaPath -AlsoRemoveCuDnnRoot

If you omit -CudaPath, it will use the current Machine/User CUDA_PATH to decide which PATH segments to remove.
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [ValidateSet('Machine','User')]
  [string]$Scope = 'Machine',
  [string]$CudaPath,
  [switch]$AlsoRemoveCudaPath,
  [switch]$AlsoRemoveCuDnnRoot,
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

function Get-Env([string]$name, [EnvironmentVariableTarget]$t){
  return [Environment]::GetEnvironmentVariable($name,$t)
}

function Set-Env([string]$name, [string]$val, [EnvironmentVariableTarget]$t){
  if ($DryRun) { Write-Host "[DryRun] Set $name ($t) = $val" -ForegroundColor DarkCyan; return }
  [Environment]::SetEnvironmentVariable($name,$val,$t)
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

$target = if ($Scope -eq 'Machine') { [EnvironmentVariableTarget]::Machine } else { [EnvironmentVariableTarget]::User }
if ($target -eq [EnvironmentVariableTarget]::Machine -and -not (Test-IsAdmin)){
  throw "Need Administrator rights for -Scope Machine. Please re-run in an elevated PowerShell."
}

$cudaRoot = if ($CudaPath) { $CudaPath } else { Get-Env 'CUDA_PATH' $target }
if (-not $cudaRoot) { Write-Host "[remove-cuda-env] CUDA_PATH not set; will try to clean common CUDA bin patterns." -ForegroundColor Yellow }

$path = Get-Env 'Path' $target
$sep = ';'
$parts = if ($path) { $path.Split($sep) } else { @() }

function Test-PathShouldRemove([string]$p){
  if (-not $p) { return $false }
  if ($cudaRoot) {
    return $p.TrimStart() -like (Join-Path $cudaRoot '*')
  }
  # Fallback: remove typical CUDA bin paths
  return ($p -match 'NVIDIA GPU Computing Toolkit\\CUDA') -and ($p -match '\\bin(\\|$)')
}

$kept = New-Object System.Collections.Generic.List[string]
$removed = New-Object System.Collections.Generic.List[string]
foreach($p in $parts){
  if (Test-PathShouldRemove $p) { $removed.Add($p) } else { $kept.Add($p) }
}

if ($removed.Count -gt 0) {
  if ($DryRun) {
    Write-Host "[DryRun] Would remove from PATH ($target):" -ForegroundColor DarkYellow
    $removed | ForEach-Object { Write-Host "  $_" }
  } else {
    Set-Env 'Path' ($kept -join $sep) $target
    Write-Host "[remove-cuda-env] Removed PATH entries ($target):" -ForegroundColor Green
    $removed | ForEach-Object { Write-Host "  $_" }
  }
} else {
  Write-Host "[remove-cuda-env] No CUDA-related PATH entries found for $Scope." -ForegroundColor Yellow
}

if ($AlsoRemoveCuDnnRoot) {
  if ($DryRun) { Write-Host "[DryRun] Would clear CUDNN_ROOT ($target)" }
  else { Set-Env 'CUDNN_ROOT' $null $target }
}

if ($AlsoRemoveCudaPath) {
  if ($DryRun) { Write-Host "[DryRun] Would clear CUDA_PATH ($target)" }
  else { Set-Env 'CUDA_PATH' $null $target }
}

Send-EnvChangeBroadcast
Write-Host "[remove-cuda-env] Done. Open a NEW terminal to see changes." -ForegroundColor Green
