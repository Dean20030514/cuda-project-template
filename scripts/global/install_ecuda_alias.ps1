<#!
Synopsis: Install a convenient 'ecuda' function into your PowerShell profile.

After running this script, you can type 'ecuda' in any new PowerShell to enable
CUDA + cuDNN environment for the current session (using the repo's enable script).

It will append function definitions to your PowerShell profile (CurrentUser-AllHosts):
  - ecuda               -> dot-sources enable_cuda_env.ps1 (takes optional args)
  - ecuda-persist-user  -> calls install_cuda_env_persistent.ps1 -Scope User
  - ecuda-persist-machine -> calls install_cuda_env_persistent.ps1 -Scope Machine (requires admin)
  - dcuda-persist       -> calls remove_cuda_env_persistent.ps1 (uninstall)

Usage:
  pwsh -NoProfile -File install_ecuda_alias.ps1

Then open a NEW PowerShell and run:
  ecuda              # to enable CUDA/cuDNN in that session
  ecuda -Quiet       # less output
  ecuda -CudaPath "C:\\..." -CuDnnRoot "C:\\..."  # override paths
#>

[CmdletBinding()]
param(
  # Where to write aliases: CurrentUser AllHosts (default) or CurrentUser CurrentHost
  [ValidateSet('CurrentUserAllHosts','CurrentUserCurrentHost')]
  [string]$ProfileScope = 'CurrentUserAllHosts'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ProfilePath([string]$scope){
  switch ($scope) {
    'CurrentUserAllHosts'    { return $PROFILE.CurrentUserAllHosts }
    'CurrentUserCurrentHost' { return $PROFILE.CurrentUserCurrentHost }
    default { return $PROFILE }
  }
}

# 使用 $PSScriptRoot 动态获取脚本路径，避免硬编码用户路径
$enablePath   = Join-Path $PSScriptRoot 'enable_cuda_env.ps1'
$persistPath  = Join-Path $PSScriptRoot 'install_cuda_env_persistent.ps1'
$removePath   = Join-Path $PSScriptRoot 'remove_cuda_env_persistent.ps1'

if (-not (Test-Path $enablePath)) { throw "enable_cuda_env.ps1 not found at $enablePath" }

$profilePath = Get-ProfilePath $ProfileScope
$profileDir = Split-Path -Path $profilePath
if (-not (Test-Path $profileDir)) {
  New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}
if (-not (Test-Path $profilePath)) {
  New-Item -ItemType File -Path $profilePath -Force | Out-Null
}

$content = Get-Content -Path $profilePath -ErrorAction SilentlyContinue

# Build the function block
$enableLiteral  = $enablePath
$persistLiteral = $persistPath
$removeLiteral  = $removePath

# Build as single-quoted here-string to avoid any expansion, then replace tokens
$block = @'
# ===== CUDA quick aliases (added by install_ecuda_alias.ps1) =====
function ecuda {
  param(
    [string]
    [Alias('cuda')]
    $CudaPath,
    [string]
    $CuDnnRoot,
    [ValidateSet('amd64','x64','x86','arm64')]
    [string]
    $VsArch = 'amd64',
    [switch]
    $Quiet
  )
  # dot-source to apply into current session
  . '__ENABLE__' -CudaPath $CudaPath -CuDnnRoot $CuDnnRoot -VsArch $VsArch -Quiet:$Quiet
}

function ecuda-persist-user {
  pwsh -NoProfile -ExecutionPolicy Bypass -File '__PERSIST__' -Scope User
}

function ecuda-persist-machine {
  Write-Host 'Requires Administrator PowerShell' -ForegroundColor Yellow
  pwsh -NoProfile -ExecutionPolicy Bypass -File '__PERSIST__' -Scope Machine
}

function dcuda-persist {
  param(
    [ValidateSet('Machine','User')]
    [string]$Scope = 'Machine'
  )
  pwsh -NoProfile -ExecutionPolicy Bypass -File '__REMOVE__' -Scope $Scope -AlsoRemoveCudaPath -AlsoRemoveCuDnnRoot
}
# ===== end CUDA quick aliases =====
'@

$block = $block.Replace('__ENABLE__',  $enableLiteral)
$block = $block.Replace('__PERSIST__', $persistLiteral)
$block = $block.Replace('__REMOVE__',  $removeLiteral)

# Avoid duplicate blocks by a simple marker search
$marker = '# ===== CUDA quick aliases (added by install_ecuda_alias.ps1) ====='
$contentText = if ($content) { $content -join "`n" } else { '' }
if ($contentText -and ($contentText -like ('*' + $marker + '*'))) {
  Write-Host '[install-ecuda-alias] Aliases already present in profile. Skipping append.' -ForegroundColor Yellow
} else {
  Add-Content -Path $profilePath -Value "`n$block`n"
  Write-Host "[install-ecuda-alias] Aliases added to: $profilePath" -ForegroundColor Green
}

Write-Host '[install-ecuda-alias] Open a NEW PowerShell and run: ecuda' -ForegroundColor Cyan
