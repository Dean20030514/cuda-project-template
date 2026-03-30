<#
.SYNOPSIS
    Visual Studio 开发环境辅助模块
.DESCRIPTION
    提供查找和导入 Visual Studio 开发者命令行环境的公共函数。
    供仓库内各构建脚本使用，避免代码重复。
#>

<#
.SYNOPSIS
    查找 Visual Studio 的 VsDevCmd.bat 路径
.DESCRIPTION
    按优先级查找 VS 开发者命令行脚本：
    1. VS 2022 稳定版（带 VC 工具）
    2. VS 2022 任意版本
    3. VS 预览版/Insiders（带 VC 工具）
    4. VS 预览版/Insiders 任意版本
    5. 常见安装路径备选
.OUTPUTS
    [string] VsDevCmd.bat 的完整路径
.EXAMPLE
    $vsdev = Find-VsDevCmd
    Write-Host "Using: $vsdev"
#>
function Find-VsDevCmd {
    [CmdletBinding()]
    [OutputType([string])]
    param()

    $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found: $vswhere. Please install Visual Studio 2022+ Build Tools or Community."
    }

    $inst = $null

    # 1. 优先稳定版，带 VC 工具（VS 2022 v17 或 VS 2026 v18+）
    $inst = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -version "[17.0,)" -property installationPath 2>$null

    # 2. 任意稳定版
    if (-not $inst) {
        $inst = & $vswhere -latest -products * `
            -version "[17.0,)" -property installationPath 2>$null
    }

    # 3. 预览版/Insiders，带 VC 工具
    if (-not $inst) {
        $inst = & $vswhere -latest -prerelease -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
    }

    # 4. 预览版/Insiders 任意
    if (-not $inst) {
        $inst = & $vswhere -latest -prerelease -products * `
            -property installationPath 2>$null
    }

    # 5. 备选常见路径
    if (-not $inst) {
        $candidates = @(
            'C:\Program Files\Microsoft Visual Studio\18\BuildTools',
            'C:\Program Files\Microsoft Visual Studio\18\Community',
            'C:\Program Files\Microsoft Visual Studio\18\Professional',
            'C:\Program Files\Microsoft Visual Studio\18\Enterprise',
            'C:\Program Files\Microsoft Visual Studio\2022\BuildTools',
            'C:\Program Files\Microsoft Visual Studio\2022\Community',
            'C:\Program Files\Microsoft Visual Studio\2022\Professional',
            'C:\Program Files\Microsoft Visual Studio\2022\Enterprise'
        )
        foreach ($c in $candidates) {
            $vsdevPath = Join-Path $c 'Common7\Tools\VsDevCmd.bat'
            if (Test-Path $vsdevPath) {
                $inst = $c
                break
            }
        }
    }

    if (-not $inst) {
        throw "Visual Studio with MSVC not found. Please install VS Build Tools/Community 2022+."
    }

    $vsdev = Join-Path $inst 'Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path $vsdev)) {
        throw "VsDevCmd.bat not found at: $vsdev"
    }

    return $vsdev
}

<#
.SYNOPSIS
    判断 VS 安装是否为 CUDA 官方支持的版本（VS 2022 或更高）
.PARAMETER VsDevCmdPath
    VsDevCmd.bat 的路径
.OUTPUTS
    [bool] 是否为 CUDA 官方支持的 VS 版本
#>
function Test-IsVs2022 {
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$VsDevCmdPath
    )

    $vsInstRoot = Split-Path (Split-Path (Split-Path $VsDevCmdPath))
    return $vsInstRoot -match '\\Microsoft Visual Studio\\2022\\'
}

<#
.SYNOPSIS
    获取 CUDA 允许不支持编译器的标志
.DESCRIPTION
    如果不是 CUDA 官方支持的 VS 版本，返回 '-allow-unsupported-compiler' 标志
.PARAMETER VsDevCmdPath
    VsDevCmd.bat 的路径
.OUTPUTS
    [string] CUDA 编译器标志（空或 '-allow-unsupported-compiler'）
#>
function Get-CudaUnsupportedCompilerFlag {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [string]$VsDevCmdPath
    )

    if (Test-IsVs2022 -VsDevCmdPath $VsDevCmdPath) {
        return ''
    } else {
        return '-allow-unsupported-compiler'
    }
}

<#
.SYNOPSIS
    从 cmd.exe 导入环境变量到当前 PowerShell 会话
.PARAMETER CmdLine
    要执行的 cmd 命令行（通常包含 VsDevCmd.bat 调用）
#>
function Import-CmdEnvironment {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$CmdLine
    )

    $output = cmd.exe /v:on /c $CmdLine
    foreach ($line in $output) {
        $idx = $line.IndexOf('=')
        if ($idx -gt 0) {
            $name = $line.Substring(0, $idx)
            $value = $line.Substring($idx + 1)
            # 更新常用的构建环境变量
            if ($name -match '^(PATH|INCLUDE|LIB|LIBPATH|ExtensionSdkDir|WindowsSdkDir|VSINSTALLDIR|VCToolsInstallDir)$' -or 
                $name -match '^VSCMD_') {
                Set-Item -Path "env:$name" -Value $value
            }
        }
    }
}

# 导出模块成员
Export-ModuleMember -Function Find-VsDevCmd, Test-IsVs2022, Get-CudaUnsupportedCompilerFlag, Import-CmdEnvironment
