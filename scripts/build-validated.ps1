#requires -version 5.1
<#
.SYNOPSIS
  Rebuild the configured x64 runtime with verified external G-buffer shader data.
.DESCRIPTION
  Reuses an existing Release build and complete _output distribution. It does not
  clean artifacts, install auxiliary targets, deploy, or launch a game. A source
  junction is accepted only when it names the configured physical source tree.
#>
[CmdletBinding()]
param(
    [string]$BuildDir = '_Comp64Release',
    [string]$Python = 'python',
    [string]$MesonExe = 'meson',
    [string]$MesonPythonPath = $env:REMIX_MESON_PYTHON_PATH,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$requestedRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$localSettings = Join-Path $PSScriptRoot 'build.local.ps1'
$explicitSettings = @{}
foreach ($name in @('Python', 'MesonExe', 'MesonPythonPath')) {
    if ($PSBoundParameters.ContainsKey($name)) { $explicitSettings[$name] = Get-Variable -Name $name -ValueOnly }
}
if (Test-Path -LiteralPath $localSettings) { . $localSettings }
foreach ($name in $explicitSettings.Keys) { Set-Variable -Name $name -Value $explicitSettings[$name] }
$Python = (Get-Command $Python -ErrorAction Stop).Source
$helper = Join-Path $requestedRoot 'scripts-common\gbuffer_blob_build.py'
$requestedBuild = if ([IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $requestedRoot $BuildDir }
$configText = & $Python $helper config --runtime $requestedRoot --build-dir $requestedBuild
if ($LASTEXITCODE -ne 0) { throw 'Configured source/build physical-directory verification failed.' }
$configuration = ($configText -join "`n") | ConvertFrom-Json
$repoRoot = $configuration.runtime
$build = $configuration.build
$helper = Join-Path $repoRoot 'scripts-common\gbuffer_blob_build.py'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -version '[18.0,19.0)' -property installationPath
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($vs -join ''))) { throw 'Visual Studio 18 x64 C++ tools are required.' }
$vs = @($vs)[0]
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
$toolset = (Get-Content -LiteralPath (Join-Path $vs 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt') -Raw).Trim()
$compiler = (Get-Content -LiteralPath (Join-Path $build 'meson-info\intro-compilers.json') -Raw | ConvertFrom-Json).host.cpp
if ($compiler.id -ne 'msvc' -or $toolset -notmatch '^14\.(\d+)\.') { throw 'Unsupported configured compiler/toolset.' }
$expectedCompiler = '19.' + $Matches[1] + '.'
if (-not $compiler.version.StartsWith($expectedCompiler)) { throw "Configured compiler $($compiler.version) differs from selected toolset $toolset." }
if ($MesonPythonPath) {
    $MesonPythonPath = [IO.Path]::GetFullPath($MesonPythonPath)
    if (-not (Test-Path -LiteralPath (Join-Path $MesonPythonPath 'mesonbuild\mesonmain.py'))) { throw 'MesonPythonPath must contain mesonbuild/mesonmain.py.' }
} else {
    $MesonExe = (Get-Command $MesonExe -ErrorAction Stop).Source
}
Write-Host "Configured physical source: $repoRoot"
Write-Host "Configured build alias:     $build"
Write-Host "MSVC: $toolset; Meson configured: $($configuration.meson_version); compiler /MP1; Ninja -j1"
if ($CheckOnly) {
    Write-Host 'Read-only preflight passed. No build, cache patch, or output copy performed.'
    return
}

function Invoke-CheckedTool {
    param([string]$Executable, [string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Executable @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($code -ne 0) { throw "$Executable failed with exit code $code." }
}

$work = Join-Path $build 'validated-runtime'
[IO.Directory]::CreateDirectory($work) | Out-Null
$environmentBefore = @{}
Get-ChildItem Env: | ForEach-Object { $environmentBefore[$_.Name] = $_.Value }
try {
    # Only vcvarsall is run through cmd; all build arguments below use native argument arrays.
    $environmentScript = Join-Path $work 'vcvars.cmd'
    foreach ($path in @($vcvars, $environmentScript)) {
        if ($path -match '[%"!&|<>^\r\n]') { throw 'Unsupported cmd metacharacter in tool/build path.' }
    }
    [IO.File]::WriteAllText($environmentScript, "@echo off`r`ncall `"$vcvars`" x64 >nul`r`nif errorlevel 1 exit /b %errorlevel%`r`nset`r`n", [Text.Encoding]::ASCII)
    $environmentLines = & $env:ComSpec /d /v:off /c $environmentScript
    if ($LASTEXITCODE -ne 0) { throw 'vcvarsall failed.' }
    foreach ($line in $environmentLines) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
    # _CL_ is appended after command-line options, overriding the configured /MP.
    $env:_CL_ = ($env:_CL_ + ' /MP1').Trim()

    $mesonRunner = $MesonExe
    $mesonPrefix = @()
    if ($MesonPythonPath) {
        $loader = Join-Path $work 'meson_entry.py'
        $moduleLiteral = ConvertTo-Json -InputObject $MesonPythonPath -Compress
        $loaderText = @"
import platform
import sys
platform._wmi = None
sys.path.insert(0, $moduleLiteral)
from mesonbuild.mesonmain import main
raise SystemExit(main())
"@
        [IO.File]::WriteAllText($loader, $loaderText, (New-Object Text.UTF8Encoding($false)))
        $mesonRunner = $Python
        $mesonPrefix = @($loader)
    }
    $versionText = & $mesonRunner @mesonPrefix --version
    if ($LASTEXITCODE -ne 0 -or ($versionText -join '').Trim() -ne $configuration.meson_version) {
        throw 'Selected Meson version differs from the existing configured build. Use its matching Meson package.'
    }
    $baseline = Join-Path $work 'supporting-files-before.json'
    Invoke-CheckedTool -Executable $Python -Arguments @($helper, 'snapshot-output', '--runtime', $repoRoot, '--build-dir', $build, '--report', $baseline)
    Invoke-CheckedTool -Executable $mesonRunner -Arguments ($mesonPrefix + @('setup', '--reconfigure', $build, $repoRoot))
    Invoke-CheckedTool -Executable $Python -Arguments @($helper, 'normalize-ninja', '--runtime', $repoRoot, '--build-dir', $build)
    Invoke-CheckedTool -Executable $mesonRunner -Arguments ($mesonPrefix + @('compile', '-C', $build, '-j', '1', 'rtx_shaders'))
    Invoke-CheckedTool -Executable $Python -Arguments @($helper, 'normalize-ninja', '--runtime', $repoRoot, '--build-dir', $build)
    Invoke-CheckedTool -Executable $Python -Arguments @($helper, 'apply', '--runtime', $repoRoot, '--build-dir', $build, '--report', (Join-Path $work 'gbuffer-apply.json'))
    Invoke-CheckedTool -Executable $mesonRunner -Arguments ($mesonPrefix + @('compile', '-C', $build, '-j', '1', 'd3d9'))
    Invoke-CheckedTool -Executable $Python -Arguments @($helper, 'verify', '--runtime', $repoRoot, '--build-dir', $build)
    Invoke-CheckedTool -Executable $Python -Arguments @($helper, 'publish', '--runtime', $repoRoot, '--build-dir', $build, '--baseline', $baseline, '--report', (Join-Path $work 'build-report.json'))
    Write-Host "Validated runtime copied to $repoRoot\_output\d3d9.dll"
    Write-Host "Report: $work\build-report.json"
} finally {
    Get-ChildItem Env: | Where-Object { -not $environmentBefore.ContainsKey($_.Name) } | ForEach-Object {
        [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process')
    }
    foreach ($name in $environmentBefore.Keys) { [Environment]::SetEnvironmentVariable($name, $environmentBefore[$name], 'Process') }
}
