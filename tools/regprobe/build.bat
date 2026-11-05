@echo off
rem Builds tools\regprobe\regprobe.exe. Needs the Vulkan SDK (VULKAN_SDK) and the MSVC x64 tools; run from a
rem "x64 Native Tools" prompt, or let this script call vcvarsall of the newest Visual Studio found by vswhere.
setlocal
cd /d "%~dp0"
if "%VULKAN_SDK%"=="" (
  echo VULKAN_SDK is not set. Install the Vulkan SDK.
  exit /b 1
)
where cl >nul 2>&1
if not errorlevel 1 goto build
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`call "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if "%VSDIR%"=="" (
  echo No Visual Studio with the x64 C++ tools found.
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
:build
cl /nologo /O2 /EHsc /std:c++17 /W3 regprobe.cpp /I "%VULKAN_SDK%\Include" /Fe:regprobe.exe /link "%VULKAN_SDK%\Lib\vulkan-1.lib"
exit /b %errorlevel%
