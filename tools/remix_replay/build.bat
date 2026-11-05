@echo off
rem Builds remix_replay.exe (and remix_testhost.exe) with the VS 18 x64 toolchain. No meson.
setlocal
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not defined VSCMD_VER (
  call %VCVARS% x64 >nul || exit /b 1
)
set HERE=%~dp0
set ROOT=%HERE%..\..
if not exist "%HERE%bin" mkdir "%HERE%bin"
set FLAGS=/nologo /std:c++20 /O2 /EHsc /W3 /MD /DUNICODE /D_UNICODE /D_CRT_SECURE_NO_WARNINGS /I"%ROOT%\public\include"
cl %FLAGS% "%HERE%remix_replay.cpp" /Fo"%HERE%bin\\" /Fe"%HERE%bin\remix_replay.exe" /link user32.lib || exit /b 1
if exist "%HERE%remix_testhost.cpp" (
  cl %FLAGS% "%HERE%remix_testhost.cpp" "%HERE%remix_testhost_d3d9.cpp" /Fo"%HERE%bin\\" /Fe"%HERE%bin\remix_testhost.exe" /link user32.lib d3d11.lib dxgi.lib || exit /b 1
)
echo Built %HERE%bin\remix_replay.exe
