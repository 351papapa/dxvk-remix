@echo off
rem Builds remix_mod_owners.exe into tools\remix_mod_owners\bin, with the USD DLLs and plugin folder it runs with.
rem Needs the VS 18 x64 toolchain and the packman USD (external\usd_stock_x64), or USD_ROOT set to a USD release folder.
setlocal
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not defined VSCMD_VER (
  call %VCVARS% x64 >nul || exit /b 1
)
set HERE=%~dp0
set ROOT=%HERE%..\..
if not defined USD_ROOT set USD_ROOT=%ROOT%\external\usd_stock_x64\release
if not exist "%USD_ROOT%\include\pxr\pxr.h" (
  echo USD not found in %USD_ROOT%. Set USD_ROOT to the release folder of the packman USD.
  exit /b 1
)
if not exist "%HERE%bin" mkdir "%HERE%bin"
set FLAGS=/nologo /std:c++17 /O2 /EHsc /W3 /MD /DNOMINMAX /DUNICODE /D_UNICODE /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4273 /I"%USD_ROOT%\include"
set LIBS=usd_usd.lib usd_sdf.lib usd_pcp.lib usd_ar.lib usd_kind.lib usd_tf.lib usd_vt.lib usd_gf.lib usd_js.lib usd_plug.lib usd_trace.lib usd_work.lib usd_ts.lib usd_arch.lib tbb12.lib
cl %FLAGS% "%HERE%remix_mod_owners.cpp" /Fo"%HERE%bin\\" /Fe"%HERE%bin\remix_mod_owners.exe" /link /LIBPATH:"%USD_ROOT%\lib" %LIBS% || exit /b 1

rem The DLLs the runtime ships, and USD's plugin folder, which USD finds beside its DLLs.
for %%L in (ar arch js gf kind sdf sdr pcp tf trace usd usdLux usdGeom usdRender usdShade usdSkel work vt plug ts) do (
  copy /y "%USD_ROOT%\lib\usd_%%L.dll" "%HERE%bin\" >nul || exit /b 1
)
copy /y "%USD_ROOT%\bin\tbb12.dll" "%HERE%bin\" >nul || exit /b 1
copy /y "%USD_ROOT%\bin\tbbmalloc.dll" "%HERE%bin\" >nul || exit /b 1
robocopy "%USD_ROOT%\lib\usd" "%HERE%bin\usd" /e /nfl /ndl /njh /njs /np >nul
if errorlevel 8 exit /b 1
echo Built %HERE%bin\remix_mod_owners.exe
exit /b 0
