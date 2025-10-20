@echo off
setlocal
echo Setting up Visual Studio Build Tools environment...
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64

echo.
echo === Compiling CUDA kernel (matrix_mult_cuda.cu) ===
nvcc -c -arch=sm_75 matrix_mult_cuda.cu -o matrix_mult_cuda.obj

echo.
echo === Compiling CPU code and linking with CUDA object ===
cl /EHsc /Fe:matrix_mult_cuda.exe matrix_mult_win.c matrix_mult_cuda.obj ^
   /I"%CUDA_PATH%\include" ^
   /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib

if %errorlevel% neq 0 (
    echo [ERROR] Compilation or linking failed.
    pause
    exit /b %errorlevel%
)

echo.
echo === Running benchmark ===
matrix_mult_cuda.exe

echo.
echo === Generating performance graph ===
python "%~dp0perf_plot.py"

echo.
echo Done.
pause
endlocal
