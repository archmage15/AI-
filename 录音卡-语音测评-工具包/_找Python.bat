@echo off
chcp 65001 >nul
setlocal
cd /d %~dp0
set PYTHONUTF8=1

REM ============================================================
REM  通用启动器（供本工具包内所有 .bat 调用）
REM  调用方式：  call "%~dp0..\_找Python.bat"
REM  成功后返回变量：%PYEXE%
REM ============================================================

set "PYEXE="
set "PKGROOT=%~dp0"

REM ---- 1. 优先用工具包自带环境 .venv ----
if exist "%PKGROOT%.venv\Scripts\python.exe" (
    set "PYEXE=%PKGROOT%.venv\Scripts\python.exe"
    goto :found
)

REM ---- 2. 退回系统 PATH 里的 python ----
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%i"
)
if defined PYEXE goto :found

REM ---- 3. 退回 py 启动器 ----
for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%i"
)
if defined PYEXE goto :found

REM ---- 都没找到 ----
echo.
echo ============================================================
echo   [错误] 没找到 Python
echo ============================================================
echo.
echo   请先做这两件事之一：
echo.
echo   (A) 推荐：双击运行工具包根目录的  安装依赖.bat
echo       它会自动建好一个独立环境 .venv，之后所有脚本都能用。
echo.
echo   (B) 或者自己装 Python 3.10 以上版本，安装时勾选
echo       "Add python.exe to PATH"，然后重开本窗口再试。
echo.
echo ============================================================
pause
exit /b 1

:found
endlocal & set "PYEXE=%PYEXE%"
exit /b 0
