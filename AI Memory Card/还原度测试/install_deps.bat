@echo off
chcp 65001 >nul
cd /d %~dp0
setlocal enabledelayedexpansion

rem ============================================================
rem  还原度测试 - 一键装依赖
rem
rem  What it does:
rem    1. locate a Python (3.11+)
rem    2. create an isolated .venv in this folder
rem    3. install dependencies (torch from official PyPI)
rem    4. self-check: can we import the key libraries
rem
rem  After this, 一键跑还原度.bat auto-uses this .venv.
rem ============================================================

echo ============================================================
echo    Install dependencies
echo ============================================================
echo.

rem ---------- 1. locate python ----------
set "PYEXE="
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%i"
)
if not defined PYEXE (
    for /f "delims=" %%i in ('where py 2^>nul') do (
        if not defined PYEXE set "PYEXE=%%i"
    )
)
if not defined PYEXE (
    for %%p in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
    ) do (
        if not defined PYEXE if exist %%p set "PYEXE=%%~p"
    )
)

if not defined PYEXE (
    echo [ERROR] Python not found.
    echo.
    echo   Install Python 3.11+ from https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add Python to PATH" on the first screen.
    echo.
    pause
    exit /b 1
)

echo [1/4] Python found:
"%PYEXE%" -c "import sys; print('      version', sys.version.split()[0]); print('      path   ', sys.executable)"
echo.

rem ---------- 2. create venv ----------
if exist "%~dp0.venv\Scripts\python.exe" (
    echo [2/4] .venv already exists, skipping creation.
) else (
    echo [2/4] creating .venv ^(about 30 seconds^) ...
    "%PYEXE%" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] failed to create .venv
        pause
        exit /b 1
    )
)
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo [ERROR] not found: %VPY%
    pause
    exit /b 1
)
echo        OK
echo.

rem ---------- 3. install ----------
rem All packages come from official PyPI. The Tsinghua mirror lags
rem badly behind for this project: it lacks the pinned versions (e.g.
rem torch 2.14.0, numpy 2.5.3 ...), so pip reports "from versions: none".
rem Official PyPI has them all.
set "MIRROR=https://pypi.org/simple"

echo [3/4] upgrading pip ...
"%VPY%" -m pip install --upgrade pip -i %MIRROR% --quiet
echo.

echo       installing torch ^(CPU build, about 200MB, slowest step^) ...
"%VPY%" -m pip install torch==2.14.0 -i %MIRROR%
if errorlevel 1 (
    echo [NOTE] torch install failed. Usually a network issue.
    echo        Just run this .bat again - pip skips what is already there.
    pause
    exit /b 1
)
echo.

echo       installing the rest ...
"%VPY%" -m pip install -r "%~dp0requirements.txt" -i %MIRROR%
if errorlevel 1 (
    echo.
    echo [NOTE] something failed above. Run this .bat once more.
    echo        pip will skip packages already installed.
    echo.
    pause
    exit /b 1
)
echo.

rem ---------- 4. self check ----------
echo [4/4] self-check: importing key libraries ...
"%VPY%" -c "import numpy, soundfile, torch, transformers, huggingface_hub, matplotlib, wordninja; print('      all OK'); print('      torch', torch.__version__)"
if errorlevel 1 (
    echo.
    echo [ERROR] self-check failed. The error above usually names the missing lib.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    DONE
echo ============================================================
echo.
echo    You can now double-click 一键跑还原度.bat.
echo.
echo    The FIRST run downloads the phoneme model ^(about 1.2GB^).
echo    Slow once, fast afterwards.
echo.
pause
