@echo off
chcp 65001 >nul
cd /d %~dp0

rem ============================================================
rem  Shared launcher: find a *usable* Python and run given .py
rem  Usage:  call _runpy.bat  "C:\path\to\script.py"
rem
rem  "usable" means: it can import numpy AND sounddevice.
rem  We test each candidate instead of trusting PATH order,
rem  because a bare python on PATH often lacks the deps.
rem ============================================================

set "TARGET=%~1"
if "%TARGET%"=="" (
    echo [ERROR] no target script given.
    pause
    exit /b 2
)

if not exist "%TARGET%" (
    echo [ERROR] script not found: %TARGET%
    echo         make sure it sits in the same folder as this .bat
    pause
    exit /b 2
)

set "PROBE=import numpy, sounddevice"
set "PYEXE="

rem ---- 1. bundled .venv next to this file ----
if exist "%~dp0.venv\Scripts\python.exe" (
    call :try_python "%~dp0.venv\Scripts\python.exe"
)
rem ---- .venv one level up ----
if not defined PYEXE if exist "%~dp0..\.venv\Scripts\python.exe" (
    call :try_python "%~dp0..\.venv\Scripts\python.exe"
)

rem ---- 2. every python on PATH ----
if not defined PYEXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYEXE call :try_python "%%i"
    )
)

rem ---- 3. py launcher ----
if not defined PYEXE (
    for /f "delims=" %%i in ('where py 2^>nul') do (
        if not defined PYEXE call :try_python "%%i"
    )
)

rem ---- 4. common install locations ----
if not defined PYEXE (
    for %%p in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
    ) do (
        if not defined PYEXE if exist %%p call :try_python "%%~p"
    )
)

rem ---- 5. WorkBuddy managed env (this machine only, harmless elsewhere) ----
if not defined PYEXE (
    if exist "%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe" (
        call :try_python "%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
    )
)

if defined PYEXE goto :run

echo ============================================================
echo   No usable Python found
echo ============================================================
echo.
echo   Found a Python, but it is missing the required libraries.
echo.
echo   Fix it by double-clicking:  install_deps.bat
echo   That creates a private .venv next to this file and
echo   installs everything into it.
echo.
pause
exit /b 1

:run
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"%PYEXE%" "%TARGET%"

if errorlevel 1 (
    echo.
    echo [NOTE] the script exited with a non-zero code.
    echo        scroll up for the error details.
)
pause
exit /b 0

rem ============================================================
rem  :try_python  -- keep the first interpreter that passes PROBE
rem  %~1 = candidate path ; sets PYEXE when it works
rem ============================================================
:try_python
if defined PYEXE exit /b 0
"%~1" -c "%PROBE%" >nul 2>&1
if not errorlevel 1 set "PYEXE=%~1"
exit /b 0
