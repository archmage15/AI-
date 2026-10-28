@echo off
chcp 65001 >nul
cd /d %~dp0
rem ============================================================
rem  Single-sentence mode (same pipeline as long-text)
rem  Put one sentence as your audio, run this to score just it.
rem  Ref audio goes in audio\ref\ subfolder (analyze it once first).
rem  Your audio goes in audio\ root (named as imitation).
rem ============================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "PYEXE="
set "CAND=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if exist "%CAND%" set "PYEXE=%CAND%"
if not defined PYEXE if exist "%~dp0.venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYEXE (
    echo [ERROR] No usable Python found. Run install_deps.bat first.
    pause
    exit /b 1
)
"%PYEXE%" "%~dp0scripts\run_all.py" < nul
echo.
echo ============================================================
echo  Done. Results in output\
echo ============================================================
pause
