@echo off
chcp 65001 >nul
cd /d %~dp0
rem ============================================================
rem  Analyze reference audio ONCE, cache the result.
rem  After this, run_all reads the cache and only analyzes
rem  your imitation audio (saves xunfei quota + model time).
rem
rem  Ref audio:  audio\yuan\yuan.mp3   (put it there first)
rem  Cache out:  audio\yuan\yuan_cache.json
rem  Needs xunfei config json in project root.
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
"%PYEXE%" "%~dp0scripts\analyze_reference.py" < nul
echo.
echo ============================================================
echo  Done. Cache written to audio\yuan\yuan_cache.json
echo ============================================================
pause
