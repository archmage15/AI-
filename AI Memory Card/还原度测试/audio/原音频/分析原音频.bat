@echo off
chcp 65001 >nul
cd /d %~dp0..\..\
rem ============================================================
rem  分析原音频（一次性跑完，生成缓存）
rem
rem  跑一次后，长文本模式一键跑还原度会直接读这份缓存，
rem  不再重跑原音频的讯飞转写 + 音素识别（省额度 + 省时间）。
rem
rem  目录约定：
rem    audio\原音频\原音频.*  放范本
rem    缓存输出到 audio\原音频\原音频缓存.json
rem ============================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "PYEXE="
set "CAND=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if exist "%CAND%" set "PYEXE=%CAND%"
if not defined PYEXE if exist "%~dp0..\..\.venv\Scripts\python.exe" set "PYEXE=%~dp0..\..\.venv\Scripts\python.exe"
if not defined PYEXE (
    echo [ERROR] No usable Python found. Run install_deps.bat first.
    pause
    exit /b 1
)
"%PYEXE%" "%~dp0..\..\scripts\analyze_reference.py" < nul
echo.
echo ============================================================
echo  完成。缓存已写到 audio\原音频\原音频缓存.json
echo  以后「长文本模式一键跑还原度」会直接读这份缓存。
echo ============================================================
pause
