@echo off
chcp 65001 >nul
cd /d %~dp0
rem ============================================================
rem  长文本模式一键跑还原度（默认：任何音频都按台词切句评测）
rem
rem  由 scripts\run_all.py 跑：
rem    1. 读原音频缓存（audio\原音频\原音频缓存.json，由「分析原音频.bat」生成）
rem       —— 没有缓存 → 提示先跑 分析原音频.bat
rem    2. 只对模仿音频跑音素模型 + 节奏重音，跟缓存的原音频参数比
rem    3. overall_score 出综合报告
rem
rem  目录约定：
rem    audio\原音频\原音频.*  放范本（先跑一次 分析原音频.bat 建缓存）
rem    audio\模仿.*          放你的跟读（每次换）
rem    output\               结果（自动生成）
rem ============================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem ------------------------------------------------------------
rem  Log switches (optional). Remove the "rem" to turn one on.
rem
rem  LOG_QUIET=1   terminal shows only the result blocks,
rem                step-by-step progress goes to the log files only.
rem                Use this for everyday runs to keep the window clean.
rem
rem  LOG_VERBOSE=1 also log intermediate numbers (phoneme counts,
rem                word counts, sub-scores) into the log files.
rem
rem  LOG_OFF=1     do not write log files at all (terminal unchanged).
rem ------------------------------------------------------------
rem set LOG_QUIET=1
rem set LOG_VERBOSE=1
rem set LOG_OFF=1

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
echo  Done. Results:
echo    input  audio  : audio\
echo    output results: output\
echo ============================================================
pause
