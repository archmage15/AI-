@echo off
chcp 65001 >nul
cd /d %~dp0
rem ============================================================
rem  长文本模式一键跑还原度（默认：任何音频都按台词切句评测）
rem
rem  由 scripts\run_all.py 跑「长文本模式」：
rem    1. 先调讯飞转写取台词文本当真值（不依赖手动放 转写文本.txt）
rem       —— 没识别到文本 → 报「没识别到文本」直接退出
rem    2. 按文本标点切句 → compare_long（四维度逐句评测）
rem    3. overall_score 出综合报告
rem
rem  目录约定：
rem    audio\      放 原音频.* / 模仿.*（mp3 / wav / flac / ogg 任意格式）
rem    output\     结果（自动生成）
rem    scripts\    所有 .py（run_all.py 是入口）
rem    讯飞配置.json 必需：长文本自动转写用（app_id + secret_key）
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
