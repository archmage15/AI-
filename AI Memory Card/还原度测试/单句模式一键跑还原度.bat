@echo off
chcp 65001 >nul
cd /d %~dp0
rem ============================================================
rem  单句模式一键跑还原度（强制单句，不看时长、不切句）
rem
rem  用途：一句一句练。把这一句当整体评，不按标点切句。
rem
rem  目录约定（和「长文本模式」一样）：
rem    audio\原音频.*   —— 这一句的原声（范本）
rem    audio\模仿.*     —— 你读的这一句
rem    output\          —— 结果（自动生成）
rem
rem  注意：单句模式不会自动重新生成参考音素表。
rem        如果换了**新句子**，先手动跑一次：
rem          python scripts\gen_ref_phonemes.py "新句子的文本"
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

"%PYEXE%" "%~dp0scripts\run_all.py" --单句 < nul

echo.
echo ============================================================
echo  Done. Results:
echo    input  audio : audio\
echo    output results: output\
echo ============================================================
pause
