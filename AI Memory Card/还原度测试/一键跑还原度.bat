@echo off
chcp 65001 >nul
cd /d %~dp0
rem ============================================================
rem  Run full fidelity compare - all three steps in one go.
rem  Put this .bat in the ROOT of the "haindu ceshi" folder.
rem
rem  Folder layout it expects (all ASCII names):
rem    %~dp0audio\       input wav files  (yuanyinpin.wav / mofang.wav)
rem    %~dp0output\      results written by the scripts
rem    %~dp0scripts\     the three .py programs
rem    %~dp0models\      phoneme model (read only, auto-downloaded if missing)
rem    %~dp0xunfei.json   optional API key file, read if present
rem
rem  Steps:
rem    [1] phoneme -> pronunciation + sound change
rem    [2] rhythm  -> rhythm / pause / pitch
rem    [3] overall -> weighted score + correction report
rem  Pauses only once at the end.
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
rem
rem  Full logs are written to the  output\logs  folder:
rem    one file appended across runs (full history),
rem    one file rewritten each run (latest run only).
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

echo ============================================================
echo  [1/3] Phoneme compare  (pronunciation + sound change)
echo ============================================================
"%PYEXE%" "%~dp0scripts\compare_phoneme.py" < nul

echo.
echo ============================================================
echo  [2/3] Rhythm / intonation compare
echo ============================================================
"%PYEXE%" "%~dp0scripts\compare_rhythm.py" < nul

echo.
echo ============================================================
echo  [3/3] Overall fidelity score + report
echo ============================================================
"%PYEXE%" "%~dp0scripts\overall_score.py" < nul

echo.
echo ============================================================
echo  Done. Results:
echo    input  audio  : audio\
echo    output results: output\
echo ============================================================
pause
