@echo off
chcp 65001 >nul
cd /d %~dp0
set PYTHONUTF8=1
"C:\Users\24048\.workbuddy\binaries\python\envs\default\Scripts\python.exe" xunfei_ise_test.py
pause
