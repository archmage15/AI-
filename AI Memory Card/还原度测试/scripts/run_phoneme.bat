@echo off
chcp 65001 >nul
cd /d %~dp0
call "%~dp0..\_runpy.bat" "%~dp0compare_phoneme.py"
