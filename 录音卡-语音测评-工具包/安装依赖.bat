@echo off
chcp 65001 >nul
setlocal
cd /d %~dp0
set PYTHONUTF8=1

echo ============================================================
echo   录音卡 · 语音测评工具包 —— 环境安装
echo ============================================================
echo.
echo 本脚本会做三件事：
echo   1. 找一台能用的 Python（3.10 以上）
echo   2. 在工具包目录下建一个独立环境 .venv
echo   3. 把要用的库从国内镜像装进去
echo.
echo 全程约 5～15 分钟（取决于网速，torch 比较大）。
echo 中途不要关窗口。装完这个窗口会自己停住。
echo.
echo ------------------------------------------------------------
pause
echo.

REM ================= 1. 找一个 Python 来建环境 =================
echo [1/3] 正在找 Python ...
set "BASEPY="

REM 1.1 系统 PATH 里的 python
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined BASEPY set "BASEPY=%%i"
)

REM 1.2 py 启动器
if not defined BASEPY (
    for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do (
        if not defined BASEPY set "BASEPY=%%i"
    )
)

if not defined BASEPY (
    echo.
    echo [错误] 本机没有找到 Python。
    echo.
    echo 请先去 https://www.python.org/downloads/ 下载安装 Python 3.10 以上版本，
    echo 安装时务必勾选 "Add python.exe to PATH"，装完重开本脚本。
    echo.
    pause
    exit /b 1
)

echo      找到: %BASEPY%
"%BASEPY%" -c "import sys;print('     版本:', sys.version.split()[0]);raise SystemExit(0 if sys.version_info>=(3,10) else 1)"
if errorlevel 1 (
    echo.
    echo [错误] 这个 Python 版本低于 3.10，装不了。请升级后再试。
    echo.
    pause
    exit /b 1
)

REM ================= 2. 建独立环境 .venv =================
echo.
echo [2/3] 正在建独立环境 .venv（就在本工具包目录下）...
if exist ".venv\Scripts\python.exe" (
    echo      已存在，跳过创建。
) else (
    "%BASEPY%" -m venv ".venv"
    if errorlevel 1 (
        echo.
        echo [错误] 建环境失败。常见原因：Python 安装不完整，缺少 venv 模块。
        echo        可试试重新安装 Python 并勾选 "pip" 和 "venv" 组件。
        echo.
        pause
        exit /b 1
    )
    echo      建好了。
)

set "VENVPY=%~dp0.venv\Scripts\python.exe"

REM ================= 3. 装依赖 =================
echo.
echo [3/3] 正在装依赖（走清华镜像，慢的话请等）...
echo.
"%VENVPY%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 echo    （pip 自升级失败，不影响，继续）
echo.

"%VENVPY%" -m pip install -r "%~dp0requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo.
    echo ============================================================
    echo   [错误] 依赖安装失败
    echo ============================================================
    echo   把上面的报错信息复制给 AI，让它帮你判断。
    echo   常见原因：
    echo     - 网络断了或镜像暂时不通
    echo     - Python 版本太新，某个包还没有对应的安装包
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   安装完成
echo ============================================================
echo.
echo   现在可以双击各个测试文件夹里的 .bat 跑脚本了。
echo.
echo   注意：第一次跑「还原度测试 / 运行音素对比.bat」时，
echo   它会自动下载一个约 1.2GB 的音素模型，需要等一会儿。
echo   下载过一次之后就不再下了。
echo.
echo ============================================================
pause
