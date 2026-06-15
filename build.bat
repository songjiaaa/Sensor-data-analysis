@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在检查打包依赖...
pip install -r requirements-dev.txt -q
echo 正在打包，请稍候...
pyinstaller --noconfirm --clean build_exe.spec
if %ERRORLEVEL% EQU 0 (
    echo.
    echo 打包完成！
    echo 可执行文件: dist\静态数据噪声分析工具.exe
    explorer dist
) else (
    echo 打包失败，请检查上方错误信息。
    pause
)
