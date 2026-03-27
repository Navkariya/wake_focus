@echo off
REM Wake Focus - Windows .exe Build Script
REM Must be run ON WINDOWS (PyInstaller does not cross-compile)
REM
REM Prerequisites:
REM   pip install pyinstaller PySide6 opencv-python-headless mediapipe ultralytics paho-mqtt

echo ========================================
echo  Wake Focus - Windows Build
echo ========================================

REM Activate venv if needed
REM call venv\Scripts\activate

REM Install PyInstaller
pip install pyinstaller

REM Build
pyinstaller packaging/windows/wake_focus.spec --noconfirm --clean

echo.
echo Build complete!
echo Executable: dist\WakeFocus\WakeFocus.exe
echo.
pause
