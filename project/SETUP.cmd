@echo off
setlocal
cd /d "%~dp0"
echo Creating a local Python environment. Python 3.12 or later is required.
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" spam_detector.py train
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto failed
echo Setup complete. Double-click START_DEMO.cmd.
pause
exit /b 0
:failed
echo Setup failed. Read the error above and the README troubleshooting section.
pause
exit /b 1
