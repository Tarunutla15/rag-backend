@echo off
REM Run backend server with port check

echo Checking if port 8000 is available...
netstat -ano | findstr :8000 >nul
if %errorlevel% == 0 (
    echo Port 8000 is in use. Killing processes...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

echo Starting backend server...
cd /d %~dp0
..\venv\Scripts\activate
python -m app.main



