@echo off
REM Run backend server on alternative port 8001

echo Starting backend server on port 8001...
cd /d %~dp0
..\venv\Scripts\activate
set PORT=8001
python -c "from app.config import settings; settings.PORT = 8001; import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=8001)"



