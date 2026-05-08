@echo off
REM Activate virtual environment and run the FastAPI server
cd ..
call venv\Scripts\activate.bat
cd backend
python -m app.main



