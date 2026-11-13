@echo off
echo =========================================================================
echo   TimeMixer Adaptive-FMM Research Suite (Production Grade)
echo =========================================================================
cd /d "%~dp0"
echo [1/3] Starting Backend API Server (Port 8000)...
start "TimeMixer Backend (Port 8000)" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 2 /nobreak >nul

echo [2/3] Starting Vite React Frontend (Port 5173)...
start "TimeMixer Frontend (Port 5173)" cmd /k "cd frontend && npm run dev"
timeout /t 3 /nobreak >nul

echo [3/3] Opening Browser at http://localhost:5173...
start http://localhost:5173

echo =========================================================================
echo Both servers launched!
echo - API Backend: http://127.0.0.1:8000
echo - Interactive Frontend: http://localhost:5173
echo - Swagger Docs: http://127.0.0.1:8000/docs
echo =========================================================================

