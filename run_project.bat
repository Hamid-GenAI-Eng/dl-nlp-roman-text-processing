@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo Roman Urdu Deep Learning + NLP Project
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Please install Python 3.10+ and try again.
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js / npm not found. Please install Node.js LTS and try again.
  pause
  exit /b 1
)

if not exist "backend\venv" (
  echo Creating Python virtual environment...
  python -m venv backend\venv
)

echo Installing backend dependencies...
call backend\venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  npm --prefix frontend install
)

echo Starting backend and frontend...
start "Roman Urdu Backend" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate.bat && uvicorn app:app --reload --host 127.0.0.1 --port 8000"
start "Roman Urdu Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo.
pause
