Set-Location $PSScriptRoot
Write-Host "Roman Urdu Deep Learning + NLP Project" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Write-Error "Python not found"; exit 1 }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Write-Error "Node.js/npm not found"; exit 1 }

if (-not (Test-Path "backend/venv")) { python -m venv backend/venv }
& "backend/venv/Scripts/python.exe" -m pip install --upgrade pip
& "backend/venv/Scripts/pip.exe" install -r backend/requirements.txt

if (-not (Test-Path "frontend/node_modules")) { npm --prefix frontend install }

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot/backend'; ./venv/Scripts/activate; uvicorn app:app --reload --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot/frontend'; npm run dev"
Write-Host "Backend: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Green
