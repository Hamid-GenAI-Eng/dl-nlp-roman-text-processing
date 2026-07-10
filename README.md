# Roman Urdu Sentiment Intelligence Pipeline

Deep Learning + NLP university project for Roman Urdu sentiment analysis.

## Quick Run on Windows

Double-click:

```bat
run_project.bat
```

Or run manually:

### Backend
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Frontend
```powershell
cd frontend
npm install
npm run dev
```

Open frontend URL shown by Vite, usually:

```text
http://localhost:5173
```

## Important

- Do not run `npm install` inside backend. Backend is Python/FastAPI.
- If you accidentally run `npm install` in root, it will not fail because root package.json is provided.
- API backend runs at `http://127.0.0.1:8000`.

## Models Included

- Naive Bayes
- Logistic Regression
- Linear SVM
- Deep Neural Network / MLP Classifier

The project focuses on NLP preprocessing plus a lightweight neural network model to keep installation stable on normal student laptops without heavy TensorFlow/PyTorch setup.
