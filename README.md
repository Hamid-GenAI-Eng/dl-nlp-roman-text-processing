# Roman Urdu Sentiment Intelligence Pipeline

Deep Learning + NLP university project for Roman Urdu sentiment analysis.

This project implements a complete pipeline to clean, process, and classify the sentiment of Roman Urdu text using a variety of Machine Learning and Deep Learning architectures.

## The BERT + BiLSTM Implementation

To enhance the sentiment classification quality and capture the deep contextual meaning of complex sentences (like sarcasm or mixed sentiments), we have integrated a **BERT + BiLSTM** architecture.

### How it works:
1. **BERT (Feature Extraction):** We utilize Hugging Face's `distilbert-base-uncased` model. Instead of relying on traditional TF-IDF token counting, the text is passed through the pre-trained BERT model to generate rich, dense, contextual embeddings for every word. To ensure the model runs efficiently on standard computers without GPUs, the core BERT weights are "frozen" during our training phase.
2. **BiLSTM (Sequence Processing):** The contextual embeddings extracted from BERT are then fed into a PyTorch **Bidirectional Long Short-Term Memory (BiLSTM)** network. The BiLSTM processes the sentence from both left-to-right and right-to-left. This allows the network to understand the overall sequence and dependencies of the words.
3. **Classification:** Finally, we use "mean pooling" across the BiLSTM outputs to create a single sentence representation, which is passed through a fully connected linear layer to classify the sentiment into **Positive**, **Negative**, or **Neutral**.

## Step-by-Step Guide to Run the Project

### Method 1: Quick Start (Recommended)
Simply double-click the `run_project.bat` script in the root directory. This will automatically create the virtual environment, install all Python and Node.js dependencies, and start both the backend and frontend servers in separate windows.

### Method 2: Manual Setup

**Step 1: Start the Backend (FastAPI)**
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
*Note: We have disabled automatic training on startup so the server boots instantly. You will trigger the deep learning training manually via the frontend UI.*

**Step 2: Start the Frontend (React + Vite)**
Open a new terminal window:
```powershell
cd frontend
npm install
npm run dev
```

**Step 3: Access the UI & Train the Models**
1. Open your browser and navigate to the frontend URL (usually `http://localhost:5173`).
2. Go to the **ML + DL Training Center** tab (the brain icon).
3. Click the **"Retrain Models"** button. The backend will automatically download the `distilbert` pre-trained weights (approx 250MB) and train the neural networks. Once finished, you'll see the accuracy and F1 scores!
4. Head to the **Sentiment Playground** to test the system with your own Roman Urdu reviews!

## Models Included

- **Classical ML Models**: Naive Bayes, Logistic Regression, Linear SVM
- **Basic Deep Learning**: Scikit-Learn MLP Classifier, PyTorch LSTM Classifier
- **Advanced Deep Learning**: **BERT + BiLSTM** (Highest Accuracy)

## Important Notes

- Do not run `npm install` inside the `backend` directory. Backend is strictly Python/FastAPI.
- The API backend runs at `http://127.0.0.1:8000`.
