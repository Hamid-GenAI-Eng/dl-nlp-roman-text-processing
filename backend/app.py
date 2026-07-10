from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocessing import preprocess_steps, preprocess_text
from model_manager import SentimentModelManager

app = FastAPI(title="Roman Urdu Deep Learning + NLP API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = SentimentModelManager()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Roman Urdu Deep Learning + NLP API is running", "status": "ok"}

@app.get("/status")
def status():
    return {"status": "running", "engine": "FastAPI + Scikit-learn Neural Network", "trained": manager.trained}

@app.post("/preprocess")
def preprocess(req: TextRequest):
    return {"steps": preprocess_steps(req.text), "final_output": preprocess_text(req.text)}

@app.get("/train")
def train():
    return {"metrics": manager.train()}

@app.get("/metrics")
def metrics():
    return {"metrics": manager.metrics}

@app.post("/predict")
def predict(req: TextRequest):
    return manager.predict(req.text)
