from __future__ import annotations

import torch  # IMPORT TORCH FIRST TO PREVENT DLL CONFLICT (WinError 1114)
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModel

from preprocessing import preprocess_text

class LSTMClassifierModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Take output of last time step
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        return logits

class PyTorchLSTMClassifier:
    def __init__(self, vocab_size=3000, embed_dim=64, hidden_dim=64, max_len=40, lr=0.005, epochs=10, batch_size=64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.word2idx = {}
        self.classes_ = []
        self.model = None
        self.device = torch.device("cpu")
        
    def _tokenize(self, text):
        return str(text).lower().split()
        
    def _build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        
        most_common = counter.most_common(self.vocab_size)
        self.word2idx = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
        
    def _text_to_sequence(self, texts):
        sequences = []
        for text in texts:
            tokens = self._tokenize(text)
            seq = [self.word2idx.get(tok, 0) for tok in tokens][:self.max_len]
            # Pre-pad sequence (pad from left) to align final state with the last actual token
            if len(seq) < self.max_len:
                seq = [0] * (self.max_len - len(seq)) + seq
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.long)

    def fit(self, X, y):
        X = list(X)
        y = list(y)
        
        self.classes_ = sorted(list(set(y)))
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}
        y_indices = torch.tensor([class_to_idx[val] for val in y], dtype=torch.long)
        
        self._build_vocab(X)
        X_seq = self._text_to_sequence(X)
        
        self.model = LSTMClassifierModel(
            vocab_size=len(self.word2idx) + 1,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.classes_)
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = torch.utils.data.TensorDataset(X_seq, y_indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        X = list(X)
        self.model.eval()
        X_seq = self._text_to_sequence(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[idx] for idx in pred_indices])

class BERTBiLSTMClassifierModel(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        bert_hidden_size = self.bert.config.hidden_size
        self.lstm = nn.LSTM(bert_hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = bert_out.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        
        # Pool across sequence dimension
        pooled_out = torch.mean(lstm_out, dim=1)
        logits = self.fc(pooled_out)
        return logits

class PyTorchBERTBiLSTMClassifier:
    def __init__(self, bert_model_name="distilbert-base-uncased", hidden_dim=64, max_len=40, lr=0.005, epochs=5, batch_size=32):
        self.bert_model_name = bert_model_name
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.classes_ = []
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        
    def fit(self, X, y):
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        X = list(X)
        y = list(y)
        
        self.classes_ = sorted(list(set(y)))
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}
        y_indices = torch.tensor([class_to_idx[val] for val in y], dtype=torch.long)
        
        encodings = self.tokenizer(X, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        self.model = BERTBiLSTMClassifierModel(
            bert_model_name=self.bert_model_name,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.classes_)
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, y_indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_ids, batch_mask, batch_y in loader:
                batch_ids = batch_ids.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_ids, batch_mask)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        X = list(X)
        self.model.eval()
        encodings = self.tokenizer(X, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[idx] for idx in pred_indices])

SAMPLE_DATA = [
    ("mashallah product bohat acha recommended", "Positive"),
    ("service bohat behtareen fast delivery", "Positive"),
    ("camera quality bohat achi hai", "Positive"),
    ("bilkul zabardast experience", "Positive"),
    ("product pyara aur useful", "Positive"),
    ("alhamdulillah order bohat acha mila", "Positive"),
    ("fast charging aur battery achi", "Positive"),
    ("excellent product paisay vasool", "Positive"),
    ("quality best hai recommend karta hun", "Positive"),
    ("bohat acha app easy to use", "Positive"),
    ("product kharab nikla", "Negative"),
    ("delivery bohat slow aur service bekar", "Negative"),
    ("camera quality buri hai", "Negative"),
    ("bilkul recommend nahi karta", "Negative"),
    ("battery timing kharab", "Negative"),
    ("waste of money product bekar", "Negative"),
    ("order late aya aur packing kharab", "Negative"),
    ("app crash hoti hai", "Negative"),
    ("support response nahi deta", "Negative"),
    ("quality low aur price zyada", "Negative"),
    ("product normal hai", "Neutral"),
    ("delivery time average tha", "Neutral"),
    ("abhi use kar raha hun", "Neutral"),
    ("features theek hain", "Neutral"),
    ("price market jaisi hai", "Neutral"),
    ("order receive ho gaya", "Neutral"),
    ("color same hai picture jaisa", "Neutral"),
    ("maine product check karna hai", "Neutral"),
    ("service average hai", "Neutral"),
    ("normal experience tha", "Neutral"),
]

# Add Roman Urdu noisy variants so preprocessing/demo behaves realistically.
AUGMENTED = [
    ("MshaAllah! Ye product boht hi achha hy, bilkul slow nhi hai? 100% recommended.", "Positive"),
    ("Masha Allah product bht acha hai", "Positive"),
    ("ye product bohot behtreen hai", "Positive"),
    ("bilkul accha phone hai", "Positive"),
    ("delivery sloww hai aur product khrab", "Negative"),
    ("nhi recommend karta product kharaab", "Negative"),
    ("boht bekaar service", "Negative"),
    ("product theek hai kuch khaas nahi", "Neutral"),
]


def build_synthetic_dataset():
    positive_terms = ["bohat acha", "behtareen", "zabardast", "recommended", "excellent", "best quality", "fast delivery", "paisay vasool", "pyara product", "alhamdulillah acha"]
    negative_terms = ["kharab", "bohat slow", "bekar", "recommend nahi", "waste of money", "buri quality", "late delivery", "battery kharab", "low quality", "support nahi"]
    neutral_terms = ["normal", "average", "theek", "receive ho gaya", "abhi use kar raha", "price same", "color same", "features theek", "ordinary", "kuch khas nahi"]
    subjects = ["product", "service", "app", "camera", "delivery", "battery", "quality", "order", "phone", "software"]
    rows = []
    for term in positive_terms:
        for sub in subjects:
            rows.append((f"mashallah {sub} {term}", "Positive"))
    for term in negative_terms:
        for sub in subjects:
            rows.append((f"{sub} {term}", "Negative"))
    for term in neutral_terms:
        for sub in subjects:
            rows.append((f"{sub} {term}", "Neutral"))
    rows.extend(SAMPLE_DATA + AUGMENTED)
    return rows


@dataclass
class PredictionResult:
    model: str
    label: str
    confidence: float


class SentimentModelManager:
    def __init__(self) -> None:
        self.models: Dict[str, Pipeline] = {}
        self.metrics: List[dict] = []
        self.trained = False
        # Do not automatically train on startup, as downloading BERT blocks the server
        # self.train()

    def _dataset(self) -> pd.DataFrame:
        import os
        csv_path = "Roman_Urdu_DataSet.csv"
        # Check relative to this file's location
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "Roman_Urdu_DataSet.csv")
            
        if os.path.exists(csv_path):
            try:
                # Load real dataset
                df = pd.read_csv(csv_path, header=None, names=["comment", "sentiment", "extra"], encoding="utf-8")
                # Drop rows with missing text or sentiment
                df = df.dropna(subset=["comment", "sentiment"])
                # Strip and clean labels
                df["sentiment"] = df["sentiment"].str.strip().str.replace("Neative", "Negative")
                # Filter only correct sentiment classes
                df = df[df["sentiment"].isin(["Positive", "Negative", "Neutral"])]
                
                # Sample dataset to keep training extremely fast yet representative (stratified sampling)
                df = df.groupby("sentiment", group_keys=False).apply(
                    lambda x: x.sample(min(len(x), 1500), random_state=42),
                    include_groups=True
                )
                df = df.reset_index(drop=True)
                df = df.rename(columns={"comment": "comment", "sentiment": "sentiment"})
                
                # Append user's custom challenging reviews for robust learning on edge cases
                custom_reviews = [
                    ("Yaar app ka product overall acha hai, lekin delivery itni late hui k mood hi off ho gaya. Packaging 10/10 thi magar experience utna khas nahi.", "Negative"),
                    ("Wah bhai wah... itni zabardast service thi k 3 din call hi nahi uthai kisi ne 😂👏 outstanding!", "Negative"),
                    ("Honestly speaking interface clean hai but UX is totally messed up. Login process unnecessary complicated lagta hai.", "Negative"),
                    ("Yarrr me ny kl ordr kia tha mgr abi tk nhe aya... ye kya scn h?? 😑", "Negative"),
                    ("Mujhay ye bohat acha laga.", "Positive"),
                    ("Mujhe ye bohat acha laga.", "Positive"),
                    ("Mjhy ye boht acha lga.", "Positive"),
                    ("Muje bohat acha lga.", "Positive"),
                    ("😍❤️🔥🔥💯", "Positive"),
                    ("Thanks... umeed nahi thi itni disappointing service mile gi.", "Negative"),
                    ("Product quality zabardast hai, bas customer support improve kar lo to perfect ho jaye.", "Positive"),
                    ("Seriously bhai backend optimization ki zarurat hai warna app users lose kare gi.", "Negative"),
                    ("Ye feature full scene on hai boss 😎", "Positive"),
                    ("LOL bhai ye kya bana diya 😂 fr fr expected better ngl", "Negative"),
                    (" Mujhe ye bilkul bura nahi laga.", "Positive"),
                    ("Aisa nahi hai k service buri nahi thi.", "Negative"),
                    ("Pehli dafa use kiya to acha laga, doosri dafa itna issue aya k uninstall hi kar diya.", "Negative"),
                    ("Dekho honestly agar sirf design ki baat karun to app bohat premium lagti hai, animations smooth hain aur colors bhi ache hain. Lekin performance itni inconsistent hai k kabhi sab kuch fast hota hai aur kabhi loading hi khatam nahi hoti. Overall average experience.", "Neutral"),
                    ("MashAllah itni fast delivery thi k parcel mere birthday k 5 din baad mila 😂", "Negative"),
                    ("Mujhe koi khas expectation nahi thi, lekin surprisingly acha nikla.", "Positive"),
                    ("Chalo theek hi hai... guzara ho jata hai.", "Neutral"),
                    ("Bhai software ka dashboard toh clean hai but performance literally 'Allah hi hafiz' 😂", "Negative"),
                    ("Interface mast hai lekin developer ne optimization ka janaza nikal diya.", "Negative"),
                    ("Sirf mere sath hi issue aa raha hai ya sab ka app itna slow chal raha hai?", "Negative"),
                    ("Hmmm... pata nahi acha bhi laga aur nahi bhi.", "Neutral"),
                    ("Bohattttttttt achaaaaaa haiiiiiiii ❤️", "Positive"),
                    ("BhAi Ye KaFi AcHa Ha 😂", "Positive"),
                    ("yebohatachahaimagardeliverylatehai", "Negative"),
                    ("Achaaa", "Positive"),
                    ("Acha", "Positive"),
                    ("Accha", "Positive"),
                    ("Achha", "Positive"),
                    ("Achaa", "Positive"),
                    ("Inshallah next update better hogi.", "Positive"),
                    ("In Sha Allah next update better hogi.", "Positive"),
                    ("inshAllah next update better hogi.", "Positive"),
                    ("Great job 👍 bas pura system hi crash kar diya.", "Negative"),
                    ("Result dekh k khushi bhi hui aur thori frustration bhi, kyun k expected aur bhi zyada tha.", "Neutral"),
                    ("Bhai sach bolun to mujhe pehle laga ye bhi baqi apps ki tarah fake promises kare gi, lekin use karne k baad opinion change ho gaya. Haan kuch bugs hain, notifications kabhi time pe nahi aati aur search bhi weak hai, lekin overall kaafi useful app hai. Agar developers regular updates dete rahein to future bright lag raha hai.", "Positive"),
                    ("Wah kya baat hai! Paisay bhi le liye aur service bhi zero. Dil khush kar diya 😂👏", "Negative"),
                    ("Itna bhi bura nahi tha jitna log keh rahe thay.", "Positive"),
                    ("Acha hai... agar crash na kare to.", "Negative"),
                    ("Product 9/10, support 2/10, delivery 10/10, overall pata nahi.", "Neutral"),
                    ("Allah apko aur taraqi de ❤️ bas customer support improve kar lo.", "Positive")
                ]
                
                df_custom = pd.DataFrame(custom_reviews, columns=["comment", "sentiment"])
                # Multiply custom samples to increase their decision weight
                df_custom = pd.concat([df_custom] * 5, ignore_index=True)
                
                # Append to dataset
                df = pd.concat([df[["comment", "sentiment"]], df_custom], ignore_index=True)
                df = df.reset_index(drop=True)
            except Exception as e:
                # Fallback to synthetic if any error occurs
                print(f"Error loading CSV, falling back to synthetic dataset: {e}")
                df = pd.DataFrame(build_synthetic_dataset(), columns=["comment", "sentiment"])
        else:
            print("CSV not found, falling back to synthetic dataset.")
            df = pd.DataFrame(build_synthetic_dataset(), columns=["comment", "sentiment"])
            
        df["clean_comment"] = df["comment"].apply(preprocess_text)
        return df

    def _make_pipeline(self, estimator) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=2500, ngram_range=(1, 2))),
            ("model", estimator),
        ])

    def train(self) -> List[dict]:
        df = self._dataset()
        X = df["clean_comment"]
        y = df["sentiment"]
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )

        model_specs = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Linear SVM": LinearSVC(),
            "Deep Neural Network (MLP)": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=200,
                random_state=42,
                early_stopping=True,
            ),
            "PyTorch LSTM": PyTorchLSTMClassifier(epochs=10, lr=0.005),
            "BERT + BiLSTM": PyTorchBERTBiLSTMClassifier(epochs=5, lr=0.005),
        }

        metrics = []
        for name, estimator in model_specs.items():
            if name in ["PyTorch LSTM", "BERT + BiLSTM"]:
                estimator.fit(X_train, y_train)
                preds = estimator.predict(X_test)
                self.models[name] = estimator
            else:
                pipe = self._make_pipeline(estimator)
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                self.models[name] = pipe

            metrics.append({
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, preds)) * 100, 2),
                "f1_score": round(float(f1_score(y_test, preds, average="weighted")) * 100, 2),
                "type": "Deep Learning" if ("Neural" in name or "LSTM" in name or "BERT" in name) else "Machine Learning",
            })
            
        self.metrics = metrics
        self.trained = True
        return metrics

    def predict(self, text: str) -> dict:
        if not self.trained:
            self.train()
        clean = preprocess_text(text)
        results = []
        for name, pipe_or_model in self.models.items():
            if name in ["PyTorch LSTM", "BERT + BiLSTM"]:
                label = pipe_or_model.predict([clean])[0]
                probs = pipe_or_model.predict_proba([clean])[0]
                confidence = float(np.max(probs)) * 100
            else:
                label = pipe_or_model.predict([clean])[0]
                confidence = 0.0
                model = pipe_or_model.named_steps["model"]
                if hasattr(model, "predict_proba"):
                    probs = pipe_or_model.predict_proba([clean])[0]
                    confidence = float(np.max(probs)) * 100
                elif hasattr(model, "decision_function"):
                    score = pipe_or_model.decision_function([clean])
                    confidence = min(99.0, max(55.0, float(np.max(np.abs(score))) * 20 + 55))
            results.append({"model": name, "label": str(label), "confidence": round(confidence, 2)})

        labels = [r["label"] for r in results]
        consensus = max(set(labels), key=labels.count)
        return {"input": text, "clean_text": clean, "consensus": str(consensus), "predictions": results}
