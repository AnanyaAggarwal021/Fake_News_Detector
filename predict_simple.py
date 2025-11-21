# predict_simple.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load small model
model = pickle.load(open("simple_model.pkl", "rb"))

app = FastAPI(title="Simple Fake News Detector")

class NewsItem(BaseModel):
    text: str

@app.post("/predict")
def predict(item: NewsItem):
    pred = model.predict([item.text])[0]
    return {
        "prediction": "Real" if pred == 1 else "Fake",
        "label": int(pred)
    }

@app.get("/")
def root():
    return {"message": "Simple Fake News Detector API running"}
