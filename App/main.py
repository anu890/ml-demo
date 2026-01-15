from fastapi import FastAPI
import pickle
import pandas as pd
import os

app = FastAPI()
MODEL_PATH = "Model/dummy_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

@app.get("/")
def home():
    return {"status": "API is online", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: dict):
    if not model:
        return {"error": "Model file not found."}
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
