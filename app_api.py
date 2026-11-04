
import os
import time
import joblib

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API de prédiction des sentiments des commentaires YouTube",
    version="1.0.0"
)

# Autoriser les requêtes de l'extension Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Chargement du modèle
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")


class PredictionRequest(BaseModel):
    comments: List[str]


@app.get("/")
def home():
    return {"message": "YouTube Sentiment Analysis API"}


@app.get("/health")
def health():
    return {
        "status": "OK" if model is not None else "ERROR",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict_batch")
def predict_batch(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas disponible."
        )

    comments = request.comments

    if not comments:
        raise HTTPException(
            status_code=400,
            detail="La liste des commentaires est vide."
        )

    if len(comments) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 commentaires par requête."
        )

    if any(not isinstance(c, str) or not c.strip() for c in comments):
        raise HTTPException(
            status_code=400,
            detail="Les commentaires doivent être des textes non vides."
        )

    start_time = time.perf_counter()

    try:
        predictions = model.predict(comments)

        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(comments)

        results = []

        for i, comment in enumerate(comments):

            label = int(predictions[i])

            sentiments = {
                -1: "Négatif",
                0: "Neutre",
                1: "Positif"
            }

            sentiment = sentiments.get(label, "Inconnu")

            if probabilities is not None:
                confidence = float(max(probabilities[i])) * 100
            else:
                confidence = None

            results.append({
                "comment": comment,
                "sentiment": sentiment,
                "label": label,
                "confidence": round(confidence, 2)
                if confidence is not None else None
            })

        statistics = {}

        for sentiment in ["Négatif", "Neutre", "Positif"]:

            count = sum(
                1 for result in results
                if result["sentiment"] == sentiment
            )

            percentage = (count / len(comments)) * 100

            statistics[sentiment] = {
                "count": count,
                "percentage": round(percentage, 2)
            }

        processing_time = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "total_comments": len(comments),
            "results": results,
            "statistics": statistics,
            "processing_time_ms": round(processing_time, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(e)}"
        )
