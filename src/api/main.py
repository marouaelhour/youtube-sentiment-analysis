
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd
import time
import os

from src.api.schemas import PredictionRequest



app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API de classification des sentiments des commentaires YouTube",
    version="1.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



MODEL_PATH = "models/best_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Modèle introuvable. Exécute d'abord train_model.py"
    )

model = joblib.load(MODEL_PATH)



labels = {
    -1: "Négatif",
    0: "Neutre",
    1: "Positif"
}



@app.get("/health")
def health():

    return {
        "status": "OK",
        "model_loaded": True,
        "model_path": MODEL_PATH
    }



@app.post("/predict_batch")
def predict_batch(request: PredictionRequest):

    if not request.comments:
        raise HTTPException(
            status_code=400,
            detail="La liste des commentaires est vide"
        )

    if len(request.comments) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 commentaires par batch"
        )

    try:
        start = time.perf_counter()

      
        comments = [
            str(comment).strip()
            for comment in request.comments
        ]

        if any(not comment for comment in comments):
            raise HTTPException(
                status_code=400,
                detail="Un commentaire est vide"
            )

       
        predictions = model.predict(comments)

        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(comments)

            confidences = probabilities.max(axis=1) * 100

        elif hasattr(model, "decision_function"):
            scores = model.decision_function(comments)

            
            if scores.ndim == 1:
                scores = pd.DataFrame(scores).values

            exp_scores = pd.DataFrame(scores).apply(
                lambda column: column,
                axis=0
            ).values

            exp_scores = __import__("numpy").exp(
                exp_scores - exp_scores.max(axis=1, keepdims=True)
            )

            probabilities = exp_scores / exp_scores.sum(
                axis=1,
                keepdims=True
            )

            confidences = probabilities.max(axis=1) * 100

        else:
            confidences = [None] * len(predictions)

       
        results = []

        for i, prediction in enumerate(predictions):

            results.append({
                "comment": comments[i],
                "sentiment": labels[int(prediction)],
                "label": int(prediction),
                "confidence": (
                    round(float(confidences[i]), 2)
                    if confidences[i] is not None
                    else None
                )
            })

      
        counts = pd.Series(predictions).value_counts()

        total = len(predictions)

        statistics = {
            labels[label]: {
                "count": int(counts.get(label, 0)),
                "percentage": round(
                    int(counts.get(label, 0)) / total * 100,
                    2
                )
            }
            for label in [-1, 0, 1]
        }

        elapsed = (time.perf_counter() - start) * 1000

        return {
            "total_comments": total,
            "results": results,
            "statistics": statistics,
            "processing_time_ms": round(elapsed, 2)
        }

    except HTTPException:
        raise

    except Exception as e:
        print("Erreur de prédiction :", str(e))

        raise HTTPException(
            status_code=500,
            detail="Erreur pendant la prédiction"
        )
