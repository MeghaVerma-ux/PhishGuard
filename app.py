
from pathlib import Path
import logging
import json
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model.joblib"
HTML_PATH = BASE_DIR / "static" / "index.html"
SAMPLE_PATH = BASE_DIR / "data" / "sample_input.json"
IMPORTANCE_PATH = BASE_DIR / "results" / "feature_importance.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PhishGuard - Phishing Detection",
    description="Ensemble Learning Based Website Classification System",
    version="1.0.0"
)

# Load trained model
bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]
feature_names = list(bundle["feature_names"])
best_model_name = bundle["best_model_name"]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": best_model_name,
        "features_required": len(feature_names)
    }


@app.get("/")
def home():
    if not HTML_PATH.exists() or HTML_PATH.stat().st_size == 0:
        raise HTTPException(
            status_code=404,
            detail="Website file missing or empty"
        )

    return FileResponse(HTML_PATH)


@app.get("/sample")
def sample():
    try:
        if not SAMPLE_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Sample input file not found"
            )

        with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
            sample_data = json.load(file)

        return sample_data

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Sample loading failed")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/feature-importance")
def get_feature_importance():
    try:
        if not IMPORTANCE_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Feature importance file not found"
            )

        df = pd.read_csv(IMPORTANCE_PATH).head(10)

        records = df.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0).to_dict(orient="records")

        return {
            "model": best_model_name,
            "method": "Permutation Importance",
            "features": records
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Feature importance loading failed")
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict")
def predict(request: PredictRequest):

    received = request.features

    missing = set(feature_names) - set(received)
    extra = set(received) - set(feature_names)

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {list(missing)[:10]}"
        )

    if extra:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected features: {list(extra)[:10]}"
        )

    try:
        values = [received[name] for name in feature_names]

        if not np.isfinite(values).all():
            raise HTTPException(
                status_code=422,
                detail="Features must contain finite numbers"
            )

        input_df = pd.DataFrame(
            [values],
            columns=feature_names
        )

        prediction = int(pipeline.predict(input_df)[0])

        confidence = None

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)[0]
            probability = float(max(probabilities))

            confidence = round(
                probability * 100 if probability <= 1 else probability,
                2
            )

        return {
            "prediction": prediction,
            "predicted_label": prediction,
            "result": f"Class {prediction}",
            "confidence": confidence,
            "model": best_model_name,
            "message": "Prediction generated successfully"
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(error))
