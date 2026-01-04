
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import xgboost as xgb

from core.serving.artifacts import load_breakout_artifacts
from core.serving.schemas import BreakoutPredictRequest, BreakoutPredictResponse, BreakoutPrediction


repo_root = Path(__file__).resolve().parents[1]
app = FastAPI()

ARTIFACTS: Dict[str, Any] = {}
FEATURE_COLUMNS: List[str] = []
THRESHOLD: float = 0.5


@app.on_event("startup")
def _startup() -> None:
    global ARTIFACTS, FEATURE_COLUMNS, THRESHOLD
    ARTIFACTS = load_breakout_artifacts(repo_root)
    FEATURE_COLUMNS = list(ARTIFACTS.get("feature_columns", []))
    THRESHOLD = float(ARTIFACTS.get("threshold", 0.5))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=BreakoutPredictResponse)
def predict(req: BreakoutPredictRequest) -> BreakoutPredictResponse:
    if not ARTIFACTS:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    model = ARTIFACTS.get("model")
    preprocessor = ARTIFACTS.get("preprocessor")
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Invalid artifacts: missing model/preprocessor")

    if not FEATURE_COLUMNS:
        raise HTTPException(status_code=500, detail="Invalid artifacts: missing feature_columns")

    try:
        df_in = pd.DataFrame(req.records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid records payload: {e}")

    for c in FEATURE_COLUMNS:
        if c not in df_in.columns:
            df_in[c] = np.nan
    X_df = df_in[FEATURE_COLUMNS].copy()

    try:
        X = preprocessor.transform(X_df)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            dmat = xgb.DMatrix(X)
            proba = model.predict(dmat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    t = float(req.threshold_override) if req.threshold_override is not None else THRESHOLD
    preds = (proba >= t)

    out = []
    for i in range(len(df_in)):
        artist_name = df_in["artist_name"].iloc[i] if "artist_name" in df_in.columns else None
        month = df_in["month"].iloc[i] if "month" in df_in.columns else None

        out.append(
            BreakoutPrediction(
                artist_name=str(artist_name) if artist_name is not None else None,
                month=str(month) if month is not None else None,
                probability=float(proba[i]),
                is_breakout=bool(preds[i]),
            )
        )

    return BreakoutPredictResponse(threshold=t, predictions=out)
