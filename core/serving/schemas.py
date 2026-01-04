
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class BreakoutPredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    return_probabilities: bool = True
    threshold_override: Optional[float] = None

    class Config:
        extra = "allow"


class BreakoutPrediction(BaseModel):
    artist_name: Optional[str] = None
    month: Optional[str] = None
    probability: float
    is_breakout: bool


class BreakoutPredictResponse(BaseModel):
    threshold: float
    predictions: List[BreakoutPrediction]
