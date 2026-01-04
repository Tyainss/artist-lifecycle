
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from pydantic import ConfigDict


class BreakoutPredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    return_probabilities: bool = True
    threshold_override: Optional[float] = None

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "records": [
                    {
                        "month": "2021-06-01",
                        "artist_name": "Ichiko Aoba",
                        "months_since_first_seen": 9,
                        "is_first_month": 0,
                        "plays_t": 14,
                        "plays_per_track_t": 1.0,
                        "days_active_t": 1,
                        "last_play_gap_days_t": 9.04582175925926,
                        "track_novelty_rate_t": 0.0,
                        "share_t": 0.0040287769784172,
                        "total_monthly_scrobbles": 3475,
                        "plays_prev3_mean": 0.3333333333333333,
                        "delta_1m": 14,
                        "trend_slope_3m": 6.5,
                        "active_months_count_before_t": 7
                    }
                ],
                "return_probabilities": True,
                "threshold_override": None,
            }
        },
    )


class BreakoutPrediction(BaseModel):
    artist_name: Optional[str] = None
    month: Optional[str] = None
    probability: float
    is_breakout: bool


class BreakoutPredictResponse(BaseModel):
    threshold: float
    predictions: List[BreakoutPrediction]
