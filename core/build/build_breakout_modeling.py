import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class BreakoutModelingConfig:
    horizon_months: int
    cold_start_buffer_months: int

    active_plays_min: int
    discovered_within_months: int
    lookback_months: int

    core_plays_min: int
    core_share_min: float


def _parse_breakout_modeling_config(breakout_cfg: dict) -> BreakoutModelingConfig:
    labels_cfg = breakout_cfg.get("labels", {})
    eligibility_cfg = breakout_cfg.get("eligibility", {})
    core_cfg = breakout_cfg.get("core_definition", {})

    horizon_months = int(labels_cfg.get("horizon_months", 2))
    cold_start_buffer_months = int(labels_cfg.get("cold_start_buffer_months", 6))

    active_plays_min = int(eligibility_cfg.get("active_plays_min", 3))
    discovered_within_months = int(eligibility_cfg.get("discovered_within_months", 6))
    lookback_months = int(eligibility_cfg.get("lookback_months", 3))

    core_plays_min = int(core_cfg.get("plays_min", 20))
    core_share_min = float(core_cfg.get("share_min", 0.01))

    if horizon_months <= 0:
        raise ValueError("breakout.labels.horizon_months must be > 0")
    if cold_start_buffer_months < 0:
        raise ValueError("breakout.labels.cold_start_buffer_months must be >= 0")
    if lookback_months <= 0:
        raise ValueError("breakout.eligibility.lookback_months must be > 0")
    if discovered_within_months < 0:
        raise ValueError("breakout.eligibility.discovered_within_months must be >= 0")
    if active_plays_min < 0:
        raise ValueError("breakout.eligibility.active_plays_min must be >= 0")

    return BreakoutModelingConfig(
        horizon_months=horizon_months,
        cold_start_buffer_months=cold_start_buffer_months,
        active_plays_min=active_plays_min,
        discovered_within_months=discovered_within_months,
        lookback_months=lookback_months,
        core_plays_min=core_plays_min,
        core_share_min=core_share_min,
    )


def _validate_features_input(df: pd.DataFrame) -> None:
    required = [
        "artist_name",
        "month",
        "plays_t",
        "share_t",
        "months_since_first_seen",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in breakout features: {missing}")


def build_breakout_modeling_table(
    features: pd.DataFrame,
    breakout_cfg: dict,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Build a model-ready breakout table (eligible + labeled + censored + cold-start applied).

    Unit of analysis: (artist_name, month) snapshot rows.

    Output includes:
      - artist_name
      - month
      - y (breakout label)
      - leakage-safe feature columns (as provided by build_breakout_features)
    """
    cfg = _parse_breakout_modeling_config(breakout_cfg)
    log = logger or logging.getLogger(__name__)

    df = features.copy()
    _validate_features_input(df)

    df["month"] = pd.to_datetime(df["month"], errors="raise")
    df = df.sort_values(["artist_name", "month"], kind="stable").reset_index(drop=True)

    # Core definition (used only for label creation)
    df["is_core"] = (df["plays_t"] >= cfg.core_plays_min) | (df["share_t"] >= cfg.core_share_min)

    # Label: y(t)=1 if core(t+1) ... core(t+H)
    core_map = df[["artist_name", "month", "is_core"]].copy()

    future_flags = []
    for k in range(1, cfg.horizon_months + 1):
        shifted = core_map.copy()
        shifted["month"] = shifted["month"] - pd.DateOffset(months=k)
        shifted = shifted.rename(columns={"is_core": f"core_t_plus_{k}"})
        future_flags.append(shifted)

    df_labeled = df.drop(columns=["is_core"]).copy()
    for shifted in future_flags:
        df_labeled = df_labeled.merge(shifted, on=["artist_name", "month"], how="left")

    for k in range(1, cfg.horizon_months + 1):
        col = f"core_t_plus_{k}"
        df_labeled[col] = df_labeled[col].fillna(False).astype(bool)

    df_labeled["y"] = False
    for k in range(1, cfg.horizon_months + 1):
        df_labeled["y"] = df_labeled["y"] | df_labeled[f"core_t_plus_{k}"]
    df_labeled["y"] = df_labeled["y"].astype(int)

    # Censor last H months (no future to label)
    max_month = df_labeled["month"].max()
    label_cutoff = max_month - pd.DateOffset(months=cfg.horizon_months)
    df_labeled = df_labeled.loc[df_labeled["month"] <= label_cutoff].copy()

    # Eligibility rules (Option A)
    active_now = df_labeled["plays_t"] >= cfg.active_plays_min
    recently_discovered = df_labeled["months_since_first_seen"] <= cfg.discovered_within_months

    base = df_labeled[["artist_name", "month", "plays_t", "share_t"]].copy()

    lags = []
    for lag in range(1, cfg.lookback_months):
        shifted = base.copy()
        shifted["month"] = shifted["month"] + pd.DateOffset(months=lag)
        shifted = shifted.rename(
            columns={
                "plays_t": f"plays_t_minus_{lag}",
                "share_t": f"share_t_minus_{lag}",
            }
        )
        lags.append(shifted)

    df_elig = df_labeled.merge(lags[0], on=["artist_name", "month"], how="left") if lags else df_labeled
    for shifted in lags[1:]:
        df_elig = df_elig.merge(shifted, on=["artist_name", "month"], how="left")

    play_cols = ["plays_t"] + [f"plays_t_minus_{lag}" for lag in range(1, cfg.lookback_months)]
    share_cols = ["share_t"] + [f"share_t_minus_{lag}" for lag in range(1, cfg.lookback_months)]

    for c in play_cols + share_cols:
        if c not in df_elig.columns:
            df_elig[c] = 0
    df_elig[play_cols] = df_elig[play_cols].fillna(0)
    df_elig[share_cols] = df_elig[share_cols].fillna(0)

    max_plays_recent = df_elig[play_cols].max(axis=1)
    max_share_recent = df_elig[share_cols].max(axis=1)

    not_core_recently = (max_plays_recent < cfg.core_plays_min) & (max_share_recent < cfg.core_share_min)

    df_elig["is_eligible"] = active_now & recently_discovered & not_core_recently

    # Cold-start buffer: drop first N months globally from modeling
    if cfg.cold_start_buffer_months > 0:
        min_month = df_elig["month"].min()
        cold_start_cutoff = min_month + pd.DateOffset(months=cfg.cold_start_buffer_months)
        df_elig = df_elig.loc[df_elig["month"] >= cold_start_cutoff].copy()

    out = df_elig.loc[df_elig["is_eligible"]].copy()

    # Keep features + label, drop intermediate helper cols and absolute date feature
    drop_cols = [
        "is_eligible",
        *[f"core_t_plus_{k}" for k in range(1, cfg.horizon_months + 1)],
        *[f"plays_t_minus_{lag}" for lag in range(1, cfg.lookback_months)],
        *[f"share_t_minus_{lag}" for lag in range(1, cfg.lookback_months)],
    ]
    if "first_seen_month" in out.columns:
        drop_cols.append("first_seen_month")

    drop_cols = [c for c in drop_cols if c in out.columns]
    out = out.drop(columns=drop_cols)

    out = out.sort_values(["month", "artist_name"], kind="stable").reset_index(drop=True)

    log.info(
        "Built breakout modeling table | rows=%s | months=%s->%s | positives=%s (%.2f%%)",
        f"{len(out):,}",
        out["month"].min().date() if len(out) else None,
        out["month"].max().date() if len(out) else None,
        int(out["y"].sum()) if "y" in out.columns else None,
        (out["y"].mean() * 100) if len(out) and "y" in out.columns else 0.0,
    )

    return out
