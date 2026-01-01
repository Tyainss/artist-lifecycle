import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================
# Validation
# ===========================
def validate_snapshots_input(df: pd.DataFrame) -> None:
    required = [
        "month",
        "artist_name",
        "plays_t",
        "unique_tracks_t",
        "plays_per_track_t",
        "days_active_t",
        "last_play_gap_days_t",
        "share_t",
        "total_monthly_scrobbles",
        "first_seen_month",
        "months_since_first_seen",
        "is_first_month",
        "track_novelty_rate_t",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required snapshot columns: {missing}")


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["artist_name", "month"], kind="stable").reset_index(drop=True)

def ensure_full_month_grid_per_artist(snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures each artist has a row for every calendar month from first_seen_month to the global max month.
    """
    df = snapshots.copy()

    max_month = df["month"].max()

    month_totals = (
        df[["month", "total_monthly_scrobbles"]]
        .drop_duplicates(subset=["month"])
        .sort_values("month", kind="stable")
        .reset_index(drop=True)
    )

    artists = (
        df[["artist_name", "first_seen_month"]]
        .drop_duplicates(subset=["artist_name"])
        .sort_values("artist_name", kind="stable")
        .reset_index(drop=True)
    )

    rows = []
    for artist_name, first_seen_month in artists.itertuples(index=False):
        months = pd.date_range(start=first_seen_month, end=max_month, freq="MS")
        tmp = pd.DataFrame({"artist_name": artist_name, "month": months})
        tmp["first_seen_month"] = first_seen_month
        rows.append(tmp)

    grid = pd.concat(rows, ignore_index=True)
    out = grid.merge(df, on=["artist_name", "month", "first_seen_month"], how="left", sort=False)

    # Fill month totals for inserted rows
    out = out.drop(columns=["total_monthly_scrobbles"]).merge(month_totals, on="month", how="left", sort=False)

    # Inserted months = no plays
    fill_zero_int = ["plays_t", "unique_tracks_t", "days_active_t", "is_first_month"]
    for c in fill_zero_int:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype("Int64")

    fill_zero_float = ["plays_per_track_t", "track_novelty_rate_t", "share_t"]
    for c in fill_zero_float:
        if c in out.columns:
            out[c] = out[c].fillna(0.0).astype(float)

    # last_play_gap_days_t: undefined if no plays; keep NaN for now
    if "last_play_gap_days_t" in out.columns:
        out["last_play_gap_days_t"] = out["last_play_gap_days_t"].astype(float)

    # Recompute months_since_first_seen and is_first_month for inserted rows
    out["months_since_first_seen"] = (
        (out["month"].dt.year - out["first_seen_month"].dt.year) * 12
        + (out["month"].dt.month - out["first_seen_month"].dt.month)
    ).astype("Int64")
    out["is_first_month"] = (out["months_since_first_seen"] == 0).astype("Int64")


    return _ensure_sorted(out)

# ===========================
# Single-column helpers
# ===========================
def add_plays_lag_1m(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_sorted(df.copy())
    g = out.groupby("artist_name", sort=False)
    out["plays_t_minus_1"] = g["plays_t"].shift(1).fillna(0).astype("Int64")
    return out


def add_plays_prev3_mean(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_sorted(df.copy())
    g = out.groupby("artist_name", sort=False, group_keys=False)

    # strict look-back: exclude current month (shift(1)), then rolling mean over 3 months
    out["plays_prev3_mean"] = (
        g["plays_t"].apply(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        .fillna(0.0)
        .astype(float)
    )
    return out


def add_delta_1m(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "plays_t_minus_1" not in out.columns:
        raise ValueError("plays_t_minus_1 must exist before calling add_delta_1m()")
    out["delta_1m"] = (out["plays_t"] - out["plays_t_minus_1"]).astype("Int64")
    return out

def add_delta_1m_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentage change vs previous month.
    If previous month plays == 0, delta_1m_pct is NaN and we expose a flag.
    """
    out = df.copy()
    if "delta_1m" not in out.columns:
        raise ValueError("delta_1m must exist before calling add_delta_1m_pct()")
    if "plays_t_minus_1" not in out.columns:
        raise ValueError("plays_t_minus_1 must exist before calling add_delta_1m_pct()")

    prev = out["plays_t_minus_1"].astype(float)
    out["prev_month_plays_was_zero"] = (prev == 0).astype(int)

    denom = prev.replace(0, np.nan)
    out["delta_1m_pct"] = (out["delta_1m"].astype(float) / denom).astype(float)
    return out

def add_ratio_to_prev3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "plays_prev3_mean" not in out.columns:
        raise ValueError("plays_prev3_mean must exist before calling add_ratio_to_prev3()")
    out["ratio_to_prev3"] = out["plays_t"] / (1.0 + out["plays_prev3_mean"].astype(float))
    return out


def _trend_slope_3m_series(values: pd.Series) -> pd.Series:
    """
    Linear slope over last 3 months including current (t-2, t-1, t).
    If fewer than 2 points, slope=0.
    """
    arr = values.to_numpy(dtype=float)
    slopes = np.zeros(len(arr), dtype=float)

    for i in range(len(arr)):
        start = max(0, i - 2)
        window = arr[start : i + 1]
        if len(window) < 2:
            slopes[i] = 0.0
            continue
        xs = np.arange(len(window), dtype=float)
        slopes[i] = float(np.polyfit(xs, window, 1)[0])

    return pd.Series(slopes, index=values.index)


def add_trend_slope_3m(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_sorted(df.copy())
    g = out.groupby("artist_name", sort=False, group_keys=False)
    out["trend_slope_3m"] = g["plays_t"].apply(_trend_slope_3m_series).astype(float)
    return out


def add_cumulative_plays_before_t(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_sorted(df.copy())
    g = out.groupby("artist_name", sort=False, group_keys=False)
    out["cumulative_plays_before_t"] = (
        g["plays_t"].apply(lambda s: s.shift(1).cumsum()).fillna(0).astype("Int64")
    )
    return out


def add_active_months_count_before_t(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_sorted(df.copy())
    g = out.groupby("artist_name", sort=False, group_keys=False)

    # cumulative count of months with plays>0 excluding current month
    out["active_months_count_before_t"] = (
        g["plays_t"]
        .apply(lambda s: (s > 0).astype(int).cumsum().shift(1))
        .fillna(0)
        .astype("Int64")
    )
    return out


def add_share_delta_1m(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_sorted(df.copy())
    g = out.groupby("artist_name", sort=False)
    out["share_delta_1m"] = out["share_t"] - g["share_t"].shift(1).fillna(0.0)
    return out


# ===========================
# Small family orchestrators (readability only)
# ===========================
def add_momentum_family(df: pd.DataFrame) -> pd.DataFrame:
    out = add_plays_lag_1m(df)
    out = add_plays_prev3_mean(out)
    out = add_delta_1m(out)
    # out = add_delta_1m_pct(out)
    # out = add_ratio_to_prev3(out)
    out = add_trend_slope_3m(out)
    return out


def add_history_family(df: pd.DataFrame) -> pd.DataFrame:
    out = add_cumulative_plays_before_t(df)
    out = add_active_months_count_before_t(out)
    return out


def add_share_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # out = add_share_delta_1m(df)
    return out


# ===========================
# Core V1 composition
# ===========================
def build_breakout_features(snapshots: pd.DataFrame, enforce_month_grid: bool = False) -> pd.DataFrame:
    """
    Build leakage-safe breakout features from artist-month snapshots.
    Features use only months <= t (with explicit shift for "before_t" features).
    """
    validate_snapshots_input(snapshots)

    df = _ensure_sorted(snapshots.copy())
    if enforce_month_grid:
        df = ensure_full_month_grid_per_artist(df)

    df = add_momentum_family(df)
    df = add_history_family(df)
    df = add_share_family(df)

    cols = [
        "month",
        "artist_name",
        "first_seen_month",
        "months_since_first_seen",
        "is_first_month",
        "plays_t",
        "unique_tracks_t",
        "plays_per_track_t",
        "days_active_t",
        "last_play_gap_days_t",
        "track_novelty_rate_t",
        "share_t",
        "total_monthly_scrobbles",
        "plays_t_minus_1",
        "plays_prev3_mean",
        "delta_1m",
        # "delta_1m_pct",
        # "prev_month_plays_was_zero",
        # "ratio_to_prev3",
        "trend_slope_3m",
        "cumulative_plays_before_t",
        "active_months_count_before_t",
        # "share_delta_1m",
    ]

    if "genre_bucket" in df.columns:
        cols.append("genre_bucket")

    out = (
        df[cols]
        .sort_values(["month", "artist_name"], kind="stable")
        .reset_index(drop=True)
    )
    return out
