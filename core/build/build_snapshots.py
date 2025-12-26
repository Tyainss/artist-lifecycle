import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ===========================
# Validation / helpers
# ===========================
def validate_scrobbles_input(df: pd.DataFrame) -> None:
    required = ["month", "played_at_utc", "artist_name", "track_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required scrobbles columns: {missing}")


def add_played_day_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["played_day_utc"] = out["played_at_utc"].dt.floor("D")
    return out


def add_month_end_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Month end = last second of the month (UTC). Used for last scrobble gap.
    """
    out = df.copy()
    out["month_end_utc"] = out["month"] + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
    return out


# ===========================
# Aggregations
# ===========================
def build_total_scrobbles_by_month(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("month", as_index=False, sort=False)
        .size()
        .rename(columns={"size": "total_scrobbles_month"})
    )


def build_artist_month_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate scrobbles to (month, artist_name) stats.
    """
    out = (
        df.groupby(["month", "artist_name"], as_index=False, sort=False)
        .agg(
            plays_t=("played_at_utc", "size"),
            days_active_t=("played_day_utc", "nunique"),
            last_played_at_utc=("played_at_utc", "max"),
        )
    )
    return out


def add_last_play_gap_days(artist_month: pd.DataFrame) -> pd.DataFrame:
    """
    Gap from last scrobble within the month to month_end_utc.
    Similar to SOTW's last_scrobble_gap_days logic.
    """
    out = artist_month.copy()
    out["month_end_utc"] = out["month"] + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)

    gap_days = (out["month_end_utc"] - out["last_played_at_utc"]).dt.total_seconds() / 86400.0
    out["last_play_gap_days_t"] = gap_days

    neg = out["last_play_gap_days_t"] < 0
    out["last_play_gap_days_t_was_negative"] = neg.fillna(False).astype(int)
    if neg.any():
        out.loc[neg, "last_play_gap_days_t"] = 0.0

    out = out.drop(columns=["month_end_utc"])
    return out


def add_share_of_attention(artist_month: pd.DataFrame, total_by_month: pd.DataFrame) -> pd.DataFrame:
    out = artist_month.merge(total_by_month, on="month", how="left", sort=False)
    out["share_t"] = out["plays_t"] / out["total_scrobbles_month"]
    return out


def add_first_seen_month(artist_month: pd.DataFrame) -> pd.DataFrame:
    first_seen = (
        artist_month.groupby("artist_name", as_index=False, sort=False)["month"]
        .min()
        .rename(columns={"month": "first_seen_month"})
    )
    out = artist_month.merge(first_seen, on="artist_name", how="left", sort=False)
    return out


def add_months_since_first_seen(artist_month: pd.DataFrame) -> pd.DataFrame:
    out = artist_month.copy()
    m = out["month"].dt.year * 12 + out["month"].dt.month
    f = out["first_seen_month"].dt.year * 12 + out["first_seen_month"].dt.month
    out["months_since_first_seen"] = (m - f).astype("int")
    return out


# ===========================
# Orchestrator
# ===========================
def build_artist_month_snapshots(scrobbles: pd.DataFrame) -> pd.DataFrame:
    """
    Build the artist-month backbone from canonical scrobbles.

    Input columns expected:
      - month (UTC tz-aware month start)
      - played_at_utc (UTC tz-aware timestamp)
      - artist_name
      - track_name

    Output columns:
      - month
      - artist_name
      - plays_t
      - days_active_t
      - last_play_gap_days_t
      - last_play_gap_days_t_was_negative
      - total_scrobbles_month
      - share_t
      - first_seen_month
      - months_since_first_seen
    """
    validate_scrobbles_input(scrobbles)

    df = scrobbles.copy()
    df = df.dropna(subset=["month", "played_at_utc", "artist_name", "track_name"])
    df["artist_name"] = df["artist_name"].astype("string").str.strip()
    df["track_name"] = df["track_name"].astype("string").str.strip()
    df = df[df["artist_name"].str.len() > 0]
    df = df[df["track_name"].str.len() > 0]

    df = add_played_day_utc(df)

    total_by_month = build_total_scrobbles_by_month(df)
    artist_month = build_artist_month_agg(df)
    artist_month = add_last_play_gap_days(artist_month)
    artist_month = add_share_of_attention(artist_month, total_by_month)
    artist_month = add_first_seen_month(artist_month)
    artist_month = add_months_since_first_seen(artist_month)

    ordered = [
        "month",
        "artist_name",
        "plays_t",
        "days_active_t",
        "last_play_gap_days_t",
        "last_play_gap_days_t_was_negative",
        "total_scrobbles_month",
        "share_t",
        "first_seen_month",
        "months_since_first_seen",
    ]

    out = (
        artist_month[ordered]
        .sort_values(["month", "artist_name"], kind="stable")
        .reset_index(drop=True)
    )
    return out
