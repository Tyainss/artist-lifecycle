import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ===========================
# Raw scrobbles -> canonical scrobbles
# ===========================
def validate_raw_scrobbles_input(df: pd.DataFrame) -> None:
    required = ["artist_name", "track_name", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw scrobbles columns: {missing}")


def clean_artist_and_track(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["artist_name"] = out["artist_name"].astype("string").str.strip()
    out["track_name"] = out["track_name"].astype("string").str.strip()

    out = out.dropna(subset=["artist_name", "track_name"])
    out = out[out["artist_name"].str.len() > 0]
    out = out[out["track_name"].str.len() > 0]
    return out


def parse_played_at_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses df['date'] into a UTC tz-aware timestamp column: played_at_utc.

    Your SOTW cleaned export uses date formats like:
      - YYYY-MM-DD HH:MM:SS
    and should parse cleanly with utc=True.
    """
    out = df.copy()

    played = pd.to_datetime(out["date"], utc=True, errors="raise")
    if played.isna().any():
        bad = out.loc[played.isna(), "date"].head(5).tolist()
        raise ValueError(f"Found invalid 'date' values. Examples: {bad}")

    out["played_at_utc"] = played
    return out


def add_month_start_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a tz-aware month start column: month (YYYY-MM-01 00:00:00+00:00)
    """
    out = df.copy()
    month_str = out["played_at_utc"].dt.strftime("%Y-%m-01")
    out["month"] = pd.to_datetime(month_str, utc=True, errors="raise")
    return out


def prepare_scrobbles_for_snapshots(raw_scrobbles: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw scrobble-level data and returns canonical columns for snapshot building:
      - month
      - played_at_utc
      - artist_name
      - track_name
    """
    validate_raw_scrobbles_input(raw_scrobbles)

    df = raw_scrobbles.copy()
    df = clean_artist_and_track(df)
    df = parse_played_at_utc(df)
    df = add_month_start_utc(df)

    return df[["month", "played_at_utc", "artist_name", "track_name"]].copy()


# ===========================
# Canonical scrobbles -> artist-month snapshot table
# ===========================
def validate_scrobbles_input(df: pd.DataFrame) -> None:
    required = ["month", "played_at_utc", "artist_name", "track_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required canonical scrobbles columns: {missing}")


def add_played_day_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["played_day_utc"] = out["played_at_utc"].dt.floor("D")
    return out


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
            unique_tracks_t=("track_name", "nunique"),
            days_active_t=("played_day_utc", "nunique"),
            last_played_at_utc=("played_at_utc", "max"),
        )
    )
    return out

def add_plays_per_track(artist_month: pd.DataFrame) -> pd.DataFrame:
    out = artist_month.copy()
    denom = out["unique_tracks_t"].replace(0, pd.NA)
    out["plays_per_track_t"] = (out["plays_t"] / denom).fillna(0.0)
    return out

def add_track_novelty_rate(scrobbles: pd.DataFrame, artist_month: pd.DataFrame) -> pd.DataFrame:
    """
    track_novelty_rate_t = (# unique tracks in month t that were never seen before t) / unique_tracks_t
    Identity = (artist_name, track_name).
    """
    df = scrobbles[["month", "artist_name", "track_name"]].copy()

    track_first_seen = (
        df.groupby(["artist_name", "track_name"], as_index=False, sort=False)["month"]
        .min()
        .rename(columns={"month": "track_first_seen_month"})
    )
    df = df.merge(track_first_seen, on=["artist_name", "track_name"], how="left", sort=False)

    new_tracks = (
        df[df["month"] == df["track_first_seen_month"]]
        .groupby(["month", "artist_name"], as_index=False, sort=False)["track_name"]
        .nunique()
        .rename(columns={"track_name": "new_tracks_t"})
    )

    out = artist_month.merge(new_tracks, on=["month", "artist_name"], how="left", sort=False)
    out["new_tracks_t"] = out["new_tracks_t"].fillna(0).astype(int)

    denom = out["unique_tracks_t"].replace(0, pd.NA)
    out["track_novelty_rate_t"] = (out["new_tracks_t"] / denom).fillna(0.0)
    return out

def add_last_play_gap_days(artist_month: pd.DataFrame) -> pd.DataFrame:
    """
    Gap from last scrobble within the month to month_end_utc.
    Clipped to 0 if negative (defensive).
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
    out["total_monthly_scrobbles"] = out["total_scrobbles_month"]
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

def add_is_first_month(artist_month: pd.DataFrame) -> pd.DataFrame:
    out = artist_month.copy()
    out["is_first_month"] = (out["months_since_first_seen"] == 0).astype(int)
    return out

def build_artist_month_snapshots(scrobbles: pd.DataFrame) -> pd.DataFrame:
    """
    Build the artist-month backbone from canonical scrobbles.
    """
    validate_scrobbles_input(scrobbles)

    df = scrobbles.copy()
    df = df.dropna(subset=["month", "played_at_utc", "artist_name", "track_name"])
    df = add_played_day_utc(df)

    total_by_month = build_total_scrobbles_by_month(df)
    artist_month = build_artist_month_agg(df)
    artist_month = add_plays_per_track(artist_month)
    artist_month = add_last_play_gap_days(artist_month)
    artist_month = add_share_of_attention(artist_month, total_by_month)
    artist_month = add_first_seen_month(artist_month)
    artist_month = add_months_since_first_seen(artist_month)
    artist_month = add_is_first_month(artist_month)
    artist_month = add_track_novelty_rate(df, artist_month)

    ordered = [
        "month",
        "artist_name",
        "plays_t",
        "unique_tracks_t",
        "plays_per_track_t",
        "days_active_t",
        "last_play_gap_days_t",
        "last_play_gap_days_t_was_negative",
        "total_scrobbles_month",
        "total_monthly_scrobbles",
        "share_t",
        "first_seen_month",
        "months_since_first_seen",
        "is_first_month",
        "track_novelty_rate_t",
    ]

    out = (
        artist_month[ordered]
        .sort_values(["month", "artist_name"], kind="stable")
        .reset_index(drop=True)
    )
    return out
