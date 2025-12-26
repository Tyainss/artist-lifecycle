import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def write_csv(path: Path, df: pd.DataFrame, append: bool = False) -> None:
    """
    Always write LF line-endings and UTF-8 so git diffs are stable across OS.
    Avoid duplicate headers when appending.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        df.to_csv(
            path,
            mode="a",
            index=False,
            header=False,
            encoding="utf-8",
            lineterminator="\n",
        )
    else:
        df.to_csv(
            path,
            index=False,
            header=True,
            encoding="utf-8",
            lineterminator="\n",
        )


def read_csv(
    path: Path,
    usecols: list[str] | None = None,
    dtype: dict | None = None,
    parse_dates: list[str] | None = None,
    safe: bool = False,
) -> pd.DataFrame:
    """
    Wrapper around pandas.read_csv with consistent defaults.

    - usecols: same semantics as pandas.
    - safe=True: if requested columns are missing, read file without usecols and
      return only the intersection (no exception).
    """
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, parse_dates=parse_dates)
    except Exception:
        if not safe:
            raise
        df = pd.read_csv(path, dtype=dtype, parse_dates=parse_dates)
        if usecols:
            keep = [c for c in usecols if c in df.columns]
            return df[keep]
        return df


def write_json(path: Path, data: dict, indent: int = 2) -> None:
    """
    Write a JSON file with UTF-8 encoding and pretty-printing.
    Creates parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=indent,
        )


# ---------------------------
# Project-specific helpers
# ---------------------------

def validate_required_columns(df: pd.DataFrame, required: list[str], source_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        available = list(df.columns)
        raise ValueError(
            f"Missing required columns for {source_name}: {missing}. Available columns: {available}"
        )


def parse_utc_datetime(series: pd.Series, source_name: str, column_name: str) -> pd.Series:
    """
    Parse a datetime column as UTC. Raises ValueError on invalid values.
    """
    dt = pd.to_datetime(series, utc=True, errors="raise")
    if dt.isna().any():
        bad = series[dt.isna()].head(5).tolist()
        raise ValueError(
            f"Found invalid datetimes in {source_name}.{column_name}. Examples: {bad}"
        )
    return dt


def month_start_from_utc(played_at_utc: pd.Series) -> pd.Series:
    """
    Produce tz-aware month start (YYYY-MM-01 00:00:00+00:00).
    """
    month_str = played_at_utc.dt.strftime("%Y-%m-01")
    return pd.to_datetime(month_str, utc=True, errors="raise")


def load_scrobbles(project_root: Path, sources_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load scrobbles from configs/sources.yaml and return canonical columns:
      - month (UTC, month start)
      - played_at_utc (UTC)
      - artist_name
      - track_name
    """
    sc = sources_cfg.get("scrobbles")
    if not isinstance(sc, dict):
        raise ValueError("configs/sources.yaml must define a 'scrobbles' mapping.")

    rel_path = sc.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError("sources.scrobbles.path must be a non-empty string.")

    required = sc.get("required_columns")
    if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
        raise ValueError("sources.scrobbles.required_columns must be a list of strings.")

    path = project_root / rel_path
    if not path.is_file():
        raise FileNotFoundError(f"Scrobbles file not found: {path}")

    df = read_csv(path)
    validate_required_columns(df, required, source_name="scrobbles")

    df = df.copy()

    df["artist_name"] = df["artist_name"].astype("string").str.strip()
    df["track_name"] = df["track_name"].astype("string").str.strip()

    df["played_at_utc"] = parse_utc_datetime(df["date"], source_name="scrobbles", column_name="date")
    df["month"] = month_start_from_utc(df["played_at_utc"])

    # Drop bad/empty essentials
    df = df.dropna(subset=["artist_name", "track_name", "played_at_utc", "month"])
    df = df[df["artist_name"].str.len() > 0]
    df = df[df["track_name"].str.len() > 0]

    return df[["month", "played_at_utc", "artist_name", "track_name"]]
