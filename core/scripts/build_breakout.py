import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from common.config_manager import ConfigManager
from core.build.build_features import build_breakout_features
from core.build.build_snapshots import build_artist_month_snapshots


def setup_logging(project_cfg: dict) -> None:
    logging_cfg = project_cfg.get("logging", {})
    level = logging_cfg.get("level", "INFO").upper()
    fmt = logging_cfg.get("fmt", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    datefmt = logging_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def validate_required_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def parse_scrobbles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "played_at_utc" in out.columns:
        out["played_at_utc"] = pd.to_datetime(out["played_at_utc"], utc=True, errors="raise")
    elif "date" in out.columns:
        out["played_at_utc"] = pd.to_datetime(out["date"], utc=True, errors="raise")
    else:
        raise ValueError("Expected a datetime column: 'played_at_utc' or 'date'")

    # month start timestamps (UTC tz-aware)
    month_naive = out["played_at_utc"].dt.to_period("M").dt.to_timestamp()
    out["month"] = month_naive.dt.tz_localize("UTC")

    return out


def main(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    cfg = ConfigManager(repo_root)

    project_cfg = cfg.project()
    sources_cfg = cfg.sources()
    breakout_cfg = cfg.breakout()

    setup_logging(project_cfg)
    logger = logging.getLogger(__name__)

    paths = project_cfg["paths"]
    curated_dir = repo_root / paths["curated"]
    processed_dir = repo_root / paths["processed"]
    features_dir = repo_root / paths["features"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    scrobbles_rel = sources_cfg["scrobbles"]["path"]
    scrobbles_path = repo_root / scrobbles_rel
    if not scrobbles_path.is_file():
        raise FileNotFoundError(f"Scrobbles file not found: {scrobbles_path}")

    required_cols = sources_cfg["scrobbles"].get("required_columns", [])
    logger.info(f"Reading scrobbles: {scrobbles_path}")
    scrobbles = pd.read_csv(scrobbles_path, low_memory=False)

    if required_cols:
        validate_required_columns(scrobbles, required_cols, label="scrobbles")

    scrobbles = parse_scrobbles(scrobbles)

    # Keep only columns needed by snapshots builder (+ optional genre_bucket if present)
    base_cols = ["month", "played_at_utc", "artist_name", "track_name"]
    optional_cols = []
    if "genre_bucket" in scrobbles.columns:
        optional_cols.append("genre_bucket")

    scrobbles = scrobbles[base_cols + optional_cols].copy()

    logger.info("Building artist-month snapshots")
    snapshots = build_artist_month_snapshots(scrobbles)

    snapshots_filename = project_cfg["breakout"]["snapshots_filename"]
    snapshots_out = processed_dir / snapshots_filename
    snapshots.to_csv(snapshots_out, index=False)
    logger.info(f"Wrote snapshots: {snapshots_out} | rows={len(snapshots)}")

    features_cfg = breakout_cfg.get("features", {})

    enforce_month_grid = features_cfg.get("enforce_month_grid", False)

    feature_set = features_cfg.get("feature_set", "full")
    feature_sets = features_cfg.get("feature_sets", {})
    if not isinstance(feature_sets, dict):
        raise ValueError("breakout.features.feature_sets must be a mapping (dict).")

    feature_set_cfg = feature_sets.get(feature_set)
    if feature_set_cfg is None:
        available = sorted(feature_sets.keys())
        raise ValueError(
            f"Unknown breakout.features.feature_set='{feature_set}'. "
            f"Available: {available}"
        )

    drop_cols = feature_set_cfg.get("drop_cols", [])
    if not isinstance(drop_cols, list) or not all(isinstance(x, str) for x in drop_cols):
        raise ValueError(
            f"breakout.features.feature_sets.{feature_set}.drop_cols must be a list of strings."
        )

    logger.info(
        f"Building breakout features | enforce_month_grid={enforce_month_grid} | "
        f"feature_set={feature_set} | drop_cols={drop_cols}"
    )
    features = build_breakout_features(
        snapshots,
        enforce_month_grid=enforce_month_grid,
        drop_cols=drop_cols,
    )

    # Choose output filename by feature_set (so reduced sets don't overwrite canonical full output)
    features_filename_by_set = features_cfg.get("features_filename_by_set", {})
    if features_filename_by_set:
        if not isinstance(features_filename_by_set, dict):
            raise ValueError("breakout.features.features_filename_by_set must be a mapping (dict).")
        if feature_set not in features_filename_by_set:
            available = sorted(features_filename_by_set.keys())
            raise ValueError(
                f"Missing features_filename_by_set entry for feature_set='{feature_set}'. "
                f"Available: {available}"
            )
        features_filename = features_filename_by_set[feature_set]
        if not isinstance(features_filename, str) or not features_filename.strip():
            raise ValueError(
                f"Invalid features_filename_by_set['{feature_set}']: must be a non-empty string."
            )
    else:
        # Fallback to project.yaml default
        features_filename = project_cfg["breakout"]["features_filename"]
        
    features_out = features_dir / features_filename
    features.to_csv(features_out, index=False)
    logger.info(f"Wrote features: {features_out} | rows={len(features)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=".")
    args = parser.parse_args()

    main(Path(args.repo_root))
