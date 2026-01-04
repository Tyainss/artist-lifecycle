import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from common.config_manager import ConfigManager


@dataclass(frozen=True)
class SplitSpec:
    train_frac: float
    val_frac: float
    test_frac: float


def _setup_logging(project_cfg: dict) -> logging.Logger:
    logging_cfg = project_cfg.get("logging", {})
    level = logging_cfg.get("level", "INFO")
    fmt = logging_cfg.get("fmt", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    datefmt = logging_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    return logging.getLogger("train_breakout")


def _resolve_paths(repo_root: Path, project_cfg: dict) -> Dict[str, Path]:
    paths_cfg = project_cfg["paths"]
    return {
        "features_dir": repo_root / paths_cfg["features"],
        "models_dir": repo_root / paths_cfg["models"],
        "metrics_dir": repo_root / paths_cfg["metrics"],
    }


def _load_modeling_table(modeling_path: Path) -> pd.DataFrame:
    df = pd.read_csv(modeling_path)
    if "month" not in df.columns:
        raise ValueError("Modeling table is missing required column: 'month'")
    if "y" not in df.columns:
        raise ValueError("Modeling table is missing required column: 'y'")
    df["month"] = pd.to_datetime(df["month"], errors="raise")
    df["y"] = df["y"].astype(int)
    return df


def _time_based_split_by_month(
    df: pd.DataFrame,
    date_col: str,
    split: SplitSpec,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[pd.Timestamp]]]:
    months = sorted(df[date_col].dropna().unique())
    n_months = len(months)
    if n_months < 5:
        raise ValueError(f"Not enough months to split reliably (n_months={n_months}).")

    n_train = int(n_months * split.train_frac)
    n_val = int(n_months * split.val_frac)
    n_test = n_months - n_train - n_val

    train_months = months[:n_train]
    val_months = months[n_train:n_train + n_val]
    test_months = months[n_train + n_val:]

    df_train = df[df[date_col].isin(train_months)].copy()
    df_val = df[df[date_col].isin(val_months)].copy()
    df_test = df[df[date_col].isin(test_months)].copy()

    logger.info(
        "Split months into train/val/test: %s/%s/%s months; rows=%s/%s/%s; positives=%s/%s/%s",
        len(train_months),
        len(val_months),
        len(test_months),
        f"{len(df_train):,}",
        f"{len(df_val):,}",
        f"{len(df_test):,}",
        int(df_train["y"].sum()),
        int(df_val["y"].sum()),
        int(df_test["y"].sum()),
    )

    month_splits = {
        "train_months": train_months,
        "val_months": val_months,
        "test_months": test_months,
    }
    return df_train, df_val, df_test, month_splits


def _select_feature_columns(
    df: pd.DataFrame,
    id_cols: List[str],
    target_col: str,
    date_col: str,
    cat_cols: List[str],
) -> List[str]:
    excluded = set(id_cols + [target_col, date_col] + cat_cols)
    return [c for c in df.columns if c not in excluded]


def _build_preprocessor_xgb(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    XGBoost does not require feature scaling.
    One-hot encode categoricals and pass numeric features through unchanged.
    """

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )


def _pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_proba))


def _roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_proba))


def _choose_threshold_max_fbeta(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.concatenate([thresholds, [1.0]])

    f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-12)
    best_idx = int(np.argmax(f_beta))
    return float(thresholds[best_idx])


def _choose_threshold_alert_budget(
    df_val: pd.DataFrame,
    y_proba: np.ndarray,
    date_col: str,
    max_median_alerts: int,
    beta: float,
    min_tp_on_val: int,
) -> float:
    """
    Notebook policy:
      - Choose threshold using validation only
      - Constraints: median alerts/month <= max_median_alerts AND TP >= min_tp_on_val
      - Objective: maximize F-beta; tie-break toward fewer alerts
    """
    y_true = df_val["y"].to_numpy(dtype=int)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.concatenate([thresholds, [1.0]])

    rows = []
    for t in thresholds:
        t = float(t)
        y_pred = (y_proba >= t).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fbeta = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec + 1e-12)

        alerts = df_val.assign(_pred=y_pred).groupby(date_col)["_pred"].sum()
        median_alerts = float(alerts.median()) if len(alerts) else 0.0

        rows.append({"t": t, "tp": tp, "median_alerts": median_alerts, "f_beta": float(fbeta)})

    grid = pd.DataFrame(rows)
    feasible = grid[(grid["median_alerts"] <= float(max_median_alerts)) & (grid["tp"] >= int(min_tp_on_val))]
    if len(feasible) == 0:
        fallback = grid[grid["tp"] >= int(min_tp_on_val)]
        best = (fallback if len(fallback) else grid).sort_values("f_beta", ascending=False).iloc[0]
        return float(best["t"])

    best = feasible.sort_values(["f_beta", "median_alerts"], ascending=[False, True]).iloc[0]
    return float(best["t"])


def _tune_xgboost_on_val(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    y_val: np.ndarray,
    base_params: dict,
    grid: List[dict],
    num_boost_round: int,
    early_stopping_rounds: int,
    logger: logging.Logger,
) -> Tuple[dict, int, float]:
    best_params: Optional[dict] = None
    best_iter: Optional[int] = None
    best_val_pr_auc = -1.0

    for params in grid:
        run_params = dict(base_params)
        run_params.update(params)

        booster = xgb.train(
            params=run_params,
            dtrain=dtrain,
            num_boost_round=int(num_boost_round),
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=int(early_stopping_rounds),
            verbose_eval=False,
        )

        best_iteration = int(booster.best_iteration)
        p_val = booster.predict(dval, iteration_range=(0, best_iteration + 1))
        pr = _pr_auc(y_val, p_val)
        logger.info("XGB grid params=%s | best_iter=%s | val PR-AUC=%.6f", params, best_iteration, pr)

        if pr > best_val_pr_auc:
            best_val_pr_auc = pr
            best_params = params
            best_iter = best_iteration

    if best_params is None or best_iter is None:
        raise RuntimeError("XGBoost tuning failed: no model was trained.")

    return best_params, int(best_iter), float(best_val_pr_auc)


def _default_xgb_grid() -> List[dict]:
    return [
        {"max_depth": 3, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "eta": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "eta": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
    ]


def run(repo_root: Path) -> Path:
    cm = ConfigManager(repo_root)
    project_cfg = cm.project()
    breakout_cfg = cm.breakout()

    logger = _setup_logging(project_cfg)
    paths = _resolve_paths(repo_root, project_cfg)

    modeling_filename = project_cfg["breakout"]["modeling_filename"]
    modeling_path = paths["features_dir"] / modeling_filename
    if not modeling_path.exists():
        raise FileNotFoundError(f"Breakout modeling file not found: {modeling_path}")

    df = _load_modeling_table(modeling_path)

    split_cfg = project_cfg.get("split", {"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2})
    split = SplitSpec(
        train_frac=float(split_cfg.get("train_frac", 0.6)),
        val_frac=float(split_cfg.get("val_frac", 0.2)),
        test_frac=float(split_cfg.get("test_frac", 0.2)),
    )

    id_cols = ["artist_name", "month"]
    date_col = "month"
    target_col = "y"
    cat_cols = ["genre_bucket"] if "genre_bucket" in df.columns else []
    num_cols = _select_feature_columns(
        df,
        id_cols=id_cols,
        target_col=target_col,
        date_col=date_col,
        cat_cols=cat_cols,
    )

    df_train, df_val, df_test, month_splits = _time_based_split_by_month(
        df, date_col=date_col, split=split, logger=logger
    )

    X_train_df = df_train[num_cols + cat_cols]
    y_train = df_train[target_col].to_numpy(dtype=int)

    X_val_df = df_val[num_cols + cat_cols]
    y_val = df_val[target_col].to_numpy(dtype=int)

    X_test_df = df_test[num_cols + cat_cols]
    y_test = df_test[target_col].to_numpy(dtype=int)

    preprocessor = _build_preprocessor_xgb(num_cols=num_cols, cat_cols=cat_cols)
    logger.info("Fitting preprocessor on train split...")
    preprocessor.fit(X_train_df)

    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    modeling_cfg = breakout_cfg.get("modeling", {})
    xgb_cfg = modeling_cfg.get("xgboost", {})

    num_boost_round = int(xgb_cfg.get("n_estimators", 2000))
    early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 50))

    mode = str(xgb_cfg.get("mode", "final")).lower()
    final_params_cfg = xgb_cfg.get("final_params", {})

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "seed": int(xgb_cfg.get("random_state", 42)),
        "nthread": int(xgb_cfg.get("n_jobs", -1)),
        "scale_pos_weight": float(xgb_cfg.get("scale_pos_weight", scale_pos_weight)),
        "reg_lambda": float(xgb_cfg.get("reg_lambda", 1.0)),
        "min_child_weight": float(xgb_cfg.get("min_child_weight", 1.0)),
    }
    params.update(final_params_cfg)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    logger.info("Training XGBoost (final) | params=%s", final_params_cfg)
    booster_es = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )

    best_iteration = int(booster_es.best_iteration)
    best_n_estimators = int(best_iteration + 1)

    p_val = booster_es.predict(dval, iteration_range=(0, best_n_estimators))
    
    val_pr_auc = _pr_auc(y_val, p_val)
    val_roc_auc = _roc_auc(y_val, p_val)
    logger.info(
        "Best XGB val PR-AUC=%.5f | ROC-AUC=%.5f | best_iter=%s",
        val_pr_auc,
        val_roc_auc,
        best_iteration,
    )

    thresh_cfg = modeling_cfg.get("thresholding", {})
    policy = str(thresh_cfg.get("policy", "max_fbeta"))
    beta = float(thresh_cfg.get("beta", 0.5))

    if policy == "alerts_budget":
        max_median_alerts = int(thresh_cfg.get("max_median_alerts", 3))
        min_tp_on_val = int(thresh_cfg.get("min_tp_on_val", 1))
        threshold = _choose_threshold_alert_budget(
            df_val=df_val,
            y_proba=p_val,
            date_col=date_col,
            max_median_alerts=max_median_alerts,
            beta=beta,
            min_tp_on_val=min_tp_on_val,
        )
    else:
        threshold = _choose_threshold_max_fbeta(y_true=y_val, y_proba=p_val, beta=beta)

    logger.info(
        "Chosen threshold on validation | policy=%s | beta=%.2f | T=%.6f",
        policy,
        beta,
        threshold,
    )

    trainval_df = pd.concat([df_train, df_val], axis=0).copy()
    X_trainval_df = trainval_df[num_cols + cat_cols]
    y_trainval = trainval_df[target_col].to_numpy(dtype=int)

    preprocessor_tv = _build_preprocessor_xgb(num_cols=num_cols, cat_cols=cat_cols)
    logger.info("Fitting preprocessor on train+val split.")
    preprocessor_tv.fit(X_trainval_df)

    X_trainval = preprocessor_tv.transform(X_trainval_df)
    X_test = preprocessor_tv.transform(X_test_df)

    n_pos_tv = int(y_trainval.sum())
    n_neg_tv = int(len(y_trainval) - n_pos_tv)
    scale_pos_weight_tv = float(n_neg_tv / max(n_pos_tv, 1))

    params_tv = dict(params)
    params_tv["scale_pos_weight"] = float(scale_pos_weight_tv)

    dtrain_tv = xgb.DMatrix(X_trainval, label=y_trainval)
    dtest = xgb.DMatrix(X_test, label=y_test)

    final_model = xgb.train(
        params=params_tv,
        dtrain=dtrain_tv,
        num_boost_round=int(best_n_estimators),
        evals=[(dtrain_tv, "trainval")],
        verbose_eval=False,
    )

    p_test = final_model.predict(dtest)
    test_pr_auc = _pr_auc(y_test, p_test)
    test_roc_auc = _roc_auc(y_test, p_test)

    logger.info("Test PR-AUC=%.5f | ROC-AUC=%.5f", test_pr_auc, test_roc_auc)

    model_filename = project_cfg["breakout"].get("model_filename", "model.bin")

    model_dir = paths["models_dir"] / "breakout"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_filename

    artifacts = {
        "mode": "breakout",
        "model_family": "xgboost",
        "model": final_model,
        "preprocessor": preprocessor_tv,
        "feature_columns": num_cols + cat_cols,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "threshold": float(threshold),
        "meta": {
            "final_params": dict(final_params_cfg),
            "best_iter": int(best_iteration),
            "best_n_estimators": int(best_n_estimators),
            "scale_pos_weight_train": float(params["scale_pos_weight"]),
            "scale_pos_weight_trainval": float(scale_pos_weight_tv),
            "val_metrics": {"pr_auc": float(val_pr_auc), "roc_auc": float(val_roc_auc)},
            "test_metrics": {"pr_auc": float(test_pr_auc), "roc_auc": float(test_roc_auc)},
            "month_splits": {k: [str(x) for x in v] for k, v in month_splits.items()},
            "thresholding": {"policy": policy, "beta": float(beta)},
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    logger.info("Saved model artifacts: %s", model_path)

    metrics_filename = project_cfg["breakout"].get("metrics_filename", "metrics.json")
    metrics_dir = paths["metrics_dir"] / "breakout"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / metrics_filename

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "breakout",
                "model_family": "xgboost",
                "model_path": str(model_path),
                "threshold": float(threshold),
                "val_pr_auc": float(val_pr_auc),
                "val_roc_auc": float(val_roc_auc),
                "test_pr_auc": float(test_pr_auc),
                "test_roc_auc": float(test_roc_auc),
                 "final_params": dict(final_params_cfg),
                "best_n_estimators": int(best_n_estimators),
                "scale_pos_weight_train": float(params["scale_pos_weight"]),
                "scale_pos_weight_trainval": float(scale_pos_weight_tv),
            },
            f,
            indent=2,
        )

    logger.info("Saved metrics: %s", metrics_path)
    return model_path
