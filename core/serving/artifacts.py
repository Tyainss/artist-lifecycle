
import pickle
from pathlib import Path
from typing import Any, Dict

from common.config_manager import ConfigManager


def load_breakout_artifacts(repo_root: Path) -> Dict[str, Any]:
    cm = ConfigManager(repo_root)
    project_cfg = cm.project()

    paths_cfg = project_cfg["paths"]
    models_dir = repo_root / paths_cfg["models"]

    model_filename = project_cfg["breakout"].get("model_filename", "model.bin")
    model_path = models_dir / "breakout" / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifacts not found: {model_path}")

    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    return artifacts
