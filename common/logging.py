import logging


def setup_logging(project_cfg: dict) -> None:
    """
    Initialize logging using values from project config.

    Expects:
      project_cfg["logging"]["level"]
      project_cfg["logging"]["fmt"]
      optional: project_cfg["logging"]["datefmt"]
    """
    log_cfg = project_cfg["logging"]
    level = getattr(logging, str(log_cfg["level"]).upper())
    fmt = log_cfg["fmt"]
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
