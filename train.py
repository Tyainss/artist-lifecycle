
import argparse
from pathlib import Path

from core.modeling.train.train_breakout import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run(repo_root)


if __name__ == "__main__":
    main()
