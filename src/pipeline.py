from pathlib import Path
import sys
import pandas as pd

def _add_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def _configure_paths() -> None:
    root = Path(__file__).resolve().parent
    project_root = root.parent
    _add_sys_path(project_root / "src" / "data" / "collect")
    _add_sys_path(project_root / "src" / "data" / "validate")
    _add_sys_path(project_root / "src" / "features" / "engineer")
    _add_sys_path(project_root / "src" / "features" / "preprocess")
    _add_sys_path(project_root / "src" / "features" / "visualize")
    _add_sys_path(project_root / "src" / "features" / "select")
    _add_sys_path(project_root / "src" / "models")

_configure_paths()

import data.collect.collect as _collect
import data.validate.validate as _validate
import features.engineer.engineer as _engineer
import features.preprocess.preprocess as _preprocess
import features.visualize.visualize as _visualize
import features.select._select as _select
import models.train as _train

MATCHES_DIR = Path("data/matches")
TRAIN_PATH  = MATCHES_DIR / "preprocessed_train.csv"
TEST_PATH   = MATCHES_DIR / "preprocessed_test.csv"

def main():
    _collect.main()
    df = pd.read_csv("data/matches/collected.csv")
    _validate.main(df)
    df = _engineer.main(df)
    train, test = _preprocess.main(df)

    # Save splits where train.py expects them
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    _visualize.main(train)

    # _select.main() must save features_linear.csv and features_trees.csv
    # to data/matches/ — check _select.py if the error persists
    _select.main(train)

    _train.main()

if __name__ == "__main__":
    main()