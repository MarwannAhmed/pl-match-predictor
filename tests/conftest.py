from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECT_DIR = PROJECT_ROOT / "src" / "data" / "collect"
VALIDATE_DIR = PROJECT_ROOT / "src" / "data" / "validate"
FEATURES_DIR = PROJECT_ROOT / "src" / "features" / "engineer"
PREPROCESS_DIR = PROJECT_ROOT / "src" / "features" / "preprocess"

if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))

if str(VALIDATE_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATE_DIR))

if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))