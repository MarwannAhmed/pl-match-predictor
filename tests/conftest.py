from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECT_DIR = PROJECT_ROOT / "src" / "data" / "collect"
VALIDATE_DIR = PROJECT_ROOT / "src" / "data" / "validate"
FEATURES_DIR = PROJECT_ROOT / "src" / "features" / "engineer"
PREPROCESS_DIR = PROJECT_ROOT / "src" / "features" / "preprocess"
VISUALIZE_DIR = PROJECT_ROOT / "src" / "features" / "visualize"
SELECT_DIR = PROJECT_ROOT / "src" / "features" / "select"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
PIPELINE_DIR = PROJECT_ROOT / "src"

if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))

if str(VALIDATE_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATE_DIR))

if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

if str(VISUALIZE_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZE_DIR))

if str(SELECT_DIR) not in sys.path:
    sys.path.insert(0, str(SELECT_DIR))

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))