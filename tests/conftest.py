from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECT_DIR = PROJECT_ROOT / "src" / "data" / "collect"
VALIDATE_DIR = PROJECT_ROOT / "src" / "data" / "validate"

if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))

if str(VALIDATE_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATE_DIR))