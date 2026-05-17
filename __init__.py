import sys
from pathlib import Path

# Ensure the repo root is on sys.path so that 'utils' is importable as a
# top-level package regardless of whether this package is invoked from
# ~/Work/ (python -m spacenorm_yolo.*) or from the repo root.
_repo_root = str(Path(__file__).parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
