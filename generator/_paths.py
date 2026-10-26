"""Importing this module puts use_numpy/ and the repo root on sys.path, so the generators can import the simulations
(and utils.py) when run from any working directory: `import _paths  # noqa: F401` before the first project import."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "use_numpy"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
