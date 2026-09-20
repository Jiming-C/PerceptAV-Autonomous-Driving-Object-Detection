"""Make the project and its scripts importable from the tests.

Doing it here rather than in each test file keeps the sys.path juggling out of
the way of the imports themselves, and means `pytest` works from any directory
without the package having to be installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
