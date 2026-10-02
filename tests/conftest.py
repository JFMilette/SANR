"""pytest setup: fit/dream/ on the path, so that tests/test_dream.py (copied
unchanged with fit/dream/dream.py) can `from dream import ...`."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'fit' / 'dream'))
