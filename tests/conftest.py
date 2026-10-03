"""pytest setup: fit/dream/ on the path, so that tests/test_dream.py (copied
unchanged with fit/dream/dream.py) can `from dream import ...`."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'fit' / 'dream'))


def pytest_configure(config):
    # tests over ~20 s; pytest.ini leaves them out unless -m slow is given
    config.addinivalue_line('markers', 'slow: long sampler and benchmark '
                            'tests (python -m pytest -m slow tests)')
