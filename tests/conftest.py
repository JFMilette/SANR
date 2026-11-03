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


import gc                                                    # noqa: E402

import pytest                                                # noqa: E402


@pytest.fixture(autouse=True)
def _qt_leftovers():
    """After each test, run the Qt events it left queued (zero-delay
    timers, deferred deletions) and collect its widgets, so that a later
    test's processEvents() never delivers them to deleted objects (an
    intermittent segmentation fault when the GUI tests run together)."""
    yield
    app = None
    if 'PyQt6.QtWidgets' in sys.modules:
        app = sys.modules['PyQt6.QtWidgets'].QApplication.instance()
    if app is not None:
        app.processEvents()
        gc.collect()
        app.processEvents()
