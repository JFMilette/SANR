"""selftest.py, the check `SANR --self-test <file>` runs on a built SANR,
passes from the source too.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import model.stack as S                                         # noqa: E402
import selftest                                                 # noqa: E402


@pytest.mark.skipif(S._transfer is None, reason='no compiled kernel')
def test_self_test_passes(tmp_path):
    report = tmp_path / 'report.txt'
    assert selftest.run(str(report)) == 0
    lines = report.read_text().splitlines()
    assert lines[-1] == 'OK'
    assert any(l.startswith('worker pool') for l in lines)


def test_self_test_reports_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(S, '_transfer', None)
    report = tmp_path / 'report.txt'
    assert selftest.run(str(report)) == 1
    text = report.read_text()
    assert 'compiled kernel' in text and text.rstrip().endswith('FAILED')
