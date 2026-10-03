"""Experimental tab import: the Q shift entered in the import dialog, kept
in the session and applied again when it is reopened.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
from tabs import ExperimentalTab, SimulationTab                 # noqa: E402
from tabs.experimental import ImportDialog, parse_data          # noqa: E402

Q = np.linspace(0.01, 0.1, 20)


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def data_file(tmp_path):
    p = tmp_path / 'r.dat'
    p.write_text('# Q R dR\n' + ''.join('%.6g %.6g %.6g\n' % (q, 1 / q, 0.1)
                                        for q in Q))
    return p


def test_parse_shift():
    lines = ['%g %g' % (q, 1 / q) for q in Q]
    d, _ = parse_data(lines, 0, q_shift=-0.002)
    assert np.allclose(d['Q'], Q - 0.002) and np.allclose(d['R'], 1 / Q)


def test_dialog_shift(app, data_file):
    dlg = ImportDialog(str(data_file), q_shift=0.001)
    assert np.allclose(dlg.data['Q'], Q + 0.001)      # starts at the last one
    dlg.q_shift.setValue(-0.0015)
    assert np.allclose(dlg.data['Q'], Q - 0.0015)
    assert 'shifted by -0.0015' in dlg.preview.text()
    dlg.q_shift.setValue(0.0)
    assert np.allclose(dlg.data['Q'], Q)


def test_session_keeps_shift(app, data_file, monkeypatch):
    exp = ExperimentalTab(simulation=SimulationTab())
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        lambda *a, **k: (str(data_file), ''))

    def accept(dlg):
        dlg.q_shift.setValue(0.0007)
        return QtWidgets.QDialog.DialogCode.Accepted
    monkeypatch.setattr(ImportDialog, 'exec', accept)
    exp.import_data()
    d = next(iter(exp.sets[0]['channels'].values()))
    assert np.allclose(d['Q'], Q + 0.0007) and exp.last_q_shift == 0.0007
    state = exp.session_state(str(data_file.parent))
    again = ExperimentalTab(simulation=SimulationTab())
    assert again.restore_state(state, str(data_file.parent)) == []
    d = next(iter(again.sets[0]['channels'].values()))
    assert np.allclose(d['Q'], Q + 0.0007) and d['q_shift'] == 0.0007
