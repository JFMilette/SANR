"""Fit box: the Q grid of the Licorne resolution scheme (one grid for every
channel, as Licorne's q.dat), chosen in a small dialog when the channels
are not on the same Q points.

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
from fit.problem import FitProblem                              # noqa: E402
from tabs import ExperimentalTab, SimulationTab                 # noqa: E402

QA = np.linspace(0.01, 0.2, 120)
QB = np.linspace(0.012, 0.19, 60)


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def channel(Q):
    one = np.ones_like(Q)
    return dict(Q=Q, R=0.1 * one, dR=0.01 * one, dQ=0 * one,
                theta=np.nan * one, path='x.dat')


def tab(qa, qb, licorne=True):
    sim = SimulationTab()
    sim.res_on.setChecked(True)
    sim.res_conv.setCurrentIndex(3 if licorne else 0)
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    exp.sets = [dict(name='d', visible=True,
                     channels={'+': channel(qa), '-': channel(qb)})]
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    return exp


def answer(monkeypatch, pick):
    """Make the dialog return the item starting with `pick` (None:
    cancel); record its calls."""
    calls = []

    def get_item(parent, title, label, items, current, editable):
        calls.append((items, current))
        if pick is None:
            return '', False
        return [i for i in items if i.startswith(pick)][0], True
    monkeypatch.setattr(QtWidgets.QInputDialog, 'getItem', get_item)
    return calls


def test_choose_the_grid(app, monkeypatch):
    exp = tab(QA, QB)
    calls = answer(monkeypatch, 'R-')
    *_, grid = exp._fit_inputs(exp.fit_report)
    assert len(calls) == 1 and np.array_equal(grid, QB)
    # remembered: the next dialog starts on R-
    exp._fit_inputs(exp.fit_report)
    assert calls[1][0][calls[1][1]].startswith('R-')
    # the fit problem computes on that grid
    s, vec, data, qmin, qmax, grid = exp._fit_inputs(exp.fit_report)
    # R+ reaches past R-'s points: those take the grid's end value
    with pytest.warns(UserWarning, match=r'outside the chosen Q grid.*R\+'):
        fp = FitProblem(exp.simulation.stack, data, 'chi2', qmin, qmax, grid)
    assert len(fp.groups) == 1 and np.array_equal(fp.groups[0][0], QB)


def test_cancel_starts_nothing(app, monkeypatch):
    exp = tab(QA, QB)
    answer(monkeypatch, None)
    assert exp._fit_inputs(exp.fit_report) is None
    assert 'No Q grid' in exp.fit_report.text()


def test_no_dialog_when_shared_or_quadrature(app, monkeypatch):
    calls = answer(monkeypatch, 'R+')
    exp = tab(QA, QA.copy())
    *_, grid = exp._fit_inputs(exp.fit_report)
    assert np.array_equal(grid, QA)
    exp = tab(QA, QB, licorne=False)
    *_, grid = exp._fit_inputs(exp.fit_report)
    assert grid is None and not calls
