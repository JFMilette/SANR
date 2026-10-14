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


def test_stop_and_continue(app):
    """Stop the DE fit, Continue in the fit window: the history goes on
    from the generation it stopped at."""
    import time
    exp = tab(QA, QA, licorne=False)
    exp.fit_maxiter.setValue(10000)
    exp.fit_tol.setValue(exp.fit_tol.minimum())
    exp.fit_polish.setChecked(False)
    exp.fit_restarts.setValue(1)

    def run_until(n):
        t0 = time.monotonic()
        while exp._fit is not None and time.monotonic() - t0 < 60:
            app.processEvents()
            if len(exp.fit_window.entries) >= n and \
                    exp.fit_window.btn_stop.isEnabled():
                exp.stop_fit()
        assert exp._fit is None
    exp.start_fit()
    win = exp.fit_window
    assert not win.btn_continue.isEnabled()
    run_until(3)
    gen = int(win.entries[-1][0]) if win.entries[-1][0].isdigit() else \
        int(win.entries[-2][0])
    assert win.btn_continue.isEnabled() and exp._de_state['generation'] == gen
    kept = len(win.entries)
    win.btn_continue.click()
    assert exp._fit is not None and not win.btn_continue.isEnabled()
    run_until(kept + 3)
    labels = [e[0] for e in win.entries]
    assert labels[:kept] == [e[0] for e in win.entries[:kept]]
    later = [int(l) for l in labels[kept:] if l.isdigit()]
    assert later and later[0] == gen + 1
    assert win.btn_continue.isEnabled()


def test_local_lbfgs_fit(app):
    """The local page: L-BFGS-B alone in the fit window, counted in
    iterations from the start (0), its report on its own page; nothing to
    continue; its Max iter kept in the session."""
    import time
    exp = tab(QA, QA, licorne=False)
    exp.fit_workers.setValue(1)
    exp.lb_maxiter.setValue(50)
    exp.fit_report.setText('untouched')
    exp.start_lbfgs()
    win = exp.fit_window
    assert win.unit == 'iteration' and not win.polish
    assert win.history.headerItem().text(0) == 'Iteration'
    t0 = time.monotonic()
    while exp._fit is not None and time.monotonic() - t0 < 60:
        app.processEvents()
    assert exp._fit is None and not win.running
    labels = [e[0] for e in win.entries]
    assert labels[0] == '0' and all(l.isdigit() for l in labels)
    assert 'Done</b> after' in exp.lb_report.text() and \
        'iteration(s)' in exp.lb_report.text()
    assert exp.fit_report.text() == 'untouched'
    assert not win.btn_continue.isEnabled()
    assert exp.btn_lbfgs.isEnabled() and exp.btn_lbfgs_window.isEnabled()
    assert exp.session_state('.')['fit']['lbfgs_maxiter'] == 50


def test_multistart_in_fit_window(app):
    """The multistart page's Run follows the starts in the fit window and
    lists the minima there, as in its table; one is sent to the Simulation
    tab from the window."""
    import time
    exp = tab(QA, QA, licorne=False)
    exp.fit_workers.setValue(1)
    exp.ms_starts.setValue(4)
    exp.start_multistart()
    win = exp.fit_window
    assert win.isVisible() and win.running and win.unit == 'minimum'
    assert exp.btn_ms_window.isEnabled() and win.btn_stop.isEnabled()
    t0 = time.monotonic()
    while exp._ms is not None and time.monotonic() - t0 < 60:
        app.processEvents()
    assert exp._ms is None and not win.running
    m = exp._ms_minima
    assert len(win.entries) == len(m) == exp.ms_table.rowCount()
    assert [e[1] for e in win.entries] == [r['chi2'] for r in m]
    assert win.entries[0][0].startswith('1 — ')
    assert win.history.currentItem() is win.history.topLevelItem(0)
    assert 'Done</b>' in win.status.text()
    assert not win.btn_continue.isEnabled() and win.btn_send.isEnabled()
    win.history.setCurrentItem(win.history.topLevelItem(len(m) - 1))
    win.btn_send.click()
    assert exp.simulation.stack.layers[1].thickness == \
        pytest.approx(m[-1]['x'][0])
    assert 'Sent <b>minimum %d' % len(m) in exp.ms_report.text()
