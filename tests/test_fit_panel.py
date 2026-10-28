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
    exp.btn_fit.click()                 # the button, as in the GUI
    win = exp.fit_window
    assert win is not None and win.method == 'de'
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


def test_least_squares_fit(app):
    """The least-squares method: run_lsq in the fit window, counted in
    iterations from the start (0), as fit as L-BFGS-B, from worker
    processes too; its report its own; its Max iter and the method chosen
    kept in the session."""
    import time
    exp = tab(QA, QA, licorne=False)
    ends = {}
    for workers in (1, 2):
        exp.fit_workers.setValue(workers)
        exp.lsq_maxiter.setValue(50)
        exp.fit_report.setText('untouched')
        exp.btn_lsq.click()
        win = exp.fit_window
        assert win.method == 'lsq' and win.unit == 'iteration'
        t0 = time.monotonic()
        while exp._fit is not None and time.monotonic() - t0 < 60:
            app.processEvents()
        assert exp._fit is None and not win.running
        labels = [e[0] for e in win.entries]
        assert labels[0] == '0' and all(l.isdigit() for l in labels[:-1])
        assert 'Done</b> after' in exp.lsq_report.text()
        assert exp.fit_report.text() == 'untouched'
        assert exp.btn_lsq.isEnabled() and exp.btn_lsq_window.isEnabled()
        ends[workers] = win.entries[-1][1]
    assert ends[2] == pytest.approx(ends[1], rel=1e-3, abs=1e-6)
    exp.lb_maxiter.setValue(1000)
    exp.start_lbfgs()
    t0 = time.monotonic()
    while exp._fit is not None and time.monotonic() - t0 < 60:
        app.processEvents()
    assert ends[1] <= exp.fit_window.entries[-1][1] * (1 + 1e-3) + 1e-6
    exp.fit_method.setCurrentIndex(exp.fit_method.findData('lsq'))
    state = exp.session_state('.')
    assert state['fit']['lsq_maxiter'] == 50 and \
        state['fit']['method'] == 'lsq'


def test_global_methods(app):
    """Basin hopping, dual annealing, SHGO and DIRECT run from their pages
    in the fit window, counted in their own units from the start (0), and
    get near the minimum L-BFGS-B finds; their options reach run_global
    and are kept in the session."""
    import time
    from fit.de.global_fit import UNITS
    exp = tab(QA, QA, licorne=False)
    exp.fit_workers.setValue(1)
    exp.lb_maxiter.setValue(1000)
    exp.start_lbfgs()
    t0 = time.monotonic()
    while exp._fit is not None and time.monotonic() - t0 < 60:
        app.processEvents()
    best = exp.fit_window.entries[-1][1]
    w = exp._global_widgets
    w['basinhopping']['niter'].setValue(3)
    w['dual_annealing']['maxiter'].setValue(20)
    w['shgo']['n'].setValue(16)
    w['direct']['maxiter'].setValue(30)
    for k, method in enumerate(('basinhopping', 'dual_annealing', 'shgo',
                                'direct')):
        exp.fit_report.setText('untouched')
        exp._global_buttons[k].click()
        win = exp.fit_window
        assert win.method == method and win.unit == UNITS[method]
        t0 = time.monotonic()
        while exp._fit is not None and time.monotonic() - t0 < 120:
            app.processEvents()
        assert exp._fit is None and not win.running
        labels = [e[0] for e in win.entries]
        if method == 'shgo':                # ends with its minima instead
            assert labels[0] == '1 — global'
        else:
            assert labels[0] == '0' and all(l.isdigit() for l in labels[:-1])
        assert 'Done</b> after' in exp._global_reports[method].text()
        assert exp.fit_report.text() == 'untouched'
        assert exp._global_windows[k].isEnabled()
        assert min(e[1] for e in win.entries) <= best * 1.5 + 1e-3, method
    assert exp._global_options('direct')['maxfun'] is None
    w['shgo']['sampling_method'].setCurrentIndex(1)
    w['direct']['locally_biased'].setChecked(False)
    state = exp.session_state('.')
    w['shgo']['sampling_method'].setCurrentIndex(0)
    w['direct']['locally_biased'].setChecked(True)
    w['basinhopping']['niter'].setValue(100)
    exp.restore_state(state, '.')
    assert exp._global_options('shgo')['sampling_method'] == 'halton'
    assert exp._global_options('direct')['locally_biased'] is False
    assert exp._global_options('basinhopping')['niter'] == 3


def test_shgo_minima_in_window_and_table(app):
    """At its end, SHGO's global and local minima are the fit window's
    entries (global first) and fill the multistart table."""
    import time
    exp = tab(QA, QA, licorne=False)
    exp.fit_workers.setValue(1)
    exp._global_widgets['shgo']['n'].setValue(16)
    exp.start_fit('shgo')
    t0 = time.monotonic()
    while exp._fit is not None and time.monotonic() - t0 < 120:
        app.processEvents()
    win = exp.fit_window
    labels = [e[0] for e in win.entries]
    assert labels[0] == '1 — global'
    assert all(l.endswith('— local') for l in labels[1:])
    costs = [e[1] for e in win.entries]
    assert costs == sorted(costs)
    assert 'global' in win.status.text()
    assert exp.ms_table.rowCount() == len(labels)
    assert exp.ms_report.text().startswith('SHGO:')
    assert win.btn_send.isEnabled()


def test_method_box_shows_its_options(app):
    """The Method box shows only the chosen method's options; Start DREAM
    from the multistart table switches it to DREAM; it is restored from
    the session."""
    exp = tab(QA, QA, licorne=False)
    keys = [exp.fit_method.itemData(k) for k in range(exp.fit_method.count())]
    assert keys == ['de', 'basinhopping', 'dual_annealing', 'shgo',
                    'direct', 'lbfgs', 'lsq', 'multistart', 'dream']
    for key in keys:
        exp.fit_method.setCurrentIndex(exp.fit_method.findData(key))
        assert [k for k, p in exp._method_pages.items()
                if not p.isHidden()] == [key]
    state = exp.session_state('.')
    exp.fit_method.setCurrentIndex(0)
    exp.restore_state(state, '.')
    assert exp.fit_method.currentData() == 'dream'
    exp.fit_method.setCurrentIndex(0)
    exp.start_dream = lambda: None
    exp._ms_dream()
    assert exp.fit_method.currentData() == 'dream'


def test_multistart_in_fit_window(app, monkeypatch):
    """The multistart page's Run follows the starts in the fit window and
    lists the minima there, as in its table; one is sent to the Simulation
    tab from the window."""
    import time
    exp = tab(QA, QA, licorne=False)
    exp.fit_workers.setValue(1)
    exp.ms_starts.setValue(4)
    from fit.de.de_window import DEWindow
    fills = []                      # (running, starts listed) per refill
    fill = DEWindow._fill_minima

    def record(self, minima):
        fill(self, minima)
        fills.append((self.running, sum(m['hits'] for m in minima)))
    monkeypatch.setattr(DEWindow, '_fill_minima', record)
    rows = []           # starts in the page's table, Apply enabled
    status = type(exp)._ms_status

    def record_table(self, info):
        status(self, info)
        if info['phase'] == 'done':
            rows.append((sum(m['hits'] for m in self._ms_minima),
                         self.btn_ms_apply.isEnabled()))
    monkeypatch.setattr(type(exp), '_ms_status', record_table)
    exp.start_multistart()
    win = exp.fit_window
    assert win.isVisible() and win.running and win.unit == 'minimum'
    assert exp.btn_ms_window.isEnabled() and win.btn_stop.isEnabled()
    t0 = time.monotonic()
    while exp._ms is not None and time.monotonic() - t0 < 60:
        app.processEvents()
    # the list grew start by start while it ran, not only at the end
    assert [n for running, n in fills if running] == [1, 2, 3, 4, 5]
    # and so did the page's table, its buttons waiting for the end
    assert rows == [(n, False) for n in range(1, 6)]
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


def test_each_fit_a_tab(app):
    """With page_tabs, every fit opens a tab of its own, named short; the
    older one stays, can still send its entries but not continue, and
    its × closes it."""
    import time
    exp = tab(QA, QA, licorne=False)
    exp.sets[0]['name'] = 'a dataset with a long name'
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    exp.fit_workers.setValue(1)
    exp.lb_maxiter.setValue(20)
    tabs = QtWidgets.QTabWidget()
    tabs.addTab(QtWidgets.QWidget(), 'tool')
    exp.page_tabs = tabs

    def fit():
        exp.start_lbfgs()
        t0 = time.monotonic()
        while exp._fit is not None and time.monotonic() - t0 < 60:
            app.processEvents()
        return exp.fit_window
    first = fit()
    second = fit()
    assert first is not second and tabs.count() == 3
    assert tabs.currentWidget() is second
    assert tabs.tabText(1) == 'L-BFGS-B 1 · a dataset with a…'
    assert tabs.tabText(2).startswith('L-BFGS-B 2 · ')
    assert 'a dataset with a long name' in tabs.tabToolTip(1)
    assert not first.btn_continue.isEnabled()
    # the old tab's entry is sent, with its own problem
    exp.simulation.stack.layers[1].thickness = 1.0
    first.history.setCurrentItem(first.history.topLevelItem(0))
    first.btn_send.click()
    assert exp.simulation.stack.layers[1].thickness == \
        pytest.approx(first.entries[0][2][0])
    exp.close_page(first)
    assert tabs.count() == 2 and exp.fit_window is second
    exp.close_page(second)
    assert tabs.count() == 1 and exp.fit_window is None
    assert not exp.btn_lbfgs_window.isEnabled()


def test_residuals_plot(app):
    """The fit tab's last plot: (R_data - R_model) / sigma of every fitted
    point with the fit's sigma, so their mean square is the cost listed for
    the entry (chi^2, error floor included); under log R, dR is used."""
    import time
    from fit.de.de_window import RESIDUALS
    from tabs.simulation import REFL_QUANTITIES
    for cost, floor in ((0, 2.0), (1, 0.0)):
        exp = tab(QA, QB, licorne=False)
        exp.fit_workers.setValue(1)
        exp.lb_maxiter.setValue(5)
        exp.fit_cost.setCurrentIndex(cost)
        exp.fit_floor.setValue(floor)
        exp.start_lbfgs()
        t0 = time.monotonic()
        while exp._fit is not None and time.monotonic() - t0 < 60:
            app.processEvents()
        win = exp.fit_window
        view = win.views[0]
        view.quantity.setCurrentText(RESIDUALS)
        assert view.quantity.currentIndex() == len(REFL_QUANTITIES)
        assert view.formula.text() == '(R_data − R_model) / σ'
        assert not view.logy.isEnabled() and not view.show_err.isEnabled()
        res = view.residuals()
        assert [n for n, *_ in res] == ['R+', 'R-']
        assert [len(q) for _, q, _ in res] == [len(QA), len(QB)]
        r = np.concatenate([r for *_, r in res])
        label, last, x = win.entries[-1]
        if cost == 0:
            assert np.mean(r**2) == pytest.approx(last, rel=1e-9)
        else:                         # data: R = 0.1, dR = 0.01
            m = np.concatenate(win.problem.model(x))
            assert np.allclose(r, (0.1 - m) / 0.01)
        legend = [lab.text for _, lab in view.refl_plot.legend.items]
        assert legend[0].startswith('R+  χ²/N')
        # another entry: computed again for it
        win.history.setCurrentItem(win.history.topLevelItem(0))
        assert view.residuals() is not res
        view.quantity.setCurrentIndex(0)
        assert view.logy.isEnabled() and view.show_err.isEnabled()


def test_strategy_reaches_the_fit_and_the_session(app, monkeypatch):
    """The DE page's Strategy goes to the DE fit and is kept in the
    session."""
    import time
    from fit.de import de_window
    seen = []

    class Thread(de_window.DEThread):
        def __init__(self, problem, method='de', **options):
            seen.append(options.get('strategy'))
            super().__init__(problem, method, **options)
    import fit.panel as P
    monkeypatch.setattr(P, 'DEThread', Thread)
    exp = tab(QA, QA, licorne=False)
    exp.fit_workers.setValue(1)
    exp.fit_maxiter.setValue(1)
    exp.fit_pop.setValue(5)
    exp.fit_polish.setChecked(False)
    i = exp.fit_strategy.findData('rand1bin')
    assert exp.fit_strategy.currentData() == 'best1bin' and i > 0
    exp.fit_strategy.setCurrentIndex(i)
    exp.start_fit()
    t0 = time.monotonic()
    while exp._fit is not None and time.monotonic() - t0 < 60:
        app.processEvents()
    assert seen == ['rand1bin'] and exp._fit is None
    assert 'Done</b>' in exp.fit_window.status.text()
    state = exp.session_state('.')
    assert state['fit']['strategy'] == 'rand1bin'
    exp.fit_strategy.setCurrentIndex(0)
    exp.restore_state(state, '.')
    assert exp.fit_strategy.currentData() == 'rand1bin'


def test_multistart_de_follows_polish_box(app, monkeypatch):
    """The short DEs of the multistart polish only if the DE page's polish
    box is ticked."""
    import fit.panel as P
    seen = []
    monkeypatch.setattr(P.MultistartThread, 'start',
                        lambda self: seen.append(self.options))
    exp = tab(QA, QA, licorne=False)
    exp.ms_method.setCurrentIndex(exp.ms_method.findData('de'))
    for ticked in (False, True):
        exp.fit_polish.setChecked(ticked)
        exp.start_multistart()
        assert seen[-1]['method'] == 'de'
        assert seen[-1]['de_options']['polish'] is ticked
        exp._ms = None
