"""DREAM page and window: the new sampler settings reach sample(), and the
window reports each chain's chi^2/N and chains in different minima.

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
from fit.dream.dream_window import DreamWindow                  # noqa: E402
from fit.dream.dream_fit import sample                          # noqa: E402
from test_dream_fit import small_problem                        # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_page_defaults_and_options(app, monkeypatch):
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    Q = np.linspace(0.01, 0.1, 40)
    one = np.ones_like(Q)
    exp.sets = [dict(name='d', visible=True, channels={
        '+': dict(Q=Q, R=0.1 * one, dR=0.01 * one, dQ=0 * one,
                  theta=np.nan * one, path='x.dat')})]
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    assert exp.dream_rhat.value() == 1.1
    assert exp.dream_snooker.value() == 0.1 and exp.dream_temper.isChecked()
    exp.dream_start.setCurrentIndex(exp.dream_start.findData('current'))
    seen = {}
    monkeypatch.setattr(exp, '_launch_dream',
                        lambda problem, options, report: seen.update(options))
    exp.start_dream()
    assert seen['p_snooker'] == 0.1 and seen['x0'] == 'current'
    assert seen['beta_start'] == pytest.approx(1e-3)
    exp.dream_temper.setChecked(False)
    exp.start_dream()
    assert seen['beta_start'] == 1.0


def test_window_chain_chi2(app):
    pp = small_problem()
    res = sample(pp, n_gen=300, seed=0, stop_when_converged=False)
    win = DreamWindow()
    win.start(pp, 300, 'test')
    win.finish(res)
    chi2 = win.trace.chain_chi2
    assert len(chi2) == 8
    ref = [pp.chi2_per_point(x) for x in res.chains[-1]]
    assert np.allclose(chi2, ref, rtol=1e-8)
    win.trace.combo.setCurrentIndex(0)
    assert 'χ²/N per chain now' in win.trace.info.text()
    win.close()


def test_error_scale_suggested(app):
    """After an optimum with chi^2/N > 2 the DREAM page fits the error
    scale and says why; below, it is left alone."""
    from fit.problem import FitProblem
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    Q = np.linspace(0.01, 0.1, 40)
    R = sim.stack.reflectivities(Q, [(np.array([1.0, 0, 0]), None)])[:, 0]
    x = [p[2] for p in sim.stack.free_parameters()]
    for k, ticked in ((3.0, True), (1.0, False)):
        exp.dream_sigma.setChecked(False)
        p = FitProblem(sim.stack, [dict(Q=Q, R=R * (1 + 0.05 * (-1) ** np.arange(40)),
                                        dR=0.05 * R / k, P0=np.array([1.0, 0, 0]),
                                        P=None)])
        exp._suggest_sigma(p, x)
        assert exp.dream_sigma.isChecked() == ticked
        assert ('underestimate the misfit by ×3' in exp.dream_note.text()) \
            == ticked


def test_error_floor_reaches_the_problems(app, monkeypatch):
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    Q = np.linspace(0.01, 0.1, 40)
    one = np.ones_like(Q)
    exp.sets = [dict(name='d', visible=True, channels={
        '+': dict(Q=Q, R=0.1 * one, dR=0.001 * one, dQ=0 * one,
                  theta=np.nan * one, path='x.dat')})]
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    exp.fit_floor.setValue(5.0)
    exp.dream_floor.setChecked(True)
    seen = {}
    monkeypatch.setattr(exp, '_launch_dream',
                        lambda problem, options, report: seen.update(p=problem))
    exp.start_dream()
    pp = seen['p']
    assert pp.fit.error_floor == 0.05 and pp.floor_free
    assert np.allclose(pp.fit.channels[0][3], np.hypot(0.001, 0.005))
    assert np.allclose(pp.data[0][1], 0.001)        # the fitted floor wins
    assert pp.names[-1] == 'log10_error_floor'


def test_chain_curves_over_the_bands(app):
    """Each chain's best kept state is drawn in its Traces colour on the
    Reflectance, SA and SLD plots of a dataset's tab, and the tab's
    "chains" box shows or hides them; its tooltip lists their chi^2/N."""
    from fit.dream.dream_fit import chain_best, chain_curves, profile_bands
    from fit.dream.dream_fit import posterior_problem
    from fit.dream.dream_window import CHAIN_COLOURS
    from test_dream_fit import film, free, measure
    data = measure(film(), ['+', '-'])
    for d, name in zip(data, ('R+', 'R-')):
        d['name'] = name                # as the app names them: SA exists
    pp = posterior_problem(free(film(), ()), data)
    res = sample(pp, n_gen=300, seed=0, stop_when_converged=False)
    best = chain_best(res)
    assert best.shape == (8, 3)
    z = profile_bands(res, pp)[0]
    cc = chain_curves(res, pp, z)
    assert [c['chain'] for c in cc] == list(range(8))
    for c, x in zip(cc, best):
        R = pp.fit.model(x)
        assert all(np.array_equal(a, b) for a, b in zip(c['R'], R))
        assert np.allclose(c['A'], (R[0] - R[1]) / (R[0] + R[1]))
        assert c['chi2'] == pytest.approx(pp.chi2_per_point(x))
        assert c['nuc'].shape == c['mag'].shape == z.shape
    win = DreamWindow()
    win.start(pp, 300, 'test')
    win.finish(res)
    win.tabs.setCurrentIndex(3)
    page, pred, prof, view, box = win.part_pages[0]
    assert page.currentIndex() == 1, page.widget(0).text()
    chains = [it for kind, _, it in pred.items if kind == 'chains']
    assert len(chains) == 8 * 2                     # R+ and R-, per chain
    assert chains[3].opts['pen'].color().name() == CHAIN_COLOURS[3]
    assert len(prof._chain_items) == 8 * 2          # NSLD and MSLD
    assert box.isChecked() and 'chain 8: χ²/N' in box.toolTip()
    view.setCurrentText('SLD')
    win.show()
    app.processEvents()
    assert all(it.isVisible() for it in prof._chain_items)
    box.setChecked(False)
    assert not any(it.isVisible() for it in prof._chain_items)
    view.setCurrentText('SA')
    sa = [it for kind, _, it in pred.items if kind == 'chains']
    assert len(sa) == 8 and not any(it.isVisible() for it in sa)
    box.setChecked(True)
    assert all(it.isVisible() for it in sa)
    # Load into model: the MAP, or one chain's best kept state
    assert win.send_from.count() == 9
    assert win.send_from.itemText(0).startswith('MAP — χ²/N')
    assert win.send_from.itemText(3).startswith('chain 3 — χ²/N')
    sent = []
    win.sendRequested.connect(lambda *a: sent.append(a))
    win.btn_map.click()
    win.send_from.setCurrentIndex(3)
    win.btn_map.click()
    assert np.array_equal(sent[0][0], res.best()[0]) and sent[0][2] == 'MAP'
    assert np.array_equal(sent[1][0], best[2]) and sent[1][1:] == (0, 'chain 3')
    win.close()


def test_log_p_line_never_widens_the_window(app):
    """With log p shown, the trace line lists every chain's chi^2/N and is
    rewritten at each live update: it wraps, so 26 chains (classic DREAM
    on 25 parameters) do not force the window wider than a laptop screen
    (it reached 1507 px before, and the window stopped responding)."""
    pp = small_problem()
    win = DreamWindow()
    win.resize(900, 700)
    win.start(pp, 30000, 'test')
    win.show()
    win.tabs.setCurrentWidget(win.trace)
    base = win.minimumSizeHint().width()
    N = 26
    g = np.arange(100) * 8
    X = np.random.default_rng(0).normal(size=(100, N, pp.ndim))
    lp = -650 + np.random.default_rng(1).normal(size=(100, N))
    win.snapshot(g, X, lp, None)
    win.trace.combo.setCurrentIndex(0)                  # log p
    win.trace.chain_chi2 = list(1.2345 + 0.001 * np.arange(N))
    win.trace._draw()
    win.status.setText('Generation 12345 / 30000 — acceptance 0.123 — '
                       'max R̂ 1.234 — elapsed 1 h 05 min' * 3)
    app.processEvents()
    assert 'χ²/N per chain now' in win.trace.info.text()
    assert win.minimumSizeHint().width() == base and win.width() == 900
    win.close()


def test_log_p_line_never_widens_the_window(app):
    """With log p shown, the trace line lists every chain's chi^2/N and is
    rewritten at each live update: it wraps, so 26 chains (classic DREAM
    on 25 parameters) do not force the window wider than a laptop screen
    (it reached 1507 px before, and the window stopped responding)."""
    pp = small_problem()
    win = DreamWindow()
    win.resize(900, 700)
    win.start(pp, 30000, 'test')
    win.show()
    win.tabs.setCurrentWidget(win.trace)
    base = win.minimumSizeHint().width()
    N = 26
    g = np.arange(100) * 8
    X = np.random.default_rng(0).normal(size=(100, N, pp.ndim))
    lp = -650 + np.random.default_rng(1).normal(size=(100, N))
    win.snapshot(g, X, lp, None)
    win.trace.combo.setCurrentIndex(0)                  # log p
    win.trace.chain_chi2 = list(1.2345 + 0.001 * np.arange(N))
    win.trace._draw()
    win.status.setText('Generation 12345 / 30000 — acceptance 0.123 — '
                       'max R̂ 1.234 — elapsed 1 h 05 min' * 3)
    app.processEvents()
    assert 'χ²/N per chain now' in win.trace.info.text()
    assert win.minimumSizeHint().width() == base and win.width() == 900
    win.close()
