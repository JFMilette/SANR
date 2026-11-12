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
    assert exp.dream_rhat.value() == 1.05 and exp.dream_ess.value() == 400
    assert exp.dream_snooker.value() == 0.1 and exp.dream_temper.isChecked()
    exp.dream_start.setCurrentIndex(exp.dream_start.findData('current'))
    seen = {}
    monkeypatch.setattr(exp, '_launch_dream',
                        lambda problem, options, report: seen.update(options))
    exp.start_dream()
    assert seen['p_snooker'] == 0.1 and seen['x0'] == 'current'
    assert seen['rhat_target'] == 1.05 and seen['ess_target'] == 400
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


def test_rhat_tab(app):
    """The R̂ tab: the max R-hat over the parameters at each generation
    it was computed, the run's target, where it was first reached; the
    live thread hands the same curve as it goes."""
    from fit.dream.dream_window import DreamThread, rhat_curve
    pp = small_problem()
    # a 10 % adaptation phase (auto_adapt would take half of 600)
    res = sample(pp, n_gen=600, seed=0, stop_when_converged=False,
                 adapt_until=60, rhat_target=1.2)
    win = DreamWindow()
    win.start(pp, 600, 'test', adapt_until=60)
    assert win.tabs.tabText(2) == 'R̂' and win.tabs.widget(2) is win.rhat
    assert len(win.rhat.gens) == 0
    win.finish(res)
    g, r = rhat_curve(res.rhat_history)
    assert len(g) > 3 and g[0] > res.adapt_until
    assert np.array_equal(win.rhat.gens, g)
    assert np.allclose(win.rhat.rmax, [np.max(R) for _, R in
                                       res.rhat_history])
    assert win.rhat.rhat_target == 1.2 and win.rhat.target.isVisible()
    assert win.rhat.target.value() == pytest.approx(np.log10(1.2))
    assert 'target 1.2' in win.rhat.info.text()
    win.close()
    # live: the thread's curve ends as the result's
    got = []
    th = DreamThread(pp, n_gen=600, seed=0, stop_when_converged=False,
                     adapt_until=60, rhat_target=1.2)
    th.rhat.connect(lambda gg, rr, t: got.append((gg, rr, t)))
    th.run()                                  # in this thread
    app.processEvents()
    g2, r2 = rhat_curve(th.result.rhat_history)
    assert got and got[-1][2] == 1.2
    last_g, last_r = got[-1][0], got[-1][1]
    assert np.array_equal(last_g, g2[:len(last_g)])
    assert np.allclose(last_r, r2[:len(last_r)])


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
    win.tabs.setCurrentWidget(win.part_pages[0][0])
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
    # Load into model: the MAP, the median or one chain's best kept state
    med = np.median(res.samples(), axis=0)
    assert win.send_from.count() == 10
    assert win.send_from.itemText(0).startswith('MAP — χ²/N')
    assert win.send_from.itemText(1) == 'median — χ²/N %.4g' %         pp.chi2_per_point(med)
    assert win.send_from.itemText(4).startswith('chain 3 — χ²/N')
    assert win.send_from.itemIcon(1).isNull()           # no chain colour
    assert win.send_from.itemIcon(4).pixmap(10, 10).toImage().pixelColor(
        5, 5).name() == CHAIN_COLOURS[2]
    sent = []
    win.sendRequested.connect(lambda *a: sent.append(a))
    win.btn_map.click()
    win.send_from.setCurrentIndex(4)
    win.btn_map.click()
    win.send_from.setCurrentIndex(1)
    win.btn_map.click()
    assert np.array_equal(sent[0][0], res.best()[0]) and sent[0][1] == 'MAP'
    assert np.array_equal(sent[1][0], best[2]) and sent[1][1] == 'chain 3'
    assert np.array_equal(sent[2][0], med) and sent[2][1] == 'median'
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


def test_continue_any_ended_run(app, monkeypatch):
    """Continue, with the '+ gen.' box beside it, adds that many
    generations to any ended run (not only one stopped with Stop), all of
    them; a stopped run's box starts at what it had left."""
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
    seen = {}
    monkeypatch.setattr(exp, '_launch_dream',
                        lambda problem, options, report: seen.update(options))
    exp.start_dream()
    win, pp = exp.dream_window, exp._dream_problem
    assert not win.continue_gen.isEnabled()
    # a run that went to its end
    res = sample(pp, n_gen=300, seed=0, stop_when_converged=False)
    win.finish(res)
    assert win.btn_continue.isEnabled() and win.continue_gen.isEnabled()
    win.continue_gen.setValue(500)
    seen.clear()
    win.btn_continue.click()
    assert seen['resume'] is res and seen['n_gen'] == 800
    assert seen['stop_when_converged'] is False
    assert not win.continue_gen.isEnabled()          # running again
    # a run stopped with Stop: the box offers what it had left
    stopped = sample(pp, n_gen=1000, seed=0, stop_when_converged=False,
                     callback=lambda t, *a: t >= 200)
    win.finish(stopped)
    assert win.continue_gen.value() == 800 and win.btn_continue.isEnabled()
    win.close()


def test_prune_and_continue(app, monkeypatch):
    """The page's adaptation phase (auto) and sample window reach
    sample(); Prune & continue, offered after a burn-in that went on past
    the adaptation phase, continues a repruned copy of the run."""
    import fit.dream.dream_window as dw
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
    opts = exp.page_options('dream')
    assert opts['adaptation'] == 0 and opts['window'] == 0.5
    seen = {}
    monkeypatch.setattr(exp, '_launch_dream',
                        lambda problem, options, report: seen.update(options))
    exp.start_dream()
    assert seen['adapt_until'] == 5000 and seen['discard'] == 0.5
    exp.dream_adapt.setValue(1000)
    exp.dream_window_box.setCurrentIndex(
        exp.dream_window_box.findData(0.0))
    exp.start_dream()
    assert seen['adapt_until'] == 1000 and seen['discard'] == 0.0
    win, pp = exp.dream_window, exp._dream_problem
    assert win.trace.adapt_until == 1000 and win.trace.discard == 0.0
    res = sample(pp, n_gen=600, seed=0, stop_when_converged=False,
                 adapt_until=60)
    monkeypatch.setattr(dw, 'burn_in_end', lambda r: None)
    win.finish(res)
    assert win.btn_continue.isEnabled() and not win.btn_prune.isEnabled()
    monkeypatch.setattr(dw, 'burn_in_end', lambda r: 300)
    win.finish(res)
    assert win.burn_in == 300 and win.btn_prune.isEnabled()
    win.continue_gen.setValue(400)
    seen.clear()
    win.btn_prune.click()
    assert seen['n_gen'] == 1000 and seen['resume'] is not res
    assert seen['resume'].state['adapt_until'] == 300
    assert res.state['adapt_until'] == 60          # the run itself kept
    assert win.trace.adapt_until == 300 and not win.btn_prune.isEnabled()
    win.close()
