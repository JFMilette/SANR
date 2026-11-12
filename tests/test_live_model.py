"""Model curves drawn against data: the live simulation datasets on the
Simulation tab's grid with their own model's dataset's fixed norms, whatever
the Fit tab fits; with the 'licorne' resolution scheme the DE window
computes on the data's Q points, as the fit does
(tabs.experimental.model_grid).

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
from fit.de.de_window import DEWindow                           # noqa: E402
from fit.problem import FitProblem                              # noqa: E402
from model.polarisation import vectors_pair                     # noqa: E402
from tabs.experimental import model_grid                        # noqa: E402
from test_licorne_resolution import Q1, fixture1_stack          # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def tab(scheme='licorne', norm=1.0):
    """Fixture 1 in the Simulation tab (its own grid: 300 points), and a
    dataset of R+ / R- on q.dat's 204 points."""
    sim = SimulationTab()
    st = fixture1_stack(scheme)
    st.background = 1e-6
    st.layers[2].fit_entry('thickness')['vary'] = True
    d = st.to_dict()
    d['simulation'] = dict(qmin=0.01, qmax=0.22, nq=300)
    sim.restore_state(d)
    sim.recompute()
    assert sim.stack.resolution['scheme'] == scheme
    exp = ExperimentalTab(simulation=sim)
    R = sim.stack.reflectivities(Q1, [(np.array([1.0, 0, 0]), None)])[:, 0]
    one = np.ones_like(Q1)
    exp.sets = [dict(name='d', visible=True, channels={
        ch: dict(Q=Q1, R=R, dR=0.1 * R, dQ=0 * one, theta=np.nan * one,
                 path='x.dat', norm=n)
        for ch, n in (('+', norm), ('-', 1.0))})]
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    return sim, exp


def live_set(exp, link=None):
    """The live dataset (always there) of model `link` (default: the first
    one listed), up to date."""
    exp._update_live()
    return next(s for s in exp.sets if s.get('live')
                and link in (None, s.get('link')))


def test_de_window_draws_the_fitted_model(app):
    sim, exp = tab()
    s, vec, data, qmin, qmax, grid = exp._fit_inputs(exp.fit_report)
    fp = FitProblem(sim.stack, data, 'chi2', qmin, qmax, grid)
    win = DEWindow(exp)
    grids = model_grid(sim.stack, sim.last_Q, s['channels'], grid)
    win.start(fp, [dict(name='d', data={},
                        vectors=dict(exp.sim_vectors(), **vec),
                        grids=grids)], 'd', 'χ²')
    x = fp.to_x(fp.x0())
    for (name, *_), m in zip(fp.channels, fp.model(x)):
        drawn = win.views[0]._model[name[1:]]
        assert np.array_equal(drawn['Q'], Q1)
        assert np.max(np.abs(drawn['R'][Q1 <= qmax] / m - 1)) <= 1e-12
    win.close()


def test_live_curves_on_the_simulation_grid(app):
    """Every live curve is on its model's Simulation grid (Q min, Q max, Q
    points), whatever the resolution scheme and whatever the Fit tab
    fits, and follows it."""
    for scheme in ('licorne', 'quadrature'):
        sim, exp = tab(scheme)
        for link in (sim.shown_model(), 'd'):
            live = live_set(exp, link)['channels']
            assert live and all(np.array_equal(c['Q'], sim.last_Q)
                                for c in live.values())
        # no dataset chosen in the Fit tab: the same curves
        before = live_set(exp, 'd')['channels']['+']['R'].copy()
        exp.fit_set.setCurrentIndex(-1)
        assert np.array_equal(live_set(exp, 'd')['channels']['+']['R'],
                              before)
        # the shown model's Q max: its curve follows; model d keeps its
        # own range (tests/test_model_q_range)
        d_grid = sim.q_grid('d').copy()
        sim.qmax.setValue(0.15)
        sim._timer.stop()
        sim.recompute()
        assert sim.last_Q[-1] == 0.15
        assert all(np.array_equal(c['Q'], sim.last_Q) for c in live_set(
            exp, sim.shown_model())['channels'].values())
        assert all(np.array_equal(c['Q'], d_grid)
                   for c in live_set(exp, 'd')['channels'].values())


def test_live_curves_take_the_norms(app):
    """A live curve takes the norms of its own model's dataset ('d' is
    fitted with its own model, DATASET MODELS), another model's none."""
    sim, exp = tab('quadrature', norm=0.5)
    other = live_set(exp, sim.shown_model())
    s = live_set(exp, 'd')
    live = s['channels']
    Q = sim.last_Q
    pair = vectors_pair(*exp.channel_vectors(s, '+'))
    st = sim.stack
    bare = st.reflectivities(Q, [pair])[:, 0] - st.background
    assert np.allclose(live['+']['R'], 0.5 * bare + st.background,
                       rtol=1e-12, atol=0)
    assert np.allclose(other['channels']['+']['R'], bare + st.background,
                       rtol=1e-12, atol=0)


def test_frozen_copy_sends_its_model_back(app, monkeypatch):
    sim, exp = tab()
    t0 = sim.stack.layers[2].thickness
    exp.freeze_simulation(exp.sets.index(live_set(exp)))
    frozen = next(s for s in exp.sets if s.get('model'))
    assert frozen['model']['layers'][2]['thickness'] == t0
    # kept through a session
    exp.restore_state(exp.session_state('.'), '.')
    frozen = next(s for s in exp.sets if s.get('model'))
    assert frozen['model']['layers'][2]['thickness'] == t0
    sim.stack.layers[2].thickness = t0 + 25.0
    sim.refresh_list(select=2)
    sim.recompute()
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', lambda *a: (
        QtWidgets.QMessageBox.StandardButton.Yes))
    exp.send_model(exp.sets.index(frozen))
    assert sim.stack.layers[2].thickness == t0
    # the live copy follows the Simulation tab again
    live = live_set(exp)
    assert np.allclose(live['channels']['+']['R'],
                       frozen['channels']['+']['R'])


def test_send_model_keeps_the_polarisation(app, monkeypatch):
    """Send model to Simulation replaces the layers and model settings,
    not the Simulation tab's polarisation (it used to reset to the
    default); a session without one still opens with the default."""
    from model.polarisation import default_vectors
    sim, exp = tab()
    exp.freeze_simulation(exp.sets.index(live_set(exp)))
    frozen = next(s for s in exp.sets if s.get('model'))
    vec = sim.pol.vectors()
    vec['+'] = [[0.98, 0.0, 0.0], [0.0, 0.0, 0.0]]
    vec['-'] = [[-0.98, 0.0, 0.0], [0.0, 0.0, 0.0]]
    sim.pol.set_vectors(vec)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', lambda *a: (
        QtWidgets.QMessageBox.StandardButton.Yes))
    exp.send_model(exp.sets.index(frozen))
    assert sim.pol.vectors()['+'] == vec['+']
    assert sim.pol.vectors()['-'] == vec['-']
    state = sim.session_state()
    del state['simulation']['polarisation']          # an older session
    sim.restore_state(state)
    assert sim.pol.vectors() == default_vectors()
