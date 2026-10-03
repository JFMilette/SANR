"""Model curves drawn against data: with the 'licorne' resolution scheme the
live simulation dataset and the DE window compute on the data's Q points,
as the fit does (tabs.experimental.model_grid), with the fitted dataset's
fixed norms.

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


def test_de_window_draws_the_fitted_model(app):
    sim, exp = tab()
    s, vec, data, qmin, qmax, grid = exp._fit_inputs(exp.fit_report)
    fp = FitProblem(sim.stack, data, 'chi2', qmin, qmax, grid)
    win = DEWindow(exp)
    grids = model_grid(sim.stack, sim.last_Q, s['channels'], grid)
    win.start(fp, {}, dict(exp.sim_vectors(), **vec), grids, 'd', 'χ²')
    x = fp.to_x(fp.x0())
    for (name, *_), m in zip(fp.channels, fp.model(x)):
        drawn = win._model[name[1:]]
        assert np.array_equal(drawn['Q'], Q1)
        assert np.max(np.abs(drawn['R'][Q1 <= qmax] / m - 1)) <= 1e-12
    win.close()


def test_live_curves_on_the_data_grid(app):
    sim, exp = tab()
    exp.add_simulation()
    live = exp.sets[-1]['channels']
    assert all(np.array_equal(c['Q'], Q1) for c in live.values())
    # the quadrature keeps the Simulation tab's grid
    sim, exp = tab('quadrature')
    exp.add_simulation()
    assert all(np.array_equal(c['Q'], sim.last_Q)
               for c in exp.sets[-1]['channels'].values())


def test_live_curves_take_the_norms(app):
    sim, exp = tab('quadrature', norm=0.5)
    exp.add_simulation()
    live = exp.sets[-1]['channels']
    Q = sim.last_Q
    pair = vectors_pair(*exp.channel_vectors(exp.sets[-1], '+'))
    st = sim.stack
    bare = st.reflectivities(Q, [pair])[:, 0] - st.background
    assert np.allclose(live['+']['R'], 0.5 * bare + st.background,
                       rtol=1e-12, atol=0)
