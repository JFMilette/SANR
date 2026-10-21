"""model.stack Q SHIFT: the model moved along Q, fitted like the scale and
background, and its row in the Simulation tab's general parameters.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import model.stack as S                                 # noqa: E402
from fit.de.de_fit import run_lbfgs                     # noqa: E402
from fit.problem import FitProblem                      # noqa: E402

Q = np.linspace(0.005, 0.15, 300)
UP = (np.array([1.0, 0, 0]), None)


def film(shift=0.0, resolution=False):
    st = S.Stack([S.Layer('vacuum', 0, 0, 0, 0, 0),
                  S.Layer('film', 150, 4e-6, 0, 1e-6, 0, 3.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0)])
    st.q_shift = shift
    if resolution:
        st.resolution.update(enabled=True, mode='tof', tof_dlambda=0.02,
                             tof_angles=[dict(theta=0.01, dtheta=0.0002,
                                              qmax=None)])
    st.build_sublayers()
    return st


def test_shift_moves_the_curve():
    """R at Q is the unshifted R at Q - shift; with the resolution on, the
    instrument's sigma stays the one at the measured Q; below zero, |Q|."""
    d = 0.0012
    R = film(d).reflectivities(Q, [UP])
    assert np.allclose(R, film().reflectivities(Q - d, [UP]), rtol=1e-12)
    st0, st = film(resolution=True), film(d, resolution=True)
    want = st0.reflectivities(Q - d, [UP], sigma=st0.resolution_sigma(Q))
    assert np.allclose(st.reflectivities(Q, [UP]), want, rtol=1e-12)
    assert not np.allclose(st.reflectivities(Q, [UP]),
                           st0.reflectivities(Q - d, [UP]), rtol=1e-6)
    # Licorne's convolution: the measured Q's weights, shifted R
    lic0, lic = film(resolution=True), film(d, resolution=True)
    for st_ in (lic0, lic):
        st_.resolution.update(scheme='licorne', licorne_fun=3)
    from model.licorne_resolution import kernel
    W = kernel(Q, lic0.resolution_sigma(Q), 3)
    lic0.resolution['enabled'] = False
    ideal = lic0.reflectivities(Q - d, [UP])
    assert np.allclose(lic.reflectivities(Q, [UP]), W @ ideal, rtol=1e-12)
    q = np.array([0.0005, 0.002])
    assert np.allclose(film(0.001).reflectivities(q, [UP]),
                       film().reflectivities(np.abs(q - 0.001), [UP]))


def test_saved_and_fitted():
    """In to_dict (an old dict without it loads 0), fitted like the
    background: a fit of the shift alone finds the one in the data."""
    st = film(0.0007)
    d = st.to_dict()
    assert S.Stack.from_dict(d).q_shift == 0.0007
    d.pop('q_shift')
    assert S.Stack.from_dict(d).q_shift == 0.0
    assert S.default_fit('q_shift')['min'] == -0.002
    R = st.reflectivities(Q, [UP])[:, 0]
    model = film()
    model.fit['q_shift'] = dict(vary=True, min=-0.002, max=0.002)
    assert [p[:2] for p in model.free_parameters()] == [(None, 'q_shift')]
    p = FitProblem(model, [dict(Q=Q, R=R, dR=0.01 * R, P0=UP[0], P=None,
                                name='R+')])
    assert p.names == ['stack.q_shift']
    x, cost, *_ = run_lbfgs(p)
    assert x[0] == pytest.approx(0.0007, abs=1e-7) and cost < 1e-8


def test_simulation_tab_row():
    """Below Scale and Background: its value, bounds and Fit box reach the
    stack, a fit result is written back, the session keeps it."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    from tabs import SimulationTab
    sim = SimulationTab()
    rows = list(sim.stack_rows)
    assert rows[:3] == ['scale', 'background', 'q_shift']
    val, bmin, bmax, cb = sim.stack_rows['q_shift']
    val.setValue(0.0005)
    bmin.setValue(-0.001)
    bmax.setValue(0.001)
    cb.setChecked(True)
    sim.recompute()
    assert sim.stack.q_shift == 0.0005
    assert sim.stack.fit['q_shift'] == dict(vary=True, min=-0.001, max=0.001)
    params = [p for p in sim.stack.free_parameters() if p[1] == 'q_shift']
    sim.set_parameters(params, [-0.0003])
    assert val.value() == pytest.approx(-0.0003)
    state = sim.session_state()
    fresh = SimulationTab()                # a session opened elsewhere
    fresh.restore_state(state)
    assert fresh.stack.q_shift == pytest.approx(-0.0003)
    assert fresh.stack_rows['q_shift'][0].value() == pytest.approx(-0.0003)
