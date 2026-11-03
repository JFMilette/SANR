"""fit.problem TARGET: fitting (R+ + R-)/2 or the spin asymmetry of a
dataset instead of its channels.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import model.stack as S                                 # noqa: E402
from fit.de.de_fit import run_lbfgs                     # noqa: E402
from fit.joint import JointProblem                      # noqa: E402
from fit.problem import FitProblem                      # noqa: E402
from model.polarisation import vectors_pair             # noqa: E402

Q = np.linspace(0.008, 0.12, 200)
PAIRS = {'+': ([1.0, 0, 0], [0, 0, 0]), '-': ([-1.0, 0, 0], [0, 0, 0])}


def film(rho=1e-6, nsld=4e-6, scale=1.0):
    """vacuum / magnetic film 120 A (M along x) / Si, no background."""
    st = S.Stack([S.Layer('vacuum', 0, 0, 0, 0, 0),
                  S.Layer('film', 120, nsld, 0, rho, 0, 3.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0)])
    st.background, st.scale = 0.0, scale
    st.build_sublayers()
    return st


def measure(st, channels='+-', q=Q):
    out = []
    for c in channels:
        R = st.reflectivities(q, [vectors_pair(*PAIRS[c])], warn=False)[:, 0]
        P0, P = vectors_pair(*PAIRS[c])
        out.append(dict(Q=q, R=R, dR=0.02 * R, P0=P0, P=P, name='R' + c))
    return out


def vary(st, attr, lo, hi):
    st.layers[1].fit[attr] = {'vary': True, 'min': lo, 'max': hi}
    return st


def test_sum_and_asym_residuals():
    """Values and propagated errors, point by point."""
    data = measure(film(rho=1.5e-6))
    st = vary(film(rho=1e-6), 'MSLD_rho', 0, 5e-6)
    a, b = data[0]['R'], data[1]['R']
    da, db = data[0]['dR'], data[1]['dR']
    m = FitProblem(st, data).model([1e-6])
    for target in ('sum', 'asym'):
        p = FitProblem(st, data, target=target)
        assert p.npoints == len(Q)
        if target == 'sum':
            want = ((m[0] + m[1]) / 2 - (a + b) / 2) / (np.hypot(da, db) / 2)
        else:
            sa = (a - b) / (a + b)
            want = ((m[0] - m[1]) / (m[0] + m[1]) - sa) / (
                2 * np.hypot(b * da, a * db) / (a + b)**2)
        assert np.allclose(p.residuals([1e-6]), want, rtol=1e-12)


def test_sum_hardly_sees_the_msld():
    """(R+ + R-)/2 has no first-order term in M: a 20 % MSLD error costs
    far less than in the channels while the MSLD is small next to the NSLD
    (second order: the ratio grows with it), and the asymmetry sees it
    most."""
    ratio = []
    for rho in (0.25e-6, 1e-6):
        data = measure(film(rho=rho))
        cost = {}
        for target in ('channels', 'sum', 'asym'):
            p = FitProblem(vary(film(), 'MSLD_rho', 0, 5e-6), data,
                           target=target)
            cost[target] = p(p.to_u([1.2 * rho]))
        assert cost['asym'] > cost['channels']
        ratio.append(cost['sum'] / cost['channels'])
    assert ratio[0] < 0.03 and ratio[0] < ratio[1] < 0.25


def test_asym_fit_ignores_a_wrong_scale():
    """Data 20 % above the model's scale: the channels' fit of the MSLD is
    pulled off, the asymmetry's finds it (the scale cancels in SA)."""
    data = measure(film(rho=1.2e-6, scale=1.2))
    found = {}
    for target in ('channels', 'asym'):
        p = FitProblem(vary(film(rho=0.5e-6), 'MSLD_rho', 0, 3e-6), data,
                       target=target)
        x, *_ = run_lbfgs(p)
        found[target] = x[0]
    assert found['asym'] == pytest.approx(1.2e-6, rel=1e-4)
    assert abs(found['channels'] - 1.2e-6) > 1e-8


def test_pairs_at_common_q_only():
    """R- without two of R+'s points: those are left out, with a warning."""
    data = measure(film())
    keep = np.ones(len(Q), bool)
    keep[[10, 50]] = False
    data[1] = {k: v[keep] if isinstance(v, np.ndarray) and len(v) == len(Q)
               else v for k, v in data[1].items()}
    st = vary(film(), 'thickness', 100, 140)
    with pytest.warns(UserWarning, match='2 point'):
        p = FitProblem(st, data, target='sum')
    assert p.npoints == len(Q) - 2
    assert np.allclose(p.residuals([120.0]), 0, atol=1e-9)


def test_pairs_points_a_little_apart_in_q():
    """R- a few 1e-4 of Q off R+ (two spin states reduced apart): every
    point paired, no warning, the model taken at each channel's own Q."""
    import warnings
    from fit.problem import PAIR_RTOL
    data = measure(film())
    data[1] = measure(film(), q=Q * (1 + 0.4 * PAIR_RTOL))[1]
    st = vary(film(), 'thickness', 100, 140)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        p = FitProblem(st, data, target='sum')
    assert p.npoints == len(Q)
    assert np.allclose(p.residuals([120.0]), 0, atol=1e-9)


def test_needs_the_pair():
    st = vary(film(), 'thickness', 100, 140)
    with pytest.raises(ValueError, match='needs both R\\+ and R−'):
        FitProblem(st, measure(film(), '+'), target='sum')
    with pytest.raises(ValueError, match='unknown target'):
        FitProblem(st, measure(film()), target='R+')


def test_spin_flip_still_fitted_and_nsf_pair():
    """With the four channels, R++ and R-- are combined and R+- and R-+
    still fitted on their own, as with 'channels'."""
    st = film()
    lab = {'++': ([1.0, 0, 0], [1.0, 0, 0]), '+-': ([1.0, 0, 0], [-1.0, 0, 0]),
           '-+': ([-1.0, 0, 0], [1.0, 0, 0]), '--': ([-1.0, 0, 0], [-1.0, 0, 0])}
    data = []
    for c, pair in lab.items():
        P0, P = vectors_pair(*pair)
        R = st.reflectivities(Q, [(P0, P)], warn=False)[:, 0] + 1e-7
        data.append(dict(Q=Q, R=R, dR=0.02 * R, P0=P0, P=P, name='R' + c))
    st = vary(st, 'thickness', 100, 140)
    p = FitProblem(st, data, target='asym')
    names = [ch[0] for ch in p.channels]
    assert (names[p.combined[0]], names[p.combined[1]]) == ('R++', 'R--')
    assert [names[k] for k in p.others] == ['R+-', 'R-+']
    assert p.npoints == 3 * len(Q)
    # the spin-flip residuals are those of the channels fit
    every = FitProblem(st, data).residuals([110.0])
    r = p.residuals([110.0])
    assert np.array_equal(r[len(Q):], every[len(Q):3 * len(Q)])


def test_log_cost():
    """'sum' in log10; 'asym' as a plain difference (SA can be < 0)."""
    data = measure(film(rho=1.5e-6))
    st = vary(film(rho=1e-6), 'MSLD_rho', 0, 5e-6)
    m = FitProblem(st, data).model([1e-6])
    a, b = data[0]['R'], data[1]['R']
    p = FitProblem(st, data, cost='log', target='sum')
    assert np.allclose(p.residuals([1e-6]),
                       np.log10((m[0] + m[1]) / 2) - np.log10((a + b) / 2))
    p = FitProblem(st, data, cost='log', target='asym')
    assert np.allclose(p.residuals([1e-6]), (m[0] - m[1]) / (m[0] + m[1])
                       - (a - b) / (a + b))


def test_joint_needs_one_target():
    st = vary(film(), 'thickness', 100, 140)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = FitProblem(st, measure(film()), target='sum')
        b = FitProblem(st, measure(film(rho=2e-6)), target='sum')
        c = FitProblem(st, measure(film()))
    j = JointProblem([('a', a), ('b', b)])
    assert j.target == 'sum' and j.npoints == 2 * len(Q)
    with pytest.raises(ValueError, match='target'):
        JointProblem([('a', a), ('c', c)])


def test_panel_box(monkeypatch):
    """The Fit to box reaches the problem (DREAM keeps the channels), the
    fit window shows SA, and the session keeps the choice."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    one = np.ones_like(Q)
    exp.sets = [dict(name='d', visible=True, channels={
        c: dict(Q=Q, R=0.1 * one * (1.1 if c == '+' else 1), dR=0.01 * one,
                dQ=0 * one, theta=np.nan * one, path='x.dat')
        for c in '+-'})]
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    exp.fit_target.setCurrentIndex(exp.fit_target.findData('asym'))
    built = exp._problem(exp.fit_report, 'chi2')
    assert built[0].target == 'asym' and built[0].npoints == len(Q)
    assert exp._problem(exp.dream_report, 'chi2',
                        'channels')[0].target == 'channels'
    state = exp.session_state(str(pathlib.Path.cwd()))
    assert state['fit']['target'] == 'asym'
    exp.fit_target.setCurrentIndex(0)
    exp.restore_state(state, str(pathlib.Path.cwd()))
    assert exp.fit_target.currentData() == 'asym'
    exp._open_fit_window(*built[:4], 'lbfgs')
    win = exp.fit_window
    assert win.views[0].quantity.currentText() == 'SA half-polarized'
    assert win.history.headerItem().text(1) == 'χ² of SA'


def test_pairs_stay_within_one_angle():
    """Channels with different numbers of rows (R- without its last point),
    two angles overlapping in Q, R- of angle 2 a little off R+: where the
    nearest R- point is the other angle's, the theta column pairs each
    point with its own angle's."""
    q1 = np.linspace(0.01, 0.05, 40)
    q2 = np.linspace(0.04, 0.12, 40)
    # an angle-1 point 1e-5 of Q from an angle-2 point, whose R- is 3e-4
    # of Q up: the nearest R- of that angle-2 point is angle 1's
    q1[np.argmin(np.abs(q1 - q2[3]))] = q2[3] * (1 + 1e-5)
    th = np.concatenate([np.full(40, 0.008), np.full(40, 0.019)])
    q_up = np.concatenate([q1, q2])
    q_dn = np.concatenate([q1, q2 * (1 + 3e-4)])[:-1]
    data = [measure(film(), q=q_up)[0], measure(film(), q=q_dn)[1]]
    data[0]['theta'], data[1]['theta'] = th, th[:-1] * (1 + 1e-4)
    st = vary(film(), 'thickness', 100, 140)
    with pytest.warns(UserWarning, match='1 point'):
        p = FitProblem(st, data, target='sum')
    assert p.npoints == 79
    assert np.allclose(p.residuals([120.0]), 0, atol=1e-9)
    # without theta, the crossing pair is lost too
    for d in data:
        d.pop('theta')
    with pytest.warns(UserWarning, match='3 point'):
        FitProblem(st, data, target='sum')


def test_same_rows_paired_row_by_row():
    """As many rows: row k with row k, however far apart their Q (a
    stitched file in file order, R- of one angle well off R+), and only
    the rows fitted in both."""
    from fit.problem import PAIR_RTOL
    q = np.concatenate([np.linspace(0.04, 0.12, 30),
                        np.linspace(0.01, 0.05, 30)])     # Q falls back
    q_dn = q * np.where(np.arange(60) < 30, 1 + 5 * PAIR_RTOL, 1.0)
    data = [measure(film(), q=q)[0], measure(film(), q=q_dn)[1]]
    data[1]['R'] = data[1]['R'].copy()
    data[1]['R'][7] = np.nan                              # not fitted
    st = vary(film(), 'thickness', 100, 140)
    with pytest.warns(UserWarning, match='1 point.*one channel'):
        p = FitProblem(st, data, target='sum')
    assert p.npoints == 59
    assert np.allclose(p.residuals([120.0]), 0, atol=1e-9)


def test_plots_pair_row_by_row():
    from tabs.experimental import unpolarized
    q = np.array([0.03, 0.04, 0.01, 0.02])                # two angles
    D = {'+': dict(Q=q, R=np.array([1., 2, 3, 4]), dR=0.1 * np.ones(4),
                   dQ=0 * q),
         '-': dict(Q=q * 1.01, R=np.array([3., 4, 5, 6]),
                   dR=0.1 * np.ones(4), dQ=0 * q)}
    Q, v, e, _ = unpolarized(D)
    assert np.array_equal(Q, q) and np.allclose(v, [2, 3, 4, 5])
    assert np.allclose(e, 0.5 * np.hypot(0.1, 0.1))
