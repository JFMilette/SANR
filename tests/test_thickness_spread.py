"""model.stack THICKNESS SPREAD: the incoherent average over a lateral
spread of the film thickness (plans/thickness_spread_plan_v1.md, sec. 7),
its fit parameter and its row in the Simulation tab.

Run from the repo root: python -m pytest tests (test 4: -m slow)
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import model.stack as S                                 # noqa: E402
from fit.de.de_fit import run_de                        # noqa: E402
from fit.joint import JointProblem                      # noqa: E402
from fit.problem import FitProblem, parameter_names     # noqa: E402

Q = np.linspace(0.004, 0.2, 400)
PAIRS = [(np.array([0.98, 0, 0]), None), (np.array([-0.98, 0, 0]), None)]
FULL = [(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])),
        (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])),
        (np.array([-1.0, 0, 0]), np.array([-1.0, 0, 0]))]
# the last commit before the Q shift and the thickness spread
BASE_COMMIT = 'a66284c'
PRIVATE = HERE / 'data' / 'private' / 'thickness_spread'


def film(scheme='rms', spread=0.0, nodes='auto', front=None, d=(120, 180)):
    """vacuum (or a magnetic fronting) / two films (D = 300 A) / Si."""
    f = S.Layer('front', 0, 0, 0, 0, 0) if front is None else front
    st = S.Stack([f,
                  S.Layer('a', d[0], 4e-6, 0, 1.2e-6, 0.1, 4.0,
                          roughness_sublayer=8),
                  S.Layer('b', d[1], 2e-6, 0, 0, 0, 6.0,
                          roughness_sublayer=8),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0,
                          roughness_sublayer=8)],
                 roughness_scheme=scheme)
    st.thickness_spread, st.thickness_spread_nodes = spread, nodes
    st.build_sublayers()
    return st


def with_resolution(st, scheme='quadrature'):
    st.resolution.update(enabled=True, mode='tof', tof_dlambda=0.02,
                         tof_angles=[dict(theta=0.01, dtheta=0.0002,
                                          qmax=None)],
                         scheme=scheme, licorne_fun=3)
    return st


def explicit(st, Q, pairs, qmax, **kw):
    """SUM_k w_k R of Stack.from_dict copies at the scaled thicknesses."""
    out = 0.0
    for w, f in st.spread_nodes(qmax):
        c = S.Stack.from_dict(st.to_dict())
        c.thickness_spread = 0.0
        for l in c.layers[1:-1]:
            l.thickness *= f
        c.build_sublayers()
        out = out + w * c.reflectivities(Q, pairs, **kw)
    return out


def base_stack_module():
    """model/stack.py as at BASE_COMMIT, or skip."""
    try:
        src = subprocess.run(['git', 'show', '%s:model/stack.py' % BASE_COMMIT],
                             cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('needs git and commit %s' % BASE_COMMIT)
    path = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / 'stack_base.py'
    path.write_text(src)
    spec = importlib.util.spec_from_file_location('stack_base', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 1 ----------------------------------------------------------------------
def test_zero_spread_is_the_code_before_it():
    """sigma_t = 0: bit for bit the reflectivities of the code before the
    spread, both roughness schemes, a magnetic fronting, resolution on and
    off, an explicit sigma."""
    base = base_stack_module()
    mag = S.Layer('Fe', 0, 8e-6, 0, 2e-6, 0.0)
    cases = [(film('rms'), PAIRS, {}), (film('licorne'), PAIRS, {}),
             (with_resolution(film('rms')), PAIRS, {}),
             (with_resolution(film('licorne'), 'licorne'), PAIRS, {}),
             (film('rms', front=mag), FULL, {}),
             (with_resolution(film('rms')), PAIRS,
              {'sigma': 0.03 * Q})]
    for st, pairs, kw in cases:
        old = base.Stack.from_dict(st.to_dict())
        old.build_sublayers()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert np.array_equal(st.reflectivities(Q, pairs, **kw),
                                  old.reflectivities(Q, pairs, **kw))


# 2, 3 -------------------------------------------------------------------
@pytest.mark.parametrize('scheme', ['rms', 'licorne'])
@pytest.mark.parametrize('res', [None, 'quadrature', 'licorne'])
def test_matches_an_explicit_average(scheme, res):
    """sigma_t = 7 %, 7 nodes: the scaled copies' average, resolution off,
    quadrature or Licorne's (the resolved average is the average of the
    resolved nodes: the average commutes with the resolution)."""
    st = film(scheme, 0.07, 7)
    if res:
        with_resolution(st, res)
    sig = st.resolution_sigma(Q)
    qmax = float(Q.max() + 3 * sig.max())
    want = explicit(st, Q, PAIRS, qmax)
    assert np.allclose(st.reflectivities(Q, PAIRS), want, rtol=1e-12,
                       atol=0)
    assert [l.thickness for l in st.layers[1:-1]] == [120, 180]


def test_magnetic_fronting_and_explicit_sigma():
    mag = S.Layer('Fe', 0, 8e-6, 0, 2e-6, 0.0)
    st = film('rms', 0.05, 9, front=mag)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert np.allclose(st.reflectivities(Q, FULL),
                           explicit(st, Q, FULL, Q.max()), rtol=1e-12)
    st = with_resolution(film('rms', 0.05, 9))
    s = 0.02 * Q
    assert np.allclose(st.reflectivities(Q, PAIRS, sigma=s),
                       explicit(st, Q, PAIRS, float(Q.max() + 3 * s.max()),
                                sigma=s), rtol=1e-12)


def test_auto_count_by_convergence():
    """'auto' (plan v2 A1): from 1.5 phi + 2 (at least 5), up by 4 until n
    and n + 4 nodes agree to SPREAD_TOL on the probe, odd, cached; an int
    is used as it is; even or < 3 is refused."""
    st = film('rms', 0.072)
    n = st.spread_count(0.2)
    assert n % 2 == 1 and S.SPREAD_NODES_MIN <= n <= S.SPREAD_NMAX
    q = np.linspace(0.2 / 200, 0.2, 200)
    d = np.abs(st._spread_probe(q, 0.072, n)
               / st._spread_probe(q, 0.072, n + 4) - 1)
    assert d.max() < S.SPREAD_TOL
    assert len(st._spread_counts) == 1
    assert st.spread_count(0.2) == n                   # from the cache
    assert st.spread_count(0.2, 0.001) == 5
    w, f = zip(*st.spread_nodes(0.2))
    assert sum(w) == pytest.approx(1.0, abs=1e-14) and len(w) == n
    st.thickness_spread_nodes = 7
    assert st.spread_count(5.0) == 7
    for bad in (4, 1):
        st.thickness_spread_nodes = bad
        with pytest.raises(ValueError, match='odd'):
            st.spread_count(0.2)


def test_gaussian_positivity():
    """(plan v2 test 7) sigma_t = 0.25 Gaussian is refused (use the
    log-normal), also by a fit whose upper bound reaches it; at 0.15 with
    41 nodes only negligible-weight nodes are left out, the weights
    renormalised."""
    with pytest.raises(ValueError, match='use the log-normal'):
        S.spread_factors(0.25, 'gaussian', 9)
    f, w = S.spread_factors(0.15, 'gaussian', 41)
    x, h = np.polynomial.hermite.hermgauss(41)
    lost = (h / np.sqrt(np.pi))[1 + np.sqrt(2) * 0.15 * x <= S.SPREAD_F_MIN]
    assert len(f) < 41 and f.min() > S.SPREAD_F_MIN
    assert 0 < lost.sum() < 1e-9 and w.sum() == pytest.approx(1, abs=1e-14)
    st = spread_free(film('rms', 0.05), hi=0.25)
    with pytest.raises(ValueError, match='use the log-normal'):
        FitProblem(st, data_of(film()))
    st.thickness_spread_dist = 'lognormal'          # fine for it
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        FitProblem(st, data_of(film()))
    with pytest.raises(ValueError, match='unknown'):
        S.spread_factors(0.1, 'uniform', 9)


# 4 ----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize('spread', [0.03, 0.07, 0.10, 0.15])
def test_auto_nodes_converge(spread):
    """'auto' within 5e-3 relative of a 241-point brute force over
    +-6 sigma_t (D = 300 A, Q up to 0.2)."""
    st = film('rms', spread)
    f = np.linspace(1 - 6 * spread, 1 + 6 * spread, 241)
    w = np.exp(-0.5 * ((f - 1) / spread) ** 2)
    w = w / np.trapezoid(w, f) * np.gradient(f)
    w[[0, -1]] /= 2
    w /= w.sum()
    ref = 0.0
    for wk, fk in zip(w, f):
        c = S.Stack.from_dict(st.to_dict())
        c.thickness_spread = 0.0
        for l in c.layers[1:-1]:
            l.thickness *= fk
        c.build_sublayers()
        ref = ref + wk * c.reflectivities(Q, PAIRS)
    assert np.max(np.abs(st.reflectivities(Q, PAIRS) / ref - 1)) < 5e-3


# 5 ----------------------------------------------------------------------
def test_critical_edge_preserved():
    """Below 0.9 Q_c of the substrate (the largest edge), R does not move:
    total reflection whatever the thicknesses."""
    st0, st = film(), film('rms', 0.10)
    qc = max(st0.critical_edges())
    q = np.linspace(0.002, 0.9 * qc, 60)
    assert np.max(np.abs(st.reflectivities(q, PAIRS)
                         - st0.reflectivities(q, PAIRS))) < 1e-6


# 6 ----------------------------------------------------------------------
def test_serialisation():
    st = film('rms', 0.064, 9)
    d = st.to_dict()
    back = S.Stack.from_dict(json.loads(json.dumps(d)))
    assert (back.thickness_spread, back.thickness_spread_nodes) == (0.064, 9)
    for k in ('thickness_spread', 'thickness_spread_nodes'):
        d.pop(k)
    old = S.Stack.from_dict(d)
    assert (old.thickness_spread, old.thickness_spread_nodes) == (0.0, 'auto')
    assert S.default_fit('thickness_spread')['max'] == 0.15
    assert 'thickness_spread' not in S.Stack.LICORNE_EXACT


# 7 ----------------------------------------------------------------------
def data_of(st, noise=0.0, seed=0):
    R = st.reflectivities(Q, PAIRS)
    rng = np.random.default_rng(seed)
    out = []
    for j, (P0, P) in enumerate(PAIRS):
        r = R[:, j] * (1 + noise * rng.normal(size=Q.size))
        out.append(dict(Q=Q, R=r, dR=max(noise, 0.01) * R[:, j], P0=P0,
                        P=P, name=('R+', 'R-')[j]))
    return out


def spread_free(st, hi=0.15):
    st.fit['thickness_spread'] = dict(vary=True, min=0.0, max=hi)
    return st


def test_fit_registration():
    st = spread_free(film('rms', 0.05))
    params = st.free_parameters()
    assert [p[:2] for p in params] == [(None, 'thickness_spread')]
    assert parameter_names(st, params) == ['stack.thickness_spread']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parts = [(n, FitProblem(st, data_of(film('rms', s))))
                 for n, s in (('A', 0.04), ('B', 0.06))]
    jp = JointProblem(parts, {'B': {'stack.thickness_spread': 'own'}})
    assert jp.names == ['stack.thickness_spread',
                        'stack.thickness_spread [B]']


# 8 ----------------------------------------------------------------------
def test_recovery():
    """sigma_t = 5 % in 3 % noise: a fit of sigma_t alone finds it to
    +-0.005."""
    data = data_of(film('rms', 0.05, 11), noise=0.03, seed=1)
    st = spread_free(film('rms', 0.0))         # 'auto', frozen by the fit
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = FitProblem(st, data)
    x, cost, *_ = run_de(p, seed=0, maxiter=30, popsize=10)
    assert x[0] == pytest.approx(0.05, abs=0.005)


# 9 ----------------------------------------------------------------------
def test_frozen_nodes():
    """'auto' with sigma_t free: the fit's copy gets one int (from the
    upper bound), the caller's stack keeps 'auto', the cost is that of
    the int at every sigma_t (it cannot jump), and the message says so;
    a joint fit gives every dataset the largest count."""
    st = spread_free(film('rms', 0.03), hi=0.15)
    with pytest.warns(UserWarning,
                      match=r'thickness spread \(whole stack, gaussian\): '
                            r'\d+ nodes'):
        p = FitProblem(st, data_of(film('rms', 0.06)))
    n = p.stack.thickness_spread_nodes
    assert isinstance(n, int) and st.thickness_spread_nodes == 'auto'
    assert n == st.spread_count(p.spread_qmax, 0.15) == p.spread_nodes
    fixed = spread_free(film('rms', 0.03, n), hi=0.15)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = FitProblem(fixed, data_of(film('rms', 0.06)))
    u = np.linspace(0, 1, 41)
    c = np.array([p(np.array([v])) for v in u])
    assert np.array_equal(c, [q(np.array([v])) for v in u])
    # smooth: no step much above its neighbours
    d2 = np.abs(np.diff(c, 2))
    assert d2.max() < 20 * np.median(d2) + 1e-12
    # a joint fit: the larger Q range's count for both
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = FitProblem(st, data_of(film('rms', 0.06))[:1])
        short = [dict(d, Q=d['Q'][:100], R=d['R'][:100], dR=d['dR'][:100])
                 for d in data_of(film('rms', 0.06))]
        b = FitProblem(st, short)
    assert b.spread_nodes < a.spread_nodes
    jp = JointProblem([('A', a), ('B', b)])
    assert [x.stack.thickness_spread_nodes for x in jp.parts] == \
        [a.spread_nodes] * 2
    # no spread at all: nothing frozen, nothing said
    plain = film()
    plain.layers[1].fit['thickness'] = dict(vary=True, min=100, max=140)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter('always')
        none = FitProblem(plain, data_of(film()))
    assert none.spread_nodes is None
    assert none.stack.thickness_spread_nodes == 'auto'
    assert not any('thickness spread' in str(w.message) for w in said)


# 10 ---------------------------------------------------------------------
def _ref_key(d, *names):
    for n in names:
        if n in d:
            return np.asarray(d[n], dtype=float)
    pytest.skip('reference file has none of %s' % (names,))


def test_acceptance_S1_4K_4p8T():
    """The prototype's fit (plan sec. 1): its session, 7 nodes, sigma_Q =
    the data's dQ, reproduces the reference curves to rtol 1e-9 and
    chi^2 = 497.97 +- 0.01 on the 4.8 T data."""
    sess = PRIVATE / 'S1_4K_4p8T_thickness_spread.json'
    ref = PRIVATE / 'thickness_spread_reference_4p8T.json'
    files = sorted(PRIVATE.glob('REF_M_*_Off_Off.dat')) + \
        sorted(PRIVATE.glob('REF_M_*_On_Off.dat'))
    if not (sess.exists() and ref.exists() and len(files) == 2):
        pytest.skip('needs the private files of %s (plan sec. 0.1)' % PRIVATE)
    d = json.loads(sess.read_text())
    st = S.Stack.from_dict(d)
    st.thickness_spread_nodes = 7
    st.build_sublayers()
    r = json.loads(ref.read_text())
    q, dq = _ref_key(r, 'Q', 'q'), _ref_key(r, 'dQ', 'dq', 'sigma_Q')
    Rp = _ref_key(r, 'R+', 'Rp', 'R_plus', 'Rplus')
    Rm = _ref_key(r, 'R-', 'Rm', 'R_minus', 'Rminus')
    R = st.reflectivities(q, PAIRS, sigma=dq)
    assert np.allclose(R[:, 0], Rp, rtol=1e-9, atol=0)
    assert np.allclose(R[:, 1], Rm, rtol=1e-9, atol=0)
    data = []
    for f, (P0, P), name in zip(files, PAIRS, ('R+', 'R-')):
        a = np.loadtxt(f, comments='#')
        data.append(dict(Q=a[:, 0], R=a[:, 1], dR=a[:, 2], dQ=a[:, 3],
                         P0=P0, P=P, name=name))
    st.resolution.update(enabled=True, mode='data')
    for e in list(st.fit.values()) + [e for l in st.layers
                                       for e in l.fit.values()]:
        e['vary'] = False
    st.fit['scale'] = dict(st.fit.get('scale', {}), vary=True,
                           min=0.0, max=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = FitProblem(st, data)
    chi2 = float(np.sum(p.residuals(np.array([st.scale])) ** 2))
    assert chi2 == pytest.approx(497.97, abs=0.01)


# GUI --------------------------------------------------------------------
def test_simulation_tab_row():
    """Below the Q shift, in %: value, bounds and Fit reach the stack as
    a fraction, the nodes box its setting, a fit result comes back in %,
    the session keeps both; Licorne-exact on keeps the spread."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    from tabs import SimulationTab
    sim = SimulationTab()
    assert list(sim.stack_rows)[-1] == 'thickness_spread'
    val, bmin, bmax, cb = sim.stack_rows['thickness_spread']
    assert (bmin.value(), bmax.value()) == (0.0, 15.0)         # %
    val.setValue(7.23)
    bmax.setValue(12.0)
    cb.setChecked(True)
    sim.spread_nodes.setCurrentIndex(sim.spread_nodes.findData(9))
    sim.recompute()
    assert sim.stack.thickness_spread == pytest.approx(0.0723)
    assert sim.stack.fit['thickness_spread']['max'] == pytest.approx(0.12)
    assert sim.stack.thickness_spread_nodes == 9
    sim.lic_exact.setChecked(True)
    sim.recompute()
    assert sim.lic_exact.isChecked()
    assert sim.stack.thickness_spread == pytest.approx(0.0723)
    params = [p for p in sim.stack.free_parameters()
              if p[1] == 'thickness_spread']
    sim.set_parameters(params, [0.051])
    assert val.value() == pytest.approx(5.1)
    state = sim.session_state()
    fresh = SimulationTab()                # a session opened elsewhere
    fresh.restore_state(state)
    assert fresh.stack.thickness_spread == pytest.approx(0.051)
    assert fresh.stack_rows['thickness_spread'][0].value() == \
        pytest.approx(5.1)
    assert fresh.spread_nodes.currentData() == 9
    # a count not among the choices (a session's 13) is kept
    state['thickness_spread_nodes'] = 13
    fresh.restore_state(state)
    assert fresh.spread_nodes.currentData() == 13
    fresh.recompute()
    assert fresh.stack.thickness_spread_nodes == 13


# ========================================================================
# plan v2: target, distributions, convergence-checked nodes
# ========================================================================
V1_COMMIT = '49eead7'           # thickness spread v1, before plan v2


def stack_module_at(commit):
    try:
        src = subprocess.run(['git', 'show', '%s:model/stack.py' % commit],
                             cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('needs git and commit %s' % commit)
    path = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / (
        'stack_%s.py' % commit)
    path.write_text(src)
    spec = importlib.util.spec_from_file_location('stack_' + commit, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def explicit_layers(st, Q, pairs, qmax, which):
    """SUM_k w_k R of from_dict copies with the layers `which` scaled."""
    out = 0.0
    for w, f in st.spread_nodes(qmax):
        c = S.Stack.from_dict(st.to_dict())
        c.thickness_spread = 0.0
        for i in which:
            c.layers[i].thickness *= f
        c.build_sublayers()
        out = out + w * c.reflectivities(Q, pairs)
    return out


# v2 1 -------------------------------------------------------------------
def test_v2_defaults_are_v1():
    """Target 'stack' and the Gaussian: bit for bit v1's results (an
    explicit node count: 'auto' itself changed, A1)."""
    v1 = stack_module_at(V1_COMMIT)
    mag = S.Layer('Fe', 0, 8e-6, 0, 2e-6, 0.0)
    cases = [(film('rms', 0.07, 7), PAIRS), (film('licorne', 0.07, 7), PAIRS),
             (with_resolution(film('rms', 0.07, 7)), PAIRS),
             (with_resolution(film('licorne', 0.07, 7), 'licorne'), PAIRS),
             (film('rms', 0.05, 9, front=mag), FULL)]
    for st, pairs in cases:
        old = v1.Stack.from_dict(st.to_dict())
        old.build_sublayers()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert np.array_equal(st.reflectivities(Q, pairs),
                                  old.reflectivities(Q, pairs))


# v2 2 -------------------------------------------------------------------
@pytest.mark.parametrize('scheme', ['rms', 'licorne'])
@pytest.mark.parametrize('dist', ['gaussian', 'lognormal'])
def test_layer_target_matches_an_explicit_average(scheme, dist):
    st = film(scheme, 0.1, 9)
    st.thickness_spread_dist = dist
    for i in (1, 2):
        st.set_spread_target(i)
        assert np.allclose(st.reflectivities(Q, PAIRS),
                           explicit_layers(st, Q, PAIRS, Q.max(), [i]),
                           rtol=1e-12, atol=0)
        assert [l.thickness for l in st.layers[1:-1]] == [120, 180]


# v2 3 -------------------------------------------------------------------
@pytest.mark.parametrize('scheme', ['rms', 'licorne'])
def test_only_the_target_changes(scheme):
    """Target = the lower film: every node's slabs above its upper
    interface window are the nominal ones; below, they differ (but at the
    middle node, f = 1: the nominal stack)."""
    st = film(scheme, 0.1, 9)
    st.set_spread_target(2)
    nominal = st._slab_arrays[1]
    top = st.windows()[0][0]
    bottoms = top + np.cumsum(nominal[3])
    above = int(np.sum(bottoms <= st.windows()[1][0] + 1e-9))
    assert above > 0
    f = [f for _, f in st.spread_nodes(Q.max())]
    for fk, (w, (_, arrays)) in zip(f, st._spread_sequences(Q.max())):
        for a, b in zip(arrays, nominal):
            assert np.array_equal(a[:above], b[:above])
        same = all(np.array_equal(a, b) for a, b in zip(arrays, nominal))
        assert same == (fk == 1.0)


# v2 4 -------------------------------------------------------------------
def test_one_layer_stack_targets_agree():
    st = S.Stack([S.Layer('vacuum', 0, 0, 0, 0, 0),
                  S.Layer('film', 150, 4e-6, 0, 1e-6, 0, 4.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0)])
    st.thickness_spread, st.thickness_spread_nodes = 0.08, 11
    st.build_sublayers()
    whole = st.reflectivities(Q, PAIRS)
    st.set_spread_target(1)
    assert np.array_equal(st.reflectivities(Q, PAIRS), whole)


# v2 5 -------------------------------------------------------------------
@pytest.mark.parametrize('sigma', [0.02, 0.07, 0.3])
def test_lognormal_moments(sigma):
    """Mean 1 and standard deviation sigma_t, so the fitted thickness is
    the mean thickness and sigma_t means what it means for the Gaussian."""
    f, w = S.spread_factors(sigma, 'lognormal', 41)
    assert abs(np.sum(w * f) - 1) < 1e-10
    assert abs(np.sum(w * (f - 1) ** 2) - sigma ** 2) < 1e-10
    assert f.min() > 0


# v2 6 -------------------------------------------------------------------
def test_lognormal_tends_to_the_gaussian():
    """Same mean and variance; they differ by the log-normal's third
    central moment, skewness 3 sigma_t times sigma_t^3 = 3 sigma_t^4: a
    change ~ sigma_t^4 (Q D)^3 / 2 of R, below 1e-5 at 0.5 % here (D =
    300 A, Q to 0.2; the plan's 1e-6 is tighter than that physics), and
    16 times smaller at half the spread."""
    d = []
    for sigma in (0.005, 0.0025):
        g, ln = film('rms', sigma, 9), film('rms', sigma, 9)
        ln.thickness_spread_dist = 'lognormal'
        d.append(np.max(np.abs(ln.reflectivities(Q, PAIRS)
                               / g.reflectivities(Q, PAIRS) - 1)))
    assert d[0] < 1e-5
    assert 12 < d[0] / d[1] < 20


# v2 8 -------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize('target', ['stack', 1])
@pytest.mark.parametrize('dist,sigma', [('gaussian', 0.05),
                                        ('gaussian', 0.15),
                                        ('lognormal', 0.05),
                                        ('lognormal', 0.3)])
def test_v2_auto_converges(target, dist, sigma):
    """A1's count within 2e-3 relative of a 241-point brute force, for the
    whole stack and for a 40 A single layer.  Except a 30 % log-normal on
    the whole 220 A stack (phi 13): Gauss-Hermite needs over 100 nodes
    there (2.8e-2 at 41, 8.5e-3 at 81 against a 4001-point reference),
    so A1 stops at SPREAD_NMAX and says so, as the plan has it."""
    st = film('rms', sigma, d=(40, 180))
    st.thickness_spread_dist = dist
    st.set_spread_target(target)
    which = st._spread_layers()
    if (target, dist, sigma) == ('stack', 'lognormal', 0.3):
        with pytest.warns(UserWarning, match='do not reach'):
            assert st.spread_count(Q.max()) >= S.SPREAD_NMAX - 4
        return
    if dist == 'gaussian':
        f = np.linspace(max(1 - 6 * sigma, S.SPREAD_F_MIN), 1 + 6 * sigma,
                        241)
        w = np.exp(-0.5 * ((f - 1) / sigma) ** 2)
    else:
        s = np.sqrt(np.log1p(sigma ** 2))
        z = np.linspace(-6, 6, 241)
        f, w = np.exp(-s ** 2 / 2 + s * z), np.exp(-0.5 * z ** 2)
    w = w / w.sum()
    ref = 0.0
    for wk, fk in zip(w, f):
        c = S.Stack.from_dict(st.to_dict())
        c.thickness_spread = 0.0
        for i in which:
            c.layers[i].thickness *= fk
        c.build_sublayers()
        ref = ref + wk * c.reflectivities(Q, PAIRS)
    assert np.max(np.abs(st.reflectivities(Q, PAIRS) / ref - 1)) < 2e-3


# v2 9 -------------------------------------------------------------------
def test_target_follows_its_layer():
    """Inserting a layer above the target or moving it: the target
    follows; deleting it: back to the whole stack, no spread, not fitted,
    and the status bar says so.  Kept through to_dict / from_dict."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    from tabs import SimulationTab
    win = QtWidgets.QMainWindow()
    sim = SimulationTab()
    win.setCentralWidget(sim)
    L = sim.stack.layers
    while len(L) < 5:                       # two interior layers at least
        sim.list.setCurrentRow(1)
        sim.add_layer()
    target = L[2]
    sim.spread_target.setCurrentIndex(sim.spread_target.findData(2))
    sim.spread.setValue(8.0)
    sim.stack_rows['thickness_spread'][3].setChecked(True)
    sim.recompute()
    assert sim.stack.thickness_spread_target == 2
    d = S.Stack.from_dict(sim.stack.to_dict())
    assert d.thickness_spread_target == 2
    sim.list.setCurrentRow(1)
    sim.add_layer()                          # inserted at 2, above it
    assert L.index(target) == 3 == sim.stack.thickness_spread_target
    assert sim.spread_target.currentData() == 3
    sim.list.setCurrentRow(3)
    sim.move_layer(-1)
    assert L.index(target) == 2 == sim.stack.thickness_spread_target
    sim.list.setCurrentRow(2)
    sim.remove_layer()
    assert target not in L
    st = sim.stack
    assert st.thickness_spread_target == 'stack'
    assert st.thickness_spread == 0 and not st.fit['thickness_spread']['vary']
    assert sim.spread.value() == 0 and sim.spread_target.currentData() == \
        'stack'
    assert 'deleted' in win.statusBar().currentMessage()


# v2 10 ------------------------------------------------------------------
def test_target_validation():
    st = film()
    for bad in (0, len(st.layers) - 1, 'top', 1.0, True, -1):
        with pytest.raises(ValueError, match='target'):
            st.set_spread_target(bad)
    st.set_spread_target(np.int64(2))
    assert st.thickness_spread_target == 2 and \
        type(st.thickness_spread_target) is int
    d = st.to_dict()
    d['thickness_spread_target'] = 9
    with pytest.raises(ValueError, match='target'):
        S.Stack.from_dict(d)
    d['thickness_spread_target'], d['thickness_spread_dist'] = 1, 'box'
    with pytest.raises(ValueError, match='distribution'):
        S.Stack.from_dict(d)
    for k in ('thickness_spread_target', 'thickness_spread_dist'):
        d.pop(k)
    old = S.Stack.from_dict(d)
    assert (old.thickness_spread_target, old.thickness_spread_dist) == \
        ('stack', 'gaussian')


# v2 11 ------------------------------------------------------------------
def test_v2_frozen_nodes_with_a_layer_target():
    st = spread_free(film('rms', 0.05, d=(40, 180)), hi=0.4)
    st.thickness_spread_dist = 'lognormal'
    st.set_spread_target(1)
    with pytest.warns(UserWarning, match=r'thickness spread \(a, lognormal\)'):
        p = FitProblem(st, data_of(film('rms', 0.1, d=(40, 180))))
    n = p.stack.thickness_spread_nodes
    assert isinstance(n, int) and st.thickness_spread_nodes == 'auto'
    fixed = S.Stack.from_dict(st.to_dict())
    fixed.thickness_spread_nodes = n
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = FitProblem(fixed, data_of(film('rms', 0.1, d=(40, 180))))
    u = np.linspace(0, 1, 41)
    c = np.array([p(np.array([v])) for v in u])
    assert np.array_equal(c, [q(np.array([v])) for v in u])
    d2 = np.abs(np.diff(c, 2))
    assert d2.max() < 20 * np.median(d2) + 1e-12


# v2 12 ------------------------------------------------------------------
def test_v2_recovery_of_a_one_layer_lognormal():
    """A 40 A top layer with a log-normal spread of 25 % (3 % noise): the
    fit with that target finds 0.25 +- 0.04; with the whole stack as the
    target, its best chi^2 is clearly worse (by > 25 over the points)."""
    q = Q[::2]
    truth = film('rms', 0.25, 21, d=(40, 180))
    truth.thickness_spread_dist = 'lognormal'
    truth.set_spread_target(1)
    rng = np.random.default_rng(3)
    R = truth.reflectivities(q, PAIRS)
    data = [dict(Q=q, R=R[:, j] * (1 + 0.03 * rng.normal(size=q.size)),
                 dR=0.03 * R[:, j], P0=P0, P=P, name=('R+', 'R-')[j])
            for j, (P0, P) in enumerate(PAIRS)]
    best = {}
    for target in (1, 'stack'):
        st = spread_free(film('rms', 0.1, 21, d=(40, 180)), hi=0.5)
        st.thickness_spread_dist = 'lognormal'
        st.set_spread_target(target)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = FitProblem(st, data)
        best[target] = run_de(p, seed=0, maxiter=15, popsize=8)
    assert best[1][0][0] == pytest.approx(0.25, abs=0.04)
    n = sum(len(d['Q']) for d in data)
    assert (best['stack'][1] - best[1][1]) * n > 25


# v2 GUI and reporting ---------------------------------------------------
def test_v2_simulation_tab_and_report():
    """Spread row: target (whole stack, then the layers), distribution,
    the Gaussian's error as a red note (no dialog), the bounds clipped to
    the Gaussian's 15 % and widened for the log-normal, the session keeps
    both; the fit tab reports target, distribution and what the spread
    means (Stack.spread_summary)."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    from tabs import SimulationTab
    sim = SimulationTab()
    c = sim.spread_target
    names = [c.itemText(k) for k in range(c.count())]
    assert names == ['Whole stack'] + [l.name for l in
                                       sim.stack.layers[1:-1]]
    _, bmin, bmax, cb = sim.stack_rows['thickness_spread']
    sim.spread_dist.setCurrentIndex(sim.spread_dist.findData('lognormal'))
    assert bmax.value() == 50.0                       # widened
    sim.spread.setValue(25.0)
    sim.recompute()
    assert sim.spread_error.isHidden()
    sim.spread_dist.setCurrentIndex(sim.spread_dist.findData('gaussian'))
    assert bmax.value() == 15.0                       # clipped
    sim.recompute()
    assert not sim.spread_error.isHidden()
    assert 'log-normal' in sim.spread_error.text()
    sim.spread_dist.setCurrentIndex(sim.spread_dist.findData('lognormal'))
    c.setCurrentIndex(1)
    sim.recompute()
    assert sim.stack.thickness_spread_target == 1
    state = sim.session_state()
    fresh = SimulationTab()
    fresh.restore_state(state)
    assert (fresh.stack.thickness_spread_target,
            fresh.stack.thickness_spread_dist) == (1, 'lognormal')
    assert fresh.spread_target.currentData() == 1
    assert fresh.spread_dist.currentData() == 'lognormal'
    # what the spread means
    st = film('rms', 0.2, d=(40, 180))
    st.set_spread_target(1)
    assert st.spread_summary() == [('RMS thickness variation',
                                    pytest.approx(8.0), 'Å')]
    st.thickness_spread_dist = 'lognormal'
    rows = dict((k, v) for k, v, _ in st.spread_summary())
    s2 = np.log1p(0.04)
    assert rows['median factor'] == pytest.approx(np.exp(-s2 / 2))
    assert rows['mode thickness'] == pytest.approx(40 * np.exp(-1.5 * s2))
    # in the fit tab's parameter table
    from fit.de.de_window import DEWindow
    st = spread_free(film('rms', 0.1, 9, d=(40, 180)), hi=0.4)
    st.thickness_spread_dist = 'lognormal'
    st.set_spread_target(1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = FitProblem(st, data_of(film('rms', 0.1, d=(40, 180))))
    win = DEWindow()
    view = dict(name='d', data={}, vectors={}, grids={None: Q})
    win.start(p, [view], 'd', 'χ²')
    rows = [[win.params.topLevelItem(k).text(c) for c in range(3)]
            for k in range(win.params.topLevelItemCount())]
    assert rows[0][:2] == ['a', 'Thickness spread (log-normal)']
    assert [r[1].strip() for r in rows[1:]] == [
        'RMS thickness variation', 'median factor', 'median thickness',
        'mode factor', 'mode thickness']
    win.close()
