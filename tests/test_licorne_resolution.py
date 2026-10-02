"""Licorne's resolution convolution (model.licorne_resolution, Stack
resolution['scheme'] = 'licorne') against the oracle's resolut
(tests/licorne_reference.py), and the size of its difference from a
converged Gaussian average on fixture 1.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import model.stack as S                                         # noqa: E402
import licorne_reference as lic                                 # noqa: E402
from licorne_inputs import oracle_inputs                        # noqa: E402
from model.licorne_io import load_licorne_model                 # noqa: E402
from model.licorne_resolution import kernel                     # noqa: E402

FIX1 = HERE / 'data' / 'licorne' / 'fixture1'
X = np.array([1.0, 0.0, 0.0])
# fixture 1's q.dat grid and the TOF resolution of IPTS 27232 S1 best_103
Q1 = np.linspace(0.0106348, 0.24841, 204)
BEST_103 = {'enabled': True, 'mode': 'tof', 'tof_dlambda': 0.005,
            'tof_angles': [{'theta': 0.006, 'dtheta': 3e-4, 'qmax': 0.04},
                           {'theta': 0.010, 'dtheta': 5e-4, 'qmax': 0.12},
                           {'theta': 0.017, 'dtheta': 5e-4, 'qmax': None}]}


def grids():
    """(name, q, sigma): uniform, geometric and jittered grids with
    sigma / spacing from 0.3 to 8, and a TOF-like jump in sigma."""
    rng = np.random.default_rng(4)
    out = []
    for name, q in (('uniform', np.linspace(0.01, 0.25, 204)),
                    ('geometric', np.geomspace(0.005, 0.3, 250)),
                    ('jittered', np.sort(np.linspace(0.01, 0.25, 300)
                                         + rng.uniform(-3e-4, 3e-4, 300)))):
        ratio = np.interp(q, [q[0], q[-1]], [0.3, 8.0])
        out.append((name, q, ratio * np.gradient(q)))
    q = np.linspace(0.01, 0.2, 400)
    jump = np.where(q < 0.04, 0.4, np.where(q <= 0.12, 2.0, 6.0))
    out.append(('tof jump', q, jump * np.gradient(q)))
    return out


GRIDS = grids()


@pytest.mark.parametrize('mode', [1, 2, 3])
@pytest.mark.parametrize('grid', GRIDS, ids=[g[0] for g in GRIDS])
def test_kernel_matches_oracle(grid, mode):
    _, q, sigma = grid
    R = np.random.default_rng(mode).uniform(1e-6, 1.0, len(q))
    W = kernel(q, sigma, mode)
    ref = lic.resolut(R, q, sigma, mode)
    assert np.max(np.abs(W @ R - ref) / np.abs(ref)) <= 1e-13


@pytest.mark.parametrize('mode', [2, 3])
def test_kernel_row_sums(mode):
    """Licorne does not normalise its sum: on a regular grid the rows of W
    sum to about 1 (the Gaussian's mass within +-3 sigma) once sigma spans
    a few points, and to exactly 1 for a point left as is (sigma < half
    its spacing).  On an irregular grid, or just above that threshold, only
    a few unevenly spaced neighbours enter and a row is off by up to
    ~20 % (mode 2's one-sided rectangle rule even where sigma is well
    sampled)."""
    worst = {}
    for name, q, sigma in GRIDS:
        W = kernel(q, sigma, mode)
        rows = np.asarray(W.sum(1)).ravel()
        dqc = np.gradient(q)
        skipped = sigma[1:-1] < dqc[1:-1] / 2
        assert np.all(rows[1:-1][skipped] == 1.0)
        worst[name] = np.max(np.abs(rows - 1))
        if name != 'jittered':
            resolved = sigma >= 1.5 * dqc
            assert np.all((rows[resolved] > 0.98) & (rows[resolved] < 1.01))
    assert 0.1 < worst['jittered'] < 0.2
    with pytest.raises(ValueError):
        kernel([0.1, 0.1, 0.2, 0.3], [0.01] * 4, mode)
    with pytest.raises(ValueError):
        kernel([0.1, 0.2, 0.3, 0.4], [0.01, np.inf, 0.01, 0.01], mode)
    with pytest.raises(ValueError):
        kernel([0.1, 0.2, 0.3, 0.4], [0.01] * 4, 4)


def fixture1_stack(scheme='licorne'):
    st = load_licorne_model(FIX1 / 'parameters.m', FIX1 / 'profile.dat')
    st.licorne_renorm, st.licorne_outer = 'licorne', 'licorne'
    st.step_fallback = False
    st.resolution.update(BEST_103, scheme=scheme, licorne_fun=3)
    st.build_sublayers()
    return st


def test_fixture1_resolution_benchmark():
    """D3 on fixture 1's grid with the best_103 TOF resolution.  The
    'licorne' scheme equals the oracle end to end (P along Licorne x, which
    the import keeps as sample x); against a converged Gaussian average it
    is ~10 % off just above the critical edge, while the quadrature stays
    within 0.5 %."""
    st = fixture1_stack()
    st.background = 1e-6
    sigma = st.resolution_sigma(Q1)
    layers, sub = oracle_inputs(FIX1)
    for s in (1, -1):
        ref = lic.licorne_R(Q1, sigma, layers, sub, s * X, np.zeros(3),
                            norm=2, background=1e-6, res_mode=3)
        R = st.reflectivities(Q1, [(s * X, None)])[:, 0]
        assert np.max(np.abs(R / ref - 1)) <= 1e-12
    # converged reference: the ideal R on a fine grid, averaged over +-6
    # sigma with a fine quadrature
    st.background = 0.0
    pairs = [(X, None), (-X, None)]
    fine = np.linspace(5e-4, 0.3, 120001)
    Rf = st._trace(fine, pairs)
    x = np.linspace(-6, 6, 2401)
    w = np.exp(-x**2 / 2)
    conv = np.stack([np.trapezoid(np.interp(Q1[:, None] + x * sigma[:, None],
                                            fine, Rf[:, c]) * w, x, axis=1)
                     / np.trapezoid(w, x) for c in range(2)], -1)
    in_range = Q1 <= 0.22
    err = st.reflectivities(Q1, pairs)[in_range] / conv[in_range] - 1
    i = np.unravel_index(np.argmax(np.abs(err)), err.shape)
    print('\nlicorne scheme: %+.4f at Q = %.4f' % (err[i], Q1[in_range][i[0]]))
    assert 0.05 < np.max(np.abs(err)) < 0.15
    assert 0.015 < Q1[in_range][i[0]] < 0.02
    st.resolution['scheme'] = 'quadrature'
    err = st.reflectivities(Q1, pairs)[in_range] / conv[in_range] - 1
    print('quadrature:     %.4f' % np.max(np.abs(err)))
    assert np.max(np.abs(err)) < 0.005


def test_tof_band_edge():
    """Licorne's TOF template: QP2 = (Q >= Q1) & (Q <= Q2), so Q = Q2 uses
    angle 2 (and Q = Q1 too)."""
    st = S.Stack([S.Layer('vac', 0), S.Layer('Si', 0, 2.07e-6)])
    st.resolution.update(BEST_103)
    a = BEST_103['tof_angles']
    q = np.array([0.04, 0.12, np.nextafter(0.12, 1)])
    lam = 4 * np.pi * np.sin(np.array([a[1]['theta'], a[1]['theta'],
                                       a[2]['theta']])) / q
    th = np.array([a[1]['theta'], a[1]['theta'], a[2]['theta']])
    dth = np.array([a[1]['dtheta'], a[1]['dtheta'], a[2]['dtheta']])
    expect = q * np.hypot(dth / th, 0.005 / lam)
    assert np.allclose(st.resolution_sigma(q), expect, rtol=1e-14)


def test_licorne_scheme_unsorted_Q():
    st = fixture1_stack()
    pairs = [(X, None), (-X, X)]
    R = st.reflectivities(Q1, pairs)
    perm = np.random.default_rng(1).permutation(len(Q1))
    assert np.array_equal(st.reflectivities(Q1[perm], pairs), R[perm])
    with pytest.raises(ValueError, match='distinct'):
        st.reflectivities(np.r_[Q1, Q1[5]], pairs)
    # the kernel is cached per grid; a new grid gives a new kernel
    assert len(st._licorne_W) == 1
    st.reflectivities(Q1[:100], pairs)
    assert len(st._licorne_W) == 2
    # a magnetic fronting has no Licorne equivalent
    st.fronting.MSLD_rho = 1e-6
    with pytest.raises(ValueError, match='magnetic fronting'):
        st.reflectivities(Q1, pairs)


def test_round_trip_and_old_sessions():
    import json
    st = fixture1_stack()
    d = json.loads(json.dumps(st.to_dict()))
    assert S.Stack.from_dict(d).resolution == st.resolution
    for k in ('scheme', 'licorne_fun'):
        del d['resolution'][k]
    old = S.Stack.from_dict(d)
    assert old.resolution['scheme'] == 'quadrature'
    assert old.resolution['licorne_fun'] == 3


def test_fit_problem_grids():
    """The fit evaluates the 'licorne' scheme on each channel's own grid:
    one call for channels sharing their Q (Licorne's q.dat), separate
    calls, with a warning, when they differ."""
    import warnings
    from fit.problem import FitProblem
    st = fixture1_stack()
    st.layers[2].fit_entry('thickness')['vary'] = True
    one = np.ones_like(Q1)
    data = [dict(Q=Q1, R=one, dR=0.1 * one, P0=X, P=None, name='R+'),
            dict(Q=Q1, R=one, dR=0.1 * one, P0=-X, P=None, name='R-')]
    fp = FitProblem(st, data)
    x = [p[2] for p in fp.params]
    assert len(fp.groups) == 1
    ref = st.reflectivities(Q1, [(X, None), (-X, None)])
    for m, r in zip(fp.model(x), ref.T):
        assert np.array_equal(m, r)
    data[1].update(Q=Q1[::2], R=one[::2], dR=0.1 * one[::2])
    with pytest.warns(UserWarning, match='different Q grids'):
        fp = FitProblem(st, data)
    assert len(fp.groups) == 2
    assert np.array_equal(fp.model(x)[1],
                          st.reflectivities(Q1[::2], [(-X, None)])[:, 0])
    # the quadrature scheme keeps one call on the union of the points
    st.resolution['scheme'] = 'quadrature'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert len(FitProblem(st, data).groups) == 1
