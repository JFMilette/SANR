"""Stack.interface_order: additive or ordered ('non-crossing') interfaces in
the 'rms' roughness scheme (model.stack PROFILE), and the crossing
diagnostic Stack.interface_crossing.

The additive reference (tests/data/rms_additive_reference.npz) was frozen
from the code before interface_order existed by
scripts/make_rms_reference.py.

Run from the repo root: python -m pytest tests
"""
import copy
import json
import pathlib
import sys
import timeit

import numpy as np
import pytest
from scipy.special import ndtri

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import model.stack as S                                        # noqa: E402
from model.polarisation import CHANNEL_NAMES, channel_pair     # noqa: E402
import supermatrix as sm                                       # noqa: E402

DATA = HERE / 'data'
REF = np.load(DATA / 'rms_additive_reference.npz')
CASES = sorted({k.split('/')[0] for k in REF.files})
X = np.array([1.0, 0, 0])
PAIRS = [channel_pair(c, X) for c in CHANNEL_NAMES]
LAB = [channel_pair(c, X) for c in ('++', '+-', '-+', '--')]
Q = np.linspace(0.004, 0.12, 200)
MODES = ('vector', 'angle', 'step')
ORDERS = S.INTERFACE_ORDERS


def crossing(order, mode='vector', model='tanh', s=(2.0, 15.0), fe_img=-1e-8):
    """vacuum / FM 20 / spacer 30 / FM2 40 / Si: the 20 A FM has sigma
    s[0] above, s[1] below, so its interfaces cross when s[1] > s[0]."""
    L = [S.Layer('vacuum'),
         S.Layer('FM', 20, 4e-6, fe_img, 2e-6, 0.1, s[0], model, 6),
         S.Layer('spacer', 30, 2e-6, 0, 0, 0.0, s[1], model, 6),
         S.Layer('FM2', 40, 6e-6, 0, 1e-6, 0.35, 3.0, model, 6,
                 MSLD_phi=0.05),
         S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0, model, 6)]
    st = S.Stack(L, magnetic_smearing=mode, interface_order=order)
    st.build_sublayers()
    return st


def session(name):
    path = DATA / 'show_mistake.json' if name == 'show_mistake' else \
        HERE.parent / 'sessions' / 'session_Cr2Te3.json'
    with open(path) as f:
        return S.Stack.from_dict(json.load(f))


def outputs(st, z):
    """profile, slabs and R (resolution as the stack has it) of st."""
    prof = st.profile(z)
    slabs = [[s.thickness, complex(s.NSLD_real, s.NSLD_img), s.MSLD_rho,
              s.MSLD_theta, s.MSLD_phi] for s in st.build_sublayers()]
    return prof, np.array(slabs), st.reflectivities(Q, PAIRS, warn=False)


def occupancies(F):
    one = np.ones((1,) + F.shape[1:])
    return np.concatenate([one, F]) - np.concatenate([F, 0 * one])


# -- T1, T10: 'additive' is the code from before the setting ------------------
@pytest.mark.parametrize('case', CASES)
def test_additive_matches_reference(case, monkeypatch):
    # R frozen bit for bit with the numpy transfer matrix; the compiled one
    # equals it to rounding (tests/test_transfer.py)
    monkeypatch.setattr(S, 'USE_COMPILED', False)
    g = lambda k: REF['%s/%s' % (case, k)]                     # noqa: E731
    st = S.Stack.from_dict(json.loads(str(g('stack'))))
    assert st.interface_order == 'additive'                    # no key: old
    for a, k in zip(st.profile(g('z')), ('nsld', 'rho', 'theta', 'phi')):
        assert np.array_equal(a, g(k)), k
    slabs = st.build_sublayers()
    assert np.array_equal([s.thickness for s in slabs], g('slab_t'))
    assert np.array_equal([complex(s.NSLD_real, s.NSLD_img) for s in slabs],
                          g('slab_nsld'))
    for k in ('rho', 'theta', 'phi'):
        assert np.array_equal([getattr(s, 'MSLD_' + k) for s in slabs],
                              g('slab_' + k)), k
    assert np.array_equal(st.reflectivities(g('Q'), PAIRS, warn=False),
                          g('R_res_on'), equal_nan=True)
    st.resolution['enabled'] = False
    assert np.array_equal(st.reflectivities(g('Q'), PAIRS, warn=False),
                          g('R_res_off'), equal_nan=True)


@pytest.mark.parametrize('name', ['session_Cr2Te3', 'show_mistake'])
def test_old_sessions_load_additive(name):
    st = session(name)
    assert st.interface_order == 'additive'
    g = lambda k: REF['%s/%s' % (name, k)]                     # noqa: E731
    assert np.array_equal(st.profile(g('z'))[1], g('rho'))


@pytest.mark.parametrize('order', ORDERS)
def test_round_trip(order):
    st = crossing(order)
    d = st.to_dict()
    assert d['interface_order'] == order
    back = S.Stack.from_dict(json.loads(json.dumps(d)))
    assert back.interface_order == order and back.to_dict() == d


def test_new_stacks_are_ordered_and_bad_orders_refused():
    assert S.Stack([S.Layer('a'), S.Layer('b')]).interface_order == 'ordered'
    st = crossing('ordered')
    st.interface_order = 'sideways'
    with pytest.raises(ValueError, match='interface order'):
        st.build_sublayers()


# -- T2: nothing changes where nothing crosses ---------------------------------
@pytest.mark.parametrize('model', ['tanh', 'erf', 'none'])
@pytest.mark.parametrize('mode', MODES)
def test_ordered_is_additive_without_crossing(model, mode):
    a, b = (crossing(o, mode, model, s=(5.0, 5.0))
            for o in ('additive', 'ordered'))
    for st in (a, b):
        for l in st.layers[1:]:
            l.roughness_sigma = 5.0
    z = np.linspace(-40, 140, 3001)
    F = a._rms_steps(z, 'additive')
    assert np.all(F[:-1] >= F[1:])                 # the premise: no crossing
    pa, sa, Ra = outputs(a, z)
    pb, sb, Rb = outputs(b, z)
    assert all(np.array_equal(x, y) for x, y in zip(pa, pb))
    assert np.array_equal(sa, sb) and np.array_equal(Ra, Rb)


# -- T3: ordered and independent occupancies are a partition of unity -----------
@pytest.mark.parametrize('order', ['ordered', 'independent'])
def test_random_stacks_stay_in_range(order):
    rng = np.random.default_rng(12)
    for _ in range(200):
        n = rng.integers(2, 7)
        L = [S.Layer('front', 0, *rng.uniform(-1e-6, 8e-6, 1), 0, 0, 0)]
        for i in range(n + 1):
            L.append(S.Layer(
                'L%d' % i, rng.uniform(0.3, 80.0), rng.uniform(-1e-6, 8e-6),
                -rng.uniform(0, 1e-7), rng.uniform(0, 3e-6),
                rng.uniform(-1, 1), rng.uniform(0, 25.0),
                rng.choice(['tanh', 'erf', 'none']), 6,
                MSLD_phi=rng.uniform(-0.25, 0.25)))
        st = S.Stack(L, interface_order=order)
        Z = st._interfaces()
        z = np.linspace(-80, Z[-1] + 80, 4001)
        F = st._rms_steps(z)
        occ = occupancies(F)
        assert np.all(occ >= 0)
        assert np.max(np.abs(occ.sum(0) - 1)) <= 1e-15
        assert np.all(np.diff(F, axis=1) >= 0)          # steps stay steps
        for mode in MODES:
            st.magnetic_smearing = mode
            for name, v in st._linear_quantities().items():
                p = S._combine(F, v)
                tol = 1e-15 * max(np.max(np.abs(v)), 1e-300)
                assert p.min() >= v.min() - tol and p.max() <= v.max() + tol, \
                    name


# -- T4: one magnetic layer keeps its angle -------------------------------------
def test_single_magnetic_layer_keeps_its_angle():
    def one(order):
        st = crossing(order)
        st.layers[3].MSLD_rho = 0.0
        st.layers[1].MSLD_theta = 0.3
        _, rho, theta, _ = st.profile(np.linspace(-40, 140, 4001))
        return theta[st.angle_defined(rho)]
    assert np.max(np.abs(one('ordered') - 0.3)) <= 1e-12
    assert np.max(np.abs(one('independent') - 0.3)) <= 1e-12
    assert np.max(np.abs(one('additive') - 0.3)) > 0.49     # the half turn


# -- T5: the kink sits at the erf crossing ---------------------------------------
def test_kink_at_the_crossing():
    s0, s1, t = 2.0, 8.0, 10.0
    L = [S.Layer('vac'), S.Layer('A', t, 4e-6, 0, 0, 0, s0, 'erf'),
         S.Layer('Si', 0, 2e-6, 0, 0, 0, s1, 'erf')]
    st = S.Stack(L)
    zs = (s1 * 0.0 - s0 * t) / (s1 - s0)
    z = np.linspace(-12.0, 5.0, 1701) + 0.0037         # z* not on the grid
    F = st._rms_steps(z)
    f = st._rms_steps(z, 'additive')
    assert np.array_equal(F[0], np.where(z < zs, f[1], f[0]))
    d2 = np.abs(np.diff(st.profile(z)[0].real, 2))
    assert abs(z[np.argmax(d2) + 1] - zs) <= z[1] - z[0]


def test_independent_has_no_kink():
    """Same two erf interfaces: 'independent' is smooth through z*."""
    s0, s1, t = 2.0, 8.0, 10.0
    L = [S.Layer('vac'), S.Layer('A', t, 4e-6, 0, 0, 0, s0, 'erf'),
         S.Layer('Si', 0, 2e-6, 0, 0, 0, s1, 'erf')]
    z = np.linspace(-12.0, 5.0, 1701) + 0.0037
    d2 = {o: np.abs(np.diff(S.Stack(L, interface_order=o).profile(z)[0].real,
                            2)).max() for o in ('ordered', 'independent')}
    assert d2['independent'] < 0.05 * d2['ordered']


def test_independent_equal_sigmas_gain_material():
    """Unlike 'ordered', 'independent' changes a thin layer even where
    nothing crosses: +2.5 % (erf) / +4.3 % (tanh) of its moment at
    t = 2 sigma (model.stack PROFILE)."""
    for model, gain in (('erf', 0.0251), ('tanh', 0.0429)):
        L = [S.Layer('v'), S.Layer('A', 10, 0, 0, 1e-6, 0, 5.0, model),
             S.Layer('s', 0, 0, 0, 0, 0, 5.0, model)]
        z = np.linspace(-150, 160, 300001)
        m = {o: S.Stack(L, interface_order=o).profile(z)[1].sum()
             * (z[1] - z[0]) / 1e-5 for o in ORDERS}
        assert m['additive'] == pytest.approx(1.0, abs=1e-9)
        assert m['ordered'] == m['additive']
        assert m['independent'] - 1 == pytest.approx(gain, abs=5e-4)


# -- T6: Monte Carlo oracle ----------------------------------------------------------
@pytest.mark.parametrize('order', ['ordered', 'independent'])
def test_monte_carlo_oracle(order):
    """Interface heights from one uniform U per draw ('ordered', comonotone)
    or one per interface ('independent'), each interface kept above the
    one under it, h'_j = min(h_j, h'_{j+1}): the empirical occupancy of
    every layer is F_{k-1} - F_k."""
    L = [S.Layer('vac'), S.Layer('A', 6, 1e-6, 0, 0, 0, 2.0, 'erf'),
         S.Layer('B', 3, 2e-6, 0, 0, 0, 7.0, 'tanh'),
         S.Layer('C', 12, 3e-6, 0, 0, 0, 0.0, 'none'),
         S.Layer('D', 8, 4e-6, 0, 0, 0, 9.0, 'erf'),
         S.Layer('Si', 0, 5e-6, 0, 0, 0, 4.0, 'tanh')]
    st = S.Stack(L, interface_order=order)
    Z = st._interfaces()
    rng = np.random.default_rng(7)
    h = []
    for Zj, l in zip(Z, L[1:]):
        if order == 'independent' or not h:
            U = rng.uniform(size=1_000_000)
        s = l.roughness_sigma if l.roughness_model != 'none' else 0.0
        if l.roughness_model == 'erf':
            h.append(Zj + s * ndtri(U))
        elif l.roughness_model == 'tanh':
            h.append(Zj + s * np.sqrt(2 * np.pi) / 4 * np.log(U / (1 - U)))
        else:
            h.append(np.full(U.shape, Zj))
    h = np.array(h)
    for j in range(len(h) - 2, -1, -1):
        h[j] = np.minimum(h[j], h[j + 1])
    z = np.linspace(-15, 45, 61) + 0.013
    occ = occupancies(st._rms_steps(z))
    for i, zi in enumerate(z):
        k = np.sum(h < zi, axis=0)                     # layer at depth zi
        emp = np.bincount(k, minlength=len(L)) / U.size
        assert np.max(np.abs(emp - occ[:, i])) <= 3e-3


# -- T7: slabs sample the profile ----------------------------------------------------
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('mode', MODES)
def test_slabs_sample_the_profile(order, mode):
    st = crossing(order, mode)
    z = st.windows()[0][0] + np.concatenate(
        [[0], np.cumsum([s.thickness for s in st.sublayers])])
    nsld, rho, theta, phi = st.profile(0.5 * (z[:-1] + z[1:]))
    slab = np.array([complex(s.NSLD_real, s.NSLD_img) for s in st.sublayers])
    assert np.max(np.abs(slab - nsld)) <= 1e-18
    prof = rho * S.cos_turns(phi) * np.exp(2j * np.pi * theta)
    slab = np.array([S.Stack.inplane_rho(s) * np.exp(2j * np.pi * s.MSLD_theta)
                     for s in st.sublayers])
    assert np.max(np.abs(slab - prof)) <= 1e-18


# -- T8: physics on a crossing stack ---------------------------------------------------
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('mode', MODES)
def test_physics_with_crossing(order, mode):
    st = crossing(order, mode, fe_img=0.0)
    R = st.reflectivities(Q, LAB)
    assert np.max(np.abs(R[:, 1] - R[:, 2])) < 1e-15            # reciprocity
    # independent oracle: Ruehm, Toperverg & Dosch, PRB 60, 16073, Eq. 5
    lay = [(complex(l.NSLD_real, l.NSLD_img), S.Stack.inplane_rho(l),
            [S.cos_turns(l.MSLD_theta), S.sin_turns(l.MSLD_theta), 0.0],
            l.thickness) for l in st.sublayers]
    Rh = [sm.R_hat(q / 2, lay, 2.07e-6) for q in Q[::10]]
    rng = np.random.default_rng(2)
    for _ in range(3):
        P0, P = (v / np.linalg.norm(v) * 0.9 for v in rng.normal(size=(2, 3)))
        got = st.reflectivity(Q[::10], P0, P)
        want = np.array([sm.refl_eq5(r, P0, P) for r in Rh])
        assert np.max(np.abs(got - want) / want) <= 1e-11
    # rotating every M and P about the film normal changes nothing
    d = 0.137
    rot = copy.deepcopy(st)
    for l in rot.layers:
        l.MSLD_theta += d
    rot.build_sublayers()
    P = np.array([S.cos_turns(d), S.sin_turns(d), 0.0])
    a = rot.reflectivities(Q, [(P, P), (P, -P)])
    b = st.reflectivities(Q, [(X, X), (X, -X)])
    assert np.max(np.abs(a - b) / b) <= 1e-11
    # total reflection below every critical edge: unitary
    b = st.backing
    b.NSLD_real, b.MSLD_rho, b.MSLD_theta = 10e-6, 2e-6, 0.2
    st.build_sublayers()
    q = np.linspace(0.002, 0.019, 100)
    for _ in range(3):
        v = rng.normal(size=3)
        assert np.max(np.abs(st.reflectivity(q, v / np.linalg.norm(v)) - 1)) \
            <= 1e-13


# -- T9: the Licorne scheme ignores the setting ------------------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_licorne_scheme_ignores_the_order(mode):
    out = []
    for order in ORDERS:
        st = crossing(order, mode)
        st.roughness_scheme = 'licorne'
        out.append(outputs(st, np.linspace(-40, 140, 2001)))
    pa, sa, Ra = out[0]
    for pb, sb, Rb in out[1:]:
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb))
        assert np.array_equal(sa, sb) and np.array_equal(Ra, Rb)
    assert crossing('ordered', mode).roughness_scheme == 'rms'
    st = crossing('ordered', mode)
    st.roughness_scheme = 'licorne'
    assert st.interface_crossing() == []


# -- T11: one evaluation of the steps for every quantity --------------------------------
def _old_profile(st, z):
    """Stack.profile ('vector') as it was computed before: the steps
    evaluated again for every quantity."""
    Z = st._interfaces()
    lay = st.layers[1:]

    def one(vals):
        out = np.full(z.shape, vals[0], dtype=vals.dtype)
        for j, l in enumerate(lay):
            out = out + (vals[j + 1] - vals[j]) * S._step(
                z - Z[j], max(l.roughness_sigma, 0.0), l.roughness_model)
        return out
    rho = np.array([l.MSLD_rho for l in st.layers])
    th = np.array([l.MSLD_theta for l in st.layers])
    ph = np.array([l.MSLD_phi for l in st.layers])
    mp = one(rho * S.cos_turns(ph) * np.exp(2j * np.pi * th))
    mz = one(rho * S.sin_turns(ph))
    a = np.abs(mp)
    return (one(st._nsld()), np.hypot(a, mz),
            S.wrap_turns(np.angle(mp) / (2 * np.pi)),
            np.arctan2(mz, a) / (2 * np.pi))


@pytest.mark.parametrize('name', ['session_Cr2Te3', 'show_mistake'])
def test_not_slower(name):
    """profile() (build_sublayers is the same at the slab centres)."""
    st = session(name)
    assert st.magnetic_smearing == 'vector'
    z = np.linspace(-60, 300, 4000)
    old = _old_profile(st, z)
    assert all(np.array_equal(a, b) for a, b in zip(old, st.profile(z)))
    # old and new timed in turn, so that a busy machine slows both alike
    t = {'old': [], 'additive': [], 'ordered': [], 'independent': []}
    for _ in range(7):
        t['old'].append(timeit.timeit(lambda: _old_profile(st, z), number=20))
        for order in ORDERS:
            st.interface_order = order
            t[order].append(timeit.timeit(lambda: st.profile(z), number=20))
    for order in ORDERS:
        assert min(t[order]) <= 1.25 * min(t['old']), order


# -- T12: acceptance on show_mistake.json ---------------------------------------------------
def test_show_mistake():
    st = session('show_mistake')
    z = np.linspace(-60, 260, 32001)
    dz = z[1] - z[0]
    nominal = sum(l.thickness * l.MSLD_rho for l in st.layers[1:-1])
    res = {}
    for order in ORDERS:
        st.interface_order = order
        nsld, rho, theta, phi = st.profile(z)
        Mx = rho * S.cos_turns(phi) * S.cos_turns(theta)
        th = theta[st.angle_defined(rho)]
        res[order] = (Mx, np.ptp(th), Mx.sum() * dz, nsld.real.sum() * dz)
    Mx, dth, iM, iN = res['additive']
    assert Mx.min() == pytest.approx(-0.4354e-6, rel=1e-3)
    assert abs(z[np.argmin(Mx)] - 69.2) <= 0.1
    assert iM == pytest.approx(nominal, rel=1e-3)
    assert dth == pytest.approx(0.5, abs=1e-9)
    Mx, dth, iM, iN_o = res['ordered']
    assert Mx.min() >= -1e-20 and dth <= 1e-12
    assert iM == pytest.approx(16.685e-6, rel=1e-3)
    assert iN_o == pytest.approx(iN, rel=1e-4)
    # the diagnostic: L3's occupancy goes negative inside CrSb, M_x to
    # -163 % of CrSb's
    names = [l.name for l in st.layers]
    big = [c for c in st.interface_crossing()
           if max(map(abs, c['excursion'].values())) > 1.0]
    assert [(names[c['layer']], names[c['host']]) for c in big] == \
        [('L3', 'CrSb')]
    assert big[0]['excursion']['M_x'] == pytest.approx(-163.4, abs=0.5)
    assert big[0]['zrange'][0] < 69.2 < big[0]['zrange'][1]
    # the Cr2Te3 session hardly crosses
    for c in session('session_Cr2Te3').interface_crossing():
        assert c['occupancy'] > -1e-3
        assert max(map(abs, c['excursion'].values())) < 1.0


# -- T13: smooth in the parameters ------------------------------------------------------------
def _sweep(order, attr, layer, ps, nsub=1):
    with open(DATA / 'show_mistake.json') as f:
        st = S.Stack.from_dict(json.load(f))
    st.interface_order, st.resolution['enabled'] = order, False
    for l in st.layers:
        l.roughness_sublayer *= nsub
    q = np.linspace(0.01, 0.2, 40)
    out = []
    for p in ps:
        setattr(st.layers[layer], attr, p)
        st.build_sublayers()
        out.append(np.log(st.reflectivities(q, LAB[::3], warn=False)).ravel())
    return np.array(out)


# BAT's sigma (the L3 / BAT interface) through L3's own 1.2 A, where the
# interfaces start to cross; L3's thickness down to 0
SWEEPS = [('roughness_sigma', 4, 0.9, 1.5), ('thickness', 3, 0.0, 0.6)]


@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('attr, layer, lo, hi', SWEEPS)
def test_continuous_in_the_parameters(order, attr, layer, lo, hi):
    """No jump in R: four times finer steps, four times smaller changes."""
    a = np.abs(np.diff(_sweep(order, attr, layer, np.linspace(lo, hi, 25)),
                       axis=0)).max()
    b = np.abs(np.diff(_sweep(order, attr, layer, np.linspace(lo, hi, 97)),
                       axis=0)).max()
    assert b <= 0.35 * a


@pytest.mark.parametrize('attr, layer, lo, hi', SWEEPS)
def test_ordered_kinks_shrink_with_the_slabs(attr, layer, lo, hi):
    """'ordered' R has slope kinks in a parameter where a crossing point
    passes a slab centre (the slabs sample the kinked profile at points);
    they are small and shrink as the slabs get thinner."""
    ps = np.linspace(lo, hi, 61)
    kink = [np.abs(np.diff(_sweep('ordered', attr, layer, ps, n), 3,
                           axis=0)).max() for n in (1, 4)]
    slope = np.abs(np.diff(_sweep('ordered', attr, layer, ps), axis=0)).max()
    assert kink[0] <= 0.1 * slope
    assert kink[1] <= 0.6 * kink[0]


@pytest.mark.parametrize('attr, layer, lo, hi', SWEEPS)
def test_independent_has_no_kinks(attr, layer, lo, hi):
    """'independent' R is as smooth in the parameters as 'additive': its
    third differences stay at the additive level, far below 'ordered'."""
    ps = np.linspace(lo, hi, 61)
    d3 = {o: np.abs(np.diff(_sweep(o, attr, layer, ps), 3, axis=0)).max()
          for o in ORDERS}
    assert d3['independent'] <= 3 * d3['additive']
    assert d3['independent'] <= 0.01 * d3['ordered']


# -- the fit windows take the stack as it is ---------------------------------------------------
def test_de_and_dream_with_ordered(tmp_path):
    """Smoke test: DE finds an ordered truth, DREAM samples it, a saved run
    reloads ordered and its profile bands are finite."""
    from fit.de.de_fit import run_de
    from fit.dream.dream_fit import (load_run, posterior_problem,
                                     profile_bands, sample, save_result)
    q = np.linspace(0.006, 0.15, 120)
    p = [channel_pair(c, X) for c in ('++', '--')]
    R = crossing('ordered').reflectivities(q, p)
    data = [dict(Q=q, R=R[:, j], dR=0.03 * R[:, j] + 1e-9, P0=P0, P=P)
            for j, (P0, P) in enumerate(p)]
    st = crossing('ordered')
    fm, sp = st.layers[1], st.layers[2]
    fm.MSLD_rho, sp.roughness_sigma = 1.5e-6, 12.0
    fm.fit['MSLD_rho'] = {'vary': True, 'min': 0.5e-6, 'max': 4e-6}
    sp.fit['roughness_sigma'] = {'vary': True, 'min': 5.0, 'max': 20.0}
    pp = posterior_problem(st, data)
    assert pp.fit.stack.interface_order == 'ordered'
    x, cost, *_ = run_de(pp.fit, seed=0, maxiter=60)
    assert cost < 1e-4
    assert x == pytest.approx([2e-6, 15.0], rel=1e-3)
    res = sample(pp, n_gen=300, n_chains=3, seed=1, stop_when_converged=False)
    f = tmp_path / 'run.npz'
    save_result(f, res, pp)
    _, pp2, _ = load_run(f)
    assert pp2.fit.stack.interface_order == 'ordered'
    z, nuc, mag = profile_bands(res, pp, n=10)
    assert np.all(np.isfinite(nuc)) and np.all(np.isfinite(mag))
