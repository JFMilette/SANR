"""Licorne roughness scheme (model.roughness, Stack roughness_scheme='licorne'),
the 'step' magnetisation-angle rule and the Licorne importer.

Section numbers (8.1 ...) are those of the implementation plan.  SLDs are in
A^-2 and angles in turns, as everywhere in model.stack.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfinv
from scipy.stats import norm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import model.roughness as rg                                    # noqa: E402
import model.stack as S                                         # noqa: E402
from model.licorne_io import (licorne_direction,                # noqa: E402
                              load_licorne_model, read_table)

DATA = HERE / 'data' / 'licorne'
FIX1 = DATA / 'fixture1'
V127 = ['v127_chi3_137', 'v127_r2_6_508']
X = np.array([1.0, 0, 0])
LAB = [(X, X), (X, -X), (-X, X), (-X, -X)]
Q = np.linspace(0.004, 0.2, 400)
MODES = S.SMEARING_MODES

# fixture 1 rows: 5 windows x 6 sublayers, 4 cores, the substrate
CORES = [6, 13, 20, 27]
CLIPPED = list(range(7, 13))                     # layer 1 / layer 2
UNCLIPPED = (list(range(0, 6)) + list(range(14, 20)) + list(range(21, 27))
             + list(range(28, 34)))


# -- helpers -----------------------------------------------------------------
def fixture1(renorm='manual'):
    st = load_licorne_model(FIX1 / 'parameters.m', FIX1 / 'profile.dat')
    st.licorne_renorm = renorm
    st.build_sublayers()
    return st


def cols(st):
    """thickness, Re NSLD, Im NSLD, rho, theta, phi of the slabs."""
    return [np.array([getattr(s, a) for s in st.sublayers]) for a in
            ('thickness', 'NSLD_real', 'NSLD_img', 'MSLD_rho', 'MSLD_theta',
             'MSLD_phi')]


def centres(st):
    e = st.windows()[0][0] + np.concatenate(
        [[0], np.cumsum([s.thickness for s in st.sublayers])])
    return 0.5 * (e[:-1] + e[1:])


def unit(theta, phi):
    """Unit M in the sample frame from (theta, phi) in turns."""
    theta, phi = np.asarray(theta), np.asarray(phi)
    return np.stack([S.cos_turns(phi) * S.cos_turns(theta),
                     S.cos_turns(phi) * S.sin_turns(theta),
                     S.sin_turns(phi)], -1)


def m_par_perp(rho, theta, phi):
    """In-plane M along x (m_par) and along y (m_perp)."""
    mp = rho * S.cos_turns(phi)
    return mp * S.cos_turns(theta), mp * S.sin_turns(theta)


def rel(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    return float(np.max(np.abs(a[ok] - b[ok]) / np.abs(b[ok])))


def licorne_fitted_F(values, s, fa):
    """Least-squares F_b of v_i = f_a + (F_b - f_a) s_i (layer a unchanged)."""
    return fa + np.sum((values - fa) * s) / np.sum(s * s)


def bilayer(rho_a, th_a, rho_b, th_b, mode, fallback=True, scheme='licorne',
            sigma=8.0, N=6, kind='tanh', ta=200.0, tb=200.0):
    """vacuum / A / B / substrate, only the A/B interface rough."""
    L = [S.Layer('vac', 0, 0, 0, 0, 0),
         S.Layer('A', ta, 4e-6, 0, rho_a, th_a, 0.0, kind, N),
         S.Layer('B', tb, 2e-6, 0, rho_b, th_b, sigma, kind, N),
         S.Layer('sub', 0, 2.07e-6, 0, 0, 0, 0.0, kind, N)]
    st = S.Stack(L, roughness_scheme=scheme, magnetic_smearing=mode)
    st.step_fallback = fallback
    st.build_sublayers()
    return st


def window_slabs(st, j):
    """(indices, offsets x) of the slabs of Licorne window j."""
    geo = st._licorne_interfaces()
    i, out = 0, None
    for lo, hi, (what, w) in st._licorne_regions(geo):
        if what == 'layer':
            i += 1
            continue
        n = w['N']
        if w['a'] == j:
            h = (w['la'] + w['lb']) / n
            out = (np.arange(i, i + n), -w['la'] + h * (np.arange(n) + 0.5))
        i += n
    return out


def magnetic(scheme, mode, thetas=(0.10, 0.35, 0.0), rough=8.0, N=6,
             fe_img=-1e-8, rot=0.0):
    """vac / Fe 80 / Cr 25 / Co 60 / Si, as tests/test_stack.mk; every
    theta (fronting and backing included) turned by rot."""
    L = [S.Layer('front', 0, 0, 0, 0, rot),
         S.Layer('Fe', 80, 8e-6, fe_img, 5e-6, thetas[0] + rot, rough,
                 'tanh', N),
         S.Layer('Cr', 25, 3e-6, 0, 0, thetas[2] + rot, rough, 'tanh', N),
         S.Layer('Co', 60, 6e-6, 0, 4e-6, thetas[1] + rot, rough, 'tanh', N),
         S.Layer('Si', 0, 2.07e-6, 0, 0, rot, rough, 'tanh', N)]
    st = S.Stack(L, roughness_scheme=scheme, magnetic_smearing=mode)
    st.build_sublayers()
    return st


# -- 8.1 constants -----------------------------------------------------------
def test_constants():
    assert abs(rg.ALPHA['erf'] - 0.953873) < 1e-6
    assert abs(rg.ALPHA['tanh'] - 1.098612) < 1e-6
    assert abs(rg.gamma_licorne(1.0, 'tanh') - 2.47584) < 1e-4
    assert abs(rg.gamma_licorne(1.0, 'erf') - 2.0913) < 1e-4
    for kind in ('tanh', 'erf'):
        for sL in (0.3, 8.0):
            k, g = rg.scale_k(sL, kind), rg.gamma_licorne(sL, kind)
            assert abs(rg.g_licorne(k * g, kind) - 0.97) < 1e-14
            # 25-75 % width = 1.3 sigma_L for both functions
            assert abs(rg.profile_fraction(0.65 * sL, sL, kind) - 0.75) < 1e-14


def test_sigma_conversion():
    # exactly 1.3 / (2 erfinv(1/2) sqrt 2) = 0.9636914; the plan's 0.963690
    # is that value truncated
    assert abs(rg.sigma_rms_from_licorne(1.0, 'erf') - 0.963690) < 2e-6
    sL = 7.3
    sr = rg.sigma_rms_from_licorne(sL, 'erf')
    x = np.linspace(-30, 30, 101)
    assert np.max(np.abs(rg.profile_fraction(x, sL, 'erf')
                         - norm.cdf(x / sr))) < 1e-14
    assert abs(rg.sigma_licorne_from_rms(sr, 'erf') - sL) < 1e-12
    # tanh: the width-matched (25-75 %) value, documented as such
    assert rg.sigma_rms_from_licorne(sL, 'tanh') == sr
    with pytest.raises(ValueError):
        rg.sigma_rms_from_licorne(1.0, 'none')


# -- 8.2 closed-form J -------------------------------------------------------
def test_closed_form_J_matches_quad():
    rng = np.random.default_rng(2)
    for _ in range(200):
        kind = ('tanh', 'erf')[rng.integers(2)]
        sL = rng.uniform(0.1, 20)
        k = rg.scale_k(sL, kind)
        la, lb = rng.uniform(0, 3 * sL, 2)
        ref = quad(lambda x: rg.g_licorne(k * x, kind), -la, lb,
                   epsabs=1e-12, epsrel=1e-12, limit=200)[0]
        assert abs(rg.window_integral(la, lb, k, kind) - ref) < 1e-10
    # no overflow far out in the tails
    assert np.isfinite(rg.window_integral(0, 1e4, 1.0, 'tanh'))


def test_J_reference():
    for kind, ref in (('erf', 9.410551), ('tanh', 9.177263)):
        J = rg.window_integral(5.0, 16.73, rg.scale_k(8.0, kind), kind)
        assert abs(J - ref) < 1e-6


# -- 8.3 kernel ----------------------------------------------------------------
@pytest.mark.parametrize('kind', ['tanh', 'erf'])
def test_symmetric_unclipped(kind):
    fa, fb, sL = 3.0, 8.0, 8.0
    x, h, v, s, Fa, Fb, la, lb = rg.licorne_interface(fa, fb, 100, 100, sL, 6,
                                                      kind)
    assert la == lb == rg.gamma_licorne(sL, kind)
    assert Fa[0] == fa and Fb[0] == fb
    assert abs(rg.window_integral(la, lb, rg.scale_k(sL, kind), kind)) < 1e-12
    assert np.max(np.abs(v[:, 0] + v[::-1, 0] - (fa + fb))) < 1e-12   # odd
    assert np.max(np.abs(s + s[::-1] - 1)) < 1e-14
    f = lambda z: rg.licorne_profile_continuous([z], fa, fb, 100, 100, sL,
                                                kind)[0, 0]
    I = quad(f, -la, lb, points=[0], epsabs=1e-12, limit=200)[0]
    assert abs(I - (la * fa + lb * fb)) < 1e-9


def _integral(fa, fb, ta, tb, sL, kind, renorm):
    Fa, Fb, la, lb, _ = rg.renormalised(fa, fb, ta, tb, sL, kind, renorm)
    f = lambda z: rg.licorne_profile_continuous([z], fa, fb, ta, tb, sL,
                                                kind, renorm)[0, 0]
    return quad(f, -la, lb, points=[0], epsabs=1e-12, limit=200)[0], la, lb


def test_clipped_erf():
    Fa, Fb, la, lb, _ = rg.renormalised(3.0, 8.0, 10.0, 100.0, 8.0, 'erf')
    assert abs(Fa - 3.9414) < 5e-5 and Fb == 8.0
    assert la == 5.0 and lb == rg.gamma_licorne(8.0, 'erf')
    I, la, lb = _integral(3.0, 8.0, 10.0, 100.0, 8.0, 'erf', 'manual')
    assert abs(I - 148.843) < 1e-3
    # the renormalised window holds exactly the nominal amount of material
    assert abs(I - (la * 3.0 + lb * 8.0)) < 1e-6
    I0, _, _ = _integral(3.0, 8.0, 10.0, 100.0, 8.0, 'erf', 'none')
    assert abs(I0 - 143.044) < 1e-3


@pytest.mark.parametrize('kind', ['tanh', 'erf'])
def test_mirror(kind):
    a = rg.licorne_interface(3.0, 8.0, 10.0, 100.0, 8.0, 6, kind)
    b = rg.licorne_interface(8.0, 3.0, 100.0, 10.0, 8.0, 6, kind)
    assert b[4][0] == 8.0 and abs(b[5][0] - a[4][0]) < 1e-12   # F swapped
    assert np.max(np.abs(b[2][::-1] - a[2])) < 1e-12           # mirrored
    I, la, lb = _integral(8.0, 3.0, 100.0, 10.0, 8.0, kind, 'manual')
    assert abs(I - (la * 8.0 + lb * 3.0)) < 1e-6


def test_renormalisation_cases():
    # equal thicknesses, both clipped: neither side changes
    Fa, Fb, la, lb, _ = rg.renormalised(3.0, 8.0, 10.0, 10.0, 8.0, 'tanh')
    assert (Fa, Fb, la, lb) == (3.0, 8.0, 5.0, 5.0)
    # a semi-infinite medium is never renormalised, its thin neighbour is
    Fa, Fb, *_ = rg.renormalised(3.0, 8.0, np.inf, 10.0, 8.0, 'tanh')
    assert Fa == 3.0 and Fb != 8.0
    Fa, Fb, *_ = rg.renormalised(3.0, 8.0, 10.0, np.inf, 8.0, 'tanh')
    assert Fa != 3.0 and Fb == 8.0
    Fa, Fb, la, lb, _ = rg.renormalised(3.0, 8.0, np.inf, np.inf, 8.0, 'tanh')
    assert (Fa, Fb) == (3.0, 8.0) and la == lb == rg.gamma_licorne(8, 'tanh')
    # unclipped thin side (gamma < t/2): no renormalisation
    Fa, Fb, *_ = rg.renormalised(3.0, 8.0, 30.0, 100.0, 2.0, 'tanh')
    assert (Fa, Fb) == (3.0, 8.0)


def test_sharp_interface():
    for sL, kind in ((0.0, 'tanh'), (5.0, 'none')):
        x, h, v, s, Fa, Fb, la, lb = rg.licorne_interface(3.0, 8.0, 10, 10,
                                                          sL, 6, kind)
        assert len(x) == 0 and h == 0 and la == lb == 0
    with pytest.raises(ValueError):
        rg.licorne_interface(3.0, 8.0, 10, 10, 1.0, 6, 'gauss')


def test_sampling():
    x, h, v, s, Fa, Fb, la, lb = rg.licorne_interface(
        [3.0, -1.0], [8.0, 2.0], 10.0, 100.0, 8.0, 7, 'tanh')
    assert abs(h - (la + lb) / 7) < 1e-15
    assert np.allclose(x, -la + h * (np.arange(7) + 0.5), atol=1e-14)
    assert np.allclose(s, (np.tanh(rg.scale_k(8.0, 'tanh') * x) + 1) / 2)
    assert np.allclose(v, Fa + (Fb - Fa) * s[:, None])


# -- 4. stack in the Licorne scheme ------------------------------------------
def test_defaults_unchanged():
    L = [S.Layer('a', 0), S.Layer('b', 10), S.Layer('c', 0)]
    st = S.Stack(L)
    assert st.roughness_scheme == 'rms' and st.magnetic_smearing == 'vector'
    assert st.step_fallback and st.licorne_renorm == 'manual'
    assert S.Stack(L, roughness_scheme='licorne').magnetic_smearing == 'step'
    assert S.Stack(L, roughness_scheme='licorne',
                   magnetic_smearing='angle').magnetic_smearing == 'angle'
    with pytest.raises(ValueError):
        S.Stack(L, roughness_scheme='nc')
    st.magnetic_smearing = 'rotate'
    with pytest.raises(ValueError):
        st.build_sublayers()


def test_to_dict_round_trip():
    import json
    st = magnetic('licorne', 'angle')
    st.step_fallback, st.licorne_renorm = False, 'none'
    d = json.loads(json.dumps(st.to_dict()))
    assert S.Stack.from_dict(d).to_dict() == st.to_dict()
    # a file written before the Licorne scheme existed loads as 'rms'
    for k in ('roughness_scheme', 'licorne_renorm', 'step_fallback'):
        del d[k]
    old = S.Stack.from_dict(d)
    assert old.roughness_scheme == 'rms' and old.magnetic_smearing == 'angle'


def test_rms_none_function_is_sharp():
    st = bilayer(0, 0, 0, 0, 'vector', scheme='rms', kind='none')
    assert st.windows()[1] == (200.0, 200.0)
    nsld = st.profile(np.array([199.999, 200.001]))[0].real
    assert list(nsld) == [4e-6, 2e-6]


def test_window_edge_jumps():
    """1.5 % of the step at each unclipped window edge, in the profile."""
    st = bilayer(0, 0, 0, 0, 'step')
    la, lb = st._licorne_interfaces()[1]['la'], st._licorne_interfaces()[1]['lb']
    eps = 1e-9
    z = 200 + np.array([-la - eps, -la + eps, lb - eps, lb + eps])
    n = st.profile(z)[0].real
    step = 2e-6 - 4e-6
    assert n[0] == 4e-6 and n[3] == 2e-6
    assert abs((n[1] - 4e-6) / step - 0.015) < 1e-6
    assert abs((2e-6 - n[2]) / step - 0.015) < 1e-6


def test_thin_layer_jump_at_midpoint():
    """A thin layer clipped by both interfaces: no core, two F, a jump at
    its midpoint, in the slabs and in the profile."""
    L = [S.Layer('vac', 0, 0, 0, 0, 0),
         S.Layer('A', 100, 4e-6, 0, 0, 0, 0.0, 'tanh', 6),
         S.Layer('thin', 10, 1e-6, 0, 3e-6, 0, 8.0, 'tanh', 6),
         S.Layer('B', 100, 6e-6, 0, 0, 0, 3.0, 'tanh', 6),
         S.Layer('sub', 0, 2e-6, 0, 0, 0, 0.0, 'tanh', 6)]
    st = S.Stack(L, roughness_scheme='licorne')
    st.build_sublayers()
    geo = st._licorne_interfaces()
    assert geo[1]['lb'] == 5.0 and geo[2]['la'] == 5.0
    # the thin layer has no core slab: 6 + 6 window slabs between A's cores
    th = cols(st)[0]
    assert len(th) == 1 + 6 + 6 + 1 and abs(th.sum() - 210) < 1e-12
    Fb1, Fa2 = geo[1]['Fb'], geo[2]['Fa']
    assert abs(Fb1[0] - Fa2[0]) > 1e-8                  # two different F
    n = st.profile(np.array([105 - 1e-9, 105 + 1e-9]))[0].real
    assert abs(n[1] - n[0]) > 1e-8
    assert np.all(Fb1[2] > 0) and np.all(Fa2[2] > 0)   # rho stays positive


def test_thin_nonmagnetic_layer_takes_neighbour_angle():
    """A thin rho = 0 layer renormalised by a magnetic neighbour gets
    F_rho > 0 in its half of the window; 'step' gives those slabs the
    neighbour's angle (fallback), never the thin layer's own."""
    L = [S.Layer('vac', 0, 0, 0, 0, 0),
         S.Layer('M', 100, 4e-6, 0, 3e-6, 0.1, 0.0, 'tanh', 6),
         S.Layer('thin', 10, 1e-6, 0, 0, 0.4, 8.0, 'tanh', 6),
         S.Layer('sub', 0, 2e-6, 0, 0, 0, 0.0, 'tanh', 6)]
    st = S.Stack(L, roughness_scheme='licorne')
    st.build_sublayers()
    w = st._licorne_interfaces()[1]
    assert w['Fb'][2] > 0 and np.all(cols(st)[3] >= 0)
    i, x = window_slabs(st, 1)
    th = cols(st)[4][i]
    assert np.allclose(th, 0.1, atol=1e-12)
    st.step_fallback = False
    st.build_sublayers()
    th = cols(st)[4][i]
    assert np.allclose(th[x > 0], 0.4) and np.allclose(th[x <= 0], 0.1)


# -- 5. the 'step' rule -------------------------------------------------------
def test_step_magnetic_against_nonmagnetic():
    """rho = 5e-6 at angle 0 against rho = 0: 'step' (fallback), 'angle'
    and 'vector' give the same slabs when the non-magnetic angle is 0;
    with an arbitrary non-magnetic angle only 'step' without the fallback
    (and 'angle', which turns M by design) leaks it."""
    ref = cols(bilayer(5e-6, 0.0, 0, 0.0, 'step'))
    for mode in ('angle', 'vector'):
        c = cols(bilayer(5e-6, 0.0, 0, 0.0, mode))
        for a, b in zip(c, ref):
            assert np.max(np.abs(a - b)) <= 1e-18
    for mode in MODES:
        rho, th, ph = cols(bilayer(5e-6, 0.0, 0, 0.37, mode))[3:]
        par, perp = m_par_perp(rho, th, ph)
        if mode == 'angle':
            assert np.max(np.abs(perp)) > 1e-7
        else:
            assert np.max(np.abs(perp)) == 0.0
    st = bilayer(5e-6, 0.0, 0, 0.37, 'step', fallback=False)
    i, x = window_slabs(st, 1)
    _, _, _, rho, th, ph = cols(st)
    perp = m_par_perp(rho, th, ph)[1][i]
    assert np.all(perp[x > 0] != 0) and np.all(perp[x <= 0] == 0)


def test_step_antiparallel():
    sa, sb = 5e-6, 5e-6
    st = bilayer(sa, 0.0, sb, 0.5, 'step')
    i, x = window_slabs(st, 1)
    _, _, _, rho, th, ph = cols(st)
    par, perp = m_par_perp(rho, th, ph)
    assert np.max(np.abs(perp)) == 0.0
    assert np.allclose(rho[i], 5e-6, rtol=1e-14)                 # no dip
    assert np.all(par[i][x <= 0] > 0) and np.all(par[i][x > 0] < 0)
    # the profile flips exactly at the interface
    # (+-1e-6: within STEP_TIE of x = 0 a depth still counts as the tie)
    _, r, t, p = st.profile(np.array([200 - 1e-6, 200 + 1e-6]))
    assert list(m_par_perp(r, t, p)[0]) == [5e-6, -5e-6]
    # 'vector': no perpendicular part, |M| dips to 0 at the interface
    v = bilayer(sa, 0.0, sb, 0.5, 'vector')
    assert np.max(np.abs(m_par_perp(*cols(v)[3:])[1])) < 1e-21
    assert v.profile(np.array([200.0]))[1][0] < 1e-20
    # 'angle': a 90 deg wall whose sense depends on writing pi or -pi
    for th_b, sign in ((0.5, 1), (-0.5, -1)):
        a = bilayer(sa, 0.0, sb, th_b, 'angle')
        _, r, t, p = a.profile(np.array([200.0]))
        assert abs(m_par_perp(r, t, p)[1][0] - sign * 5e-6) < 1e-20


def test_step_tie_goes_to_upper_layer():
    """Odd N in a symmetric window: the middle slab is at x = 0."""
    st = bilayer(5e-6, 0.1, 5e-6, 0.3, 'step', N=5)
    i, x = window_slabs(st, 1)
    assert abs(x[2]) < 1e-12
    assert abs(st.sublayers[i[2]].MSLD_theta - 0.1) < 1e-12


def test_step_both_nonmagnetic():
    st = bilayer(0, 0.2, 0, 0.4, 'step')
    i, _ = window_slabs(st, 1)
    assert all(st.sublayers[k].MSLD_theta == 0 for k in i)
    st = bilayer(0, 0.2, 0, 0.4, 'step', fallback=False)
    assert {round(st.sublayers[k].MSLD_theta, 9) for k in i} == {0.2, 0.4}


@pytest.mark.parametrize('fallback', [True, False])
def test_step_rms_scheme(fallback):
    """'step' in the 'rms' scheme: largest occupancy, same fallback."""
    st = bilayer(5e-6, 0.0, 5e-6, 0.5, 'step', scheme='rms')
    _, _, _, rho, th, ph = cols(st)
    assert np.max(np.abs(m_par_perp(rho, th, ph)[1])) == 0.0
    c = centres(st)
    near = np.abs(c - 200) < 20
    assert np.allclose(rho[near], 5e-6, rtol=1e-12)
    assert np.all(np.where(c[near] < 200, th[near] == 0, th[near] == 0.5))
    st = bilayer(5e-6, 0.0, 0, 0.37, 'step', scheme='rms', fallback=fallback)
    _, _, _, rho, th, ph = cols(st)
    perp = m_par_perp(rho, th, ph)[1]
    assert (np.max(np.abs(perp)) == 0) == fallback


# -- 6. profile consistency ---------------------------------------------------
def _angled():
    """Magnetic stack with distinct angles, a thin non-magnetic layer, a
    thin magnetic one and an odd N (tie slab at x = 0)."""
    L = [S.Layer('vac', 0, 0, 0, 0, 0),
         S.Layer('A', 60, 4e-6, -1e-8, 3e-6, 0.10, 5.0, 'tanh', 6,
                 MSLD_phi=0.05),
         S.Layer('thin', 9, 2e-6, 0, 0, 0.40, 6.0, 'erf', 5),
         S.Layer('B', 12, 6e-6, 0, 2e-6, 0.65, 7.0, 'tanh', 4),
         S.Layer('C', 80, 3e-6, 0, 1e-6, -0.2, 2.0, 'tanh', 3),
         S.Layer('sub', 0, 2.07e-6, 0, 0, 0, 1.0, 'erf', 6)]
    return L


@pytest.mark.parametrize('scheme', ['licorne', 'rms'])
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fallback', [True, False])
def test_profile_at_slab_centres(scheme, mode, fallback):
    # (the rms slicer puts slab edges on a 1e-6 A grid, so fixture 1's
    # unrounded depths only line up in the Licorne scheme)
    stacks = [S.Stack(_angled(), roughness_scheme=scheme,
                      magnetic_smearing=mode)]
    if scheme == 'licorne':
        stacks.append(fixture1())
    for st in stacks:
        st.roughness_scheme, st.magnetic_smearing = scheme, mode
        st.step_fallback = fallback
        st.build_sublayers()
        _, re_, im_, rho, th, ph = cols(st)
        n, r, t, p = st.profile(centres(st))
        scale = max(np.max(np.abs(re_)), 1e-6)
        assert np.max(np.abs(n.real - re_)) <= 1e-12 * scale
        assert np.max(np.abs(n.imag - im_)) <= 1e-12 * scale
        assert np.max(np.abs(r - rho)) <= 1e-12 * scale
        # m_par, m_perp, m_z and the angles where there is an M
        for a, b in zip(m_par_perp(r, t, p), m_par_perp(rho, th, ph)):
            assert np.max(np.abs(a - b)) <= 1e-12 * scale
        assert np.max(np.abs(r * S.sin_turns(p) - rho * S.sin_turns(ph))) \
            <= 1e-12 * scale
        on = rho > 1e-3 * scale
        dth = (t[on] - th[on] + 0.5) % 1.0 - 0.5
        assert np.max(np.abs(dth), initial=0) <= 1e-12
        assert np.max(np.abs(p[on] - ph[on]), initial=0) <= 1e-12


def test_profile_shows_step_angle_jump():
    st = bilayer(5e-6, 0.1, 5e-6, 0.3, 'step')
    t = st.profile(np.array([200 - 1e-6, 200 + 1e-6]))[2]
    assert np.allclose(t, [0.1, 0.3], atol=1e-12)


# -- 8.4 fixture 1 (Licorne 1.4.2 export) -------------------------------------
REF1 = read_table(FIX1 / 'profile_sublayers.dat')


def test_fixture1_geometry():
    st = fixture1()
    th = cols(st)[0]
    assert len(st.sublayers) + 1 == len(REF1) == 35
    assert np.allclose(th, REF1[:-1, 1], rtol=1e-5, atol=0)
    assert np.allclose(th[CORES], [47.0009, 18.7335, 48.5175, 75.319],
                       rtol=1e-5)
    # Licorne's depth 0 is the top of the first window
    lo = st.windows()[0][0]
    assert abs(-lo - 0.000848556) < 2e-9
    depth = np.concatenate([[0], np.cumsum(th)])            # z - lo
    assert np.allclose(depth, REF1[:, 0], rtol=0, atol=2e-3)
    w = st._licorne_interfaces()[1]
    assert abs(w['la'] - 24.3219) < 1e-4 and w['lb'] == 41.5003 / 2


def test_fixture1_unclipped_windows():
    st = fixture1()
    _, re_, im_, rho, th, ph = cols(st)
    u = UNCLIPPED
    assert rel(re_[u], REF1[u, 2]) <= 1e-5
    mag = [i for i in u if REF1[i, 4] != 0]
    assert rel(rho[mag], REF1[mag, 4]) <= 1e-5
    assert np.all(rho[[i for i in u if REF1[i, 4] == 0]] == 0)
    # normalised profile in each unclipped window, ours and Licorne's
    s_ref = [0.02968, 0.10985, 0.33238, 0.66762, 0.89015, 0.97032]
    # nominal value above / below each window: the neighbouring core, or
    # the fronting / substrate (slab i is nom[i + 1])
    nom = np.concatenate([[0.0], re_, [st.backing.NSLD_real]])
    for k in range(0, len(u), 6):
        rows = np.array(u[k:k + 6])
        fa, fb = nom[rows[0]], nom[rows[-1] + 2]
        for v, tol in ((re_, 1e-5), (REF1[:-1, 2], 1e-4)):
            s = (v[rows] - fa) / (fb - fa)
            assert np.max(np.abs(s - s_ref)) <= tol


def test_fixture1_clipped_window():
    """Layer 1 / layer 2: clipped on the layer-2 side, which Licorne
    renormalises by ~93 % of the manual formula (TODO: J_eff, unexplained)."""
    err = {}
    for renorm in ('manual', 'none'):
        _, re_, _, rho, _, _ = cols(fixture1(renorm))
        err[renorm] = (rel(re_[CLIPPED], REF1[CLIPPED, 2]),
                       rel(rho[CLIPPED], REF1[CLIPPED, 4]))
    assert err['manual'][0] <= 1e-3 and err['manual'][1] <= 1e-3
    assert abs(err['manual'][0] - 4.6e-4) < 1e-5
    assert abs(err['manual'][1] - 2.7e-4) < 1e-5
    assert err['none'][0] > 5e-3 > 10 * err['manual'][0]
    st = fixture1()
    w = st._licorne_interfaces()[1]
    assert abs(w['Fb'][0] - 1.55113e-6) < 5e-12
    assert abs(w['Fb'][2] - 2.30705e-6) < 1e-11     # 2.3070446e-6 exactly
    # Licorne's own F_b: renormalised the same way, by ~93 %
    s = rg.profile_fraction(window_slabs(st, 1)[1], w['sigma'], 'tanh')
    for c, ours, nominal in ((2, w['Fb'][0], 1.53998e-6),
                             (4, w['Fb'][2], 2.31522e-6)):
        fa = REF1[CORES[0], c]
        F_lic = licorne_fitted_F(REF1[CLIPPED, c], s, fa)
        assert 0.90 < (F_lic - nominal) / (ours - nominal) < 0.95
    F_lic = licorne_fitted_F(REF1[CLIPPED, 2], s, 4.69889e-6)
    tb, fa, fb = 41.5003, 4.69889e-6, 1.53998e-6
    J_eff = tb * (fb - fa) / (F_lic - fa) - tb / 2 - w['la']
    assert abs(J_eff - (-3.4357)) < 2e-3


def test_fixture1_angles_stay_zero():
    _, _, _, _, th, ph = cols(fixture1())
    assert np.all(th == 0) and np.all(ph == 0)
    assert np.all(REF1[:, 5:7] == 0)


def test_import_parameters_only():
    st = load_licorne_model(FIX1 / 'parameters.m')
    assert st.roughness_scheme == 'licorne'
    assert st.magnetic_smearing == 'step'
    assert st.layers[0].name == 'vacuum' and st.layers[0].NSLD_real == 0
    assert st.layers[2].NSLD_real == 1.54e-6       # 5 figures in parameters.m
    assert [l.roughness_model for l in st.layers[1:]] == ['tanh'] * 5
    assert [l.roughness_sublayer for l in st.layers[1:]] == [6] * 5
    assert st.layers[2].roughness_sigma == 9.82371
    assert st.backing.MSLD_rho == 0
    r2 = load_licorne_model(DATA / V127[1] / 'parameters.m',
                            axis_map=np.eye(3))
    assert r2.layers[1].NSLD_img == -3e-8          # MATLAB complex literal
    assert r2.backing.roughness_sublayer == 10


def test_import_axis_map(tmp_path):
    p = DATA / V127[0] / 'parameters.m'
    with pytest.raises(ValueError, match='axis_map'):
        load_licorne_model(p)
    with pytest.raises(ValueError, match='orthogonal'):
        load_licorne_model(p, axis_map=2 * np.eye(3))
    # Licorne x -> sample x, Licorne z -> sample y, Licorne y -> sample -z:
    # theta = 90, phi = phi_L lies at turns (cos phi_L, 0, -sin phi_L)
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    st = load_licorne_model(p, axis_map=R)
    L2 = st.layers[2]
    u = unit(L2.MSLD_theta, L2.MSLD_phi)
    assert np.allclose(u, R @ licorne_direction(4.72957, 90), atol=1e-12)
    bad = tmp_path / 'parameters.m'
    bad.write_text(p.read_text().replace("'tanh'", "'NC'"))
    with pytest.raises(NotImplementedError):
        load_licorne_model(bad, axis_map=np.eye(3))


# -- 8.5 the angle rule against Licorne ---------------------------------------
def _slab_dirs(st):
    return unit([s.MSLD_theta for s in st.sublayers],
                [s.MSLD_phi for s in st.sublayers])


def _licorne_dirs(ref):
    return np.array([licorne_direction(p, t) for p, t in ref[:, 5:7]])


@pytest.mark.parametrize('name', V127)
def test_v127_profiles(name):
    """Licorne 1.2.7 exports with angles: unclipped windows (N = 3 and 10),
    values and geometry."""
    d = DATA / name
    st = load_licorne_model(d / 'parameters.m', d / 'profile.dat',
                            axis_map=np.eye(3))
    st.build_sublayers()
    ref = read_table(d / 'profile_sublayers.dat')
    th, re_, im_, rho, _, _ = cols(st)
    assert len(th) + 1 == len(ref)
    assert np.allclose(th, ref[:-1, 1], rtol=1e-5, atol=0)
    assert rel(re_, ref[:-1, 2]) <= 1e-5
    assert np.allclose(rho, ref[:-1, 4], rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize('name', V127)
def test_v127_angle_rule(name):
    """Licorne's per-sublayer angles are the 'step' rule exactly, with the
    tie at x = 0 going to the upper layer and only the fronting and
    substrate (no angle in Licorne) falling back: step_fallback False.
    The default fallback differs only in the window slabs on the side of
    an interior non-magnetic layer, which take the other layer's angle."""
    d = DATA / name
    ref = read_table(d / 'profile_sublayers.dat')[:-1]
    st = load_licorne_model(d / 'parameters.m', d / 'profile.dat',
                            axis_map=np.eye(3))
    st.step_fallback = False
    st.build_sublayers()
    assert np.max(np.abs(_slab_dirs(st) - _licorne_dirs(ref))) < 1e-7
    st.step_fallback = True
    st.build_sublayers()
    lic = _licorne_dirs(ref)
    expect = lic.copy()
    n = len(st.layers)
    for j in range(n - 1):
        if window_slabs(st, j) is None:
            continue
        i, x = window_slabs(st, j)
        for side, k, other in ((x <= 1e-9, j, j + 1), (x > 1e-9, j + 1, j)):
            if 0 < k < n - 1 and st.layers[k].MSLD_rho == 0:
                o = st.layers[other]
                expect[i[side]] = unit(o.MSLD_theta, o.MSLD_phi) \
                    if o.MSLD_rho != 0 else unit(0.0, 0.0)
    assert np.max(np.abs(_slab_dirs(st) - expect)) < 1e-7
    assert np.max(np.abs(expect - lic)) > 0.01


@pytest.mark.parametrize('case', ['antiparallel', 'perpendicular'])
def test_fixture2_angle_rule(case):
    """Plan fixture 2: 50 / 50 A bilayer, rho = 5e-6, theta = 90,
    phi = 0 / 180 (antiparallel) or 0 / 90, sigma_L = 10, tanh, N = 6."""
    d = DATA / 'fixture2' / case
    if not (d / 'profile_sublayers.dat').exists():
        pytest.skip('needs a Licorne export, see tests/data/licorne/README.md')
    prof = d / 'profile.dat'
    st = load_licorne_model(d / 'parameters.m',
                            prof if prof.exists() else None,
                            axis_map=np.eye(3))
    st.build_sublayers()
    ref = read_table(d / 'profile_sublayers.dat')[:-1]
    rho = cols(st)[3]
    # rho following the smeared profile, no dip: not 'vector'
    assert np.allclose(rho, ref[:, 4], rtol=1e-5, atol=1e-12)
    # the angle jumping at x = 0, no intermediate angle: 'step', not 'angle'
    assert np.max(np.abs(_slab_dirs(st) - _licorne_dirs(ref))) < 1e-7


# -- 8.6 physics invariants in the Licorne scheme ------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_sharp_matches_rms_and_fresnel(mode):
    a = magnetic('licorne', mode, rough=0.0)
    b = magnetic('rms', mode, rough=0.0)
    assert rel(a.reflectivities(Q, LAB), b.reflectivities(Q, LAB)) <= 1e-13
    L = [S.Layer('vac', 0), S.Layer('Si', 0, 2.07e-6)]
    st = S.Stack(L, roughness_scheme='licorne', magnetic_smearing=mode)
    q = np.sqrt(Q**2 - 16 * np.pi * 2.07e-6 + 0j)
    assert rel(st.reflectivity(Q, X, X), np.abs((Q - q) / (Q + q))**2) <= 1e-12


@pytest.mark.parametrize('mode', MODES)
def test_spin_flip_parallel_and_perpendicular(mode):
    par = magnetic('licorne', mode, thetas=(0, 0, 0)).reflectivities(Q, LAB)
    assert np.max(par[:, 1]) < 1e-25 and np.max(par[:, 2]) < 1e-25
    perp = magnetic('licorne', mode, thetas=(0, 0, 0), rot=0.25)
    R = perp.reflectivities(Q, LAB)
    assert np.max(R[:, 1]) > 1e-4
    # M along y, P along x: R++ + R+- = (|r_u|^2 + |r_d|^2)/2
    assert rel(R[:, 0] + R[:, 1], 0.5 * (par[:, 0] + par[:, 3])) <= 1e-11


@pytest.mark.parametrize('mode', MODES)
def test_collinear_antiparallel_spin_flip(mode):
    """Fe at 0, Co at 1/2 turn, P along x: 'step' and 'vector' stay
    collinear (no spin flip), 'angle' turns M through 90 deg."""
    L = [S.Layer('vac', 0),
         S.Layer('Fe', 80, 8e-6, 0, 5e-6, 0.0, 8.0, 'tanh', 6),
         S.Layer('Co', 60, 6e-6, 0, 4e-6, 0.5, 8.0, 'tanh', 6),
         S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 8.0, 'tanh', 6)]
    st = S.Stack(L, roughness_scheme='licorne', magnetic_smearing=mode)
    R = st.reflectivities(Q, LAB)
    sf = np.max(R[:, 1:3] / R[:, [0]])
    if mode == 'angle':
        assert sf > 1e-6
    else:
        assert sf < 1e-20


@pytest.mark.parametrize('mode', MODES)
def test_reciprocity_flux_rotation(mode):
    st = magnetic('licorne', mode, fe_img=0.0)
    R = st.reflectivities(Q, LAB)
    assert np.max(np.abs(R[:, 1] - R[:, 2])) < 1e-15
    # total reflection below every critical edge: unitary
    b = st.backing
    b.NSLD_real, b.MSLD_rho, b.MSLD_theta = 10e-6, 2e-6, 0.2
    st.build_sublayers()
    q = np.linspace(0.002, 0.019, 200)
    rng = np.random.default_rng(3)
    for _ in range(5):
        v = rng.normal(size=3)
        assert np.max(np.abs(st.reflectivity(q, v / np.linalg.norm(v)) - 1)) \
            <= 1e-13
    # rotating every M and P about the film normal changes nothing
    d = 0.137
    rot = magnetic('licorne', mode, rot=d)
    P = np.array([S.cos_turns(d), S.sin_turns(d), 0.0])
    st = magnetic('licorne', mode)
    assert rel(rot.reflectivities(Q, [(P, P), (P, -P)]),
               st.reflectivities(Q, [(X, X), (X, -X)])) <= 1e-11


def test_thin_sublayers():
    """Fixture 1 has 0.000283 A sublayers: R is finite down to small Q and
    within (Q sigma)^2 of the same model with that interface made sharp."""
    q = np.concatenate([np.geomspace(1e-6, 1e-3, 30), Q])
    nsf = [(X, X), (-X, -X)]
    st = fixture1()
    assert min(s.thickness for s in st.sublayers) < 3e-4
    R = st.reflectivities(q, nsf)
    assert np.all(np.isfinite(R))
    st.layers[1].roughness_sigma = 0.0
    st.build_sublayers()
    assert rel(R, st.reflectivities(q, nsf)) < 1e-6


@pytest.mark.xfail(strict=True, reason='model.stack._slab_matrices has no '
                   'sinh(Sd)/S regularisation: S == 0 exactly gives 0/0')
def test_slab_at_its_own_critical_Q():
    L = [S.Layer('vac', 0), S.Layer('a', 10, 4e-6 / np.pi),
         S.Layer('s', 0, 2e-6)]
    st = S.Stack(L, roughness_scheme='licorne')
    q = np.array([np.sqrt(16 * np.pi * L[1].NSLD_real)])
    with np.errstate(all='ignore'):
        assert np.isfinite(st.reflectivity(q, X, X)).all()


# -- 8.7 cross-model check -----------------------------------------------------
def _thick(scheme, sigma, N, tail=3.0):
    L = [S.Layer('vac', 0),
         S.Layer('A', 200, 6e-6, -1e-9, 2e-6, 0, sigma, 'erf', N),
         S.Layer('B', 150, 2e-6, 0, 0, 0, sigma, 'erf', N),
         S.Layer('C', 250, 4e-6, 0, 1e-6, 0, sigma, 'erf', N),
         S.Layer('s', 0, 2.07e-6, 0, 0, 0, sigma, 'erf', N)]
    st = S.Stack(L, tail=tail, roughness_scheme=scheme)
    return st.reflectivities(Q2, [(X, X), (-X, -X)])


Q2 = np.linspace(0.005, 0.25, 600)


def test_cross_model(monkeypatch):
    """Thick layers, sigma_rms = 0.96369 sigma_L (erf): Licorne's R
    converges in N to a limit that differs from the rms model by the
    truncation at +-2.09 sigma_L alone (it vanishes when the window is
    widened).  Residuals are printed (pytest -s)."""
    sL = 8.0
    ref = _thick('rms', rg.sigma_rms_from_licorne(sL, 'erf'), 400, tail=6.0)
    R = {N: _thick('licorne', sL, N) for N in (6, 24, 96, 384)}
    lo = Q2 < 0.1
    print('\nN    max|R_lic/R_rms - 1|   (Q < 0.1)    |R_N/R_384 - 1|')
    for N, RN in R.items():
        d = np.abs(RN / ref - 1)
        print('%-4d %-22.4g %-12.4g %.3g' % (N, d.max(), d[lo].max(),
                                             np.abs(RN / R[384] - 1).max()))
    conv = [np.abs(R[N] / R[384] - 1).max() for N in (6, 24, 96)]
    assert conv[0] > conv[1] > conv[2] and conv[2] < 2e-3
    floor = np.abs(R[384] / ref - 1)
    assert 0.1 < floor.max() < 0.3 and floor[lo].max() < 0.03
    monkeypatch.setitem(rg._DELTA, 'erf', erfinv(0.99999))
    wide = np.abs(_thick('licorne', sL, 384) / ref - 1)
    print('window to the 99.999 %% point: %.3g (%.3g for Q < 0.1)'
          % (wide.max(), wide[lo].max()))
    assert wide.max() < 1e-3 and wide.max() < floor.max() / 100
