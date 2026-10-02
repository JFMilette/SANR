"""Physics of model.stack and the pair helpers of model.polarisation.

Run from the repo root: python -m pytest tests
"""
import json
import pathlib
import sys
import warnings

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import model.stack as S                                        # noqa: E402
from model.polarisation import (channel_pair, default_vectors,  # noqa: E402
                                in_plane, pair_vectors, vectors_pair)
import supermatrix as sm                                       # noqa: E402

Q = np.linspace(0.004, 0.12, 500)
PH = (0.08, -0.17)                   # MSLD_phi of Fe and Co, turns
X = np.array([1.0, 0, 0])
Z = np.array([0, 0, 1.0])
LAB = ['++', '+-', '-+', '--']
NAMES = LAB + ['+', '-']
FRONT_MAG = (1e-6, 0.4e-6, 0.1)      # NSLD, rho, theta of a magnetic fronting


def mk(phis=(0, 0), front=(0, 0, 0), mode='vector', rough=4.0, fe_img=-1e-8):
    """vac / Fe 80 / Cr 25 / Co 60 / Si."""
    L = [S.Layer('front', 0, front[0], 0, front[1], front[2]),
         S.Layer('Fe', 80, 8e-6, fe_img, 5e-6, 0.10, rough, MSLD_phi=phis[0]),
         S.Layer('Cr', 25, 3e-6, 0, 0, 0.0, rough),
         S.Layer('Co', 60, 6e-6, 0, 4e-6, 0.35, rough, MSLD_phi=phis[1]),
         S.Layer('Si', 0, 2.07e-6, 0, 0, 0, rough)]
    st = S.Stack(L)
    st.magnetic_smearing = mode
    st.build_sublayers()
    return st


def pairs(names, n=X):
    return [channel_pair(c, n) for c in names]


def rel(a, b):
    """max relative difference where both are finite and b != 0; the NaN
    patterns must agree."""
    assert np.array_equal(np.isnan(a), np.isnan(b)), 'NaN patterns differ'
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    return float(np.max(np.abs(a[ok] - b[ok]) / np.abs(b[ok])))


def rand_P(rng):
    v = rng.normal(size=3)
    return v * rng.uniform(0.5, 1) / np.linalg.norm(v)


# -- Halperin ---------------------------------------------------------------
def test_only_inplane_part_is_seen():
    a = mk(PH).reflectivities(Q, pairs(LAB))
    b = mk()
    b.layers[1].MSLD_rho *= np.cos(2*np.pi*PH[0])
    b.layers[3].MSLD_rho *= np.cos(2*np.pi*PH[1])
    b.build_sublayers()
    assert rel(a, b.reflectivities(Q, pairs(LAB))) <= 1e-12


def test_out_of_plane_is_nonmagnetic():
    R = mk((0.25, -0.25)).reflectivities(Q, pairs(LAB))
    assert np.max(R[:, 1]) < 1e-25                      # no spin flip
    assert np.array_equal(R[:, 0], R[:, 3])             # R++ == R--


@pytest.mark.parametrize('mode', ['vector', 'angle'])
def test_slabs_sample_the_profile(mode):
    """What each slab puts into the transfer matrix is the profile at its
    centre."""
    st = mk(PH, mode=mode)
    z = st.windows()[0][0] + np.concatenate(
        [[0], np.cumsum([s.thickness for s in st.sublayers])])
    _, rho, theta, phi = st.profile(0.5*(z[:-1] + z[1:]))
    prof = rho * S.cos_turns(phi) * np.exp(2j*np.pi*theta)
    slab = np.array([S.Stack.inplane_rho(s)*np.exp(2j*np.pi*s.MSLD_theta)
                     for s in st.sublayers])
    assert np.max(np.abs(slab - prof)) <= 1e-18
    mz = np.array([s.MSLD_rho*S.sin_turns(s.MSLD_phi) for s in st.sublayers])
    assert np.max(np.abs(mz - rho*S.sin_turns(phi))) <= 1e-18


# -- arbitrary polarisation --------------------------------------------------
def test_matches_supermatrix_eq5():
    """Independent oracle: Ruehm, Toperverg & Dosch, PRB 60, 16073, Eq. 5."""
    rng = np.random.default_rng(0)
    st = mk(PH, rough=0.0)
    lay = [(complex(l.NSLD_real, l.NSLD_img), S.Stack.inplane_rho(l),
            in_plane(2*np.pi*l.MSLD_theta), l.thickness)
           for l in st.sublayers]
    Rh = [sm.R_hat(q/2, lay, 2.07e-6) for q in Q]
    for _ in range(10):
        P0, P = rand_P(rng), rand_P(rng)
        assert rel(st.reflectivity(Q, P0, P),
                   np.array([sm.refl_eq5(R, P0, P) for R in Rh])) <= 1e-11


@pytest.mark.parametrize('front', [(0, 0, 0), FRONT_MAG], ids=['nonmag', 'mag'])
@pytest.mark.parametrize('qf', [False, True])
@pytest.mark.parametrize('res', [False, True])
def test_batched_equals_single(front, qf, res):
    st = mk(PH, front)
    st.q_in_fronting, st.resolution['enabled'] = qf, res
    st.background = 2e-7
    n = st.fronting_direction() if front[1] else X
    p = pairs(NAMES, n)
    R = st.reflectivities(Q, p)
    single = np.stack([st.reflectivity(Q, *pp) for pp in p], -1)
    assert rel(R, single) <= 1e-14


@pytest.mark.parametrize('res', [False, True])
def test_no_analyser_counts_background_once(res):
    st = mk(PH)
    st.resolution['enabled'] = res
    st.background = 3e-7
    R = st.reflectivities(Q, pairs(NAMES))
    assert rel(R[:, 4], R[:, 0] + R[:, 1] - st.background) <= 1e-12
    assert rel(R[:, 5], R[:, 3] + R[:, 2] - st.background) <= 1e-12


@pytest.mark.parametrize('res', [False, True])
def test_batched_equals_single_magnetic_backing(res):
    """Magnetic substrate at an angle to both films: its eigenbasis enters
    the boundary condition; batching must not change any channel."""
    st = mk(PH)
    b = st.backing
    b.MSLD_rho, b.MSLD_theta, b.MSLD_phi = 1.5e-6, 0.3, 0.05
    st.build_sublayers()
    st.resolution['enabled'] = res
    st.background = 2e-7
    p = pairs(NAMES)
    R = st.reflectivities(Q, p)
    single = np.stack([st.reflectivity(Q, *pp) for pp in p], -1)
    assert rel(R, single) <= 1e-14
    assert rel(R[:, 4], R[:, 0] + R[:, 1] - st.background) <= 1e-12


@pytest.mark.parametrize('res', [False, True])
def test_scale_then_background(res):
    st = mk(PH)
    st.resolution['enabled'] = res
    R1 = st.reflectivities(Q, pairs(NAMES))
    st.scale, st.background = 0.97, 3e-7
    R2 = st.reflectivities(Q, pairs(NAMES))
    assert rel(R2, 0.97 * R1 + 3e-7) <= 1e-13
    # R+ still holds the background once
    assert rel(R2[:, 4], R2[:, 0] + R2[:, 1] - 3e-7) <= 1e-12


def test_stack_parameters_are_free_parameters():
    st = mk(PH)
    assert st.free_parameters() == []
    st.fit_entry('background')['vary'] = True
    st.layers[1].fit['thickness'] = {'vary': True, 'min': 70, 'max': 90}
    st.fit['scale'] = {'vary': True, 'min': 0.9, 'max': 1.1}
    assert [(i, a) for i, a, *_ in st.free_parameters()] == \
        [(1, 'thickness'), (None, 'scale'), (None, 'background')]
    assert st.owner(None) is st and st.owner(1) is st.layers[1]
    st.scale = 0.95
    back = S.Stack.from_dict(json.loads(json.dumps(st.to_dict())))
    assert back.scale == 0.95 and back.fit == st.fit
    # files written before scale existed
    d = st.to_dict()
    del d['scale'], d['fit']
    old = S.Stack.from_dict(d)
    assert old.scale == 1.0 and old.free_parameters()[1:] == []


def test_efficiency():
    st = mk(PH)
    R = st.reflectivities(Q, [(0.95*X, X), (X, X), (-X, X)])
    assert rel(R[:, 0], 0.975*R[:, 1] + 0.025*R[:, 2]) <= 1e-12


@pytest.mark.parametrize('qf', [False, True])
def test_magnetic_fronting(qf):
    st = mk(PH, FRONT_MAG, rough=0.0)
    st.layers[0].MSLD_phi = 0.05
    st.q_in_fronting = qf
    st.build_sublayers()
    lab = st._lab_channels(Q)                  # ++ +- -+ -- along M_f
    m = st.fronting_direction()
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        assert rel(st.reflectivity(Q, m, -m), lab[:, 1]) <= 1e-12
        st.reflectivity(Q, m)
        n_along = len(W)
        # P along z is transverse: it dephases to an unpolarised beam
        Rz = st.reflectivity(Q, Z, Z)
        n_trans = len(W) - n_along
    assert rel(Rz, lab.sum(1)/4) <= 1e-12
    assert n_along == 0 and n_trans == 1


def test_total_reflection_is_unitary():
    """Vacuum / non-absorbing film / magnetic backing below both of its
    critical edges: every incident polarisation is totally reflected."""
    rng = np.random.default_rng(1)
    st = mk(PH, fe_img=0.0)
    b = st.backing
    b.NSLD_real, b.MSLD_rho, b.MSLD_theta = 10e-6, 2e-6, 0.2
    st.build_sublayers()
    q = np.linspace(0.002, 0.019, 200)         # < sqrt(16 pi 8e-6) = 0.0200
    for _ in range(10):
        assert np.max(np.abs(st.reflectivity(q, rand_P(rng)) - 1)) <= 1e-14


# -- serialisation and helpers ------------------------------------------------
def test_to_dict_round_trip():
    st = mk(PH, FRONT_MAG)
    st.background, st.q_in_fronting = 1e-7, True
    back = S.Stack.from_dict(json.loads(json.dumps(st.to_dict())))
    assert back.to_dict() == st.to_dict()


@pytest.mark.parametrize('f', ['session_Cr2Te3.json', 'session_Cr2Te3_2.json'])
def test_example_models_load(f):
    with open(HERE.parent / f) as fh:
        st = S.Stack.from_dict(json.load(fh))
    st.build_sublayers()
    assert np.all(np.isfinite(st.reflectivities(Q, pairs(LAB))))


def test_pair_helpers():
    n = in_plane(0.5)
    P0, P = channel_pair('-+', n)
    assert np.allclose(P0, -n) and np.allclose(P, n)
    assert channel_pair('+', n)[1] is None
    with pytest.raises(ValueError):
        channel_pair('+x', n)
    vec = default_vectors()
    assert vec['+'] == [[1, 0, 0], [0, 0, 0]]
    assert vec['-'] == [[-1, 0, 0], [0, 0, 0]]
    assert vec['+-'] == [[1, 0, 0], [-1, 0, 0]]
    P0, P = vectors_pair([1, 0, 0], [0, 0, 0])      # Pa = 0: no analyser
    assert P is None and pair_vectors(P0, P) == [[1, 0, 0], [0, 0, 0]]
    assert mk().fronting_direction() is None
    assert np.allclose(mk(front=FRONT_MAG).fronting_direction(),
                       in_plane(0.2*np.pi))
    with pytest.raises(ValueError):
        mk().reflectivity(Q, [1, 1, 0])             # |P| > 1


def test_tof_resolution_matches_licorne_script():
    """sigma_Q of the three-angle TOF setup, as in the Licorne script."""
    st = mk()
    st.resolution.update(enabled=True, mode='tof')
    q = np.linspace(0.005, 0.2, 400)
    ref = np.zeros_like(q)
    for band, th, dth in ((q < 0.04, 0.006, 3e-4),
                          ((q >= 0.04) & (q <= 0.12), 0.010, 5e-4),
                          (q > 0.12, 0.017, 5e-4)):
        lam = 4*np.pi*np.sin(th) / q[band]
        ref[band] = q[band] * np.sqrt((dth/th)**2 + (0.005/lam)**2)
    assert rel(st.resolution_sigma(q), ref) <= 1e-14


def test_resolution_grid_follows_local_sigma():
    """The adaptive grid matches the ideal R at every node (brute force)."""
    st = mk()
    for mode in ('mono', 'tof'):
        st.resolution.update(enabled=True, mode=mode)
        p = pairs(['++', '--'])
        sig = st.resolution_sigma(Q)
        x = np.linspace(-S.RES_SPAN, S.RES_SPAN, S.RES_NODES)
        w = np.exp(-0.5 * x**2)
        Qn = np.maximum(np.abs(Q[:, None] + x * sig[:, None]), 1e-6)
        Ri = st._trace(Qn.ravel(), p).reshape(len(Q), len(x), -1)
        brute = (Ri * w[None, :, None]).sum(1) / w.sum()
        assert rel(st.reflectivities(Q, p), brute) <= 1e-2


@pytest.mark.parametrize('res_mode', ['mono', 'tof'])
@pytest.mark.parametrize('magnetic_backing', [False, True])
def test_resolution_grid_accuracy(res_mode, magnetic_backing, monkeypatch):
    """The coarse grid refined at the critical edges stays within a small
    fraction of a 1 % error bar of a 16x finer uniform grid."""
    st = mk(PH)
    if magnetic_backing:
        b = st.backing
        b.MSLD_rho, b.MSLD_theta, b.MSLD_phi = 1.5e-6, 0.3, 0.05
        st.build_sublayers()
    st.resolution.update(enabled=True, mode=res_mode)
    q = np.linspace(0.004, 0.15, 600)
    p = pairs(NAMES)
    R = st.reflectivities(q, p)
    monkeypatch.setattr(S, 'RES_GRID_STEP', 16 * S.RES_GRID_STEP)
    monkeypatch.setattr(S.Stack, 'critical_edges', lambda self: [])
    ref = st.reflectivities(q, p)
    assert np.max(np.abs(R - ref) / (0.01 * ref + 1e-7)) < 0.3


def test_critical_edges():
    st = mk(PH)                                     # vacuum / ... / Si
    assert st.critical_edges() == pytest.approx([np.sqrt(16 * np.pi * 2.07e-6)])
    b = st.backing
    b.MSLD_rho = 1e-6                               # in plane: two edges
    assert st.critical_edges() == pytest.approx(
        [np.sqrt(16 * np.pi * 1.07e-6), np.sqrt(16 * np.pi * 3.07e-6)])
