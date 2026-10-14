"""The compiled transfer-matrix product (model/_transfer.pyx) against the
numpy one of model.stack: general and collinear paths, edge cases.

Run from the repo root: python -m pytest tests  (skipped when the extension
is not built: python setup.py build_ext --inplace)
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import model.stack as S                                        # noqa: E402

T = pytest.importorskip('model._transfer')

Q = np.linspace(0.002, 0.25, 400)


def numpy_product(st, Q):
    old = S.USE_COMPILED
    S.USE_COMPILED = False
    try:
        return st.build_transfer_matrix(Q)
    finally:
        S.USE_COMPILED = old


def mk(thetas, rhos=(5e-6, 0, 4e-6), img=-1e-8, rough=4.0):
    """vac / Fe 80 / Cr 25 / Co 60 / Si, M of Fe and Co at `thetas`."""
    L = [S.Layer('front', 0, 0, 0, 0, 0),
         S.Layer('Fe', 80, 8e-6, img, rhos[0], thetas[0], rough),
         S.Layer('Cr', 25, 3e-6, 0, rhos[1], thetas[1], rough),
         S.Layer('Co', 60, 6e-6, 0, rhos[2], thetas[2], rough),
         S.Layer('Si', 0, 2.07e-6, 0, 0, 0, rough)]
    st = S.Stack(L)
    st.build_sublayers()
    return st


def close(st, Q, tol=1e-11):
    a = st.build_transfer_matrix(Q)
    b = numpy_product(st, Q)
    scale = np.abs(b).max(axis=(1, 2))[:, None, None]
    assert np.all(np.abs(a - b) <= tol * scale)


@pytest.mark.parametrize('thetas', [(0.1, 0.0, 0.35),     # general
                                    (0.1, 0.0, 0.1),      # collinear
                                    (0.1, 0.0, 0.6),      # antiparallel
                                    (0.0, 0.0, 0.0)])
def test_matches_numpy(thetas):
    st = mk(thetas)
    expect = None if thetas[0] != thetas[2] and \
        abs((thetas[2] - thetas[0]) % 1 - 0.5) > 1e-9 else 'collinear'
    rho = [st.inplane_rho(l) for l in st.sublayers]
    th = [l.MSLD_theta for l in st.sublayers]
    assert (T.axis(rho, th) is not None) == (expect == 'collinear')
    close(st, Q)


def test_nonmagnetic_and_absorbing():
    close(mk((0, 0, 0), rhos=(0, 0, 0), img=-5e-7), Q)


def test_complex_q_and_critical_edge():
    """Q from sqrt(Q2 + 0j) with Q2 < 0, and Q exactly at Si's edge (S = 0
    in a slab)."""
    st = mk((0.1, 0.0, 0.35), rough=0.0)
    edge = np.sqrt(16*np.pi*2.07e-6)
    q2 = np.concatenate([np.linspace(-1e-3, 0.01, 50), [edge**2]])
    close(st, np.sqrt(q2 + 0j))


def test_thick_evanescent_layer():
    """exp(S d) large below the edge of a thick dense layer."""
    L = [S.Layer('front', 0, 0, 0, 0, 0),
         S.Layer('Ni', 5000, 9.4e-6, 0, 2e-6, 0.2, 0.0),
         S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 0.0)]
    st = S.Stack(L)
    st.build_sublayers()
    close(st, np.linspace(0.002, 0.03, 100), tol=1e-9)


def test_reflectivities_match():
    st = mk((0.1, 0.0, 0.35))
    pairs = [(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])),
             (np.array([1.0, 0, 0]), None)]
    a = st.reflectivities(Q, pairs)
    old = S.USE_COMPILED
    S.USE_COMPILED = False
    try:
        b = st.reflectivities(Q, pairs)
    finally:
        S.USE_COMPILED = old
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=0)
