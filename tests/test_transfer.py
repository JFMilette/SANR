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


# ---- products: several slab sets, the slabs they share at the top once

def _node_slabs(st, target, factors):
    """The slab arrays of st with layer `target` at each thickness factor
    (the thickness-spread nodes of a one-layer target)."""
    d0 = st.layers[target].thickness
    out = []
    for f in factors:
        st.layers[target].thickness = d0 * f
        st.build_sublayers()
        out.append(st._slab_arrays[1])
    st.layers[target].thickness = d0
    st.build_sublayers()
    return out


@pytest.mark.parametrize('thetas', [(0.1, 0.0, 0.35),     # general
                                    (0.1, 0.0, 0.1),      # collinear
                                    (0.1, 0.0, 0.6),      # antiparallel
                                    (0.0, 0.0, 0.0)])
@pytest.mark.parametrize('target', [1, 2, 3])
def test_products_are_the_single_products(thetas, target):
    """Bit for bit, whichever layer the sets differ in (the top one: no
    slab shared; the bottom one: most of them)."""
    st = mk(thetas)
    slabs = _node_slabs(st, target, (0.9, 0.97, 1.0, 1.04, 1.12))
    shared = min(T._shared(slabs[0], s) for s in slabs[1:])
    assert (shared > 0) == (target > 1)
    Qc = np.sqrt(Q**2 + 0j)
    for a, s in zip(T.products(slabs, Qc), slabs):
        assert np.array_equal(a, T.product(*s, Qc))


def test_products_edge_cases():
    Qc = np.sqrt(np.linspace(-1e-4, 0.05, 300) + 0j)     # below the edge too
    col = _node_slabs(mk((0.1, 0.0, 0.1)), 3, (0.9, 1.0, 1.1))
    gen = _node_slabs(mk((0.1, 0.0, 0.35)), 3, (0.9, 1.0, 1.1))
    non = _node_slabs(mk((0, 0, 0), rhos=(0, 0, 0)), 3, (0.9, 1.0, 1.1))
    flip = _node_slabs(mk((0.6, 0.0, 0.1)), 3, (0.9, 1.0, 1.1))
    cases = [col[:1],                       # one set
             [col[1], col[1], col[1]],      # identical sets: all shared
             [col[0], gen[1], col[2]],      # collinear and not, mixed
             [col[0], non[1], flip[2]],     # collinear along different axes
             [col[0], col[1][:3] + (col[1][3] * 2,)],     # nothing shared
             [tuple(a[:5] for a in col[0]), col[1]],      # one is a prefix
             [gen[0], tuple(a[:7] for a in gen[0])],
             [list(map(list, non[0])), non[2]]]           # lists, no M
    for slabs in cases:
        for a, s in zip(T.products(slabs, Qc), slabs):
            assert np.array_equal(a, T.product(*s, Qc))
    assert T.products([], Qc) == []


def test_spread_reflectivity_is_the_per_node_one():
    """A thickness spread through products (Stack._node_matrices) gives
    the reflectivity of one product per node exactly, and the numpy
    product's to rounding."""
    pairs = [(np.array([1.0, 0, 0]), None), (np.array([-1.0, 0, 0]), None),
             (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))]
    for thetas in ((0.1, 0.0, 0.35), (0.1, 0.0, 0.1)):
        for target in ('stack', 3, 2):
            st = mk(thetas)
            st.thickness_spread = 0.04
            st.thickness_spread_nodes = 7
            st.set_spread_target(target)
            R = st.reflectivities(Q, pairs)
            products = T.products
            del T.products            # as a kernel built before it had them
            try:
                assert np.array_equal(st.reflectivities(Q, pairs), R)
            finally:
                T.products = products
            S.USE_COMPILED = False
            try:
                Rn = st.reflectivities(Q, pairs)
            finally:
                S.USE_COMPILED = True
            assert np.allclose(R, Rn, rtol=1e-9, atol=1e-300)
            block = S.NODE_BLOCK
            S.NODE_BLOCK = 7 * 37         # Q in blocks of 37: the same R
            try:
                assert np.array_equal(st.reflectivities(Q, pairs), R)
            finally:
                S.NODE_BLOCK = block


def test_slabs_are_layers_when_read():
    """build_sublayers' slabs: a sequence whose Layers are only made when
    one is read, the values those of the arrays the product multiplies."""
    st = mk((0.1, 0.0, 0.35))
    sub = st.sublayers
    assert sub._layers is None and len(sub) > 10 and sub
    st.build_transfer_matrix(Q)
    st.reflectivities(Q, [(np.array([1.0, 0, 0]), None)])
    assert st.sublayers is sub and sub._layers is None      # never read
    n, rho, theta, d = st._slab_arrays[1]
    assert all(isinstance(l, S.Layer) for l in sub)
    assert sub[0] is sub[0] and sub[-1] is list(sub)[-1]
    assert [l.thickness for l in sub] == list(d)
    assert [complex(l.NSLD_real, l.NSLD_img) for l in sub] == list(n)
    assert [st.inplane_rho(l) for l in sub] == list(rho)
    assert [l.MSLD_theta for l in sub] == list(theta)
    assert [l.name for l in sub[:2]] == ['front_000', 'front_001']
    assert sub == list(sub) and [st.fronting] + sub == [st.fronting, *sub]
    # read back one Layer at a time, the product is the arrays' one
    assert np.array_equal(st.build_transfer_matrix(Q, (list(sub), None)),
                          st.build_transfer_matrix(Q))
    import copy
    import pickle
    for clone in (copy.deepcopy(st), pickle.loads(pickle.dumps(st))):
        assert np.array_equal(clone.build_transfer_matrix(Q),
                              st.build_transfer_matrix(Q))
        assert len(clone.sublayers) == len(sub)
    empty = S.Stack([S.Layer('a'), S.Layer('b', NSLD_real=2e-6)])
    empty.build_sublayers()
    assert len(empty.sublayers) == 0 and not empty.sublayers
    assert list(empty.sublayers) == []
