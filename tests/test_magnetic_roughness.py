"""Layer.MSLD_sigma: the magnetic profile's own roughness (model.stack
MAGNETIC ROUGHNESS), in the model, a fit and the Simulation tab.

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

import model.stack as S                                         # noqa: E402
from fit.de.de_fit import run_de                                # noqa: E402
from fit.problem import FitProblem                              # noqa: E402

X = np.array([1.0, 0, 0])
Q = np.linspace(0.01, 0.15, 80)
Z = np.linspace(-40, 160, 2001)


def film(sig=4.0, msig=None, mode='vector', order='ordered', model='tanh'):
    """vacuum / Fe 60 / Cr 20 / Co 40 / Si; Fe and Co's interfaces get the
    magnetic sigma msig."""
    L = [S.Layer('v'),
         S.Layer('Fe', 60, 8e-6, 0, 5e-6, 0.1, sig, model, 8, MSLD_sigma=msig),
         S.Layer('Cr', 20, 3e-6, 0, 0, 0.0, sig, model, 8, MSLD_sigma=msig),
         S.Layer('Co', 40, 6e-6, 0, 4e-6, 0.35, sig, model, 8, MSLD_phi=0.05,
                 MSLD_sigma=msig),
         S.Layer('Si', 0, 2.07e-6, 0, 0, 0, sig, model, 8)]
    return S.Stack(L, magnetic_smearing=mode, interface_order=order)


@pytest.mark.parametrize('order', S.INTERFACE_ORDERS)
@pytest.mark.parametrize('mode', S.SMEARING_MODES)
def test_each_profile_takes_its_sigma(mode, order):
    """The NSLD is that of the structural sigma, the magnetic profile that
    of a stack whose roughness is the magnetic sigma everywhere it is
    given."""
    st = film(4.0, 9.0, mode, order)
    nsld, rho, theta, phi = st.profile(Z)
    assert np.array_equal(nsld, film(4.0, None, mode, order).profile(Z)[0])
    mag = film(4.0, None, mode, order)
    for l in mag.layers[1:4]:
        l.roughness_sigma = 9.0
    for a, b in zip((rho, theta, phi), mag.profile(Z)[1:]):
        assert np.array_equal(a, b)


@pytest.mark.parametrize('mode', S.SMEARING_MODES)
def test_slabs_sample_the_profile(mode):
    st = film(3.0, 8.0, mode)
    st.build_sublayers()
    w = st.windows()
    assert w[0][1] - w[0][0] == pytest.approx(2 * 3.0 * 8.0)  # the wider sigma
    e = w[0][0] + np.concatenate([[0], np.cumsum([s.thickness
                                                  for s in st.sublayers])])
    nsld, rho, theta, phi = st.profile(0.5 * (e[:-1] + e[1:]))
    assert np.max(np.abs([complex(s.NSLD_real, s.NSLD_img)
                          for s in st.sublayers] - nsld)) <= 1e-18
    prof = rho * S.cos_turns(phi) * np.exp(2j * np.pi * theta)
    slab = np.array([S.Stack.inplane_rho(s) * np.exp(2j * np.pi * s.MSLD_theta)
                     for s in st.sublayers])
    assert np.max(np.abs(slab - prof)) <= 1e-18


def test_none_changes_nothing():
    for scheme in S.ROUGHNESS_SCHEMES:
        a, b = film(4.0, None), film(4.0, 4.0)
        a.roughness_scheme = b.roughness_scheme = scheme
        for x, y in zip(a.profile(Z), b.profile(Z)):
            assert np.allclose(x, y, rtol=1e-15, atol=0)


def licorne(sig, msig, mode, thin=12.0):
    st = film(sig, msig, mode)
    st.roughness_scheme = 'licorne'
    st.layers[2].thickness = thin          # Cr: clipped and renormalised
    return st


@pytest.mark.parametrize('mode', S.SMEARING_MODES)
def test_licorne_magnetic_window(mode):
    """In the Licorne scheme the NSLD is the Licorne profile of the
    structural sigma, the magnetic profile the Licorne profile (truncated,
    clipped, renormalised) of the magnetic sigma."""
    st = licorne(4.0, 9.0, mode)
    nsld, rho, theta, phi = st.profile(Z)
    assert np.array_equal(nsld, licorne(4.0, None, mode).profile(Z)[0])
    mag = licorne(4.0, None, mode)
    for l in mag.layers[1:4]:
        l.roughness_sigma = 9.0
    for x, y in zip((rho, theta, phi), mag.profile(Z)[1:]):
        assert np.allclose(x, y, rtol=1e-14, atol=1e-20)
    # Cr's renormalisation shows: the magnetic window is not the rms one
    rms = film(4.0, 9.0, mode)
    assert np.max(np.abs(rho - rms.profile(Z)[1])) > 1e-8


@pytest.mark.parametrize('mode', S.SMEARING_MODES)
@pytest.mark.parametrize('sigmas', [(4.0, 9.0), (9.0, 2.0), (0.0, 5.0)])
def test_licorne_slabs_sample_the_profile(mode, sigmas):
    """Both windows' slabs together: each slab is the profile at its centre
    and each window edge (a jump) is a slab edge."""
    st = licorne(*sigmas, mode)
    st.build_sublayers()
    e = st.windows()[0][0] + np.concatenate(
        [[0], np.cumsum([s.thickness for s in st.sublayers])])
    nsld, rho, theta, phi = st.profile(0.5 * (e[:-1] + e[1:]))
    assert np.max(np.abs([complex(s.NSLD_real, s.NSLD_img)
                          for s in st.sublayers] - nsld)) <= 1e-18
    assert np.max(np.abs([s.MSLD_rho for s in st.sublayers] - rho)) <= 1e-18
    assert np.max(np.abs(np.exp(2j * np.pi * np.array(
        [s.MSLD_theta for s in st.sublayers])) - np.exp(2j * np.pi * theta))) \
        <= 1e-9
    for w in st._licorne_interfaces():
        m = w['mag']
        if m is None:
            continue
        for edge in (-m['la_n'], m['lb_n'], -m['la'], m['lb']):
            assert np.min(np.abs(e - (w['Z'] + edge))) < 1e-9
    assert np.all(np.isfinite(st.reflectivities(Q, [(X, X), (X, -X)])))


def test_saved_and_fitted():
    st = film(4.0, None)
    assert 'MSLD_sigma' not in [a for _, a, *_ in
                                (p for p in st.free_parameters())]
    st.layers[1].fit['MSLD_sigma'] = {'vary': True, 'min': 0, 'max': 15}
    assert st.free_parameters() == []                  # not separate: no
    st.layers[1].MSLD_sigma = 6.0
    assert st.free_parameters() == [(1, 'MSLD_sigma', 6.0, 0, 15)]
    back = S.Stack.from_dict(st.to_dict())
    assert back.layers[1].MSLD_sigma == 6.0 and back.layers[2].MSLD_sigma \
        is None
    d = st.to_dict()
    for l in d['layers']:
        del l['MSLD_sigma']                            # a file from before
    assert S.Stack.from_dict(d).layers[1].MSLD_sigma is None
    # DE finds Fe's magnetic sigma from the polarised channels
    truth = film(4.0, None)
    truth.layers[1].MSLD_sigma = 9.0
    pairs = [(X, X), (-X, -X)]
    R = truth.reflectivities(Q, pairs)
    data = [dict(Q=Q, R=R[:, j], dR=0.01 * R[:, j], P0=P0, P=P)
            for j, (P0, P) in enumerate(pairs)]
    x, cost, *_ = run_de(FitProblem(st, data), maxiter=40, seed=0)
    assert abs(x[0] - 9.0) < 1e-2 and cost < 1e-4


def test_simulation_tab_row():
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tabs import SimulationTab
    sim = SimulationTab()
    sim.restore_state(film(4.0, None).to_dict())
    sim.list.setCurrentRow(1)                                    # Fe
    ed = sim.editor
    val = ed.rows['MSLD_sigma'][0]
    assert not ed.msig_own.isChecked() and not val.isEnabled()
    assert val.value() == 4.0                          # follows Roughness σ
    ed.msig_own.setChecked(True)
    assert val.isEnabled() and sim.stack.layers[1].MSLD_sigma == 4.0
    val.setValue(7.5)
    assert sim.stack.layers[1].MSLD_sigma == 7.5
    assert sim.stack.layers[1].roughness_sigma == 4.0
    sim.list.setCurrentRow(2)
    sim.list.setCurrentRow(1)                          # reloaded
    assert ed.msig_own.isChecked() and val.value() == 7.5
    ed.msig_own.setChecked(False)
    assert sim.stack.layers[1].MSLD_sigma is None and val.value() == 4.0
    sim.list.setCurrentRow(0)                          # the fronting
    assert not ed.msig_own.isEnabled()
