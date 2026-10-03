"""fit.priors: the densities, their use in the DE cost (its optimum is the
MAP) and in DREAM, the round trips through the session and saved runs, the
tension report and the Simulation tab's prior dialog.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import numpy as np
import pytest
from scipy.integrate import quad

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import model.stack as S                                         # noqa: E402
from fit.de.de_fit import run_de                                # noqa: E402
from fit.dream.dream_fit import (PosteriorProblem, load_run,    # noqa: E402
                                 posterior_warnings, sample, save_result)
from fit.priors import Prior, complete, log_prior, tension      # noqa: E402
from fit.problem import FitProblem                              # noqa: E402

X = np.array([1.0, 0, 0])
Q = np.linspace(0.01, 0.1, 30)


def film():
    """vacuum / Fe 80 A / Si, every parameter fixed."""
    return S.Stack([S.Layer('front', 0, 0, 0, 0, 0),
                    S.Layer('Fe', 80, 8e-6, 0, 2e-6, 0.0, 4.0),
                    S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])


@pytest.mark.parametrize('spec', [
    {'kind': 'gauss', 'mu': 3.0, 'tau': 0.2},
    {'kind': 'gauss', 'mu': 50.0},
    {'kind': 'soft_box', 'a': -1.0, 'b': 2.0, 's': 0.3},
    {'kind': 'soft_box', 'a': 10.0, 'b': 12.0}])
def test_normalised(spec):
    spec = complete(spec)
    lo, hi = -100.0, 200.0
    pts = [spec.get('mu', spec.get('a')), spec.get('b', spec.get('mu'))]
    total = quad(lambda x: np.exp(log_prior(spec, x)), lo, hi,
                 points=pts, limit=200, epsabs=1e-12, epsrel=1e-12)[0]
    assert abs(total - 1) < 1e-8


def test_defaults_and_refusals():
    assert complete({'kind': 'gauss', 'mu': -40.0})['tau'] == 4.0
    assert complete({'kind': 'soft_box', 'a': 0, 'b': 5})['s'] == 0.5
    with pytest.raises(ValueError, match='needs its tolerance'):
        complete({'kind': 'gauss', 'mu': 0.0})
    with pytest.raises(ValueError):
        complete({'kind': 'soft_box', 'a': 2, 'b': 1})
    with pytest.raises(ValueError):
        complete({'kind': 'uniform'})


def test_de_optimum_is_the_map():
    """The background is linear in the model: with Gaussian data and a
    Gaussian prior the optimum is the precision-weighted mean."""
    st = film()
    e = st.fit_entry('background')
    e.update(vary=True, min=0.0, max=1e-5)
    mu, tau = 3e-6, 4e-7
    e['prior'] = {'kind': 'gauss', 'mu': mu, 'tau': tau}
    R0 = st.reflectivities(Q, [(X, None)])[:, 0]        # background 0
    rng = np.random.default_rng(0)
    sig = 2e-7 * (1 + np.arange(len(Q)) % 3)
    R = R0 + 2e-6 + rng.normal(0, sig)
    p = FitProblem(st, [dict(Q=Q, R=R, dR=sig, P0=X, P=None)])
    w = 1 / sig**2
    exact = (np.sum(w * (R - R0)) + mu / tau**2) / (w.sum() + 1 / tau**2)
    x, cost, *_ = run_de(p, maxiter=60, seed=0, tol=1e-10)
    assert abs(x[0] / exact - 1) < 1e-6
    # the cost is chi^2/N plus the prior's share; the residuals are not
    assert cost == pytest.approx(np.mean(p.residuals(x)**2)
                                 + ((x[0] - mu) / tau)**2 / p.npoints)


def test_dream_samples_the_prior():
    """A parameter the data cannot see (the angle of M with rho = 0): its
    posterior is its prior."""
    st = film()
    st.layers[1].MSLD_rho = 0.0
    e = st.layers[1].fit_entry('MSLD_theta')
    e.update(vary=True, min=0.0, max=1.0)
    e['prior'] = {'kind': 'gauss', 'mu': 0.5, 'tau': 0.1}
    R = st.reflectivities(Q, [(X, None)])[:, 0]
    pp = PosteriorProblem(FitProblem(st, [dict(Q=Q, R=R, dR=0.05 * R,
                                                P0=X, P=None)]))
    assert list(pp.priors) == ['Fe.MSLD_theta']
    r = sample(pp, n_gen=4000, seed=1, stop_when_converged=False)
    s = r.samples(0.5)[:, 0]
    assert abs(s.mean() - 0.5) < 0.05 * 0.5
    assert abs(s.std() / 0.1 - 1) < 0.05


def test_round_trips(tmp_path):
    st = film()
    e = st.layers[1].fit_entry('thickness')
    e.update(vary=True, min=50.0, max=120.0)
    e['prior'] = {'kind': 'soft_box', 'a': 70.0, 'b': 90.0, 's': 2.0}
    back = S.Stack.from_dict(st.to_dict())
    assert back.layers[1].fit['thickness']['prior'] == e['prior']
    back.layers[1].fit['thickness']['prior']['a'] = 0.0     # a deep copy
    assert e['prior']['a'] == 70.0
    R = st.reflectivities(Q, [(X, None)])[:, 0]
    pp = PosteriorProblem(FitProblem(st, [dict(Q=Q, R=R, dR=0.05 * R,
                                                P0=X, P=None)]))
    res = sample(pp, n_gen=200, seed=0, stop_when_converged=False)
    save_result(tmp_path / 'r.npz', res, pp)
    _, loaded, _ = load_run(tmp_path / 'r.npz')
    assert loaded.priors['Fe.thickness'].spec == complete(e['prior'])


def test_tension():
    g = {'kind': 'gauss', 'mu': 100.0, 'tau': 5.0}
    assert tension(g, 112.0) is None
    assert '3.4 τ' in tension(g, 117.0)
    b = {'kind': 'soft_box', 'a': 0.0, 'b': 1.0, 's': 0.1}
    assert tension(b, 1.2) is None and 's outside' in tension(b, 1.5)
    assert Prior(g).penalty(100.0) == 0 and Prior(b).penalty(0.5) == 0


def test_tension_reported():
    st = film()
    e = st.layers[1].fit_entry('thickness')
    e.update(vary=True, min=50.0, max=120.0)
    e['prior'] = {'kind': 'gauss', 'mu': 60.0, 'tau': 1.0}   # truth: 80
    R = st.reflectivities(Q, [(X, None)])[:, 0]
    pp = PosteriorProblem(FitProblem(st, [dict(Q=Q, R=R, dR=0.01 * R,
                                                P0=X, P=None)]))
    res = sample(pp, n_gen=600, seed=0, stop_when_converged=False)
    warn = posterior_warnings(res, problem=pp)
    assert any('Prior–data tension: Fe.thickness' in w for w in warn)


def test_prior_dialog():
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tabs.simulation import PriorDialog
    d = PriorDialog('NSLD real', None, scale=1e6, unit='10⁻⁶ Å⁻²')
    d.kind.setCurrentIndex(1)
    d.edits['mu'].setText('6.5')
    d._accept()
    assert d.spec == {'kind': 'gauss', 'mu': 6.5e-6,
                      'tau': pytest.approx(6.5e-7)}
    d = PriorDialog('NSLD real', d.spec, scale=1e6)
    assert d.edits['mu'].text() == '6.5' and d.kind.currentIndex() == 1
    d.kind.setCurrentIndex(0)
    d._accept()
    assert d.spec is None
    d = PriorDialog('x', None)
    d.kind.setCurrentIndex(2)
    d.edits['a'].setText('3')
    d._accept()
    assert d.result() == 0 and 'lower and upper' in d.error.text()
