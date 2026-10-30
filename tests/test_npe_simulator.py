"""fit.npe.simulator: the NPE training simulator (plans/npe_plan_v1.md S3)
and the prerequisite P0.2 (one reflectivities call per Q grid).  Fast, and
without torch or sbi.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys
import warnings

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import model.stack as S                                     # noqa: E402
from fit.joint import JointProblem                          # noqa: E402
from fit.npe.simulator import (Simulator, UnitPrior, choose_coarse,  # noqa
                               coarse_problem, fidelity, prior_logpdf)
from fit.priors import Prior, log_prior                     # noqa: E402
from fit.problem import FitProblem                          # noqa: E402
from model.polarisation import channel_pair, in_plane       # noqa: E402
from test_dream_fit import film, free, measure              # noqa: E402
from test_mixture import mixed, pair                        # noqa: E402

X = in_plane(0.0)


def problem(channels=('+', '-'), res=False):
    st = free(film())
    st.resolution['enabled'] = res
    return FitProblem(st, measure(film(), channels,
                                  np.random.default_rng(0)), 'chi2')


def joint():
    def data(rho):
        st = film()
        st.layers[1].MSLD_rho = rho
        return measure(st, ('+', '-'), np.random.default_rng(1))
    st = free(film())
    return JointProblem([(n, FitProblem(st, data(r), 'chi2'))
                         for n, r in (('A', 5e-6), ('B', 3e-6))],
                        {'B': {'Fe.MSLD_rho': 'own'}})


def flat(p, u):
    return np.concatenate(p.model(p.to_x(u)))


# -- P0.2 ---------------------------------------------------------------------
@pytest.mark.parametrize('res', [False, True])
def test_one_reflectivities_call_per_grid(res, monkeypatch):
    p = problem(('++', '--', '+-', '-+'), res)
    calls = []
    real = p.stack.reflectivities

    def counted(Q, pairs, **kw):
        calls.append(len(pairs))
        return real(Q, pairs, **kw)

    monkeypatch.setattr(p.stack, 'reflectivities', counted)
    p.model(p.to_x(p.x0()))
    assert len(calls) == len(p.groups) == 1 and calls == [4]


# -- S3.1 bitwise -------------------------------------------------------------
@pytest.mark.parametrize('make', [problem, joint], ids=['fit', 'joint'])
def test_simulate_is_the_model_bitwise(make):
    p = make()
    sim = Simulator(p, noise=False, seed=0)
    u = sim.sample_prior(6)
    R, ok = sim.simulate(u)
    assert ok.all() and R.shape == (6, sim.n_points)
    for r, row in zip(R, u):
        assert np.array_equal(r, flat(p, row))
    uk, f = sim.draw(u)
    assert np.array_equal(uk, u) and f.dtype == np.float32
    assert np.array_equal(f, sim.features(R))


def test_joint_own_parameters():
    p = joint()
    assert p.names == ['Fe.thickness', 'Fe.MSLD_rho', 'Fe.MSLD_rho [B]',
                       'Fe.roughness_sigma', 'stack.background']
    assert Simulator(p).d == 5


# -- S3.2 ties ----------------------------------------------------------------
def test_tied_parameters_are_one():
    st = S.Stack([S.Layer('v', 0), S.Layer('A', 60, 4e-6, 0, 0, 0, 3.0),
                  S.Layer('B', 40, 3e-6, 0, 0, 0, 3.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0)])
    st.layers[1].fit['NSLD_real'] = {'vary': True, 'min': 1e-6, 'max': 7e-6}
    st.layers[2].fit['NSLD_real'] = {'vary': True, 'min': 2e-6, 'max': 6e-6}
    st.layers[1].fit['thickness'] = {'vary': True, 'min': 40, 'max': 80}
    st.tie([(1, 'NSLD_real'), (2, 'NSLD_real')])
    Q = np.linspace(0.01, 0.15, 60)
    R = st.reflectivities(Q, [(X, None)])[:, 0]
    p = FitProblem(st, [dict(Q=Q, R=R, dR=0.02 * R, P0=X, P=None)])
    sim = Simulator(p, noise=False)
    assert sim.d == 2
    u = np.array([0.3, 0.7])
    sim.simulate(u[None])
    a, b = (p.stack.layers[i].NSLD_real for i in (1, 2))
    assert a == b == p.to_x(u)[1]


# -- S3.3 prior ---------------------------------------------------------------
def with_prior():
    st = free(film())
    st.layers[1].fit['thickness']['prior'] = dict(kind='gauss', mu=70.0,
                                                  tau=8.0)
    st.layers[1].fit['roughness_sigma']['prior'] = dict(kind='soft_box',
                                                        a=2.0, b=5.0, s=1.0)
    return FitProblem(st, measure(film(), ('+',)), 'chi2')


def test_prior_draws_and_density():
    p = with_prior()
    prior = UnitPrior(p)
    rng = np.random.default_rng(3)
    u = prior.sample(200_000, rng)
    assert u.min() >= 0 and u.max() <= 1
    # the thickness: N(70, 8) truncated to [60, 100]
    g = np.linspace(60, 100, 20001)
    w = np.exp(-0.5 * ((g - 70) / 8) ** 2)
    mean = np.trapezoid(g * w, g) / np.trapezoid(w, g)
    sd = np.sqrt(np.trapezoid((g - mean) ** 2 * w, g) / np.trapezoid(w, g))
    t = p.to_x(u)[:, 0]
    assert abs(t.mean() - mean) < 4 * sd / np.sqrt(len(t))
    assert abs(t.std() / sd - 1) < 0.01
    # the others uniform
    assert abs(u[:, 1].mean() - 0.5) < 0.005
    # log_prob = the fit.priors terms + a constant, normalised over [0, 1]^d
    r = rng.random((50, p.lo.size))
    x = p.to_x(r)
    want = [sum(log_prior(q.spec, v) for q, v in zip(p.priors, xi)
                if q is not None) for xi in x]
    d = prior.log_prob(r) - np.array(want)
    assert np.ptp(d) < 1e-9
    v = np.array([0.3, 0.5])
    assert np.allclose(prior_logpdf(p.priors[0].spec, v * 40 + 60),
                       [Prior(p.priors[0].spec)(a) for a in v * 40 + 60])
    assert prior.log_prob(np.array([[1.2, 0.5, 0.5, 0.5]]))[0] == -np.inf


def test_prior_terms_are_normalised():
    p = with_prior()
    prior = UnitPrior(p)
    s = np.linspace(0, 1, 20001)
    for j in (0, 2):
        dens = np.exp(prior._term(s, j) - prior.log_z[j])
        assert abs(np.trapezoid(dens, s) - 1) < 1e-6


# -- S3.4 reproducibility -----------------------------------------------------
def test_same_seed_same_simulations_with_any_workers():
    p = problem(res=True)
    out = []
    for n in (1, 3):
        sim = Simulator(p, seed=11)
        u = sim.sample_prior(12)
        out.append(sim.draw(u, n_workers=n))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])


def test_failures_are_dropped():
    p = problem()
    sim = Simulator(p, noise=False)
    good = p.model

    def model(x):
        if x[0] > 80:
            raise ValueError('boom')
        return good(x)

    p.model = model
    u = np.array([[0.1, 0.5, 0.5, 0.5], [0.9, 0.5, 0.5, 0.5]])
    R, ok = sim.simulate(u)
    assert ok.tolist() == [True, False] and np.isnan(R[1]).all()
    uk, f = sim.draw(u)
    assert len(uk) == len(f) == 1
    assert sim.failure_rate == 0.5


# -- S3.5 noise ---------------------------------------------------------------
def test_noise_is_the_likelihoods():
    p = FitProblem(free(film()), measure(film(), ('+', '-')), 'chi2',
                   error_floor=0.02)
    sim = Simulator(p, seed=2)
    R = np.tile(sim.observed, (4000, 1))
    spread = (sim.add_noise(R) - R).std(axis=0)
    assert np.allclose(spread / sim.sigma, 1, atol=0.06)
    from fit.dream.dream_fit import PosteriorProblem
    want = np.concatenate([s for _, s in PosteriorProblem(p).data])
    assert np.array_equal(sim.sigma, want)


def test_fitted_error_scale_and_floor():
    """With the error scale and floor fitted, they are the last
    parameters and each draw's noise is the likelihood's at its values."""
    from fit.dream.dream_fit import (FLOOR_NAME, SIGMA_NAME,
                                     PosteriorProblem)
    fit = problem()
    pp = PosteriorProblem(fit, sigma_free=True, floor_free=True)
    sim = Simulator(pp, seed=4)
    assert sim.d == len(fit.params) + 2 and sim.k == len(fit.params)
    assert sim.names[-2:] == [SIGMA_NAME, FLOOR_NAME]
    u = sim.sample_prior(3)
    R, ok = sim.simulate(u)
    assert ok.all() and np.array_equal(R[0], flat(fit, u[0, :sim.k]))
    E = sim.errors(u)
    for e, row in zip(E, u):
        assert np.allclose(e, np.concatenate(pp.errors(pp.to_x(row))))
    rows = np.repeat(u[:1], 4000, axis=0)
    base = np.tile(R[0], (4000, 1))
    spread = (sim.add_noise(base, u=rows) - base).std(axis=0)
    assert np.allclose(spread / E[0], 1, atol=0.06)
    with pytest.raises(ValueError, match='fitted'):
        sim.add_noise(base)
    uk, f = sim.draw(u)
    assert uk.shape == (3, sim.d) and np.isfinite(f).all()


def test_features_are_finite_for_negative_R():
    sim = Simulator(problem())
    f = sim.features(np.full(sim.n_points, -1e-7))
    assert np.isfinite(f).all() and (f < 0).all()


# -- S3.6 fidelity ------------------------------------------------------------
def test_fidelity_of_the_full_problem_is_zero():
    p = problem(res=True)
    f = fidelity(p, p, n_draws=10, rng=0)
    assert f['worst'] == 0 and f['gap'].shape == (20, 2)


def test_coarse_problem_changes_slabs_and_grid_only():
    p = problem(res=True)
    for l in p.stack.layers:
        l.roughness_sublayer = 8
    c = coarse_problem(p, n_sub=3, grid_step=2)
    assert [l.roughness_sublayer for l in c.stack.layers] == [3] * 3
    assert c.stack.res_grid_step == 2 and p.stack.res_grid_step is None
    assert [l.roughness_sublayer for l in p.stack.layers] == [8] * 3
    f = fidelity(p, c, n_draws=10, rng=0)
    assert f['worst'] > 0


def test_choose_coarse_keeps_the_threshold():
    p = problem(res=True)
    for l in p.stack.layers:
        l.roughness_sublayer = 8
    c, rep = choose_coarse(p, threshold=1e9, n_draws=10, rng=0)
    assert rep['settings'] is not None and c is not p
    assert all(v < 1e9 for v in rep['p99'].values())
    c, rep = choose_coarse(p, threshold=0.0, n_draws=10, rng=0)
    assert rep['settings'] is None and c is p


# -- S3.7 generality ----------------------------------------------------------
def magnetic_fronting():
    st = S.Stack([S.Layer('front', 0, 1e-6, 0, 0.4e-6, 0.1),
                  S.Layer('Fe', 80, 8e-6, 0, 5e-6, 0.3, 4.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])
    n = st.fronting_direction()
    Q = np.linspace(0.01, 0.12, 60)
    pairs = [channel_pair(c, n) for c in ('++', '--', '+-', '-+')]
    st.build_sublayers()
    R = st.reflectivities(Q, pairs, warn=False)
    st.layers[1].fit['MSLD_theta'] = {'vary': True, 'min': 0.0, 'max': 0.5}
    st.layers[1].fit['thickness'] = {'vary': True, 'min': 60, 'max': 100}
    return FitProblem(st, [dict(Q=Q, R=R[:, j], dR=0.02 * R[:, j] + 1e-8,
                                P0=P0, P=P)
                           for j, (P0, P) in enumerate(pairs)], 'chi2')


def licorne_scheme():
    st = free(film())
    st.roughness_scheme = 'licorne'
    return FitProblem(st, measure(film(), ('+', '-')), 'chi2')


def spread():
    st = free(film())
    st.thickness_spread = 0.05
    st.fit['thickness_spread'] = dict(vary=True, min=0.0, max=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return FitProblem(st, measure(film(), ('+', '-')), 'chi2')


def mixture():
    a, _, _ = mixed()
    a.layers[-2].fit['thickness'] = {'vary': True, 'min': 70, 'max': 110}
    Q = np.linspace(0.01, 0.15, 80)
    R = a.reflectivities(Q, [pair('++'), pair('--')])
    return FitProblem(a, [dict(Q=Q, R=R[:, j], dR=0.03 * R[:, j], P0=P0, P=P)
                          for j, (P0, P) in enumerate((pair('++'),
                                                       pair('--')))], 'chi2')


@pytest.mark.parametrize('make', [magnetic_fronting, licorne_scheme, spread,
                                  mixture])
def test_every_model_feature_runs(make):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = make()
    sim = Simulator(p, seed=0)
    u, f = sim.draw(sim.sample_prior(4))
    assert len(u) == 4 and np.isfinite(f).all()
    c = coarse_problem(p, n_sub=2)
    fd = fidelity(p, c, n_draws=4, rng=0)
    assert np.isfinite(fd['worst'])
