"""DREAM(ZS) snooker updates and the tempered burn-in (fit/dream/dream.py):
exactness on targets with known answers, resume, and a two-mode target
where three chains without them converge falsely to one mode.

Run from the repo root: python -m pytest tests   (-m slow for the slow ones)
"""
import numpy as np
import pytest
from scipy import stats
from dream import dream

# d = 10, modes at 0.25 and 0.75 in every coordinate, sd 0.03, weights
# 0.3 / 0.7, bounds [0, 1]: 3 chains without snooker or tempering find
# only mode 1 for seeds 3 and 5 (R-hat ~ 1.0 all the same)
TWO_MODES = dict(d=10, mu=(0.25, 0.75), sd=0.03, w=(0.3, 0.7))


def two_modes_split(X):
    """(log prior, log likelihood) of TWO_MODES, vectorised; the prior is
    flat on the bounds."""
    X = np.atleast_2d(X)
    sd, (w1, w2), (m1, m2) = TWO_MODES['sd'], TWO_MODES['w'], TWO_MODES['mu']
    a = np.log(w1) - 0.5 * (((X - m1) / sd) ** 2).sum(1)
    b = np.log(w2) - 0.5 * (((X - m2) / sd) ** 2).sum(1)
    return np.zeros(len(X)), np.logaddexp(a, b)


def two_modes(X):
    return np.add(*two_modes_split(X))


def mode2_weight(s):
    return float(np.mean(s.mean(1) > 0.5))


@pytest.mark.parametrize('p', [0.1, 0.5])
def test_snooker_flat_box(p):
    r = dream(lambda X: np.zeros(len(X)), [(0, 1)] * 5, n_gen=20000,
              n_chains=3, vectorized=True, seed=1, p_snooker=p)
    s = r.samples(0.5)[::10]
    ks = max(stats.kstest(s[:, j], 'uniform').statistic for j in range(5))
    edge = np.mean((s < 0.05) | (s > 0.95)) / 0.1
    assert ks < 0.04 and 0.95 <= edge <= 1.05, (ks, edge)


@pytest.mark.slow
@pytest.mark.parametrize('p', [0.1, 0.5])
def test_snooker_half_normal(p):
    """Mass piled against a bound.  80000 generations: at 20000 the mean of
    one seed scatters by 2-4 % (no bias over seeds)."""
    sd = np.array([0.05, 0.1, 0.2, 0.3])
    r = dream(lambda X: -0.5 * ((X / sd) ** 2).sum(1), [(0, 1)] * 4,
              n_gen=80000, n_chains=3, vectorized=True, seed=2, p_snooker=p,
              thin=4)
    # the half-normal mean, truncated at 1 (sd 0.3: 0.9 % of the mass)
    a = 1 / sd
    mean = sd * stats.norm.pdf(0) - sd * stats.norm.pdf(a)
    mean = mean / (stats.norm.cdf(a) - 0.5)
    ratio = r.samples(0.5).mean(0) / mean
    assert np.all((ratio > 0.96) & (ratio < 1.04)), ratio


@pytest.mark.parametrize('p', [0.1, 0.5])
def test_snooker_gaussian(p):
    d = 10
    var = np.arange(1, d + 1, dtype=float)
    S = 0.5 * np.sqrt(np.outer(var, var))
    np.fill_diagonal(S, var)
    Si = np.linalg.inv(S)
    r = dream(lambda X: -0.5 * np.einsum('ij,jk,ik->i', X, Si, X),
              [(-20, 20)] * d, n_gen=20000, n_chains=3, vectorized=True,
              seed=1, thin=2, p_snooker=p)
    s = r.samples(0.5)
    m_err = np.abs(s.mean(0)) / np.sqrt(var)
    c_err = np.abs(np.cov(s.T) - S) / np.sqrt(np.outer(var, var))
    assert m_err.max() < 0.15 and c_err.max() < 0.15, (m_err.max(),
                                                       c_err.max())


@pytest.mark.parametrize('temper', [False, True])
def test_snooker_resume(temper):
    """One run of 400 generations equals 200 + resume to 400, bit for
    bit, also within the tempered phase."""
    kw = dict(n_chains=4, vectorized=True, seed=5, p_snooker=0.3,
              adapt_until=300)
    if temper:
        kw.update(logpdf_split=two_modes_split, beta_start=1e-3)
    one = dream(two_modes, [(0, 1)] * 10, n_gen=400, **kw)
    half = dream(two_modes, [(0, 1)] * 10, n_gen=200, **kw)
    two = dream(two_modes, [(0, 1)] * 10, n_gen=400, resume=half.state, **kw)
    assert np.array_equal(one.chains, two.chains)
    assert np.array_equal(one.logp, two.logp)
    assert one.n_evals == two.n_evals


def test_tempering_off_is_identical():
    base = dict(n_gen=600, n_chains=4, vectorized=True, seed=3,
                p_snooker=0.1)
    a = dream(two_modes, [(0, 1)] * 10, **base)
    b = dream(two_modes, [(0, 1)] * 10, beta_start=1.0, **base)
    c = dream(two_modes, [(0, 1)] * 10, beta_start=1.0,
              logpdf_split=two_modes_split, **base)
    assert np.array_equal(a.chains, b.chains)
    assert np.array_equal(a.chains, c.chains)


def test_snooker_off_is_identical():
    """p_snooker = 0 draws no extra random numbers: the chains of before."""
    base = dict(n_gen=300, n_chains=3, vectorized=True, seed=4)
    a = dream(two_modes, [(0, 1)] * 10, **base)
    b = dream(two_modes, [(0, 1)] * 10, p_snooker=0.0, **base)
    assert np.array_equal(a.chains, b.chains)


def test_settings_refused():
    with pytest.raises(ValueError, match='logpdf_split'):
        dream(two_modes, [(0, 1)] * 10, n_gen=10, vectorized=True,
              beta_start=1e-3)
    with pytest.raises(ValueError, match='adaptation phase'):
        dream(two_modes, [(0, 1)] * 10, n_gen=100, vectorized=True,
              adapt_until=10, temper_until=20, beta_start=1e-3,
              logpdf_split=two_modes_split)
    with pytest.raises(ValueError, match='archive'):
        dream(two_modes, [(0, 1)] * 10, n_gen=10, vectorized=True,
              archive=False, p_snooker=0.1)


@pytest.mark.slow
def test_two_modes_found():
    """8 chains, snooker 0.1, tempered burn-in from 1e-3: every seed sees
    both modes, in about the right proportion."""
    w = []
    for seed in range(1, 6):
        r = dream(two_modes, [(0, 1)] * 10, n_gen=30000, n_chains=8,
                  vectorized=True, seed=seed, p_snooker=0.1,
                  logpdf_split=two_modes_split, beta_start=1e-3,
                  adapt_until=3000, temper_until=3000)
        w.append(mode2_weight(r.samples(0.5)))
    w = np.array(w)
    print('mode-2 weights', np.round(w, 3))
    assert np.all((w >= 0.05) & (w <= 0.95)), w
    assert abs(w.mean() - 0.70) <= 0.10, w
