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


def test_split_rhat_sees_drift():
    """Chains that all drift the same way agree with each other, so the
    classic R-hat is ~1; the split R-hat sees the drift."""
    from dream import gelman_rubin, split_rhat
    rng = np.random.default_rng(0)
    n = 2000
    good = rng.normal(size=(n, 4, 2))
    drift = good + np.linspace(0, 3, n)[:, None, None]
    assert split_rhat(good).max() < 1.01
    assert gelman_rubin(drift).max() < 1.01 and split_rhat(drift).min() > 1.2


def test_archive_pruned_at_end_of_adaptation():
    """At adapt_until the archive keeps only the states archived in the
    last archive_keep of the adaptation phase (no prior draws), at least
    10 d of them; z0's rows stay; archive_keep None keeps everything."""
    kw = dict(n_chains=4, vectorized=True, seed=1, archive_every=10)
    r = dream(two_modes, [(0, 1)] * 10, n_gen=1000, adapt_until=1000, **kw)
    st = r.state
    # events after generation 750: 760 .. 1000, 4 rows each = 100 = 10 d
    assert st['m'] == 100 and st['archive_keep'] == 0.25
    assert np.array_equal(st['Z'], np.concatenate(
        [r.chains[r.generations == g][0] for g in range(760, 1001, 10)]))
    # a short adaptation phase: the 10 d minimum keeps older rows too
    kw.update(n_gen=200, adapt_until=200)
    short = dream(two_modes, [(0, 1)] * 10, **kw)
    assert short.state['m'] == 100
    full = dream(two_modes, [(0, 1)] * 10, archive_keep=None, **kw)
    assert full.state['m'] == 100 + 20 * 4
    assert np.array_equal(short.state['Z'], full.state['Z'][-100:])
    z0 = np.random.default_rng(2).uniform(0.2, 0.8, (30, 10))
    zr = dream(two_modes, [(0, 1)] * 10, z0=z0, **kw)
    assert zr.state['m'] == 100 and np.array_equal(zr.state['Z'][:30], z0)


def test_reprune_extends_the_burn_in():
    """reprune: the burn-in goes on to `after`, the archive keeps only
    what was archived after it (at least 10 d); resumed, nothing before
    it is a sample."""
    from dream import reprune
    kw = dict(n_chains=4, vectorized=True, seed=3, archive_every=10,
              adapt_until=200)
    r = dream(two_modes, [(0, 1)] * 10, n_gen=1000, **kw)
    st = reprune(r.state, 600)
    assert st['adapt_until'] == 600 and r.state['adapt_until'] == 200
    # 40 events after 600, 4 rows each: 160 rows, the last ones
    assert st['m'] == 160
    assert np.array_equal(st['Z'], r.state['Z'][-160:])
    late = reprune(r.state, 980)               # 8 rows: 10 d = 100 kept
    assert late['m'] == 100
    with pytest.raises(ValueError):
        reprune(r.state, 100)
    more = dream(two_modes, [(0, 1)] * 10, n_gen=1400, resume=st, **kw)
    assert more.adapt_until == 600
    assert more.generations[more.keep()].min() == 700        # last half
    assert more.generations[more.keep(0.0)].min() == 601


def test_window_from_end_of_adaptation():
    """discard 0: R-hat and samples() from the end of the adaptation
    phase on; the window is kept in the state and on resume."""
    kw = dict(n_chains=4, vectorized=True, seed=3, adapt_until=200)
    r = dream(two_modes, [(0, 1)] * 10, n_gen=1000, discard=0.0, **kw)
    assert r.discard == 0.0 and r.state['discard'] == 0.0
    assert r.generations[r.keep()].min() == 201
    assert len(r.samples()) == 800 * 4
    half = dream(two_modes, [(0, 1)] * 10, n_gen=500, discard=0.0, **kw)
    more = dream(two_modes, [(0, 1)] * 10, n_gen=1000, resume=half.state,
                 **kw)
    assert more.discard == 0.0 and np.array_equal(more.chains, r.chains)


def test_stop_needs_ess():
    """stop_when_converged stops once R-hat is below the target, but not
    before the effective sample size reaches ess_target."""
    kw = dict(n_gen=4000, n_chains=4, vectorized=True, seed=1,
              stop_when_converged=True, rhat_target=1.2,
              min_gen_after_converged=0)
    f = lambda X: -0.5 * ((X - 0.5) / 0.1) ** 2 @ np.ones(2)   # noqa: E731
    early = dream(f, [(0, 1)] * 2, **kw)
    assert early.generations[-1] < 4000
    never = dream(f, [(0, 1)] * 2, ess_target=1e9, **kw)
    assert never.generations[-1] == 4000
