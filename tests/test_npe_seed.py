"""fit.npe.seed (plans/npe_plan_v1.md D1) with a stand-in network, and the
initial DREAM(ZS) archive z0 (D2).  Fast, without torch or sbi.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fit.dream.dream_fit import posterior_problem, sample   # noqa: E402
from fit.npe.seed import allocate, format_clusters, seed_chains  # noqa: E402
from test_dream_fit import film, free, measure               # noqa: E402

MODES = np.array([[0.25, 0.3], [0.7, 0.75]])
SD = 0.03


class Fit:
    """The parts of a FitProblem seed_chains uses: [0, 1]^2."""
    params = [None, None]
    names = ['a', 'b']
    lo, hi = np.zeros(2), np.ones(2)

    def to_x(self, u):
        return np.asarray(u, dtype=float)

    def to_u(self, x):
        return np.asarray(x, dtype=float)


class Posterior:
    """Two Gaussian modes holding 70 % and 30 % of the mass."""
    n_extra = 0
    names = Fit.names

    def __init__(self):
        self.fit = Fit()

    def to_x(self, u):
        return np.asarray(u, dtype=float)

    def __call__(self, x):
        d = ((np.asarray(x) - MODES) ** 2).sum(axis=1) / SD ** 2
        return float(np.log(0.7 * np.exp(-d[0] / 2) + 0.3 * np.exp(-d[1] / 2)))

    def chi2_per_point(self, x):
        return -2 * self(x)


class Network:
    """A proposal with the right modes and the wrong weights (50 / 50),
    a bit too wide: the importance weights must correct it."""

    def check(self, problem):
        pass

    def sample(self, n, round=-1, seed=None):
        rng = np.random.default_rng(seed)
        k = rng.integers(0, 2, n)
        return MODES[k] + rng.normal(0, 1.5 * SD, (n, 2))

    def log_prob(self, u, round=-1):
        s = 1.5 * SD
        d = ((u[:, None, :] - MODES[None]) ** 2).sum(-1) / s ** 2
        return np.log(0.5 * np.exp(-d / 2).sum(axis=1) / (2 * np.pi * s**2))


def test_both_modes_get_chains_by_their_weight():
    seeds = seed_chains(Network(), Posterior(), 10, rng=0, n_samples=4000)
    assert seeds.x0.shape == (10, 2)
    big = [c for c in seeds.clusters if c['weight'] > 0.05]
    assert len(big) == 2
    w = sorted(c['weight'] for c in big)
    assert w[1] == pytest.approx(0.7, abs=0.05)
    assert sorted(c['chains'] for c in big) == [3, 7]
    near = [np.argmin(((x - MODES) ** 2).sum(axis=1)) for x in seeds.x0]
    assert sorted(np.bincount(near)) == [3, 7]
    assert len(np.unique(seeds.x0, axis=0)) == 10       # distinct starts
    assert 'weight' in format_clusters(seeds)


def test_allocation():
    assert allocate([0.6, 0.3, 0.1], 10, 1e-3).tolist() == [6, 3, 1]
    # every cluster above min_weight gets one chain first
    assert allocate([0.98, 0.0105, 0.0095], 8, 1e-2).tolist() == [7, 1, 0]
    assert allocate([0.98, 0.0105, 0.0095], 8, 1e-3).tolist() == [6, 1, 1]
    # more clusters than chains: the heaviest ones
    assert allocate([0.1, 0.5, 0.4], 2, 1e-3).tolist() == [0, 1, 1]
    assert allocate([1e-9, 1e-8], 3, 0.5).tolist() == [0, 3]


def problem():
    st = free(film())
    return posterior_problem(st, measure(film(), ('+', '-')))


def test_initial_archive():
    pp = problem()
    rng = np.random.default_rng(0)
    z0 = pp.to_x(rng.random((40, pp.ndim)))
    res = sample(pp, n_gen=30, seed=1, z0=z0, stop_when_converged=False)
    assert np.allclose(res.state['Z'][:40], pp.to_u(z0), rtol=0, atol=1e-15)
    a = sample(pp, n_gen=30, seed=1, stop_when_converged=False)
    b = sample(pp, n_gen=30, seed=1, z0=None, stop_when_converged=False)
    assert np.array_equal(a.chains, b.chains)
    assert not np.array_equal(a.state['Z'][:40], res.state['Z'][:40])
    with pytest.raises(ValueError, match='z0'):
        sample(pp, n_gen=10, seed=1, z0=z0[:3])
