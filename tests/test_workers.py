"""Worker pools that receive the problem once (fit.workers): the tasks
carry only a token, and the results are those of a single process.

Run from the repo root: python -m pytest tests
"""
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fit.de.de_fit import _Cost, run_lbfgs                  # noqa: E402
from fit.dream.dream_fit import PosteriorProblem, sample    # noqa: E402
from fit.problem import FitProblem                          # noqa: E402
from fit.workers import lookup                              # noqa: E402
from test_dream_fit import film, free, measure              # noqa: E402


def problem():
    return FitProblem(free(film()), measure(film(), ['++', '--']))


def test_token_replaces_the_problem_in_the_pickle():
    p = problem()
    cost = _Cost(p, cancelled=lambda: False)
    full = len(pickle.dumps(cost))
    cost.token = 'some-pool'
    small = pickle.loads(pickle.dumps(cost))
    assert len(pickle.dumps(cost)) < full / 10
    # outside a worker of that pool there is no problem (a drained task)
    assert small.problem is None and lookup('some-pool') is None
    assert small.cancelled is None
    cost.token = None
    back = pickle.loads(pickle.dumps(cost))
    u = p.x0()
    assert back.problem(u) == p(u)


def test_dream_workers_give_the_same_chains():
    pp = PosteriorProblem(problem())
    one = sample(pp, n_gen=30, n_chains=4, seed=3, stop_when_converged=False)
    two = sample(pp, n_gen=30, n_chains=4, seed=3, stop_when_converged=False,
                 n_workers=2)
    assert np.array_equal(one.chains, two.chains)
    assert np.array_equal(one.logp, two.logp)


def test_lbfgs_workers_give_the_same_fit():
    p = problem()
    a = run_lbfgs(p, maxiter=20)
    b = run_lbfgs(p, maxiter=20, workers=2)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_pool_starts_at_once_and_cleans_up():
    """start() returns without waiting for the processes (the problem goes
    through a file, not the start pipe); stop() removes that file."""
    import time
    from fit import workers
    big = np.zeros(400_000)                     # 3.2 MB, as a joint problem
    t = time.perf_counter()
    pool, token = workers.start(2, big)
    assert time.perf_counter() - t < 0.5
    path = workers._FILES[token]
    assert pathlib.Path(path).exists()
    workers.stop(pool, token)
    assert not pathlib.Path(path).exists() and token not in workers._FILES
