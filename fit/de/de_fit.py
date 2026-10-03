"""
Differential-evolution fit of a fit.problem.FitProblem with scipy.

The optimiser works in the problem's normalised coordinates, including in
the final L-BFGS-B polish.

WORKERS -- run_de(workers=n > 1) evaluates each generation's trial vectors
in n processes (scipy's updating='deferred': the population is replaced once
per generation instead of member by member, which may take a few more
generations).  The processes are started once per fit (about a second) and
receive a pickled copy of the problem; a stop request is seen while a
generation is being evaluated and ends the processes at once.
"""

import math
import multiprocessing

import numpy as np
from scipy.optimize import differential_evolution


class FitCancelled(Exception):
    pass


class _Cost:
    """problem(u), raising FitCancelled once cancelled() is True.  Pickled
    for the worker processes without cancelled (they are stopped from the
    parent instead, see _PoolMap)."""

    def __init__(self, problem, cancelled):
        self.problem, self.cancelled = problem, cancelled

    def __getstate__(self):
        return {'problem': self.problem, 'cancelled': None}

    def __call__(self, u):
        if self.cancelled is not None and self.cancelled():
            raise FitCancelled
        return self.problem(u)


class _PoolMap:
    """map() over a process pool for differential_evolution's `workers`,
    polling cancelled() while a generation is out."""

    POLL = 0.05                               # s between cancel checks

    def __init__(self, n, cancelled):
        self.n, self.cancelled = n, cancelled
        # spawn: a fresh interpreter, never a fork of the GUI process
        self.pool = multiprocessing.get_context('spawn').Pool(n)

    def __call__(self, func, iterable):
        items = list(iterable)
        # one chunk per process: the problem is pickled once per chunk
        res = self.pool.map_async(func, items,
                                  chunksize=max(1, math.ceil(len(items) / self.n)))
        while not res.ready():
            res.wait(self.POLL)
            if self.cancelled is not None and self.cancelled():
                raise FitCancelled
        return res.get()

    def close(self):
        self.pool.terminate()
        self.pool.join()


def run_de(problem, maxiter=200, popsize=15, tol=1e-3, mutation=(0.5, 1),
           polish=True, seed=None, callback=None, cancelled=None, workers=1,
           full_output=False, x0=None, atol=0.0, init='latinhypercube'):
    """Differential evolution on `problem`.

    callback(x, cost, generation) is called after every generation with the
    best parameters so far.  cancelled() is polled at every cost evaluation
    (every POLL seconds with workers > 1); when it returns True the run
    stops at once, without polishing.  workers > 1 evaluates each
    generation in that many processes (see WORKERS).  mutation is scipy's
    F: a constant, or (min, max) to draw a new F every generation.  Returns
    (x, cost, generations, message) with x in physical units; full_output
    adds the final population in physical units, best first (None if
    stopped).  x0: the normalised member the population starts with
    (default the current values, problem.x0()); tol, atol and init are
    scipy's (stop once the population's cost spread <= atol + tol x |its
    mean|; init 'latinhypercube' or 'sobol')."""
    x0 = problem.x0() if x0 is None else np.clip(x0, 0.0, 1.0)
    best = [problem.to_x(x0), None, 0]                # x, cost, generation
    cost = _Cost(problem, cancelled)
    pool = _PoolMap(workers, cancelled) if workers > 1 else None

    def cb(intermediate_result):
        best[:] = [problem.to_x(intermediate_result.x),
                   float(intermediate_result.fun), best[2] + 1]
        if callback is not None:
            callback(*best)

    try:
        res = differential_evolution(
            cost, [(0.0, 1.0)] * len(problem.params), x0=x0,
            maxiter=maxiter, popsize=popsize, tol=tol, atol=atol, init=init,
            mutation=mutation,
            polish=polish,
            seed=seed, callback=cb,
            workers=pool if pool is not None else 1,
            updating='deferred' if pool is not None else 'immediate')
    except FitCancelled:
        if best[1] is None:                  # stopped before one generation
            best[1] = problem(problem.to_u(best[0]))
        out = best[0], best[1], best[2], 'stopped'
        return out + (None,) if full_output else out
    finally:
        if pool is not None:
            pool.close()
    out = problem.to_x(res.x), float(res.fun), best[2], res.message
    if full_output:
        pop = res.population[np.argsort(res.population_energies)]
        out += (np.array([problem.to_x(u) for u in pop]),)
    return out


def run_de_restarts(problem, restarts=1, seed=None, callback=None,
                    cancelled=None, **options):
    """run_de `restarts` times, with seeds seed + k: (x, cost, generations,
    message) of the best run, and [(k, u, cost)] of every run's end (for
    fit.multistart.merge).  callback(x, cost, generation) counts the
    generations over the runs; a stop ends them all.  A converged DE can
    still sit in a local minimum (session A: 'success' at chi^2/N 10.28
    with tol 1e-3, against 4.93): the spread of the runs shows it."""
    ends, best, done = [], None, 0
    for k in range(restarts):
        def cb(x, cost, gen, done=done):
            if callback is not None:
                callback(x, cost, done + gen)
        x, cost, gen, msg = run_de(
            problem, seed=None if seed is None else seed + k, callback=cb,
            cancelled=cancelled, **options)
        ends.append((k, problem.to_u(x), cost))
        if best is None or cost < best[1]:
            best = [x, cost, 0, str(msg)]
        done += gen
        if msg == 'stopped':
            best[3] = 'stopped'
            break
    best[2] = done
    if restarts > 1 and best[3] != 'stopped':
        best[3] = 'best of %d runs: %s' % (len(ends), best[3])
    return tuple(best), ends
