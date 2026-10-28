"""
scipy's other global optimisers on a fit.problem.FitProblem (the Fit box's
basinhopping, DIRECT, SHGO and dual annealing methods), in the normalised
box [0, 1]^d and with the cost DE minimises (fit.problem COST).

METHODS -- run_global(problem, method, ...):
  'basinhopping'   : hops from the current values: a random step of at most
                     `stepsize` in every normalised coordinate (kept in the
                     box, scipy adapts its size), then L-BFGS-B, the new
                     minimum kept by the Metropolis test at temperature `T`
                     (in cost units: chi^2/N for the 'chi2' cost); `niter`
                     hops.  The L-BFGS-B gradients go to the pool.
  'direct'         : DIRECT (DIviding RECTangles), deterministic, from the
                     centre of the box; `maxfun` evaluations (None: 1000 per
                     parameter), `maxiter` iterations, `locally_biased`
                     (DIRECT-L) or not.  One process.
  'shgo'           : simplicial homology global optimisation: `n` sampling
                     points per iteration (`sampling_method` 'sobol',
                     'halton' or 'simplicial'), `iters` iterations, L-BFGS-B
                     from each candidate minimum.  The sampling goes to the
                     pool.
  'dual_annealing' : generalised simulated annealing from the current
                     values, `maxiter` iterations from `initial_temp`, with
                     an L-BFGS-B local search unless `local_search` is off
                     (scipy's iteration cap, DA_LOCAL_MAXITER).  The
                     annealing is a chain, one point at a time; the local
                     search's gradients go to the pool.

HISTORY -- callback(x, cost, step) with the current values (step 0), then
with the best point so far at each of the method's own reports: every hop
(basinhopping), iteration (DIRECT), step of a local fit (SHGO: scipy gives
its callback to them) or new minimum found (dual annealing): UNITS.

PROGRESS -- status(info), info = {'phase': 'global', 'method', 'step',
'maxiter' (None for SHGO and dual annealing), 'evals' (the cost evaluations of this
process; None when a pool evaluates too), 'best'}, at each history entry
and at most every PROGRESS_EVERY s between them.

MINIMA -- SHGO ends with every local minimum its local fits reached
(scipy's res.xl, res.funl), the global one first: minima(ends) gets them
as fit.multistart ends [(k, u, cost)], k = 1, 2, ... by cost (none when
stopped).

cancelled() is polled at every evaluation (and while the pool works);
the run then returns the best point so far with the message 'stopped'.
"""

import math
import time

import numpy as np
from scipy.optimize import basinhopping, direct, dual_annealing, shgo

from fit.de.de_fit import FitCancelled, _Cost, _PoolMap

METHODS = ('basinhopping', 'direct', 'shgo', 'dual_annealing')
LABELS = {'basinhopping': 'Basin hopping', 'direct': 'DIRECT',
          'shgo': 'SHGO', 'dual_annealing': 'Dual annealing'}
UNITS = {'basinhopping': 'hop', 'direct': 'iteration', 'shgo': 'step',
         'dual_annealing': 'minimum'}
# methods that use the workers (dual annealing: with its local search)
POOLED = ('basinhopping', 'shgo', 'dual_annealing')
PROGRESS_EVERY = 0.2                        # s between progress reports


def DA_LOCAL_MAXITER(d):
    """scipy's iteration cap of dual annealing's L-BFGS-B, d parameters
    (scipy.optimize._dual_annealing.LocalSearchWrapper)."""
    return min(max(6 * d, 100), 1000)


class _Tracked(_Cost):
    """problem(u) as _Cost, keeping how many evaluations this process made
    and the best point among them; on_eval() after each."""

    DROP = ('cancelled', 'on_eval')

    def __init__(self, problem, cancelled, on_eval=None):
        super().__init__(problem, cancelled)
        self.on_eval = on_eval
        self.n, self.best = 0, (None, math.inf)

    def __call__(self, u):
        c = super().__call__(u)
        self.n += 1
        if c < self.best[1]:
            self.best = (np.array(u, dtype=float), c)
        if self.on_eval is not None:
            self.on_eval()
        return c


class _Step:
    """basinhopping's random step, kept in the box; scipy adapts
    `stepsize`."""

    def __init__(self, stepsize, rng):
        self.stepsize, self.rng = stepsize, rng

    def __call__(self, u):
        s = self.stepsize
        return np.clip(u + self.rng.uniform(-s, s, len(u)), 0.0, 1.0)


def run_global(problem, method, callback=None, status=None, cancelled=None,
               workers=1, x0=None, seed=None, niter=100, T=1.0,
               stepsize=0.1, maxfun=None, maxiter=1000, locally_biased=True,
               n=128, iters=1, sampling_method='sobol', initial_temp=5230.0,
               local_search=True, minima=None):
    """One of METHODS on problem (options: see METHODS) from x0
    (normalised; default the current values); minima: see MINIMA.  Returns (x, cost, steps,
    message) with x in physical units, as fit.de.de_fit.run_lbfgs."""
    if method not in METHODS:
        raise ValueError('method must be one of %s' % ', '.join(METHODS))
    u0 = problem.x0() if x0 is None else np.clip(x0, 0.0, 1.0)
    d = len(u0)
    bounds = [(0.0, 1.0)] * d
    step, last = [0], [0.0]
    top = {'basinhopping': niter, 'direct': maxiter}.get(method)
    pooled = workers > 1 and method in POOLED and \
        (method != 'dual_annealing' or local_search)

    def report(now=False):
        t = time.monotonic()
        if status is None or not (now or t - last[0] >= PROGRESS_EVERY):
            return
        last[0] = t
        c = cost.best[1]
        status({'phase': 'global', 'method': method, 'step': step[0],
                'maxiter': top, 'evals': None if pooled else cost.n,
                'best': c if math.isfinite(c) else None})

    cost = _Tracked(problem, cancelled, report)

    def entry(u=None, c=None):
        """A history entry: the best point so far (u, c: one the method
        found, maybe in the pool)."""
        step[0] += 1
        if u is not None and c < cost.best[1]:
            cost.best = (np.clip(np.array(u, dtype=float), 0.0, 1.0),
                         float(c))
        if callback is not None:
            callback(problem.to_x(cost.best[0]), float(cost.best[1]),
                     step[0])
        report(now=True)

    pool = None
    try:
        c0 = cost(u0)
        if callback is not None:
            callback(problem.to_x(u0), float(c0), 0)
        report(now=True)
        if pooled:
            pool = _PoolMap(workers, cancelled, cost)
        if method == 'basinhopping':
            local = dict(method='L-BFGS-B', bounds=bounds,
                         options={} if pool is None else {'workers': pool})
            res = basinhopping(
                cost, np.copy(u0), niter=niter, T=T, stepsize=stepsize,
                minimizer_kwargs=local,
                take_step=_Step(stepsize, np.random.default_rng(seed)),
                callback=lambda x, f, accept: entry(x, f))
        elif method == 'direct':
            res = direct(cost, bounds, maxfun=maxfun or 1000 * d,
                         maxiter=maxiter, locally_biased=locally_biased,
                         callback=lambda xk: entry())
        elif method == 'shgo':
            # in the pool, a local fit cannot hand the pool on: L-BFGS-B
            # alone there
            res = shgo(cost, bounds, n=n, iters=iters,
                       sampling_method=sampling_method,
                       minimizer_kwargs=dict(method='L-BFGS-B'),
                       callback=lambda xk: entry(xk, problem(xk)),
                       workers=1 if pool is None else pool)
        else:
            # scipy's own local search, its gradients in the pool
            local = None if pool is None else dict(
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': DA_LOCAL_MAXITER(d), 'workers': pool})
            res = dual_annealing(
                cost, bounds, maxiter=maxiter, initial_temp=initial_temp,
                no_local_search=not local_search, x0=np.copy(u0),
                minimizer_kwargs=local, rng=seed,
                callback=lambda x, f, context: entry(x, f))
    except FitCancelled:
        u, c = cost.best
        return problem.to_x(u), float(c), step[0], 'stopped'
    finally:
        if pool is not None:
            pool.close()
    u, c = np.clip(res.x, 0.0, 1.0), float(res.fun)
    if cost.best[1] < c:                # e.g. a sample better than the end
        u, c = cost.best
    if method == 'shgo' and minima is not None:
        ends = [(np.clip(np.asarray(v, dtype=float), 0.0, 1.0), float(f))
                for v, f in zip(np.atleast_2d(res.xl),
                                np.atleast_1d(res.funl))]
        if not any(np.array_equal(v, u) for v, _ in ends):
            ends.append((np.asarray(u, dtype=float), c))
        ends.sort(key=lambda e: e[1])
        minima([(k + 1, v, f) for k, (v, f) in enumerate(ends)])
    msg = res.message
    if isinstance(msg, (list, tuple)):
        msg = '; '.join(str(m) for m in msg)
    return problem.to_x(u), float(c), step[0], str(msg)
