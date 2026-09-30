"""
Fit a Stack to measured reflectivity channels with scipy's differential
evolution.

PARAMETERS -- the varied ones are Stack.free_parameters(): every Layer.fit
entry with vary = True, bounded by its [min, max].  The optimiser works in
normalised coordinates u in [0, 1]^n, x = min + u (max - min), so SLDs
(~1e-6) and thicknesses (~100) are on the same footing, including in the
final L-BFGS-B polish.

DATA -- a list of measured channels, each a dict
  {'Q', 'R', 'dR', 'P0': vector, 'P': vector or None, 'name' (optional)}
with (P0, P) its polarisation pair (see model.stack POLARISATION and
model.polarisation); 'name' labels it in warnings.  Channels measured along
different axes can be fitted together.  The model is ONE
Stack.reflectivities call per trial on the union of every channel's Q
points, with the stack's own resolution and background (added once per
channel).

COST -- mean over all points of the squared residual:
  'chi2' : (R_model - R) / dR, i.e. the reduced chi^2 up to the number of
           free parameters.  Points with dR <= 0 use dR = R (relative
           residual) so data without errors can still be fitted.
  'log'  : log10(R_model) - log10(R), equal weight per decade; points with
           R <= 0 are ignored.

WORKERS -- run_de(workers=n > 1) evaluates each generation's trial vectors
in n processes (scipy's updating='deferred': the population is replaced once
per generation instead of member by member, which may take a few more
generations).  The processes are started once per fit (about a second) and
receive a pickled copy of the problem; a stop request is seen while a
generation is being evaluated and ends the processes at once.
"""

import copy
import math
import multiprocessing
import warnings

import numpy as np
from scipy.optimize import differential_evolution


COSTS = ('chi2', 'log')


class FitProblem:
    """Cost of a copy of `stack` against `data` as a function of the
    normalised free parameters."""

    def __init__(self, stack, data, cost='chi2', qmin=-np.inf, qmax=np.inf):
        if cost not in COSTS:
            raise ValueError('unknown cost %r' % cost)
        # the caller's stack may be edited while the fit runs
        self.stack = copy.deepcopy(stack)
        self.cost = cost
        self.params = self.stack.free_parameters()
        if not self.params:
            raise ValueError('no parameter is varied: tick Fit on at least '
                             'one layer parameter in the Simulation tab')
        bad = [p for p in self.params if not p[3] < p[4]]
        if bad:
            raise ValueError('empty bounds for %s' % ', '.join(
                '%s.%s' % (self.stack.layers[i].name, a)
                for i, a, *_ in bad))
        # HALPERIN: only rho cos(phi) reaches the neutrons
        for i in sorted({p[0] for p in self.params}):
            varied = {p[1] for p in self.params if p[0] == i}
            if {'MSLD_rho', 'MSLD_phi'} <= varied:
                warnings.warn(
                    '%s: MSLD_rho and MSLD_phi are both varied, but only '
                    'rho cos(phi) is measurable (Halperin) and the sign of '
                    'phi is never determined' % self.stack.layers[i].name,
                    stacklevel=2)
        self.lo = np.array([p[3] for p in self.params], dtype=float)
        self.hi = np.array([p[4] for p in self.params], dtype=float)

        self.channels = []                 # [(name, idx, R, sigma)]
        self.pairs = []                    # (P0, P) of each channel
        Qs = []
        for k, e in enumerate(data):
            Q, R, dR = (np.asarray(e[c], dtype=float) for c in ('Q', 'R', 'dR'))
            ok = np.isfinite(Q) & np.isfinite(R) & (Q >= qmin) & (Q <= qmax)
            if cost == 'log':
                ok &= R > 0
                sigma = None
            else:
                dR = np.where(np.isfinite(dR) & (dR > 0), dR, np.abs(R))
                ok &= dR > 0
                sigma = dR[ok]
            if ok.any():
                P = e.get('P')
                self.pairs.append((np.asarray(e['P0'], dtype=float),
                                   None if P is None else
                                   np.asarray(P, dtype=float)))
                self.channels.append((e.get('name', 'channel %d' % k),
                                      R[ok], sigma))
                Qs.append(Q[ok])
        if not self.channels:
            raise ValueError('no data points to fit in the chosen Q range')
        self.Q, inv = np.unique(np.concatenate(Qs), return_inverse=True)
        # indices of each channel's points in self.Q
        splits = np.cumsum([len(q) for q in Qs])[:-1]
        self.channels = [(n, idx, R, s) for (n, R, s), idx
                         in zip(self.channels, np.split(inv, splits))]
        self.npoints = sum(len(ch[2]) for ch in self.channels)

        # said once here; the cost function itself never warns
        m = self.stack.fronting_direction()
        if m is not None:
            lost = [n for (n, *_), (P0, P) in zip(self.channels, self.pairs)
                    if any(v is not None and
                           np.linalg.norm(v - (v @ m)*m) > 1e-9
                           for v in (P0, P))]
            if lost:
                warnings.warn(
                    'magnetic fronting: the polarisation of %s is partly '
                    'transverse to its magnetisation; that part precesses '
                    'and is averaged out, only the projection on M_f is '
                    'fitted' % ', '.join(lost), stacklevel=2)

    # -- parameters ---------------------------------------------------------
    def to_x(self, u):
        """Physical values of normalised u, never outside [lo, hi]."""
        return self.lo + np.clip(u, 0.0, 1.0) * (self.hi - self.lo)

    def to_u(self, x):
        return np.clip((np.asarray(x) - self.lo) / (self.hi - self.lo), 0, 1)

    def x0(self):
        """Current values, normalised and clipped into the bounds."""
        return self.to_u([p[2] for p in self.params])

    def apply(self, x, stack=None):
        """Write parameter values x into `stack` (default: the fit's copy)."""
        stack = self.stack if stack is None else stack
        for (i, attr, *_), v in zip(self.params, x):
            setattr(stack.layers[i], attr, float(v))

    # -- cost ---------------------------------------------------------------
    def model(self, x):
        """Model reflectivity of each channel at its data points."""
        self.apply(x)
        self.stack.build_sublayers()
        R = self.stack.reflectivities(self.Q, self.pairs, warn=False)
        return [R[idx, k] for k, (_, idx, _, _) in enumerate(self.channels)]

    def residuals(self, x):
        out = []
        for m, (_, _, R, sigma) in zip(self.model(x), self.channels):
            if self.cost == 'log':
                with np.errstate(divide='ignore', invalid='ignore'):
                    out.append(np.log10(m) - np.log10(R))
            else:
                out.append((m - R) / sigma)
        r = np.concatenate(out)
        # a model that cannot be evaluated (NaN, R <= 0 in log) is heavily
        # penalised rather than aborting the fit
        return np.where(np.isfinite(r), r, 1e3)

    def __call__(self, u):
        return float(np.mean(self.residuals(self.to_x(u))**2))


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
           polish=True, seed=None, callback=None, cancelled=None, workers=1):
    """Differential evolution on `problem`.

    callback(x, cost, generation) is called after every generation with the
    best parameters so far.  cancelled() is polled at every cost evaluation
    (every POLL seconds with workers > 1); when it returns True the run
    stops at once, without polishing.  workers > 1 evaluates each
    generation in that many processes (see WORKERS).  mutation is scipy's
    F: a constant, or (min, max) to draw a new F every generation.  Returns
    (x, cost, generations, message) with x in physical units."""
    best = [problem.to_x(problem.x0()), None, 0]      # x, cost, generation
    cost = _Cost(problem, cancelled)
    pool = _PoolMap(workers, cancelled) if workers > 1 else None

    def cb(intermediate_result):
        best[:] = [problem.to_x(intermediate_result.x),
                   float(intermediate_result.fun), best[2] + 1]
        if callback is not None:
            callback(*best)

    try:
        res = differential_evolution(
            cost, [(0.0, 1.0)] * len(problem.params), x0=problem.x0(),
            maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation,
            polish=polish,
            seed=seed, callback=cb,
            workers=pool if pool is not None else 1,
            updating='deferred' if pool is not None else 'immediate')
    except FitCancelled:
        if best[1] is None:                  # stopped before one generation
            best[1] = problem(problem.to_u(best[0]))
        return best[0], best[1], best[2], 'stopped'
    finally:
        if pool is not None:
            pool.close()
    return problem.to_x(res.x), float(res.fun), best[2], res.message
