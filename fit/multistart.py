"""
Multistart local optimisation of a fit.problem.FitProblem: where a fit can
end, and how often.

STARTS -- start 0 is the current values (problem.x0()); starts 1..n are the
first n points of a scrambled Sobol sequence in the normalised box [0, 1]^d
(scipy.stats.qmc.Sobol, `seed`), spread evenly over the bounds.

LOCAL FIT -- 'lbfgs': scipy's L-BFGS-B from the start (maxiter 3000); 'de':
a short differential evolution (fit.de.de_fit.run_de, maxiter 100 with the
polish, seed + k) whose population holds the start.

MINIMA -- the end points are merged when they agree to MERGE_U in every
normalised coordinate and to MERGE_COST in cost: one minimum, its best
member kept, `hits` how many starts reached it, `from_current` whether the
current values' start is among them.  Sorted by cost (the problem's: chi^2/N
for the 'chi2' cost).  A parameter within EDGE of either end of its bounds
is `at_bound`: the bound, not the data, holds it there.

run_multistart(workers > 1) runs the starts in that many 'spawn' processes;
cancelled() is polled between starts (at every cost evaluation with one
worker) and returns the minima found so far.
"""

import math
import multiprocessing

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from fit.de.de_fit import FitCancelled, run_de

METHODS = ('lbfgs', 'de')
EDGE = 0.005              # normalised distance from a bound: on it
MERGE_U = 0.02            # max |u_a - u_b| of one minimum
MERGE_COST = 1e-3         # max |cost_a - cost_b| of one minimum


def starting_points(problem, n_starts, seed=None):
    """[(the current values), n Sobol points], normalised."""
    U = [problem.x0()]
    if n_starts > 0:
        m = max(0, math.ceil(math.log2(n_starts)))
        U += list(qmc.Sobol(len(problem.params), seed=seed)
                  .random_base2(m)[:n_starts])
    return U


class _Local:
    """One local fit from (k, u0) -> (k, u, cost); picklable."""

    def __init__(self, problem, method, seed, de_options, cancelled=None):
        self.problem, self.method = problem, method
        self.seed, self.de_options = seed, dict(de_options or {})
        self.cancelled = cancelled

    def __getstate__(self):
        return dict(self.__dict__, cancelled=None)

    def _cost(self, u):
        if self.cancelled is not None and self.cancelled():
            raise FitCancelled
        return self.problem(u)

    def __call__(self, item):
        k, u0 = item
        if self.method == 'lbfgs':
            r = minimize(self._cost, u0, method='L-BFGS-B',
                         bounds=[(0.0, 1.0)] * len(u0),
                         options={'maxiter': 3000})
            u = np.clip(r.x, 0.0, 1.0)
            return k, u, float(self.problem(u))
        opts = dict(dict(maxiter=100, polish=True), **self.de_options)
        x, cost, _, msg = run_de(
            self.problem, x0=u0, cancelled=self.cancelled,
            seed=None if self.seed is None else self.seed + k, **opts)
        if msg == 'stopped':
            raise FitCancelled
        return k, self.problem.to_u(x), float(cost)


def merge(problem, ends):
    """Distinct minima of ends [(start k, u, cost)] (see MINIMA)."""
    out = []
    for k, u, cost in sorted(ends, key=lambda e: e[2]):
        for m in out:
            if np.max(np.abs(u - m['u'])) < MERGE_U and \
                    abs(cost - m['chi2']) < MERGE_COST:
                m['hits'] += 1
                m['from_current'] |= k == 0
                break
        else:
            out.append(dict(chi2=cost, x=problem.to_x(u), u=np.array(u),
                            hits=1, at_bound=(u < EDGE) | (u > 1 - EDGE),
                            from_current=k == 0))
    return out


def run_multistart(problem, n_starts=24, method='lbfgs', seed=None,
                   workers=1, cancelled=None, de_options=None,
                   callback=None):
    """Local fits from the current values and n_starts Sobol points:
    [{'chi2' (the cost), 'x' (physical), 'u', 'hits', 'at_bound' (bool per
    parameter), 'from_current'}], best first.  callback(done, total) after
    each start."""
    if method not in METHODS:
        raise ValueError('method must be one of %s' % ', '.join(METHODS))
    items = list(enumerate(starting_points(problem, n_starts, seed)))
    ends = []
    if workers > 1:
        local = _Local(problem, method, seed, de_options)
        pool = multiprocessing.get_context('spawn').Pool(workers)
        try:
            res = pool.imap_unordered(local, items)
            for _ in items:
                while True:
                    try:
                        ends.append(res.next(timeout=0.05))
                        break
                    except multiprocessing.TimeoutError:
                        if cancelled is not None and cancelled():
                            return merge(problem, ends)
                if callback is not None:
                    callback(len(ends), len(items))
        finally:
            pool.terminate()
            pool.join()
    else:
        local = _Local(problem, method, seed, de_options, cancelled)
        for item in items:
            try:
                ends.append(local(item))
            except FitCancelled:
                break
            if callback is not None:
                callback(len(ends), len(items))
    return merge(problem, ends)
