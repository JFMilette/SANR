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

PROGRESS -- progress(info) follows the starts as they run (from the worker
processes through a queue, read while waiting for the ends): info =
{'start': k, 'phase', 'step', 'maxiter', 'best'}, phase 'begin' as start
k begins, 'de' as a short DE evaluates its members (step: the generation
in progress, 0 for the first population, 'evals' of its 'size' members
evaluated), 'polish' after each iteration of its
polish (maxiter None), 'lbfgs' after each L-BFGS-B iteration, at most every
PROGRESS_EVERY s per start besides 'begin', each generation's end and the
polish's first; then 'done' (best: its end cost, u: its end point) once
its end is in.
'best' is the lowest cost of the start so far (None before any).  A
start's reports may arrive after its 'done'.
"""

import math
import multiprocessing
import queue
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from fit.de.de_fit import FitCancelled, run_de

METHODS = ('lbfgs', 'de')
EDGE = 0.005              # normalised distance from a bound: on it
MERGE_U = 0.02            # max |u_a - u_b| of one minimum
MERGE_COST = 1e-3         # max |cost_a - cost_b| of one minimum
LBFGS_MAXITER = 3000
PROGRESS_EVERY = 0.2      # s between the reports of one start

_QUEUE = None             # a worker process's progress queue (_set_queue)


def _set_queue(q):
    global _QUEUE
    _QUEUE = q


class _Counted:
    """problem, with on_call(cost) after each cost evaluation (for the
    short DE's progress)."""

    def __init__(self, problem, on_call):
        self._problem, self._on_call = problem, on_call

    def __getattr__(self, name):
        return getattr(self._problem, name)

    def __call__(self, u):
        c = self._problem(u)
        self._on_call(c)
        return c


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

    def __init__(self, problem, method, seed, de_options, cancelled=None,
                 progress=None):
        self.problem, self.method = problem, method
        self.seed, self.de_options = seed, dict(de_options or {})
        self.cancelled, self.progress = cancelled, progress
        self._last = 0.0                # time of the last report

    def __getstate__(self):
        # in a worker, reports go through _QUEUE
        return dict(self.__dict__, cancelled=None, progress=None)

    def _report(self, k, phase, step, maxiter, best, now=False, **more):
        """A PROGRESS report of start k, unless one went out less than
        PROGRESS_EVERY s ago (now: send it anyway)."""
        t = time.monotonic()
        if not now and t - self._last < PROGRESS_EVERY:
            return
        self._last = t
        info = dict(start=k, phase=phase, step=step, maxiter=maxiter,
                    best=None if best is None else float(best), **more)
        if self.progress is not None:
            self.progress(info)
        elif _QUEUE is not None:
            _QUEUE.put(info)

    def _cost(self, u):
        if self.cancelled is not None and self.cancelled():
            raise FitCancelled
        return self.problem(u)

    def __call__(self, item):
        k, u0 = item
        self._report(k, 'begin', 0, None, None, now=True)
        if self.method == 'lbfgs':
            it = [0]

            def cb(intermediate_result):
                it[0] += 1
                self._report(k, 'lbfgs', it[0], LBFGS_MAXITER,
                             intermediate_result.fun)
            r = minimize(self._cost, u0, method='L-BFGS-B',
                         bounds=[(0.0, 1.0)] * len(u0), callback=cb,
                         options={'maxiter': LBFGS_MAXITER})
            u = np.clip(r.x, 0.0, 1.0)
            return k, u, float(self.problem(u))
        opts = dict(dict(maxiter=100, polish=True), **self.de_options)
        # scipy's population: popsize x d, a power of 2 for Sobol
        size = opts.get('popsize', 15) * len(u0)
        if opts.get('init') == 'sobol':
            size = 2 ** math.ceil(math.log2(size))
        # generation in progress (0: the first population), its members
        # evaluated, lowest cost, polishing
        run = dict(gen=0, evals=0, best=None, polish=False)

        def evaluated(c):
            if run['polish']:
                return
            if run['gen'] == 0 and run['evals'] == size:
                # scipy's first report comes after generation 1 too
                run.update(gen=1, evals=0)
            run['evals'] += 1
            if np.isfinite(c) and (run['best'] is None or c < run['best']):
                run['best'] = c
            self._report(k, 'de', run['gen'], opts['maxiter'], run['best'],
                         evals=run['evals'], size=size)

        def status(info):
            if info['phase'] == 'polish':
                run['polish'] = True
                self._report(k, 'polish', info['iteration'], None,
                             info['best'], now=not info['iteration'])
            else:
                run.update(gen=info['generation'] + 1, evals=0,
                           best=info['best'])
                self._report(k, 'de', run['gen'], info['maxiter'],
                             run['best'], now=True, evals=0, size=size)
        x, cost, _, msg = run_de(
            _Counted(self.problem, evaluated), x0=u0,
            cancelled=self.cancelled, status=status,
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
                   callback=None, progress=None):
    """Local fits from the current values and n_starts Sobol points:
    [{'chi2' (the cost), 'x' (physical), 'u', 'hits', 'at_bound' (bool per
    parameter), 'from_current'}], best first.  callback(done, total) after
    each start; progress(info): see PROGRESS."""
    if method not in METHODS:
        raise ValueError('method must be one of %s' % ', '.join(METHODS))
    items = list(enumerate(starting_points(problem, n_starts, seed)))
    ends = []

    def ended(end):
        ends.append(end)
        if progress is not None:
            progress(dict(start=end[0], phase='done', step=None,
                          maxiter=None, best=end[2], u=end[1]))
        if callback is not None:
            callback(len(ends), len(items))
    if workers > 1:
        local = _Local(problem, method, seed, de_options)
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue() if progress is not None else None
        pool = ctx.Pool(workers, initializer=_set_queue, initargs=(q,))

        def drain():
            while q is not None:
                try:
                    progress(q.get_nowait())
                except queue.Empty:
                    return
        try:
            res = pool.imap_unordered(local, items)
            for _ in items:
                while True:
                    drain()
                    try:
                        end = res.next(timeout=0.05)
                        break
                    except multiprocessing.TimeoutError:
                        if cancelled is not None and cancelled():
                            return merge(problem, ends)
                drain()
                ended(end)
        finally:
            pool.terminate()
            pool.join()
            if q is not None:
                q.close()
    else:
        local = _Local(problem, method, seed, de_options, cancelled,
                       progress)
        for item in items:
            try:
                end = local(item)
            except FitCancelled:
                break
            ended(end)
    return merge(problem, ends)
