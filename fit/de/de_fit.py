"""
Differential-evolution fit of a fit.problem.FitProblem with scipy.

The optimiser works in the problem's normalised coordinates, including in
the final L-BFGS-B polish.

WORKERS -- run_de(workers=n > 1) evaluates each generation's trial vectors
in n processes (scipy's updating='deferred': the population is replaced once
per generation instead of member by member, which may take a few more
generations).  The processes are started once per fit (about a second) and
each receives a pickled copy of the problem then, once (fit.workers); the
tasks carry only the trial vectors.  A stop request is seen while a
generation is being evaluated and ends the processes at once.

POLISH -- the final L-BFGS-B polish is run here rather than by scipy, which
sends every single evaluation through the pool (the problem pickled each
time) and reports nothing while it runs: its finite-difference gradients
go to the pool as one batch, single evaluations stay in this process, and
status() reports each of its iterations.  The polished point is kept if it
is better and inside the bounds; unlike scipy, also when L-BFGS-B reports
a failure (e.g. a line search that gives up): its cost is evaluated exactly
at that point, so better is better (a Cr2Te3 run dropped 76.7 -> 57.6).

CONTINUE -- run_de(state=d) keeps in the dict d, after every generation,
the population ('population', normalised, best first), its costs
('energies') and the generation count ('generation'); run_de(resume=d)
starts from that population instead of a new one (scipy's init), counting
on from that generation: a stopped fit goes on where it was.  The scipy
random state is not kept (a new seed draws the next generations), and the
population's costs are evaluated again once.  run_de_restarts adds the run
('run'), the generations of the runs before it ('done') and their ends
('ends') to d, and resumes that run (from scratch if it was stopped
before its first generation), then any runs left.

STATUS -- status(info) is called after every generation with info = {
'phase': 'de', 'generation', 'maxiter', 'best', 'median' (of the
population's costs), 'spread' (their std / |mean|: the run stops once it
is <= 'tol', with atol 0), 'tol'}, then {'phase': 'polish', 'iteration',
'best'} as the polish starts (iteration 0) and after each of its iterations.

LOCAL -- run_lbfgs runs L-BFGS-B alone from the current values (the Fit
tool box's local page): the same cost, bounds and workers as the polish.
"""

import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

from fit.workers import Wrapper, start as start_pool, stop as stop_pool


class FitCancelled(Exception):
    pass


class _Cost(Wrapper):
    """problem(u), raising FitCancelled once cancelled() is True.  Pickled
    for the worker processes without cancelled (they are stopped from the
    parent instead, see _PoolMap), and without the problem once a pool holds
    it (fit.workers)."""

    DROP = ('cancelled',)

    def __init__(self, problem, cancelled):
        self.problem, self.cancelled = problem, cancelled

    def __call__(self, u):
        if self.cancelled is not None and self.cancelled():
            raise FitCancelled
        return self.problem(u)


class _PoolMap:
    """map() over a process pool for differential_evolution's `workers`,
    polling cancelled() while a generation is out."""

    POLL = 0.05                               # s between cancel checks

    def __init__(self, n, cancelled, cost):
        self.n, self.cancelled, self.cost = n, cancelled, cost
        # spawn: a fresh interpreter, never a fork of the GUI process; each
        # process gets the problem once, the tasks only cost's token
        self.pool, cost.token = start_pool(n, cost.problem)

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
        stop_pool(self.pool, self.cost.token)
        self.cost.token = None


def run_de(problem, maxiter=200, popsize=15, tol=1e-3, mutation=(0.5, 1),
           polish=True, seed=None, callback=None, cancelled=None, workers=1,
           full_output=False, x0=None, atol=0.0, init='latinhypercube',
           status=None, state=None, resume=None):
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
    mean|; init 'latinhypercube' or 'sobol').  status(info): see STATUS;
    the polish: see POLISH.  state, resume: see CONTINUE (with resume, x0,
    init and popsize are unused)."""
    if resume is not None:
        pop = np.clip(np.asarray(resume['population'], dtype=float), 0.0, 1.0)
        start = dict(init=pop)
        best = [problem.to_x(pop[0]), float(resume['energies'][0]),
                int(resume['generation'])]
    else:
        x0 = problem.x0() if x0 is None else np.clip(x0, 0.0, 1.0)
        start = dict(x0=x0, init=init)
        best = [problem.to_x(x0), None, 0]            # x, cost, generation
    gen0 = best[2]                         # generations before this call
    cost = _Cost(problem, cancelled)
    pool = _PoolMap(workers, cancelled, cost) if workers > 1 \
        else None

    def cb(intermediate_result):
        best[:] = [problem.to_x(intermediate_result.x),
                   float(intermediate_result.fun), best[2] + 1]
        if state is not None:
            e = np.asarray(intermediate_result.population_energies)
            order = np.argsort(e)
            state.update(population=np.array(
                intermediate_result.population)[order],
                energies=e[order], generation=best[2])
        if callback is not None:
            callback(*best)
        if status is not None:
            e = np.asarray(intermediate_result.population_energies)
            e = e[np.isfinite(e)]
            mean = abs(float(np.mean(e))) if e.size else np.nan
            status({'phase': 'de', 'generation': best[2],
                    'maxiter': gen0 + maxiter,
                    'best': best[1], 'tol': tol,
                    'median': float(np.median(e)) if e.size else np.nan,
                    'spread': float(np.std(e)) / mean if mean else np.nan})

    def polish_cb(intermediate_result):
        it[0] += 1
        if status is not None:
            status({'phase': 'polish', 'iteration': it[0],
                    'best': float(intermediate_result.fun)})

    it = [0]
    try:
        res = differential_evolution(
            cost, [(0.0, 1.0)] * len(problem.params), **start,
            maxiter=maxiter, popsize=popsize, tol=tol, atol=atol,
            mutation=mutation,
            polish=False,
            seed=seed, callback=cb,
            workers=pool if pool is not None else 1,
            updating='deferred' if pool is not None else 'immediate')
        polished = None
        if polish:
            if status is not None:
                status({'phase': 'polish', 'iteration': 0,
                        'best': float(res.fun)})
            pr = minimize(cost, np.copy(res.x), method='L-BFGS-B',
                          bounds=[(0.0, 1.0)] * len(problem.params),
                          callback=polish_cb,
                          options={} if pool is None else {'workers': pool})
            if pr.fun < res.fun and np.all((pr.x >= 0.0) & (pr.x <= 1.0)):
                polished = pr
                res.x, res.fun = pr.x, pr.fun
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
        if polished is not None:                 # as scipy: it is the best
            pop[0] = polished.x
        out += (np.array([problem.to_x(u) for u in pop]),)
    return out


def run_lbfgs(problem, maxiter=1000, callback=None, cancelled=None,
              workers=1, status=None, x0=None):
    """L-BFGS-B alone on `problem`, from x0 (normalised; default the
    current values, problem.x0()): a local fit, much faster than DE but it
    ends in the minimum nearest the start.  callback(x, cost, iteration) is
    called with the start (iteration 0) and after every iteration;
    status(info) with {'phase': 'lbfgs', 'iteration', 'maxiter', 'best'}
    likewise.  cancelled and workers as in run_de (the finite-difference
    gradients go to the pool as one batch, see POLISH).  Returns (x, cost,
    iterations, message) with x in physical units."""
    u0 = problem.x0() if x0 is None else np.clip(x0, 0.0, 1.0)
    cost = _Cost(problem, cancelled)
    best = [problem.to_x(u0), None, 0]                # x, cost, iteration

    def report():
        if callback is not None:
            callback(*best)
        if status is not None:
            status({'phase': 'lbfgs', 'iteration': best[2],
                    'maxiter': maxiter, 'best': best[1]})

    def cb(intermediate_result):
        best[:] = [problem.to_x(intermediate_result.x),
                   float(intermediate_result.fun), best[2] + 1]
        report()

    pool = None
    try:
        best[1] = float(cost(u0))
        report()
        pool = _PoolMap(workers, cancelled, cost) if workers > 1 \
        else None
        res = minimize(cost, np.copy(u0), method='L-BFGS-B',
                       bounds=[(0.0, 1.0)] * len(problem.params), callback=cb,
                       options=dict(maxiter=maxiter, **(
                           {} if pool is None else {'workers': pool})))
    except FitCancelled:
        if best[1] is None:                  # stopped before the start's cost
            best[1] = problem(problem.to_u(best[0]))
        return best[0], best[1], best[2], 'stopped'
    finally:
        if pool is not None:
            pool.close()
    return problem.to_x(res.x), float(res.fun), best[2], str(res.message)


def run_de_restarts(problem, restarts=1, seed=None, callback=None,
                    cancelled=None, status=None, state=None, resume=None,
                    **options):
    """run_de `restarts` times, with seeds seed + k: (x, cost, generations,
    message) of the best run, and [(k, u, cost)] of every run's end (for
    fit.multistart.merge).  callback(x, cost, generation) counts the
    generations over the runs, status(info) gets run_de's info with 'run'
    (1, 2, ...) and 'runs' added; a stop ends them all.  A converged DE can
    still sit in a local minimum (session A: 'success' at chi^2/N 10.28
    with tol 1e-3, against 4.93): the spread of the runs shows it.  state,
    resume: see CONTINUE."""
    ends, best, done, first = [], None, 0, 0
    if resume is not None:
        resume = dict(resume)            # state may be the same dict
        ends = list(resume.get('ends', []))
        done, first = resume.get('done', 0), resume.get('run', 0)
        if ends:
            k, u, c = min(ends, key=lambda e: e[2])
            best = [problem.to_x(u), c, 0, '']
    for k in range(first, restarts):
        def cb(x, cost, gen, done=done):
            if callback is not None:
                callback(x, cost, done + gen)
        def st(info, k=k):
            if status is not None:
                status(dict(info, run=k + 1, runs=restarts))

        run = {} if state is None else state
        if state is not None:
            state.clear()
            state.update(run=k, done=done, ends=list(ends))
        x, cost, gen, msg = run_de(
            problem, seed=None if seed is None else seed + k, callback=cb,
            cancelled=cancelled, status=st, state=run,
            resume=resume if k == first and resume is not None
            and 'population' in resume else None, **options)
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
