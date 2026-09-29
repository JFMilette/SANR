"""
Fit a Stack to measured reflectivity channels with scipy's differential
evolution.

PARAMETERS -- the varied ones are Stack.free_parameters(): every Layer.fit
entry with vary = True, bounded by its [min, max].  The optimiser works in
normalised coordinates u in [0, 1]^n, x = min + u (max - min), so SLDs
(~1e-6) and thicknesses (~100) are on the same footing, including in the
final L-BFGS-B polish.

DATA -- a list of channels (name, Q, R, dR) with name one of '++', '+-',
'-+', '--' (lab channels of calc_reflectance) or '+' / '-' (half-polarized,
R+ = R++ + R+-, R- = R-- + R-+).  The model is evaluated once per trial on
the union of every channel's Q points, with the stack's own resolution and
background.

COST -- mean over all points of the squared residual:
  'chi2' : (R_model - R) / dR, i.e. the reduced chi^2 up to the number of
           free parameters.  Points with dR <= 0 use dR = R (relative
           residual) so data without errors can still be fitted.
  'log'  : log10(R_model) - log10(R), equal weight per decade; points with
           R <= 0 are ignored.
"""

import copy

import numpy as np
from scipy.optimize import differential_evolution


# columns of calc_reflectance summed for each data channel
CHANNEL_COLUMNS = {'++': (4,), '+-': (5,), '-+': (6,), '--': (7,),
                   '+': (4, 5), '-': (7, 6)}
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
        self.lo = np.array([p[3] for p in self.params], dtype=float)
        self.hi = np.array([p[4] for p in self.params], dtype=float)

        self.channels = []                 # [(name, columns, idx, R, sigma)]
        Qs = []
        for name, Q, R, dR in data:
            Q, R, dR = (np.asarray(a, dtype=float) for a in (Q, R, dR))
            ok = np.isfinite(Q) & np.isfinite(R) & (Q >= qmin) & (Q <= qmax)
            if cost == 'log':
                ok &= R > 0
                sigma = None
            else:
                dR = np.where(np.isfinite(dR) & (dR > 0), dR, np.abs(R))
                ok &= dR > 0
                sigma = dR[ok]
            if ok.any():
                self.channels.append((name, CHANNEL_COLUMNS[name], R[ok],
                                      sigma))
                Qs.append(Q[ok])
        if not self.channels:
            raise ValueError('no data points to fit in the chosen Q range')
        self.Q, inv = np.unique(np.concatenate(Qs), return_inverse=True)
        # indices of each channel's points in self.Q
        splits = np.cumsum([len(q) for q in Qs])[:-1]
        self.channels = [(n, c, idx, R, s) for (n, c, R, s), idx
                         in zip(self.channels, np.split(inv, splits))]
        self.npoints = sum(len(ch[3]) for ch in self.channels)

    # -- parameters ---------------------------------------------------------
    def to_x(self, u):
        return self.lo + np.asarray(u) * (self.hi - self.lo)

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
        R = self.stack.calc_reflectance(self.Q)
        return [R[idx][:, list(cols)].sum(1) for _, cols, idx, _, _
                in self.channels]

    def residuals(self, x):
        out = []
        for m, (_, _, _, R, sigma) in zip(self.model(x), self.channels):
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


def run_de(problem, maxiter=200, popsize=15, tol=1e-3, polish=True,
           seed=None, callback=None, cancelled=None):
    """Differential evolution on `problem`.

    callback(x, cost, generation) is called after every generation with the
    best parameters so far.  cancelled() is polled at every cost evaluation;
    when it returns True the run stops at once, without polishing.  Returns
    (x, cost, generations, message) with x in physical units."""
    best = [problem.to_x(problem.x0()), None, 0]      # x, cost, generation

    def cost(u):
        if cancelled is not None and cancelled():
            raise FitCancelled
        return problem(u)

    def cb(intermediate_result):
        best[:] = [problem.to_x(intermediate_result.x),
                   float(intermediate_result.fun), best[2] + 1]
        if callback is not None:
            callback(*best)

    try:
        res = differential_evolution(
            cost, [(0.0, 1.0)] * len(problem.params), x0=problem.x0(),
            maxiter=maxiter, popsize=popsize, tol=tol, polish=polish,
            seed=seed, callback=cb)
    except FitCancelled:
        if best[1] is None:                  # stopped before one generation
            best[1] = problem(problem.to_u(best[0]))
        return best[0], best[1], best[2], 'stopped'
    return problem.to_x(res.x), float(res.fun), best[2], res.message
