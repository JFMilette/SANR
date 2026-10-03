"""
The fit problem shared by every fitting method (fit.de, fit.dream): which
parameters vary, the measured data, and the model and cost of a parameter
vector.

PARAMETERS -- the varied ones are Stack.free_parameters(): every Layer.fit
and Stack.fit (scale, background) entry with vary = True, bounded by its
[min, max].  The methods work in normalised coordinates u in [0, 1]^n,
x = min + u (max - min) (FitProblem.to_x / to_u), so SLDs (~1e-6) and
thicknesses (~100) are on the same footing.

DATA -- a list of measured channels, each a dict
  {'Q', 'R', 'dR', 'P0': vector, 'P': vector or None, 'name' (optional),
   'norm' (optional, default 1)}
with (P0, P) its polarisation pair (see model.stack POLARISATION and
model.polarisation); 'name' labels it in warnings; 'norm' is a fixed factor
on that channel's model before the background (Licorne's Norm_factor, see
model.licorne_io NORMALISATION), never a fit parameter.  Channels measured along
different axes can be fitted together.  The model is ONE
Stack.reflectivities call per trial on the union of every channel's Q
points, with the stack's own resolution, scale and background (added once
per channel).  With the 'licorne' resolution scheme the convolution depends
on the grid it runs on (model.stack RESOLUTION), as in Licorne, which
computes every channel on one q.dat.  So either `grid` (e.g. the Q points of
one chosen channel) is that one grid: the model is computed on it and each
channel takes it at its own points (exactly on the grid's points, by
interpolation of log R between them, the end value outside it, with a
warning); or, without `grid`, each channel's grid is all its finite Q points, the Q
range and unusable R or dR only choosing the fitted ones among them
(Licorne's QPOINTS), and each distinct grid is evaluated on its own (one
call per grid; a warning says so when the channels' grids differ).

COST -- mean over all points of the squared residual, plus, for parameters
with a prior from other measurements (fit.priors), -2 (log p - its maximum)
/ N of each: the DE optimum is then DREAM's MAP (zero when every such
value sits at its expected one); residuals() never includes it:
  'chi2' : (R_model - R) / dR, i.e. the reduced chi^2 up to the number of
           free parameters.  Points with dR <= 0 use dR = R (relative
           residual) so data without errors can still be fitted.
  'log'  : log10(R_model) - log10(R), equal weight per decade; points with
           R <= 0 are ignored.
DREAM (fit.dream.dream_fit) uses the 'chi2' residuals as a Gaussian
likelihood.

A FitProblem holds a copy of the stack and plain arrays: it is pickled to
the worker processes of either method.
"""

import copy
import warnings

import numpy as np

from fit.priors import prior_terms


COSTS = ('chi2', 'log')


def _interp(q, Q, R):
    """R (on the increasing grid Q) at the points q: linear in log R when R
    is positive everywhere (it falls off by decades), else linear; the end
    value outside the grid."""
    if np.all(R > 0):
        return np.exp(np.interp(q, Q, np.log(R)))
    return np.interp(q, Q, R)


class FitProblem:
    """Cost of a copy of `stack` against `data` as a function of the
    normalised free parameters."""

    def __init__(self, stack, data, cost='chi2', qmin=-np.inf, qmax=np.inf,
                 grid=None):
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
                '%s.%s' % ('stack' if i is None else
                           self.stack.layers[i].name, a)
                for i, a, *_ in bad))
        # HALPERIN: only rho cos(phi) reaches the neutrons
        for i in sorted({p[0] for p in self.params if p[0] is not None}):
            varied = {p[1] for p in self.params if p[0] == i}
            if {'MSLD_rho', 'MSLD_phi'} <= varied:
                warnings.warn(
                    '%s: MSLD ρ and the out-of-plane angle MSLD θ '
                    '(MSLD_phi) are both varied, but only ρ cos θ is '
                    'measurable (Halperin) and the sign of θ is never '
                    'determined' % self.stack.layers[i].name,
                    stacklevel=2)
        self.lo = np.array([p[3] for p in self.params], dtype=float)
        self.hi = np.array([p[4] for p in self.params], dtype=float)
        # Prior or None per parameter (fit.priors)
        self.priors = prior_terms(self.stack, self.params)

        self.channels = []                 # [(name, idx, R, sigma)]
        self.pairs = []                    # (P0, P) of each channel
        self.norms = []                    # fixed factor of each channel
        Qs = []
        full = []                          # (finite Q, fitted ones among them)
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
                self.norms.append(float(e.get('norm', 1.0)))
                self.channels.append((e.get('name', 'channel %d' % k),
                                      R[ok], sigma))
                Qs.append(Q[ok])
                fin = np.isfinite(Q)
                full.append((Q[fin], np.flatnonzero(ok[fin])))
        if not self.channels:
            raise ValueError('no data points to fit in the chosen Q range')
        self.Q, inv = np.unique(np.concatenate(Qs), return_inverse=True)
        # indices of each channel's points in self.Q
        splits = np.cumsum([len(q) for q in Qs])[:-1]
        self.channels = [(n, idx, R, s) for (n, R, s), idx
                         in zip(self.channels, np.split(inv, splits))]
        self.npoints = sum(len(ch[2]) for ch in self.channels)
        self.grid = None if grid is None else np.asarray(grid, dtype=float)
        self.groups = self._groups(Qs, full)

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

    def _groups(self, Qs, full):
        """[(Q grid, [(channel k, indices of its points in the grid or None,
        its points)])]: the union of every channel's Q, or with the
        'licorne' resolution scheme the one `grid`, else one grid per
        distinct channel grid (see DATA).  Indices None: the channel's
        points are interpolated on the grid.  Qs: each channel's fitted Q;
        full: each channel's (finite Q, indices of the fitted ones), the
        Licorne grid even outside the Q range or where R, dR are unusable,
        as Licorne computes the model on every point of q.dat before it
        selects QPOINTS (fit_reflection.m)."""
        res = self.stack.resolution
        if not (res.get('enabled') and res.get('scheme') == 'licorne'):
            return [(self.Q, [(k, ch[1], q) for k, (ch, q)
                              in enumerate(zip(self.channels, Qs))])]
        # a repeated Q (stitched TOF angles) is kept, as in Licorne's resolut
        if self.grid is not None:
            g = np.sort(self.grid[np.isfinite(self.grid)])
            members, out = [], []
            for k, (q, ch) in enumerate(zip(Qs, self.channels)):
                i = np.clip(np.searchsorted(g, q), 0, len(g) - 1)
                members.append((k, i if np.array_equal(g[i], q) else None,
                                q))
                n = int(np.sum((q < g[0]) | (q > g[-1])))
                if n:
                    out.append('%s (%d)' % (ch[0], n))
            if out:
                warnings.warn('Licorne resolution scheme: points outside the '
                              'chosen Q grid (%.4g - %.4g) take its end '
                              'value: %s' % (g[0], g[-1], ', '.join(out)),
                              stacklevel=3)
            return [(g, members)]
        grids = {}
        for k, ((qf, sel), q) in enumerate(zip(full, Qs)):
            order = np.argsort(qf, kind='stable')
            g = qf[order]
            pos = np.empty(len(qf), dtype=int)
            pos[order] = np.arange(len(qf))
            grids.setdefault(g.tobytes(), (g, []))[1].append((k, pos[sel], q))
        if len(grids) > 1:
            warnings.warn('Licorne resolution scheme: the channels are on '
                          'different Q grids; each grid is convolved on its '
                          'own', stacklevel=3)
        return list(grids.values())

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
            setattr(stack.owner(i), attr, float(v))

    # -- cost ---------------------------------------------------------------
    def model(self, x):
        """Model reflectivity of each channel at its data points."""
        self.apply(x)
        self.stack.build_sublayers()
        out = [None] * len(self.channels)
        for Q, members in self.groups:
            norms = [self.norms[k] for k, *_ in members]
            R = self.stack.reflectivities(
                Q, [self.pairs[k] for k, *_ in members], warn=False,
                norms=None if all(n == 1 for n in norms) else norms)
            for c, (k, idx, q) in enumerate(members):
                out[k] = R[idx, c] if idx is not None else \
                    _interp(q, Q, R[:, c])
        return out

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

    def prior_penalty(self, x):
        """sum of -2 (log p - its maximum) over the parameters' priors."""
        return float(sum(p.penalty(v) for p, v in zip(self.priors, x)
                         if p is not None))

    def __call__(self, u):
        x = self.to_x(u)
        cost = float(np.mean(self.residuals(x)**2))
        if any(self.priors):
            cost += self.prior_penalty(x) / self.npoints
        return cost
