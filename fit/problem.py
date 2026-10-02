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
on the grid it runs on (model.stack RESOLUTION), so each distinct grid of
fitted points is evaluated on its own instead (one call per grid; a warning
says so when the channels' grids differ).

COST -- mean over all points of the squared residual:
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

        self.channels = []                 # [(name, idx, R, sigma)]
        self.pairs = []                    # (P0, P) of each channel
        self.norms = []                    # fixed factor of each channel
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
                self.norms.append(float(e.get('norm', 1.0)))
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
        self.groups = self._groups(Qs)

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

    def _groups(self, Qs):
        """[(Q grid, [(channel k, indices of its points in the grid)])]: the
        union of every channel's Q, or one grid per distinct channel grid
        with the 'licorne' resolution scheme (see DATA)."""
        res = self.stack.resolution
        if not (res.get('enabled') and res.get('scheme') == 'licorne'):
            return [(self.Q, [(k, ch[1]) for k, ch
                              in enumerate(self.channels)])]
        grids = {}
        for k, (q, ch) in enumerate(zip(Qs, self.channels)):
            g = np.unique(q)
            if len(g) != len(q):
                raise ValueError('%s has repeated Q points: the Licorne '
                                 'resolution scheme needs distinct Q' % ch[0])
            grids.setdefault(g.tobytes(), (g, []))[1].append(
                (k, np.searchsorted(g, q)))
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
            norms = [self.norms[k] for k, _ in members]
            R = self.stack.reflectivities(
                Q, [self.pairs[k] for k, _ in members], warn=False,
                norms=None if all(n == 1 for n in norms) else norms)
            for c, (k, idx) in enumerate(members):
                out[k] = R[idx, c]
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

    def __call__(self, u):
        return float(np.mean(self.residuals(self.to_x(u))**2))
