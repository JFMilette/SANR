"""
The fit problem shared by every fitting method (fit.de, fit.dream): which
parameters vary, the measured data, and the model and cost of a parameter
vector.

PARAMETERS -- the varied ones are Stack.free_parameters(): every Layer.fit
and Stack.fit (scale, background) entry with vary = True, bounded by its
[min, max], with tied layers (model.stack ties) as one parameter.  The methods work in normalised coordinates u in [0, 1]^n,
x = min + u (max - min) (FitProblem.to_x / to_u), so SLDs (~1e-6) and
thicknesses (~100) are on the same footing.

DATA -- a list of measured channels, each a dict
  {'Q', 'R', 'dR', 'P0': vector, 'P': vector or None, 'name' (optional),
   'norm' (optional, default 1), 'dQ' (optional)}
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

RESOLUTION -- with the stack's resolution on in the 'data' mode (model.stack
RESOLUTION) and the quadrature scheme, every point is averaged over its own
dQ (Stack.data_sigma; the dQ table where it has none): the model is then ONE
call on every channel's points one after the other, a Q repeated in two
channels or two stitched angles taken with each one's dQ.  With the
'licorne' scheme (any of its modes 1-3) each grid is convolved with the dQ
of its own points: those of the channel whose finite Q are the grid (the
dQ table where a point has none, or where no channel's Q are the grid,
e.g. a grid shared by channels whose dQ differ: the first one's is used).

COST -- mean over all points of the squared residual, plus, for parameters
with a prior from other measurements (fit.priors), -2 (log p - its maximum)
/ N of each: the DE optimum is then DREAM's MAP (zero when every such
value sits at its expected one); residuals() never includes it:
  'chi2' : (R_model - R) / dR, i.e. the reduced chi^2 up to the number of
           free parameters.  Points with dR <= 0 use dR = R (relative
           residual) so data without errors can still be fitted.  An
           error_floor f > 0 uses sqrt(dR^2 + (f R)^2) instead: a relative
           error (e.g. of the normalisation between angles) that counting
           errors leave out.
  'log'  : log10(R_model) - log10(R), equal weight per decade; points with
           R <= 0 are ignored.
DREAM (fit.dream.dream_fit) uses the 'chi2' residuals as a Gaussian
likelihood.

TARGET -- what the residuals compare (DE, L-BFGS-B and multistart; DREAM
always samples the channels):
  'channels' : every channel's R (the default).
  'sum'      : (R+ + R-)/2 of the channels named 'R+' and 'R-' (else 'R++'
               and 'R--'), error sqrt(dR+^2 + dR-^2)/2.  Its magnetic part
               is only of second order in M (the first-order cross term
               cancels): nearly the nuclear reflectivity, for the NSLD of
               datasets whose magnetism differs.  Nearly only: a 20 % MSLD
               error costs ~2 % of what it costs the channels at MSLD /
               NSLD ~ 0.06, ~20 % at 0.25 (tests/test_fit_target.py), so
               the MSLD should still sit near its value.
  'asym'     : the spin asymmetry SA = (R+ - R-)/(R+ + R-) of the same pair,
               error 2 sqrt((R- dR+)^2 + (R+ dR-)^2)/(R+ + R-)^2; first
               order in M, and the scale cancels in it (up to the
               background): for the MSLD.  With the 'log' cost its residual
               is SA_model - SA (SA can be negative: no log).
The two channels are combined at their common Q points (row by row when
their fitted points are the same, else matched to 1e-6 Q), only where
both are fitted; the model is computed as for 'channels' and combined the
same way.  The dataset's other channels (spin-flip R+-, R-+) are still
fitted one by one, as with 'channels' (`others`); a dataset without the
pair cannot be fitted.  Fitting 'sum' and
'asym' together would hold the same information as the two channels:
each alone is a deliberate choice of part of it.

THICKNESS SPREAD -- with the stack's thickness_spread_nodes 'auto', the
node count (model.stack NODES) would change with sigma_t and the cost jump
where it does: the fit's copy of the stack gets it frozen once, by the
convergence check at the upper bound of sigma_t (its value when not
fitted), for the stack's distribution and target, on the largest Q
computed (+ 3 sigma_Q and the largest |Q shift| allowed); the caller's
stack keeps 'auto'.  The target and distribution are model choices, not
fit parameters: compare them by separate fits (BIC, bounds, DREAM).  A message says how many nodes, i.e. how many
times slower each evaluation is.  fit.joint gives every dataset the
largest count.  With a mixture of models (model.stack MIXTURE) there is
no thickness spread: a stack with both is refused here.

A FitProblem holds a copy of the stack and plain arrays: it is pickled to
the worker processes of either method.
"""

import copy
import warnings

import numpy as np

from fit.priors import prior_terms
from model.stack import spread_factors


COSTS = ('chi2', 'log')
TARGETS = ('channels', 'sum', 'asym')
# the (up, down) channels a 'sum' or 'asym' target combines, by preference
TARGET_PAIRS = (('R+', 'R-'), ('R++', 'R--'))
TARGET_NAMES = {'sum': '(R+ + R−)/2', 'asym': 'the spin asymmetry'}


def parameter_names(stack, params):
    """'layer.attr' of each free_parameters() entry ('stack.attr' for the
    stack's own, 'tie(Fe.thickness, Co.thickness)' for a tie); layers
    sharing a name get their index: '#2 Fe.thickness'."""
    names = [l.name for l in stack.layers]
    out = []
    for i, attr, *_ in params:
        if i is None:
            out.append('stack.%s' % attr)
        elif isinstance(i, tuple):
            out.append(stack.entry_name(i))
        elif names.count(names[i]) > 1:
            out.append('#%d %s.%s' % (i, names[i], attr))
        else:
            out.append('%s.%s' % (names[i], attr))
    return out


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
                 grid=None, error_floor=0.0, target='channels'):
        if cost not in COSTS:
            raise ValueError('unknown cost %r' % cost)
        if target not in TARGETS:
            raise ValueError('unknown target %r' % target)
        self.target = target
        # the caller's stack may be edited while the fit runs
        self.stack = copy.deepcopy(stack)
        self.cost = cost
        self.params = self.stack.free_parameters()
        self.names = parameter_names(self.stack, self.params)
        if not self.params:
            raise ValueError('no parameter is varied: tick Fit on at least '
                             'one layer parameter in the Simulation tab')
        bad = [p for p in self.params if not p[3] < p[4]]
        if bad:
            raise ValueError('empty bounds for %s' % ', '.join(
                self.stack.entry_name(i) if isinstance(i, tuple) else
                '%s.%s' % ('stack' if i is None else
                           self.stack.layers[i].name, a)
                for i, a, *_ in bad))
        # HALPERIN: only rho cos(phi) reaches the neutrons
        for i in sorted({p[0] for p in self.params
                         if isinstance(p[0], int)}):
            varied = {p[1] for p in self.params if p[0] == i}
            lo = [p[3] for p in self.params if p[:2] == (i, 'MSLD_rho')]
            if {'MSLD_rho', 'MSLD_theta'} <= varied and lo and lo[0] < 0:
                warnings.warn(
                    '%s: MSLD ρ may be negative and the in-plane angle MSLD '
                    'φ (MSLD_theta) is varied too: -ρ at φ is +ρ at φ + '
                    '180°, so the fit has two equivalent minima; keep ρ >= '
                    '0 or fix φ' % self.stack.layers[i].name, stacklevel=2)
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
        self.error_floor = float(error_floor)
        self.raw_sigma = []                # each channel's dR, no floor
        Qs = []
        dQs = []                           # each channel's dQ, NaN if none
        full = []                          # (finite Q, fitted ones among them)
        full_dq = []                       # dQ of each channel's finite Q
        for k, e in enumerate(data):
            Q, R, dR = (np.asarray(e[c], dtype=float) for c in ('Q', 'R', 'dR'))
            dQ = np.asarray(e['dQ'], dtype=float) if e.get('dQ') is not None \
                else np.full_like(Q, np.nan)
            ok = np.isfinite(Q) & np.isfinite(R) & (Q >= qmin) & (Q <= qmax)
            if cost == 'log':
                ok &= R > 0
                sigma = None
            else:
                dR = np.where(np.isfinite(dR) & (dR > 0), dR, np.abs(R))
                ok &= dR > 0
                sigma = dR[ok]
                self.raw_sigma.append(sigma)
                if self.error_floor > 0:
                    sigma = np.hypot(sigma, self.error_floor * R[ok])
            if ok.any():
                P = e.get('P')
                self.pairs.append((np.asarray(e['P0'], dtype=float),
                                   None if P is None else
                                   np.asarray(P, dtype=float)))
                self.norms.append(float(e.get('norm', 1.0)))
                self.channels.append((e.get('name', 'channel %d' % k),
                                      R[ok], sigma))
                Qs.append(Q[ok])
                dQs.append(dQ[ok])
                fin = np.isfinite(Q)
                full.append((Q[fin], np.flatnonzero(ok[fin])))
                full_dq.append(dQ[fin])
        if not self.channels:
            raise ValueError('no data points to fit in the chosen Q range')
        self.Q, inv = np.unique(np.concatenate(Qs), return_inverse=True)
        # indices of each channel's points in self.Q
        splits = np.cumsum([len(q) for q in Qs])[:-1]
        self.channels = [(n, idx, R, s) for (n, R, s), idx
                         in zip(self.channels, np.split(inv, splits))]
        self.npoints = sum(len(ch[2]) for ch in self.channels)
        # 'sum' / 'asym': (up k, down k, their points, value, error or
        # None), and the channels fitted on their own besides
        self.combined = None
        self.others = list(range(len(self.channels)))
        if target != 'channels':
            self.combined = self._combine_data()
            self.others = [k for k in self.others
                           if k not in self.combined[:2]]
            self.npoints = len(self.combined[4]) + sum(
                len(self.channels[k][2]) for k in self.others)
            if any(p[:2] == (None, 'scale') for p in self.params) and \
                    target == 'asym':
                warnings.warn('the spin asymmetry hardly depends on the '
                              'scale (only through the background): fix it',
                              stacklevel=2)
        self.grid = None if grid is None else np.asarray(grid, dtype=float)
        self.groups = self._groups(Qs, full)
        # sigma_Q of each group's points, None: the resolution's own
        self.sigmas = [None] * len(self.groups)
        if self._pointwise():
            self.sigmas = [self.stack.data_sigma(self.groups[0][0],
                                                 np.concatenate(dQs))]
        elif self._licorne_data():
            self.sigmas = [self._grid_sigma(g, members, full, full_dq)
                           for g, members in self.groups]

        self.stack.check_spread_or_mixture()
        self.spread_nodes = self._freeze_spread_nodes()

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

    def _bound(self, attr, which):
        """The stack parameter attr's fit bound (which: 3 min, 4 max) if it
        is varied, else None."""
        for p in self.params:
            if p[:2] == (None, attr):
                return p[which]
        return None

    def _freeze_spread_nodes(self):
        """The thickness spread's node count, frozen on the fit's copy of
        the stack when it is 'auto' (see THICKNESS SPREAD); None without a
        spread."""
        st = self.stack
        hi = self._bound('thickness_spread', 4)
        shift = [abs(st.q_shift)] + [abs(b) for b in (
            self._bound('q_shift', 3), self._bound('q_shift', 4))
            if b is not None]
        qmax = sigma = 0.0
        for (Q, _), sg in zip(self.groups, self.sigmas):
            qmax = max(qmax, float(np.max(np.abs(Q))))
            sg = st.resolution_sigma(Q) if sg is None else sg
            sigma = max(sigma, float(np.max(sg, initial=0.0)))
        self.spread_qmax = qmax + max(shift) + 3 * sigma
        s = st.thickness_spread if hi is None else hi
        if not s > 0:
            return None
        n = st.thickness_spread_nodes = st.spread_count(self.spread_qmax, s)
        # a spread too wide for its upper bound (a Gaussian reaching zero
        # thickness): refused now, not in the middle of the fit
        spread_factors(s, st.thickness_spread_dist, n)
        warnings.warn('thickness spread (%s, %s): %d nodes, each evaluation '
                      'about %d× slower'
                      % (st.spread_target_name(), st.thickness_spread_dist,
                         n, n), stacklevel=3)
        return n

    def _combine_data(self):
        """The pair of a 'sum' / 'asym' target and its data (see TARGET)."""
        names = [ch[0] for ch in self.channels]
        pair = next((p for p in TARGET_PAIRS if all(n in names for n in p)),
                    None)
        if pair is None:
            raise ValueError('fitting %s needs both R+ and R− (or R++ and '
                             'R−−) fitted; the channels are %s'
                             % (TARGET_NAMES[self.target], ', '.join(names)))
        a, b = (names.index(n) for n in pair)
        qa, qb = self.Q[self.channels[a][1]], self.Q[self.channels[b][1]]
        if len(qa) == len(qb) and np.array_equal(qa, qb):
            ia = ib = np.arange(len(qa))
        else:
            # each up point with the nearest down point, if within 1e-6 Q
            j = np.abs(qa[:, None] - qb[None, :]).argmin(axis=1)
            near = np.abs(qb[j] - qa) <= 1e-6 * np.abs(qa)
            ia, ib = np.flatnonzero(near), j[near]
            ib, first = np.unique(ib, return_index=True)
            ia = ia[first]
            n = len(qa) + len(qb) - 2 * len(ia)
            if n:
                warnings.warn('%s and %s: %d point(s) without a partner at '
                              'the same Q are left out'
                              % (pair[0], pair[1], n), stacklevel=3)
        Ra, Rb = self.channels[a][2][ia], self.channels[b][2][ib]
        sa, sb = self.channels[a][3], self.channels[b][3]
        T = Ra + Rb
        ok = T > 0
        if self.target == 'sum':
            v = T / 2
            s = None if sa is None else np.hypot(sa[ia], sb[ib]) / 2
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                v = (Ra - Rb) / T
                s = None if sa is None else \
                    2 * np.hypot(Rb * sa[ia], Ra * sb[ib]) / T**2
        if s is not None:
            ok &= np.isfinite(s) & (s > 0)
        if not ok.any():
            raise ValueError('no Q point where %s and %s can be combined'
                             % pair)
        return (a, b, ia[ok], ib[ok], v[ok], None if s is None else s[ok])

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
        if self._pointwise():
            # every channel's points in turn, each with its own dQ
            end = np.cumsum([len(q) for q in Qs])
            return [(np.concatenate(Qs),
                     [(k, np.arange(e - len(q), e), q) for k, (q, e)
                      in enumerate(zip(Qs, end))])]
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

    def _licorne_data(self):
        """The 'licorne' scheme with the data's dQ (see RESOLUTION)."""
        res = self.stack.resolution
        return bool(res.get('enabled')) and res.get('mode') == 'data' and \
            res.get('scheme') == 'licorne'

    def _grid_sigma(self, g, members, full, full_dq):
        """sigma_Q at the points of Licorne grid g: the dQ of the first
        member channel whose finite Q are g, None (the dQ table) if none
        is."""
        for k, *_ in members:
            q, dq = full[k][0], full_dq[k]
            order = np.argsort(q, kind='stable')
            if np.array_equal(q[order], g):
                return self.stack.data_sigma(g, dq[order])
        return None

    def _pointwise(self):
        """Each point averaged over its own dQ (see RESOLUTION)."""
        res = self.stack.resolution
        return bool(res.get('enabled')) and res.get('mode') == 'data' and \
            res.get('scheme', 'quadrature') != 'licorne'

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
        for (Q, members), sigma in zip(self.groups, self.sigmas):
            norms = [self.norms[k] for k, *_ in members]
            R = self.stack.reflectivities(
                Q, [self.pairs[k] for k, *_ in members], warn=False,
                norms=None if all(n == 1 for n in norms) else norms,
                sigma=sigma)
            for c, (k, idx, q) in enumerate(members):
                out[k] = R[idx, c] if idx is not None else \
                    _interp(q, Q, R[:, c])
        return out

    def residuals(self, x):
        model = self.model(x)
        out = [] if self.combined is None else \
            [self._combined_residuals(model)]
        for k in self.others:
            m, (_, _, R, sigma) = model[k], self.channels[k]
            if self.cost == 'log':
                with np.errstate(divide='ignore', invalid='ignore'):
                    out.append(np.log10(m) - np.log10(R))
            else:
                out.append((m - R) / sigma)
        r = np.concatenate(out)
        # a model that cannot be evaluated (NaN, R <= 0 in log) is heavily
        # penalised rather than aborting the fit
        return np.where(np.isfinite(r), r, 1e3)

    def _combined_residuals(self, model):
        """Residuals of a 'sum' / 'asym' target (see TARGET)."""
        a, b, ia, ib, v, s = self.combined
        ma, mb = model[a][ia], model[b][ib]
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.target == 'sum':
                m = (ma + mb) / 2
                if self.cost == 'log':
                    return np.log10(m) - np.log10(v)
            else:
                m = (ma - mb) / (ma + mb)
            return m - v if s is None else (m - v) / s

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
