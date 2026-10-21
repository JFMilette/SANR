"""
Joint fit of several datasets (e.g. one sample measured at several fields
or temperatures) with one model: a JointProblem has the interface of
fit.problem.FitProblem over one FitProblem per dataset, so that DE
(fit.de), multistart (fit.multistart) and DREAM (fit.dream) take either.

PARAMETERS -- every dataset sees the stack's varied parameters
(Stack.free_parameters(), named as fit.problem.parameter_names).  In each
dataset each one is
  'shared'  fitted, one value for every dataset sharing it (the default)
  'own'     fitted, a value of this dataset alone
  'fixed'   not fitted in this dataset: a value of its own, `values`
            {dataset name: {parameter name: value}}, else its value when
            the fit started
`modes` {dataset name: {parameter name: mode}} gives those that are not
'shared'; `fixed` holds the values a 'fixed' parameter takes, one row per
dataset (K, n).  The joint parameters are, for each stack parameter in turn, the
shared one (if any dataset shares it), then one per dataset where it is
'own', named 'name [dataset]'; each has the parameter's bounds and prior and
starts at its current value.  part_x(x, k) gives the stack's parameter
values that dataset k sees.

COST -- the residuals of every dataset one after the other: the mean over
all their points (every point weighs the same, whichever dataset it is
in), plus the priors / N, as FitProblem; every dataset with the same
target (fit.problem TARGET).  The channels (`channels`, `pairs`,
`norms`, `raw_sigma`, their points in `Q`) are every dataset's in turn,
`channel_part` the dataset of each.

The helpers parts, part_x and part_index take a FitProblem (one dataset) or
a JointProblem alike.
"""

import numpy as np

from fit.problem import FitProblem

MODES = ('shared', 'own', 'fixed')


class JointProblem:
    """Cost of one stack against several datasets (see the module
    docstring)."""

    def __init__(self, parts, modes=None, fixed=None, values=None):
        """parts: [(dataset name, FitProblem)], built from the same stack
        with the same cost; fixed: the stack parameters' values for 'fixed',
        (n,) for every dataset or (K, n) one row per dataset (default: their
        values in the FitProblems, i.e. at the start); values {dataset:
        {parameter name: value}} replace them where that parameter is
        'fixed' in that dataset (a value given for another mode is
        ignored)."""
        if not parts:
            raise ValueError('no dataset to fit')
        self.part_names = [n for n, _ in parts]
        if len(set(self.part_names)) != len(self.part_names):
            raise ValueError('two datasets have the same name')
        self.parts = [p for _, p in parts]
        first = self.parts[0]
        for p in self.parts[1:]:
            if p.names != first.names or p.cost != first.cost or \
                    p.target != first.target:
                raise ValueError('the datasets are not fitted with the same '
                                 'parameters, cost and target')
        self.stack, self.cost = first.stack, first.cost
        self.target = first.target
        self.error_floor, self.grid = first.error_floor, None
        modes = modes or {}
        for m in modes.values():
            bad = set(m.values()) - set(MODES)
            if bad:
                raise ValueError('unknown mode %s (one of %s)'
                                 % (', '.join(map(repr, bad)),
                                    ', '.join(MODES)))
        self.modes = {n: {a: v for a, v in (modes.get(n) or {}).items()
                          if v != 'shared'} for n in self.part_names}
        base = first.params
        f = np.array([p[2] for p in base], dtype=float) \
            if fixed is None else np.asarray(fixed, dtype=float)
        self.fixed = np.array(np.broadcast_to(f, (len(self.parts), len(base))))
        for k, n in enumerate(self.part_names):
            for name, v in ((values or {}).get(n) or {}).items():
                if name not in first.names:
                    raise ValueError('a value for an unknown parameter %r'
                                     % name)
                if self.modes[n].get(name) == 'fixed':
                    self.fixed[k, first.names.index(name)] = float(v)

        # the joint parameters; index[k][i]: joint index of the stack's
        # parameter i in dataset k, None where it is fixed
        self.params, self.names, self.scopes, self.priors = [], [], [], []
        self.index = [[None] * len(base) for _ in self.parts]
        for i, (p, name, prior) in enumerate(zip(base, first.names,
                                                 first.priors)):
            m = [self.modes[s].get(name, 'shared') for s in self.part_names]
            groups = [(name, None, [k for k, v in enumerate(m)
                                    if v == 'shared'])]
            groups += [('%s [%s]' % (name, s), s, [k])
                       for k, (v, s) in enumerate(zip(m, self.part_names))
                       if v == 'own']
            for label, scope, ks in groups:
                if not ks:
                    continue
                for k in ks:
                    self.index[k][i] = len(self.params)
                self.params.append(p)
                self.names.append(label)
                self.scopes.append(scope)      # None: shared
                self.priors.append(prior)
        if not self.params:
            raise ValueError('no parameter is fitted in any dataset')
        self.lo = np.array([p[3] for p in self.params], dtype=float)
        self.hi = np.array([p[4] for p in self.params], dtype=float)
        self._maps = [(np.array([j for j in idx if j is not None], dtype=int),
                       np.array([i for i, j in enumerate(idx)
                                 if j is not None], dtype=int))
                      for idx in self.index]

        # a thickness spread: every dataset with the largest node count any
        # of them froze (fit.problem THICKNESS SPREAD)
        counts = [p.spread_nodes for p in self.parts
                  if getattr(p, 'spread_nodes', None)]
        if counts:
            for p in self.parts:
                p.stack.thickness_spread_nodes = p.spread_nodes = max(counts)
        self.channels, self.pairs, self.norms = [], [], []
        self.raw_sigma, self.channel_part = [], []
        Qs, off = [], 0
        for k, p in enumerate(self.parts):
            self.channels += [(name, idx + off, R, s)
                              for name, idx, R, s in p.channels]
            self.channel_part += [k] * len(p.channels)
            self.pairs += p.pairs
            self.norms += p.norms
            self.raw_sigma += p.raw_sigma
            Qs.append(p.Q)
            off += len(p.Q)
        self.Q = np.concatenate(Qs)
        self.npoints = sum(p.npoints for p in self.parts)

    # the same as FitProblem's
    to_x = FitProblem.to_x
    to_u = FitProblem.to_u
    x0 = FitProblem.x0
    prior_penalty = FitProblem.prior_penalty
    __call__ = FitProblem.__call__

    def part_x(self, x, k):
        """Values of the stack's parameters (parts[k].params) that dataset
        k sees at the joint parameters' values x."""
        v = self.fixed[k].copy()
        J, I = self._maps[k]
        v[I] = np.asarray(x, dtype=float)[J]
        return v

    def apply(self, x, stack=None, part=0):
        """Write dataset `part`'s values at x into `stack` (default: that
        dataset's copy)."""
        self.parts[part].apply(self.part_x(x, part), stack)

    def model(self, x):
        """Model reflectivity of every dataset's channels in turn."""
        out = []
        for k, p in enumerate(self.parts):
            out += p.model(self.part_x(x, k))
        return out

    def residuals(self, x):
        return np.concatenate([p.residuals(self.part_x(x, k))
                               for k, p in enumerate(self.parts)])


def parts(problem):
    """The FitProblem of each dataset."""
    return problem.parts if isinstance(problem, JointProblem) else [problem]


def part_x(problem, x, k=0):
    """Values of the stack's parameters (parts(problem)[k].params) for
    dataset k at x (extra entries of x, e.g. DREAM's error scale, are
    ignored)."""
    if isinstance(problem, JointProblem):
        return problem.part_x(x, k)
    return np.asarray(x, dtype=float)[:len(problem.params)]


def part_index(problem, k=0):
    """Index in x of each of the stack's parameters in dataset k, None
    where it is not fitted there."""
    if isinstance(problem, JointProblem):
        return list(problem.index[k])
    return list(range(len(problem.params)))
