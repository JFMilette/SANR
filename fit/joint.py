"""
Joint fit of several datasets (e.g. one sample measured at several fields
or temperatures) with one model: a JointProblem has the interface of
fit.problem.FitProblem over one FitProblem per dataset, so that DE
(fit.de), multistart (fit.multistart) and DREAM (fit.dream) take either.

PARAMETERS -- each dataset is fitted with its own stack: the model linked
to it (fit.panel MODELS), so the datasets may vary different parameters
(Stack.free_parameters(), named as fit.problem.parameter_names).  The
parameters are matched by name across the datasets; in each dataset that
has it, each one is
  'shared'  fitted, one value for every dataset sharing it (the default)
  'own'     fitted, a value of this dataset alone
  'fixed'   not fitted in this dataset: it keeps its value in that
            dataset's stack (its model's)
`modes` {dataset name: {parameter name: mode}} gives those that are not
'shared'; `fixed` holds the values a 'fixed' parameter takes, one row per
dataset of its own parameters.  The joint parameters are, for each name in
the order the datasets first give it, the shared one (if any dataset shares
it), then one per dataset where it is 'own', named 'name [dataset]'; each
takes its bounds, prior and start from one dataset it stands for
(`origin`), i.e. from that dataset's model: a shared one from the dataset
`sources` names for it, else the first sharing it.  part_x(x, k) gives the values
of dataset k's own parameters.

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

    def __init__(self, parts, modes=None, fixed=None, sources=None):
        """parts: [(dataset name, FitProblem)] with the same cost and
        target, each built from its dataset's own stack (the model linked to
        it: PARAMETERS); fixed: the values a dataset's parameters keep where
        'fixed', one row per dataset of its own parameters (or one row for
        all, when they have the same parameters); default: their values in
        that dataset's FitProblem, i.e. its model's; sources: {parameter
        name: dataset} a shared parameter takes its bounds, start and prior
        from (default, or when that dataset does not share it: the first
        one sharing it)."""
        if not parts:
            raise ValueError('no dataset to fit')
        self.part_names = [n for n, _ in parts]
        if len(set(self.part_names)) != len(self.part_names):
            raise ValueError('two datasets have the same name')
        self.parts = [p for _, p in parts]
        first = self.parts[0]
        for p in self.parts[1:]:
            if p.cost != first.cost or p.target != first.target:
                raise ValueError('the datasets are not fitted with the same '
                                 'cost and target')
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
        self.sources = {a: n for a, n in (sources or {}).items()
                        if n in self.part_names}
        if fixed is None:
            fixed = [[q[2] for q in p.params] for p in self.parts]
        elif np.ndim(fixed) == 1 and len(fixed) and np.ndim(fixed[0]) == 0:
            fixed = [fixed] * len(self.parts)       # one row for all
        self.fixed = [np.array(f, dtype=float) for f in fixed]
        if [len(f) for f in self.fixed] != [len(p.params)
                                            for p in self.parts]:
            raise ValueError('the fixed values do not match the datasets\' '
                             'parameters')

        # every dataset's parameters, by name, in the order they first
        # appear; index[k][i]: joint index of parameter i of dataset k,
        # None where it is fixed; origin[j]: (dataset, its parameter) the
        # joint parameter j takes its bounds, start and prior from
        names = []
        for p in self.parts:
            names += [n for n in p.names if n not in names]
        self.params, self.names, self.scopes, self.priors = [], [], [], []
        self.origin = []
        self.index = [[None] * len(p.params) for p in self.parts]
        for name in names:
            has = [k for k, p in enumerate(self.parts) if name in p.names]
            m = {k: self.modes[self.part_names[k]].get(name, 'shared')
                 for k in has}
            groups = [(name, None, [k for k in has if m[k] == 'shared'])]
            groups += [('%s [%s]' % (name, self.part_names[k]),
                        self.part_names[k], [k])
                       for k in has if m[k] == 'own']
            for label, scope, ks in groups:
                if not ks:
                    continue
                src = self.sources.get(name) if scope is None else None
                k0 = next((k for k in ks if self.part_names[k] == src),
                          ks[0])
                i0 = self.parts[k0].names.index(name)
                for k in ks:
                    self.index[k][self.parts[k].names.index(name)] = \
                        len(self.params)
                self.params.append(self.parts[k0].params[i0])
                self.names.append(label)
                self.scopes.append(scope)      # None: shared
                self.priors.append(self.parts[k0].priors[i0])
                self.origin.append((k0, i0))
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
                if getattr(p, 'spread_nodes', None):
                    p.stack.thickness_spread_nodes = p.spread_nodes = \
                        max(counts)
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
