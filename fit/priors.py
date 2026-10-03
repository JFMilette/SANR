"""
Priors from other measurements: what a parameter should be, known before
these data (e.g. a thickness from X-ray reflectivity), without a known
distribution.  The fit bounds stay the physical support; a prior says where
inside them the value is expected.

SPEC -- Layer.fit[attr]['prior'] (or Stack.fit[attr]['prior'] for scale and
background) is None or, in the parameter's own units (model.stack
FIT_PARAMS: Angstrom, A^-2, turns):
  {'kind': 'gauss',    'mu': expected, 'tau': tolerance}
  {'kind': 'soft_box', 'a': lower, 'b': upper, 's': softness}
complete() fills the defaults: tau = 0.1 |mu| when only mu is given (mu = 0
needs tau), s = 0.1 (b - a) when s is blank.

LOG DENSITIES, normalised over the real line (the hard bounds truncate
them, which is harmless for MCMC and DE):
  gauss     -((x - mu) / tau)^2 / 2 - log(tau sqrt(2 pi))
  soft_box  Z = (b - a) + s sqrt(2 pi);  -log Z inside [a, b], and
            -(dist / s)^2 / 2 - log Z outside (dist to the nearer edge)

USE -- fit.problem.FitProblem adds -2 (log p - its maximum) / N of every
prior to the DE cost, so that the DE optimum is DREAM's MAP;
fit.dream.dream_fit.PosteriorProblem adds log p to the log posterior.

TENSION -- tension(spec, x): a text when x lies more than 3 tau from mu
(gauss) or more than 3 s outside [a, b] (soft box): the data pull against
the other measurement.
"""

import math

KINDS = ('gauss', 'soft_box')
TENSION = 3.0             # in tau (gauss) or s (soft box)


def complete(spec):
    """spec with its defaults filled, checked; None stays None."""
    if spec is None:
        return None
    spec = dict(spec)
    kind = spec.get('kind')
    if kind == 'gauss':
        mu = float(spec['mu'])
        tau = spec.get('tau')
        if tau is None:
            if mu == 0:
                raise ValueError('a prior expecting 0 needs its tolerance ±')
            tau = 0.1 * abs(mu)
        if not float(tau) > 0:
            raise ValueError('the tolerance ± of a prior must be > 0')
        return {'kind': kind, 'mu': mu, 'tau': float(tau)}
    if kind == 'soft_box':
        a, b = float(spec['a']), float(spec['b'])
        if not b > a:
            raise ValueError('a soft box needs lower < upper')
        s = spec.get('s')
        s = 0.1 * (b - a) if s is None else float(s)
        if not s > 0:
            raise ValueError('the softness of a soft box must be > 0')
        return {'kind': kind, 'a': a, 'b': b, 's': s}
    raise ValueError('unknown prior kind %r (one of %s)'
                     % (kind, ', '.join(KINDS)))


def log_prior(spec, x):
    """log density of x under a complete spec."""
    if spec['kind'] == 'gauss':
        tau = spec['tau']
        return -0.5 * ((x - spec['mu']) / tau) ** 2 - \
            math.log(tau * math.sqrt(2 * math.pi))
    a, b, s = spec['a'], spec['b'], spec['s']
    lz = math.log((b - a) + s * math.sqrt(2 * math.pi))
    dist = a - x if x < a else x - b if x > b else 0.0
    return -0.5 * (dist / s) ** 2 - lz


def log_prior_max(spec):
    """The largest log density of a complete spec (at mu, or inside the
    box)."""
    return log_prior(spec, spec['mu'] if spec['kind'] == 'gauss'
                     else spec['a'])


class Prior:
    """log density of one parameter (picklable, for worker processes)."""

    def __init__(self, spec):
        self.spec = complete(spec)
        self.top = log_prior_max(self.spec)

    def __call__(self, x):
        return log_prior(self.spec, float(x))

    def penalty(self, x):
        """-2 (log p(x) - its maximum): 0 at mu or inside the box."""
        return -2.0 * (self(x) - self.top)


def prior_terms(stack, params):
    """[Prior or None] aligned with params (Stack.free_parameters())."""
    out = []
    for i, attr, *_ in params:
        spec = stack.owner(i).fit.get(attr, {}).get('prior')
        out.append(None if spec is None else Prior(spec))
    return out


def tension(spec, x):
    """Text if x pulls against the prior (see TENSION), else None."""
    spec = complete(spec)
    if spec['kind'] == 'gauss':
        k = abs(x - spec['mu']) / spec['tau']
        if k > TENSION:
            return '%.3g τ from the expected %.6g' % (k, spec['mu'])
        return None
    dist = spec['a'] - x if x < spec['a'] else x - spec['b'] \
        if x > spec['b'] else 0.0
    k = dist / spec['s']
    if k > TENSION:
        return '%.3g s outside [%.6g, %.6g]' % (k, spec['a'], spec['b'])
    return None
