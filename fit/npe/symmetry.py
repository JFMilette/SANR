"""
Symmetry probe (plans/npe_plan_v1.md Y1): which transformations of the free
parameters leave a fit problem's model unchanged on its own channels, found
numerically, so that mirror modes are known before NPE or DREAM meets them
and bounds can be chosen to keep one of them (or both, on purpose).

CANDIDATES -- for each layer whose in-plane angle MSLD_theta is free (a
tie counts as one layer):
  theta -> -theta            mirror about the sample x axis
  theta -> theta + 1/2       M reversed (turns)
  rho -> -rho, theta + 1/2   the same vector (only when MSLD_rho is free:
                             two parameter values for one model)
and the same for every such layer at once when there are several; for each
layer whose elevation MSLD_phi is free, phi -> -phi (only rho cos(phi)
reaches the neutrons, HALPERIN), and for all of them at once.  In a joint
fit a layer's own copies in each dataset move together.  The transformed
values may leave the bounds: the model is evaluated there anyway.

VERDICT -- over n_draws prior draws x, with m = problem.model:
  'exact'   max |m(T x) - m(x)| / max |m(x)| <= EXACT (1e-12)
  'approx'  max |m(T x) - m(x)| / sigma < APPROX (0.1) at every point
  'broken'  otherwise
Nothing is hard-coded: a magnetic fronting or an analyser breaks some of
them, and the probe says which.
"""

import re

import numpy as np

from fit.npe.simulator import Simulator

EXACT = 1e-12
APPROX = 0.1


def _layer(name, attr):
    """The layer label of a parameter name (dataset suffix and attr
    removed)."""
    name = re.sub(r' \[[^\]]*\]$', '', name)
    suffix = '.' + attr
    return name[:-len(suffix)] if name.endswith(suffix) else name


def candidates(problem):
    """[(label, function x -> transformed x)] (see CANDIDATES)."""
    p = Simulator(problem).problem
    by = {}                       # layer -> attr -> [indices]
    for j, (prm, name) in enumerate(zip(p.params, p.names)):
        if prm[1] in ('MSLD_theta', 'MSLD_rho', 'MSLD_phi'):
            by.setdefault(_layer(name, prm[1]), {}).setdefault(
                prm[1], []).append(j)

    def move(changes):
        def T(x):
            x = np.array(x, dtype=float, copy=True)
            for idx, f in changes:
                x[idx] = f(x[idx])
            return x
        return T

    def flip(v):
        return -v

    def half(v):
        return v + 0.5

    out = []
    theta = [(k, v['MSLD_theta']) for k, v in by.items() if 'MSLD_theta' in v]
    for k, idx in theta:
        out.append(('%s: θ → −θ' % k, move([(idx, flip)])))
        out.append(('%s: θ → θ + ½' % k, move([(idx, half)])))
        if 'MSLD_rho' in by[k]:
            out.append(('%s: ρ → −ρ, θ → θ + ½' % k,
                        move([(idx, half), (by[k]['MSLD_rho'], flip)])))
    if len(theta) > 1:
        every = sum((i for _, i in theta), [])
        out.append(('all layers: θ → −θ', move([(every, flip)])))
        out.append(('all layers: θ → θ + ½', move([(every, half)])))
    phi = [(k, v['MSLD_phi']) for k, v in by.items() if 'MSLD_phi' in v]
    for k, idx in phi:
        out.append(('%s: φ → −φ' % k, move([(idx, flip)])))
    if len(phi) > 1:
        out.append(('all layers: φ → −φ',
                    move([(sum((i for _, i in phi), []), flip)])))
    return out


def probe(problem, n_draws=20, rng=None):
    """[{'name', 'verdict', 'rel', 'gap'}] of every candidate (see
    VERDICT): rel the largest relative change, gap the largest change in
    sigma, over the draws."""
    sim = Simulator(problem)
    p = sim.problem
    rng = np.random.default_rng(rng)
    X = p.to_x(sim.sample_prior(n_draws, rng))
    base = []
    for x in X:
        with np.errstate(all='ignore'):
            base.append(np.concatenate(p.model(x)))
    out = []
    for name, T in candidates(p):
        rel = gap = 0.0
        for x, m in zip(X, base):
            with np.errstate(all='ignore'):
                mt = np.concatenate(p.model(T(x)))
            d = np.abs(mt - m)
            if not np.all(np.isfinite(d)):
                rel = gap = np.inf
                break
            rel = max(rel, float(d.max() / max(np.abs(m).max(), 1e-300)))
            gap = max(gap, float((d / sim.sigma).max()))
        verdict = 'exact' if rel <= EXACT else \
            'approx' if gap < APPROX else 'broken'
        out.append(dict(name=name, verdict=verdict, rel=rel, gap=gap))
    return out


def format_probe(rows):
    if not rows:
        return 'No magnetic angle is fitted: nothing to probe.'
    w = max(len(r['name']) for r in rows)
    return '\n'.join('%-*s  %-7s  rel %.2g, %.3g σ'
                     % (w, r['name'], r['verdict'], r['rel'], r['gap'])
                     for r in rows)
