"""
Scenario studies with the amortised (round-1) network of an NPE run
(plans/npe_plan_v1.md A1): can these data tell the scenarios apart?

  python scripts/npe_scenarios.py network.npe scenarios.json [--dream 2]

network.npe: "Save network…" of an NPE tab (it holds its fit problem).
scenarios.json: {"name": {"parameter": value, ...}, ...}, values in the
fit's own units (A, A^-2, turns) by the fit's parameter names
(fit.problem.parameter_names, e.g. "Fe.MSLD_rho"); a parameter a scenario
leaves out keeps its value in the fit's stack (a fitted error scale or
floor: log10_sigma_scale 0, log10_error_floor -2, unless given).  E.g. a moment at an
interface against one spread through the layer.

For each scenario: synthetic data from the fit problem's model with the
likelihood's noise (fit.npe.simulator), the round-1 network's posterior at
those data (--samples), and the posterior mass of each scenario's region:
the share of the samples nearer to that scenario than to any other in u
(the normalised fit parameters, the noise ones left out).  A row whose diagonal holds nearly all the
mass (>= --separated) is told apart from the others.  --dream k checks the
first k scenarios with DREAM (fit.dream.dream_fit.sample, --n-gen, chains
started at round-1 samples): the same masses from its posterior.

Refused unless the round-1 network passed its calibration (the NPE tab's
"calibrate round 1"; fit.npe.train calibrate), as the plan requires;
--force runs it anyway, with a warning.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fit.dream.dream_fit import EXTRA_START                  # noqa: E402
from fit.npe.simulator import Simulator                       # noqa: E402
from fit.npe.train import load_npe                            # noqa: E402


def scenario_x(pp, values):
    """Physical parameter vector of a scenario of the PosteriorProblem
    pp: the fit's current values (EXTRA_START for the noise), with the
    scenario's."""
    fit = pp.fit
    x = np.r_[fit.to_x(fit.x0()),
              [EXTRA_START[n] for n in pp.names[len(fit.params):]]]
    unknown = set(values) - set(pp.names)
    if unknown:
        raise ValueError('unknown parameters %s (the fit has %s)'
                         % (sorted(unknown), pp.names))
    for n, v in values.items():
        x[pp.names.index(n)] = float(v)
    return x


def masses(u, centres):
    """Share of the rows of u nearest each centre (u space)."""
    d = ((u[:, None, :] - centres[None]) ** 2).sum(-1)
    k = d.argmin(axis=1)
    return np.bincount(k, minlength=len(centres)) / len(u)


def with_data(pp, R):
    """A copy of the PosteriorProblem pp whose channels hold the flat
    data R."""
    from fit.dream.dream_fit import PosteriorProblem
    p = copy.deepcopy(pp.fit)
    ends = np.cumsum([len(ch[2]) for ch in p.channels])[:-1]
    p.channels = [(n, idx, r, s) for (n, idx, _, s), r
                  in zip(p.channels, np.split(R, ends))]
    return PosteriorProblem(p, sigma_free=pp.sigma_free,
                            floor_free=pp.floor_free)


def main():
    a = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('network')
    a.add_argument('scenarios')
    a.add_argument('--samples', type=int, default=5000)
    a.add_argument('--separated', type=float, default=0.95)
    a.add_argument('--dream', type=int, default=0)
    a.add_argument('--n-gen', type=int, default=5000)
    a.add_argument('--workers', type=int, default=1)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--force', action='store_true')
    o = a.parse_args()
    res = load_npe(o.network)
    cal = res.meta.get('calibration')
    if cal is None or not cal['passed']:
        msg = ('the round-1 network was %s: its posteriors are not '
               'trustworthy for scenario studies'
               % ('never calibrated' if cal is None else 'not calibrated'))
        if not o.force:
            sys.exit(msg + ' (--force to run anyway)')
        print('WARNING: ' + msg)
    pp = res.posterior_problem()
    res.check(pp)
    with open(o.scenarios) as fh:
        scen = json.load(fh)
    names = list(scen)
    X = np.array([scenario_x(pp, scen[n]) for n in names])
    U = pp.to_u(X)
    k = len(pp.fit.params)
    C = U[:, :k]                    # the regions: in the fit parameters
    sim = Simulator(pp, seed=o.seed)
    R, ok = sim.simulate(U)
    if not ok.all():
        sys.exit('the model fails at %s'
                 % ', '.join(n for n, g in zip(names, ok) if not g))
    R = sim.add_noise(R, u=U)
    w = max(len(n) for n in names)
    print('posterior mass of each scenario\'s region (rows: the data of '
          'that scenario; round-1 network, %d samples)' % o.samples)
    print('%-*s  %s  separated' % (w, '', '  '.join('%8s' % n[:8]
                                                    for n in names)))
    M = []
    for i, n in enumerate(names):
        u = res.sample(o.samples, round=0, x=sim.features(R[i]),
                       seed=o.seed + i)
        m = masses(u[:, :k], C)
        M.append(m)
        print('%-*s  %s  %s' % (w, n, '  '.join('%8.3f' % v for v in m),
                                'yes' if m[i] >= o.separated else 'NO'))
    if o.dream:
        from fit.dream.dream_fit import sample
        print('\nDREAM check (%d generations):' % o.n_gen)
        for i, n in enumerate(names[:o.dream]):
            q = with_data(pp, R[i])
            u0 = res.sample(8, round=0, x=sim.features(R[i]),
                            seed=o.seed + 100 + i)
            r = sample(q, n_gen=o.n_gen, x0=q.to_x(u0), seed=o.seed + i,
                       n_workers=o.workers, stop_when_converged=False)
            m = masses(q.to_u(r.samples(0.5))[:, :k], C)
            print('%-*s  %s  (network: %s)'
                  % (w, n, '  '.join('%8.3f' % v for v in m),
                     ' '.join('%.3f' % v for v in M[i])), flush=True)


if __name__ == '__main__':
    main()
