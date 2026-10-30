"""
Scenario studies with the amortised (round-1) network of an NPE run
(plans/npe_plan_v1.md A1): can these data tell the scenarios apart?

  python scripts/npe_scenarios.py network.npe scenarios.json [--dream 2]

network.npe: "Save network…" of an NPE tab (it holds its fit problem).
scenarios.json: {"name": {"parameter": value, ...}, ...}, values in the
fit's own units (A, A^-2, turns) by the fit's parameter names
(fit.problem.parameter_names, e.g. "Fe.MSLD_rho"); a parameter a scenario
leaves out keeps its value in the fit's stack.  E.g. a moment at an
interface against one spread through the layer.

For each scenario: synthetic data from the fit problem's model with the
likelihood's noise (fit.npe.simulator), the round-1 network's posterior at
those data (--samples), and the posterior mass of each scenario's region:
the share of the samples nearer to that scenario than to any other in u
(the normalised parameters).  A row whose diagonal holds nearly all the
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

from fit.npe.simulator import Simulator                       # noqa: E402
from fit.npe.train import load_npe                            # noqa: E402
from fit.saved import problem_from_state                      # noqa: E402


def scenario_x(problem, values):
    """Physical parameter vector of a scenario: the fit's current values,
    with the scenario's."""
    x = problem.to_x(problem.x0())
    unknown = set(values) - set(problem.names)
    if unknown:
        raise ValueError('unknown parameters %s (the fit has %s)'
                         % (sorted(unknown), problem.names))
    for n, v in values.items():
        x[problem.names.index(n)] = float(v)
    return x


def masses(u, centres):
    """Share of the rows of u nearest each centre (u space)."""
    d = ((u[:, None, :] - centres[None]) ** 2).sum(-1)
    k = d.argmin(axis=1)
    return np.bincount(k, minlength=len(centres)) / len(u)


def with_data(problem, R):
    """A copy of problem whose channels hold the flat data R."""
    p = copy.deepcopy(problem)
    ends = np.cumsum([len(ch[2]) for ch in p.channels])[:-1]
    p.channels = [(n, idx, r, s) for (n, idx, _, s), r
                  in zip(p.channels, np.split(R, ends))]
    return p


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
    problem = problem_from_state(res.meta['problem'])
    res.check(problem)
    with open(o.scenarios) as fh:
        scen = json.load(fh)
    names = list(scen)
    X = np.array([scenario_x(problem, scen[n]) for n in names])
    C = problem.to_u(X)
    sim = Simulator(problem, seed=o.seed)
    R, ok = sim.simulate(C)
    if not ok.all():
        sys.exit('the model fails at %s'
                 % ', '.join(n for n, g in zip(names, ok) if not g))
    R = sim.add_noise(R)
    w = max(len(n) for n in names)
    print('posterior mass of each scenario\'s region (rows: the data of '
          'that scenario; round-1 network, %d samples)' % o.samples)
    print('%-*s  %s  separated' % (w, '', '  '.join('%8s' % n[:8]
                                                    for n in names)))
    M = []
    for k, n in enumerate(names):
        u = res.sample(o.samples, round=0, x=sim.features(R[k]),
                       seed=o.seed + k)
        m = masses(u, C)
        M.append(m)
        print('%-*s  %s  %s' % (w, n, '  '.join('%8.3f' % v for v in m),
                                'yes' if m[k] >= o.separated else 'NO'))
    if o.dream:
        from fit.dream.dream_fit import PosteriorProblem, sample
        print('\nDREAM check (%d generations):' % o.n_gen)
        for k, n in enumerate(names[:o.dream]):
            pp = PosteriorProblem(with_data(problem, R[k]))
            u0 = res.sample(8, round=0, x=sim.features(R[k]),
                            seed=o.seed + 100 + k)
            r = sample(pp, n_gen=o.n_gen, x0=pp.to_x(u0), seed=o.seed + k,
                       n_workers=o.workers, stop_when_converged=False)
            m = masses(problem.to_u(r.samples(0.5)), C)
            print('%-*s  %s  (network: %s)'
                  % (w, n, '  '.join('%8.3f' % v for v in m),
                     ' '.join('%.3f' % v for v in M[k])), flush=True)


if __name__ == '__main__':
    main()
