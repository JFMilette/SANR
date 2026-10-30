"""
Speed of the NPE training simulator on this machine (plans/npe_plan_v1.md
B0; not part of pytest): problem.model(x) over random prior draws
(fit.npe.simulator, build_sublayers included), with the compiled transfer
product on and off (model.stack USE_COMPILED), the resolution on and off,
and every interface cut into each of --sublayers slabs.  Prints the time
per draw, the draws per second on one core, the slab-count range, and how
long --sims simulations take on --workers cores.

The session is a Licorne session folder (as benchmarks/bench_fit.py, with
--vary / --untie), a DREAM run (.npz of "Save chains…") or a network (.npe
of "Save network…"); the latter two hold their fit problem.

  python benchmarks/bench_npe_sim.py tests/data/private/session_A --untie
"""
import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model.stack as S                                      # noqa: E402
from fit.joint import parts                                  # noqa: E402
from fit.npe.simulator import Simulator, _stacks             # noqa: E402
from fit.problem import FitProblem                           # noqa: E402


def load_problem(path, vary=None, untie=False):
    """The fit problem of a Licorne folder, a DREAM .npz or an NPE .npe."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if path.endswith('.npz'):
            from fit.dream.dream_fit import load_run
            return load_run(path)[1].fit
        if path.endswith('.npe'):
            import json
            import torch
            from fit.saved import problem_from_state
            d = torch.load(path, weights_only=False)
            return problem_from_state(json.loads(d['meta'])['problem'])
        from bench_fit import load
        return load(path, vary, untie)[0]


def variant(problem, n_sub, resolution):
    """problem rebuilt with n_sub slabs per interface and the resolution on
    or off (each dataset's FitProblem built again from its inputs)."""
    out = []
    for p in parts(problem):
        st = copy.deepcopy(p.stack)
        for s in _stacks_of(st):
            for l in s.layers:
                l.roughness_sublayer = n_sub
            s.resolution['enabled'] = resolution and \
                bool(p.stack.resolution.get('enabled'))
        i = p.inputs
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out.append(FitProblem(st, i['data'], 'chi2', i['qmin'], i['qmax'],
                                  i['grid'], i['error_floor']))
    if len(out) == 1:
        return out[0]
    from fit.joint import JointProblem
    return JointProblem(list(zip(problem.part_names, out)), problem.modes,
                        problem.fixed, problem.sources)


def _stacks_of(st):
    yield st
    yield from getattr(st, '_mix_stacks', {}).values()


def timing(problem, u):
    """(ms per draw, slab counts) over the rows u."""
    slabs = []
    problem.model(problem.to_x(u[0]))
    t = time.perf_counter()
    for row in u:
        problem.model(problem.to_x(row))
        slabs.append(sum(len(st.sublayers) for st in _stacks(problem)))
    return 1e3 * (time.perf_counter() - t) / len(u), slabs


def main():
    a = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('session')
    a.add_argument('--vary', default=None)
    a.add_argument('--untie', action='store_true')
    a.add_argument('--draws', type=int, default=200)
    a.add_argument('--sublayers', type=int, nargs='+', default=[3, 5, 8, 16])
    a.add_argument('--sims', type=int, default=100_000)
    a.add_argument('--workers', type=int, default=os.cpu_count())
    a.add_argument('--seed', type=int, default=0)
    o = a.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    p = load_problem(o.session, o.vary, o.untie)
    res_on = bool(p.stack.resolution.get('enabled'))
    u = Simulator(p).sample_prior(o.draws, np.random.default_rng(o.seed))
    print('%s: %d free parameters, %d points, resolution %s (%s)'
          % (o.session, len(p.params), p.npoints,
             'on' if res_on else 'off',
             p.stack.resolution.get('scheme', 'quadrature')))
    print('%9s %9s %11s %9s %10s %12s %14s'
          % ('sublayers', 'compiled', 'resolution', 'slabs', 'ms/draw',
             'draws/s/core', '%d sims, %d cores' % (o.sims, o.workers)))
    compiled = S.USE_COMPILED
    try:
        for n_sub in o.sublayers:
            for res in ([True, False] if res_on else [False]):
                q = variant(p, n_sub, res)
                for comp in (True, False):
                    S.USE_COMPILED = comp and compiled
                    ms, slabs = timing(q, u)
                    print('%9d %9s %11s %9s %10.3g %12.0f %14s'
                          % (n_sub, 'yes' if S.USE_COMPILED else 'no',
                             'on' if res else 'off',
                             '%d-%d' % (min(slabs), max(slabs)), ms,
                             1e3 / ms, '%.1f min' % (o.sims * ms / 6e4 /
                                                     o.workers)), flush=True)
    finally:
        S.USE_COMPILED = compiled


if __name__ == '__main__':
    main()
