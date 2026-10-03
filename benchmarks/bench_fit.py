"""
Fitting benchmark on a saved Licorne session (not part of pytest).

  landscape  L-BFGS-B from the current values and from n random points in
             the bounds: the distribution of local minima, how often each is
             reached, and which parameters end on a bound.
  dream      DREAM runs (fit.dream.dream_fit.sample) with the given settings
             and seeds: MAP chi^2/N, each chain's chi^2/N at the end, R-hat,
             acceptance, parameters with > 5 % of their samples at a bound.

The free parameters are the session's own fit flags (default), or --vary
"i:attr,i:attr,..." with i the layer index in the SANR stack (0 = fronting).
Results are appended as JSON lines to --out and printed as a table.

Examples (run from the repo root):
  python benchmarks/bench_fit.py landscape tests/data/private/session_A --starts 24
  python benchmarks/bench_fit.py landscape tests/data/private/session_B \
      --vary 1:thickness,2:thickness,3:thickness,1:roughness_sigma,2:roughness_sigma,3:roughness_sigma,4:roughness_sigma
  python benchmarks/bench_fit.py dream tests/data/private/session_A --seeds 1 2 3 \
      --n-gen 5000 --chains 3
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fit.problem import FitProblem                      # noqa: E402
from model.licorne_io import load_licorne_session        # noqa: E402


def load(folder, vary=None):
    """(FitProblem, names) of a Licorne session, cost chi2, its own data."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lic = load_licorne_session(folder)
    st = lic['stack']
    if vary:
        for l in st.layers:
            for e in l.fit.values():
                e['vary'] = False
        for item in vary.split(','):
            i, attr = item.split(':')
            st.layers[int(i)].fit_entry(attr)['vary'] = True
    Q = np.loadtxt(os.path.join(folder, 'q.dat'), comments='#')
    data = []
    for c in lic['channels']:
        E = np.loadtxt(c['path'], comments='#')
        P = c['pol'][1]
        data.append(dict(Q=Q, R=E[:, 0], dR=E[:, 1], P0=np.array(c['pol'][0]),
                         P=None if not np.any(P) else np.array(P),
                         name='R' + c['channel'], norm=c['norm']))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = FitProblem(st, data, cost='chi2')
    names = ['%s.%s' % (st.layers[i].name, a) for i, a, *_ in p.params]
    return p, names


def at_bound(u, edge):
    return (u < edge) | (u > 1 - edge)


def landscape(p, names, starts, seed):
    rng = np.random.default_rng(seed)
    U0 = [p.x0()] + [rng.uniform(0, 1, len(p.params)) for _ in range(starts)]
    runs = []
    for k, u0 in enumerate(U0):
        r = minimize(p, u0, method='L-BFGS-B', bounds=[(0, 1)] * len(u0),
                     options=dict(maxiter=3000))
        runs.append(dict(start='current' if k == 0 else 'random',
                         chi2=float(r.fun), x=list(map(float, p.to_x(r.x))),
                         at_bound=at_bound(r.x, 0.005).tolist()))
    c = np.array([r['chi2'] for r in runs])
    best = runs[int(np.argmin(c))]
    print('%d local fits: best chi2/N %.4f (from %s start), median %.3f, worst %.3f'
          % (len(c), c.min(), best['start'], np.median(c), c.max()))
    rnd = c[1:]
    print('random starts within +0.1 / +1 of the best: %.2f / %.2f'
          % (np.mean(rnd < c.min() + 0.1), np.mean(rnd < c.min() + 1)))
    B = np.array([r['at_bound'] for r in runs])
    print('fits with >= 1 parameter on a bound: %.2f' % B.any(1).mean())
    for j, n in enumerate(names):
        print('  %-28s on a bound in %.2f of fits; best %.6g'
              % (n, B[:, j].mean(), best['x'][j]))
    return dict(kind='landscape', names=names, runs=runs)


def dream_runs(p, names, seeds, n_gen, chains, archive, sigma_free, extra):
    from fit.dream.dream_fit import PosteriorProblem, sample
    out = []
    print('%4s %10s %40s %8s %6s  %s' % ('seed', 'chi2 MAP', 'chi2/N of each chain at the end',
                                         'Rhat', 'acc', 'on a bound (>5 %)'))
    for seed in seeds:
        pp = PosteriorProblem(p, sigma_free=sigma_free)
        t = time.time()
        r = sample(pp, n_gen=n_gen, seed=seed, n_chains=chains, archive=archive,
                   stop_when_converged=False, **extra)
        nP = len(p.params)
        end = [pp.chi2_per_point(x[:nP]) for x in r.chains[-1]]
        s = r.samples(0.5)
        u = (s[:, :nP] - p.lo) / (p.hi - p.lo)
        frac = np.mean(at_bound(u, 0.01), axis=0)
        rec = dict(kind='dream', seed=seed, n_gen=n_gen, chains=chains,
                   archive=archive, sigma_free=sigma_free, extra=extra,
                   names=pp.names,
                   chi2_map=pp.chi2_per_point(r.best()[0][:nP]),
                   chi2_end=end, rhat=list(map(float, r.rhat_history[-1][1])),
                   acc=float(r.acceptance[r.adapt_until:].mean()),
                   median=list(map(float, np.median(s, 0))),
                   q025=list(map(float, np.percentile(s, 2.5, 0))),
                   q975=list(map(float, np.percentile(s, 97.5, 0))),
                   at_bound=frac.tolist(), seconds=time.time() - t)
        out.append(rec)
        print('%4d %10.4f %40s %8.2f %6.3f  %s'
              % (seed, rec['chi2_map'], np.round(end, 2).tolist()[:8],
                 max(rec['rhat']), rec['acc'],
                 [n for n, f in zip(names, frac) if f > 0.05]), flush=True)
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('mode', choices=['landscape', 'dream'])
    a.add_argument('session')
    a.add_argument('--vary', default=None)
    a.add_argument('--starts', type=int, default=24)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    a.add_argument('--n-gen', type=int, default=5000)
    a.add_argument('--chains', type=int, default=None)
    a.add_argument('--classic', action='store_true', help='no archive')
    a.add_argument('--sigma-free', action='store_true')
    a.add_argument('--p-snooker', type=float, default=None)
    a.add_argument('--temper', action='store_true',
                   help='tempered burn-in (beta_start 1e-3)')
    a.add_argument('--out', default='benchmarks/results.jsonl')
    o = a.parse_args()
    p, names = load(o.session, o.vary)
    print('%s: %d free parameters, %d points, chi2/N at the session values %.4f'
          % (o.session, len(names), p.npoints, p(p.x0())))
    if o.mode == 'landscape':
        recs = [landscape(p, names, o.starts, o.seed)]
    else:
        extra = {}
        if o.p_snooker is not None:
            extra['p_snooker'] = o.p_snooker
        if o.temper:
            extra['beta_start'] = 1e-3
        recs = dream_runs(p, names, o.seeds, o.n_gen, o.chains, not o.classic,
                          o.sigma_free, extra)
    os.makedirs(os.path.dirname(os.path.abspath(o.out)), exist_ok=True)
    with open(o.out, 'a') as fh:
        for r in recs:
            fh.write(json.dumps(dict(r, session=o.session)) + '\n')


if __name__ == '__main__':
    main()
