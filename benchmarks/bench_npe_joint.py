"""
NPE-seeded DREAM on a joint fit of a saved SANR session (plans/npe_plan_v1.md
V3.3; not part of pytest).  The session is opened as the GUI opens it (its
models, datasets and resolution), the datasets named are fitted together as
the Fit tab would (fit.panel, the NPE method's problem), each parameter in
the mode the session's joint fit gives it, a dataset that joint fit lacks
taking the modes of --like (default: the first dataset it has).  With the
Licorne resolution, the Q grid of a dataset whose channels are not on the
same points is the one of channel --grid (default: the first channel, as
pressing OK in the Fit tab's dialog), said for each dataset.  Printed:
chi^2/N at the session's values (the current fit), the time per model
evaluation, the training rounds, the clusters fit.npe.seed finds, and the
DREAM result from the seeds; appended as a JSON line to --out.

  python benchmarks/bench_npe_joint.py sessions/39100/oct29/magnetic_fit_1.53.json \\
      --sets "S1 4K 0.1T" "S1 4K 2T" "S1 4K 4.8T" "S1 10K 4.8T" \\
             "S1 20K 4.8T" "S1 250K 4.8T" --workers 8
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def joint_problem(session, sets, like=None, grid=None):
    """(PosteriorProblem, warnings) of the joint fit of `sets` in session."""
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def get_item(parent, title, label, items, current, editable):
        # the Licorne Q grid dialog of fit.panel, answered without a window
        k = next((i for i, t in enumerate(items)
                  if grid and t.startswith('R%s ' % grid)), current)
        print('%s: Q grid of %s' % (label.split('\n')[0], items[k]))
        return items[k], True

    QtWidgets.QInputDialog.getItem = staticmethod(get_item)
    from tabs import ExperimentalTab, SimulationTab
    from fit.panel import JOINT
    with open(session) as fh:
        d = json.load(fh)
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim.restore_state(d)
        missing = exp.restore_state(d['experimental'],
                                    os.path.dirname(os.path.abspath(session)))
    if missing:
        raise SystemExit('channels not found: %s' % missing)
    names = [s['name'] for s in exp.sets]
    bad = [n for n in sets if n not in names]
    if bad:
        raise SystemExit('no dataset %s in the session (it has %s)'
                         % (bad, names))
    old = d['experimental'].get('fit', {}).get('joint') or {}
    modes = dict(old.get('modes') or {})
    like = like or next((n for n in old.get('sets', []) if n in modes), None)
    for n in sets:
        if n not in modes and like is not None:
            modes[n] = dict(modes[like])
    exp._joint = dict(sets=list(sets), modes=modes,
                      sources=dict(old.get('sources') or {}))
    exp._refresh_fit_sets()
    exp.fit_set.setCurrentIndex(exp.fit_set.findData(JOINT))
    built = exp._npe_problem()
    if built is None:
        raise SystemExit('no fit problem: %s' % exp.npe_report.text())
    pp, inputs, qmin, qmax, note = built
    app.processEvents()
    return pp, note


def main():
    a = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('session')
    a.add_argument('--sets', nargs='+', required=True)
    a.add_argument('--like', default=None)
    a.add_argument('--grid', default=None,
                   help="channel ('+', '-', ...) whose Q the Licorne "
                        "resolution runs on")
    a.add_argument('--seed', type=int, default=1)
    a.add_argument('--rounds', type=int, default=5)
    a.add_argument('--sims', type=int, default=20000)
    a.add_argument('--samples', type=int, default=10000)
    a.add_argument('--chains', type=int, default=8)
    a.add_argument('--n-gen', type=int, default=5000)
    a.add_argument('--workers', type=int, default=1)
    a.add_argument('--full', action='store_true',
                   help='train on the full-fidelity model')
    a.add_argument('--only-problem', action='store_true',
                   help='stop after the chi^2 at the session values')
    a.add_argument('--out', default=os.path.join(ROOT, 'benchmarks',
                                                 'results.jsonl'))
    o = a.parse_args()
    pp, note = joint_problem(o.session, o.sets, o.like, o.grid)
    fit = pp.fit
    x_now = fit.to_x(fit.x0())
    chi_now = pp.chi2_per_point(x_now)
    t = time.perf_counter()
    for _ in range(3):
        fit.model(x_now)
    per = (time.perf_counter() - t) / 3
    print('%d datasets, %d parameters, %d points; chi2/N at the session '
          'values %.5g; %.3g s per model evaluation'
          % (len(o.sets), pp.ndim, fit.npoints, chi_now, per), flush=True)
    if note:
        print('warnings:', note.replace('<br>', '; '))
    if o.only_problem:
        return
    from fit.dream.dream_fit import sample, summary_table
    from fit.npe.seed import format_clusters, seed_chains
    from fit.npe.simulator import describe_coarse
    from fit.npe.train import run_npe
    t0 = time.time()

    def say(event, info):
        if event == 'round':
            print('round %d: %d sims (%d failed), %s %.3g, %d epochs, '
                  'loss %.5g, %.0f s' % (
                      info['round'], info['n_sims'], info['n_failed'],
                      info['proposal'], info['acceptance'], info['epochs'],
                      info['best_validation_loss'], info['seconds']),
                  flush=True)
        elif event == 'coarse':
            print('training simulator:', describe_coarse(info), flush=True)

    res = run_npe(fit, coarse=not o.full, rounds=o.rounds,
                  sims_per_round=o.sims, seed=o.seed, n_workers=o.workers,
                  callback=say)
    t_train = time.time() - t0
    seeds = seed_chains(res, pp, o.chains, rng=o.seed, n_samples=o.samples,
                        n_workers=o.workers)
    print(format_clusters(seeds), flush=True)
    t_seed = time.time() - t0 - t_train
    r = sample(pp, n_gen=o.n_gen, x0=seeds.x0, seed=o.seed,
               n_workers=o.workers, stop_when_converged=False)
    rows, best = summary_table(r, pp)
    rhat = max(row['rhat'] for row in rows)
    print('DREAM from the seeds: MAP chi2/N %.5g (session values %.5g), '
          'max R-hat %.3g; %.0f s training, %.0f s seeding, %.0f s in all'
          % (best['chi2'], chi_now, rhat, t_train, t_seed, time.time() - t0),
          flush=True)
    rec = dict(kind='npe_joint', session=o.session, sets=o.sets,
               seed=o.seed, names=pp.names, chi2_session=chi_now,
               chi2_map=best['chi2'], rhat=rhat,
               rounds=[{k: v for k, v in s.items() if 'loss' not in k}
                       for s in res.rounds], coarse=res.meta['coarse'],
               clusters=[dict(weight=c['weight'], chains=c['chains'],
                              chi2=c['chi2'], at_bound=c['at_bound'])
                         for c in seeds.clusters],
               seconds=dict(train=t_train, seed=t_seed,
                            total=time.time() - t0),
               median=[row['median'] for row in rows],
               x_session=list(map(float, x_now)))
    with open(o.out, 'a') as fh:
        fh.write(json.dumps(rec, default=float) + '\n')


if __name__ == '__main__':
    main()
