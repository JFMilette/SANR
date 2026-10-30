"""NPE-seeded DREAM on synthetic data (plans/npe_plan_v1.md V1): known
parameters recovered, a known mirror ambiguity found (both modes seeded and
sampled with their weight), a joint fit of three datasets.  `slow`;
skipped without torch, sbi and scikit-learn.

Run from the repo root: python -m pytest -m slow tests/test_npe_recovery.py
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fit.npe import available                               # noqa: E402

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not available(),
                                 reason='needs torch, sbi, scikit-learn')]

import model.stack as S                                     # noqa: E402
from fit.dream.dream_fit import (PosteriorProblem, sample,  # noqa: E402
                                 summary_table)
from fit.joint import JointProblem                          # noqa: E402
from fit.problem import FitProblem                          # noqa: E402
from model.polarisation import channel_pair, in_plane       # noqa: E402
from test_dream_fit import within3sd                         # noqa: E402

Q = np.linspace(0.008, 0.15, 120)
FULL = ('++', '--', '+-', '-+')
HALF = ('+', '-')
TRUTH = {'Fe.MSLD_theta': 0.1, 'Co.MSLD_theta': -0.15,
         'Fe.MSLD_rho': 5e-6, 'Cu.thickness': 30.0}
WORKERS = 4


def multilayer():
    """vacuum / Fe 60 / Cu 30 / Co 50 / Si, noncollinear in-plane M."""
    return S.Stack([S.Layer('front', 0, 0, 0, 0, 0),
                    S.Layer('Fe', 60, 8e-6, 0, 5e-6, 0.1, 4.0),
                    S.Layer('Cu', 30, 6.5e-6, 0, 0, 0, 3.0),
                    S.Layer('Co', 50, 2.3e-6, 0, 4e-6, -0.15, 4.0),
                    S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])


def measured(st, channels, seed, err=0.03):
    st.build_sublayers()
    pairs = [channel_pair(c, in_plane(0.0)) for c in channels]
    R = st.reflectivities(Q, pairs, warn=False)
    rng = np.random.default_rng(seed)
    out = []
    for j, ((P0, P), c) in enumerate(zip(pairs, channels)):
        dR = err * R[:, j] + 2e-7
        out.append(dict(Q=Q, R=R[:, j] + rng.normal(size=Q.size) * dR,
                        dR=dR, P0=P0, P=P, name='R' + c))
    return out


def free(st, fe_theta=(0.0, 0.5)):
    """The four TRUTH parameters free, starting elsewhere."""
    fe, cu, co = st.layers[1:4]
    fe.MSLD_theta, co.MSLD_theta, fe.MSLD_rho, cu.thickness = \
        0.3, 0.2, 3e-6, 40.0
    fe.fit['MSLD_theta'] = {'vary': True, 'min': fe_theta[0],
                            'max': fe_theta[1]}
    co.fit['MSLD_theta'] = {'vary': True, 'min': -0.5, 'max': 0.5}
    fe.fit['MSLD_rho'] = {'vary': True, 'min': 1e-6, 'max': 8e-6}
    cu.fit['thickness'] = {'vary': True, 'min': 15.0, 'max': 50.0}
    return st


def seeded_dream(fit, seed, n_gen=3000, sims=4000):
    from fit.npe.seed import seed_chains
    from fit.npe.train import run_npe
    pp = PosteriorProblem(fit)
    res = run_npe(fit, coarse=False, rounds=3, sims_per_round=sims,
                  seed=seed, n_workers=WORKERS)
    seeds = seed_chains(res, pp, 8, rng=seed, n_samples=3000,
                        n_workers=WORKERS)
    run = sample(pp, n_gen=n_gen, x0=seeds.x0, seed=seed,
                 n_workers=WORKERS, stop_when_converged=False)
    return pp, seeds, run


@pytest.mark.parametrize('channels', [HALF, FULL], ids=['half', 'full'])
def test_recovery(channels):
    fit = FitProblem(free(multilayer()), measured(multilayer(), channels, 1),
                     'chi2')
    pp, seeds, run = seeded_dream(fit, seed=2)
    rows, best = summary_table(run, pp)
    for name, v in TRUTH.items():
        assert within3sd(rows, name, v), (name, v, rows)
    assert best['chi2'] < 1.5


def test_mirror_modes_are_found_and_weighed():
    """Without an analyser, theta -> -theta of every layer at once is an
    exact symmetry (fit.npe.symmetry): with bounds that allow both, the
    seeds cover both modes and DREAM samples each with half the weight
    (equal by the symmetry, so no long reference run is needed)."""
    from fit.npe.symmetry import probe
    fit = FitProblem(free(multilayer(), fe_theta=(-0.5, 0.5)),
                     measured(multilayer(), HALF, 3), 'chi2')
    sym = {r['name']: r['verdict'] for r in probe(fit, n_draws=4, rng=0)}
    assert sym['all layers: θ → −θ'] == 'exact'
    pp, seeds, run = seeded_dream(fit, seed=5)
    j = fit.names.index('Fe.MSLD_theta')
    signs = {np.sign(c['best_x'][j]) for c in seeds.clusters
             if c['chains'] and abs(abs(c['best_x'][j]) - 0.1) < 0.03}
    assert signs == {-1.0, 1.0}, [c['best_x'] for c in seeds.clusters]
    s = run.samples(0.5)[:, j]
    plus = np.mean(s > 0)
    assert 0.25 < plus < 0.75, plus


def test_joint_shared_and_own():
    """Three datasets of one film at three fields: thickness and
    roughness shared, the magnetisation of each its own."""
    rho = {'1T': 5e-6, '0.5T': 3.5e-6, '0.1T': 2e-6}

    def film(r=5e-6):
        return S.Stack([S.Layer('front', 0, 0, 0, 0, 0),
                        S.Layer('Fe', 80, 8e-6, 0, r, 0.0, 4.0),
                        S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])

    st = film()
    fe = st.layers[1]
    fe.thickness, fe.MSLD_rho, fe.roughness_sigma = 70.0, 4e-6, 2.0
    fe.fit['thickness'] = {'vary': True, 'min': 60, 'max': 100}
    fe.fit['MSLD_rho'] = {'vary': True, 'min': 0.5e-6, 'max': 8e-6}
    fe.fit['roughness_sigma'] = {'vary': True, 'min': 0, 'max': 10}
    parts = [(n, FitProblem(st, measured(film(r), HALF, k), 'chi2'))
             for k, (n, r) in enumerate(rho.items())]
    jp = JointProblem(parts, {n: {'Fe.MSLD_rho': 'own'} for n in rho})
    pp, seeds, run = seeded_dream(jp, seed=7)
    rows, _ = summary_table(run, pp)
    assert within3sd(rows, 'Fe.thickness', 80.0)
    assert within3sd(rows, 'Fe.roughness_sigma', 4.0)
    for n, r in rho.items():
        assert within3sd(rows, 'Fe.MSLD_rho [%s]' % n, r)
