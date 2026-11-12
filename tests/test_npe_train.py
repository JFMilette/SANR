"""fit.npe.train: sequential NPE on a one-layer toy (plans/npe_plan_v1.md
T2): the network's posterior against a brute-force grid posterior of the
exact likelihood, the bounds, the truncated proposals, save / load, and
the calibration.  `slow`; skipped without torch, sbi and scikit-learn.

Run from the repo root: python -m pytest -m slow tests/test_npe_train.py
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fit.npe import available                               # noqa: E402

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not available(),
                                 reason='needs torch, sbi, scikit-learn')]

import model.stack as S                                     # noqa: E402
from fit.dream.dream_fit import PosteriorProblem            # noqa: E402
from fit.problem import FitProblem                          # noqa: E402
from model.polarisation import channel_pair, in_plane       # noqa: E402

Q = np.linspace(0.01, 0.15, 100)
TRUTH = (80.0, 8e-6)
BOUNDS = ((70.0, 90.0), (6e-6, 10e-6))


def toy(seed=0, shift=0.0):
    """vacuum / film / Si, R+ with 5 % errors; thickness and NSLD free;
    shift moves the data (another data set)."""
    st = S.Stack([S.Layer('front', 0, 0, 0, 0, 0),
                  S.Layer('F', TRUTH[0], TRUTH[1], 0, 0, 0, 0.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 0.0)])
    p = [channel_pair('+', in_plane(0.0))]
    R = st.reflectivities(Q, p)[:, 0]
    dR = 0.05 * R + 1e-8
    R = R + np.random.default_rng(seed).normal(size=Q.size) * dR + shift
    f = st.layers[1]
    f.fit['thickness'] = {'vary': True, 'min': BOUNDS[0][0],
                          'max': BOUNDS[0][1]}
    f.fit['NSLD_real'] = {'vary': True, 'min': BOUNDS[1][0],
                          'max': BOUNDS[1][1]}
    return FitProblem(st, [dict(Q=Q, R=R, dR=dR, P0=p[0][0], P=p[0][1],
                                name='R+')], 'chi2')


@pytest.fixture(scope='module')
def trained():
    from fit.npe.train import run_npe
    p = toy()
    res = run_npe(p, coarse=False, rounds=3, sims_per_round=3000, seed=4,
                  n_workers=4)
    return p, res


def grid_posterior(p, n=161):
    """Mean and sd of each parameter under the exact posterior on an
    n x n grid in u."""
    pp = PosteriorProblem(p)
    g = (np.arange(n) + 0.5) / n
    U = np.array(np.meshgrid(g, g, indexing='ij')).reshape(2, -1).T
    lp = np.array([pp(p.to_x(u)) for u in U])
    w = np.exp(lp - lp.max())
    w /= w.sum()
    X = p.to_x(U)
    mean = w @ X
    sd = np.sqrt(w @ (X - mean) ** 2)
    return mean, sd


def test_matches_the_grid_posterior(trained):
    """Stated tolerance, for both parameters: the median within 1.5
    posterior sd of the grid's mean, the spread (interquartile range /
    1.349) within a factor 2 of its sd, and at least 90 % of the samples
    within 5 sd.  Robust statistics: the flow keeps thin tails far out (the
    posterior sd is 0.3 % of the bounds), which a plain sd would weigh;
    DREAM, seeded from the core, never sees them."""
    p, res = trained
    mean, sd = grid_posterior(p)
    X = p.to_x(res.sample(20000, seed=1))
    med = np.median(X, axis=0)
    s = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
    assert np.all(np.abs(med - mean) < 1.5 * sd), (med, mean, sd)
    assert np.all((s / sd > 0.5) & (s / sd < 2)), (s, sd)
    core = np.all(np.abs(X - mean) < 5 * sd, axis=1).mean()
    assert core > 0.9, core


def test_samples_stay_in_the_bounds(trained):
    p, res = trained
    for r in range(len(res.networks)):
        u = res.sample(5000, round=r, seed=2)
        assert u.min() > 0 and u.max() < 1
    # far from the data too: the amortised network at other features
    sim = res.simulator
    x = sim.features(sim.observed * 3)
    u = res.sample(2000, round=0, x=x, seed=3)
    assert u.min() > 0 and u.max() < 1


def test_rounds_are_truncated(trained):
    p, res = trained
    r = res.rounds
    assert [s['proposal'] for s in r][0] == 'prior'
    assert all(s['acceptance'] < 1 for s in r[1:])
    assert all(s['n_sims'] + s['n_failed'] == 3000 for s in r)
    assert res.meta['names'] == p.names


def test_envelope_draws_the_truncated_prior(trained, monkeypatch):
    """The envelope path (sharp posteriors) gives nearly the distribution
    of plain rejection from the prior."""
    import torch
    from fit.npe import train as T
    p, res = trained
    post = res.posterior()
    inside = T.density_region(post, 1e-3)
    rng = np.random.default_rng(5)
    monkeypatch.setattr(T, 'PRIOR_ACCEPT', 0.0)       # exact, however slow
    a, ia = T.truncated_draws(res.prior, post, inside, 4000, rng)
    monkeypatch.setattr(T, 'PRIOR_ACCEPT', 1.1)
    b, ib = T.truncated_draws(res.prior, post, inside, 4000, rng)
    assert ia['proposal'] == 'prior' and ib['proposal'] == 'envelope'
    assert ib['acceptance'] > 0
    with torch.no_grad():
        assert inside(torch.as_tensor(T.to_theta(b),
                                      dtype=torch.float32)).all()
    assert len(np.unique(b, axis=0)) == len(b)        # no repeated draw
    # an approximation (fit.npe.train ROUNDS): here the region is mostly
    # the flow's thin tails, its hardest case
    sa, sb = a.std(axis=0), b.std(axis=0)
    assert np.all(np.abs(a.mean(axis=0) - b.mean(axis=0)) < 0.25 * sa)
    assert np.all(np.abs(sb / sa - 1) < 0.35)


def test_save_load(trained, tmp_path):
    from fit.npe.train import load_npe
    p, res = trained
    path = tmp_path / 'toy.npe'
    res.save(path)
    back = load_npe(path, p)
    assert np.array_equal(back.sample(500, seed=9), res.sample(500, seed=9))
    assert np.allclose(back.log_prob(res.sample(5, seed=1)),
                       res.log_prob(res.sample(5, seed=1)))
    with pytest.raises(ValueError, match='other data'):
        load_npe(path, toy(shift=1e-9))
    other = toy()
    other.hi = other.hi * 1.01
    with pytest.raises(ValueError, match='bounds'):
        back.check(other)
    # the same parameters and data, another model (the substrate's NSLD,
    # kept fixed): refused; another start of the fitted ones: taken
    from fit.saved import problem_from_state, problem_state
    d = problem_state(toy())
    d['stack']['layers'][2]['NSLD_real'] = 2.2e-6
    with pytest.raises(ValueError, match='another model'):
        back.check(problem_from_state(d))
    d = problem_state(toy())
    d['stack']['layers'][1]['thickness'] *= 1.05
    back.check(problem_from_state(d))


def test_calibration(trained):
    from fit.npe.train import calibrate
    p, res = trained
    cal = calibrate(res, res.simulator, n=200, n_posterior=300, seed=1)
    assert cal['round'] == 1 and len(cal['ks_pvals']) == 2
    assert res.meta['calibration'] is cal
    assert isinstance(cal['passed'], bool)
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 3
    with pytest.raises(RuntimeError, match='stopped'):
        calibrate(res, res.simulator, n=200, n_posterior=300, seed=1,
                  cancelled=stop)


def test_each_simulation_keeps_its_split():
    """fit.npe.train ROUNDS: a simulation in the training set stays
    there in the later rounds (sbi alone splits all of them again each
    round), the new ones are split in the fraction asked."""
    import torch
    from fit.npe import train as T

    class Fake:
        def __init__(self):
            self.n, self.seen = 0, []

        def get_simulations(self, starting_round=0):
            return (torch.zeros(self.n, 1),)

        def get_dataloaders(self, starting_round=0, training_batch_size=200,
                            validation_fraction=0.1, resume_training=False,
                            dataloader_kwargs=None):
            assert resume_training          # the indices are already set
            self.seen.append((self.train_indices.clone(),
                              self.val_indices.clone()))
    f = Fake()
    T._fixed_split(f)
    for n in (1000, 2000, 3000):
        f.n = n
        f.get_dataloaders(validation_fraction=0.1)
    (t1, v1), (t2, v2), (t3, v3) = f.seen
    assert len(v1) == 100 and len(v2) == 200 and len(v3) == 300
    assert set(v1.tolist()) <= set(v2.tolist()) <= set(v3.tolist())
    assert set(t1.tolist()) <= set(t2.tolist()) <= set(t3.tolist())
    assert not set(t3.tolist()) & set(v3.tolist())
    assert len(t3) + len(v3) == 3000


def test_fitted_error_scale():
    """The error scale as an NPE parameter, on data whose stated errors are
    half the real ones (log10 s = log10 2).  The network itself hardly
    learns s (it lies in the scatter of the points around the curve, which
    a few thousand simulations do not teach it): its samples stay close to
    the prior.  The seeds, chosen by the exact posterior (fit.npe.seed), do
    have it right: what the network is used for."""
    from fit.dream.dream_fit import SIGMA_NAME, PosteriorProblem
    from fit.npe.train import run_npe
    p = toy()
    for ch in p.channels:                # state half the noise drawn
        ch[3][:] /= 2                    # (raw_sigma is the same array)
    pp = PosteriorProblem(p, sigma_free=True)
    res = run_npe(pp, coarse=False, rounds=2, sims_per_round=3000, seed=6,
                  n_workers=4)
    assert res.names[-1] == SIGMA_NAME and res.meta['noise']['sigma_free']
    from fit.npe.seed import seed_chains
    seeds = seed_chains(res, pp, 8, rng=3, n_samples=4000, n_workers=4)
    s = seeds.x0[:, -1]
    assert np.all(np.abs(s - np.log10(2)) < 0.1), s
    res.check(pp)
    with pytest.raises(ValueError, match='parameters'):
        res.check(p)                     # without the error scale
    back = res.posterior_problem()
    assert back.names == pp.names and back.sigma_free


def test_every_epoch_is_reported():
    """The 'epoch' events: one per epoch of each round, the last of a round
    holding the losses its statistics end with."""
    from fit.npe.simulator import Simulator
    from fit.npe.train import train
    got = []
    res = train(Simulator(toy(), seed=1), rounds=2, sims_per_round=400,
                seed=1, training=dict(max_num_epochs=6),
                callback=lambda e, i: got.append(i) if e == 'epoch' else None)
    for r, stats in enumerate(res.rounds, 1):
        mine = [e for e in got if e['round'] == r]
        assert [e['epoch'] for e in mine] == list(range(1, len(mine) + 1))
        assert mine[-1]['validation_loss'] == stats['validation_loss']
        assert mine[-1]['patience'] == 20


def test_network_grows_with_the_parameters():
    from fit.npe.train import EMBEDDING, FLOW, network_size
    assert network_size(5) == (EMBEDDING, FLOW)
    (hidden, n_out), flow = network_size(39)
    assert n_out == 96 and n_out >= 2 * 39 and min(hidden) >= 2 * n_out
    assert flow['hidden_features'] == 128
    (hidden, n_out), _ = network_size(200)
    assert n_out >= 400 and min(hidden) >= 2 * n_out
