"""fit.dream.dream_fit: DREAM posteriors of synthetic PNR data (plan §5.2
tests 1, 3, 4, 5) and the plumbing (failures, save / load, cancel, pickling).

Run from the repo root: python -m pytest tests
"""
import pathlib
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import model.stack as S                                     # noqa: E402
from fit.dream.dream_fit import (SIGMA_NAME, format_summary, load_result,  # noqa: E402
                         load_run, posterior_problem, posterior_warnings,
                         sample, save_result, summary_table)
from fit.de.de_fit import run_de                           # noqa: E402
from model.polarisation import channel_pair, in_plane      # noqa: E402

Q = np.linspace(0.006, 0.15, 150)
N = in_plane(0.0)
TRUTH = dict(thickness=80.0, MSLD_rho=5e-6, roughness_sigma=4.0)


def film(scale=1.0, background=1e-6):
    """vacuum / Fe 80 A / Si."""
    st = S.Stack([S.Layer('front', 0, 0, 0, 0, 0),
                  S.Layer('Fe', 80, 8e-6, 0, 5e-6, 0.0, 4.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])
    st.scale, st.background = scale, background
    return st


def measure(st, channels, noise_rng=None, err=0.03):
    """Channels of st at Q, dR = err R + 1e-7; Gaussian noise of dR if
    noise_rng is given."""
    p = [channel_pair(c, N) for c in channels]
    R = st.reflectivities(Q, p, warn=False)
    out = []
    for j, (P0, P) in enumerate(p):
        dR = err * R[:, j] + 1e-7
        r = R[:, j] if noise_rng is None else \
            R[:, j] + dR * noise_rng.normal(size=Q.size)
        out.append(dict(Q=Q, R=r, dR=dR, P0=P0, P=P))
    return out


def free(st, nuisance=('background',)):
    """Fe thickness, MSLD_rho, roughness and the stack's `nuisance` varied,
    starting off the truth."""
    fe = st.layers[1]
    fe.thickness, fe.MSLD_rho, fe.roughness_sigma = 75.0, 4e-6, 3.0
    fe.fit['thickness'] = {'vary': True, 'min': 60, 'max': 100}
    fe.fit['MSLD_rho'] = {'vary': True, 'min': 2e-6, 'max': 8e-6}
    fe.fit['roughness_sigma'] = {'vary': True, 'min': 0, 'max': 10}
    bounds = {'scale': (0.8, 1.2), 'background': (0.0, 5e-6)}
    start = {'scale': 1.0, 'background': 0.0}
    for a in nuisance:
        st.fit[a] = {'vary': True, 'min': bounds[a][0], 'max': bounds[a][1]}
        setattr(st, a, start[a])
    return st


def within3sd(rows, name, value):
    """value within 3 posterior sigma (16 / 84 % half-widths) of the median:
    on one noisy dataset each 95 % interval misses the truth 1 time in 20,
    so with five parameters a 95 % check would often fail by design (the
    coverage test of the plan is the statistical check)."""
    r = {r['name']: r for r in rows}[name]
    return r['median'] - 3 * r['minus'] <= value <= r['median'] + 3 * r['plus']


# -- 1. noise-free consistency ----------------------------------------------
@pytest.fixture(scope='module')
def noise_free():
    data = measure(film(), ['++', '--', '+-', '-+'])
    pp = posterior_problem(free(film()), data)
    return pp, sample(pp, n_gen=8000, seed=1, stop_when_converged=False)


def test_noise_free_recovers_truth(noise_free):
    pp, res = noise_free
    rows, best = summary_table(res, pp)
    truth = dict(TRUTH, background=1e-6)
    for r, name in zip(rows, ['thickness', 'MSLD_rho', 'roughness_sigma',
                              'background']):
        sd = (r['plus'] + r['minus']) / 2
        assert abs(r['median'] - truth[name]) < 0.2 * sd, r
        assert r['rhat'] < 1.2
    assert best['chi2'] < 1e-3
    assert posterior_warnings(res, pp.bounds) == []
    assert 'Fe.thickness' in format_summary(res, pp)


# -- 3, 4, 5. noisy half-polarised data, scale and background free ------------
@pytest.fixture(scope='module')
def noisy():
    truth = film(scale=0.97, background=2e-6)
    data = measure(truth, ['+', '-'], np.random.default_rng(5))
    pp = posterior_problem(free(film(), ('scale', 'background')), data)
    return pp, sample(pp, n_gen=10000, seed=2, stop_when_converged=False)


def test_nuisance_parameters_contain_truth(noisy):
    """Background counted once in R+ / R-: its posterior holds the truth,
    and so do the scale and the moment fitted together with it."""
    pp, res = noisy
    rows, _ = summary_table(res, pp)
    assert pp.names == ['Fe.thickness', 'Fe.MSLD_rho', 'Fe.roughness_sigma',
                        'stack.scale', 'stack.background']
    for name, value in (('Fe.thickness', 80.0), ('Fe.MSLD_rho', 5e-6),
                        ('Fe.roughness_sigma', 4.0), ('stack.scale', 0.97),
                        ('stack.background', 2e-6)):
        assert within3sd(rows, name, value), name
    assert max(r['rhat'] for r in rows) < 1.2


def test_de_agrees_with_posterior(noisy):
    """The DE optimum lies inside every 95 % interval, and no sampled
    state has a lower chi^2 than it (uniform prior: MAP = min chi^2)."""
    pp, res = noisy
    rows, best = summary_table(res, pp)
    x, cost, *_ = run_de(pp.fit, seed=0, tol=1e-8)
    for r, v in zip(rows, x):
        assert r['lo95'] <= v <= r['hi95'], r['name']
    assert cost <= best['chi2'] + 1e-6
    assert cost == pytest.approx(pp.chi2_per_point(x))


# -- sigma_free -----------------------------------------------------------------
def test_sigma_free_finds_underestimated_errors():
    """Noise twice the stated dR: log10 of the error factor near 0.3."""
    data = measure(film(), ['++', '--'], np.random.default_rng(3))
    for d in data:
        d['dR'] = d['dR'] / 2
    pp = posterior_problem(free(film(), ()), data, sigma_free=True)
    assert pp.names[-1] == SIGMA_NAME and pp.bounds[-1] == (-1.0, 1.0)
    res = sample(pp, n_gen=4000, seed=4, min_gen_after_converged=1000)
    s = res.samples(0.5)[:, -1]
    assert abs(np.median(s) - np.log10(2)) < 0.05


# -- plumbing -----------------------------------------------------------------
def small_problem(**kw):
    data = measure(film(), ['+', '-'])
    return posterior_problem(free(film(), ()), data, **kw)


def test_failures_are_minus_inf():
    pp = small_problem()
    x = np.array([80.0, 5e-6, 4.0])
    assert np.isfinite(pp(x))
    assert pp(np.array([120.0, 5e-6, 4.0])) == -np.inf      # outside bounds

    def broken(_):
        raise ZeroDivisionError
    pp.fit.model = broken
    assert pp(x) == -np.inf
    pp.fit.model = lambda _: [np.full(150, np.nan)] * 2
    assert pp(x) == -np.inf


def test_priors():
    pp = small_problem(priors={'Fe.thickness': lambda t: -0.5 * (t - 80)**2})
    x = np.array([82.0, 5e-6, 4.0])
    assert pp(x) == pytest.approx(pp.loglike(x) - 2.0)
    with pytest.raises(ValueError):
        small_problem(priors={'nope': lambda v: 0.0})


def test_pickle_and_seed():
    pp = small_problem()
    pp(np.array([80.0, 5e-6, 4.0]))                  # builds the closure
    back = pickle.loads(pickle.dumps(pp))
    x = np.array([79.0, 4.5e-6, 3.0])
    assert back(x) == pp(x)
    a = sample(pp, n_gen=300, seed=7, stop_when_converged=False)
    b = sample(back, n_gen=300, seed=7, stop_when_converged=False)
    assert np.array_equal(a.chains, b.chains)


def test_cancel_returns_partial():
    seen = []

    def cb(t, X, logp, rhat):
        seen.append(X)
        return t >= 50
    res = sample(small_problem(), n_gen=1000, seed=0, callback=cb)
    assert res.generations[-1] == 50 and len(res.acceptance) == 50
    assert seen[0].shape == (3, 3) and 60 <= seen[0][0, 0] <= 100


def test_save_load_round_trip(tmp_path):
    pp = small_problem()
    res = sample(pp, n_gen=400, seed=0, stop_when_converged=False)
    f = tmp_path / 'run.npz'
    save_result(f, res, pp, sources=['a.dat', 'b.dat'])
    back, meta = load_result(f)
    for k in ('chains', 'logp', 'generations', 'acceptance', 'pCR'):
        assert np.array_equal(getattr(back, k), getattr(res, k)), k
    assert [g for g, _ in back.rhat_history] == \
        [g for g, _ in res.rhat_history]
    assert all(np.array_equal(a, b) for (_, a), (_, b)
               in zip(back.rhat_history, res.rhat_history))
    assert back.names == pp.names and meta['sources'] == ['a.dat', 'b.dat']
    assert meta['bounds'] == [list(b) for b in pp.bounds]
    assert S.Stack.from_dict(meta['stack']).to_dict() == \
        pp.fit.stack.to_dict()


def test_de_start():
    pp = small_problem()
    res = sample(pp, n_gen=200, seed=0, x0='de', stop_when_converged=False,
                 de_options=dict(maxiter=30))
    # every chain starts at a DE population member: near the optimum
    assert np.all(np.abs(res.chains[0][:, 0] - 80.0) < 5.0)


def test_load_run_rebuilds_the_problem(tmp_path):
    """A saved run holds its data: the posterior is rebuilt from the file
    alone and gives the same log density; a changed file is refused."""
    data = measure(film(), ['++', '-'])
    data[1]['dR'][:5] = 0.0                 # dR <= 0: |R| used, as fitted
    pp = posterior_problem(free(film(), ('scale', 'background')), data,
                           sigma_free=True)
    res = sample(pp, n_gen=300, seed=0, stop_when_converged=False)
    f = tmp_path / 'run.npz'
    save_result(f, res, pp)
    back, problem, meta = load_run(f)
    assert np.array_equal(back.chains, res.chains)
    assert problem.names == pp.names and problem.bounds == pp.bounds
    assert problem.sigma_free
    assert [c[0] for c in problem.fit.channels] == ['channel 0', 'channel 1']
    assert problem.fit.pairs[1][1] is None
    x = res.best()[0]
    assert problem(x) == pp(x)
    # the data no longer match their hash
    with np.load(f) as z:
        arrays = dict(z)
    arrays['ch0_R'] = arrays['ch0_R'] * 1.01
    np.savez(tmp_path / 'bad.npz', **arrays)
    with pytest.raises(ValueError, match='hash'):
        load_run(tmp_path / 'bad.npz')


def test_continue_equals_one_run(tmp_path):
    """Stop a run, continue it (also from a saved file): the chains are
    those of one uninterrupted run."""
    pp = small_problem()
    full = sample(pp, n_gen=400, seed=3, stop_when_converged=False)
    part = sample(pp, n_gen=400, seed=3, stop_when_converged=False,
                  callback=lambda t, *a: t >= 150)
    assert part.generations[-1] == 150
    assert part.state['cancelled'] and part.state['n_gen'] == 400
    assert not full.state['cancelled']
    cont = sample(pp, n_gen=400, resume=part, stop_when_converged=False)
    assert np.array_equal(cont.chains, full.chains)
    assert np.array_equal(cont.logp, full.logp)
    assert cont.n_evals == full.n_evals
    # through a file: save the stopped run, load it, continue
    save_result(tmp_path / 'part.npz', part, pp)
    back, problem, _ = load_run(tmp_path / 'part.npz')
    assert back.state['cancelled'] and back.state['n_gen'] == 400
    cont2 = sample(problem, n_gen=400, resume=back, stop_when_converged=False)
    assert np.array_equal(cont2.chains, full.chains)
    # and continuing a finished run goes further
    more = sample(pp, n_gen=600, resume=full, stop_when_converged=False)
    assert more.generations[-1] == 600
    assert np.array_equal(more.chains[:len(full.chains)], full.chains)
    with pytest.raises(ValueError, match='exceed'):
        sample(pp, n_gen=400, resume=full)
