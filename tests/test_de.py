"""fit.problem and fit.de.de_fit: differential-evolution fits of polarised
channels.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import model.stack as S                                 # noqa: E402
from fit.de.de_fit import run_de, run_lbfgs            # noqa: E402
from fit.problem import FitProblem                      # noqa: E402
from model.polarisation import channel_pair, in_plane   # noqa: E402

LAB = ['++', '+-', '-+', '--']


def film(phi=0.0, theta=0.1, front=(0, 0, 0)):
    """front / Fe 80 A / Si."""
    L = [S.Layer('front', 0, front[0], 0, front[1], front[2]),
         S.Layer('Fe', 80, 8e-6, 0, 5e-6, theta, 4.0, MSLD_phi=phi),
         S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)]
    st = S.Stack(L)
    st.build_sublayers()
    return st


def measure(st, Q, n, noise=0.0, rng=None):
    """Synthetic data of the four lab channels along n, dR = 1 % of R."""
    p = [channel_pair(c, n) for c in LAB]
    R = st.reflectivities(Q, p, warn=False)
    out = []
    for j, (c, (P0, P)) in enumerate(zip(LAB, p)):
        dR = 0.01 * R[:, j]
        r = R[:, j] + (dR * rng.normal(size=Q.size) if noise else 0)
        out.append(dict(Q=Q, R=r, dR=dR, P0=P0, P=P, name='R' + c))
    return out


def vary(st, attr, lo, hi, layer=1):
    st.layers[layer].fit[attr] = {'vary': True, 'min': lo, 'max': hi}


def test_fit_recovers_phi():
    """Only rho cos(phi) is seen, so the fit finds |phi|, not its sign."""
    Q = np.linspace(0.004, 0.12, 300)
    data = measure(film(phi=0.1), Q, in_plane(0.0))
    st = film(phi=0.0)
    vary(st, 'MSLD_phi', -0.25, 0.25)
    with warnings.catch_warnings():
        warnings.simplefilter('error')              # nothing to warn about
        prob = FitProblem(st, data)
    x, cost, *_ = run_de(prob, seed=0)
    assert abs(abs(x[0]) - 0.1) <= 1e-3 and cost < 1e-6


def test_rho_and_phi_together_warn():
    st = film()
    vary(st, 'MSLD_rho', 0, 1e-5)
    vary(st, 'MSLD_phi', -0.25, 0.25)
    with pytest.warns(UserWarning, match='Fe'):
        FitProblem(st, measure(film(), np.linspace(0.01, 0.1, 50),
                               in_plane(0.0)))


def test_two_axes_break_the_mirror():
    """Along one axis, theta and its mirror -theta give the same data; a
    second axis 90 deg away tells them apart."""
    rng = np.random.default_rng(3)
    Q = np.linspace(0.005, 0.15, 400)
    L = [S.Layer('vac'), S.Layer('M', 80, 3e-6, 0, 1e-6, 0.1, 3.0),
         S.Layer('InP', 0, 1.82e-6, 0, 0, 0, 3.0)]
    truth = S.Stack(L)
    truth.build_sublayers()
    one = measure(truth, Q, in_plane(0.0), noise=1, rng=rng)
    two = one + measure(truth, Q, in_plane(np.pi/2), noise=1, rng=rng)
    st = S.Stack([S.Layer(**vars(l)) for l in L])
    st.layers[1].MSLD_theta = 0.0
    vary(st, 'MSLD_theta', -0.5, 0.5)

    p1 = FitProblem(st, one)
    assert abs(p1(p1.to_u([-0.1])) / p1(p1.to_u([0.1])) - 1) < 1e-6
    p2 = FitProblem(st, two)
    assert p2(p2.to_u([-0.1])) > 10 * p2(p2.to_u([0.1]))
    x, *_ = run_de(p2, seed=0)
    assert abs(x[0] - 0.1) <= 1e-3


def test_transverse_polarisation_warns_once():
    """Magnetic fronting, data along an axis 0.4 rad off its M: one warning
    when the problem is built, none while fitting."""
    front = (1e-6, 0.4e-6, 0.1)
    st = film(front=front)
    Q = np.linspace(0.01, 0.12, 200)
    data = measure(st, Q, in_plane(2*np.pi*0.1 + 0.4))
    vary(st, 'thickness', 70, 90)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        prob = FitProblem(st, data)
        n_init = len(W)
        run_de(prob, maxiter=3, popsize=5, polish=False, seed=0)
    assert n_init == 1 and len(W) == 1


def test_workers_fit_like_one_process():
    """Two processes find the same thickness and roughness as one."""
    Q = np.linspace(0.004, 0.12, 200)
    data = measure(film(), Q, in_plane(0.0))
    st = film()
    st.layers[1].thickness = 70.0
    vary(st, 'thickness', 60.0, 100.0)
    vary(st, 'roughness_sigma', 1.0, 8.0)
    prob = FitProblem(st, data)
    x1, c1, *_ = run_de(prob, seed=0)
    x2, c2, *_ = run_de(prob, seed=0, workers=2)
    assert np.allclose(x1, [80.0, 4.0], rtol=1e-4) and c1 < 1e-6
    assert np.allclose(x2, [80.0, 4.0], rtol=1e-4) and c2 < 1e-6


def test_workers_stop_at_once():
    """A stop request ends a parallel fit mid-generation."""
    import time
    Q = np.linspace(0.004, 0.12, 200)
    prob = FitProblem(_varied(), measure(film(), Q, in_plane(0.0)))
    # stop requested from the start: the first generation is still out in
    # the workers (which take ~1 s to start) and must be abandoned
    t0 = time.perf_counter()
    x, cost, gen, msg = run_de(prob, maxiter=10**6, tol=0, workers=2,
                               cancelled=lambda: True)
    assert msg == 'stopped' and gen == 0 and np.isfinite(cost)
    assert time.perf_counter() - t0 < 1.0


def _varied():
    st = film()
    vary(st, 'thickness', 60.0, 100.0)
    return st


def test_fit_stays_inside_bounds():
    """Truth outside [min, max]: the result is pinned at the bounds, and no
    trial model ever leaves them."""
    Q = np.linspace(0.004, 0.12, 200)
    data = measure(film(), Q, in_plane(0.0))          # 80 A, sigma 4 A
    st = film()
    st.layers[1].thickness = 95.0                      # starts above max
    vary(st, 'thickness', 50.0, 70.0)
    prob = FitProblem(st, data)
    seen = []
    residuals = prob.residuals
    prob.residuals = lambda x: seen.append(np.array(x)) or residuals(x)
    x, *_ = run_de(prob, maxiter=20, seed=0)
    X = np.array(seen)
    assert np.all(X >= prob.lo) and np.all(X <= prob.hi)
    assert x[0] == pytest.approx(70.0)


def test_fit_recovers_scale_and_background():
    """Half-polarised data: the background is counted once per channel, so
    scale and background are found together with the film."""
    Q = np.linspace(0.004, 0.2, 300)
    truth = film()
    truth.scale, truth.background = 0.97, 2e-6
    n = in_plane(0.0)
    p = [channel_pair(c, n) for c in ('+', '-')]
    R = truth.reflectivities(Q, p, warn=False)
    data = [dict(Q=Q, R=R[:, j], dR=0.01 * R[:, j], P0=P0, P=P)
            for j, (P0, P) in enumerate(p)]
    st = film()
    st.fit['scale'] = {'vary': True, 'min': 0.8, 'max': 1.2}
    st.fit['background'] = {'vary': True, 'min': 0.0, 'max': 1e-5}
    vary(st, 'thickness', 60.0, 100.0)
    prob = FitProblem(st, data)
    x, cost, *_ = run_de(prob, seed=0, tol=1e-8)
    assert [a for _, a, *_ in prob.params] == ['thickness', 'scale',
                                               'background']
    assert x == pytest.approx([80.0, 0.97, 2e-6], rel=1e-3)
    assert cost < 1e-6


def test_status_reports_generations_and_polish():
    Q = np.linspace(0.004, 0.12, 200)
    data = measure(film(), Q, in_plane(0.0))
    st = film()
    st.layers[1].thickness = 70.0
    vary(st, 'thickness', 50, 110)
    vary(st, 'NSLD_real', 6e-6, 10e-6)
    prob = FitProblem(st, data)
    seen = []
    x, cost, gen, msg = run_de(prob, maxiter=5, popsize=5, seed=0,
                               status=seen.append)
    de = [i for i in seen if i['phase'] == 'de']
    pol = [i for i in seen if i['phase'] == 'polish']
    assert [i['generation'] for i in de] == list(range(1, gen + 1))
    assert all(i['spread'] >= 0 and i['median'] >= i['best'] for i in de)
    assert pol and pol[0]['iteration'] == 0          # announced first
    assert cost <= de[-1]['best'] + 1e-12            # never worse
    # with workers, the polish's gradients go to the pool
    seen.clear()
    x2, c2, *_ = run_de(prob, maxiter=5, popsize=5, seed=0, workers=2,
                        status=seen.append)
    assert any(i['phase'] == 'polish' and i['iteration'] > 0 for i in seen)


def test_stop_and_continue():
    """A fit stopped mid-way goes on from its last population (fit.de.de_fit
    CONTINUE): the generations count on, the cost never goes back up, and
    it reaches the same optimum; stopped before its first generation, it
    leaves nothing to continue."""
    Q = np.linspace(0.004, 0.12, 200)
    prob = FitProblem(_varied(), measure(film(), Q, in_plane(0.0)))
    calls = [0]

    def after(n):
        def cancelled():
            calls[0] += 1
            return calls[0] > n
        return cancelled
    state = {}
    x, cost, gen, msg = run_de(prob, seed=0, tol=1e-8, polish=False,
                               cancelled=after(300), state=state)
    assert msg == 'stopped' and gen == state['generation'] > 0
    assert state['energies'][0] == cost
    assert np.allclose(prob.to_x(state['population'][0]), x)
    gens = []
    x2, c2, g2, msg2 = run_de(prob, seed=1, tol=1e-8, resume=state,
                              callback=lambda x, c, g: gens.append((g, c)))
    assert gens[0][0] == gen + 1 and gens[0][1] <= cost
    assert np.all(np.diff([c for _, c in gens]) <= 0)
    assert np.allclose(x2, [80.0], rtol=1e-4) and c2 < 1e-6
    state = {}
    run_de(prob, seed=0, cancelled=lambda: True, state=state)
    assert state == {}


def test_lbfgs_alone_from_the_current_values():
    """run_lbfgs: from a start near the truth it reaches it, reporting the
    start as iteration 0 then every iteration; with workers the same; a
    stop request ends it at once with the start's cost."""
    Q = np.linspace(0.004, 0.12, 200)
    data = measure(film(), Q, in_plane(0.0))
    st = film()
    st.layers[1].thickness = 78.0
    st.layers[1].roughness_sigma = 4.5
    vary(st, 'thickness', 60.0, 100.0)
    vary(st, 'roughness_sigma', 1.0, 8.0)
    prob = FitProblem(st, data)
    seen, info = [], []
    x, cost, it, msg = run_lbfgs(prob, callback=lambda *a: seen.append(a),
                                 status=info.append)
    assert np.allclose(x, [80.0, 4.0], rtol=1e-3) and cost < 1e-4
    assert [s[2] for s in seen] == list(range(it + 1)) and it > 0
    assert np.allclose(seen[0][0], [78.0, 4.5]) and seen[0][1] > cost
    assert info[-1] == {'phase': 'lbfgs', 'iteration': it, 'maxiter': 1000,
                        'best': seen[-1][1]}
    x2, c2, *_ = run_lbfgs(prob, workers=2)
    assert np.allclose(x2, x, rtol=1e-4)
    x, c, it, msg = run_lbfgs(prob, cancelled=lambda: True)
    assert msg == 'stopped' and it == 0 and np.allclose(x, [78.0, 4.5])
    assert np.isclose(c, seen[0][1])


@pytest.mark.parametrize('strategy', ['best1bin', 'currenttobest1bin',
                                      'randtobest1bin', 'rand1bin'])
def test_each_strategy_recovers_phi(strategy):
    """Every strategy of the DE page reaches the minimum of a 1-parameter
    fit (rand1bin more slowly), and an unknown one is refused."""
    Q = np.linspace(0.004, 0.12, 150)
    data = measure(film(phi=0.1), Q, in_plane(0.0))
    st = film(phi=0.0)
    vary(st, 'MSLD_phi', 0.0, 0.25)
    prob = FitProblem(st, data)
    x, cost, *_ = run_de(prob, seed=0, strategy=strategy, maxiter=60)
    assert abs(x[0] - 0.1) <= 1e-3 and cost < 1e-6
    with pytest.raises(ValueError, match='unknown strategy'):
        run_de(prob, strategy='best3bin')


def test_stopped_before_the_first_evaluation():
    """Stop before any cost was computed: every method returns the start,
    its cost and 'stopped' (not a cost of None, nor a TypeError)."""
    from fit.de.de_fit import run_lsq
    from fit.de.global_fit import METHODS, run_global
    Q = np.linspace(0.004, 0.12, 60)
    prob = FitProblem(_varied(), measure(film(), Q, in_plane(0.0)))
    u0 = prob.x0()
    runs = [lambda: run_de(prob, cancelled=lambda: True),
            lambda: run_lbfgs(prob, cancelled=lambda: True),
            lambda: run_lsq(prob, cancelled=lambda: True)]
    runs += [lambda m=m: run_global(prob, m, cancelled=lambda: True)
             for m in METHODS]
    for run in runs:
        x, cost, steps, msg = run()
        assert msg == 'stopped' and steps == 0
        assert np.allclose(x, prob.to_x(u0))
        assert isinstance(cost, float) and cost == pytest.approx(prob(u0))


class _Valley:
    """A cheap stand-in for a FitProblem: Rosenbrock's valley in n
    normalised parameters, slow enough for L-BFGS-B to need a few hundred
    iterations."""

    def __init__(self, n):
        self.params = [None] * n
        self.calls = 0

    def x0(self):
        return (np.tile([-1.2, 1.0], len(self.params))[:len(self.params)]
                + 3) / 6

    def to_x(self, u):
        return np.array(u, dtype=float)

    to_u = to_x

    def __call__(self, u):
        from scipy.optimize import rosen
        self.calls += 1
        return float(rosen(6 * np.asarray(u) - 3))


def test_lbfgs_stops_on_its_iterations_only(monkeypatch):
    """fit.de.de_fit L-BFGS-B LIMITS: scipy's default 15000 evaluations
    stopped a 62-parameter run after 200 iterations whatever Max iter
    said; the iterations are now the only limit, in run_lbfgs, the DE
    polish and the multistart, and a run that ends on it says so."""
    import fit.de.de_fit as D
    import fit.multistart as M
    assert D.lbfgs_maxfun(1000, 3) == 84000
    assert D.lbfgs_maxfun(10, 3) == 15000              # scipy's, at least
    assert D.lbfgs_maxfun(1000, 62) == 1000 * 63 * 21
    p = _Valley(62)
    x, cost, it, msg = run_lbfgs(p, maxiter=1000)
    assert msg.startswith('CONVERGENCE') and cost < 1e-6
    assert it > 250 and p.calls > 15000        # past scipy's default limit
    # Max iter is what stops a run, and the message says so
    x, cost, it, msg = run_lbfgs(_Valley(62), maxiter=40)
    assert it == 40 and D.LIMIT in msg and 'Max iter (40' in msg
    # the multistart's L-BFGS-B: converged, not cut
    k, u, c, limited = M._Local(p, 'lbfgs', 0, None)((0, p.x0()))
    assert c < 1e-6 and not limited
    monkeypatch.setattr(M, 'LBFGS_MAXITER', 30)
    k, u, c, limited = M._Local(p, 'lbfgs', 0, None)((0, p.x0()))
    assert limited
    m = M.merge(p, [(0, u, c, True), (1, u, c, False), (2, 1 - u, c + 1)])
    assert [(r['hits'], r['unfinished']) for r in m] == [(2, 1), (1, 0)]
    # the DE polish: its iterations and the same evaluation limit
    seen = []
    minimize = D.minimize

    def spy(*a, **k):
        seen.append(k['options'])
        return minimize(*a, **k)
    monkeypatch.setattr(D, 'minimize', spy)
    small = _Valley(3)
    *_, msg = run_de(small, maxiter=3, seed=0, polish_maxiter=7)
    assert seen[-1]['maxiter'] == 7 and seen[-1]['maxfun'] == 15000
    assert D.LIMIT in msg and 'polish' in msg
    *_, msg = run_de(small, maxiter=3, seed=0)
    assert seen[-1]['maxiter'] == D.POLISH_MAXITER
    assert seen[-1]['maxfun'] == D.lbfgs_maxfun(D.POLISH_MAXITER, 3)
    assert D.LIMIT not in msg
    monkeypatch.setattr(D, 'minimize', minimize)
    k, u, c, limited = M._Local(small, 'de', 0, dict(
        maxiter=3, polish_maxiter=2))((0, small.x0()))
    assert limited
