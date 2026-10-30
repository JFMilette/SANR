"""
The simulator NPE trains on (plans/npe_plan_v1.md S1, S2): the fit
problem's own model, prior and noise, so that every model feature SANR has
is supported by construction and nothing of the physics is written again.

PROBLEM -- a fit.dream.dream_fit.PosteriorProblem, the one DREAM samples,
or the fit.problem.FitProblem / fit.joint.JointProblem (the 'chi2' cost) it
is made of, as a PosteriorProblem without a fitted error scale or floor.
The parameters are its own in its normalised coordinates u in [0, 1]^d:
the fit's free ones (ties are one parameter, as in every fit), then, when
the PosteriorProblem fits them (sigma_free, floor_free), the error scale
log10_sigma_scale and the floor log10_error_floor over their bounds.

PRIOR -- UnitPrior: uniform in [0, 1]^d (the bounds) times the fit.priors
term of each parameter that has one (by name, as the PosteriorProblem has
them) (a Gaussian or a soft box from another
measurement), each normalised over the bounds, so that it is exactly the
prior DREAM samples (fit.dream.dream_fit POSTERIOR) up to a constant.
sample() draws the uniform and each prior term by rejection.

SIMULATION -- Simulator.simulate(u): fit.model(fit.to_x(u)) of each row
(its fit parameters; the noise parameters only enter the noise), every
channel of every dataset one after the other, float64.  A row
whose model fails (an exception, a non-finite value) is reported in `ok`
and must be dropped, never replaced.  n_workers > 1 evaluates the rows in
'spawn' processes that get the problem once (fit.workers); the prior draws
and the noise are drawn in the calling process, so the simulations depend
on the seed alone, not on the number of workers.

NOISE -- add_noise: Gaussian with each point's error as the likelihood has
it (PosteriorProblem.errors): the FitProblem's sigma (dR, or |R| where dR
<= 0, with any fixed error_floor), or with the error scale s and floor f of
each draw sqrt((s dR)^2 + (f R)^2), R the measured one.  `sigma` stays the
FitProblem's (the unit of the fidelity gaps and of the features).

FEATURES -- features(R): asinh(R / c_k), c_k the median error of channel k:
linear around the background, logarithmic over the decades of R above it,
and finite for the noisy R <= 0 near the background.  float32.  The network
standardises them itself (fit.npe.train).

FIDELITY -- coarse_problem(problem, n_sub, grid_step): a copy whose stacks
(each dataset's, and a mixture's models) cut every interface into at most
n_sub slabs (roughness_sublayer, either roughness scheme) and/or compute the
quadrature resolution on a grid of step sigma / grid_step (model.stack
Stack.res_grid_step); a frozen thickness spread keeps its node count.
fidelity(problem, coarse) gives, on prior draws and on draws with every
roughness at a corner of its bounds (where slicing errs most), max over Q of
|R_coarse - R_full| / sigma for each channel; choose_coarse takes the
fastest setting whose 99th percentile (FIDELITY_PERCENTILE) is below the
threshold in every channel (FIDELITY_THRESHOLD sigma), else the problem
itself.  Only training uses the coarse problem: DREAM always gets the full
one.
"""

import copy
import time

import numpy as np

from fit.joint import parts
from fit.workers import Wrapper, start as start_pool, stop as stop_pool

FIDELITY_THRESHOLD = 0.3          # sigma
FIDELITY_PERCENTILE = 99.0
# tried by choose_coarse: (n_sub, grid_step), None keeping the problem's
COARSE_CANDIDATES = [(n, g) for n in (None, 2, 3, 4) for g in (None, 4, 2)
                     if (n, g) != (None, None)]
ROUGHNESS_ATTRS = ('roughness_sigma', 'MSLD_sigma')
CHUNKS_PER_WORKER = 8             # simulate: chunks per process (progress)


# -- prior -------------------------------------------------------------------
def prior_logpdf(spec, x):
    """fit.priors.log_prior of a complete spec at the values x (array)."""
    x = np.asarray(x, dtype=float)
    if spec['kind'] == 'gauss':
        tau = spec['tau']
        return -0.5 * ((x - spec['mu']) / tau) ** 2 - \
            np.log(tau * np.sqrt(2 * np.pi))
    a, b, s = spec['a'], spec['b'], spec['s']
    lz = np.log((b - a) + s * np.sqrt(2 * np.pi))
    dist = np.where(x < a, a - x, np.where(x > b, x - b, 0.0))
    return -0.5 * (dist / s) ** 2 - lz


class UnitPrior:
    """The fit's prior in u (see PRIOR); picklable."""

    GRID = 4097                    # points of the normalisation integral

    def __init__(self, problem):
        self.lo = np.asarray(problem.lo, dtype=float)
        self.hi = np.asarray(problem.hi, dtype=float)
        self.d = len(self.lo)
        priors = problem.priors                   # a list, or by name
        if isinstance(priors, dict):
            priors = [priors.get(n) for n in problem.names]
        if any(p is not None and not hasattr(p, 'spec') for p in priors):
            raise ValueError('a prior given as a function cannot be drawn: '
                             'give it as a fit.priors spec')
        self.specs = [None if p is None else dict(p.spec) for p in priors]
        # log max of each term (rejection) and log of its integral over
        # [0, 1] in u, both relative to that max
        self.top = np.zeros(self.d)
        self.log_z = np.zeros(self.d)
        g = np.linspace(0.0, 1.0, self.GRID)
        for j, spec in enumerate(self.specs):
            if spec is None:
                continue
            lp = prior_logpdf(spec, self._x(g, j))
            self.top[j] = lp.max()
            self.log_z[j] = np.log(np.trapezoid(np.exp(lp - lp.max()), g))

    def _x(self, u, j):
        return self.lo[j] + u * (self.hi[j] - self.lo[j])

    def _term(self, u, j):
        """log of term j at u, relative to its max (<= 0 on the bounds)."""
        return prior_logpdf(self.specs[j], self._x(u, j)) - self.top[j]

    def log_prob(self, u):
        """Normalised log density of each row of u (n, d); -inf outside
        [0, 1]^d."""
        u = np.atleast_2d(np.asarray(u, dtype=float))
        out = np.zeros(len(u))
        for j, spec in enumerate(self.specs):
            if spec is not None:
                out += self._term(u[:, j], j) - self.log_z[j]
        inside = np.all((u >= 0) & (u <= 1), axis=1)
        return np.where(inside, out, -np.inf)

    def sample(self, n, rng):
        """n draws (n, d)."""
        u = rng.random((n, self.d))
        for j, spec in enumerate(self.specs):
            if spec is None:
                continue
            col, got = np.empty(n), 0
            acc = np.exp(self.log_z[j])          # acceptance of a draw
            while got < n:
                m = int(min(max(2 * (n - got) / acc, 1024), 1e7))
                c = rng.random(m)
                c = c[np.log(rng.random(m)) < self._term(c, j)]
                k = min(len(c), n - got)
                col[got:got + k] = c[:k]
                got += k
            u[:, j] = col
        return u


# -- simulation --------------------------------------------------------------
class _Model(Wrapper):
    """Every channel's model of one row u, flat, or None where it fails
    (picklable; without the problem once a pool holds it)."""

    def __init__(self, problem):
        self.problem = problem

    def __call__(self, u):
        try:
            with np.errstate(all='ignore'):
                R = np.concatenate(self.problem.model(self.problem.to_x(u)))
        except Exception:
            return None
        return R if np.all(np.isfinite(R)) else None


def _fit_problem(problem):
    """The FitProblem / JointProblem of problem (see PROBLEM)."""
    problem = getattr(problem, 'fit', problem)    # of a PosteriorProblem
    if problem.cost != 'chi2':
        raise ValueError("NPE needs the 'chi2' cost (linear R with its "
                         "errors, the likelihood DREAM uses)")
    return problem


def posterior_of(problem):
    """The PosteriorProblem of problem (see PROBLEM)."""
    from fit.dream.dream_fit import PosteriorProblem
    if isinstance(problem, PosteriorProblem):
        return problem
    return PosteriorProblem(_fit_problem(problem))


class Simulator:
    """Prior draws, model, noise and features of a fit problem (see the
    module docstring)."""

    def __init__(self, problem, *, noise=True, seed=None):
        self.posterior = pp = posterior_of(problem)
        self.problem = p = pp.fit
        self.names = list(pp.names)
        self.d = pp.ndim
        self.k = len(p.params)               # the fit's, before the noise's
        self.lo, self.hi = pp.lo, pp.hi
        self.noise = bool(noise)
        self.rng = np.random.default_rng(seed)
        self.prior = UnitPrior(pp)
        self.channel_names = [ch[0] for ch in p.channels]
        self.sigma = np.concatenate([ch[3] for ch in p.channels])
        # the likelihood's errors before scale and floor, and the measured R
        self._e = np.concatenate([e for _, e in pp.data])
        self._r = np.concatenate([r for r, _ in pp.data])
        self.sizes = [len(ch[3]) for ch in p.channels]
        self.scales = np.array([np.median(ch[3]) for ch in p.channels])
        self._c = np.repeat(self.scales, self.sizes)
        self.observed = np.concatenate([ch[2] for ch in p.channels])
        self.n_run = self.n_failed = 0

    @property
    def n_points(self):
        return len(self.sigma)

    def sample_prior(self, n, rng=None):
        return self.prior.sample(n, self.rng if rng is None else rng)

    def simulate(self, u, n_workers=1, progress=None, cancelled=None):
        """(R (n, n_points), ok (n,)): the model of each row of u, noise
        free; rows where ok is False failed (their R is NaN).
        progress(done, total) after each chunk; cancelled() True stops
        (RuntimeError)."""
        u = np.atleast_2d(np.asarray(u, dtype=float))[:, :self.k]
        n = len(u)
        R = np.full((n, self.n_points), np.nan)
        ok = np.zeros(n, dtype=bool)
        func = _Model(self.problem)
        pool = None
        try:
            if n_workers > 1 and n > 1:
                pool, func.token = start_pool(n_workers, self.problem)
                chunk = max(1, -(-n // (n_workers * CHUNKS_PER_WORKER)))
                rows = pool.imap(func, list(u), chunksize=chunk)
            else:
                chunk = max(1, n // 50)
                rows = map(func, u)
            for i, r in enumerate(rows):
                if r is not None:
                    R[i], ok[i] = r, True
                if (i + 1) % chunk == 0 or i + 1 == n:
                    if cancelled is not None and cancelled():
                        raise RuntimeError('stopped')
                    if progress is not None:
                        progress(i + 1, n)
        finally:
            if pool is not None:
                stop_pool(pool, func.token)
            func.token = None
        self.n_run += n
        self.n_failed += int(n - ok.sum())
        return R, ok

    def errors(self, u=None):
        """The likelihood's error of every point, (n_points,), or (n,
        n_points) with the error scale and floor of the rows of u (see
        NOISE)."""
        pp = self.posterior
        if not pp.n_extra or u is None:
            return self._e if not pp.n_extra else self.sigma
        x = pp.to_x(np.atleast_2d(u))
        s = 10.0 ** x[:, self.k] if pp.sigma_free else np.ones(len(x))
        f = 10.0 ** x[:, -1] if pp.floor_free else np.zeros(len(x))
        return np.sqrt((s[:, None] * self._e) ** 2 +
                       (f[:, None] * self._r) ** 2)

    def add_noise(self, R, rng=None, u=None):
        """R plus Gaussian noise of the likelihood's errors (see NOISE),
        those of the rows of u when the noise is fitted."""
        rng = self.rng if rng is None else rng
        if self.posterior.n_extra and u is None:
            raise ValueError('the error scale or floor is fitted: give the '
                             'draws u the noise is drawn for')
        return R + rng.normal(size=np.shape(R)) * self.errors(u)

    def features(self, R):
        """asinh(R / c_k), float32 (see FEATURES)."""
        return np.arcsinh(np.asarray(R, dtype=float) / self._c).astype(
            np.float32)

    def observed_features(self):
        return self.features(self.observed)

    def draw(self, u, rng=None, **kw):
        """(u kept, features): the rows of u whose model works, simulated
        with noise (when self.noise), as features; kw go to simulate."""
        u = np.atleast_2d(np.asarray(u, dtype=float))
        R, ok = self.simulate(u, **kw)
        R, u = R[ok], u[ok]
        if self.noise:
            R = self.add_noise(R, rng, u)
        return u, self.features(R)

    @property
    def failure_rate(self):
        return self.n_failed / self.n_run if self.n_run else 0.0


# -- fidelity -----------------------------------------------------------------
def _stacks(problem):
    """Every stack problem's model uses: each dataset's, and the models of
    a mixture."""
    for p in parts(problem):
        yield p.stack
        yield from getattr(p.stack, '_mix_stacks', {}).values()


def coarse_problem(problem, n_sub=None, grid_step=None):
    """A copy of problem with at most n_sub slabs per interface and the
    quadrature resolution on a grid of step sigma / grid_step (see
    FIDELITY); None keeps the problem's."""
    c = copy.deepcopy(_fit_problem(problem))
    for st in _stacks(c):
        if n_sub is not None:
            for l in st.layers:
                l.roughness_sublayer = min(max(int(l.roughness_sublayer), 1),
                                           int(n_sub))
        if grid_step is not None:
            st.res_grid_step = float(grid_step)
    return c


def roughness_index(problem):
    """Indices of the roughness parameters (ROUGHNESS_ATTRS)."""
    return [j for j, p in enumerate(problem.params)
            if p[1] in ROUGHNESS_ATTRS]


def fidelity_draws(problem, n_draws, rng):
    """n_draws prior draws, and as many with every roughness at a random
    corner of its bounds (none without a fitted roughness)."""
    sim = Simulator(problem)
    u = sim.sample_prior(n_draws, rng)
    rough = roughness_index(sim.problem)
    if rough:
        c = sim.sample_prior(n_draws, rng)
        c[:, rough] = rng.integers(0, 2, (n_draws, len(rough)))
        u = np.vstack([u, c])
    return u


def _gaps(sim, Rf, Rc, ok):
    """max_Q |Rc - Rf| / sigma per channel, (n ok, channels)."""
    g = np.abs(Rc[ok] - Rf[ok]) / sim.sigma
    ends = np.cumsum(sim.sizes)[:-1]
    return np.column_stack([part.max(axis=1)
                            for part in np.split(g, ends, axis=1)])


def fidelity(problem, coarse, n_draws=100, rng=None, n_workers=1, u=None,
             full=None):
    """{'gap' (n, channels): max_Q |R_coarse - R_full| / sigma of each
    draw, 'channels', 'p99' (each channel's FIDELITY_PERCENTILE), 'worst'
    (their largest), 'n_failed'} on fidelity_draws (or the rows u);
    full: (R, ok) of the full problem on u, if already computed."""
    rng = np.random.default_rng(rng)
    sim = Simulator(problem)
    if u is None:
        u = fidelity_draws(sim.problem, n_draws, rng)
    Rf, okf = sim.simulate(u, n_workers) if full is None else full
    Rc, okc = Simulator(coarse).simulate(u, n_workers)
    ok = okf & okc
    if not ok.any():
        raise ValueError('the model fails at every draw')
    gap = _gaps(sim, Rf, Rc, ok)
    p99 = np.percentile(gap, FIDELITY_PERCENTILE, axis=0)
    return dict(gap=gap, channels=sim.channel_names, p99=p99,
                worst=float(p99.max()), n_failed=int((~ok).sum()))


def time_per_draw(problem, u):
    """Seconds per model evaluation over the rows u (one process)."""
    m = _Model(problem)
    m(u[0])                                      # caches, compiled kernel
    t = time.perf_counter()
    for r in u:
        m(r)
    return (time.perf_counter() - t) / len(u)


def choose_coarse(problem, threshold=FIDELITY_THRESHOLD, n_draws=100,
                  rng=None, n_workers=1, candidates=None, progress=None):
    """(problem to train on, report): the fastest coarse_problem of the
    candidates (n_sub, grid_step) whose 99th percentile gap is below
    threshold in every channel, else the problem itself (see FIDELITY).
    report: {'settings' (n_sub, grid_step) or None, 'threshold',
    'time_full', 'time', 'p99' {channel: gap}, 'tried' [(settings, time,
    worst gap or None if not tried)]}.  progress(text) as it goes."""
    problem = _fit_problem(problem)
    rng = np.random.default_rng(rng)
    say = progress or (lambda text: None)
    u = fidelity_draws(problem, n_draws, rng)
    t_full = time_per_draw(problem, u[:min(10, len(u))])
    res = problem.stack.resolution
    quadrature = bool(res.get('enabled')) and \
        res.get('scheme', 'quadrature') == 'quadrature'
    most = max(max(int(l.roughness_sublayer), 1) for st in _stacks(problem)
               for l in st.layers)
    timed = []
    for n_sub, g in candidates or COARSE_CANDIDATES:
        if (g is not None and not quadrature) or \
                (n_sub is not None and n_sub >= most):
            continue                              # changes nothing
        c = coarse_problem(problem, n_sub, g)
        timed.append(((n_sub, g), time_per_draw(c, u[:min(10, len(u))]), c))
    timed.sort(key=lambda e: e[1])
    report = dict(settings=None, threshold=float(threshold), time_full=t_full,
                  time=t_full, p99={}, tried=[])
    full = None
    for settings, t, c in timed:
        if t > 0.9 * t_full:                      # not worth it
            report['tried'].append((settings, t, None))
            continue
        say('fidelity of n_sub %s, grid step %s…' % settings)
        if full is None:
            full = Simulator(problem).simulate(u, n_workers)
        f = fidelity(problem, c, u=u, n_workers=n_workers, full=full)
        report['tried'].append((settings, t, f['worst']))
        if f['worst'] < threshold:
            report.update(settings=settings, time=t,
                          p99=dict(zip(f['channels'], f['p99'].tolist())))
            return c, report
    return problem, report


def describe_coarse(report):
    """One line of text for a choose_coarse report."""
    if report['settings'] is None:
        return ('full fidelity (%.3g ms per draw): no coarser setting was '
                'within %.2g σ' % (1e3 * report['time_full'],
                                   report['threshold']))
    n_sub, g = report['settings']
    what = []
    if n_sub is not None:
        what.append('%d slabs per interface' % n_sub)
    if g is not None:
        what.append('resolution grid σ/%g' % g)
    return ('%s: %.3g ms per draw instead of %.3g, worst 99th percentile '
            'gap %.2g σ' % (', '.join(what), 1e3 * report['time'],
                            1e3 * report['time_full'],
                            max(report['p99'].values())))
