"""
Bayesian posterior sampling of a fit with DREAM (fit.dream.dream), next to the
differential-evolution optimiser of fit.de.

POSTERIOR -- PosteriorProblem wraps a FitProblem (cost 'chi2'): the same
free parameters, in the same order and with the same bounds, and the same
model (Stack.reflectivities with the stack's resolution, scale and
background, ONE call for every channel).  The likelihood is Gaussian in
linear R with the measured dR, normalisation constant included
(dream.make_gaussian_loglike); points with dR <= 0 use dR = |R|, as in the
DE fit, and with the FitProblem's fixed error_floor.  floor_free instead
appends 'log10_error_floor' in (-4, -0.5): a relative floor f, the errors
being sqrt((s dR)^2 + (f R)^2) (after log10_sigma_scale if both).  The
prior is uniform inside the bounds, times the priors of the
stack's fit entries (fit.priors) and any `priors` {name: log density}
given (these win on a clash).  sigma_free appends 'log10_sigma_scale' in
(-1, 1): a common factor 10^s on every dR.  A model that cannot be
evaluated (exception, non-finite R) has log-likelihood -inf.  The
polarisation of each channel is fixed (the Simulation tab's pairs, or the
channel's own).

SAMPLING -- sample() runs DREAM(ZS) (archive=True: 8 chains, snooker
updates with p_snooker 0.1 and a tempered burn-in from beta_start 1e-3, see
fit.dream.dream) or classic DREAM in the normalised coordinates u in
[0, 1]^d of FitProblem (so SLDs ~1e-6 and thicknesses ~100 are on one
footing), and returns the chains in physical units.  The first 10 % of the
generations (dream adapt_until) are the adaptation phase and are never used
as posterior samples, also when x0 starts the chains from the DE optimum's
final population ('de') or around the current values ('current').  Three
plain chains (the defaults before) can all settle in one of two modes with
R-hat ~ 1 (tests/test_dream_sampler.py TWO_MODES); chains in different
minima are reported by posterior_warnings, chains all in the wrong one
cannot be told from R-hat or log p.

JOINT -- the FitProblem may be a fit.joint.JointProblem (several datasets,
one stack): the same likelihood over every dataset's points; save_result
keeps its datasets, modes and fixed values, and the bands are drawn for one
dataset at a time (`part`).

RESULTS -- save_result / load_result (.npz), summary_table, derived, and
the bands the plots draw: predictive_bands (model R and spin asymmetry of
posterior draws over the data) and profile_bands (SLD profile of the draws,
from the slabs that enter the transfer matrix).  Bands are the percentiles
BAND_PERCENTILES (2.5, 16, 50, 84, 97.5) of the draws at each point.
chain_curves gives the same curves (R, spin asymmetry, SLD profile) for
each chain's best kept state (chain_best), drawn over the bands.
"""

import copy
import hashlib
import json
import math
import os
import subprocess
import warnings

import numpy as np

from fit.de.de_fit import run_de
from fit.priors import tension
from fit.dream.dream import DreamResult, dream, ess, make_gaussian_loglike
from fit.joint import JointProblem, part_x, parts
from fit.problem import FitProblem, parameter_names  # noqa: F401 (API)
from fit.workers import Wrapper, start as start_pool, stop as stop_pool
from model.stack import Stack

BAND_PERCENTILES = (2.5, 16, 50, 84, 97.5)
SIGMA_NAME = 'log10_sigma_scale'
SIGMA_BOUNDS = (-1.0, 1.0)
FLOOR_NAME = 'log10_error_floor'
FLOOR_BOUNDS = (-4.0, -0.5)
# where the extra parameters start (jittered_start, the DE start): the
# measured dR, a 1 % floor
EXTRA_START = {SIGMA_NAME: 0.0, FLOOR_NAME: -2.0}


class PosteriorProblem:
    """Picklable bundle: FitProblem parameters + model + data -> log
    posterior (see POSTERIOR)."""

    def __init__(self, fit_problem, *, sigma_free=False, floor_free=False,
                 priors=None):
        if fit_problem.cost != 'chi2':
            raise ValueError("the likelihood needs the 'chi2' cost (linear R "
                             "with its errors)")
        self.fit = fit_problem
        self.sigma_free = bool(sigma_free)
        self.floor_free = bool(floor_free)
        self.names = list(fit_problem.names)
        self.bounds = [(float(a), float(b))
                       for a, b in zip(fit_problem.lo, fit_problem.hi)]
        if self.sigma_free:
            self.names.append(SIGMA_NAME)
            self.bounds.append(SIGMA_BOUNDS)
        if self.floor_free:
            self.names.append(FLOOR_NAME)
            self.bounds.append(FLOOR_BOUNDS)
        # the stack's priors (fit.priors), then the API's
        self.priors = {n: p for n, p in zip(self.names, fit_problem.priors)
                       if p is not None}
        self.priors.update(priors or {})
        unknown = set(self.priors) - set(self.names)
        if unknown:
            raise ValueError('priors on unknown parameters: %s'
                             % ', '.join(sorted(unknown)))
        # a fitted floor replaces the FitProblem's fixed one
        self.data = [(R, raw if self.floor_free else sigma)
                     for (_, _, R, sigma), raw in zip(fit_problem.channels,
                                                      fit_problem.raw_sigma)]
        self.lo = np.array([b[0] for b in self.bounds])
        self.hi = np.array([b[1] for b in self.bounds])
        self._loglike = None

    # the likelihood closure is rebuilt in each process
    def __getstate__(self):
        d = dict(self.__dict__)
        d['_loglike'] = None
        return d

    @property
    def ndim(self):
        return len(self.names)

    def model(self, x):
        """Model R of each measured channel at its data points (x without
        the sigma parameter)."""
        return self.fit.model(np.asarray(x, dtype=float))

    def _safe_model(self, x):
        try:
            return self.model(x)
        except Exception as exc:                  # any failure: L = 0
            raise ValueError(str(exc)) from exc

    def loglike(self, x):
        if self._loglike is None:
            self._loglike = make_gaussian_loglike(self._safe_model, self.data,
                                                  self.sigma_free,
                                                  self.floor_free)
        with np.errstate(all='ignore'):
            return float(self._loglike(x))

    def logprior(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < self.lo) or np.any(x > self.hi):
            return -np.inf
        lp = 0.0
        for j, n in enumerate(self.names):
            if n in self.priors:
                lp += float(self.priors[n](x[j]))
        return lp

    def split(self, x):
        """(log prior, log likelihood) of x; the likelihood is -inf where
        the prior is (not evaluated there).  Their sum is self(x)."""
        lp = self.logprior(x)
        if not np.isfinite(lp):
            return -np.inf, -np.inf
        ll = self.loglike(x)
        return lp, ll if np.isfinite(ll) else -np.inf

    def __call__(self, x):
        lp, ll = self.split(x)
        return lp + ll if np.isfinite(ll) else -np.inf

    def chi2_per_point(self, x):
        """chi^2 / N of x (the DE 'chi2' cost without its priors), with the
        measured dR."""
        x = np.asarray(x, dtype=float)[:len(self.fit.params)]
        return float(np.mean(self.fit.residuals(
            self.fit.to_x(self.fit.to_u(x)))**2))

    @property
    def n_extra(self):
        """Sampled parameters besides the fit's (sigma scale, floor)."""
        return self.sigma_free + self.floor_free

    def noise(self, x):
        """(s, f): the error scale and floor of state x (1, 0 when not
        sampled)."""
        k = len(self.fit.params)
        s = 10.0 ** x[k] if self.sigma_free else 1.0
        f = 10.0 ** x[-1] if self.floor_free else 0.0
        return s, f

    def errors(self, x):
        """The likelihood's error of every channel's points at state x."""
        s, f = self.noise(x)
        return [np.sqrt((s * e) ** 2 + (f * r) ** 2) for r, e in self.data]

    def chi2_from_logp(self, x, logp):
        """chi^2 / N of states x (n, d) from their log posterior logp (n,),
        without evaluating the model: the likelihood's inverse (nan where
        logp is not finite).  With the measured dR (times nothing), or with
        a fitted floor, with the likelihood's errors at that state."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = self.fit.npoints
        out = []
        for xi, lp in zip(x, np.asarray(logp, dtype=float)):
            E = self.errors(xi)
            log_norm = -0.5 * n * np.log(2 * np.pi) - sum(
                np.log(e).sum() for e in E)
            s = 1.0 if self.floor_free else self.noise(xi)[0]
            with np.errstate(invalid='ignore'):
                out.append(-2.0 * (lp - self.logprior(xi) - log_norm) * s**2)
        chi2 = np.array(out)
        return np.where(np.isfinite(chi2), chi2 / n, np.nan)

    def to_x(self, u):
        return self.lo + np.asarray(u) * (self.hi - self.lo)

    def to_u(self, x):
        return (np.asarray(x) - self.lo) / (self.hi - self.lo)


class _Unit(Wrapper):
    """problem as a function of normalised u, for dream (picklable; without
    the problem once a pool holds it, fit.workers)."""

    def __init__(self, problem):
        self.problem = problem

    def __call__(self, u):
        return self.problem(self.problem.to_x(u))

    def split(self, u):
        return self.problem.split(self.problem.to_x(u))


def _de_start(problem, n_chains, seed, de_options):
    """n_chains starting states from a DE fit's final population (physical
    units, best first); the sigma parameter, if any, starts at 0."""
    *_, pop = run_de(problem.fit, seed=seed, full_output=True,
                     **(de_options or {}))
    if pop is None:
        raise RuntimeError('the DE start was stopped')
    if len(pop) < n_chains:
        raise ValueError('the DE population (%d) is smaller than the number '
                         'of chains (%d)' % (len(pop), n_chains))
    x0 = pop[:n_chains]
    extra = [EXTRA_START[n] for n in problem.names[len(problem.fit.params):]]
    if extra:
        x0 = np.column_stack([x0, np.tile(extra, (n_chains, 1))])
    return x0


def jittered_start(problem, n_chains, seed=None, x=None):
    """n_chains jittered copies (physical units) of x, the fit parameters'
    values (default: the current ones), e.g. a minimum of fit.multistart:
    N(0, 0.002 (hi - lo)) each, clipped to the bounds; the error scale and
    floor, if sampled, around EXTRA_START."""
    rng = np.random.default_rng(seed)
    u = problem.fit.x0() if x is None else problem.fit.to_u(x)
    k = len(problem.fit.params)
    for j, n in enumerate(problem.names[k:], k):
        u = np.r_[u, (EXTRA_START[n] - problem.lo[j])
                  / (problem.hi[j] - problem.lo[j])]
    U = np.clip(u + rng.normal(0.0, 0.002, (n_chains, len(u))), 0.0, 1.0)
    return problem.to_x(U)


def sample(problem, *, n_gen=30000, archive=True, n_chains=None, x0=None,
           n_workers=1, seed=None, stop_when_converged=True,
           min_gen_after_converged=5000, callback=None, de_options=None,
           resume=None, rhat_target=1.1, p_snooker=0.1, beta_start=1e-3,
           temper_until=None):
    """DREAM posterior sample of `problem` (a PosteriorProblem), chains in
    physical units.

    x0: None (Latin hypercube in the bounds), an (n_chains, d) array,
        'de' to start from the best members of a fit.de.de_fit.run_de final
        population (run with de_options), or 'current' for jittered copies
        of the current values (jittered_start).  The DREAM(ZS) archive is
        still seeded from the prior, and the adaptation phase is still
        discarded.
    p_snooker: snooker updates (archive only; 0 without it).
    beta_start, temper_until: the tempered burn-in of fit.dream.dream
        (beta_start 1: none).
    n_workers > 1 evaluates the chains of each generation in that many
        'spawn' processes, each given the problem once as it starts
        (fit.workers) and one chunk of the chains per generation.
    callback(generation, X, logp, rhat or None) is called after every
        generation, X in physical units; returning True stops the run and
        returns the generations done so far.
    rhat_target: converged once R-hat < rhat_target for every parameter
        (stop_when_converged, and the flag and warning of the GUI).
    resume: a result of sample() on the same problem (or load_run of it):
        go on from its last generation up to n_gen generations in all, as
        if it had never stopped; its chains, archive and settings are kept
        (archive, n_chains, x0 and seed are ignored).
    Defaults are Vrugt's (fit.dream.dream); the outlier test runs only without
    the archive."""
    state = None
    if resume is not None:
        state = resume.state
        if state is None:
            raise ValueError('this run has no sampler state: it cannot be '
                             'continued')
        if state['d'] != problem.ndim:
            raise ValueError('the run has %d parameters, the problem %d'
                             % (state['d'], problem.ndim))
        n_chains, archive, x0 = state['N'], state['archive'], None
    if n_chains is None:
        n_chains = 8 if archive else max(7, problem.ndim + 1)
    if not archive:
        p_snooker = 0.0
    if isinstance(x0, str):
        if x0 == 'de':
            x0 = _de_start(problem, n_chains, seed, de_options)
        elif x0 == 'current':
            x0 = jittered_start(problem, n_chains, seed)
        else:
            raise ValueError("x0 must be None, an array, 'de' or 'current'")
    u0 = None if x0 is None else np.clip(problem.to_u(x0), 0.0, 1.0)

    def cb(t, U, logp, pCR, rhat):
        return callback(t, problem.to_x(U), logp, rhat) is True

    pool = None
    unit = _Unit(problem)
    try:
        map_func = map
        if n_workers > 1:
            # each process gets the problem once, then one chunk of the
            # generation's states per process
            pool, unit.token = start_pool(n_workers, problem)

            def map_func(func, items):
                return pool.map(func, items,
                                chunksize=-(-len(items) // n_workers))
        res = dream(unit, [(0.0, 1.0)] * problem.ndim, n_gen,
                    n_chains, names=list(problem.names), archive=archive,
                    x0=u0, map_func=map_func, seed=seed,
                    stop_when_converged=stop_when_converged,
                    min_gen_after_converged=min_gen_after_converged,
                    callback=cb if callback is not None else None,
                    resume=state, rhat_target=rhat_target,
                    p_snooker=p_snooker, beta_start=beta_start,
                    temper_until=temper_until, logpdf_split=unit.split)
    finally:
        if pool is not None:
            stop_pool(pool, unit.token)
        unit.token = None
    res.chains = problem.to_x(res.chains)
    return res


# -- results ------------------------------------------------------------------
def _git_hash():
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def data_hash(problem):
    """sha1 of every channel's R and dR as fitted."""
    h = hashlib.sha1()
    for R, dR in problem.data:
        h.update(np.ascontiguousarray(R, dtype=float).tobytes())
        h.update(np.ascontiguousarray(dR, dtype=float).tobytes())
    return h.hexdigest()


# DreamResult.state: scalars (JSON meta) and arrays (state_<name> in .npz)
# (a state saved before snooker and tempering lacks theirs: neither)
_STATE_SCALARS = ('t', 'n_gen', 'cancelled', 'rhat_target', 'd', 'N',
                  'archive', 'thin',
                  'outlier', 'adapt_until', 'archive_every', 'n_evals', 'm',
                  'rng', 'p_snooker', 'beta_start', 'temper_until')
_STATE_ARRAYS = ('X', 'logp_X', 'Z', 'pCR', 'J', 'n_id', 'hist_logp', 'acc',
                 'chains', 'logps', 'gens', 'lp_X', 'll_X')


def save_result(path, result, problem, sources=(), datasets=()):
    """Write result and what it was computed from to `path` (.npz):
    chains, logp, generations, acceptance, R-hat history, pCR, the fitted
    data of every channel (ch<k>_Q, ch<k>_R, ch<k>_dR: the points and
    errors the likelihood used), the sampler state (state_<name>, so that
    the run can be continued), and as JSON the names, bounds, each
    channel's name, polarisation pair and norm, the Licorne Q grid if any,
    data hash, data `sources`
    (e.g. file names), git hash, stack mode flags and the stack itself, so that
    load_run can rebuild the PosteriorProblem (not its `priors`, which are
    functions).  `datasets`: the name of the dataset fitted (a joint fit
    stores its own, with its modes, fixed values and each dataset's Q grid,
    and each channel's dataset)."""
    fit = problem.fit
    st = fit.stack
    meta = dict(names=list(problem.names), bounds=problem.bounds,
                sigma_free=problem.sigma_free, floor_free=problem.floor_free,
                error_floor=problem.fit.error_floor,
                adapt_until=result.adapt_until,
                n_evals=result.n_evals, data_hash=data_hash(problem),
                sources=list(sources), git=_git_hash(),
                roughness_scheme=st.roughness_scheme,
                magnetic_smearing=st.magnetic_smearing,
                resolution=st.resolution, stack=st.to_dict(),
                grid=problem.fit.grid,
                channels=[dict(name=name, P0=P0, P=P, norm=norm)
                          for (name, *_), (P0, P), norm
                          in zip(fit.channels, fit.pairs, fit.norms)],
                datasets=list(datasets))
    if isinstance(fit, JointProblem):
        meta.update(datasets=fit.part_names,
                    joint=dict(modes=fit.modes, fixed=fit.fixed,
                               grids=[p.grid for p in fit.parts]))
        for c, k in zip(meta['channels'], fit.channel_part):
            c['part'] = k
    data = {}
    # the measured dR, before any floor (meta error_floor)
    for k, ((_, idx, R, _), dR) in enumerate(zip(problem.fit.channels,
                                                 problem.fit.raw_sigma)):
        data.update({'ch%d_Q' % k: problem.fit.Q[idx], 'ch%d_R' % k: R,
                     'ch%d_dR' % k: dR})
    state = result.state
    if state is not None:                   # what sample(resume=...) needs
        meta['state'] = {k: state[k] for k in _STATE_SCALARS if k in state}
        for k in _STATE_ARRAYS:
            if state.get(k) is not None:
                data['state_' + k] = state[k]
        data['state_rhat_gens'] = np.array([g for g, _ in state['rhat_hist']],
                                           dtype=int)
        data['state_rhat'] = np.array([r for _, r in state['rhat_hist']]) \
            if state['rhat_hist'] else np.empty((0, problem.ndim))
    gens = [g for g, _ in result.rhat_history]
    rh = np.array([r for _, r in result.rhat_history]) if gens else \
        np.empty((0, problem.ndim))
    np.savez_compressed(
        path, chains=result.chains, logp=result.logp,
        generations=result.generations, acceptance=result.acceptance,
        rhat_generations=np.array(gens, dtype=int), rhat=rh,
        pCR=result.pCR, meta=json.dumps(meta, default=_plain), **data)


def _plain(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    raise TypeError('%s is not JSON serialisable' % type(obj).__name__)


def load_result(path):
    """(DreamResult, meta dict) written by save_result."""
    with np.load(path) as f:
        meta = json.loads(str(f['meta']))
        res = DreamResult(
            chains=f['chains'], logp=f['logp'],
            generations=f['generations'], acceptance=f['acceptance'],
            rhat_history=list(zip(f['rhat_generations'].tolist(),
                                  list(f['rhat']))),
            pCR=f['pCR'], adapt_until=meta['adapt_until'],
            n_evals=meta['n_evals'], names=meta['names'])
        if 'state' in meta:
            st = dict(meta['state'])
            for k in _STATE_ARRAYS:
                st[k] = f['state_' + k] if 'state_' + k in f else None
            st['rhat_hist'] = list(zip(f['state_rhat_gens'].tolist(),
                                       list(f['state_rhat'])))
            res.state = st
    return res, meta


def load_run(path):
    """(DreamResult, PosteriorProblem, meta) of a file written by
    save_result: the problem is rebuilt from the stored stack and data, and
    checked against the stored data hash.  Without priors."""
    res, meta = load_result(path)
    if 'channels' not in meta:
        raise ValueError('this file was saved without its data (an earlier '
                         'version): only fit.dream.dream_fit.load_result can read '
                         'its chains')
    with np.load(path) as f:
        data = [dict(Q=f['ch%d_Q' % k], R=f['ch%d_R' % k],
                     dR=f['ch%d_dR' % k], name=c['name'],
                     P0=np.asarray(c['P0'], dtype=float),
                     P=None if c['P'] is None
                     else np.asarray(c['P'], dtype=float),
                     norm=float(c.get('norm', 1.0)))
                for k, c in enumerate(meta['channels'])]
    grid = meta.get('grid')
    stack = Stack.from_dict(meta['stack'])
    options = dict(sigma_free=meta['sigma_free'],
                   floor_free=meta.get('floor_free', False))
    floor = meta.get('error_floor', 0.0)
    joint = meta.get('joint')
    if joint is None:
        problem = posterior_problem(
            stack, data, grid=None if grid is None else np.array(grid),
            error_floor=floor, **options)
    else:
        fits = [(name, FitProblem(
            stack, [d for d, c in zip(data, meta['channels'])
                    if c['part'] == k], 'chi2',
            grid=None if g is None else np.array(g), error_floor=floor))
            for k, (name, g) in enumerate(zip(meta['datasets'],
                                              joint['grids']))]
        problem = PosteriorProblem(JointProblem(fits, joint['modes'],
                                                joint['fixed']), **options)
    if problem.names != list(meta['names']):
        raise ValueError('the stored stack gives the parameters %s, the '
                         'chains have %s' % (problem.names, meta['names']))
    if data_hash(problem) != meta['data_hash']:
        raise ValueError('the stored data do not match their hash')
    return res, problem, meta


def summary_table(result, problem=None, discard=0.5):
    """One dict per parameter: name, median, minus / plus (16 / 84 %
    distance from the median), lo95 / hi95, rhat (last), ess; plus the MAP
    state {'x', 'logp', 'chi2' (chi^2/N, with problem)}."""
    s = result.samples(discard)
    q = np.percentile(s, [2.5, 16, 50, 84, 97.5], axis=0)
    rhat = result.rhat_history[-1][1] if result.rhat_history else \
        np.full(s.shape[1], np.nan)
    n_eff = ess(result, discard)
    names = result.names or ['x%d' % j for j in range(s.shape[1])]
    rows = [dict(name=n, median=q[2, j], minus=q[2, j] - q[1, j],
                 plus=q[3, j] - q[2, j], lo95=q[0, j], hi95=q[4, j],
                 rhat=float(rhat[j]), ess=float(n_eff[j]))
            for j, n in enumerate(names)]
    x, lp = result.best()
    best = dict(x=x, logp=lp,
                chi2=problem.chi2_per_point(x) if problem is not None
                else math.nan)
    return rows, best


def format_summary(result, problem=None, discard=0.5):
    """summary_table as text."""
    rows, best = summary_table(result, problem, discard)
    w = max(len(r['name']) for r in rows)
    out = ['%-*s  %12s %11s %11s %12s %12s %6s %7s'
           % (w, 'parameter', 'median', '-1σ', '+1σ', '2.5%', '97.5%',
              'R̂', 'ESS')]
    for r, xb in zip(rows, best['x']):
        out.append('%-*s  %12.6g %11.4g %11.4g %12.6g %12.6g %6.3f %7.0f'
                   % (w, r['name'], r['median'], r['minus'], r['plus'],
                      r['lo95'], r['hi95'], r['rhat'], r['ess']))
    out.append('MAP: log p = %.6g, χ²/N = %.5g'
               % (best['logp'], best['chi2']))
    out.append('acceptance after adaptation = %.3f, %d evaluations'
               % (result.acceptance[result.adapt_until:].mean(),
                  result.n_evals))
    return '\n'.join(out)


def derived(result, func, discard=0.5):
    """func(x) on every posterior sample x: array of its values (e.g. the
    total moment, or rho cos(phi) of one layer)."""
    return np.array([func(x) for x in result.samples(discard)])


def chains_split(result, discard=0.5, split_nats=10):
    """True when the chains' mean log posteriors over the kept samples
    differ by more than split_nats: they sit in different minima, which
    R-hat can miss.  Chains all in one wrong minimum are not detected
    this way (hence the tempered burn-in of fit.dream.dream)."""
    g_max = result.generations[-1]
    g0 = discard * g_max if discard < 1 else discard
    keep = result.generations >= max(g0, result.adapt_until)
    if keep.sum() < 1 or g_max <= result.adapt_until:
        return False
    with np.errstate(invalid='ignore'):
        m = result.logp[keep].mean(axis=0)
    m = m[np.isfinite(m)]
    return m.size > 1 and np.ptp(m) > split_nats


def chain_best(result, discard=0.5):
    """Each chain's best kept state (highest log posterior after `discard`
    and the adaptation phase), (N, d) in physical units."""
    g_max = result.generations[-1]
    g0 = discard * g_max if discard < 1 else discard
    keep = result.generations >= max(g0, result.adapt_until)
    lp = np.where(np.isfinite(result.logp[keep]), result.logp[keep], -np.inf)
    best = lp.argmax(axis=0)
    X = result.chains[keep]
    return np.array([X[k, j] for j, k in enumerate(best)])


def chain_chi2(result, problem, discard=0.5):
    """chi^2 / N of each chain's best kept state."""
    return [problem.chi2_per_point(x) for x in chain_best(result, discard)]


def posterior_warnings(result, bounds=None, discard=0.5, rhat_max=1.1,
                       acc_min=0.05, edge=0.01, edge_frac=0.05,
                       problem=None, split_nats=10):
    """Texts for: R-hat above rhat_max at the end, chains in different
    minima (chains_split; chi^2/N of each with `problem`), acceptance below
    acc_min after adaptation, a parameter with more than edge_frac of its
    samples within `edge` x its bound width of a bound (the bound cuts into
    the posterior; only with `bounds`, e.g. PosteriorProblem.bounds), and
    with `problem` a MAP value in tension with its prior (fit.priors)."""
    out = []
    names = result.names or []
    if result.rhat_history:
        R = result.rhat_history[-1][1]
        bad = [n for n, r in zip(names, R) if not r < rhat_max]
        if bad:
            out.append('Not converged (R̂ ≥ %g): %s' % (rhat_max,
                                                       ', '.join(bad)))
    else:
        out.append('No R̂ yet: the run ended within the adaptation phase.')
    if chains_split(result, discard, split_nats):
        per = '' if problem is None else ' (χ²/N per chain: %s)' % ', '.join(
            '%.4g' % c for c in chain_chi2(result, problem, discard))
        out.append('Chains sit in different minima%s: not converged, '
                   'whatever R̂ says' % per)
    acc = result.acceptance[result.adapt_until:]
    if acc.size and acc.mean() < acc_min:
        out.append('Low acceptance after adaptation: %.3f' % acc.mean())
    if problem is not None:
        out += prior_tension(problem.fit, result.best()[0], problem.names)
    if bounds is not None and result.generations[-1] > result.adapt_until:
        s = result.samples(discard)
        for j, (lo, hi) in enumerate(bounds):
            w = edge * (hi - lo)
            for side, near in (('min', s[:, j] < lo + w),
                               ('max', s[:, j] > hi - w)):
                if near.mean() > edge_frac:
                    out.append('%s hugs its %s bound: widen it?'
                               % (names[j], side))
    return out


def prior_tension(fit_problem, x, names=None):
    """'Prior–data tension' texts of the values x (physical) of
    fit_problem's parameters that pull against their priors."""
    names = names or fit_problem.names
    out = []
    for n, p, v in zip(names, fit_problem.priors, x):
        t = None if p is None else tension(p.spec, v)
        if t:
            out.append('Prior–data tension: %s = %.6g is %s' % (n, v, t))
    return out


def posterior_problem(stack, data, qmin=-np.inf, qmax=np.inf, grid=None,
                      error_floor=0.0, **kw):
    """PosteriorProblem of FitProblem(stack, data, 'chi2', qmin, qmax,
    grid, error_floor)."""
    return PosteriorProblem(FitProblem(stack, data, 'chi2', qmin, qmax, grid,
                                       error_floor), **kw)


# -- bands ---------------------------------------------------------------------
def draws(result, n=200, discard=0.5, seed=0):
    """n random posterior samples (fewer if there are fewer), (n, d)."""
    s = result.samples(discard)
    if len(s) == 0:
        raise ValueError('no posterior sample after the adaptation phase')
    rng = np.random.default_rng(seed)
    return s[rng.choice(len(s), size=min(n, len(s)), replace=False)]


def _on_q(Q, Qb, y):
    """Rows y on the points Qb at the points Q: linear, NaN outside Qb's
    range (as the Experimental tab); y itself when Qb is Q."""
    y = np.atleast_2d(y)
    if np.array_equal(Q, Qb):
        return y
    o = np.argsort(Qb)
    return np.array([np.interp(Q, Qb[o], r[o], left=np.nan, right=np.nan)
                     for r in y])


def _asymmetry_pair(names):
    """Indices (a, b) of R+ / R- or R++ / R-- among channel names, or None."""
    for a, b in (('R+', 'R-'), ('R++', 'R--')):
        if a in names and b in names:
            return names.index(a), names.index(b)
    return None


def predictive_bands(result, problem, n=200, discard=0.5, seed=0,
                     replicates=20, part=0):
    """Model R of n posterior draws at the fitted points:
    {'channels': [{'name', 'Q', 'R', 'dR', 'bands', 'predictive'}],
     'asymmetry': {'Q', 'A', 'dA', 'bands', 'predictive'} or None},
    each (5, nQ) at BAND_PERCENTILES.  'bands' is the model alone (the
    uncertainty of the parameters); 'predictive' adds to every draw the
    measurement noise of the likelihood, N(0, dR) (with the error scale and
    floor of that draw when they are sampled), `replicates` times per draw so its percentiles are
    smooth: where replicated data would fall, about as wide as the error
    bars.  The spin asymmetry (a - b) / (a + b) is given for R+ /
    R- (or R++ / R--) on a's Q points, b interpolated onto them where
    the grids differ (NaN outside b's range).  The channels of
    dataset `part` of a joint fit (fit.joint)."""
    fit = problem.fit
    nfit = len(fit.params)
    sub = parts(fit)[part]
    # the part's channels among the likelihood's
    ks = [k for k, p in enumerate(getattr(fit, 'channel_part',
                                          [0] * len(fit.channels)))
          if p == part]
    X = draws(result, n, discard, seed)
    models = [sub.model(part_x(fit, x[:nfit], part)) for x in X]
    errs = [problem.errors(x) for x in X]
    rng = np.random.default_rng(seed + 1)
    channels, reps = [], []
    for k, (name, idx, R, dR) in enumerate(sub.channels):
        M = np.array([m[k] for m in models])
        E = np.array([e[ks[k]] for e in errs])
        rep = (M[:, None] + rng.normal(size=(len(M), replicates, M.shape[1]))
               * E[:, None]).reshape(-1, M.shape[1])
        reps.append(rep)
        channels.append(dict(
            name=name, Q=sub.Q[idx], R=R, dR=dR,
            bands=np.percentile(M, BAND_PERCENTILES, axis=0),
            predictive=np.percentile(rep, BAND_PERCENTILES, axis=0)))
    asym = None
    pair = _asymmetry_pair([c['name'] for c in channels])
    if pair is not None:
        i, j = pair
        a, b = channels[i], channels[j]
        Q = a['Q']

        def on_a(y):
            # b's rows (data, model draws, replicates) at a's Q points
            return _on_q(Q, b['Q'], y)

        Ma = np.array([m[i] for m in models])
        Mb = on_a(np.array([m[j] for m in models]))
        Rb, dRb = on_a(b['R'])[0], on_a(b['dR'])[0]
        # all-NaN columns (outside b's range) give NaN percentiles
        with np.errstate(divide='ignore', invalid='ignore'), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            tot = a['R'] + Rb
            A = (Ma - Mb) / (Ma + Mb)
            Arep = (reps[i] - on_a(reps[j])) / (reps[i] + on_a(reps[j]))
            asym = dict(Q=Q, A=(a['R'] - Rb) / tot,
                        dA=2 * np.hypot(Rb * a['dR'], a['R'] * dRb) / tot**2,
                        bands=np.nanpercentile(A, BAND_PERCENTILES, axis=0),
                        predictive=np.nanpercentile(
                            Arep, BAND_PERCENTILES, axis=0))
    return dict(channels=channels, asymmetry=asym)


def chain_curves(result, problem, z, discard=0.5, part=0):
    """The curves of each chain's best kept state (chain_best) for dataset
    `part`: [{'chain' (its index), 'chi2' (chi^2 / N, every dataset),
    'R' [the model R of each channel at its points, as predictive_bands'
    channels], 'A' (the spin asymmetry at predictive_bands' asymmetry Q,
    or None), 'nuc', 'mag' (slab NSLD and in-plane MSLD at depths z, in
    A^-2, as profile_bands)}]."""
    fit = problem.fit
    nfit = len(fit.params)
    sub = parts(fit)[part]
    st = copy.deepcopy(sub.stack)
    pair = _asymmetry_pair([c[0] for c in sub.channels])
    out = []
    for j, x in enumerate(chain_best(result, discard)):
        xp = part_x(fit, x[:nfit], part)
        R = sub.model(xp)
        A = None
        if pair is not None:
            a, b = pair
            Qa = sub.Q[sub.channels[a][1]]
            Rb = _on_q(Qa, sub.Q[sub.channels[b][1]], R[b])[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                A = (R[a] - Rb) / (R[a] + Rb)
        sub.apply(xp, st)
        st.build_sublayers()
        nuc, mag = slab_profile(st, z)
        out.append(dict(chain=j, chi2=problem.chi2_per_point(x), R=R, A=A,
                        nuc=nuc, mag=mag))
    return out


def slab_profile(st, z):
    """NSLD real part and in-plane magnetic SLD (rho cos phi, the part the
    neutrons see) of the slabs of st (sublayers built) at depths z, with
    the fronting above and the backing below them."""
    lo = st.windows()[0][0]
    e = lo + np.concatenate([[0.0], np.cumsum([s.thickness
                                               for s in st.sublayers])])
    slabs = [st.fronting] + list(st.sublayers) + [st.backing]
    k = np.searchsorted(e, z, side='right')
    return (np.array([s.NSLD_real for s in slabs])[k],
            np.array([Stack.inplane_rho(s) for s in slabs])[k])


def profile_bands(result, problem, n=200, discard=0.5, seed=0, npts=1500,
                  part=0):
    """(z, nuclear bands, magnetic bands): the slab NSLD and in-plane MSLD
    of n posterior draws on a common depth grid around the MAP stack, bands
    (5, npts) in A^-2; the stack of dataset `part` of a joint fit."""
    fit = problem.fit
    nfit = len(fit.params)
    sub = parts(fit)[part]
    st = copy.deepcopy(sub.stack)
    sub.apply(part_x(fit, result.best()[0][:nfit], part), st)
    st.build_sublayers()
    lo = st.windows()[0][0]
    hi = lo + sum(s.thickness for s in st.sublayers)
    pad = 0.15 * max(hi - lo, 10.0)
    z = np.linspace(lo - pad, hi + pad, npts)
    nuc, mag = [], []
    for x in draws(result, n, discard, seed):
        sub.apply(part_x(fit, x[:nfit], part), st)
        st.build_sublayers()
        a, b = slab_profile(st, z)
        nuc.append(a)
        mag.append(b)
    return (z, np.percentile(nuc, BAND_PERCENTILES, axis=0),
            np.percentile(mag, BAND_PERCENTILES, axis=0))
