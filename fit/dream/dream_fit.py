"""
Bayesian posterior sampling of a fit with DREAM (fit.dream.dream), next to the
differential-evolution optimiser of fit.de.

POSTERIOR -- PosteriorProblem wraps a FitProblem (cost 'chi2'): the same
free parameters, in the same order and with the same bounds, and the same
model (Stack.reflectivities with the stack's resolution, scale and
background, ONE call for every channel).  The likelihood is Gaussian in
linear R with the measured dR, normalisation constant included
(dream.make_gaussian_loglike); points with dR <= 0 use dR = |R|, as in the
DE fit.  The prior is uniform inside the bounds, times any `priors`
{name: log density} given.  sigma_free appends 'log10_sigma_scale' in
(-1, 1): a common factor 10^s on every dR.  A model that cannot be
evaluated (exception, non-finite R) has log-likelihood -inf.  The
polarisation of each channel is fixed (the Simulation tab's pairs, or the
channel's own).

SAMPLING -- sample() runs DREAM(ZS) (archive=True, 3 chains) or classic
DREAM in the normalised coordinates u in [0, 1]^d of FitProblem (so SLDs
~1e-6 and thicknesses ~100 are on one footing), and returns the chains in
physical units.  The first 10 % of the generations (dream adapt_until) are
the adaptation phase and are never used as posterior samples, also when
x0='de' starts the chains from the DE optimum's final population.

RESULTS -- save_result / load_result (.npz), summary_table, derived, and
the bands the plots draw: predictive_bands (model R and spin asymmetry of
posterior draws over the data) and profile_bands (SLD profile of the draws,
from the slabs that enter the transfer matrix).  Bands are the percentiles
BAND_PERCENTILES (2.5, 16, 50, 84, 97.5) of the draws at each point.
"""

import copy
import hashlib
import json
import math
import multiprocessing
import os
import subprocess

import numpy as np

from fit.de.de_fit import run_de
from fit.dream.dream import DreamResult, dream, ess, make_gaussian_loglike
from fit.problem import FitProblem
from model.stack import Stack

BAND_PERCENTILES = (2.5, 16, 50, 84, 97.5)
SIGMA_NAME = 'log10_sigma_scale'
SIGMA_BOUNDS = (-1.0, 1.0)


def parameter_names(stack, params):
    """'layer.attr' of each free_parameters() entry ('stack.attr' for the
    stack's own); layers sharing a name get their index: '#2 Fe.thickness'."""
    names = [l.name for l in stack.layers]
    out = []
    for i, attr, *_ in params:
        if i is None:
            out.append('stack.%s' % attr)
        elif names.count(names[i]) > 1:
            out.append('#%d %s.%s' % (i, names[i], attr))
        else:
            out.append('%s.%s' % (names[i], attr))
    return out


class PosteriorProblem:
    """Picklable bundle: FitProblem parameters + model + data -> log
    posterior (see POSTERIOR)."""

    def __init__(self, fit_problem, *, sigma_free=False, priors=None):
        if fit_problem.cost != 'chi2':
            raise ValueError("the likelihood needs the 'chi2' cost (linear R "
                             "with its errors)")
        self.fit = fit_problem
        self.sigma_free = bool(sigma_free)
        self.names = parameter_names(fit_problem.stack, fit_problem.params)
        self.bounds = [(float(a), float(b))
                       for a, b in zip(fit_problem.lo, fit_problem.hi)]
        if self.sigma_free:
            self.names.append(SIGMA_NAME)
            self.bounds.append(SIGMA_BOUNDS)
        self.priors = dict(priors or {})
        unknown = set(self.priors) - set(self.names)
        if unknown:
            raise ValueError('priors on unknown parameters: %s'
                             % ', '.join(sorted(unknown)))
        self.data = [(R, sigma) for _, _, R, sigma in fit_problem.channels]
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
                                                  self.sigma_free)
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

    def __call__(self, x):
        lp = self.logprior(x)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.loglike(x)
        return lp + ll if np.isfinite(ll) else -np.inf

    def chi2_per_point(self, x):
        """chi^2 / N of x (the DE 'chi2' cost), with the measured dR."""
        x = np.asarray(x, dtype=float)[:len(self.fit.params)]
        return self.fit(self.fit.to_u(x))

    def to_x(self, u):
        return self.lo + np.asarray(u) * (self.hi - self.lo)

    def to_u(self, x):
        return (np.asarray(x) - self.lo) / (self.hi - self.lo)


class _Unit:
    """problem as a function of normalised u, for dream (picklable)."""

    def __init__(self, problem):
        self.problem = problem

    def __call__(self, u):
        return self.problem(self.problem.to_x(u))


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
    if problem.sigma_free:
        x0 = np.column_stack([x0, np.zeros(n_chains)])
    return x0


def sample(problem, *, n_gen=30000, archive=True, n_chains=None, x0=None,
           n_workers=1, seed=None, stop_when_converged=True,
           min_gen_after_converged=5000, callback=None, de_options=None,
           resume=None, rhat_target=1.2):
    """DREAM posterior sample of `problem` (a PosteriorProblem), chains in
    physical units.

    x0: None (Latin hypercube in the bounds), an (n_chains, d) array, or
        'de' to start from the best members of a fit.de.de_fit.run_de final
        population (run with de_options).  The DREAM(ZS) archive is still
        seeded from the prior, and the adaptation phase is still discarded.
    n_workers > 1 evaluates the chains of each generation in that many
        'spawn' processes.
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
        n_chains = 3 if archive else max(7, problem.ndim + 1)
    if isinstance(x0, str):
        if x0 != 'de':
            raise ValueError("x0 must be None, an array or 'de'")
        x0 = _de_start(problem, n_chains, seed, de_options)
    u0 = None if x0 is None else np.clip(problem.to_u(x0), 0.0, 1.0)

    def cb(t, U, logp, pCR, rhat):
        return callback(t, problem.to_x(U), logp, rhat) is True

    pool = None
    try:
        if n_workers > 1:
            pool = multiprocessing.get_context('spawn').Pool(n_workers)
        res = dream(_Unit(problem), [(0.0, 1.0)] * problem.ndim, n_gen,
                    n_chains, names=list(problem.names), archive=archive,
                    x0=u0, map_func=pool.map if pool else map, seed=seed,
                    stop_when_converged=stop_when_converged,
                    min_gen_after_converged=min_gen_after_converged,
                    callback=cb if callback is not None else None,
                    resume=state, rhat_target=rhat_target)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
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
_STATE_SCALARS = ('t', 'n_gen', 'cancelled', 'rhat_target', 'd', 'N',
                  'archive', 'thin',
                  'outlier', 'adapt_until', 'archive_every', 'n_evals', 'm',
                  'rng')
_STATE_ARRAYS = ('X', 'logp_X', 'Z', 'pCR', 'J', 'n_id', 'hist_logp', 'acc',
                 'chains', 'logps', 'gens')


def save_result(path, result, problem, sources=()):
    """Write result and what it was computed from to `path` (.npz):
    chains, logp, generations, acceptance, R-hat history, pCR, the fitted
    data of every channel (ch<k>_Q, ch<k>_R, ch<k>_dR: the points and
    errors the likelihood used), the sampler state (state_<name>, so that
    the run can be continued), and as JSON the names, bounds, each
    channel's name and polarisation pair, data hash, data `sources` (e.g.
    file names), git hash, stack mode flags and the stack itself, so that
    load_run can rebuild the PosteriorProblem (not its `priors`, which are
    functions)."""
    st = problem.fit.stack
    meta = dict(names=list(problem.names), bounds=problem.bounds,
                sigma_free=problem.sigma_free, adapt_until=result.adapt_until,
                n_evals=result.n_evals, data_hash=data_hash(problem),
                sources=list(sources), git=_git_hash(),
                roughness_scheme=st.roughness_scheme,
                magnetic_smearing=st.magnetic_smearing,
                resolution=st.resolution, stack=st.to_dict(),
                channels=[dict(name=name, P0=P0, P=P) for (name, *_), (P0, P)
                          in zip(problem.fit.channels, problem.fit.pairs)])
    data = {}
    for k, (_, idx, R, dR) in enumerate(problem.fit.channels):
        data.update({'ch%d_Q' % k: problem.fit.Q[idx], 'ch%d_R' % k: R,
                     'ch%d_dR' % k: dR})
    state = result.state
    if state is not None:                   # what sample(resume=...) needs
        meta['state'] = {k: state[k] for k in _STATE_SCALARS}
        for k in _STATE_ARRAYS:
            if state[k] is not None:
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
                     else np.asarray(c['P'], dtype=float))
                for k, c in enumerate(meta['channels'])]
    problem = posterior_problem(Stack.from_dict(meta['stack']), data,
                                sigma_free=meta['sigma_free'])
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


def posterior_warnings(result, bounds=None, discard=0.5, rhat_max=1.2,
                       acc_min=0.05, edge=0.01, edge_frac=0.05):
    """Texts for: R-hat above rhat_max at the end, acceptance below acc_min
    after adaptation, a parameter with more than edge_frac of its samples
    within `edge` x its bound width of a bound (the bound cuts into the
    posterior; only with `bounds`, e.g. PosteriorProblem.bounds)."""
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
    acc = result.acceptance[result.adapt_until:]
    if acc.size and acc.mean() < acc_min:
        out.append('Low acceptance after adaptation: %.3f' % acc.mean())
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


def posterior_problem(stack, data, qmin=-np.inf, qmax=np.inf, **kw):
    """PosteriorProblem of FitProblem(stack, data, 'chi2', qmin, qmax)."""
    return PosteriorProblem(FitProblem(stack, data, 'chi2', qmin, qmax), **kw)


# -- bands ---------------------------------------------------------------------
def draws(result, n=200, discard=0.5, seed=0):
    """n random posterior samples (fewer if there are fewer), (n, d)."""
    s = result.samples(discard)
    if len(s) == 0:
        raise ValueError('no posterior sample after the adaptation phase')
    rng = np.random.default_rng(seed)
    return s[rng.choice(len(s), size=min(n, len(s)), replace=False)]


def _asymmetry_pair(names):
    """Indices (a, b) of R+ / R- or R++ / R-- among channel names, or None."""
    for a, b in (('R+', 'R-'), ('R++', 'R--')):
        if a in names and b in names:
            return names.index(a), names.index(b)
    return None


def predictive_bands(result, problem, n=200, discard=0.5, seed=0,
                     replicates=20):
    """Model R of n posterior draws at the fitted points:
    {'channels': [{'name', 'Q', 'R', 'dR', 'bands', 'predictive'}],
     'asymmetry': {'Q', 'A', 'dA', 'bands', 'predictive'} or None},
    each (5, nQ) at BAND_PERCENTILES.  'bands' is the model alone (the
    uncertainty of the parameters); 'predictive' adds to every draw the
    measurement noise of the likelihood, N(0, dR) (times 10^s of that draw
    with sigma_free), `replicates` times per draw so its percentiles are
    smooth: where replicated data would fall, about as wide as the error
    bars.  The spin asymmetry (a - b) / (a + b) is given for R+ /
    R- (or R++ / R--) fitted on the same Q points."""
    fit = problem.fit
    nfit = len(fit.params)
    X = draws(result, n, discard, seed)
    models = [problem.model(x[:nfit]) for x in X]
    s = 10.0 ** X[:, -1] if problem.sigma_free else np.ones(len(X))
    rng = np.random.default_rng(seed + 1)
    channels, reps = [], []
    for k, (name, idx, R, dR) in enumerate(fit.channels):
        M = np.array([m[k] for m in models])
        rep = (M[:, None] + rng.normal(size=(len(M), replicates, M.shape[1]))
               * dR * s[:, None, None]).reshape(-1, M.shape[1])
        reps.append(rep)
        channels.append(dict(
            name=name, Q=fit.Q[idx], R=R, dR=dR,
            bands=np.percentile(M, BAND_PERCENTILES, axis=0),
            predictive=np.percentile(rep, BAND_PERCENTILES, axis=0)))
    asym = None
    pair = _asymmetry_pair([c['name'] for c in channels])
    if pair is not None:
        a, b = (channels[k] for k in pair)
        if np.array_equal(a['Q'], b['Q']):
            i, j = pair
            with np.errstate(divide='ignore', invalid='ignore'):
                tot = a['R'] + b['R']
                A = np.array([(m[i] - m[j]) / (m[i] + m[j]) for m in models])
                Arep = (reps[i] - reps[j]) / (reps[i] + reps[j])
                asym = dict(Q=a['Q'], A=(a['R'] - b['R']) / tot,
                            dA=2 * np.hypot(b['R'] * a['dR'],
                                            a['R'] * b['dR']) / tot**2,
                            bands=np.percentile(A, BAND_PERCENTILES, axis=0),
                            predictive=np.nanpercentile(
                                Arep, BAND_PERCENTILES, axis=0))
    return dict(channels=channels, asymmetry=asym)


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


def profile_bands(result, problem, n=200, discard=0.5, seed=0, npts=1500):
    """(z, nuclear bands, magnetic bands): the slab NSLD and in-plane MSLD
    of n posterior draws on a common depth grid around the MAP stack, bands
    (5, npts) in A^-2."""
    fit = problem.fit
    nfit = len(fit.params)
    st = copy.deepcopy(fit.stack)
    fit.apply(result.best()[0][:nfit], st)
    st.build_sublayers()
    lo = st.windows()[0][0]
    hi = lo + sum(s.thickness for s in st.sublayers)
    pad = 0.15 * max(hi - lo, 10.0)
    z = np.linspace(lo - pad, hi + pad, npts)
    nuc, mag = [], []
    for x in draws(result, n, discard, seed):
        fit.apply(x[:nfit], st)
        st.build_sublayers()
        a, b = slab_profile(st, z)
        nuc.append(a)
        mag.append(b)
    return (z, np.percentile(nuc, BAND_PERCENTILES, axis=0),
            np.percentile(mag, BAND_PERCENTILES, axis=0))
