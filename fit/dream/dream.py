"""
dream.py -- DiffeRential Evolution Adaptive Metropolis (DREAM)

Implementation follows J.A. Vrugt, Environ. Model. Softw. 75, 273 (2016),
Eqs. (23)-(24) and Algorithm 5, with Vrugt's defaults (Table 1):

    delta = 3        max number of chain pairs per proposal
    nCR   = 3        number of crossover values CR = {1/nCR, ..., 1}
    c     = 0.1      lambda ~ U(-c, c)
    c*    = 1e-12    zeta   ~ N(0, c*)
    p(gamma=1) = 0.2 unit jump rate for mode hopping
    outlier = 'iqr'  outlier-chain test (burn-in only)
    boundary = 'fold' (only bound handling that preserves detailed balance)

Variant ('archive' flag):
    archive=False  classic DREAM: jump vectors from the *current* states of
                   the other N-1 chains (needs N >= 2*delta+1).
    archive=True   DREAM(ZS) (ter Braak & Vrugt 2008; Laloy & Vrugt 2012):
                   jump vectors from an archive Z of past states, seeded with
                   M0 = 10*d prior draws and appended every K generations.
                   Works with N = 3 chains, cannot permanently lose a mode,
                   and makes outlier resetting unnecessary.  Snooker updates
                   are not implemented.

Both pCR adaptation and outlier-chain resetting break reversibility, so they
are restricted to the adaptation phase (first `adapt_until` generations).
Samples from that phase must be discarded.

Resuming: DreamResult.state holds everything the sampler needs to go on
(states, archive, crossover statistics, random generator, history), and
dream(..., resume=state, n_gen=N_total) continues the run up to generation
N_total, as if it had never stopped (bit for bit, apart from convergence
being detected afresh).

The target is log p(x) = log prior(x) + log L(x), passed in as `logpdf`.
A uniform prior on `bounds` is implicit (fold keeps every proposal inside);
add any informative prior terms inside `logpdf` itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class DreamResult:
    chains: np.ndarray            # (n_stored, N, d) chain states
    logp: np.ndarray              # (n_stored, N)    log densities
    generations: np.ndarray       # (n_stored,)      generation index of each row
    acceptance: np.ndarray        # (n_gen,)         fraction accepted per generation
    rhat_history: list = field(default_factory=list)   # [(gen, rhat(d,)), ...]
    pCR: np.ndarray = None        # final crossover selection probabilities
    adapt_until: int = 0
    n_evals: int = 0
    names: Optional[Sequence[str]] = None
    state: Optional[dict] = None  # sampler state for dream(resume=...)

    def samples(self, discard: float | int = 0.5) -> np.ndarray:
        """Flattened posterior sample (M, d).

        discard: fraction (<1) or generation number (>=1) to drop.  Never
        keeps anything from the adaptation phase.
        """
        g_max = self.generations[-1]
        g0 = discard * g_max if discard < 1 else discard
        g0 = max(g0, self.adapt_until)
        keep = self.generations >= g0
        return self.chains[keep].reshape(-1, self.chains.shape[-1])

    def best(self) -> tuple[np.ndarray, float]:
        """Maximum-a-posteriori state visited (not a posterior summary)."""
        k = np.unravel_index(np.nanargmax(self.logp), self.logp.shape)
        return self.chains[k].copy(), float(self.logp[k])

    def summary(self, discard: float | int = 0.5) -> str:
        s = self.samples(discard)
        q = np.percentile(s, [2.5, 16, 50, 84, 97.5], axis=0)
        rhat = self.rhat_history[-1][1] if self.rhat_history else np.full(s.shape[1], np.nan)
        names = self.names or [f"x{j}" for j in range(s.shape[1])]
        w = max(len(n) for n in names)
        lines = [f"{'param':<{w}}  {'median':>12} {'-1s':>12} {'+1s':>12} "
                 f"{'2.5%':>12} {'97.5%':>12} {'Rhat':>6}"]
        for j, n in enumerate(names):
            lines.append(f"{n:<{w}}  {q[2, j]:12.5g} {q[2, j]-q[1, j]:12.4g} "
                         f"{q[3, j]-q[2, j]:12.4g} {q[0, j]:12.5g} {q[4, j]:12.5g} "
                         f"{rhat[j]:6.3f}")
        lines.append(f"acceptance (post-adapt) = "
                     f"{self.acceptance[self.adapt_until:].mean():.3f},  "
                     f"evaluations = {self.n_evals}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def latin_hypercube(n: int, lo: np.ndarray, hi: np.ndarray, rng) -> np.ndarray:
    d = lo.size
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return lo + u * (hi - lo)


def handle_bounds(Xp: np.ndarray, X: np.ndarray, lo, hi, mode: str, rng) -> np.ndarray:
    """Vrugt (2016) Fig. 6.  'fold' preserves detailed balance."""
    if mode == "none":
        return Xp
    width = hi - lo
    if mode == "fold":
        return lo + np.mod(Xp - lo, width)
    if mode == "reflect":
        Y = np.where(Xp < lo, 2 * lo - Xp, Xp)
        Y = np.where(Y > hi, 2 * hi - Y, Y)
        bad = (Y < lo) | (Y > hi)                        # jumps > one width
        Y[bad] = (lo + rng.random(Y.shape) * width)[bad]
        return Y
    if mode == "bound":
        return np.clip(Xp, lo, hi)
    raise ValueError(f"unknown boundary mode {mode!r}")


def gelman_rubin(x: np.ndarray) -> np.ndarray:
    """R-hat of Gelman & Rubin (1992) as used in the DREAM toolbox.

    x: (n, N, d) -- n samples per chain, N chains.
    """
    n, N, _ = x.shape
    if n < 2:
        return np.full(x.shape[-1], np.inf)
    means = x.mean(axis=0)                          # (N, d)
    W = x.var(axis=0, ddof=1).mean(axis=0)          # within-chain
    B_n = means.var(axis=0, ddof=1)                 # between-chain / n
    sigma2 = (n - 1) / n * W + B_n
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.sqrt((N + 1) / N * sigma2 / W - (n - 1) / (N * n))
    return R


def ess(result: DreamResult, discard: float | int = 0.5) -> np.ndarray:
    """Effective sample size of each parameter over the pooled chains kept
    by result.samples(discard): multi-chain autocorrelation (Gelman et al.,
    BDA3 Sec. 11.5) summed with Geyer's initial positive sequence."""
    g_max = result.generations[-1]
    g0 = discard * g_max if discard < 1 else discard
    x = result.chains[result.generations >= max(g0, result.adapt_until)]
    n, N, d = x.shape
    if n < 4:
        return np.full(d, np.nan)
    xc = x - x.mean(axis=0)
    f = np.fft.rfft(xc, n=2 * n, axis=0)
    acov = np.fft.irfft(f * f.conj(), axis=0)[:n] / n        # (n, N, d), biased
    W = x.var(axis=0, ddof=1).mean(axis=0)
    var_plus = (n - 1) / n * W + x.mean(axis=0).var(axis=0, ddof=1)
    out = np.empty(d)
    for j in range(d):
        if var_plus[j] <= 0:
            out[j] = np.nan
            continue
        rho = 1.0 - (W[j] - acov[:, :, j].mean(axis=1)) / var_plus[j]
        rho[0] = 1.0
        pairs = rho[: n - n % 2].reshape(-1, 2).sum(axis=1)
        k = np.flatnonzero(pairs <= 0)
        pairs = pairs[: k[0] if k.size else pairs.size]
        pairs = np.minimum.accumulate(pairs)                 # monotone (Geyer)
        tau = max(-1.0 + 2.0 * pairs.sum(), 1.0 / np.log10(max(N * n, 10)))
        out[j] = N * n / tau
    return out


def _iqr_outliers(mean_logp: np.ndarray) -> np.ndarray:
    finite = np.isfinite(mean_logp)
    if finite.sum() < 4:
        return ~finite
    q1, q3 = np.percentile(mean_logp[finite], [25, 75])
    return (~finite) | (mean_logp < q1 - 2.0 * (q3 - q1))


# ----------------------------------------------------------------------------
# Main sampler
# ----------------------------------------------------------------------------
def dream(
    logpdf: Callable,
    bounds: Sequence[tuple[float, float]],
    n_gen: int = 5000,
    n_chains: Optional[int] = None,
    *,
    names: Optional[Sequence[str]] = None,
    delta: int = 3,
    nCR: int = 3,
    c: float = 0.1,
    c_star: float = 1e-12,
    p_unit_gamma: float = 0.2,
    beta0: float = 1.0,
    boundary: str = "fold",
    outlier: Optional[bool] = None,
    archive: bool = True,
    archive_every: int = 10,
    archive_m0: Optional[int] = None,
    outlier_every: int = 10,
    adapt_until: Optional[int] = None,
    thin: int = 1,
    x0: Optional[np.ndarray] = None,
    vectorized: bool = False,
    map_func: Callable = map,
    rhat_every: int = 100,
    rhat_target: float = 1.2,
    stop_when_converged: bool = False,
    min_gen_after_converged: int = 0,
    seed=None,
    callback: Optional[Callable] = None,
    resume: Optional[dict] = None,
) -> DreamResult:
    """Run DREAM.

    logpdf : x (d,) -> float     (or X (N, d) -> (N,) if vectorized=True)
             Return -np.inf for forbidden states.
    bounds : [(lo, hi), ...]     uniform prior support / initialisation box
    x0     : optional (N, d) initial population, e.g. the final population of
             a differential-evolution optimisation (counts as burn-in).
    map_func : e.g. multiprocessing.Pool(...).map to evaluate the N proposals
             of a generation in parallel (logpdf must then be picklable).
    callback : callback(t, X, logp_X, pCR, rhat or None) after every
             generation; returning True stops the run and returns the
             generations done so far.
    resume : the `state` of an earlier result: go on from its last
             generation up to n_gen (the new total).  Its chains, archive,
             adaptation phase, thin and variant are kept (n_chains, archive,
             adapt_until, thin, outlier and x0 are then ignored); the same
             logpdf and bounds must be given.  Convergence is detected
             afresh, so stop_when_converged runs at least
             min_gen_after_converged more generations.
    """
    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(b, float) for b in zip(*bounds))
    d = lo.size
    if n_chains is None:
        n_chains = 3 if archive else max(2 * delta + 1, d + 1)
    N = n_chains
    if not archive and N < 2 * delta + 1:
        raise ValueError(f"classic DREAM needs n_chains >= 2*delta+1 = {2*delta+1}")
    if outlier is None:
        outlier = not archive
    if adapt_until is None:
        adapt_until = max(n_gen // 10, 1)
    if resume is not None:
        st = resume
        if st['d'] != d:
            raise ValueError(f"the state has {st['d']} parameters, not {d}")
        if n_gen <= st['t']:
            raise ValueError(f"n_gen must exceed the {st['t']} generations done")
        N, archive, thin = st['N'], st['archive'], st['thin']
        outlier, adapt_until = st['outlier'], st['adapt_until']
        archive_every = st['archive_every']

    def evaluate(Xs):
        if vectorized:
            return np.asarray(logpdf(Xs), float)
        return np.fromiter(map_func(logpdf, list(Xs)), float, count=len(Xs))

    # --- initial population --------------------------------------------------
    if resume is not None:
        rng.bit_generator.state = st['rng']
        X, logp_X = st['X'].copy(), st['logp_X'].copy()
        n_evals = st['n_evals']
    elif x0 is not None:
        X = np.array(x0, float, copy=True)
        if X.shape != (N, d):
            raise ValueError(f"x0 must have shape {(N, d)}")
    else:
        X = latin_hypercube(N, lo, hi, rng)
    if resume is None:
        logp_X = evaluate(X)
        n_evals = N

    if archive and resume is not None:
        m = st['m']
        Z = np.empty((m + (n_gen // archive_every + 1) * N, d))
        Z[:m] = st['Z']
    elif archive:
        m0 = archive_m0 or max(10 * d, 2 * delta + 1)
        Z = np.empty((m0 + (n_gen // archive_every + 1) * N, d))
        Z[:m0] = latin_hypercube(m0, lo, hi, rng)
        m = m0

    # --- crossover bookkeeping -------------------------------------------------
    CR = np.arange(1, nCR + 1) / nCR
    pCR = np.full(nCR, 1.0 / nCR)
    J = np.zeros(nCR)          # accumulated squared normalised jump distance
    n_id = np.zeros(nCR)       # number of times each CR was used
    if resume is not None:
        pCR, J, n_id = (np.array(st[k], float) for k in ('pCR', 'J', 'n_id'))

    # --- storage --------------------------------------------------------------
    n_store = n_gen // thin + 1
    chains = np.empty((n_store, N, d))
    logps = np.empty((n_store, N))
    gens = np.empty(n_store, int)
    chains[0], logps[0], gens[0] = X, logp_X, 0
    k_store = 1
    hist_logp = np.empty((n_gen + 1, N))      # full log-density history (outlier test)
    hist_logp[0] = logp_X
    acc = np.zeros(n_gen)
    rhat_hist = []
    converged_at = None
    t0 = 0
    cancelled = False
    if resume is not None:               # the generations already done
        t0, k_store = st['t'], len(st['gens'])
        chains[:k_store], logps[:k_store] = st['chains'], st['logps']
        gens[:k_store] = st['gens']
        hist_logp[:t0 + 1] = st['hist_logp']
        acc[:t0] = st['acc']
        rhat_hist = [(int(g), np.array(r)) for g, r in st['rhat_hist']]
    others = [np.delete(np.arange(N), i) for i in range(N)]

    for t in range(t0 + 1, n_gen + 1):
        # ---------- proposals (Eq. 23, 24) ------------------------------------
        std_X = (Z[m // 2:m] if archive else X).std(axis=0)
        std_X[std_X == 0] = 1.0
        dX = np.zeros((N, d))
        id_cr = rng.choice(nCR, size=N, p=pCR)
        D = rng.integers(1, delta + 1, size=N)
        lam = rng.uniform(-c, c, size=(N, d))
        zeta = rng.normal(0.0, c_star, size=(N, d))
        for i in range(N):
            if archive:
                r = rng.choice(m, size=2 * D[i], replace=False)
                P = Z
            else:
                r = rng.choice(others[i], size=2 * D[i], replace=False)
                P = X
            a, b = r[: D[i]], r[D[i]:]
            A = np.flatnonzero(rng.random(d) <= CR[id_cr[i]])
            if A.size == 0:
                A = rng.integers(d, size=1)
            if rng.random() < p_unit_gamma:
                gamma = 1.0
            else:
                gamma = beta0 * 2.38 / np.sqrt(2 * D[i] * A.size)
            dX[i, A] = (zeta[i, A] + (1.0 + lam[i, A]) * gamma
                        * (P[a][:, A] - P[b][:, A]).sum(axis=0))
        Xp = handle_bounds(X + dX, X, lo, hi, boundary, rng)

        # ---------- evaluate & Metropolis (Eq. 19) ----------------------------
        logp_p = evaluate(Xp)
        n_evals += N
        with np.errstate(invalid="ignore"):
            log_ratio = logp_p - logp_X
        log_ratio = np.where(np.isnan(log_ratio), -np.inf, log_ratio)
        accept = np.log(rng.random(N)) < log_ratio
        X_old = X
        X = np.where(accept[:, None], Xp, X)
        logp_X = np.where(accept, logp_p, logp_X)
        acc[t - 1] = accept.mean()

        # ---------- adaptation phase -----------------------------------------
        if t <= adapt_until:
            move = X - X_old                     # zero for rejected proposals
            np.add.at(J, id_cr, ((move / std_X) ** 2).sum(axis=1))
            np.add.at(n_id, id_cr, 1)
            if np.all(n_id > 0) and J.sum() > 0:
                pCR = J / n_id
                pCR = pCR / pCR.sum()
                pCR = np.maximum(pCR, 1e-3)      # keep every CR selectable
                pCR /= pCR.sum()

        hist_logp[t] = logp_X
        if archive and t % archive_every == 0:
            Z[m:m + N] = X
            m += N
        if outlier and t <= adapt_until and t % outlier_every == 0:
            mean_lp = hist_logp[t // 2: t + 1].mean(axis=0)
            bad = _iqr_outliers(mean_lp)
            if bad.any() and not bad.all():
                best = np.argmax(np.where(bad, -np.inf, logp_X))
                X[bad] = X[best]
                logp_X[bad] = logp_X[best]
                hist_logp[: t + 1, bad] = hist_logp[: t + 1, best][:, None]

        # ---------- storage ----------------------------------------------------
        if t % thin == 0:
            chains[k_store], logps[k_store], gens[k_store] = X, logp_X, t
            k_store += 1

        # ---------- convergence ----------------------------------------------
        if t % rhat_every == 0 and t > adapt_until:
            sel = gens[:k_store] >= max(t // 2, adapt_until)
            R = gelman_rubin(chains[:k_store][sel])
            rhat_hist.append((t, R))
            if converged_at is None and np.all(R < rhat_target):
                converged_at = t
            if (stop_when_converged and converged_at is not None
                    and t - converged_at >= min_gen_after_converged):
                acc = acc[:t]
                break
        if callback is not None:
            stop = callback(t, X, logp_X, pCR, rhat_hist[-1][1] if rhat_hist else None)
            if stop is True:                    # cancelled: keep what we have
                acc = acc[:t]
                cancelled = True
                break

    state = dict(
        t=t, n_gen=n_gen, cancelled=cancelled,     # planned total, stopped?
        rhat_target=rhat_target,
        d=d, N=N, archive=archive, thin=thin, outlier=outlier,
        adapt_until=adapt_until, archive_every=archive_every,
        X=X.copy(), logp_X=logp_X.copy(), n_evals=n_evals,
        m=m if archive else 0, Z=Z[:m].copy() if archive else None,
        pCR=pCR.copy(), J=J.copy(), n_id=n_id.copy(),
        rng=rng.bit_generator.state, hist_logp=hist_logp[:t + 1].copy(),
        acc=acc[:t].copy(), chains=chains[:k_store].copy(),
        logps=logps[:k_store].copy(), gens=gens[:k_store].copy(),
        rhat_hist=[(g, r.copy()) for g, r in rhat_hist])
    return DreamResult(
        chains=chains[:k_store], logp=logps[:k_store], generations=gens[:k_store],
        acceptance=acc, rhat_history=rhat_hist, pCR=pCR,
        adapt_until=adapt_until, n_evals=n_evals, names=names, state=state,
    )


# ----------------------------------------------------------------------------
# Likelihood helper for reflectometry data
# ----------------------------------------------------------------------------
def make_gaussian_loglike(model: Callable, data: Sequence[tuple], sigma_free: bool = False):
    """Gaussian log-likelihood summed over polarisation channels.

    model(x) -> list of model reflectivities, one array per channel, already
                resolution-convolved and including scale/background.
    data     -> list of (R_obs, dR_obs) arrays, same order as model output.
    sigma_free: if True, the last element of x is log10(s), a common
                multiplier on all dR (accounts for under/over-estimated errors).
    """
    R_obs = [np.asarray(r, float) for r, _ in data]
    dR = [np.asarray(e, float) for _, e in data]
    n_pts = sum(r.size for r in R_obs)
    log_norm = -0.5 * n_pts * np.log(2 * np.pi) - sum(np.log(e).sum() for e in dR)

    def loglike(x):
        x = np.asarray(x, float)
        s = 1.0
        if sigma_free:
            s = 10.0 ** x[-1]
            x = x[:-1]
        try:
            R_mod = model(x)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return -np.inf
        chi2 = sum((((m - r) / e) ** 2).sum() for m, r, e in zip(R_mod, R_obs, dR))
        if not np.isfinite(chi2):
            return -np.inf
        return log_norm - n_pts * np.log(s) - 0.5 * chi2 / s**2

    return loglike
