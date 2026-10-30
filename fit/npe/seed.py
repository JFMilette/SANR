"""
DREAM starting states from a trained network (plans/npe_plan_v1.md D1):
the network proposes, the exact posterior chooses.

seed_chains(npe_result, posterior_problem, n_chains) draws n_samples from
the network's last round at the measured data, evaluates the exact log
posterior of each with the PosteriorProblem DREAM samples
(fit.dream.dream_fit), and clusters the samples in u with a Gaussian mixture
(1 to max_clusters components, the number by BIC).  Each cluster weighs the
sum of its samples' importance weights p(x | data) / q(x) (normalised over
all clusters; the uniform's Jacobian between u and x is a constant).  Every
cluster with a weight of at least min_weight shares the chains in proportion
to its weight (largest remainders), and gets one at least (taken from the
cluster with the most); a cluster's chains
start at its best distinct samples (highest posterior).  With more such
clusters than chains, the heaviest ones get a chain each.

Seeds.state() / Seeds.from_state(): plain arrays and lists, kept with the
network (fit.npe.train NPEResult.save) so that a loaded network shows the
seeds and clusters it had.

Seeds.clusters reports each cluster: weight, samples, chains, best log p
and chi^2/N, its best state and the parameters within EDGE of a bound
there: the candidate solutions, useful on their own.  The posterior is
evaluated in 'spawn' processes with n_workers > 1 (fit.workers), the
samples' order unchanged.
"""

import numpy as np

from fit.workers import Wrapper, start as start_pool, stop as stop_pool

EDGE = 0.01                       # of the bound width
CHUNKS_PER_WORKER = 50            # chunks per process: smooth progress


class _LogPost(Wrapper):
    """log posterior of a state x (picklable; fit.workers)."""

    def __init__(self, problem):
        self.problem = problem

    def __call__(self, x):
        try:
            return float(self.problem(x))
        except Exception:
            return -np.inf


def log_posterior(problem, X, n_workers=1, progress=None, cancelled=None):
    """problem(x) of each row of X (physical units); -inf where it fails.
    progress(done, total); cancelled() True stops (RuntimeError)."""
    func = _LogPost(problem)
    n = len(X)
    out = np.full(n, -np.inf)
    pool = None
    try:
        if n_workers > 1 and n > 1:
            pool, func.token = start_pool(n_workers, problem)
            chunk = max(1, -(-n // (n_workers * CHUNKS_PER_WORKER)))
            rows = pool.imap(func, list(X), chunksize=chunk)
        else:
            chunk = max(1, n // 50)
            rows = map(func, X)
        for i, v in enumerate(rows):
            out[i] = v
            if (i + 1) % chunk == 0 or i + 1 == n:
                if cancelled is not None and cancelled():
                    raise RuntimeError('stopped')
                if progress is not None:
                    progress(i + 1, n)
    finally:
        if pool is not None:
            stop_pool(pool, func.token)
        func.token = None
    return out


def cluster(u, max_clusters, rng):
    """Labels of u's rows from a Gaussian mixture of 1..max_clusters
    components, the number by BIC."""
    from sklearn.mixture import GaussianMixture
    best = None
    for k in range(1, max_clusters + 1):
        if len(u) < 10 * k:
            break
        g = GaussianMixture(k, covariance_type='full', reg_covar=1e-6,
                            random_state=int(rng.integers(2**31)))
        g.fit(u)
        bic = g.bic(u)
        if best is None or bic < best[0]:
            best = (bic, g)
    if best is None:
        return np.zeros(len(u), dtype=int)
    return best[1].predict(u)


def allocate(weights, n_chains, min_weight):
    """Chains per cluster (see the module docstring)."""
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-w)
    out = np.zeros(len(w), dtype=int)
    keep = [k for k in order if w[k] >= min_weight] or [order[0]]
    if len(keep) >= n_chains:
        out[keep[:n_chains]] = 1
        return out
    share = w[keep] / w[keep].sum() * n_chains
    got = np.floor(share).astype(int)
    left = n_chains - got.sum()
    got[np.argsort(-(share - got))[:left]] += 1
    for i in np.flatnonzero(got == 0):          # one each, from the largest
        got[np.argmax(got)] -= 1
        got[i] = 1
    out[keep] = got
    return out


class Seeds:
    """seed_chains' result: x0 (n_chains, d) physical, the clusters, and
    the samples with their labels, log posterior and log q (u)."""

    FIELDS = ('x0', 'u', 'labels', 'logp', 'logq')

    def __init__(self, x0, clusters, u, labels, logp, logq):
        self.x0, self.clusters = x0, clusters
        self.u, self.labels, self.logp, self.logq = u, labels, logp, logq

    def state(self):
        """A plain dict of numpy arrays and lists (see the module
        docstring)."""
        d = {k: np.asarray(getattr(self, k)) for k in self.FIELDS}
        d['clusters'] = [dict(c) for c in self.clusters]
        return d

    @classmethod
    def from_state(cls, d):
        return cls(clusters=[dict(c) for c in d['clusters']],
                   **{k: np.asarray(d[k]) for k in cls.FIELDS})


def seed_chains(npe_result, posterior_problem, n_chains, rng=None,
                n_samples=10_000, max_clusters=8, min_weight=1e-3,
                n_workers=1, round=-1, progress=None, cancelled=None):
    """Seeds of n_chains DREAM chains (see the module docstring)."""
    pp = posterior_problem
    npe_result.check(pp)
    rng = np.random.default_rng(rng)
    u = npe_result.sample(n_samples, round=round,
                          seed=int(rng.integers(2**31)))
    X = pp.to_x(u)
    logp = log_posterior(pp, X, n_workers, progress, cancelled)
    ok = np.isfinite(logp)
    if not ok.any():
        raise ValueError('the posterior is zero at every network sample')
    logq = npe_result.log_prob(u, round=round)
    labels = cluster(u[ok], max_clusters, rng)
    u, X, logp, logq = u[ok], X[ok], logp[ok], logq[ok]
    logw = logp - logq
    logw -= logw.max()
    ks = np.unique(labels)
    W = np.array([np.exp(logw[labels == k]).sum() for k in ks])
    W /= W.sum()
    chains = allocate(W, n_chains, min_weight)
    x0, clusters = [], []
    for k, w, c in zip(ks, W, chains):
        idx = np.flatnonzero(labels == k)
        idx = idx[np.argsort(-logp[idx])]
        _, first = np.unique(X[idx], axis=0, return_index=True)
        idx = idx[np.sort(first)]                # distinct, best first
        b = idx[0]
        picked = idx[:c]
        if c > len(picked):                      # fewer distinct samples
            picked = np.resize(picked, c)
        x0 += list(X[picked])
        ub = u[b]
        clusters.append(dict(
            label=int(k), weight=float(w), n=int(len(idx)), chains=int(c),
            best_logp=float(logp[b]), chi2=pp.chi2_per_point(X[b]),
            best_x=X[b], mean_x=pp.to_x(u[labels == k].mean(axis=0)),
            at_bound=[n for n, v in zip(pp.names, ub)
                      if v < EDGE or v > 1 - EDGE]))
    order = np.argsort([-c['weight'] for c in clusters])
    clusters = [clusters[i] for i in order]
    return Seeds(np.array(x0), clusters, u, labels, logp, logq)


def format_clusters(seeds):
    """The clusters as text."""
    out = ['%3s %9s %7s %6s %12s %9s  %s' % ('#', 'weight', 'samples',
                                             'chains', 'best log p',
                                             'χ²/N', 'at a bound')]
    for i, c in enumerate(seeds.clusters, 1):
        out.append('%3d %9.3g %7d %6d %12.6g %9.5g  %s'
                   % (i, c['weight'], c['n'], c['chains'], c['best_logp'],
                      c['chi2'], ', '.join(c['at_bound']) or '—'))
    return '\n'.join(out)
