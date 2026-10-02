"""Validation of dream.py against targets with known posteriors.

1. 10-D correlated Gaussian (Vrugt 2016, case study 3 style):
   Sigma_ii = i, rho_ij = 0.5  -> compare mean and covariance.
2. Bimodal 1-D mixture 1/6 N(-8,1) + 5/6 N(10,1) (Vrugt 2016, case 2)
   -> compare mode weight.
3. Halperin-type ridge: data constrain only u = rho*cos(phi) (in-plane
   magnetisation), uniform prior on (rho, phi).  Compare DREAM marginals
   with brute-force 2-D grid integration of the exact posterior.
"""
import numpy as np
from dream import dream, gelman_rubin


def test_gaussian(seed=1, archive=True):
    d = 10
    var = np.arange(1, d + 1, dtype=float)
    S = 0.5 * np.sqrt(np.outer(var, var))
    np.fill_diagonal(S, var)
    Si = np.linalg.inv(S)

    def logpdf(X):                       # vectorised
        return -0.5 * np.einsum("ij,jk,ik->i", X, Si, X)

    r = dream(logpdf, [(-20, 20)] * d, n_gen=20000 if archive else 10000,
              n_chains=3 if archive else 10, archive=archive,
              vectorized=True, seed=seed, thin=2)
    s = r.samples(0.5)
    m_err = np.abs(s.mean(0)) / np.sqrt(var)
    c_err = np.abs(np.cov(s.T) - S) / np.sqrt(np.outer(var, var))
    print(f"[gauss {'ZS' if archive else 'classic'}] max|mean|/sd = {m_err.max():.3f}   "
          f"max|dCov|/sd_i sd_j = {c_err.max():.3f}   "
          f"Rhat_max = {r.rhat_history[-1][1].max():.3f}   "
          f"acc = {r.acceptance[r.adapt_until:].mean():.3f}   pCR = {np.round(r.pCR, 3)}")
    assert m_err.max() < 0.15 and c_err.max() < 0.15


def test_bimodal(seed=2, archive=True):
    w = np.array([1 / 6, 5 / 6])
    mu = np.array([-8.0, 10.0])

    def logpdf(x):
        return np.log(np.sum(w * np.exp(-0.5 * (x[0] - mu) ** 2)) / np.sqrt(2 * np.pi))

    r = dream(logpdf, [(-20, 20)], n_gen=40000, n_chains=3 if archive else 10,
              archive=archive, seed=seed)
    s = r.samples(0.5)[:, 0]
    frac = np.mean(s > 0)
    print(f"[bimodal {'ZS' if archive else 'classic'}] weight(right mode) = {frac:.3f}  (exact {w[1]:.3f})")
    if archive:
        assert abs(frac - w[1]) < 0.04


def test_halperin_ridge(seed=3):
    u0, su = 2.0, 0.05                     # data constrain rho*cos(phi) only
    lo, hi = np.array([0.0, 0.0]), np.array([5.0, 90.0])   # rho, phi[deg]

    def loglike(X):
        u = X[:, 0] * np.cos(np.deg2rad(X[:, 1]))
        return -0.5 * ((u - u0) / su) ** 2

    r = dream(loglike, list(zip(lo, hi)), n_gen=30000,
              vectorized=True, seed=seed, thin=2)
    s = r.samples(0.5)

    # exact posterior on a fine grid
    rg = np.linspace(lo[0], hi[0], 2001)
    pg = np.linspace(lo[1], hi[1], 2001)
    RR, PP = np.meshgrid(rg, pg, indexing="ij")
    P = np.exp(loglike(np.c_[RR.ravel(), PP.ravel()]).reshape(RR.shape))
    P /= P.sum()
    m_rho, m_phi = (P * RR).sum(), (P * PP).sum()
    s_rho = np.sqrt((P * (RR - m_rho) ** 2).sum())
    s_phi = np.sqrt((P * (PP - m_phi) ** 2).sum())
    print(f"[ridge]   rho: DREAM {s[:,0].mean():.3f} +- {s[:,0].std():.3f}   "
          f"exact {m_rho:.3f} +- {s_rho:.3f}")
    print(f"[ridge]   phi: DREAM {s[:,1].mean():.2f} +- {s[:,1].std():.2f}   "
          f"exact {m_phi:.2f} +- {s_phi:.2f}")
    u = s[:, 0] * np.cos(np.deg2rad(s[:, 1]))
    print(f"[ridge]   u = rho cos(phi): {u.mean():.4f} +- {u.std():.4f}   "
          f"Rhat_max = {r.rhat_history[-1][1].max():.3f}")
    assert abs(s[:, 0].mean() - m_rho) < 0.1 * s_rho
    assert abs(s[:, 1].mean() - m_phi) < 0.1 * s_phi


def test_rhat_detects_nonconvergence():
    rng = np.random.default_rng(0)
    good = rng.normal(size=(1000, 8, 2))
    bad = good.copy(); bad[:, 0] += 5.0
    assert gelman_rubin(good).max() < 1.02 and gelman_rubin(bad).min() > 1.2
    print("[rhat]    good %.3f   one stuck chain %.3f"
          % (gelman_rubin(good).max(), gelman_rubin(bad).min()))


if __name__ == "__main__":
    test_rhat_detects_nonconvergence()
    test_gaussian(archive=False)
    test_gaussian(archive=True)
    test_bimodal(archive=False)      # expected to lose the minor mode
    test_bimodal(archive=True)
    test_halperin_ridge()
    print("all tests passed")
