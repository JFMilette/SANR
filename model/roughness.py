"""
Licorne interfacial roughness (Licorne manual, Appendix 10.1): pure functions.

One interface between layer a (above) and layer b (below), x measured from the
nominal interface, positive into b:

    g(y)  = tanh(y)  'tanh'      or  erf(y)  'erf'
    k     = alpha / sigma,   sigma = 1.3 sigma_L,
            alpha = 2 arctanh(1/2) (tanh) or 2 erfinv(1/2) (erf), so that the
            25-75 % width of the profile is sigma for both functions
    gamma : g(k gamma) = 0.97, i.e. gamma = 2.47584 sigma_L (tanh),
            2.09130 sigma_L (erf)
    window: [-l_a, l_b],  l_a = min(gamma, t_a/2),  l_b = min(gamma, t_b/2)
    f(x)  = F_a + (F_b - F_a) (g(k x) + 1)/2   inside the window

The window is cut into N equal sublayers sampled at their midpoints.  Outside
the window the layer keeps its nominal value, so the profile jumps by 1.5 % of
the step at every unclipped window edge: this is Licorne's behaviour and is
kept (the plot must equal the calculation).

RENORMALISATION -- when the window is clipped on the thinner side, that side's
value F is changed so that the layer keeps its nominal amount of material:

    F_a = f_b - t_a (f_b - f_a) / (l_b + t_a/2 - J)   if t_a < t_b, gamma > t_a/2
    F_b = f_a + t_b (f_b - f_a) / (t_b/2 + l_a + J)   if t_a > t_b, gamma > t_b/2

otherwise F = f; never for t_a == t_b, never for a semi-infinite medium
(t = inf).  J = int_{-l_a}^{l_b} g(k x) dx, in closed form through the even
antiderivative G (J = G(l_b) - G(l_a)).  A thin layer clipped on both sides
therefore has a different F at each of its interfaces and jumps at its
midpoint.

All values f are stacked scalars smeared with the same geometry, e.g.
[Re NSLD, Im NSLD, rho]; the renormalisation is affine in (f_a, f_b) with
weights summing to 1, so it may be applied to vector components too.
"""

import numpy as np
from scipy.special import erf, erfinv

KINDS = ('tanh', 'erf', 'none')

LICORNE_SIGMA_FACTOR = 1.3                 # sigma = 1.3 sigma_L (Nevot-Croce)
EDGE = 0.97                                # g(k gamma) at the window edge
ALPHA = {'tanh': 2.0 * np.arctanh(0.5),    # 1.0986123
         'erf': 2.0 * erfinv(0.5)}         # 0.9538726
_DELTA = {'tanh': np.arctanh(EDGE), 'erf': erfinv(EDGE)}
_G = {'tanh': np.tanh, 'erf': erf}


def _check_kind(kind):
    if kind not in KINDS:
        raise ValueError('unknown roughness function %r (expected one of %s)'
                         % (kind, ', '.join(KINDS)))


def scale_k(sigma_L, kind):
    """k = alpha / (1.3 sigma_L)."""
    return ALPHA[kind] / (LICORNE_SIGMA_FACTOR * sigma_L)


def gamma_licorne(sigma_L, kind):
    """Unclipped window half-width: g(k gamma) = 0.97.  0 for a sharp
    interface (sigma_L <= 0 or kind 'none')."""
    _check_kind(kind)
    if kind == 'none' or sigma_L <= 0:
        return 0.0
    return _DELTA[kind] / scale_k(sigma_L, kind)


def g_licorne(y, kind):
    """g(y): tanh or erf."""
    return _G[kind](y)


def antiderivative(x, k, kind):
    """Even antiderivative G of g(k x):
    tanh: logcosh(k x)/k,  erf: x erf(k x) + exp(-k^2 x^2)/(k sqrt(pi))."""
    x = np.asarray(x, dtype=float)
    if kind == 'tanh':
        y = np.abs(k * x)
        return (y + np.log1p(np.exp(-2.0 * y)) - np.log(2.0)) / k
    return x * erf(k * x) + np.exp(-(k * x)**2) / (k * np.sqrt(np.pi))


def window_integral(la, lb, k, kind):
    """J = int_{-la}^{lb} g(k x) dx = G(lb) - G(la)."""
    return float(antiderivative(lb, k, kind) - antiderivative(la, k, kind))


def window_limits(ta, tb, sigma_L, kind):
    """(l_a, l_b, gamma): the window clipped at half of each thickness."""
    gamma = gamma_licorne(sigma_L, kind)
    return min(gamma, ta / 2.0), min(gamma, tb / 2.0), gamma


def renormalised(fa, fb, ta, tb, sigma_L, kind='tanh', renorm='manual'):
    """(F_a, F_b, l_a, l_b, J): the effective values of both sides (see
    RENORMALISATION).  renorm 'none' keeps F = f."""
    if renorm not in ('manual', 'none'):
        raise ValueError("renorm must be 'manual' or 'none'")
    fa = np.asarray(fa, dtype=float)
    fb = np.asarray(fb, dtype=float)
    la, lb, gamma = window_limits(ta, tb, sigma_L, kind)
    if gamma == 0.0:
        return fa, fb, 0.0, 0.0, 0.0
    k = scale_k(sigma_L, kind)
    J = window_integral(la, lb, k, kind)
    Fa, Fb = fa, fb
    # TODO: Licorne's J differs (J_eff above); unexplained.  On fixture 1
    # (sigma_L = 9.82371, t_a = 71.3237, t_b = 41.5003, tanh) Licorne's
    # output corresponds to J_eff = -3.4357 +- 1e-4 against the exact
    # J = -3.42469: it applies ~93 % of this correction.  Licorne-Py
    # (generateSublayers.py) uses this same exact formula.  No empirical
    # factor is applied here.
    if renorm == 'manual':
        if ta < tb and gamma > ta / 2.0:
            Fa = fb - ta * (fb - fa) / (lb + ta / 2.0 - J)
        elif ta > tb and gamma > tb / 2.0:
            Fb = fa + tb * (fb - fa) / (tb / 2.0 + la + J)
    return Fa, Fb, la, lb, J


def licorne_interface(fa, fb, ta, tb, sigma_L, N, kind='tanh',
                      renorm='manual'):
    """
    One interface between layer a (above) and layer b (below), Licorne
    Appendix 10.1.

    fa, fb : (nq,) stacked scalars smeared with the same geometry, e.g.
             [Re NSLD, Im NSLD, rho]
    ta, tb : thicknesses; np.inf for the fronting/backing media (never
             clipped or renormalised)
    N      : number of equal sublayers in the window (at least 1)
    x      : measured from the nominal interface, positive into b
    Returns x_centres (N,), h (sublayer thickness), values (N, nq), s (N,)
    the profile fraction, Fa (nq,), Fb (nq,), la, lb.  A sharp interface
    (sigma_L <= 0 or kind 'none') returns an empty window (N = 0, h = 0).
    """
    _check_kind(kind)
    fa = np.atleast_1d(np.asarray(fa, dtype=float))
    fb = np.atleast_1d(np.asarray(fb, dtype=float))
    Fa, Fb, la, lb, _ = renormalised(fa, fb, ta, tb, sigma_L, kind, renorm)
    if la + lb == 0.0:
        return (np.zeros(0), 0.0, np.zeros((0,) + fa.shape), np.zeros(0),
                Fa, Fb, 0.0, 0.0)
    N = max(int(N), 1)
    h = (la + lb) / N
    x = -la + h * (np.arange(N) + 0.5)
    s = profile_fraction(x, sigma_L, kind)
    values = Fa[None, :] + (Fb - Fa)[None, :] * s[:, None]
    return x, h, values, s, Fa, Fb, la, lb


def profile_fraction(x, sigma_L, kind):
    """s(x) = (g(k x) + 1)/2; a sharp step (0 for x <= 0, else 1) when the
    interface has no width."""
    x = np.asarray(x, dtype=float)
    if gamma_licorne(sigma_L, kind) == 0.0:
        return np.where(x > 0, 1.0, 0.0)
    return 0.5 * (g_licorne(scale_k(sigma_L, kind) * x, kind) + 1.0)


def licorne_profile_continuous(x, fa, fb, ta, tb, sigma_L, kind='tanh',
                               renorm='manual'):
    """The continuous profile of one interface at offsets x (nx,) from it:
    F_a + (F_b - F_a) s(x) inside [-l_a, l_b), the nominal f_a above it and
    f_b below it (the window-edge jumps included).  Shape (nx, nq)."""
    _check_kind(kind)
    x = np.asarray(x, dtype=float)
    fa = np.atleast_1d(np.asarray(fa, dtype=float))
    fb = np.atleast_1d(np.asarray(fb, dtype=float))
    Fa, Fb, la, lb, _ = renormalised(fa, fb, ta, tb, sigma_L, kind, renorm)
    s = profile_fraction(x, sigma_L, kind)[:, None]
    inside = Fa + (Fb - Fa) * s
    out = np.where((x < -la)[:, None], fa, inside)
    return np.where((x >= lb)[:, None], fb, out)


def sigma_rms_from_licorne(sigma_L, kind):
    """rms width of the Gaussian (erf) profile equivalent to Licorne's
    sigma_L.

    erf : exact, sigma_rms = 1.3 sigma_L / (alpha_1 sqrt 2) = 0.963691 sigma_L.
    tanh: a tanh profile is not an erf profile, so there is no exact rms
          equivalent.  This returns the width of the Gaussian with the SAME
          25-75 % WIDTH (1.3 sigma_L for both Licorne functions), which is the
          erf value 0.963691 sigma_L.  (The tanh profile's own rms, from its
          sech^2 derivative, is pi/(sqrt(12) k) = 1.07315 sigma_L.)
    """
    _check_kind(kind)
    if kind == 'none':
        raise ValueError("a sharp ('none') interface has no width")
    return LICORNE_SIGMA_FACTOR * sigma_L / (ALPHA['erf'] * np.sqrt(2.0))


def sigma_licorne_from_rms(sigma_rms, kind):
    """Inverse of sigma_rms_from_licorne (same convention for 'tanh')."""
    return sigma_rms / sigma_rms_from_licorne(1.0, kind)
