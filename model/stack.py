"""
Layer / Stack model for spin-dependent neutron reflectometry.

PROFILE -- additive interface steps.  For any quantity v,

    v(z) = v_0 + SUM_j ( v_{j+1} - v_j ) * f_j( z - Z_j )

    f_j(u) = [1 + erf( u/(sigma_j*sqrt(2)) )]/2          'erf'
           = [1 + tanh( u*2/(sigma_j*sqrt(2*pi)) )]/2    'tanh', slope-matched

Interfaces superpose, so sigma may exceed a layer thickness: the layer then
smoothly fails to reach its nominal value.  No truncation, no renormalisation.

OWNERSHIP -- layer k's roughness_sigma and roughness_sublayer both describe the
interface at the TOP of layer k.  The fronting has no interface above it, so
its values are unused; the backing's describe the last interface.

SLICING -- interface j owns the window

    [ Z_j - min(tail*sigma_j, T_above/2) ,  Z_j + min(tail*sigma_j, T_below/2) ]

cut into roughness_sublayer slabs.  Clipping at the half-thicknesses means the
interface above a layer claims at most T/2 of it and the one below claims the
rest; windows meet at the midpoint and never overlap.  Uncovered parts of a
layer become one slab each.  Clipping limits the WINDOW, never the VALUES:
every slab samples the exact profile at its own centre.

MAGNETISATION -- rho and theta are smeared independently, like any other
quantity, so the angle turns smoothly from one layer's value to the next over
the roughness width (antiparallel layers rotate through 90 deg rather than
passing through |M| = 0).  theta is interpolated between the values as entered:
0 -> 350 deg turns the long way; enter -10 deg for the short way.

ANGLES -- axes of Majkrzak Fig. 1.14: sample z || Q (film normal), x and y in
the film plane.  MSLD_theta is theta_M, the in-plane angle of M from sample x
towards y (GEPORE THE(J); also the theta of Eq. 1.128 / Fig. 1.19), stored in
turns.  phi_M (out-of-plane) is not modelled: Np_z has no effect (Halperin).
Stack.alpha is the in-plane azimuth of the lab polarisation from sample x; it
is NOT GEPORE's EPS (a rotation about x): alpha = 90 deg <=> EPS = 3pi/2, and
the spin (uu, ud, du, dd) channels are EPS = 0 (P || Q).

TRANSFER MATRIX -- Majkrzak Eq. 1.128, i.e. Table 1.2 with Np_z = 0 (Halperin:
the component of B along Q is cancelled by the demagnetising field, so only
the in-plane magnetisation is seen).  Quantisation along the film normal.

SURROUND -- a magnetic fronting/backing is off-diagonal in that basis, so its
wavevector is a 2x2 operator (Stack._surround_K), not one scalar per spin.
Lab channels are taken along the fronting's M when it is magnetic and are
flux-normalised; see calc_reflectance.
"""

import numpy as np
from scipy.special import erf


# Layer attributes a fit may vary.  Layer.fit[attr] = {'vary', 'min', 'max'},
# bounds in the same units as the attribute (Angstrom, A^-2, turns).
FIT_PARAMS = ('thickness', 'NSLD_real', 'NSLD_img', 'MSLD_rho', 'MSLD_theta',
              'roughness_sigma')


def default_fit(attr, v):
    """Fixed, with bounds around v wide enough to be a sensible start."""
    if attr == 'thickness':
        w, lo = max(0.5 * v, 10.0), 0.0
    elif attr == 'roughness_sigma':
        w, lo = max(0.5 * v, 5.0), 0.0
    elif attr == 'MSLD_rho':
        w, lo = max(0.5 * v, 0.5e-6), 0.0
    elif attr == 'MSLD_theta':
        w, lo = 0.25, -np.inf
    elif attr == 'NSLD_img':
        w, lo = max(abs(v), 1e-7), -np.inf
    else:
        w, lo = max(0.5 * abs(v), 1e-6), -np.inf
    return {'vary': False, 'min': float(max(v - w, lo)), 'max': float(v + w)}


# ---------------------------------------------------------------- Layer ----
class Layer:
    """A single homogeneous medium.

    roughness_sigma / roughness_model / roughness_sublayer describe the
    interface at the TOP of this layer.  `fit` holds the fit settings of the
    FIT_PARAMS (see fit_entry).
    """

    def __init__(self, name: str = "", thickness: float = 0.0,
                 NSLD_real: float = 0.0, NSLD_img: float = 0.0,
                 MSLD_rho: float = 0.0, MSLD_theta: float = 0.0,
                 roughness_sigma: float = 0.0, roughness_model: str = "tanh",
                 roughness_sublayer: int = 5, fit: dict = None):
        self.name = name
        self.thickness = thickness              # Angstrom
        self.NSLD_real = NSLD_real              # A^-2
        self.NSLD_img = NSLD_img                # A^-2, negative = absorbing
        self.MSLD_rho = MSLD_rho                # A^-2, in-plane magnitude
        self.MSLD_theta = MSLD_theta        # turns, theta_M of Fig. 1.14 (from x)
        self.roughness_sigma = roughness_sigma  # Angstrom, Gaussian sigma
        self.roughness_model = roughness_model  # 'erf' or 'tanh'
        self.roughness_sublayer = roughness_sublayer
        self.fit = {k: dict(v) for k, v in (fit or {}).items()}

    def fit_entry(self, attr):
        """{'vary', 'min', 'max'} of `attr`, created from default_fit if unset."""
        if attr not in self.fit:
            self.fit[attr] = default_fit(attr, getattr(self, attr))
        return self.fit[attr]

    def __repr__(self):
        return ("Layer(%r, t=%.3g, nsld=%.4g%+.4gj, msld=%.4g @ %.4g turn, "
                "sigma=%.3g, N=%d)" % (self.name, self.thickness, self.NSLD_real,
                                       self.NSLD_img, self.MSLD_rho,
                                       self.MSLD_theta, self.roughness_sigma,
                                       self.roughness_sublayer))


# ------------------------------------------------------ profile helpers ----
def _step(u, sigma, model):
    u = np.asarray(u, dtype=float)
    if sigma <= 0.0:
        return np.where(u < 0.0, 0.0, 1.0)
    if model == 'tanh':
        return 0.5 * (1.0 + np.tanh(u * 2.0 / (sigma * np.sqrt(2.0 * np.pi))))
    return 0.5 * (1.0 + erf(u / (sigma * np.sqrt(2.0))))


def _profile(z, Z, sigmas, models, vals):
    """vals: asymptotic value of each medium, len(Z)+1 entries."""
    z = np.asarray(z, dtype=float)
    vals = np.asarray(vals)
    out = np.full(z.shape, vals[0], dtype=vals.dtype)
    for j in range(len(Z)):
        out = out + (vals[j + 1] - vals[j]) * _step(z - Z[j], sigmas[j], models[j])
    return out


# ---------------------------------------------------------------- Stack ----
class Stack:
    """[fronting, L1, ..., LN, backing].  Fronting and backing are
    semi-infinite; their thickness is ignored."""

    def __init__(self, layers, tail: float = 3.0):
        if len(layers) < 2:
            raise ValueError('need at least a fronting and a backing')
        self.layers = list(layers)
        self.tail = tail                 # unclipped window half-width = tail*sigma
        self.sublayers = None
        self.transfer_matrix = None
        self.alpha = 0.0                     # rad, in-plane lab polarisation azimuth
                                             # from sample x (not EPS; see ANGLES)
        self.background = 0.0                # constant added to every channel
        self.q_in_fronting = False           # True: Q is measured inside the fronting

    # -- serialisation ------------------------------------------------------
    def to_dict(self):
        """Plain-JSON description of the sample (layers, tail, alpha,
        background)."""
        return {'tail': self.tail, 'alpha': self.alpha,
                'background': self.background,
                'q_in_fronting': self.q_in_fronting,
                'layers': [dict(vars(l), fit={k: dict(v)
                                              for k, v in l.fit.items()})
                           for l in self.layers]}

    @classmethod
    def from_dict(cls, d):
        s = cls([Layer(**l) for l in d['layers']], tail=d.get('tail', 3.0))
        s.alpha = d.get('alpha', 0.0)
        s.background = d.get('background', 0.0)
        s.q_in_fronting = bool(d.get('q_in_fronting', False))
        return s

    # -- fitting ------------------------------------------------------------
    def fit_allowed(self, i, attr):
        """Semi-infinite media have no thickness; the fronting has no
        interface above it, so no roughness."""
        n = len(self.layers)
        if attr == 'thickness':
            return 0 < i < n - 1
        if attr == 'roughness_sigma':
            return i > 0
        return True

    def free_parameters(self):
        """[(layer index, attr, value, min, max)] of every varied parameter."""
        out = []
        for i, l in enumerate(self.layers):
            for attr in FIT_PARAMS:
                e = l.fit.get(attr)
                if e and e['vary'] and self.fit_allowed(i, attr):
                    out.append((i, attr, getattr(l, attr), e['min'], e['max']))
        return out

    # -- geometry -----------------------------------------------------------
    @property
    def fronting(self):
        return self.layers[0]

    @property
    def backing(self):
        return self.layers[-1]

    def _thicknesses(self):
        return np.array([np.inf] + [l.thickness for l in self.layers[1:-1]]
                        + [np.inf], dtype=float)

    def _interfaces(self):
        """Depths Z_j of every interface; interface j is the top of layer j+1."""
        t = self._thicknesses()
        return np.concatenate([[0.0], np.cumsum(t[1:-1])])

    def windows(self):
        """(lo, hi) per interface, clipped at half the adjacent thicknesses."""
        t = self._thicknesses()
        Z = self._interfaces()
        out = []
        for j, Zj in enumerate(Z):
            w = self.tail * max(self.layers[j + 1].roughness_sigma, 0.0)
            out.append((Zj - min(w, t[j] / 2.0), Zj + min(w, t[j + 1] / 2.0)))
        return out

    # -- slicing ------------------------------------------------------------
    def build_sublayers(self):
        """Replace the interior layers by slabs carrying the smeared profile.

        Stores and returns the list; fronting and backing are not included.
        """
        t = self._thicknesses()
        Z = self._interfaces()
        n_if = len(Z)
        sig = [max(self.layers[j + 1].roughness_sigma, 0.0) for j in range(n_if)]
        mod = [self.layers[j + 1].roughness_model for j in range(n_if)]
        nsub = [max(int(self.layers[j + 1].roughness_sublayer), 1)
                for j in range(n_if)]

        nsld = np.array([complex(l.NSLD_real, l.NSLD_img) for l in self.layers])
        rho = np.array([l.MSLD_rho for l in self.layers])
        theta = np.array([l.MSLD_theta for l in self.layers])

        pts = []
        for (lo, hi), m in zip(self.windows(), nsub):
            pts.append(np.array([lo]) if hi - lo < 1e-9
                       else np.linspace(lo, hi, m + 1))
        e = np.unique(np.round(np.concatenate(pts), 6))
        e = e[np.concatenate([[True], np.diff(e) > 1e-4])]

        c = 0.5 * (e[:-1] + e[1:])
        v_n = _profile(c, Z, sig, mod, nsld)
        v_rho = _profile(c, Z, sig, mod, rho)
        v_theta = _profile(c, Z, sig, mod, theta)

        self.sublayers = [
            Layer(name='%s_%03d' % (self.layers[0].name or 'slab', i),
                  thickness=th, NSLD_real=v_n[i].real, NSLD_img=v_n[i].imag,
                  MSLD_rho=float(v_rho[i]),
                  MSLD_theta=float(v_theta[i]),
                  roughness_sigma=0.0, roughness_sublayer=1)
            for i, th in enumerate(np.diff(e))]
        return self.sublayers

    def profile(self, z):
        """Exact continuous profile: (nsld complex, msld_rho, MSLD_theta in turns)."""
        Z = self._interfaces()
        n_if = len(Z)
        sig = [max(self.layers[j + 1].roughness_sigma, 0.0) for j in range(n_if)]
        mod = [self.layers[j + 1].roughness_model for j in range(n_if)]
        nsld = np.array([complex(l.NSLD_real, l.NSLD_img) for l in self.layers])
        rho = np.array([l.MSLD_rho for l in self.layers])
        theta = np.array([l.MSLD_theta for l in self.layers])
        return (_profile(z, Z, sig, mod, nsld), _profile(z, Z, sig, mod, rho),
                _profile(z, Z, sig, mod, theta))

    # -- transfer matrix ----------------------------------------------------
    @staticmethod
    def _slab_matrices(nsld, rho, theta, d, Q):
        """4x4 matrices of many homogeneous slabs at once.

        nsld (complex), rho, theta (turns), d: arrays of length nS.
        Returns shape (nS, nQ, 4, 4).  Majkrzak Eq. 1.128 (Table 1.2, Np_z = 0).
        """
        nsld = np.asarray(nsld, dtype=complex)[:, None]
        rho = np.asarray(rho, dtype=float)[:, None]
        d = np.asarray(d, dtype=float)[:, None]
        q2 = Q**2 / 4
        S1 = np.sqrt(4*np.pi*(nsld + rho) - q2)          # S2 = -S1
        S3 = np.sqrt(4*np.pi*(nsld - rho) - q2)          # S4 = -S3

        # each hyperbolic function evaluated once
        c1, s1 = np.cosh(S1*d), np.sinh(S1*d)
        c3, s3 = np.cosh(S3*d), np.sinh(S3*d)
        sS1, sS3 = s1*S1, s3*S3
        s1S, s3S = s1/S1, s3/S3

        # mu_1 = e^{i theta}, mu_3 = -mu_1 (mu_2 = mu_1, mu_4 = mu_3), so
        # 1/(mu_3-mu_1) = -1/(2 mu_1) and every entry of Eq. 1.128 reduces to
        # a half-sum/half-difference of the two eigenchannels times 1, mu_1
        # or mu_1* (|mu_1| = 1).
        mu = np.exp(1j*2*np.pi*np.asarray(theta, dtype=float))[:, None]
        muc = mu.conj()
        cp, cm = 0.5*(c1 + c3), 0.5*(c1 - c3)
        sp, sm = 0.5*(sS1 + sS3), 0.5*(sS1 - sS3)
        tp, tm = 0.5*(s1S + s3S), 0.5*(s1S - s3S)

        M = np.empty((4, 4) + S1.shape, dtype=np.complex128)
        M[0, 0] = M[1, 1] = M[2, 2] = M[3, 3] = cp
        M[1, 0] = M[3, 2] = mu * cm
        M[0, 1] = M[2, 3] = muc * cm
        M[2, 0] = M[3, 1] = sp
        M[3, 0] = mu * sm
        M[2, 1] = muc * sm
        M[0, 2] = M[1, 3] = tp
        M[1, 2] = mu * tm
        M[0, 3] = muc * tm
        return np.ascontiguousarray(M.transpose(2, 3, 0, 1))

    @staticmethod
    def layer_matrix(layer, Q):
        """4x4 matrix of one homogeneous slab.  Returns shape (nQ, 4, 4)."""
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        return Stack._slab_matrices([complex(layer.NSLD_real, layer.NSLD_img)],
                                    [layer.MSLD_rho], [layer.MSLD_theta],
                                    [layer.thickness], Q)[0]

    def build_transfer_matrix(self, Q, use_sublayers: bool = True):
        """Product matrix A = A_N ... A_1 (deepest leftmost).  Shape (nQ,4,4).

        Uses the sliced sublayers when available, else the nominal layers.
        The fronting and backing are never part of the product; they enter
        through the boundary condition.

        All slab matrices are built in one vectorised call, then multiplied
        pairwise (tree reduction): ~log2(N) batched matmuls instead of N.
        """
        Q = np.atleast_1d(np.asarray(Q))     # may be complex: only Q**2 is used
        if use_sublayers:
            if self.sublayers is None:
                self.build_sublayers()
            seq = self.sublayers
        else:
            seq = self.layers[1:-1]
        if not seq:
            M = np.broadcast_to(np.eye(4, dtype=np.complex128),
                                Q.shape + (4, 4)).copy()
        else:
            A = self._slab_matrices(
                [complex(l.NSLD_real, l.NSLD_img) for l in seq],
                [l.MSLD_rho for l in seq], [l.MSLD_theta for l in seq],
                [l.thickness for l in seq], Q)   # top -> bottom
            while len(A) > 1:
                odd = A[-1:] if len(A) % 2 else None
                A = A[1::2] @ A[0:len(A) - 1:2]   # deeper on the left
                if odd is not None:
                    A = np.concatenate([A, odd])
            M = A[0]
        self.transfer_matrix = M
        return M


    @staticmethod
    def _surround_K(layer, Q2):
        """Wavevector operator of a semi-infinite medium, in the z basis.

        Q2 is the squared vacuum-referenced Q (may be complex).  The medium's
        in-plane magnetisation is off-diagonal in the z basis, so the
        wavevector is the 2x2 operator
            K = (k+ + k-)/2 * 1 + (k+ - k-)/2 * [[0, mu*], [mu, 0]],
            k+- = (i/2) sqrt(Q2 - 16 pi (nsld +- rho)),  mu = e^{i theta},
        whose eigenvectors (1, +-mu)/sqrt(2) are the spins along +-M.
        Returns K (nQ,2,2) and q+- = sqrt(...) (the Q inside the medium).
        """
        n = layer.NSLD_real + 1j*layer.NSLD_img
        qp = np.sqrt(Q2 - 16*np.pi*(n + layer.MSLD_rho) + 0j)
        qm = np.sqrt(Q2 - 16*np.pi*(n - layer.MSLD_rho) + 0j)
        mu = np.exp(2j*np.pi*layer.MSLD_theta)
        a, b = 0.25j*(qp + qm), 0.25j*(qp - qm)
        K = np.empty(np.shape(Q2) + (2, 2), dtype=np.complex128)
        K[:, 0, 0] = K[:, 1, 1] = a
        K[:, 0, 1] = b*np.conj(mu)
        K[:, 1, 0] = b*mu
        return K, qp, qm

    def lab_axis(self):
        """Angle (rad) of the lab quantisation axis in the sample plane.

        A magnetic fronting fixes it: only spins along its M propagate as
        stationary states there, so alpha is overridden by its MSLD_theta.
        """
        f = self.fronting
        return 2*np.pi*f.MSLD_theta if f.MSLD_rho != 0 else self.alpha

    def _reflect(self, Q2):
        """Reflection matrices for squared vacuum Q.

        Returns r (z basis) and rl (lab basis), both [:, out, in], plus the
        fronting q+- (lab eigenchannels).
        """
        M = self.build_transfer_matrix(np.sqrt(Q2 + 0j))
        A, B = M[:, :2, :2], M[:, :2, 2:]
        C, D = M[:, 2:, :2], M[:, 2:, 2:]
        Kf, qf_p, qf_m = self._surround_K(self.fronting, Q2)
        Kb, _, _ = self._surround_K(self.backing, Q2)

        # psi_f = I + r, psi_f' = Kf (I - r);  back: psi_b' = Kb psi_b
        #   (C - Kb A)(I + r) + (D - Kb B) Kf (I - r) = 0
        X = C - Kb @ A
        Y = (D - Kb @ B) @ Kf
        r = -np.linalg.solve(X - Y, X + Y)

        # lab basis: columns (1, +-e^{i axis})/sqrt(2)
        e = np.exp(1j*self.lab_axis())
        U = np.array([[1, 1], [e, -e]]) / np.sqrt(2)
        Ui = np.array([[1, np.conj(e)], [1, -np.conj(e)]]) / np.sqrt(2)
        return r, Ui @ r @ U, qf_p, qf_m

    def calc_reflectance(self, Q):
        """Reflectivity + background, shape (nQ, 8): spin channels
        (uu, ud, du, dd) then lab channels (++, +-, -+, --).

        Channel 'ab' = incident a, reflected b.  Lab channels are along
        lab_axis() and flux-normalised, R_ab = |r_ab|^2 Re(q_b)/Re(q_a),
        which is what makes R+- = R-+ with a magnetic fronting.

        q_in_fronting False: Q is the vacuum-referenced 2 k0z, common to both
            spins.  Below a fronting critical edge that spin cannot propagate
            in the fronting; its spin-flip channel is NaN there.
        q_in_fronting True: Q is the wavevector transfer inside the fronting,
            as measured through a substrate (Majkrzak Eq. 1.113, GEPORE's
            QP/QM): each incident lab spin s has its own vacuum
            Q_s^2 = Q^2 + 16 pi rho_f,s, so the matrices are built once per
            incident spin.  The spin channels then use the spin-averaged
            Q^2 + 16 pi rho_f,N.

        Spin channels are |r|^2 in the film-normal basis; they are only
        meaningful as reflectivities for a non-magnetic fronting.
        """
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        f = self.fronting
        nf = f.NSLD_real + 1j*f.NSLD_img

        if not self.q_in_fronting:
            r, rl, qf_p, qf_m = self._reflect(Q**2 + 0j)
            same = np.isclose(qf_p, qf_m, rtol=1e-12, atol=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                f_pm = np.where(same, 1.0,
                                np.where(qf_p.real > 0, qf_m.real/qf_p.real, np.nan))
                f_mp = np.where(same, 1.0,
                                np.where(qf_m.real > 0, qf_p.real/qf_m.real, np.nan))
            r_pp, r_pm = rl[:, 0, 0], rl[:, 1, 0]
            r_mp, r_mm = rl[:, 0, 1], rl[:, 1, 1]
        else:
            r, rl_p, qp_p, qp_m = self._reflect(Q**2 + 16*np.pi*nf)
            if f.MSLD_rho == 0:              # one vacuum Q serves both spins
                rl_m, qm_p, qm_m = rl_p, qp_p, qp_m
            else:
                _, rl_p, qp_p, qp_m = self._reflect(
                    Q**2 + 16*np.pi*(nf + f.MSLD_rho))
                _, rl_m, qm_p, qm_m = self._reflect(
                    Q**2 + 16*np.pi*(nf - f.MSLD_rho))
            # the incident channel's q in the fronting is Q itself; the
            # flipped one leaves at a Zeeman-shifted q (0 flux if evanescent)
            f_pm = qp_m.real / Q
            f_mp = qm_p.real / Q
            r_pp, r_pm = rl_p[:, 0, 0], rl_p[:, 1, 0]
            r_mp, r_mm = rl_m[:, 0, 1], rl_m[:, 1, 1]

        R_uu = np.abs(r[:, 0, 0])**2
        R_ud = np.abs(r[:, 1, 0])**2
        R_du = np.abs(r[:, 0, 1])**2
        R_dd = np.abs(r[:, 1, 1])**2
        R_pp = np.abs(r_pp)**2
        R_pm = np.abs(r_pm)**2 * f_pm
        R_mp = np.abs(r_mp)**2 * f_mp
        R_mm = np.abs(r_mm)**2

        return np.stack([R_uu, R_ud, R_du, R_dd, R_pp, R_pm, R_mp, R_mm],
                        axis=-1) + self.background
