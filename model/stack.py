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

MAGNETISATION -- rho and phi are smeared independently, like any other
quantity, so the angle turns smoothly from one layer's value to the next over
the roughness width (antiparallel layers rotate through 90 deg rather than
passing through |M| = 0).  phi is interpolated between the values as entered:
0 -> 350 deg turns the long way; enter -10 deg for the short way.

TRANSFER MATRIX -- Majkrzak Eq. 1.128, i.e. Table 1.2 with Np_z = 0 (Halperin:
the component of B along Q is cancelled by the demagnetising field, so only
the in-plane magnetisation is seen).  Quantisation along the film normal.
"""

import numpy as np
from scipy.special import erf



# ---------------------------------------------------------------- Layer ----
class Layer:
    """A single homogeneous medium.

    roughness_sigma / roughness_model / roughness_sublayer describe the
    interface at the TOP of this layer.
    """

    def __init__(self, name: str = "", thickness: float = 0.0,
                 NSLD_real: float = 0.0, NSLD_img: float = 0.0,
                 MSLD_rho: float = 0.0, MSLD_phi: float = 0.0,
                 roughness_sigma: float = 0.0, roughness_model: str = "tanh",
                 roughness_sublayer: int = 5):
        self.name = name
        self.thickness = thickness              # Angstrom
        self.NSLD_real = NSLD_real              # A^-2
        self.NSLD_img = NSLD_img                # A^-2, negative = absorbing
        self.MSLD_rho = MSLD_rho                # A^-2, in-plane magnitude
        self.MSLD_phi = MSLD_phi            # mul of 2pi radians, in-plane angle
        self.roughness_sigma = roughness_sigma  # Angstrom, Gaussian sigma
        self.roughness_model = roughness_model  # 'erf' or 'tanh'
        self.roughness_sublayer = roughness_sublayer

    def __repr__(self):
        return ("Layer(%r, t=%.3g, nsld=%.4g%+.4gj, msld=%.4g @ %.4g turn, "
                "sigma=%.3g, N=%d)" % (self.name, self.thickness, self.NSLD_real,
                                       self.NSLD_img, self.MSLD_rho,
                                       self.MSLD_phi, self.roughness_sigma,
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
        self.alpha = 0.0                     # polarisation angle for R_pp, R_pm, R_mp, R_mm

    # -- serialisation ------------------------------------------------------
    def to_dict(self):
        """Plain-JSON description of the sample (layers, tail, alpha)."""
        return {'tail': self.tail, 'alpha': self.alpha,
                'layers': [dict(vars(l)) for l in self.layers]}

    @classmethod
    def from_dict(cls, d):
        s = cls([Layer(**l) for l in d['layers']], tail=d.get('tail', 3.0))
        s.alpha = d.get('alpha', 0.0)
        return s

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
        phi = np.array([l.MSLD_phi for l in self.layers])

        pts = []
        for (lo, hi), m in zip(self.windows(), nsub):
            pts.append(np.array([lo]) if hi - lo < 1e-9
                       else np.linspace(lo, hi, m + 1))
        e = np.unique(np.round(np.concatenate(pts), 6))
        e = e[np.concatenate([[True], np.diff(e) > 1e-4])]

        c = 0.5 * (e[:-1] + e[1:])
        v_n = _profile(c, Z, sig, mod, nsld)
        v_rho = _profile(c, Z, sig, mod, rho)
        v_phi = _profile(c, Z, sig, mod, phi)

        self.sublayers = [
            Layer(name='%s_%03d' % (self.layers[0].name or 'slab', i),
                  thickness=th, NSLD_real=v_n[i].real, NSLD_img=v_n[i].imag,
                  MSLD_rho=float(v_rho[i]),
                  MSLD_phi=float(v_phi[i]),
                  roughness_sigma=0.0, roughness_sublayer=1)
            for i, th in enumerate(np.diff(e))]
        return self.sublayers

    def profile(self, z):
        """Exact continuous profile: (nsld complex, msld_rho, MSLD_phi in turns)."""
        Z = self._interfaces()
        n_if = len(Z)
        sig = [max(self.layers[j + 1].roughness_sigma, 0.0) for j in range(n_if)]
        mod = [self.layers[j + 1].roughness_model for j in range(n_if)]
        nsld = np.array([complex(l.NSLD_real, l.NSLD_img) for l in self.layers])
        rho = np.array([l.MSLD_rho for l in self.layers])
        phi = np.array([l.MSLD_phi for l in self.layers])
        return (_profile(z, Z, sig, mod, nsld), _profile(z, Z, sig, mod, rho),
                _profile(z, Z, sig, mod, phi))

    # -- transfer matrix ----------------------------------------------------
    @staticmethod
    def _slab_matrices(nsld, rho, phi, d, Q):
        """4x4 matrices of many homogeneous slabs at once.

        nsld (complex), rho, phi (turns), d: arrays of length nS.
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

        # mu_1 = e^{i phi}, mu_3 = -mu_1 (mu_2 = mu_1, mu_4 = mu_3), so
        # 1/(mu_3-mu_1) = -1/(2 mu_1) and every entry of Eq. 1.128 reduces to
        # a half-sum/half-difference of the two eigenchannels times 1, mu_1
        # or mu_1* (|mu_1| = 1).
        mu = np.exp(1j*2*np.pi*np.asarray(phi, dtype=float))[:, None]
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
                                    [layer.MSLD_rho], [layer.MSLD_phi],
                                    [layer.thickness], Q)[0]

    def build_transfer_matrix(self, Q, use_sublayers: bool = True):
        """Product matrix A = A_N ... A_1 (deepest leftmost).  Shape (nQ,4,4).

        Uses the sliced sublayers when available, else the nominal layers.
        The fronting and backing are never part of the product; they enter
        through the boundary condition.

        All slab matrices are built in one vectorised call, then multiplied
        pairwise (tree reduction): ~log2(N) batched matmuls instead of N.
        """
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
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
                [l.MSLD_rho for l in seq], [l.MSLD_phi for l in seq],
                [l.thickness for l in seq], Q)   # top -> bottom
            while len(A) > 1:
                odd = A[-1:] if len(A) % 2 else None
                A = A[1::2] @ A[0:len(A) - 1:2]   # deeper on the left
                if odd is not None:
                    A = np.concatenate([A, odd])
            M = A[0]
        self.transfer_matrix = M
        return M


    def calc_reflectance(self, Q):
        """Reflectivity for each spin channel, shape (nQ, 4)."""
        M = self.build_transfer_matrix(Q)
        # fronting and backing
        f = self.fronting
        b = self.backing

        Qf_plus = np.sqrt(Q**2 - 16*np.pi*(f.NSLD_real+f.NSLD_img*1j+f.MSLD_rho))
        Qf_minus = np.sqrt(Q**2 - 16*np.pi*(f.NSLD_real+f.NSLD_img*1j-f.MSLD_rho))
        Qb_plus = np.sqrt(Q**2 - 16*np.pi*(b.NSLD_real+b.NSLD_img*1j+b.MSLD_rho))
        Qb_minus = np.sqrt(Q**2 - 16*np.pi*(b.NSLD_real+b.NSLD_img*1j-b.MSLD_rho))

        kf_plus = 1j*Qf_plus/2
        kf_minus = 1j*Qf_minus/2
        kb_plus = 1j*Qb_plus/2
        kb_minus = 1j*Qb_minus/2



        r11=M[:,2,0]-kb_plus*M[:,0,0]
        r12=M[:,2,1]-kb_plus*M[:,0,1]
        r13=M[:,2,2]-kb_plus*M[:,0,2]
        r14=M[:,2,3]-kb_plus*M[:,0,3]

        r21=M[:,3,0]-kb_minus*M[:,1,0]
        r22=M[:,3,1]-kb_minus*M[:,1,1]
        r23=M[:,3,2]-kb_minus*M[:,1,2]
        r24=M[:,3,3]-kb_minus*M[:,1,3]

        p1 = r11-kf_plus*r13
        p2 = r21-kf_plus*r23

        q1=r12-kf_minus*r14
        q2=r22-kf_minus*r24

        S1_plus= r11+kf_plus*r13
        S2_plus= r21+kf_plus*r23

        S1_minus= r12+kf_minus*r14
        S2_minus= r22+kf_minus*r24

        D = p1*q2 - p2*q1

        r_uu = (S2_plus*q1 - S1_plus*q2)/D
        r_ud = (S1_plus*p2 - S2_plus*p1)/D

        r_du = (S2_minus*q1 - S1_minus*q2)/D
        r_dd = (S1_minus*p2 - S2_minus*p1)/D

        R_uu = np.abs(r_uu)**2
        R_ud = np.abs(r_ud)**2
        R_du = np.abs(r_du)**2
        R_dd = np.abs(r_dd)**2


        
        e, ec = np.exp(1j*self.alpha), np.exp(-1j*self.alpha)
        r_pp = 0.5*(r_uu + r_dd + e*r_du + ec*r_ud)
        r_mm = 0.5*(r_uu + r_dd - e*r_du - ec*r_ud)
        r_pm = 0.5*(r_uu - r_dd + e*r_du - ec*r_ud)
        r_mp = 0.5*(r_uu - r_dd - e*r_du + ec*r_ud)

        R_pp = np.abs(r_pp)**2
        R_mm = np.abs(r_mm)**2
        R_pm = np.abs(r_pm)**2
        R_mp = np.abs(r_mp)**2




        return np.stack([R_uu, R_ud, R_du, R_dd,R_pp,R_pm,R_mp,R_mm], axis=-1)









