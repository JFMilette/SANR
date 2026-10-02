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

ROUGHNESS SCHEME -- Stack.roughness_scheme chooses between the additive erf /
tanh profile above ('rms', default) and Licorne's scheme ('licorne', see
model.roughness): each interface j has its own window [Z_j - l_a, Z_j + l_b],
l = min(gamma, T/2), gamma = 2.47584 sigma_L (tanh) or 2.0913 sigma_L (erf),
cut into roughness_sublayer equal slabs sampled at their midpoints; the
thinner, clipped side is renormalised to keep its amount of material
(Stack.licorne_renorm 'manual' or 'none'), and the rest of each layer is one
core slab at its nominal value.  Licorne's jumps are kept, in the slabs and
in Stack.profile alike: 1.5 % of the step at every unclipped window edge, and
at the midpoint of a thin layer whose two interfaces renormalise it
differently.  Stack.licorne_renorm 'licorne' takes the window integral J of
the renormalisation with Licorne's 50-point rule instead of exactly
(model.roughness RENORMALISATION).  Stack.licorne_outer says how thick the
fronting and backing are for the windows: 'infinite' (default), so their
windows are never clipped; or 'licorne', Licorne's pseudo-layers of 1.5 t_1
above and 1.5 t_N below (expandrough.m: M.Layers(1).thickness*1.5 for the
top row, M.Layers(N).thickness*1.5 for the substrate row), so the outer
windows are clipped at 0.75 t_1 / 0.75 t_N.  The renormalisation then only
ever changes layer 1 at the top interface and layer N at the bottom one (the
thinner side), never the fronting or backing.  Only the window geometry
changes: the cores and the semi-infinite media keep their own thickness and
values.  Licorne's top pseudo-layer is vacuum and its substrate row has
rho = 0, i.e. SANR's vacuum fronting and non-magnetic backing; for any other
fronting or backing there is no Licorne equivalent and SANR's own values
are used.  In this scheme a layer's roughness_sigma is Licorne's sigma_L,
roughness_model its roughness_fun ('tanh', 'erf' or 'none') and
roughness_sublayer its roughness_nbound, all for the interface at the TOP of
the layer; Stack.tail is unused.  model.licorne_io reads a Licorne export.

MAGNETISATION -- Stack.magnetic_smearing chooses how M crosses an interface
(default 'vector' for the 'rms' scheme, 'step' for 'licorne'):
  'step': rho is smeared as a scalar like the NSLD (renormalised with it in
      the 'licorne' scheme) and each slab takes the whole angle (theta, phi)
      of one layer: in a Licorne window the layer above for x <= 0 (the tie
      at x = 0, met by the middle slab of an odd count in a symmetric
      window, goes to the layer above, as in Licorne 1.2.7 exports) and the
      layer below for x > 0; in the 'rms' scheme the layer of largest
      occupancy f_{k-1} - f_k at the slab centre.  The angle jumps; |M| does
      not dip.  With Stack.step_fallback (default) a layer with rho == 0 has
      no angle and the other layer's is used (both non-magnetic: 0, 0).
      step_fallback False reproduces Licorne, where only the fronting and
      backing (which have no angle parameter there) fall back and an
      interior non-magnetic layer's stored angle shows wherever its window
      side carries rho > 0.  Core slabs and the semi-infinite media keep
      their own angle.
  'vector': the Cartesian components of M are smeared like any
      other quantity.  This is the lateral average of a
      rough interface (the potential is linear in M), so a non-magnetic
      neighbour only fades |M| without turning it, its theta has no effect,
      and layers at an angle D pass through |M| = cos(D/2) (0 if antiparallel).
  'angle': rho and theta are smeared independently, so the angle turns
      smoothly from one layer's value to the next over the roughness width,
      as in a magnetic twist (antiparallel layers rotate through 90 deg rather
      than passing through |M| = 0).  theta is interpolated between the values
      as entered: 0 -> 350 deg turns the long way; enter -10 deg for the short
      way.  A non-magnetic layer's theta then matters: keep it equal to its
      magnetic neighbour's unless a rotation is intended.  phi is
      interpolated the same way (not folded).

ANGLES -- axes of Majkrzak Fig. 1.14: sample z || Q (film normal), x and y in
the film plane.  MSLD_theta is theta_M, the in-plane angle of M from sample x
towards y (GEPORE THE(J); also the theta of Eq. 1.128 / Fig. 1.19), stored in
turns.  MSLD_phi is the elevation of M out of the film plane, in turns
(0 = in plane, +1/4 = along +z), so M = rho (cos phi cos theta,
cos phi sin theta, sin phi).  MSLD_rho is |M|.

HALPERIN -- the neutron sees B, and B_z is continuous across the film
surface: the z component of M is cancelled by the demagnetising field and
has no effect on the reflectivity.  Every medium (slabs, fronting, backing)
therefore enters the transfer matrix and the boundary conditions only through
its in-plane projection rho cos(phi) at angle theta (Stack.inplane_rho).
M_z is kept in the model and in the profiles but is invisible to the
neutrons, so rho and phi of one layer are degenerate in a fit: only
rho cos(phi) is measured.

TRANSFER MATRIX -- Majkrzak Eq. 1.128, i.e. Table 1.2 with Np_z = 0
(HALPERIN).  Quantisation along the film normal.

RESOLUTION -- with Stack.resolution on, every reflectivity is averaged over a
Gaussian in Q of standard deviation

    sigma_Q = Q * sqrt( (dtheta/theta)^2 + (dlambda/lambda)^2 ),

where resolution['mode'] says which of theta, lambda is fixed
(RESOLUTION_MODES, one sigma function per mode):
  'mono' -- fixed wavelength, angle scan (Licorne resolution.m, MONO mode):
            theta = asin(Q lambda / 4 pi); 'wavelength' (A), 'dlambda_rel'
            (dlambda / lambda), 'dtheta' (rad).
  'tof'  -- time of flight at a few fixed angles, each covering a Q band:
            lambda = 4 pi sin(theta) / Q; 'tof_dlambda' (A, absolute) and
            'tof_angles' = [{'theta', 'dtheta' (rad), 'qmax' (A^-1)}, ...]
            in increasing qmax.  Angle i is used for qmax_{i-1} <= Q <
            qmax_i; the last angle's qmax is ignored (open-ended).
The average is a fixed quadrature
over +-RES_SPAN sigma.  The ideal reflectivity is computed once on a Q grid
of local step sigma / RES_GRID_STEP, refined around the critical edges
(Stack.critical_edges, the square-root kinks of total reflection), and
linearly interpolated at the quadrature nodes: several times cheaper than
evaluating every node.  Against a uniform grid 16x finer, the error is a
few 0.1 % of R at most, and smaller than that of a uniform grid 4x finer at
the critical edge (the worst place for interpolation).  A magnetic fronting
with q_in_fronting False uses a uniform grid of step sigma /
RES_GRID_STEP_UNIFORM instead (see RES_GRID_STEP_UNIFORM).
A channel that is undefined (NaN) at some nodes is averaged over the others.

POLARISATION -- a measured quantity is a pair (P0, P): incident polarisation
P0 and analyser P, 3-vectors in the sample frame of ANGLES whose length is
the efficiency, P = None for no analyser.  Stack.reflectivities(Q, pairs)
returns one column per pair (Ruehm, Toperverg & Dosch, PRB 60, 16073,
Eq. 2): R = Tr{rho r rho0 r^+}, rho0 = (1 + P0.sigma)/2,
rho = (1 + P.sigma)/2, or rho = 1 for P=None (all reflected spins counted).
The background is added once per pair, after the scale factor: R+ = (n, None)
is scale (R++ + R+-) + background.  model.polarisation names the usual pairs.

With a non-magnetic fronting this holds for any backing, and all pairs share
one transfer-matrix evaluation.  With a magnetic fronting the two eigenspins
(+-M_f) have different k_z there, so a component of P0 or P transverse to M_f
precesses with depth and is averaged out over the macroscopic path through
the fronting: only P0.m_f and P.m_f survive, and R is the flux-normalised
lab channels along M_f (Stack._lab_channels) weighted by (1 +- P0.m_f)/2 and
(1 +- P.m_f)/2 (one warning per call if a pair has a transverse part).  With
the resolution on, the weights are applied AFTER averaging each lab channel:
in vacuum-referenced Q a flipped channel is NaN below a fronting edge and is
averaged over the nodes where it is defined.

SURROUND -- a magnetic fronting/backing is off-diagonal in the film-normal
basis, so its wavevector is a 2x2 operator (Stack._surround_K), not one
scalar per spin.
"""

import copy
import warnings

import numpy as np
from scipy.special import erf

from model import roughness as rough


# Layer attributes a fit may vary.  Layer.fit[attr] = {'vary', 'min', 'max'},
# bounds in the same units as the attribute (Angstrom, A^-2, turns).
FIT_PARAMS = ('thickness', 'NSLD_real', 'NSLD_img', 'MSLD_rho', 'MSLD_theta',
              'MSLD_phi', 'roughness_sigma')
# Stack attributes a fit may vary, in Stack.fit the same way; their
# free_parameters() entries have layer index None.
STACK_FIT_PARAMS = ('scale', 'background')


# Default fit bounds, the same for every layer: attr -> (min, max).
DEFAULT_BOUNDS = {
    'thickness':       (0.0, 500.0),       # Angstrom
    'NSLD_real':       (-2e-6, 10e-6),     # A^-2
    'NSLD_img':        (-1e-6, 0.0),       # A^-2
    'MSLD_rho':        (0.0, 5e-6),        # A^-2
    'MSLD_theta':      (0.0, 1.0),         # turns
    'MSLD_phi':        (-0.25, 0.25),      # turns, out of plane
    'roughness_sigma': (0.0, 10.0),        # Angstrom
    'scale':           (0.5, 1.5),
    'background':      (0.0, 1e-5),
}


def default_fit(attr):
    """Fixed, with the layer-independent DEFAULT_BOUNDS."""
    lo, hi = DEFAULT_BOUNDS[attr]
    return {'vary': False, 'min': float(lo), 'max': float(hi)}


# Resolution quadrature: RES_NODES points evenly spread over +-RES_SPAN sigma,
# interpolated from a grid of step sigma_min / RES_GRID_STEP (at most
# RES_GRID_MAX points), refined around every critical edge (Stack.
# critical_edges) where R has a square-root kink: RES_EDGE_POINTS more points
# on each side, geometrically closer towards the edge, over RES_EDGE_WIDTH
# coarse steps.  Away from the edges the coarse step is within a few % of an
# error bar of a 16x finer grid; at the edges the refinement makes it better
# than a uniform grid 4x finer (see RESOLUTION).  A magnetic fronting with
# vacuum-referenced Q (q_in_fronting False) keeps the uniform fine grid of
# step sigma / RES_GRID_STEP_UNIFORM and no edge refinement: its flipped
# channels start at the fronting's edges (NaN below, a 1/sqrt flux factor
# above), a boundary only that grid handles well.
RES_NODES = 101
RES_SPAN = 3.5
RES_GRID_STEP = 8
RES_GRID_STEP_UNIFORM = 32
RES_GRID_MAX = 50000
RES_EDGE_POINTS = 40
RES_EDGE_WIDTH = 8
# Q points per block of the transfer-matrix product (see
# Stack.build_transfer_matrix); set from a benchmark
TM_BLOCK = 256


def wrap_turns(t):
    """Angle in turns folded into [0, 1); rounding first keeps -1e-17 at 0
    instead of 1 (i.e. 0 deg, not 360)."""
    return np.round(np.asarray(t, dtype=float), 9) % 1.0


# |M| below M_TINY of the largest layer rho has no meaningful direction: its
# angle is round-off noise, or the direction of a leftover interface tail far
# from any magnetic layer (e.g. a rough interface's tail outliving a sharper
# one below it points along M_above - M_below).  See Stack.angle_defined.
M_TINY = 1e-3

ROUGHNESS_SCHEMES = ('rms', 'licorne')
# Stack.licorne_outer: thickness of the fronting / backing for the Licorne
# windows (see ROUGHNESS SCHEME)
LICORNE_OUTER = ('infinite', 'licorne')
SMEARING_MODES = ('vector', 'angle', 'step')
# magnetic_smearing when none is given, per roughness scheme
DEFAULT_SMEARING = {'rms': 'vector', 'licorne': 'step'}
# 'step' in a Licorne window: x <= STEP_TIE * window width takes the layer
# above, so round-off in a depth never moves the tie at x = 0
STEP_TIE = 1e-9


def cos_turns(t):
    """cos(2 pi t) with the quarter turns exactly 0 (cos(pi/2) = 6e-17 would
    otherwise leave a 'magnetic' residue on a layer magnetised along z)."""
    c = np.cos(2*np.pi*np.asarray(t, dtype=float))
    return np.where(np.abs(c) < 1e-12, 0.0, c)


def sin_turns(t):
    """sin(2 pi t) with the half turns exactly 0."""
    s = np.sin(2*np.pi*np.asarray(t, dtype=float))
    return np.where(np.abs(s) < 1e-12, 0.0, s)


_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
                  dtype=complex)


def _density(P):
    """(1 + P.sigma)/2 for a polarisation 3-vector P (|P| <= 1)."""
    return 0.5*(np.eye(2) + np.tensordot(np.asarray(P, dtype=float), _PAULI, 1))


def _check_P(P, name):
    P = np.asarray(P, dtype=float)
    if P.shape != (3,):
        raise ValueError('%s must be a 3-vector (x, y, z) in the sample frame'
                         % name)
    if np.linalg.norm(P) > 1 + 1e-9:
        raise ValueError('|%s| = %.4g > 1' % (name, np.linalg.norm(P)))
    return P


def _check_pairs(pairs):
    out = [(_check_P(P0, 'P0'), None if P is None else _check_P(P, 'P'))
           for P0, P in pairs]
    if not out:
        raise ValueError('no (P0, P) pair given')
    return out


def _weigh(Rab, w):
    """Rab (nQ, 4) lab channels, w (4, k) weights -> (nQ, k).  A zero weight
    drops an undefined (NaN) channel instead of spreading it."""
    Rw = Rab[:, :, None] * w[None, :, :]
    return np.where(w[None, :, :] > 0, Rw, 0.0).sum(1)



# default resolution: mono is Licorne's resolution.m (lambda = 5 A,
# dlambda = 0.01 A, dtheta = 0.7 mrad), tof its three-angle TOF setup
DEFAULT_RESOLUTION = {
    'enabled': False, 'mode': 'mono',
    'wavelength': 5.0, 'dlambda_rel': 0.002, 'dtheta': 7e-4,
    'tof_dlambda': 0.005,
    'tof_angles': [{'theta': 0.006, 'dtheta': 3e-4, 'qmax': 0.04},
                   {'theta': 0.010, 'dtheta': 5e-4, 'qmax': 0.12},
                   {'theta': 0.017, 'dtheta': 5e-4, 'qmax': None}],
}


def _sigma_mono(Q, res):
    """Fixed wavelength (see RESOLUTION); Q > 0."""
    s = np.clip(Q * res['wavelength'] / (4*np.pi), 1e-12, 1.0)
    return Q * np.hypot(res['dtheta'] / np.arcsin(s), res['dlambda_rel'])


def _sigma_tof(Q, res):
    """Fixed angles, one per Q band (see RESOLUTION); Q > 0."""
    ang = res['tof_angles']
    if not ang:
        raise ValueError('TOF resolution needs at least one angle')
    theta = np.array([a['theta'] for a in ang], dtype=float)
    dtheta = np.array([a['dtheta'] for a in ang], dtype=float)
    edges = np.array([a['qmax'] for a in ang[:-1]], dtype=float)
    if np.any(theta <= 0):
        raise ValueError('TOF angles must be > 0')
    if np.any(np.diff(edges) <= 0):
        raise ValueError('TOF Q limits must increase')
    k = np.searchsorted(edges, Q, side='right')
    lam = 4*np.pi*np.sin(theta[k]) / Q
    return Q * np.hypot(dtheta[k] / theta[k], res['tof_dlambda'] / lam)


# resolution['mode'] -> sigma_Q(Q > 0, resolution dict)
RESOLUTION_MODES = {'mono': _sigma_mono, 'tof': _sigma_tof}


def _res_grid(Qn, sigma, bins=2048, edges_q=(), step=None):
    """Q grid on which the ideal R is computed for the quadrature nodes Qn
    (nQ, RES_NODES) of widths sigma (nQ,): the local step is sigma /
    RES_GRID_STEP of the finest Gaussian whose nodes fall there, so a
    resolution that is sharp in one Q band (TOF at low Q) only refines that
    band.  At most RES_GRID_MAX points, plus the refinement around each
    critical edge in `edges_q` (see RES_EDGE_POINTS).  `step` replaces
    RES_GRID_STEP."""
    step = RES_GRID_STEP if step is None else step
    lo, hi = Qn.min(), Qn.max()
    edges = np.linspace(lo, hi, bins + 1)
    h = np.full(bins, np.inf)
    ok = sigma > 0
    idx = np.minimum(((Qn[ok] - lo) / (hi - lo) * bins).astype(int), bins - 1)
    np.minimum.at(h, idx.ravel(),
                  np.repeat(sigma[ok] / step, Qn.shape[1]))
    # a bin between two nodes of a wide Gaussian takes its neighbours' step
    i = np.arange(bins)
    fin = np.isfinite(h)
    left = np.maximum.accumulate(np.where(fin, i, 0))
    right = np.minimum.accumulate(np.where(fin, i, bins - 1)[::-1])[::-1]
    h = np.where(fin, h, np.minimum(h[left], h[right]))
    h = np.maximum(h, (hi - lo) / (RES_GRID_MAX - 1))
    # points evenly spaced in u = integral dq / h(q)
    u = np.concatenate([[0.0], np.cumsum(np.diff(edges) / h)])
    grid = np.interp(np.linspace(0, u[-1], int(np.ceil(u[-1])) + 1), u, edges)
    # the kink of R at a critical edge: the edge itself and points closing in
    # on it from both sides
    extra = []
    for qc in edges_q:
        if lo < qc < hi:
            k = min(int((qc - lo) / (hi - lo) * bins), bins - 1)
            width = RES_EDGE_WIDTH * h[k]
            d = width * np.geomspace(1e-4, 1.0, RES_EDGE_POINTS)
            extra.append(np.concatenate([[qc], qc - d, qc + d]))
    if extra:
        grid = np.unique(np.clip(np.concatenate([grid] + extra), lo, hi))
    return grid


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
                 roughness_sublayer: int = 5, fit: dict = None,
                 MSLD_phi: float = 0.0):
        self.name = name
        self.thickness = thickness              # Angstrom
        self.NSLD_real = NSLD_real              # A^-2
        self.NSLD_img = NSLD_img                # A^-2, negative = absorbing
        self.MSLD_rho = MSLD_rho                # A^-2, |M|
        self.MSLD_theta = MSLD_theta        # turns, theta_M of Fig. 1.14 (from x)
        self.MSLD_phi = MSLD_phi            # turns, elevation out of the plane
        self.roughness_sigma = roughness_sigma  # Angstrom, Gaussian sigma
        self.roughness_model = roughness_model  # 'erf', 'tanh' or 'none'
        self.roughness_sublayer = roughness_sublayer
        self.fit = {k: dict(v) for k, v in (fit or {}).items()}

    def fit_entry(self, attr):
        """{'vary', 'min', 'max'} of `attr`, created from default_fit if unset."""
        if attr not in self.fit:
            self.fit[attr] = default_fit(attr)
        return self.fit[attr]

    def __repr__(self):
        return ("Layer(%r, t=%.3g, nsld=%.4g%+.4gj, msld=%.4g @ %.4g, %+.4g "
                "turn, sigma=%.3g, N=%d)" % (self.name, self.thickness,
                                             self.NSLD_real, self.NSLD_img,
                                             self.MSLD_rho, self.MSLD_theta,
                                             self.MSLD_phi, self.roughness_sigma,
                                             self.roughness_sublayer))


# ------------------------------------------------------ profile helpers ----
def _step(u, sigma, model):
    u = np.asarray(u, dtype=float)
    if sigma <= 0.0 or model == 'none':
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

    def __init__(self, layers, tail: float = 3.0, roughness_scheme='rms',
                 magnetic_smearing=None):
        if len(layers) < 2:
            raise ValueError('need at least a fronting and a backing')
        if roughness_scheme not in ROUGHNESS_SCHEMES:
            raise ValueError('unknown roughness scheme %r' % (roughness_scheme,))
        self.layers = list(layers)
        self.tail = tail                 # unclipped window half-width = tail*sigma
        # 'rms' or 'licorne' (see ROUGHNESS SCHEME)
        self.roughness_scheme = roughness_scheme
        # 'manual', 'licorne' or 'none' (Licorne scheme only, see
        # model.roughness RENORMALISATION)
        self.licorne_renorm = 'manual'
        # 'infinite' or 'licorne' (Licorne scheme only, see ROUGHNESS SCHEME)
        self.licorne_outer = 'infinite'
        self.sublayers = None
        self.scale = 1.0                     # factor on every channel
        self.background = 0.0                # constant added to every channel
        self.fit = {}                        # STACK_FIT_PARAMS, as Layer.fit
        # True (default): Q is measured inside the fronting; False: vacuum Q
        self.q_in_fronting = True
        # instrumental resolution (see RESOLUTION): enabled, mode and the
        # parameters of each mode
        self.resolution = copy.deepcopy(DEFAULT_RESOLUTION)
        # 'vector', 'angle' or 'step' (see MAGNETISATION); None takes the
        # scheme's default
        self.magnetic_smearing = (DEFAULT_SMEARING[roughness_scheme]
                                  if magnetic_smearing is None
                                  else magnetic_smearing)
        self.step_fallback = True        # 'step': a rho = 0 layer has no angle

    # -- serialisation ------------------------------------------------------
    def to_dict(self):
        """Plain-JSON description of the sample (layers, tail, scale,
        background, resolution, smearing).  No polarisation: it belongs to
        each measurement."""
        return {'tail': self.tail,
                'scale': self.scale,
                'background': self.background,
                'fit': {k: dict(v) for k, v in self.fit.items()},
                'q_in_fronting': self.q_in_fronting,
                'resolution': copy.deepcopy(self.resolution),
                'magnetic_smearing': self.magnetic_smearing,
                'roughness_scheme': self.roughness_scheme,
                'licorne_renorm': self.licorne_renorm,
                'licorne_outer': self.licorne_outer,
                'step_fallback': self.step_fallback,
                'layers': [dict(vars(l), fit={k: dict(v)
                                              for k, v in l.fit.items()})
                           for l in self.layers]}

    @classmethod
    def from_dict(cls, d):
        """Inverse of to_dict.  Files written before the Licorne scheme
        existed load as 'rms'."""
        s = cls([Layer(**l) for l in d['layers']], tail=d['tail'],
                roughness_scheme=d.get('roughness_scheme', 'rms'),
                magnetic_smearing=d.get('magnetic_smearing'))
        s.background = d['background']
        s.scale = float(d.get('scale', 1.0))
        s.fit = {k: dict(v) for k, v in d.get('fit', {}).items()}
        s.q_in_fronting = bool(d['q_in_fronting'])
        s.resolution.update(copy.deepcopy(d['resolution']))
        s.licorne_renorm = d.get('licorne_renorm', 'manual')
        s.licorne_outer = d.get('licorne_outer', 'infinite')
        s.step_fallback = bool(d.get('step_fallback', True))
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

    def fit_entry(self, attr):
        """{'vary', 'min', 'max'} of the stack parameter `attr`, created
        from default_fit if unset."""
        if attr not in self.fit:
            self.fit[attr] = default_fit(attr)
        return self.fit[attr]

    def free_parameters(self):
        """[(layer index, attr, value, min, max)] of every varied parameter:
        the layers' in order, then the stack's own (layer index None)."""
        out = []
        for i, l in enumerate(self.layers):
            for attr in FIT_PARAMS:
                e = l.fit.get(attr)
                if e and e['vary'] and self.fit_allowed(i, attr):
                    out.append((i, attr, getattr(l, attr), e['min'], e['max']))
        for attr in STACK_FIT_PARAMS:
            e = self.fit.get(attr)
            if e and e['vary']:
                out.append((None, attr, getattr(self, attr), e['min'],
                            e['max']))
        return out

    def owner(self, i):
        """The object a free_parameters() entry with layer index i sets:
        layer i, or the stack itself for None."""
        return self if i is None else self.layers[i]

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
        """(lo, hi) per interface, clipped at half the adjacent thicknesses
        (the Licorne window [Z - l_a, Z + l_b] in that scheme)."""
        if self.roughness_scheme == 'licorne':
            return [(w['Z'] - w['la'], w['Z'] + w['lb'])
                    for w in self._licorne_interfaces()]
        t = self._thicknesses()
        Z = self._interfaces()
        out = []
        for j, Zj in enumerate(Z):
            lay = self.layers[j + 1]
            w = 0.0 if lay.roughness_model == 'none' else \
                self.tail * max(lay.roughness_sigma, 0.0)
            out.append((Zj - min(w, t[j] / 2.0), Zj + min(w, t[j + 1] / 2.0)))
        return out

    def _check_modes(self):
        if self.roughness_scheme not in ROUGHNESS_SCHEMES:
            raise ValueError('unknown roughness scheme %r'
                             % (self.roughness_scheme,))
        if self.magnetic_smearing not in SMEARING_MODES:
            raise ValueError('unknown magnetic smearing %r'
                             % (self.magnetic_smearing,))

    # -- slicing ------------------------------------------------------------
    def build_sublayers(self):
        """Replace the interior layers by slabs carrying the smeared profile.

        Stores and returns the list; fronting and backing are not included.
        """
        self._check_modes()
        if self.roughness_scheme == 'licorne':
            return self._build_licorne()
        Z = self._interfaces()
        n_if = len(Z)
        sig = [max(self.layers[j + 1].roughness_sigma, 0.0) for j in range(n_if)]
        mod = [self.layers[j + 1].roughness_model for j in range(n_if)]
        nsub = [max(int(self.layers[j + 1].roughness_sublayer), 1)
                for j in range(n_if)]

        nsld = np.array([complex(l.NSLD_real, l.NSLD_img) for l in self.layers])

        pts = []
        for (lo, hi), m in zip(self.windows(), nsub):
            pts.append(np.array([lo]) if hi - lo < 1e-9
                       else np.linspace(lo, hi, m + 1))
        e = np.unique(np.round(np.concatenate(pts), 6))
        e = e[np.concatenate([[True], np.diff(e) > 1e-4])]

        c = 0.5 * (e[:-1] + e[1:])
        v_n = _profile(c, Z, sig, mod, nsld)
        v_rho, v_theta, v_phi = self._magnetic_profile(c, Z, sig, mod)
        return self._set_sublayers(np.diff(e), v_n, v_rho, v_theta, v_phi)

    def _set_sublayers(self, th, v_n, v_rho, v_theta, v_phi):
        self.sublayers = [
            Layer(name='%s_%03d' % (self.layers[0].name or 'slab', i),
                  thickness=float(th[i]), NSLD_real=v_n[i].real,
                  NSLD_img=v_n[i].imag, MSLD_rho=float(v_rho[i]),
                  MSLD_theta=float(v_theta[i]), MSLD_phi=float(v_phi[i]),
                  roughness_sigma=0.0, roughness_sublayer=1)
            for i in range(len(th))]
        return self.sublayers

    def profile(self, z):
        """Exact continuous profile: (nsld complex, |M| = msld_rho, MSLD_theta,
        MSLD_phi), angles in turns.  The neutrons only see the in-plane part
        rho cos(phi) (HALPERIN).  In the Licorne scheme this is the truncated,
        renormalised profile the slabs sample, jumps included."""
        self._check_modes()
        if self.roughness_scheme == 'licorne':
            return self._licorne_profile(z)
        Z = self._interfaces()
        n_if = len(Z)
        sig = [max(self.layers[j + 1].roughness_sigma, 0.0) for j in range(n_if)]
        mod = [self.layers[j + 1].roughness_model for j in range(n_if)]
        nsld = np.array([complex(l.NSLD_real, l.NSLD_img) for l in self.layers])
        return (_profile(z, Z, sig, mod, nsld),) + \
            self._magnetic_profile(z, Z, sig, mod)

    def _magnetic_profile(self, z, Z, sig, mod):
        """(|M|, theta, phi in turns) at depths z, smeared as
        magnetic_smearing says (see MAGNETISATION)."""
        rho = np.array([l.MSLD_rho for l in self.layers])
        theta = np.array([l.MSLD_theta for l in self.layers])
        phi = np.array([l.MSLD_phi for l in self.layers])
        if self.magnetic_smearing == 'angle':
            # the interpolated angles are kept as is (the physics depends on
            # the path they take); only the reported theta is folded
            return (_profile(z, Z, sig, mod, rho),
                    wrap_turns(_profile(z, Z, sig, mod, theta)),
                    _profile(z, Z, sig, mod, phi))
        z = np.asarray(z, dtype=float)
        if self.magnetic_smearing == 'step':
            # occupancy of layer k: f_{k-1} - f_k (f_{-1} = 1, f_last = 0);
            # each depth takes the angle of the layer that occupies it most
            # among those that have an angle (argmax: ties go to the upper)
            f = np.stack([np.ones(z.shape)]
                         + [_step(z - Z[j], sig[j], mod[j])
                            for j in range(len(Z))] + [np.zeros(z.shape)])
            has = self._has_angle()
            occ = np.where(has.reshape((-1,) + (1,)*z.ndim), f[:-1] - f[1:],
                           -np.inf)
            k = np.argmax(occ.reshape(len(self.layers), -1), 0).reshape(z.shape)
            some = np.any(has)
            return (_profile(z, Z, sig, mod, rho),
                    wrap_turns(np.where(some, theta[k], 0.0)),
                    np.where(some, phi[k], 0.0))
        # in-plane part as one complex number, M_z separately; with phi = 0
        # everywhere mz is exactly 0 and this is the previous 2D smearing
        mp = _profile(z, Z, sig, mod,
                      rho * cos_turns(phi) * np.exp(2j*np.pi*theta))
        mz = _profile(z, Z, sig, mod, rho * sin_turns(phi))
        a = np.abs(mp)
        return (np.hypot(a, mz), wrap_turns(np.angle(mp) / (2*np.pi)),
                np.arctan2(mz, a) / (2*np.pi))

    def _has_angle(self):
        """'step': which layers have an angle to give (see MAGNETISATION).
        With step_fallback a layer needs rho != 0; without it only a
        non-magnetic fronting / backing has none (Licorne)."""
        n = len(self.layers)
        mag = np.array([l.MSLD_rho != 0 for l in self.layers])
        if self.step_fallback:
            return mag
        return mag | ((np.arange(n) > 0) & (np.arange(n) < n - 1))

    # -- Licorne scheme -------------------------------------------------------
    @staticmethod
    def _licorne_values(layer):
        """Quantities smeared through a Licorne window, all renormalised
        alike: Re NSLD, Im NSLD, rho, and the Cartesian M (for 'vector')."""
        r, th, ph = layer.MSLD_rho, layer.MSLD_theta, layer.MSLD_phi
        return np.array([layer.NSLD_real, layer.NSLD_img, r,
                         r * cos_turns(ph) * cos_turns(th),
                         r * cos_turns(ph) * sin_turns(th),
                         r * sin_turns(ph)], dtype=float)

    def _licorne_interfaces(self):
        """Per interface j (the top of layer j+1, which owns sigma_L, the
        function and N): its depth Z, window la / lb, renormalised values
        Fa / Fb (_licorne_values) and the indices a = j, b = j+1.  The
        fronting and backing take the thickness licorne_outer gives them."""
        if self.licorne_outer not in LICORNE_OUTER:
            raise ValueError('unknown licorne_outer %r' % (self.licorne_outer,))
        t = self._thicknesses()
        if self.licorne_outer == 'licorne' and len(t) > 2:
            t[0], t[-1] = 1.5 * t[1], 1.5 * t[-2]
        vals = [self._licorne_values(l) for l in self.layers]
        out = []
        for j, Zj in enumerate(self._interfaces()):
            own = self.layers[j + 1]
            sL = max(own.roughness_sigma, 0.0)
            kind = own.roughness_model
            Fa, Fb, la, lb, _ = rough.renormalised(
                vals[j], vals[j + 1], t[j], t[j + 1], sL, kind,
                self.licorne_renorm)
            out.append({'Z': float(Zj), 'la': la, 'lb': lb, 'sigma': sL,
                        'kind': kind,
                        'N': max(int(own.roughness_sublayer), 1),
                        'Fa': Fa, 'Fb': Fb, 'a': j, 'b': j + 1})
        return out

    def _licorne_window(self, w, x):
        """(nsld complex, rho, theta, phi) at offsets x from interface w,
        inside its window."""
        x = np.asarray(x, dtype=float)
        s = rough.profile_fraction(x, w['sigma'], w['kind'])
        v = w['Fa'][:, None] + (w['Fb'] - w['Fa'])[:, None] * s[None, :]
        nsld = v[0] + 1j * v[1]
        A, B = self.layers[w['a']], self.layers[w['b']]
        mode = self.magnetic_smearing
        if mode == 'vector':
            mp = v[3] + 1j * v[4]
            a = np.abs(mp)
            return (nsld, np.hypot(a, v[5]),
                    wrap_turns(np.angle(mp) / (2*np.pi)),
                    np.arctan2(v[5], a) / (2*np.pi))
        if mode == 'angle':
            # angles interpolated with the profile fraction, never
            # renormalised (an angle has no amount of material)
            return (nsld, v[2],
                    wrap_turns(A.MSLD_theta + (B.MSLD_theta - A.MSLD_theta) * s),
                    A.MSLD_phi + (B.MSLD_phi - A.MSLD_phi) * s)
        # 'step': the upper layer for x <= 0, the lower for x > 0, or the
        # other one when that layer has no angle
        has = self._has_angle()
        upper = x <= STEP_TIE * (w['la'] + w['lb'])
        first = np.where(upper, w['a'], w['b'])
        other = np.where(upper, w['b'], w['a'])
        k = np.where(has[first], first, other)
        theta = np.array([l.MSLD_theta for l in self.layers])[k]
        phi = np.array([l.MSLD_phi for l in self.layers])[k]
        none = ~has[k]
        return (nsld, v[2], wrap_turns(np.where(none, 0.0, theta)),
                np.where(none, 0.0, phi))

    def _licorne_regions(self, geo):
        """[(lo, hi, ('window', w) or ('layer', k))] from the top of the
        first window to the end of the last one: windows and the core
        slabs between them (a core of zero thickness is dropped)."""
        t = self._thicknesses()
        out = []
        for j, w in enumerate(geo):
            if w['la'] + w['lb'] > 0:
                out.append((w['Z'] - w['la'], w['Z'] + w['lb'], ('window', w)))
            if j + 1 < len(geo):
                core = t[j + 1] - w['lb'] - geo[j + 1]['la']
                if core > 0:
                    out.append((w['Z'] + w['lb'], geo[j + 1]['Z']
                                - geo[j + 1]['la'], ('layer', j + 1)))
        return out

    @staticmethod
    def _nominal(layer):
        return (complex(layer.NSLD_real, layer.NSLD_img), layer.MSLD_rho,
                float(wrap_turns(layer.MSLD_theta)), layer.MSLD_phi)

    def _build_licorne(self):
        """Licorne slabs (see ROUGHNESS SCHEME): N equal slabs per window at
        their midpoints, one core slab per layer at its nominal value."""
        geo = self._licorne_interfaces()
        th, cols = [], []
        for lo, hi, (what, w) in self._licorne_regions(geo):
            if what == 'layer':
                th.append(hi - lo)
                cols.append([np.atleast_1d(v) for v in
                             self._nominal(self.layers[w])])
                continue
            N = w['N']
            h = (w['la'] + w['lb']) / N
            x = -w['la'] + h * (np.arange(N) + 0.5)
            th.extend([h] * N)
            cols.append(list(self._licorne_window(w, x)))
        if not th:
            return self._set_sublayers([], [], [], [], [])
        v = [np.concatenate([c[i] for c in cols]) for i in range(4)]
        return self._set_sublayers(np.array(th), *v)

    def _licorne_profile(self, z):
        """profile(z) in the Licorne scheme: the semi-infinite media and the
        cores at their nominal values, the truncated, renormalised window
        function between (each region [lo, hi), so a jump sits at its
        edge)."""
        z = np.asarray(z, dtype=float)
        shape = z.shape
        z = z.ravel()
        geo = self._licorne_interfaces()
        nsld = np.empty(z.shape, dtype=complex)
        rho, theta, phi = (np.empty(z.shape) for _ in range(3))

        def put(mask, vals):
            for arr, v in zip((nsld, rho, theta, phi), vals):
                arr[mask] = v

        put(slice(None), self._nominal(self.fronting))
        put(z >= geo[-1]['Z'] + geo[-1]['lb'], self._nominal(self.backing))
        for lo, hi, (what, w) in self._licorne_regions(geo):
            m = (z >= lo) & (z < hi)
            if what == 'layer':
                put(m, self._nominal(self.layers[w]))
            elif np.any(m):
                put(m, self._licorne_window(w, z[m] - w['Z']))
        return tuple(a.reshape(shape) for a in (nsld, rho, theta, phi))

    @staticmethod
    def inplane_rho(layer):
        """rho cos(phi): the in-plane magnetic SLD the neutrons see."""
        return float(layer.MSLD_rho * cos_turns(layer.MSLD_phi))

    def angle_defined(self, rho):
        """True where the magnetisation |M| = rho (A^-2, array) is large
        enough for its direction to mean something: above M_TINY of the
        largest layer rho.  Below that the angle is round-off noise or the
        direction of a leftover interface tail (no layer's angle), and has no
        measurable effect; plots leave it blank."""
        ref = max((abs(l.MSLD_rho) for l in self.layers), default=0.0)
        return np.asarray(rho) > M_TINY * ref if ref > 0 else \
            np.zeros(np.shape(rho), dtype=bool)

    # -- transfer matrix ----------------------------------------------------
    @staticmethod
    def _slab_matrices(nsld, rho, theta, d, Q):
        """4x4 matrices of many homogeneous slabs at once.

        nsld (complex), rho (the IN-PLANE magnetic SLD, see HALPERIN),
        theta (turns), d:
        arrays of length nS.
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

    def build_transfer_matrix(self, Q):
        """Product matrix A = A_N ... A_1 (deepest leftmost) of the sliced
        sublayers.  Shape (nQ,4,4).  The fronting and backing are never part
        of the product; they enter through the boundary condition.

        All slab matrices are built in one vectorised call, then multiplied
        pairwise (tree reduction): ~log2(N) batched matmuls instead of N.
        """
        Q = np.atleast_1d(np.asarray(Q))     # may be complex: only Q**2 is used
        if self.sublayers is None:
            self.build_sublayers()
        seq = self.sublayers
        if not seq:
            M = np.broadcast_to(np.eye(4, dtype=np.complex128),
                                Q.shape + (4, 4)).copy()
        else:
            slabs = ([complex(l.NSLD_real, l.NSLD_img) for l in seq],
                     [self.inplane_rho(l) for l in seq],
                     [l.MSLD_theta for l in seq],
                     [l.thickness for l in seq])       # top -> bottom
            M = np.empty(Q.shape + (4, 4), dtype=np.complex128)
            # blocks of Q small enough for the slab matrices to stay in cache
            for j in range(0, len(Q), TM_BLOCK):
                A = self._slab_matrices(*slabs, Q[j:j + TM_BLOCK])
                while len(A) > 1:
                    odd = A[-1:] if len(A) % 2 else None
                    A = A[1::2] @ A[0:len(A) - 1:2]   # deeper on the left
                    if odd is not None:
                        A = np.concatenate([A, odd])
                M[j:j + TM_BLOCK] = A[0]
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
        rho = Stack.inplane_rho(layer)              # HALPERIN
        qp = np.sqrt(Q2 - 16*np.pi*(n + rho) + 0j)
        qm = np.sqrt(Q2 - 16*np.pi*(n - rho) + 0j)
        mu = np.exp(2j*np.pi*layer.MSLD_theta)
        a, b = 0.25j*(qp + qm), 0.25j*(qp - qm)
        K = np.empty(np.shape(Q2) + (2, 2), dtype=np.complex128)
        K[:, 0, 0] = K[:, 1, 1] = a
        K[:, 0, 1] = b*np.conj(mu)
        K[:, 1, 0] = b*mu
        return K, qp, qm

    def fronting_direction(self):
        """Unit in-plane vector of a magnetic fronting's M, None if the
        fronting is not magnetic.  Only spins along it propagate as
        stationary states in the fronting (see POLARISATION)."""
        f = self.fronting
        if self.inplane_rho(f) == 0:
            return None
        return np.array([float(cos_turns(f.MSLD_theta)),
                         float(sin_turns(f.MSLD_theta)), 0.0])

    def _reflect(self, Q2):
        """Reflection matrix r [:, out, in] in the film-normal (z) basis for
        the squared vacuum Q, plus the fronting's q+- (see _surround_K)."""
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
        return r, qf_p, qf_m

    def resolution_sigma(self, Q):
        """Standard deviation of the Q resolution at each Q (see RESOLUTION);
        zeros when the resolution is off."""
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        res = self.resolution
        if not res['enabled']:
            return np.zeros_like(Q)
        mode = res.get('mode', 'mono')
        if mode not in RESOLUTION_MODES:
            raise ValueError('unknown resolution mode %r' % (mode,))
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma = RESOLUTION_MODES[mode](np.abs(Q), res)
        return np.where(Q != 0, sigma, 0.0)

    # -- reflectivity -------------------------------------------------------
    def reflectivity(self, Q, P0, P=None):
        """reflectivities for the single pair (P0, P), shape (nQ,)."""
        return self.reflectivities(Q, [(P0, P)])[:, 0]

    def reflectivities(self, Q, pairs, warn=True):
        """scale x reflectivity + background of each (P0, P) in pairs (see
        POLARISATION), averaged over the Q resolution when it is enabled.
        Shape (nQ, len(pairs)).  warn=False silences the warning about a
        polarisation transverse to a magnetic fronting's M."""
        pairs = _check_pairs(pairs)
        m = self.fronting_direction()
        if m is None:
            R = self._resolve(Q, lambda q: self._trace(q, pairs))
        else:
            w = self._lab_weights(pairs, m, warn)
            R = _weigh(self._resolve(Q, self._lab_channels), w)
        return self.scale * R + self.background

    def critical_edges(self):
        """Q of the critical edges, where R has a square-root kink: total
        reflection by the backing for each spin pair of backing and fronting
        (in-plane magnetic SLD +-, HALPERIN).  A candidate that is not a real
        kink only costs a few grid points (see RES_EDGE_POINTS).  The edges
        of a magnetic fronting's own spins are left out on purpose: there a
        flipped channel starts (NaN below, a 1/sqrt flux factor above) and
        the coarse grid's handling of that boundary is the one to keep."""
        f, b = self.fronting, self.backing
        mf, mb = abs(self.inplane_rho(f)), abs(self.inplane_rho(b))
        v = [b.NSLD_real + sb * mb - f.NSLD_real - sf * mf
             for sb in (1, -1) for sf in (1, -1)]
        return sorted({float(np.sqrt(16 * np.pi * x)) for x in v if x > 0})

    def _resolve(self, Q, ideal):
        """Average ideal(Q) -> (nQ, k) over the Q resolution (see RESOLUTION)."""
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        sigma = self.resolution_sigma(Q)
        if not np.any(sigma > 0):
            return ideal(Q)
        x = np.linspace(-RES_SPAN, RES_SPAN, RES_NODES)
        w = np.exp(-0.5 * x**2)
        # reflectivity is even in Q; keep the nodes off Q = 0
        Qn = np.maximum(np.abs(Q[:, None] + x[None, :] * sigma[:, None]), 1e-6)
        if self.fronting_direction() is not None and not self.q_in_fronting:
            grid = _res_grid(Qn, sigma, step=RES_GRID_STEP_UNIFORM)
        else:
            grid = _res_grid(Qn, sigma, edges_q=self.critical_edges())
        Rg = ideal(grid)
        R = np.stack([np.interp(Qn, grid, Rg[:, c])
                      for c in range(Rg.shape[1])], -1)
        ok = np.isfinite(R)
        W = np.where(ok, w[None, :, None], 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(W.sum(1) > 0,
                            (np.where(ok, R, 0.0) * W).sum(1) / W.sum(1),
                            np.nan)

    def _trace(self, Q, pairs):
        """Non-magnetic fronting: Tr{rho r rho0 r^+} of every pair from ONE
        transfer-matrix evaluation, any backing; no background or resolution.
        Shape (nQ, len(pairs))."""
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        f = self.fronting
        Q2 = Q**2 + 0j
        if self.q_in_fronting:
            Q2 = Q2 + 16*np.pi*(f.NSLD_real + 1j*f.NSLD_img)
        r = self._reflect(Q2)[0]
        rho0 = np.stack([_density(P0) for P0, _ in pairs])
        rho = np.stack([np.eye(2) if P is None else _density(P)
                        for _, P in pairs])
        # = einsum('kij,qjl,klm,qim->qk', rho, r, rho0, r*), in a fixed
        # order so that a pair rounds the same in any batch
        A = rho[:, None] @ (r[None] @ rho0[:, None])
        return (A * r.conj()[None]).real.sum((-2, -1)).T

    def _lab_channels(self, Q):
        """Magnetic fronting: the channels ++, +-, -+, -- along its M, no
        background or resolution.  Shape (nQ, 4).

        Channel 'ab' = incident a, reflected b, flux-normalised:
        R_ab = |r_ab|^2 Re(q_b)/Re(q_a), which is what makes R+- = R-+.

        q_in_fronting False: Q is the vacuum-referenced 2 k0z, common to both
            spins.  Below a fronting critical edge that spin cannot propagate
            in the fronting; its spin-flip channel is NaN there.
        q_in_fronting True: Q is the wavevector transfer inside the fronting,
            as measured through a substrate (Majkrzak Eq. 1.113, GEPORE's
            QP/QM): each incident spin s has its own vacuum
            Q_s^2 = Q^2 + 16 pi rho_f,s, so the matrices are built once per
            incident spin.
        """
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        f = self.fronting
        nf = f.NSLD_real + 1j*f.NSLD_img
        rf = self.inplane_rho(f)                   # HALPERIN
        # spins along +-M: columns (1, +-e^{i theta})/sqrt(2)
        e = np.exp(1j*2*np.pi*f.MSLD_theta)
        U = np.array([[1, 1], [e, -e]]) / np.sqrt(2)
        Ui = np.array([[1, np.conj(e)], [1, -np.conj(e)]]) / np.sqrt(2)

        def lab(Q2):
            r, qp, qm = self._reflect(Q2)
            return Ui @ r @ U, qp, qm

        if not self.q_in_fronting:
            rl_p, qf_p, qf_m = lab(Q**2 + 0j)
            rl_m = rl_p
            with np.errstate(divide='ignore', invalid='ignore'):
                f_pm = np.where(qf_p.real > 0, qf_m.real/qf_p.real, np.nan)
                f_mp = np.where(qf_m.real > 0, qf_p.real/qf_m.real, np.nan)
        else:
            rl_p, _, qp_m = lab(Q**2 + 16*np.pi*(nf + rf))
            rl_m, qm_p, _ = lab(Q**2 + 16*np.pi*(nf - rf))
            # the incident channel's q in the fronting is Q itself; the
            # flipped one leaves at a Zeeman-shifted q (0 flux if evanescent)
            f_pm = qp_m.real / Q
            f_mp = qm_p.real / Q
        return np.stack([np.abs(rl_p[:, 0, 0])**2,
                         np.abs(rl_p[:, 1, 0])**2 * f_pm,
                         np.abs(rl_m[:, 0, 1])**2 * f_mp,
                         np.abs(rl_m[:, 1, 1])**2], axis=-1)

    @staticmethod
    def _lab_weights(pairs, m, warn=True):
        """(4, k) weights of the lab channels ++ +- -+ -- along m (a magnetic
        fronting's M) for each pair: (1 +- P0.m)/2 x (1 +- P.m)/2, or x 1 for
        no analyser.  Warns once if any pair has a part transverse to m."""
        w, lost = [], []
        for k, (P0, P) in enumerate(pairs):
            a0 = P0 @ m
            w_in = np.array([(1 + a0)/2, (1 - a0)/2])
            if P is None:
                w_out = np.ones(2)
            else:
                a1 = P @ m
                w_out = np.array([(1 + a1)/2, (1 - a1)/2])
            if any(v is not None and np.linalg.norm(v - (v @ m)*m) > 1e-9
                   for v in (P0, P)):
                lost.append(k)
            w.append(np.outer(w_in, w_out).ravel())
        if lost and warn:
            warnings.warn('magnetic fronting: the part of P0 / P transverse '
                          'to its magnetisation precesses and is averaged '
                          'out; only the projection on M_f is used (pair(s) '
                          '%s)' % ', '.join(map(str, lost)), stacklevel=3)
        w = np.array(w).T
        # round-off leaves ~1e-17 on a channel that is meant to be dropped
        return np.where(np.abs(w) < 1e-12, 0.0, w)
