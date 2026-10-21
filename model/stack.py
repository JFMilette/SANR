"""
Layer / Stack model for spin-dependent neutron reflectometry.

PROFILE -- interface steps.  For any quantity v,

    v(z) = v_0 + SUM_j ( v_{j+1} - v_j ) * F_j(z)

built from the step of each interface j at depth Z_j,

    f_j(u) = [1 + erf( u/(sigma_j*sqrt(2)) )]/2          'erf'
           = [1 + tanh( u*2/(sigma_j*sqrt(2*pi)) )]/2    'tanh', slope-matched

combined as Stack.interface_order says (INTERFACE_ORDERS; 'rms' scheme only):
  'ordered' (a new stack's default): F_j(z) = max_{i >= j} f_i(z - Z_i), the
      running max from the backing up.  The interfaces keep their order in
      the growth direction (the backing is the substrate, grown on first):
      the height of interface j is h'_j = min(h_j, h'_{j+1}), comonotone
      fluctuations, so where it would dip below the interface under it, it
      sits on it.  The occupancy of layer k, F_{k-1} - F_k (F_{-1} = 1,
      F_last = 0), is >= 0 and the occupancies sum to 1: every quantity is a
      weighted average of the layer values.  Where no interfaces cross
      (f_j >= f_{j+1} everywhere, e.g. equal sigmas and models) F_j = f_j and
      the profile is bit-identical to 'additive'.  Where they cross, the
      profile stays continuous with a slope kink at the crossing (for erf
      at z* = (sigma_{j+1} Z_j - sigma_j Z_{j+1}) / (sigma_{j+1} - sigma_j)),
      and a layer's integrated amount is no longer exactly t v: it gains
      the negative area 'additive' would give it.
  'independent': F_j(z) = 1 - PROD_{i >= j} (1 - f_i(z - Z_i)).  Each
      interface's height h_i fluctuates independently of the others, and
      interface j again sits at h'_j = min_{i >= j} h_i, so the occupancy
      of layer k, f_{k-1} PROD_{i >= k} (1 - f_i), is >= 0 and the
      occupancies sum to 1.  The profile is smooth everywhere (no kink),
      but unlike 'ordered' it differs from 'additive' even where nothing
      crosses: a layer only a few sigma thick gains material (equal
      sigmas: +2.5 % erf / +4.3 % tanh for t = 2 sigma, +20 / +25 % for
      t = sigma).
  'additive' (files saved before the setting existed): F_j = f_j, the
      interfaces superpose and every layer keeps its integrated amount.
      Where a wider interface reaches past a sharper neighbour the
      occupancy f_{k-1} - f_k goes negative: the NSLD or rho overshoots
      (rho < 0 in 'step' and 'angle' smearing), and in 'vector' smearing M
      points backwards (theta flips half a turn, |M| has a cusp at 0).
      Stack.interface_crossing() says where and how far.
In both, sigma may exceed a layer thickness: the layer then smoothly fails
to reach its nominal value.  No truncation, no renormalisation.

OWNERSHIP -- layer k's roughness_sigma and roughness_sublayer both describe the
interface at the TOP of layer k.  The fronting has no interface above it, so
its values are unused; the backing's describe the last interface.

MAGNETIC ROUGHNESS -- layer k's MSLD_sigma, when not None, is the width of
the magnetic profile (rho, theta, phi, however magnetic_smearing smears
them) at the interface at the top of layer k, the NSLD keeping
roughness_sigma; None (the default) uses roughness_sigma for both.  Same
function (roughness_model); in the 'rms' scheme the same interface_order.
In the Licorne scheme the magnetic values get a Licorne window of their own
(Stack._licorne_interfaces 'mag'): truncated at gamma(MSLD_sigma), clipped
at the half thicknesses and renormalised like the NSLD's.  Inside the wider
of the two windows each quantity follows its own window and keeps its
layer's core value outside it; the window is cut into both windows' N equal
slabs together, so every jump sits on a slab edge.  Licorne itself has no
magnetic roughness: with any MSLD_sigma set the result is no longer a
Licorne model.

SLICING -- interface j owns the window

    [ Z_j - min(tail*sigma_j, T_above/2) ,  Z_j + min(tail*sigma_j, T_below/2) ]

(sigma_j the larger of the interface's roughness_sigma and MSLD_sigma)

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
Stack.set_licorne_exact() makes every Licorne choice at once (the scheme,
'step' without step_fallback, licorne_renorm and licorne_outer 'licorne',
Licorne's resolution convolution: LICORNE_EXACT); the importer does so, and
set_licorne_exact(False) restores the settings from before it was turned on,
or the physical defaults when the stack was never anything but exact.

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

where resolution['mode'] says where sigma_Q comes from (RESOLUTION_MODES,
one sigma function per mode):
  'tof'  -- time of flight at a few fixed angles, each covering a Q band:
            lambda = 4 pi sin(theta) / Q; 'tof_dlambda' (A, absolute) and
            'tof_angles' = [{'theta', 'dtheta' (rad), 'qmax' (A^-1)}, ...]
            in increasing qmax.  Angle i is used for qmax_{i-1} <= Q <
            qmax_i; the last angle's qmax is ignored (open-ended).  With
            two or more limits, Q equal to the last one stays with the
            angle below it, as in Licorne's TOF template
            (QP2=(Q>=Q1)&(Q<=Q2)).
  'data' -- the measured dQ: 'data_q', 'data_dq' (A^-1, already a
            standard deviation) are a table of sigma_Q against Q, linearly
            interpolated (end values outside it), taken from a dataset
            ('data_set' its name, 'data_overlap' how the table treats two
            overlapping stitched angles: 'mean', 'first' or 'last', see
            tabs.resolution.dq_table).  It gives sigma_Q at any Q (the Simulation tab, model
            curves); a fit uses each data point's own dQ instead
            (Stack.data_sigma, fit.problem RESOLUTION), so stitched angles
            whose dQ do not join up are each taken as measured.
resolution['scheme'] says how the average is taken:
  'quadrature' (default) -- the Gaussian average below, independent of the
            Q points asked for.
  'licorne' -- Licorne's own convolution (model.licorne_resolution,
            reflection_m.m resolut, mode resolution['licorne_fun'] 1, 2 or
            3): a sum over the Q points of the call themselves, within
            +-3 sigma, unnormalised, with constant tails past the ends.
            The result therefore depends on the Q grid it is evaluated on,
            which is Licorne's behaviour: compute on the data Q to compare
            with Licorne.  Q must have no duplicates (any order); a magnetic
            fronting is refused (Licorne has none).  The weight matrix is
            cached per (Q, sigma, mode).
The quadrature average is a fixed quadrature
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
is scale (R++ + R+-) + background (times a fixed per-pair norm before the
background if given).  model.polarisation names the usual pairs.

Q SHIFT -- Stack.q_shift (A^-1, fittable like the scale and background)
moves the model along Q: the reflectivity at a measured Q is the sample's
at Q - q_shift (positive: the curve moves to higher Q), e.g. for a small
misalignment of the Q axis.  The resolution keeps the instrument's sigma_Q
at the measured Q; R is even in Q, so a point shifted below zero takes
the value at |Q - q_shift|.  Licorne's convolution weights depend only on
the spacing of the points and their sigma, which a shift leaves alone: they
stay those of the measured Q (its cached kernel), applied to the shifted
reflectivity, so fitting the shift costs nothing extra there.

THICKNESS SPREAD -- Stack.thickness_spread = sigma_t >= 0, a fraction (0.07
= 7 %).  The film's thickness varies across the illuminated area over
lateral distances much larger than the neutron's transverse coherence
length, so each region reflects independently and the measured specular
reflectivity is the area-weighted incoherent sum of the regions'
reflectivities, not the reflectivity of the averaged profile (Majkrzak,
O'Donovan & Berk 2006, sec. 1.4.4; Majkrzak et al. 2014, 2022).  Every
interior layer (all but the fronting and backing) is scaled by one factor
f, Gaussian of mean 1 and standard deviation sigma_t: a growth-rate
variation correlated across the layers.  Roughness, SLDs, magnetisation and
angles are unchanged; only the thicknesses scale:

    R_spread(Q) = SUM_k w_k R(Q; d_j -> f_k d_j),
    f_k = 1 + sqrt(2) sigma_t x_k,   w_k = h_k / sqrt(pi)

with x_k, h_k the n-point Gauss-Hermite nodes and weights
(numpy.polynomial.hermite.hermgauss; Abramowitz & Stegun 25.4.46); the w_k
sum to 1.  Stack.thickness_spread_nodes is n, an odd int >= 3, or 'auto':
n = clamp(odd_ceil(0.4 phi^2 + 3), 5, 41) with phi = Qmax D sigma_t (D the
interior thickness, Qmax the largest Q computed + 3 sigma_Q), which keeps
the largest relative error in R below 2e-3 (tests/test_thickness_spread.py:
two films of D = 300 A to Q = 0.2 need 5, 7, 11, 19, 23 and 33 nodes at
sigma_t = 3, 5, 7, 10, 12 and 15 %; the count grows as phi^2, so the
linear rule of plans/thickness_spread_plan_v1.md, fitted to a stack with
shallower fringes, gave 9 nodes and 7e-3 at 7 %).  fit.problem freezes it
for a fit, from the upper bound of sigma_t (the cost must not jump where n
changes).  The outer Gauss-Hermite nodes reach far into the tails: a node
whose factor f_k <= 0.05 (a thickness near zero, unphysical) is left out
and the others' weights renormalised; their weight is the Gaussian's tail
there (<= 2e-10 up to 15 %, a few 1e-6 at 20 %), and more than
SPREAD_DROP_MAX of it is refused (ValueError: a spread above ~25 %).  The average is linear, so it commutes with the resolution (both
schemes), the Q shift, the norms, scale and background: it is taken on the
ideal reflectivity, inside the function the resolution averages, each
node's slabs built once per parameter change (Stack._spread_sequences) by
build_sublayers itself, so both roughness schemes (the Licorne windows and
renormalisation included) scale as the nominal stack would.
critical_edges() depends only on the fronting and backing: the quadrature
grid is the same for every node.  sigma_t = 0 is the code path without it,
untouched; profile() and the slabs drawn stay the nominal (f = 1) ones.
Every f_k must be > 0.05 (ValueError otherwise).  It is not part of
LICORNE_EXACT: Licorne has no such parameter.

  C. F. Majkrzak, K. V. O'Donovan, N. F. Berk, "Polarized neutron
    reflectometry", in Neutron Scattering from Magnetic Materials
    (Elsevier, 2006), sec. 1.4.4.
  C. F. Majkrzak et al., "Determination of the effective transverse
    coherence of the neutron wave packet as employed in reflectivity
    investigations of condensed-matter structures. I. Measurements",
    Phys. Rev. A 89 (2014).
  C. F. Majkrzak et al., "The effect of transverse wavefront width on
    specular neutron reflection", J. Appl. Cryst. 55 (2022).
  S. L. Morelhao, C. I. Fornari, P. H. O. Rappl, E. Abramof, J. Appl.
    Cryst. 50, 399-410 (2017), doi:10.1107/S1600576717000760 (thickness
    distributions in Bi2Te3 films).
  A. R. J. Nelson, S. W. Prescott, J. Appl. Cryst. 52, 193-200 (2019),
    doi:10.1107/S1600576718017296 (refnx MixedReflectModel, incoherent
    sums).

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
from model.licorne_resolution import kernel as licorne_kernel

try:                       # compiled product (setup.py), else numpy's
    from model import _transfer
except ImportError:
    _transfer = None


# Layer attributes a fit may vary.  Layer.fit[attr] = {'vary', 'min', 'max'},
# bounds in the same units as the attribute (Angstrom, A^-2, turns), and
# optionally 'prior' (fit.priors SPEC, None for none).
FIT_PARAMS = ('thickness', 'NSLD_real', 'NSLD_img', 'MSLD_rho', 'MSLD_theta',
              'MSLD_phi', 'roughness_sigma', 'MSLD_sigma')
# Stack attributes a fit may vary, in Stack.fit the same way; their
# free_parameters() entries have layer index None.
STACK_FIT_PARAMS = ('scale', 'background', 'q_shift', 'thickness_spread')
# THICKNESS SPREAD: the node choices and the 'auto' rule's limits
SPREAD_NODE_CHOICES = ('auto', 5, 7, 9, 11, 15, 21)
SPREAD_NODES_MIN, SPREAD_NODES_MAX = 5, 41
SPREAD_F_MIN = 0.05             # nodes with a smaller factor are left out
SPREAD_DROP_MAX = 1e-4          # the weight they may carry


# Default fit bounds, the same for every layer: attr -> (min, max).
DEFAULT_BOUNDS = {
    'thickness':       (0.0, 500.0),       # Angstrom
    'NSLD_real':       (-2e-6, 10e-6),     # A^-2
    'NSLD_img':        (-1e-6, 0.0),       # A^-2
    'MSLD_rho':        (0.0, 5e-6),        # A^-2
    'MSLD_theta':      (0.0, 1.0),         # turns
    'MSLD_phi':        (-0.25, 0.25),      # turns, out of plane
    'roughness_sigma': (0.0, 10.0),        # Angstrom
    'MSLD_sigma':      (0.0, 10.0),        # Angstrom
    'scale':           (0.5, 1.5),
    'background':      (0.0, 1e-5),
    'q_shift':         (-0.002, 0.002),    # A^-1
    'thickness_spread': (0.0, 0.2),        # fraction
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
# the compiled product (model/_transfer.pyx) when it is built; False forces
# the numpy one (the tests compare them)
USE_COMPILED = True


def wrap_turns(t, decimals=9):
    """Angle in turns folded into [0, 1); rounding first (to `decimals`)
    keeps -1e-17 at 0 instead of 1 (i.e. 0 deg, not 360)."""
    return np.round(np.asarray(t, dtype=float), decimals) % 1.0


# wrap_turns rounding in the Licorne scheme: fine enough not to limit the
# agreement with Licorne (1e-9 turn costs ~1e-7 on a small spin-flip channel)
LICORNE_DECIMALS = 15


# |M| below M_TINY of the largest layer rho has no meaningful direction: its
# angle is round-off noise, or the direction of a leftover interface tail far
# from any magnetic layer (with interface_order 'additive', a rough
# interface's tail outliving a sharper one below it points along M_above -
# M_below; 'ordered' and 'independent' have none).  See Stack.angle_defined.
M_TINY = 1e-3

ROUGHNESS_SCHEMES = ('rms', 'licorne')
# Stack.interface_order: how the 'rms' scheme's interfaces combine (see
# PROFILE)
INTERFACE_ORDERS = ('additive', 'ordered', 'independent')
# Stack.interface_crossing: a layer is listed when its additive occupancy
# goes below -CROSSING_TINY (round-off in far tails stays above); the
# default depth grid has steps of CROSSING_STEP (A), at most
# CROSSING_POINTS points, out to CROSSING_PAD sigma beyond the interfaces
CROSSING_TINY = 1e-9
CROSSING_STEP = 0.05
CROSSING_POINTS = 20000
CROSSING_PAD = 8.0
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


def spread_nodes(n, s):
    """[(w_k, f_k)] of n Gauss-Hermite nodes at sigma_t s (THICKNESS
    SPREAD): those with f_k > SPREAD_F_MIN, renormalised; ValueError if the
    others weigh more than SPREAD_DROP_MAX."""
    x, h = np.polynomial.hermite.hermgauss(int(n))
    s = float(s)
    f = 1.0 + np.sqrt(2.0) * s * x
    w = h / np.sqrt(np.pi)
    keep = f > SPREAD_F_MIN
    lost = float(w[~keep].sum())
    if lost > SPREAD_DROP_MAX:
        raise ValueError(
            'thickness spread %.3g %%: a weight of %.2g of the thickness '
            'distribution lies at thicknesses below %g of the nominal; '
            'lower the spread' % (100 * s, lost, SPREAD_F_MIN))
    return list(zip(w[keep] / w[keep].sum(), f[keep]))


def _node_sum(seqs, R):
    """SUM_k w_k R(seq_k) over the thickness-spread nodes, in node order
    (THICKNESS SPREAD)."""
    out = None
    for w, seq in seqs:
        r = w * R(seq)
        out = r if out is None else out + r
    return out


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



# default resolution: Licorne's three-angle TOF resolution.m
DEFAULT_RESOLUTION = {
    'enabled': False, 'mode': 'tof',
    'tof_dlambda': 0.005,
    'tof_angles': [{'theta': 0.006, 'dtheta': 3e-4, 'qmax': 0.04},
                   {'theta': 0.010, 'dtheta': 5e-4, 'qmax': 0.12},
                   {'theta': 0.017, 'dtheta': 5e-4, 'qmax': None}],
    'scheme': 'quadrature', 'licorne_fun': 3,
    'data_q': [], 'data_dq': [], 'data_set': '',
    'data_overlap': 'mean',
}
# resolution['scheme'] (see RESOLUTION)
RES_SCHEMES = ('quadrature', 'licorne')
# Licorne weight matrices kept per stack (one per Q grid, sigma and mode)
LICORNE_W_CACHE = 4


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
    if len(edges) >= 2:              # Licorne's template: Q <= Q2 is angle 2
        k = np.where(Q == edges[-1], k - 1, k)
    lam = 4*np.pi*np.sin(theta[k]) / Q
    return Q * np.hypot(dtheta[k] / theta[k], res['tof_dlambda'] / lam)


def _sigma_data(Q, res):
    """The measured dQ table (see RESOLUTION); Q > 0."""
    q = np.asarray(res.get('data_q', ()), dtype=float)
    if q.size == 0:
        raise ValueError('resolution from data dQ: no dQ table yet (choose '
                         'a dataset with Show…)')
    return np.interp(Q, q, np.asarray(res['data_dq'], dtype=float))


# resolution['mode'] -> sigma_Q(Q > 0, resolution dict)
RESOLUTION_MODES = {'tof': _sigma_tof, 'data': _sigma_data}
# keys of resolution modes that no longer exist (fixed wavelength, FWHM dQ)
OLD_RESOLUTION_KEYS = ('wavelength', 'dlambda_rel', 'dtheta', 'data_fwhm')


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

    roughness_sigma / roughness_model / roughness_sublayer / MSLD_sigma
    describe the interface at the TOP of this layer.  `fit` holds the fit settings of the
    FIT_PARAMS (see fit_entry).
    """

    def __init__(self, name: str = "", thickness: float = 0.0,
                 NSLD_real: float = 0.0, NSLD_img: float = 0.0,
                 MSLD_rho: float = 0.0, MSLD_theta: float = 0.0,
                 roughness_sigma: float = 0.0, roughness_model: str = "tanh",
                 roughness_sublayer: int = 5, fit: dict = None,
                 MSLD_phi: float = 0.0, MSLD_sigma: float = None):
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
        # Angstrom, the magnetic profile's own sigma; None: roughness_sigma
        # (see MAGNETIC ROUGHNESS)
        self.MSLD_sigma = MSLD_sigma
        # (the deepcopy only when there is something to copy: the slabs of
        # build_sublayers are made by the thousand during a fit)
        self.fit = copy.deepcopy(fit) if fit else {}

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


def _steps(z, Z, sigmas, models, order):
    """F_j(z) of every interface (see PROFILE), shape (len(Z),) + z.shape:
    the steps f_j themselves for 'additive', their running max from the
    backing up for 'ordered', 1 - their running product of (1 - f) from the
    backing up for 'independent' (as sums of log1p, so that a far tail
    keeps its digits)."""
    z = np.asarray(z, dtype=float)
    f = np.stack([_step(z - Zj, s, m) for Zj, s, m in zip(Z, sigmas, models)])
    if order == 'ordered':
        f = np.maximum.accumulate(f[::-1], axis=0)[::-1]
    elif order == 'independent':
        with np.errstate(divide='ignore'):
            f = -np.expm1(np.cumsum(np.log1p(-f)[::-1], axis=0)[::-1])
    return f


def _combine(F, vals):
    """The profile v_0 + SUM_j (v_{j+1} - v_j) F_j of the medium values
    vals (len(F) + 1 entries) from the steps F of _steps."""
    vals = np.asarray(vals)
    out = np.full(F.shape[1:], vals[0], dtype=vals.dtype)
    for j in range(len(F)):
        out = out + (vals[j + 1] - vals[j]) * F[j]
    return out


# ----------------------------------------------------------------- ties ----
# A tie fits several layers' values of one attribute together.
# Stack.ties = [{'members': [[i, attr], ...],
#                'mode': 'equal' | 'licorne' | 'total',
#                'v0': [...], 'lo': [...], 'hi': [...]}]
# (v0, lo, hi: each member's value and bounds when the tie was made).
#   'equal'    every member takes the fitted value S; S starts at the first
#              member's value, its bounds are the intersection of the
#              members' (current) bounds
#   'licorne'  Licorne's coupled parameters (equal <field>_fit numbers,
#              licorne_cb.m prepare_fitstr, l. 8936-8938, and PAR = sum of
#              the members' values, l. 8747): S = sum of the members, in
#              [sum lo, sum hi]; above S0 = sum v0 each member moves from
#              its v0 towards its hi in proportion, below towards its lo:
#              m_j = v0_j + (S - S0) (hi_j - v0_j) / (sum hi - S0), S >= S0
#              m_j = v0_j + (S - S0) (lo_j - v0_j) / (sum lo - S0), S < S0
#   'total'    thickness only: the members' total thickness T (their sum
#              when the fit starts) stays fixed while each varies.  Every
#              member but the last is fitted on its own, the last takes the
#              rest, t_last = T - (the others).  A fitted member's bounds
#              are its own, narrowed so that the last stays within its own
#              whatever the others do: the room the last has when the fit
#              starts is shared equally among them (Stack.total_bounds; all
#              of it with two members).
# Stack.free_parameters() lists an 'equal' or 'licorne' tie once, as
# (('tie', k), attr, S, lo, hi), and a 'total' tie as one entry per fitted
# member m (its place in the tie), (('tie', k, m), 'thickness', t_m, lo,
# hi), when any member is ticked Fit; the members are not listed
# themselves.  Stack.owner(('tie', k)) is a TieOwner, so setattr(owner,
# attr, S) sets every member; Stack.owner(('tie', k, m)) a TotalOwner,
# whose setattr sets member m and gives the difference to the last.
TIE_MODES = ('equal', 'licorne', 'total')


class TieOwner:
    """What Stack.owner returns for tie k: reading any attribute gives its
    fitted value S, writing one sets every member (see ties); `fit` is
    empty (a tie has no prior; its bounds come from the members)."""

    def __init__(self, stack, k):
        object.__setattr__(self, '_stack', stack)
        object.__setattr__(self, '_k', k)
        object.__setattr__(self, 'fit', {})

    def __getattr__(self, name):
        return self._stack.tie_value(self._k)

    def __setattr__(self, name, value):
        self._stack.set_tie(self._k, value)

    def fit_entry(self, attr):
        lo, hi = self._stack.tie_bounds(self._k)
        return {'vary': True, 'min': lo, 'max': hi}


class TotalOwner:
    """What Stack.owner returns for member m of 'total' tie k: that layer
    (its value, `fit` and bounds, so a prior on its thickness applies),
    except that writing its thickness gives the difference to the tie's
    last member, keeping the total (see ties)."""

    def __init__(self, stack, k, m):
        i = stack.ties[k]['members'][m][0]
        object.__setattr__(self, '_stack', stack)
        object.__setattr__(self, '_k', k)
        object.__setattr__(self, '_layer', stack.layers[i])

    def __getattr__(self, name):
        return getattr(self._layer, name)

    def __setattr__(self, name, value):
        last = self._stack.layers[self._stack.ties[self._k]['members'][-1][0]]
        rest = self._layer.thickness + last.thickness
        self._layer.thickness = float(value)
        last.thickness = float(rest - value)

    def fit_entry(self, attr):
        return self._layer.fit_entry(attr)


# ---------------------------------------------------------------- Stack ----
class Stack:
    """[fronting, L1, ..., LN, backing].  Fronting and backing are
    semi-infinite; their thickness is ignored."""

    def __init__(self, layers, tail: float = 3.0, roughness_scheme='rms',
                 magnetic_smearing=None, interface_order='ordered'):
        if len(layers) < 2:
            raise ValueError('need at least a fronting and a backing')
        if roughness_scheme not in ROUGHNESS_SCHEMES:
            raise ValueError('unknown roughness scheme %r' % (roughness_scheme,))
        self.layers = list(layers)
        self.tail = tail                 # unclipped window half-width = tail*sigma
        # 'rms' or 'licorne' (see ROUGHNESS SCHEME)
        self.roughness_scheme = roughness_scheme
        # 'ordered', 'additive' or 'independent' ('rms' scheme only, see
        # PROFILE)
        self.interface_order = interface_order
        # 'manual', 'licorne' or 'none' (Licorne scheme only, see
        # model.roughness RENORMALISATION)
        self.licorne_renorm = 'manual'
        # 'infinite' or 'licorne' (Licorne scheme only, see ROUGHNESS SCHEME)
        self.licorne_outer = 'infinite'
        self.sublayers = None
        # (sublayers, their arrays) from _set_sublayers, for
        # build_transfer_matrix
        self._slab_arrays = None
        self.scale = 1.0                     # factor on every channel
        self.background = 0.0                # constant added to every channel
        self.q_shift = 0.0                   # A^-1, see Q SHIFT
        self.thickness_spread = 0.0          # fraction, see THICKNESS SPREAD
        self.thickness_spread_nodes = 'auto'  # or an odd int >= 3
        # incremented whenever self.sublayers is replaced: the key of the
        # thickness-spread nodes' slabs (_spread_sequences)
        self._sub_gen = 0
        self._spread_cache = None
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
        self._licorne_W = {}             # cached Licorne resolution kernels
        # the LICORNE_EXACT settings set_licorne_exact() overwrote, for
        # set_licorne_exact(False); None: nothing to go back to
        self._before_exact = None
        self.ties = []                   # see ties above

    # -- Licorne-exact ----------------------------------------------------------
    # settings with which the stack reproduces Licorne 1.4.2 (MATLAB path):
    # attribute -> value; 'resolution.scheme' is resolution['scheme']
    LICORNE_EXACT = {'roughness_scheme': 'licorne', 'magnetic_smearing': 'step',
                     'step_fallback': False, 'licorne_renorm': 'licorne',
                     'licorne_outer': 'licorne',
                     'resolution.scheme': 'licorne'}
    # what set_licorne_exact(False) restores when there are no earlier
    # settings (the roughness scheme and the smearing stay as they are)
    PHYSICAL = {'step_fallback': True, 'licorne_renorm': 'manual',
                'licorne_outer': 'infinite', 'resolution.scheme': 'quadrature'}

    def _setting(self, key, value=None):
        if key.startswith('resolution.'):
            k = key.split('.', 1)[1]
            if value is not None:
                self.resolution[k] = value
            return self.resolution.get(k)
        if value is not None:
            setattr(self, key, value)
        return getattr(self, key)

    def set_licorne_exact(self, on=True):
        """on: every setting of LICORNE_EXACT (Licorne's roughness scheme,
        'step' without fallback, its 50-point J, its 1.5 t outer
        pseudo-layers, its convolution on the data grid), remembering the
        ones it replaces.  off: those settings back, or the physical
        defaults of PHYSICAL when there are none (a stack built or loaded
        exact)."""
        if on:
            if not self.is_licorne_exact():
                self._before_exact = {k: self._setting(k)
                                      for k in self.LICORNE_EXACT}
            new = self.LICORNE_EXACT
        else:
            new = self._before_exact or self.PHYSICAL
            self._before_exact = None
        for k, v in new.items():
            self._setting(k, v)

    def is_licorne_exact(self):
        return all(self._setting(k) == v
                   for k, v in self.LICORNE_EXACT.items())

    # -- serialisation ------------------------------------------------------
    def to_dict(self):
        """Plain-JSON description of the sample (layers, tail, scale,
        background, resolution, smearing).  No polarisation: it belongs to
        each measurement."""
        return {'tail': self.tail,
                'scale': self.scale,
                'background': self.background,
                'q_shift': self.q_shift,
                'thickness_spread': self.thickness_spread,
                'thickness_spread_nodes': self.thickness_spread_nodes,
                'fit': copy.deepcopy(self.fit),
                'q_in_fronting': self.q_in_fronting,
                'resolution': copy.deepcopy(self.resolution),
                'magnetic_smearing': self.magnetic_smearing,
                'roughness_scheme': self.roughness_scheme,
                'interface_order': self.interface_order,
                'licorne_renorm': self.licorne_renorm,
                'licorne_outer': self.licorne_outer,
                'step_fallback': self.step_fallback,
                'ties': copy.deepcopy(self.ties),
                'layers': [dict(vars(l), fit=copy.deepcopy(l.fit))
                           for l in self.layers]}

    @classmethod
    def from_dict(cls, d):
        """Inverse of to_dict.  Files written before the Licorne scheme
        existed load as 'rms', before interface_order existed as
        'additive' (so old fits reproduce); a resolution in the removed fixed-wavelength
        ('mono') mode loads as 'tof', off."""
        s = cls([Layer(**l) for l in d['layers']], tail=d['tail'],
                roughness_scheme=d.get('roughness_scheme', 'rms'),
                magnetic_smearing=d.get('magnetic_smearing'),
                interface_order=d.get('interface_order', 'additive'))
        s.background = d['background']
        s.scale = float(d.get('scale', 1.0))
        s.q_shift = float(d.get('q_shift', 0.0))
        s.thickness_spread = float(d.get('thickness_spread', 0.0))
        s.thickness_spread_nodes = d.get('thickness_spread_nodes', 'auto')
        s.fit = copy.deepcopy(d.get('fit', {}))
        s.q_in_fronting = bool(d['q_in_fronting'])
        res = copy.deepcopy(d['resolution'])
        if res.get('mode', 'tof') not in RESOLUTION_MODES:
            res.update(mode='tof', enabled=False)
        for k in OLD_RESOLUTION_KEYS:
            res.pop(k, None)
        s.resolution.update(res)
        s.licorne_renorm = d.get('licorne_renorm', 'manual')
        s.licorne_outer = d.get('licorne_outer', 'infinite')
        s.step_fallback = bool(d.get('step_fallback', True))
        s.ties = copy.deepcopy(d.get('ties', []))
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
        if attr == 'MSLD_sigma':
            return i > 0 and self.layers[i].MSLD_sigma is not None
        return True

    def fit_entry(self, attr):
        """{'vary', 'min', 'max'} of the stack parameter `attr`, created
        from default_fit if unset."""
        if attr not in self.fit:
            self.fit[attr] = default_fit(attr)
        return self.fit[attr]

    # -- ties (see ties above) ------------------------------------------
    def tie(self, members, mode='equal'):
        """Add a tie of members [(layer index, attr)] (one attr, two or
        more layers, none already tied); returns its index."""
        members = [[int(i), str(a)] for i, a in members]
        if mode not in TIE_MODES:
            raise ValueError('tie mode must be one of %s' % ', '.join(TIE_MODES))
        if mode == 'total':
            if any(a != 'thickness' for _, a in members):
                raise ValueError('a total can only be kept for thicknesses')
            if not all(self.fit_allowed(i, a) for i, a in members):
                raise ValueError('the fronting and backing have no '
                                 'thickness')
        if len(members) < 2 or len({a for _, a in members}) != 1:
            raise ValueError('a tie needs two or more layers and one '
                             'attribute')
        if len({i for i, _ in members}) != len(members):
            raise ValueError('a layer appears twice in the tie')
        if self.tied() & {tuple(m) for m in members}:
            raise ValueError('a member is already tied')
        entries = [self.layers[i].fit_entry(a) for i, a in members]
        self.ties.append({
            'members': members, 'mode': mode,
            'v0': [float(getattr(self.layers[i], a)) for i, a in members],
            'lo': [float(e['min']) for e in entries],
            'hi': [float(e['max']) for e in entries]})
        return len(self.ties) - 1

    def untie(self, i, attr):
        """Remove the tie holding (layer i, attr), if any."""
        self.ties = [t for t in self.ties
                     if [i, attr] not in [list(m) for m in t['members']]]

    def tied(self):
        """{(layer index, attr)} of every tie member."""
        return {tuple(m) for t in self.ties for m in t['members']}

    def tie_of(self, i, attr):
        """Index of the tie holding (layer i, attr), or None."""
        for k, t in enumerate(self.ties):
            if [i, attr] in [list(m) for m in t['members']]:
                return k
        return None

    def tie_name(self, k):
        """'tie(Layer 2.NSLD_real, Layer 3.NSLD_real)', or
        'total(A.thickness, B.thickness)' for a 'total' tie."""
        t = self.ties[k]
        return '%s(%s)' % ('total' if t['mode'] == 'total' else 'tie',
                           ', '.join('%s.%s' % (self.layers[i].name, a)
                                     for i, a in t['members']))

    def entry_name(self, i):
        """Name of the free_parameters() entry of a tie: tie_name for
        ('tie', k), 'A.thickness in total(A.thickness, B.thickness)' for
        member m of a 'total' tie, ('tie', k, m)."""
        if len(i) == 2:
            return self.tie_name(i[1])
        j, a = self.ties[i[1]]['members'][i[2]]
        return '%s.%s in %s' % (self.layers[j].name, a, self.tie_name(i[1]))

    def total_bounds(self, k):
        """(T, [(lo, hi)] of every member of 'total' tie k but the last):
        each member's own bounds, narrowed so that whatever the others
        take the last stays within its own (see ties).  The room the last
        has, down to its min and up to its max, is shared equally among
        the others: with two members, the one fitted can take all of it."""
        mem = self.ties[k]['members']
        t = [float(self.layers[i].thickness) for i, _ in mem]
        es = [self.layers[i].fit_entry(a) for i, a in mem]
        T, n = sum(t), len(mem) - 1
        grow = (t[-1] - max(float(es[-1]['min']), 0.0)) / n
        shrink = (float(es[-1]['max']) - t[-1]) / n
        return T, [(max(float(e['min']), 0.0, tm - shrink),
                    min(float(e['max']), tm + grow))
                   for tm, e in zip(t[:-1], es[:-1])]

    def tie_bounds(self, k):
        t = self.ties[k]
        if t['mode'] == 'licorne':
            return float(sum(t['lo'])), float(sum(t['hi']))
        es = [self.layers[i].fit_entry(a) for i, a in t['members']]
        return (float(max(e['min'] for e in es)),
                float(min(e['max'] for e in es)))

    def tie_value(self, k):
        t = self.ties[k]
        vals = [getattr(self.layers[i], a) for i, a in t['members']]
        return float(sum(vals)) if t['mode'] == 'licorne' else float(vals[0])

    def set_tie(self, k, S):
        """Set every member of tie k from its fitted value S."""
        t = self.ties[k]
        S = float(S)
        if t['mode'] == 'equal':
            vals = [S] * len(t['members'])
        else:
            S0 = sum(t['v0'])
            end = t['hi'] if S >= S0 else t['lo']
            span = sum(end) - S0
            vals = [v if span == 0 else v + (S - S0) * (e - v) / span
                    for v, e in zip(t['v0'], end)]
        for (i, a), v in zip(t['members'], vals):
            setattr(self.layers[i], a, float(v))

    def reindex_ties(self, old_layers):
        """Follow the layers after they were added, removed or moved
        (old_layers: the list before): members of a removed layer leave
        their tie, a tie left with one member goes."""
        where = {id(l): i for i, l in enumerate(self.layers)}
        ties = []
        for t in self.ties:
            keep = [j for j, (i, _) in enumerate(t['members'])
                    if i < len(old_layers) and id(old_layers[i]) in where]
            if len(keep) < 2:
                continue
            t = {k: ([t[k][j] for j in keep] if isinstance(t[k], list)
                     else t[k]) for k in t}
            t['members'] = [[where[id(old_layers[i])], a]
                            for i, a in t['members']]
            ties.append(t)
        self.ties = ties

    def free_parameters(self):
        """[(layer index, attr, value, min, max)] of every varied parameter:
        the layers' in order, then the ties (layer index ('tie', k), or
        ('tie', k, m) for the members of a 'total' tie, when any member is
        ticked Fit), then the stack's own (layer index None)."""
        out = []
        tied = self.tied()
        for i, l in enumerate(self.layers):
            for attr in FIT_PARAMS:
                e = l.fit.get(attr)
                if e and e['vary'] and self.fit_allowed(i, attr) and \
                        (i, attr) not in tied:
                    out.append((i, attr, getattr(l, attr), e['min'], e['max']))
        for k, t in enumerate(self.ties):
            attr = t['members'][0][1]
            if not any(self.layers[i].fit.get(a, {}).get('vary')
                       and self.fit_allowed(i, a) for i, a in t['members']):
                continue
            if t['mode'] == 'total':
                for m, (lo, hi) in enumerate(self.total_bounds(k)[1]):
                    i = t['members'][m][0]
                    out.append((('tie', k, m), attr,
                                self.layers[i].thickness, lo, hi))
            else:
                out.append((('tie', k), attr, self.tie_value(k),
                            *self.tie_bounds(k)))
        for attr in STACK_FIT_PARAMS:
            e = self.fit.get(attr)
            if e and e['vary']:
                out.append((None, attr, getattr(self, attr), e['min'],
                            e['max']))
        return out

    def owner(self, i):
        """The object a free_parameters() entry with layer index i sets:
        layer i, the stack itself for None, a TieOwner for ('tie', k), a
        TotalOwner for ('tie', k, m)."""
        if i is None:
            return self
        if isinstance(i, tuple):
            return TieOwner(self, i[1]) if len(i) == 2 else \
                TotalOwner(self, i[1], i[2])
        return self.layers[i]

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
                self.tail * max(lay.roughness_sigma, 0.0,
                                lay.MSLD_sigma or 0.0)
            out.append((Zj - min(w, t[j] / 2.0), Zj + min(w, t[j + 1] / 2.0)))
        return out

    def _check_modes(self):
        if self.roughness_scheme not in ROUGHNESS_SCHEMES:
            raise ValueError('unknown roughness scheme %r'
                             % (self.roughness_scheme,))
        if self.magnetic_smearing not in SMEARING_MODES:
            raise ValueError('unknown magnetic smearing %r'
                             % (self.magnetic_smearing,))
        if self.interface_order not in INTERFACE_ORDERS:
            raise ValueError('unknown interface order %r'
                             % (self.interface_order,))

    # -- slicing ------------------------------------------------------------
    def build_sublayers(self):
        """Replace the interior layers by slabs carrying the smeared profile.

        Stores and returns the list; fronting and backing are not included.
        """
        self._check_modes()
        if self.roughness_scheme == 'licorne':
            return self._build_licorne()
        nsub = [max(int(l.roughness_sublayer), 1) for l in self.layers[1:]]
        pts = []
        for (lo, hi), m in zip(self.windows(), nsub):
            pts.append(np.array([lo]) if hi - lo < 1e-9
                       else np.linspace(lo, hi, m + 1))
        e = np.unique(np.round(np.concatenate(pts), 6))
        e = e[np.concatenate([[True], np.diff(e) > 1e-4])]

        c = 0.5 * (e[:-1] + e[1:])
        F = self._rms_steps(c)
        return self._set_sublayers(np.diff(e), _combine(F, self._nsld()),
                                   *self._magnetic_profile(
                                       self._magnetic_steps(c, F)))

    def _set_sublayers(self, th, v_n, v_rho, v_theta, v_phi):
        self._sub_gen = getattr(self, '_sub_gen', 0) + 1
        self.sublayers = [
            Layer(name='%s_%03d' % (self.layers[0].name or 'slab', i),
                  thickness=float(th[i]), NSLD_real=v_n[i].real,
                  NSLD_img=v_n[i].imag, MSLD_rho=float(v_rho[i]),
                  MSLD_theta=float(v_theta[i]), MSLD_phi=float(v_phi[i]),
                  roughness_sigma=0.0, roughness_sublayer=1)
            for i in range(len(th))]
        # the same slabs as the arrays build_transfer_matrix multiplies (the
        # in-plane rho of inplane_rho), so that it need not read them back
        # one Layer at a time; valid while self.sublayers is this list
        self._slab_arrays = (self.sublayers, (
            np.asarray(v_n, dtype=complex),
            np.asarray(v_rho, dtype=float) * cos_turns(v_phi),
            np.asarray(v_theta, dtype=float), np.asarray(th, dtype=float)))
        return self.sublayers

    def profile(self, z):
        """Exact continuous profile: (nsld complex, |M| = msld_rho, MSLD_theta,
        MSLD_phi), angles in turns.  The neutrons only see the in-plane part
        rho cos(phi) (HALPERIN).  In the Licorne scheme this is the truncated,
        renormalised profile the slabs sample, jumps included."""
        self._check_modes()
        if self.roughness_scheme == 'licorne':
            return self._licorne_profile(z)
        F = self._rms_steps(z)
        return (_combine(F, self._nsld()),) + \
            self._magnetic_profile(self._magnetic_steps(z, F))

    def _nsld(self):
        return np.array([complex(l.NSLD_real, l.NSLD_img) for l in self.layers])

    def _rms_steps(self, z, order=None, magnetic=False):
        """F_j at depths z of every interface in the 'rms' scheme (_steps),
        combined as `order` says (default interface_order); with the
        magnetic sigmas (see MAGNETIC ROUGHNESS) if `magnetic`."""
        lay = self.layers[1:]
        sig = [l.MSLD_sigma if magnetic and l.MSLD_sigma is not None
               else l.roughness_sigma for l in lay]
        return _steps(z, self._interfaces(), [max(s, 0.0) for s in sig],
                      [l.roughness_model for l in lay],
                      self.interface_order if order is None else order)

    def _magnetic_steps(self, z, F, order=None):
        """The magnetic profile's steps: F itself (the NSLD's) unless a
        layer has its own MSLD_sigma."""
        if all(l.MSLD_sigma is None for l in self.layers[1:]):
            return F
        return self._rms_steps(z, order, magnetic=True)

    def _magnetic_profile(self, F):
        """(|M|, theta, phi in turns) from the steps F (_rms_steps), smeared
        as magnetic_smearing says (see MAGNETISATION)."""
        rho = np.array([l.MSLD_rho for l in self.layers])
        theta = np.array([l.MSLD_theta for l in self.layers])
        phi = np.array([l.MSLD_phi for l in self.layers])
        if self.magnetic_smearing == 'angle':
            # the interpolated angles are kept as is (the physics depends on
            # the path they take); only the reported theta is folded
            return (_combine(F, rho), wrap_turns(_combine(F, theta)),
                    _combine(F, phi))
        shape = F.shape[1:]
        if self.magnetic_smearing == 'step':
            # occupancy of layer k: F_{k-1} - F_k (F_{-1} = 1, F_last = 0);
            # each depth takes the angle of the layer that occupies it most
            # among those that have an angle (argmax: ties go to the upper)
            f = np.concatenate([np.ones((1,) + shape), F,
                                np.zeros((1,) + shape)])
            has = self._has_angle()
            occ = np.where(has.reshape((-1,) + (1,)*len(shape)),
                           f[:-1] - f[1:], -np.inf)
            k = np.argmax(occ.reshape(len(self.layers), -1), 0).reshape(shape)
            some = np.any(has)
            return (_combine(F, rho),
                    wrap_turns(np.where(some, theta[k], 0.0)),
                    np.where(some, phi[k], 0.0))
        # in-plane part as one complex number, M_z separately; with phi = 0
        # everywhere mz is exactly 0 and this is the previous 2D smearing
        mp = _combine(F, rho * cos_turns(phi) * np.exp(2j*np.pi*theta))
        mz = _combine(F, rho * sin_turns(phi))
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

    @staticmethod
    def _licorne_values_all(layers):
        """_licorne_values of every layer at once, one row each (the same
        numbers: numpy's cos and sin give an array the bits they give each
        of its elements)."""
        a = np.array([[l.NSLD_real, l.NSLD_img, l.MSLD_rho, l.MSLD_theta,
                       l.MSLD_phi] for l in layers], dtype=float).T
        r, th, ph = a[2], a[3], a[4]
        return np.column_stack([a[0], a[1], r,
                                r * cos_turns(ph) * cos_turns(th),
                                r * cos_turns(ph) * sin_turns(th),
                                r * sin_turns(ph)])

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
        vals = self._licorne_values_all(self.layers)
        out = []
        for j, Zj in enumerate(self._interfaces()):
            own = self.layers[j + 1]
            sL = max(own.roughness_sigma, 0.0)
            kind = own.roughness_model
            Fa, Fb, la, lb, _ = rough.renormalised(
                vals[j], vals[j + 1], t[j], t[j + 1], sL, kind,
                self.licorne_renorm)
            w = {'Z': float(Zj), 'la': la, 'lb': lb, 'sigma': sL,
                 'kind': kind, 'N': max(int(own.roughness_sublayer), 1),
                 'Fa': Fa, 'Fb': Fb, 'a': j, 'b': j + 1, 'mag': None,
                 'va': vals[j], 'vb': vals[j + 1]}
            if own.MSLD_sigma is not None:
                # the magnetic profile's own Licorne window (MAGNETIC
                # ROUGHNESS); la / lb become the extent of both
                sM = max(own.MSLD_sigma, 0.0)
                FaM, FbM, laM, lbM, _ = rough.renormalised(
                    vals[j], vals[j + 1], t[j], t[j + 1], sM, kind,
                    self.licorne_renorm)
                w['mag'] = {'sigma': sM, 'la': laM, 'lb': lbM,
                            'Fa': FaM, 'Fb': FbM, 'la_n': la, 'lb_n': lb}
                w['la'], w['lb'] = max(la, laM), max(lb, lbM)
            out.append(w)
        return out

    def _licorne_window(self, w, x):
        """(nsld complex, rho, theta, phi) at offsets x from interface w,
        inside its window."""
        x = np.asarray(x, dtype=float)
        s = rough.profile_fraction(x, w['sigma'], w['kind'])
        v = w['Fa'][:, None] + (w['Fb'] - w['Fa'])[:, None] * s[None, :]
        A, B = self.layers[w['a']], self.layers[w['b']]
        m = w['mag']
        if m is not None:
            # the NSLD (rows 0, 1) and the magnetic values (rows 2-5) each
            # follow their own window, their layer's core value outside it
            sm = rough.profile_fraction(x, m['sigma'], w['kind'])
            v[2:] = (m['Fa'][2:, None]
                     + (m['Fb'] - m['Fa'])[2:, None] * sm[None, :])
            va = w['va'][:, None]
            vb = w['vb'][:, None]
            for rows, la, lb in ((slice(0, 2), m['la_n'], m['lb_n']),
                                 (slice(2, 6), m['la'], m['lb'])):
                v[rows] = np.where(x < -la, va[rows],
                                   np.where(x >= lb, vb[rows], v[rows]))
            s = np.where(x < -m['la'], 0.0, np.where(x >= m['lb'], 1.0, sm))
        nsld = v[0] + 1j * v[1]
        mode = self.magnetic_smearing
        if mode == 'vector':
            mp = v[3] + 1j * v[4]
            a = np.abs(mp)
            return (nsld, np.hypot(a, v[5]),
                    wrap_turns(np.angle(mp) / (2*np.pi), LICORNE_DECIMALS),
                    np.arctan2(v[5], a) / (2*np.pi))
        if mode == 'angle':
            # angles interpolated with the profile fraction, never
            # renormalised (an angle has no amount of material)
            return (nsld, v[2],
                    wrap_turns(A.MSLD_theta
                               + (B.MSLD_theta - A.MSLD_theta) * s,
                               LICORNE_DECIMALS),
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
        return (nsld, v[2], wrap_turns(np.where(none, 0.0, theta),
                                       LICORNE_DECIMALS),
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
                float(wrap_turns(layer.MSLD_theta, LICORNE_DECIMALS)),
                layer.MSLD_phi)

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
            if w['mag'] is None:
                h = (w['la'] + w['lb']) / N
                x = -w['la'] + h * (np.arange(N) + 0.5)
                th.extend([h] * N)
            else:
                # both windows' N equal slabs, so that every jump of either
                # profile sits on a slab edge
                m = w['mag']
                e = np.unique(np.concatenate(
                    [np.linspace(-m['la_n'], m['lb_n'], N + 1),
                     np.linspace(-m['la'], m['lb'], N + 1)]))
                e = e[np.concatenate([[True], np.diff(e) > 1e-9])]
                x = 0.5 * (e[:-1] + e[1:])
                th.extend(np.diff(e))
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

    # -- crossing interfaces (see PROFILE) ------------------------------------
    def _linear_quantities(self):
        """{name: value of every layer} of the quantities the 'rms' scheme
        smears as v_0 + SUM (v_{j+1} - v_j) F_j under magnetic_smearing
        (see MAGNETISATION); M_x, M_y, M_z are rho's Cartesian parts."""
        L = self.layers
        rho = np.array([l.MSLD_rho for l in L])
        th = np.array([l.MSLD_theta for l in L])
        ph = np.array([l.MSLD_phi for l in L])
        q = {'NSLD_real': np.array([l.NSLD_real for l in L]),
             'NSLD_img': np.array([l.NSLD_img for l in L])}
        if self.magnetic_smearing == 'vector':
            q.update(M_x=rho * cos_turns(ph) * cos_turns(th),
                     M_y=rho * cos_turns(ph) * sin_turns(th),
                     M_z=rho * sin_turns(ph))
        elif self.magnetic_smearing == 'angle':
            q.update(MSLD_rho=rho, MSLD_theta=th, MSLD_phi=ph)
        else:
            q['MSLD_rho'] = rho
        return q

    def interface_crossing(self, z=None):
        """Where the 'rms' interfaces cross and what interface_order
        'additive' does there, whatever the stack's own order (see PROFILE);
        [] in the Licorne scheme.  One entry per layer k whose additive
        occupancy f_{k-1} - f_k goes below -CROSSING_TINY at the depths z
        (default: a grid of step CROSSING_STEP over every interface):
            'layer'      k
            'occupancy'  its most negative occupancy
            'z'          the depth of that minimum
            'host'       the layer whose nominal depth range holds 'z'
            'zrange'     (lo, hi): where the occupancy is below 1 % of its
                         minimum
            'excursion'  {quantity (_linear_quantities): signed %}: how far
                         the additive profile leaves the range of the layer
                         values over zrange, where this layer's occupancy is
                         the most negative one (< 0 below the smallest
                         value, > 0 above the largest), in % of the host
                         layer's value (of the largest |value| when that
                         one is 0)."""
        if self.roughness_scheme != 'rms':
            return []
        Z = self._interfaces()
        if z is None:
            sig = max(max(l.roughness_sigma, l.MSLD_sigma or 0.0, 0.0)
                      for l in self.layers[1:])
            pad = CROSSING_PAD * sig + 10.0
            n = min(CROSSING_POINTS,
                    int((Z[-1] - Z[0] + 2 * pad) / CROSSING_STEP) + 1)
            z = np.linspace(Z[0] - pad, Z[-1] + pad, n)
        z = np.asarray(z, dtype=float).ravel()
        f = self._rms_steps(z, 'additive')
        fm = self._magnetic_steps(z, f, 'additive')
        one = np.ones((1, z.size))
        # a layer's occupancy in the NSLD or in the magnetic profile
        occ = np.minimum(*(np.concatenate([one, g]) - np.concatenate([g, 0 * one])
                           for g in (f, fm)))
        quant = self._linear_quantities()
        prof = {q: _combine(f if q.startswith('NSLD') else fm, v)
                for q, v in quant.items()}
        host = np.searchsorted(Z, z, side='right')
        worst_layer = np.argmin(occ, axis=0)
        out = []
        for k, o in enumerate(occ):
            i = int(np.argmin(o))
            if o[i] >= -CROSSING_TINY:
                continue
            sel = o < 0.01 * o[i]
            own = sel & (worst_layer == k)
            h = int(host[i])
            exc = {}
            for q, v in quant.items():
                if not own.any():
                    exc[q] = 0.0
                    continue
                p = prof[q][own]
                below = min(p.min() - v.min(), 0.0)
                above = max(p.max() - v.max(), 0.0)
                ref = abs(v[h]) or np.max(np.abs(v))
                worst = below if -below >= above else above
                exc[q] = float(100.0 * worst / ref) if ref > 0 else 0.0
            out.append({'layer': k, 'occupancy': float(o[i]),
                        'z': float(z[i]), 'host': h,
                        'zrange': (float(z[sel].min()), float(z[sel].max())),
                        'excursion': exc})
        return out

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
        # sinh(S d)/S -> d at S = 0 (Q at a slab's own critical edge)
        with np.errstate(invalid='ignore', divide='ignore'):
            s1S = np.where(S1 == 0, d, s1/S1)
            s3S = np.where(S3 == 0, d, s3/S3)

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

    def build_transfer_matrix(self, Q, seq=None):
        """Product matrix A = A_N ... A_1 (deepest leftmost) of the sliced
        sublayers.  Shape (nQ,4,4).  The fronting and backing are never part
        of the product; they enter through the boundary condition.  seq: a
        (slabs, their arrays) pair as _set_sublayers stores it in
        _slab_arrays (a thickness-spread node, _spread_sequences) instead
        of self.sublayers.

        All slab matrices are built in one vectorised call, then multiplied
        pairwise (tree reduction): ~log2(N) batched matmuls instead of N.
        With model/_transfer built (setup.py) the compiled loop computes the
        same product instead, 1.5-3x faster.
        """
        Q = np.atleast_1d(np.asarray(Q))     # may be complex: only Q**2 is used
        if seq is not None:
            seq, arrays = seq
        else:
            if self.sublayers is None:
                self.build_sublayers()
            seq, arrays = self.sublayers, None
        if not seq:
            M = np.broadcast_to(np.eye(4, dtype=np.complex128),
                                Q.shape + (4, 4)).copy()
        else:
            cached = getattr(self, '_slab_arrays', None)
            if arrays is not None:
                slabs = arrays
            elif cached is not None and cached[0] is seq:
                slabs = cached[1]
            else:
                slabs = ([complex(l.NSLD_real, l.NSLD_img) for l in seq],
                         [self.inplane_rho(l) for l in seq],
                         [l.MSLD_theta for l in seq],
                         [l.thickness for l in seq])   # top -> bottom
            if _transfer is not None and USE_COMPILED:
                return _transfer.product(*slabs, Q.ravel()).reshape(
                    Q.shape + (4, 4))
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

    def _reflect(self, Q2, seq=None):
        """Reflection matrix r [:, out, in] in the film-normal (z) basis for
        the squared vacuum Q, plus the fronting's q+- (see _surround_K);
        seq: see build_transfer_matrix."""
        M = self.build_transfer_matrix(np.sqrt(Q2 + 0j), seq)
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
        mode = res.get('mode', 'tof')
        if mode not in RESOLUTION_MODES:
            raise ValueError('unknown resolution mode %r' % (mode,))
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma = RESOLUTION_MODES[mode](np.abs(Q), res)
        return np.where(Q != 0, sigma, 0.0)

    def data_sigma(self, Q, dQ):
        """sigma_Q of measured points (Q, dQ) in the 'data' mode: their dQ
        (a standard deviation), the dQ table where a dQ is missing or
        <= 0."""
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        dQ = np.atleast_1d(np.asarray(dQ, dtype=float))
        ok = np.isfinite(dQ) & (dQ > 0)
        if ok.all():
            return dQ
        return np.where(ok, dQ, self.resolution_sigma(Q))

    # -- reflectivity -------------------------------------------------------
    def reflectivity(self, Q, P0, P=None):
        """reflectivities for the single pair (P0, P), shape (nQ,)."""
        return self.reflectivities(Q, [(P0, P)])[:, 0]

    def reflectivities(self, Q, pairs, warn=True, norms=None, sigma=None):
        """norm x scale x reflectivity + background of each (P0, P) in
        pairs (see POLARISATION), averaged over the Q resolution when it is
        enabled.  Shape (nQ, len(pairs)).  norms: a fixed factor per pair
        (default 1; Licorne's per-channel Norm_factor, model.licorne_io
        NORMALISATION).  sigma: sigma_Q at each Q instead of the
        resolution's (e.g. Stack.data_sigma); Q may then repeat.
        warn=False silences the warning about a polarisation transverse to
        a magnetic fronting's M.  q_shift: see Q SHIFT."""
        pairs = _check_pairs(pairs)
        m = self.fronting_direction()
        trace, lab = self._trace, self._lab_channels
        if self.thickness_spread:
            # THICKNESS SPREAD: the nodes' average ideal reflectivity
            Q = np.atleast_1d(np.asarray(Q, dtype=float))
            s = self.resolution_sigma(Q) if sigma is None else \
                np.broadcast_to(np.asarray(sigma, dtype=float), Q.shape)
            qmax = float(np.max(np.abs(Q - self.q_shift))
                         + 3 * np.max(s, initial=0.0)) if Q.size else 0.0
            seqs = self._spread_sequences(qmax)

            def trace(q, pairs):
                return _node_sum(seqs, lambda seq: self._trace(q, pairs, seq))

            def lab(q):
                return _node_sum(seqs, lambda seq: self._lab_channels(q, seq))
        if m is None:
            R = self._resolve(Q, lambda q: trace(q, pairs), sigma,
                              self.q_shift)
        else:
            w = self._lab_weights(pairs, m, warn)
            R = _weigh(self._resolve(Q, lab, sigma, self.q_shift), w)
        if norms is not None:
            norms = np.asarray(norms, dtype=float)
            if norms.shape != (len(pairs),):
                raise ValueError('one norm per pair')
            R = norms * R
        return self.scale * R + self.background

    def spread_count(self, qmax, spread=None):
        """The number of thickness-spread nodes (THICKNESS SPREAD) for the
        largest Q qmax and sigma_t `spread` (default the stack's)."""
        n = self.thickness_spread_nodes
        if n != 'auto':
            n = int(n)
            if n < 3 or n % 2 == 0:
                raise ValueError('thickness_spread_nodes must be an odd '
                                 'integer >= 3 or \'auto\', not %r' % (n,))
            return n
        s = self.thickness_spread if spread is None else float(spread)
        D = sum(l.thickness for l in self.layers[1:-1])
        x = 0.4 * (qmax * D * s) ** 2 + 3
        k = int(np.ceil(x))
        k += 1 - k % 2                       # the smallest odd int >= x
        return min(max(k, SPREAD_NODES_MIN), SPREAD_NODES_MAX)

    def spread_nodes(self, qmax, spread=None):
        """[(w_k, f_k)] of the thickness spread (THICKNESS SPREAD) at
        sigma_t `spread` (default the stack's): the nodes with f_k > 0.05,
        their weights renormalised."""
        return spread_nodes(self.spread_count(qmax, spread),
                            self.thickness_spread if spread is None
                            else spread)

    def _spread_sequences(self, qmax):
        """[(w_k, seq_k)]: each thickness-spread node's slabs (seq as
        build_transfer_matrix takes it), built by build_sublayers with every
        interior thickness times f_k, then the stack restored exactly (the
        saved thicknesses, the nominal slabs).  Cached until the slabs are
        rebuilt (_sub_gen), the spread or the nodes change."""
        nodes = self.spread_nodes(qmax)
        if self.sublayers is None:
            self.build_sublayers()
        key = (getattr(self, '_sub_gen', 0), float(self.thickness_spread),
               tuple(float(f) for _, f in nodes))
        cache = getattr(self, '_spread_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        inner = self.layers[1:-1]
        d0 = [float(l.thickness) for l in inner]
        saved = (self.sublayers, self._slab_arrays,
                 getattr(self, '_sub_gen', 0))
        out = []
        try:
            for w, f in nodes:
                for l, d in zip(inner, d0):
                    l.thickness = d * f
                self.build_sublayers()
                out.append((float(w), self._slab_arrays))
        finally:
            for l, d in zip(inner, d0):
                l.thickness = d
            self.sublayers, self._slab_arrays, self._sub_gen = saved
        self._spread_cache = (key, out)
        return out

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

    def _resolve(self, Q, ideal, sigma=None, shift=0.0):
        """Average ideal(Q) -> (nQ, k) over the Q resolution (see
        RESOLUTION), or over Gaussians of the given sigma_Q; the sample's
        ideal at Q - shift (see Q SHIFT), sigma that of the measured Q."""
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        sigma = self.resolution_sigma(Q) if sigma is None else \
            np.broadcast_to(np.asarray(sigma, dtype=float), Q.shape)
        if not np.any(sigma > 0):
            return ideal(np.abs(Q - shift) if shift else Q)
        scheme = self.resolution.get('scheme', 'quadrature')
        if scheme not in RES_SCHEMES:
            raise ValueError('unknown resolution scheme %r' % (scheme,))
        if scheme == 'licorne':
            # the measured Q's weights (cached), the shifted reflectivity
            return self._resolve_licorne(
                Q, sigma, ideal if not shift else
                lambda q: ideal(np.abs(q - shift)))
        if shift:
            Q = np.abs(Q - shift)
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

    def _resolve_licorne(self, Q, sigma, ideal):
        """Licorne's convolution on the points Q themselves (see
        RESOLUTION 'licorne')."""
        if self.fronting_direction() is not None:
            raise ValueError("the 'licorne' resolution scheme has no "
                             "magnetic fronting (Licorne has none)")
        order = np.argsort(Q, kind='stable')
        Qs, ss = Q[order], sigma[order]
        mode = int(self.resolution.get('licorne_fun', 3))
        key = (Qs.tobytes(), ss.tobytes(), mode)
        cache = self._licorne_W
        W = cache.get(key)
        if W is None:
            W = licorne_kernel(Qs, ss, mode)
            if len(cache) >= LICORNE_W_CACHE:
                cache.pop(next(iter(cache)))
            cache[key] = W
        Rs = W @ ideal(Qs)
        R = np.empty_like(Rs)
        R[order] = Rs
        return R

    def _trace(self, Q, pairs, seq=None):
        """Non-magnetic fronting: Tr{rho r rho0 r^+} of every pair from ONE
        transfer-matrix evaluation, any backing; no background or resolution.
        Shape (nQ, len(pairs)).  seq: see build_transfer_matrix."""
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        f = self.fronting
        Q2 = Q**2 + 0j
        if self.q_in_fronting:
            Q2 = Q2 + 16*np.pi*(f.NSLD_real + 1j*f.NSLD_img)
        r = self._reflect(Q2, seq)[0]
        rho0 = np.stack([_density(P0) for P0, _ in pairs])
        rho = np.stack([np.eye(2) if P is None else _density(P)
                        for _, P in pairs])
        # = einsum('kij,qjl,klm,qim->qk', rho, r, rho0, r*), in a fixed
        # order so that a pair rounds the same in any batch
        A = rho[:, None] @ (r[None] @ rho0[:, None])
        return (A * r.conj()[None]).real.sum((-2, -1)).T

    def _lab_channels(self, Q, seq=None):
        """Magnetic fronting: the channels ++, +-, -+, -- along its M, no
        background or resolution.  Shape (nQ, 4).  seq: see
        build_transfer_matrix.

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
            r, qp, qm = self._reflect(Q2, seq)
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
