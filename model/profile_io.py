"""
Read a layer-profile table (such as a profile.dat) into Stack layers.

FILE -- '#' lines are skipped; one row per layer from the top, the substrate
last, with the columns
  Depth Thickness Re_NSLD Im_NSLD MSLD_rho MSLD_phi MSLD_theta Roughness
(SLDs in A^-2, lengths in Angstrom).  Depth, MSLD_phi and MSLD_theta are not
read: every magnetisation lies along x.  Roughness is the sigma of the
interface at the top of the row's layer (model.stack OWNERSHIP), a tanh
profile of PROFILE_SUBLAYERS sublayers.  A vacuum fronting is put on top.
"""

import numpy as np

from model.stack import Layer

PROFILE_SUBLAYERS = 16


def load_profile(path):
    """[fronting, layers..., substrate] of the profile table at path."""
    tab = np.loadtxt(path, comments='#', ndmin=2)
    if tab.shape[1] < 8:
        raise ValueError('%d columns, a profile has 8 (Depth Thickness '
                         'Re_NSLD Im_NSLD MSLD_rho MSLD_phi MSLD_theta '
                         'Roughness)' % tab.shape[1])
    layers = [Layer('vacuum')]
    for i, r in enumerate(tab):
        last = i == len(tab) - 1
        layers.append(Layer(
            name='substrate' if last else 'Layer %d' % (i + 1),
            thickness=0.0 if last else float(r[1]),
            NSLD_real=float(r[2]), NSLD_img=float(r[3]),
            MSLD_rho=float(r[4]), roughness_sigma=float(r[7]),
            roughness_model='tanh', roughness_sublayer=PROFILE_SUBLAYERS))
    return layers
