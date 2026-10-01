"""
Read a Licorne (MATLAB) model export into a Stack with the Licorne roughness
scheme (see model.roughness and model.stack ROUGHNESS SCHEME).

FILES -- a Licorne export directory holds
  parameters.m           lines  Layers(k).field=value;  Substrate.field=value;
                         (values: numbers, MATLAB complex literals such as
                         3.3529e-06-3e-08i, strings 'tanh', vectors [a,b,c])
  profile.dat            one row per layer, then the substrate:
                         Depth Thickness Re_NSLD Im_NSLD MSLD_rho MSLD_phi
                         MSLD_theta Roughness
  profile_sublayers.dat  Licorne's own slabs: Depth Thickness Re_NSLD Im_NSLD
                         MSLD_rho MSLD_phi MSLD_theta, the substrate last.
                         Depth 0 is the top of the first window, i.e.
                         z_nominal + l_a of the first interface.

parameters.m keeps only 5 significant figures of the NSLD (1.54e-006 against
1.53998e-006 in profile.dat), so with profile.dat given the thickness, NSLD
and MSLD come from it and only roughness_fun / roughness_nbound from
parameters.m.  Licorne has no fronting parameters: the fronting is vacuum.
Its substrate has no magnetisation.

ANGLES -- Licorne's msld = [rho, phi_deg, theta_deg] with
m = rho (sin theta cos phi, sin theta sin phi, cos theta) in Licorne's axes.
How these axes map onto this code's sample frame (z = film normal) is NOT
known (TODO: Licorne's x is in plane -- the manual's superlattice example
uses theta = 90, phi = 0/180 with P along x -- but whether its y or z is the
film normal cannot be told from the manual's figure).  So axis_map, an
orthogonal 3x3 matrix with m_sample = axis_map @ m_licorne, is required as
soon as two magnetic layers point differently.  Without it a collinear model
is loaded with every M along sample +x (theta = phi = 0): only the relative
directions are then meaningful, and the polarisation must be set along x.
"""

import re

import numpy as np

from model.stack import Layer, Stack

_LINE = re.compile(r'^\s*(Substrate|Layers\((\d+)\))\.(\w+)\s*=\s*(.*?);\s*$')
_FUNCS = {'tanh': 'tanh', 'erf': 'erf', 'erfc': 'erf', 'none': 'none'}


def _value(text):
    text = text.strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.startswith('['):
        return [_value(v) for v in re.split(r'[,\s]+', text.strip('[] '))
                if v]
    if text.endswith('i'):                         # MATLAB complex literal
        return complex(text[:-1] + 'j')
    return float(text)


def read_parameters(path):
    """{'substrate': {field: value}, 'layers': [{field: value}, ...]} from a
    Licorne parameters.m (layers in order, top first)."""
    sub, layers = {}, {}
    with open(path) as fh:
        for line in fh:
            m = _LINE.match(line)
            if not m:
                continue
            who, k, field, val = m.groups()
            d = sub if who == 'Substrate' else layers.setdefault(int(k), {})
            d[field] = _value(val)
    if sorted(layers) != list(range(1, len(layers) + 1)):
        raise ValueError('%s: layer numbers are not 1..N' % path)
    return {'substrate': sub, 'layers': [layers[k] for k in sorted(layers)]}


def read_table(path):
    """A Licorne .dat table (profile.dat, profile_sublayers.dat) as a 2D
    array, the '#' header skipped."""
    return np.atleast_2d(np.loadtxt(path, comments='#'))


def licorne_direction(phi_deg, theta_deg):
    """Unit vector of Licorne's (phi, theta) in Licorne's axes."""
    p, t = np.radians(phi_deg), np.radians(theta_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def sample_angles(u):
    """(MSLD_theta, MSLD_phi) in turns of a unit vector in the sample frame
    (model.stack ANGLES)."""
    return (float(np.arctan2(u[1], u[0]) / (2*np.pi)),
            float(np.arctan2(u[2], np.hypot(u[0], u[1])) / (2*np.pi)))


def _function(name):
    key = str(name).lower()
    if key == 'nc':
        raise NotImplementedError("Licorne's 'NC' roughness (Nevot-Croce "
                                  "inside Parratt) is not supported")
    if key not in _FUNCS:
        raise ValueError('unknown Licorne roughness_fun %r' % (name,))
    return _FUNCS[key]


def load_licorne_model(parameters_m_path, profile_dat_path=None,
                       axis_map=None):
    """Stack (roughness_scheme 'licorne', magnetic_smearing 'step') of a
    Licorne export: vacuum fronting, the layers, the substrate as backing.
    See FILES for which file each value comes from and ANGLES for axis_map.
    """
    par = read_parameters(parameters_m_path)
    lay_p = par['layers'] + [par['substrate']]          # interface owners
    n = len(lay_p)
    if profile_dat_path is not None:
        tab = read_table(profile_dat_path)
        if len(tab) != n:
            raise ValueError('%s has %d rows, parameters.m %d layers + '
                             'substrate' % (profile_dat_path, len(tab), n - 1))
        thick, re_n, im_n = tab[:, 1], tab[:, 2], tab[:, 3]
        rho, phi, theta = tab[:, 4], tab[:, 5], tab[:, 6]
    else:
        nsld = [complex(p['nsld']) for p in lay_p]
        msld = [p.get('msld', [0.0, 0.0, 0.0]) for p in lay_p]
        thick = np.array([p.get('thickness', 0.0) for p in lay_p])
        re_n = np.array([v.real for v in nsld])
        im_n = np.array([v.imag for v in nsld])
        rho, phi, theta = (np.array([float(m[i]) for m in msld])
                           for i in range(3))
    # the substrate has no magnetisation in Licorne
    rho = np.array(rho, dtype=float)
    rho[-1] = 0.0

    dirs = [licorne_direction(p, t) for p, t in zip(phi, theta)]
    mag = [i for i in range(n) if rho[i] != 0]
    collinear = all(np.allclose(dirs[i], dirs[mag[0]], atol=1e-12)
                    for i in mag)
    if axis_map is None and not collinear:
        raise ValueError('the magnetic layers point in different directions:'
                         ' axis_map (Licorne axes -> sample frame) is '
                         'required, see licorne_io ANGLES')
    if axis_map is not None:
        R = np.asarray(axis_map, dtype=float)
        if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3)):
            raise ValueError('axis_map must be an orthogonal 3x3 matrix')
        angles = [sample_angles(R @ u) for u in dirs]
    else:
        angles = [(0.0, 0.0)] * n

    layers = [Layer('vacuum', 0.0, 0.0, 0.0, 0.0, 0.0)]
    for i, p in enumerate(lay_p):
        last = i == n - 1
        layers.append(Layer(
            name='substrate' if last else 'Layer %d' % (i + 1),
            thickness=0.0 if last else float(thick[i]),
            NSLD_real=float(re_n[i]), NSLD_img=float(im_n[i]),
            MSLD_rho=float(rho[i]), MSLD_theta=angles[i][0],
            MSLD_phi=angles[i][1],
            roughness_sigma=float(p.get('roughness', 0.0)),
            roughness_model=_function(p.get('roughness_fun', 'none')),
            roughness_sublayer=int(p.get('roughness_nbound', 1))))
    return Stack(layers, roughness_scheme='licorne')
