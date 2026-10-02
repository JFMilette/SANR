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
A saved Licorne session folder (load_licorne_session) also holds
  q.dat                  one Q per line, the grid of every channel
  rexp<k>.dat            measured channel k = 1..Pol_num: R dR per line;
                         header #"source file" ... with Licorne's own numbers
  rtheory<k>.dat         Licorne's computed channel k: R per line
  resolution.m           MATLAB script giving Sigma(Q); only Licorne's TOF
                         template (Theta<i>, DTheta<i>, Q<i>, DLambda) is read
and in parameters.m the top-level Polarization / Analysis (one row per
channel, Licorne axes, length = efficiency), Pol_num and Background.

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
load_licorne_session does so: a channel's Polarization / Analysis must lie
along one Licorne axis and becomes the same signed length along sample x.
"""

import os
import re

import numpy as np

from model.stack import Layer, Stack

_LINE = re.compile(r'^\s*(Substrate|Layers\((\d+)\))\.(\w+)\s*=\s*(.*?);\s*$')
# a top-level  Name=value  statement, at the start of a line or after a ';'
_STATEMENT = re.compile(r"(?:^|(?<=;))[ \t]*([A-Za-z]\w*)[ \t]*=(?!=)[ \t]*"
                        r"(\[[^\]]*\]|\{[^}]*\}|'[^']*'|[^;\n]*)", re.M)
_FUNCS = {'tanh': 'tanh', 'erf': 'erf', 'erfc': 'erf', 'none': 'none'}
# Layer attribute -> Licorne field (with _fit, _min, _max) and the index of
# the value in a vector field
_FIT_FIELDS = (('thickness', 'thickness', None), ('NSLD_real', 're_nsld', None),
               ('NSLD_img', 'im_nsld', None), ('MSLD_rho', 'msld', 0),
               ('roughness_sigma', 'roughness', None))


def _value(text):
    text = text.strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.startswith('{'):                       # cell array of strings
        return [_value(v) for v in text.strip('{} ').split(',') if v.strip()]
    if text.startswith('['):
        body = text.strip('[] ')
        if ';' in body:                            # matrix, one list per row
            return [_value('[%s]' % r) for r in body.split(';') if r.strip()]
        return [_value(v) for v in re.split(r'[,\s]+', body) if v]
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


def read_assignments(path):
    """{name: value} of the top-level  Name=value;  statements of a MATLAB
    file (parameters.m, resolution.m); '%' comments and statements whose
    value is an expression are skipped."""
    with open(path) as fh:
        text = re.sub(r'%.*', '', fh.read())
    out = {}
    for name, val in _STATEMENT.findall(text):
        try:
            out[name] = _value(val)
        except ValueError:
            pass
    return out


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


def read_resolution(path):
    """Stack.resolution entries ('tof' mode) of a resolution.m written in
    Licorne's TOF template: Theta<i>, DTheta<i> (rad) per angle, Q<i> the
    upper Q of angle i (the last one open-ended) and DLambda (A).  None if the
    script is not that template."""
    v = read_assignments(path)
    idx = sorted(int(k[5:]) for k in v if re.fullmatch(r'Theta\d+', k))
    if not idx or idx != list(range(1, len(idx) + 1)) or \
            not isinstance(v.get('DLambda'), float):
        return None
    angles = []
    for i in idx:
        qmax = v.get('Q%d' % i) if i < len(idx) else None
        if not isinstance(v.get('DTheta%d' % i), float) or \
                (i < len(idx) and not isinstance(qmax, float)):
            return None
        angles.append({'theta': v['Theta%d' % i],
                       'dtheta': v['DTheta%d' % i], 'qmax': qmax})
    return {'enabled': True, 'mode': 'tof', 'tof_dlambda': v['DLambda'],
            'tof_angles': angles}


def licorne_channel(pol, an):
    """(name, [Pi, Pa], axis) of a Licorne channel from its Polarization
    and Analysis rows: both along one Licorne axis (index `axis`), which
    becomes sample x (ANGLES).  Pa = (0, 0, 0) is no analyser."""
    pol, an = np.asarray(pol, dtype=float), np.asarray(an, dtype=float)
    axes = set(np.flatnonzero(pol)) | set(np.flatnonzero(an))
    if not np.any(pol) or len(axes) != 1:
        raise ValueError('Polarization %s / Analysis %s do not lie along one '
                         'axis' % (pol.tolist(), an.tolist()))
    ax = axes.pop()
    p, a = float(pol[ax]), float(an[ax])
    name = ('+' if p > 0 else '-') + ('' if a == 0 else '+' if a > 0 else '-')
    return name, [[p, 0.0, 0.0], [a, 0.0, 0.0]], int(ax)


def _set_fit(stack, par):
    """Licorne's <field>_fit / _min / _max as the layers' fit settings."""
    lay_p = par['layers'] + [par['substrate']]
    for i, (layer, p) in enumerate(zip(stack.layers[1:], lay_p), start=1):
        for attr, key, j in _FIT_FIELDS:
            vals = [p.get(key + s) for s in ('_fit', '_min', '_max')]
            if None in vals or not stack.fit_allowed(i, attr):
                continue
            if j is not None:
                vals = [v[j] for v in vals]
            layer.fit[attr] = {'vary': bool(vals[0]), 'min': float(vals[1]),
                               'max': float(vals[2])}


def _curve(path):
    """Values of a rexp / rtheory file: (n, ncols) array, header skipped."""
    return np.loadtxt(path, comments='#', ndmin=2)


def load_licorne_session(folder, axis_map=None):
    """A saved Licorne session folder (FILES) as a dict:
      stack     load_licorne_model of it, with the fit flags and bounds,
                the background and, if resolution.m is Licorne's TOF
                template, the resolution
      name      the folder's name
      q_path    q.dat
      channels  [{'channel', 'path' (rexp<k>.dat), 'pol' [Pi, Pa]}]
      theory    [{'channel', 'path', 'Q', 'R', 'pol'}] from rtheory<k>.dat
      qrange    (min, max) of q.dat
      notes     what was not carried over, as text
    Only the first Pol_num rows of Polarization / Analysis are used."""
    folder = os.path.abspath(folder)
    par_path = os.path.join(folder, 'parameters.m')
    prof = os.path.join(folder, 'profile.dat')
    stack = load_licorne_model(par_path,
                               prof if os.path.isfile(prof) else None,
                               axis_map)
    par = read_parameters(par_path)
    top = read_assignments(par_path)
    notes = []
    _set_fit(stack, par)
    if any(any(p.get('msld_fit', [0, 0, 0])[1:]) for p in par['layers']):
        notes.append('Fitted magnetisation angles are not carried over.')
    stack.background = float(top.get('Background', 0.0))

    res_path = os.path.join(folder, 'resolution.m')
    res = read_resolution(res_path) if os.path.isfile(res_path) else None
    if res is not None:
        stack.resolution.update(res)
    else:
        stack.resolution['enabled'] = False
        notes.append('resolution.m is not Licorne\'s TOF template: the '
                     'resolution is off.')

    q_path = os.path.join(folder, 'q.dat')
    Q = _curve(q_path)[:, 0]
    pols = top.get('Polarization', [])
    ans = top.get('Analysis', [])
    n = int(top.get('Pol_num', len(pols)))
    # Licorne direction of every magnetic layer, to check it lies along P
    mag = [licorne_direction(m[1], m[2]) for m in
           (p.get('msld', [0, 0, 0]) for p in par['layers']) if m[0] != 0]
    channels, theory, seen, skew = [], [], set(), set()
    for k in range(1, n + 1):
        rexp = os.path.join(folder, 'rexp%d.dat' % k)
        try:
            ch, pol, ax = licorne_channel(pols[k - 1], ans[k - 1])
        except (IndexError, ValueError) as exc:
            notes.append('Channel %d left out: %s' % (k, exc))
            continue
        if ch in seen:
            notes.append('Channel %d left out: R%s is already channel %d.'
                         % (k, ch, [c['k'] for c in channels
                                    if c['channel'] == ch][0]))
            continue
        if not os.path.isfile(rexp):
            notes.append('Channel %d left out: no rexp%d.dat.' % (k, k))
            continue
        seen.add(ch)
        channels.append({'channel': ch, 'path': rexp, 'pol': pol, 'k': k})
        if any(abs(abs(u[ax]) - 1) > 1e-9 for u in mag):
            skew.add('xyz'[ax])
        th = os.path.join(folder, 'rtheory%d.dat' % k)
        if os.path.isfile(th):
            R = _curve(th)[:, 0]
            if len(R) == len(Q):
                theory.append({'channel': ch, 'path': th, 'Q': Q.copy(),
                               'R': R, 'pol': pol})
    if skew and axis_map is None:
        notes.append('In Licorne the polarisation (along its %s) is not '
                     'parallel to the magnetisation (msld phi, theta); here '
                     'both lie along sample x, so the reflectivity can differ '
                     'from rtheory*.dat.' % ', '.join(sorted(skew)))
    return {'stack': stack, 'name': os.path.basename(folder),
            'q_path': q_path, 'channels': channels, 'theory': theory,
            'qrange': (float(Q.min()), float(Q.max())), 'notes': notes}
