"""
Read a Licorne (MATLAB) model export into a Stack with the Licorne roughness
scheme (see model.roughness and model.stack ROUGHNESS SCHEME).

FILES -- a Licorne export directory holds
  parameters.m           lines  Layers(k).field=value;  Substrate.field=value;
                         (values: numbers, MATLAB complex literals such as
                         3.3529e-06-3e-08i, strings 'tanh', vectors [a,b,c])
  profile.dat            one row per layer, then the substrate (its angle
                         columns are 0 or stale in exports before Licorne
                         1.2.7, so the angles come from parameters.m):
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
Licorne has no film normal: all three components of every M enter its
supermatrix, and only directions relative to each other and to P matter (a
common rotation of every M and P changes nothing).  So the import picks the
rotation R (m_sample = R m_Licorne, licorne_axis_map) that puts the plane of
the model's vectors onto the film plane, where SANR (HALPERIN) sees all of
them: the directions of every magnetic layer, every Polarization /
Analysis row in use, and every non-magnetic interior layer that shares a
rough interface with a magnetic one (its stored angle shows in its half of
that window, model.stack 'step' with step_fallback False).  Their SVD gives
the rank r:
  r <= 2  R = [e1; e2; n]: e1 the first polarisation (sample x), n the
          normal of their plane (smallest singular vector; for r = 1 any
          vector perpendicular to e1), e2 = n x e1; det R = +1
  r = 3   the vectors are not coplanar: Licorne lets a component along the
          film normal act, which specular PNR cannot see.  Refused (ValueError)
          unless axis_map is given, which then warns that the out-of-plane
          parts are lost.
An explicit axis_map (orthogonal 3x3) replaces R.

NORMALISATION -- Licorne returns (R++ + R+-)/2 for a channel without
analyser (spin_av with rho = 1: trace / 4), compensated by its per-channel
Norm_factor (2 by default); SANR's no-analyser channel is R++ + R+-.  Channel
k therefore gets the fixed norm Norm_factor(k) / 2 without analyser,
Norm_factor(k) with one (fit.problem DATA 'norm', Stack.reflectivities
norms); when every channel in use has the same norm it becomes the stack's
scale instead, and the norms stay 1.

NOT REPRODUCED -- reported in the session notes: Formalism 'Parratt' (SANR
computes the supermatrix), a second incoherent fraction (Fraction < 100,
Layers2), Q-dependent polariser / analyser efficiencies (Polarizer /
Analyser = 1, pol.dat / an.dat; |P| of Polarization / Analysis is used),
Q_mult / Rexp_mult other than 1 (not applied).  An 'NC' (Nevot-Croce)
interface is sharp in Licorne's supermatrix (expandrough.m skips it,
reflection_s never reads it) and is imported as 'none'; with Formalism
'Parratt' it is refused (NotImplementedError).
"""

import os
import re
import warnings

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


def _function(name, formalism='Supermatrix'):
    """SANR roughness_model of a Licorne roughness_fun.  'NC' is a sharp
    interface in the supermatrix (see NOT REPRODUCED)."""
    key = str(name).lower()
    if key == 'nc':
        if str(formalism).lower() == 'parratt':
            raise NotImplementedError("Licorne's 'NC' roughness (Nevot-Croce "
                                      "inside Parratt) is not supported")
        return 'none'
    if key not in _FUNCS:
        raise ValueError('unknown Licorne roughness_fun %r' % (name,))
    return _FUNCS[key]


def _channel_rows(top):
    """(Polarization, Analysis) rows of the Pol_num channels in use."""
    pols = [np.asarray(v, dtype=float) for v in top.get('Polarization', [])]
    ans = [np.asarray(v, dtype=float) for v in top.get('Analysis', [])]
    n = int(top.get('Pol_num', len(pols)))
    return pols[:n], ans[:n]


def _msld(par, profile=None):
    """(rho, phi_deg, theta_deg) arrays of the layers and the substrate:
    rho from profile.dat's table if given (6 figures), else parameters.m;
    the angles always from parameters.m (exports before Licorne 1.2.7 write
    0 or stale angles to profile.dat).  The substrate has no magnetisation
    in Licorne."""
    lay_p = par['layers'] + [par['substrate']]
    msld = [p.get('msld', [0.0, 0.0, 0.0]) for p in lay_p]
    rho, phi, theta = (np.array([float(m[i]) for m in msld])
                       for i in range(3))
    if profile is not None:
        rho = np.array(profile[:, 4], dtype=float)
    rho[-1] = 0.0
    return rho, phi, theta


def _rough(p):
    """A Licorne interface is rough with sigma_L > 0 and a function other
    than NC."""
    return float(p.get('roughness', 0.0)) > 0 and \
        str(p.get('roughness_fun', 'none')).lower() not in ('nc', 'none')


def model_vectors(par, top, profile=None):
    """Unit vectors (Licorne axes) that fix the model's orientation (see
    ANGLES): magnetic layers, a non-magnetic interior layer that shares a
    rough interface with a magnetic one, the channels' P; the first P
    (else the first M) is returned first."""
    rho, phi, theta = _msld(par, profile)
    lay_p = par['layers'] + [par['substrate']]
    n = len(lay_p) - 1                       # interior layers 0..n-1
    mag = [i for i in range(n) if rho[i] != 0]
    shared = [i for i in range(n) if rho[i] == 0 and (
        (i > 0 and rho[i - 1] != 0 and _rough(lay_p[i])) or
        (i + 1 < n and rho[i + 1] != 0 and _rough(lay_p[i + 1])))]
    m = [licorne_direction(phi[i], theta[i]) for i in mag + shared]
    pols, ans = _channel_rows(top)
    p = [v / np.linalg.norm(v) for v in pols + ans if np.any(v)]
    return p + m


def licorne_axis_map(par, top, profile=None):
    """R with m_sample = R m_Licorne (see ANGLES); ValueError for a model
    whose vectors are not coplanar."""
    vecs = model_vectors(par, top, profile)
    if not vecs:
        return np.eye(3)
    V = np.array(vecs)
    _, sv, Vt = np.linalg.svd(V)
    sv = np.r_[sv, np.zeros(3 - len(sv))]
    rank = int(np.sum(sv > 1e-9 * sv[0]))
    if rank == 3:
        raise ValueError('the Licorne model\'s magnetisations and '
                         'polarisations are not coplanar: Licorne lets a '
                         'component along the film normal act, which specular '
                         'reflectivity cannot see; give axis_map to import it '
                         'anyway (that component is then lost)')
    e1 = V[0]
    if rank == 2:
        n = Vt[2]
    else:                       # any normal: the axis least along e1
        a = np.eye(3)[np.argmin(np.abs(e1))]
        n = np.cross(a, e1)
        n /= np.linalg.norm(n)
    n = n * np.sign(n[np.argmax(np.abs(n))])     # a fixed sign: v127 -> I
    R = np.vstack([e1, np.cross(n, e1), n])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1) < 1e-12
    return R


def _out_of_plane(R, vecs):
    """Largest film-normal component of the mapped vectors."""
    return max((abs((R @ v)[2]) for v in vecs), default=0.0)


def load_licorne_model(parameters_m_path, profile_dat_path=None,
                       axis_map=None, notes=None):
    """Stack (Licorne-exact, Stack.set_licorne_exact) of a Licorne export:
    vacuum fronting, the layers, the substrate as backing.
    See FILES for which file each value comes from and ANGLES for the axes
    (licorne_axis_map unless axis_map is given).  What is not reproduced is
    appended to `notes` (a list) if given."""
    par = read_parameters(parameters_m_path)
    top = read_assignments(parameters_m_path)
    lay_p = par['layers'] + [par['substrate']]          # interface owners
    n = len(lay_p)
    tab = None
    if profile_dat_path is not None:
        tab = read_table(profile_dat_path)
        if len(tab) != n:
            raise ValueError('%s has %d rows, parameters.m %d layers + '
                             'substrate' % (profile_dat_path, len(tab), n - 1))
        thick, re_n, im_n = tab[:, 1], tab[:, 2], tab[:, 3]
    else:
        nsld = [complex(p['nsld']) for p in lay_p]
        thick = np.array([p.get('thickness', 0.0) for p in lay_p])
        re_n = np.array([v.real for v in nsld])
        im_n = np.array([v.imag for v in nsld])
    rho, phi, theta = _msld(par, tab)

    if axis_map is None:
        R = licorne_axis_map(par, top, tab)
    else:
        R = np.asarray(axis_map, dtype=float)
        if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3)):
            raise ValueError('axis_map must be an orthogonal 3x3 matrix')
        if _out_of_plane(R, model_vectors(par, top, tab)) > 1e-9:
            warnings.warn('axis_map leaves magnetisations or polarisations '
                          'out of the film plane: that component is not seen '
                          'here (HALPERIN), Licorne used it', stacklevel=2)
    angles = [sample_angles(R @ licorne_direction(p, t))
              for p, t in zip(phi, theta)]

    formalism = top.get('Formalism', 'Supermatrix')
    funs = [str(p.get('roughness_fun', 'none')) for p in lay_p]
    if notes is not None and any(f.lower() == 'nc' for f in funs):
        notes.append("'NC' interfaces are sharp in Licorne's supermatrix: "
                     "imported as 'none'.")
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
            roughness_model=_function(funs[i], formalism),
            roughness_sublayer=int(p.get('roughness_nbound', 1))))
    st = Stack(layers, roughness_scheme='licorne')
    st.set_licorne_exact()
    return st


def read_resolution(path):
    """Stack.resolution entries of a resolution.m written in one of
    Licorne's templates, None for any other script:
      TOF   Theta<i>, DTheta<i> (rad) per angle, Q<i> the upper Q of angle i
            (the last one open-ended) and DLambda (A) -> mode 'tof'
      TOF, one angle (older Licorne)  Theta, DTheta (rad), DLambda (A),
            Lambda=4*pi*sin(Theta)./Q and
            Sigma=Q.*sqrt((DTheta/Theta)^2+(DLambda./Lambda).^2)
            -> mode 'tof' with that one angle
      MONO  Lambda, DLambda (A), DTheta (rad, as DTheta=... or
            DTheta(Q > 0)=...), Theta = asin(Q Lambda / 4 pi) and
            Sigma=Q.*sqrt((DTheta./Theta).^2+(DLambda/Lambda)^2)
            -> mode 'mono'"""
    v = read_assignments(path)
    idx = sorted(int(k[5:]) for k in v if re.fullmatch(r'Theta\d+', k))
    if not idx:
        return _read_tof1(path, v) or _read_mono(path, v)
    if idx != list(range(1, len(idx) + 1)) or \
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


_MONO_SIGMA = 'sigma=q.*sqrt((dtheta./theta).^2+(dlambda/lambda)^2);'
_MONO_THETA = re.compile(r'theta=asin\((q\.?\*lambda|lambda\.?\*q)'
                         r'(/4/pi|/\(4\*pi\)|\./4\./pi)\);')
_DTHETA = re.compile(r'(?:^|;)\s*DTheta\s*(?:\(\s*Q\s*>\s*0\s*\))?\s*=\s*'
                     r'([-+0-9.eE]+)\s*;', re.M)


_TOF1_LAMBDA = 'lambda=4*pi*sin(theta)./q;'
_TOF1_SIGMA = 'sigma=q.*sqrt((dtheta/theta)^2+(dlambda./lambda).^2);'


def _read_tof1(path, v):
    """One-angle TOF template of read_resolution, None if not it."""
    with open(path) as fh:
        flat = re.sub(r'\s+', '', re.sub(r'%.*', '', fh.read())).lower()
    if _TOF1_LAMBDA not in flat or _TOF1_SIGMA not in flat or \
            not all(isinstance(v.get(k), float)
                    for k in ('Theta', 'DTheta', 'DLambda')):
        return None
    return {'enabled': True, 'mode': 'tof', 'tof_dlambda': v['DLambda'],
            'tof_angles': [{'theta': v['Theta'], 'dtheta': v['DTheta'],
                            'qmax': None}]}


def _read_mono(path, v):
    """MONO template of read_resolution, None if the script is not it."""
    with open(path) as fh:
        text = re.sub(r'%.*', '', fh.read())
    flat = re.sub(r'\s+', '', text).lower()
    dth = _DTHETA.findall(text)
    lam, dlam = v.get('Lambda'), v.get('DLambda')
    if _MONO_SIGMA not in flat or not _MONO_THETA.search(flat) or \
            len(dth) != 1 or not isinstance(lam, float) or \
            not isinstance(dlam, float) or lam <= 0:
        return None
    return {'enabled': True, 'mode': 'mono', 'wavelength': lam,
            'dlambda_rel': dlam / lam, 'dtheta': float(dth[0])}


def licorne_channel(pol, an, R=np.eye(3)):
    """(name, [Pi, Pa]) of a Licorne channel from its Polarization and
    Analysis rows mapped into the sample frame by R (see ANGLES); named by
    the signs along sample x (the first polarisation).  Pa = (0, 0, 0) is
    no analyser."""
    pol, an = np.asarray(pol, dtype=float), np.asarray(an, dtype=float)
    if not np.any(pol):
        raise ValueError('Polarization %s is zero' % pol.tolist())
    Pi, Pa = R @ pol, R @ an
    Pi, Pa = (np.where(np.abs(v) < 1e-12, 0.0, v) for v in (Pi, Pa))
    if Pi[0] == 0 or (np.any(Pa) and Pa[0] == 0):
        raise ValueError('Polarization %s / Analysis %s have no component '
                         'along the first polarisation'
                         % (pol.tolist(), an.tolist()))
    name = ('+' if Pi[0] > 0 else '-') + \
        ('' if not np.any(Pa) else '+' if Pa[0] > 0 else '-')
    return name, [[float(c) for c in Pi], [float(c) for c in Pa]]


def channel_norms(top, pols, ans):
    """Fixed norm of each channel in use (see NORMALISATION)."""
    nf = top.get('Norm_factor', [])
    nf = [float(x) for x in (nf if isinstance(nf, list) else [nf])]
    return [(nf[k] if k < len(nf) else 1.0) / (1.0 if np.any(a) else 2.0)
            for k, a in enumerate(ans[:len(pols)])]


def import_notes(top):
    """What Licorne computed that SANR does not reproduce (NOT
    REPRODUCED)."""
    notes = []
    if str(top.get('Formalism', 'Supermatrix')).lower() == 'parratt':
        notes.append("Licorne used the Parratt formalism (M along z only, no "
                     "spin flip, Nevot-Croce on 'NC' interfaces); SANR "
                     "computes the supermatrix.")
    if float(top.get('Fraction', 100)) < 100:
        notes.append('Second incoherent fraction (Fraction = %g %%, Layers2) '
                     'ignored.' % float(top['Fraction']))
    if top.get('Polarizer') == 1 or top.get('Analyser') == 1:
        notes.append('Q-dependent polariser / analyser efficiency '
                     '(pol.dat / an.dat) ignored; |P| from Polarization / '
                     'Analysis is used.')
    pols, _ = _channel_rows(top)
    rm = top.get('Rexp_mult', [])
    rm = rm if isinstance(rm, list) else [rm]
    qm = top.get('Q_mult', 1.0)
    if qm != 1 or any(float(x) != 1 for x in rm[:len(pols)]):
        notes.append('Q_mult = %g, Rexp_mult = %s are not applied.'
                     % (qm, [float(x) for x in rm[:len(pols)]]))
    return notes


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
                the background, the scale (a common Norm_factor, see
                NORMALISATION) and, if resolution.m is one of Licorne's
                templates, the resolution (Licorne's own convolution,
                mode ResolutionFun, 2 when the file has none)
      name      the folder's name
      q_path    q.dat
      channels  [{'channel', 'path' (rexp<k>.dat), 'pol' [Pi, Pa],
                  'norm', 'k'}]
      theory    [{'channel', 'path', 'Q', 'R', 'pol'}] from rtheory<k>.dat
      qrange    (min, max) of q.dat
      axis_map  R, m_sample = R m_Licorne (see ANGLES)
      notes     what was not carried over, as text
    Only the first Pol_num rows of Polarization / Analysis are used."""
    folder = os.path.abspath(folder)
    par_path = os.path.join(folder, 'parameters.m')
    prof = os.path.join(folder, 'profile.dat')
    prof = prof if os.path.isfile(prof) else None
    par = read_parameters(par_path)
    top = read_assignments(par_path)
    if axis_map is None:
        R = licorne_axis_map(par, top,
                             None if prof is None else read_table(prof))
    else:
        R = np.asarray(axis_map, dtype=float)
    notes = import_notes(top)
    stack = load_licorne_model(par_path, prof, R, notes)
    _set_fit(stack, par)
    if any(any(p.get('msld_fit', [0, 0, 0])[1:]) for p in par['layers']):
        notes.append('Fitted magnetisation angles are not carried over.')
    stack.background = float(top.get('Background', 0.0))

    res_path = os.path.join(folder, 'resolution.m')
    res = read_resolution(res_path) if os.path.isfile(res_path) else None
    if res is not None:
        stack.resolution.update(res)
        # no ResolutionFun (Licorne before 1.2.3): its convolution was mode
        # 2, as rtheory*.dat of 1.0.0, 1.1.0 and 1.2.2 sessions show
        fun = top.get('ResolutionFun', 2)
        if fun in (1, 2, 3):
            stack.resolution.update(scheme='licorne', licorne_fun=int(fun))
    else:
        stack.resolution['enabled'] = False
        notes.append('resolution.m is not one of Licorne\'s templates (TOF, '
                     'one-angle TOF, MONO): the resolution is off.')

    q_path = os.path.join(folder, 'q.dat')
    Q = _curve(q_path)[:, 0]
    pols, ans = _channel_rows(top)
    norms = channel_norms(top, pols, ans)
    channels, theory, seen = [], [], set()
    for k in range(1, int(top.get('Pol_num', len(pols))) + 1):
        rexp = os.path.join(folder, 'rexp%d.dat' % k)
        try:
            ch, pol = licorne_channel(pols[k - 1], ans[k - 1], R)
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
        channels.append({'channel': ch, 'path': rexp, 'pol': pol,
                         'norm': norms[k - 1], 'k': k})
        th = os.path.join(folder, 'rtheory%d.dat' % k)
        if os.path.isfile(th):
            Rt = _curve(th)[:, 0]
            if len(Rt) == len(Q):
                theory.append({'channel': ch, 'path': th, 'Q': Q.copy(),
                               'R': Rt, 'pol': pol})
    # one common norm is the stack's scale
    if channels and len({c['norm'] for c in channels}) == 1:
        stack.scale = channels[0]['norm']
        for c in channels:
            c['norm'] = 1.0
    # no splitting possible: P perpendicular to every magnetisation
    m = [stack.inplane_rho(l) != 0 for l in stack.layers[1:-1]]
    dirs = [np.array([np.cos(2*np.pi*l.MSLD_theta),
                      np.sin(2*np.pi*l.MSLD_theta), 0.0])
            for l, on in zip(stack.layers[1:-1], m) if on]
    P = [np.asarray(c['pol'][0]) for c in channels]
    if dirs and P and all(abs(u @ p) < 1e-9 * np.linalg.norm(p)
                          for u in dirs for p in P):
        notes.append('In this Licorne model the polarisation is '
                     'perpendicular to every magnetisation: R+ = R- (no spin '
                     'splitting) in Licorne and here.')
    return {'stack': stack, 'name': os.path.basename(folder),
            'q_path': q_path, 'channels': channels, 'theory': theory,
            'qrange': (float(Q.min()), float(Q.max())), 'axis_map': R,
            'notes': notes}
