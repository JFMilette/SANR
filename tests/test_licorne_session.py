"""Import of a saved Licorne session folder (model.licorne_io
load_licorne_session): channels, polarisation, fit bounds, background and the
TOF resolution script.

Run from the repo root: python -m pytest tests
"""
import pathlib
import re
import shutil
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from model.licorne_io import (licorne_axis_map, licorne_channel,  # noqa: E402
                              licorne_direction, load_licorne_model,
                              load_licorne_session, read_assignments,
                              read_parameters, read_resolution, read_table)
import licorne_reference as lic                                 # noqa: E402
from licorne_inputs import oracle_inputs                        # noqa: E402

DATA = HERE / 'data' / 'licorne'
FIX1 = DATA / 'fixture1'
V127 = DATA / 'v127_chi3_137'

# Licorne's TOF resolution template, as saved in IPTS 27232 S1 best_103
TOF_SCRIPT = """\
Theta1=0.006;Theta2=0.01;Theta3=0.017;

DTheta1=0.0003;DTheta2=0.0005;DTheta3=0.00050;

Q1=0.04;Q2=0.12;QP1=Q<Q1;QP2=(Q>=Q1)&(Q<=Q2);QP3=Q>Q2;

DLambda=0.005; Lambda=Q.*0; Sigma=Q.*0;

Lambda(QP1)=4*pi*sin(Theta1)./Q(QP1);

Sigma(QP1)=Q(QP1).*sqrt((DTheta1/Theta1)^2+(DLambda./Lambda(QP1)).^2);
"""


def session_folder(tmp_path, n=5, src=FIX1, k_max=2):
    """src's model (fixture1 by default) with n Q points, channels
    1..k_max and their theory."""
    d = tmp_path / 'best_1'
    d.mkdir(parents=True)
    for f in ('parameters.m', 'profile.dat'):
        shutil.copy(src / f, d / f)
    Q = np.linspace(0.01, 0.2, n)
    (d / 'q.dat').write_text('#"a.dat" 0.01 0.2 %d\n' % n
                             + ''.join('%g\n' % q for q in Q))
    for k in range(1, k_max + 1):
        (d / ('rexp%d.dat' % k)).write_text(
            '#"a.dat" 1e-6 1.0  1 0 0  0 0 0  52 2 3\n'
            + ''.join('%g %g\n' % (k / (i + 1), 0.01) for i in range(n)))
        (d / ('rtheory%d.dat' % k)).write_text(
            '#<theory> 2e-6 0.99  1 0 0  0 0 0\n'
            + ''.join('%g\n' % (1 / (i + 1)) for i in range(n)))
    (d / 'resolution.m').write_text(TOF_SCRIPT)
    return d


def test_read_assignments():
    v = read_assignments(FIX1 / 'parameters.m')
    assert v['Pol_num'] == 2 and v['Background'] == 1e-6
    assert v['Polarization'][:3] == [[1, 0, 0], [-1, 0, 0], [0, 0, -1]]
    assert v['PolAnOpt'] == ['Channel', 'Channel']
    assert v['LMFitOption2'] == np.inf
    assert 'thickness' not in v                     # Layers(1).thickness=...
    assert read_assignments(V127 / 'parameters.m')['Analysis'][1] == \
        [-0.98, 0, 0]


def test_read_resolution(tmp_path):
    p = tmp_path / 'resolution.m'
    p.write_text(TOF_SCRIPT)
    r = read_resolution(p)
    assert r['mode'] == 'tof' and r['enabled']
    assert r['tof_dlambda'] == 0.005
    assert r['tof_angles'] == [
        {'theta': 0.006, 'dtheta': 0.0003, 'qmax': 0.04},
        {'theta': 0.01, 'dtheta': 0.0005, 'qmax': 0.12},
        {'theta': 0.017, 'dtheta': 0.0005, 'qmax': None}]
    p.write_text('Sigma=Q.*0.01;\n')
    assert read_resolution(p) is None
    p.write_text(TOF_SCRIPT.replace('DTheta2=0.0005;', ''))
    assert read_resolution(p) is None


def test_licorne_channel():
    assert licorne_channel([1, 0, 0], [0, 0, 0]) == \
        ('+', [[1.0, 0, 0], [0, 0, 0]])
    assert licorne_channel([0.98, 0, 0], [-0.98, 0, 0]) == \
        ('+-', [[0.98, 0, 0], [-0.98, 0, 0]])
    # mapped by R; named by the signs along sample x
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    assert licorne_channel([0, 0, -1], [0, 0, 1], R) == \
        ('-+', [[-1.0, 0, 0], [1.0, 0, 0]])
    # a P at an angle keeps its other component
    name, pair = licorne_channel([0.6, 0.8, 0], [0, 0, 0])
    assert name == '+' and pair[0] == [0.6, 0.8, 0.0]
    with pytest.raises(ValueError):
        licorne_channel([1, 0, 0], [0, 0, 1])
    with pytest.raises(ValueError):
        licorne_channel([0, 0, 0], [0, 0, 0])


def test_load_session(tmp_path):
    d = session_folder(tmp_path)
    s = load_licorne_session(d)
    st = s['stack']
    assert s['name'] == 'best_1'
    assert st.roughness_scheme == 'licorne' and st.background == 1e-6
    assert st.resolution['enabled'] and st.resolution['mode'] == 'tof'
    # ResolutionFun = 3: Licorne's own convolution on the data grid
    assert st.resolution['scheme'] == 'licorne'
    assert st.resolution['licorne_fun'] == 3
    assert len(st.resolution['tof_angles']) == 3
    assert [c['channel'] for c in s['channels']] == ['+', '-']
    assert s['channels'][1]['pol'] == [[-1.0, 0, 0], [0, 0, 0]]
    assert s['channels'][0]['path'] == str(d / 'rexp1.dat')
    assert s['qrange'] == pytest.approx((0.01, 0.2))
    assert [t['channel'] for t in s['theory']] == ['+', '-']
    assert np.allclose(s['theory'][0]['R'], 1 / np.arange(1, 6))
    # fit bounds of parameters.m; the fronting keeps the defaults
    L2 = st.layers[2]
    assert L2.fit['thickness'] == {'vary': False, 'min': 30.0, 'max': 80.0}
    assert L2.fit['MSLD_rho'] == {'vary': False, 'min': 0.0, 'max': 3e-6}
    assert L2.fit['roughness_sigma']['max'] == 12.0
    assert 'thickness' not in st.backing.fit
    assert st.backing.fit['NSLD_real'] == {'vary': False, 'min': 5.5e-6,
                                           'max': 6.5e-6}
    assert st.fronting.fit == {}
    # fixture1: P along Licorne x, M at theta = 0 (Licorne z)
    assert any('perpendicular to every magnetisation' in n
               for n in s['notes'])
    assert not any('not parallel' in n for n in s['notes'])


def test_load_session_fit_flags_and_gaps(tmp_path):
    d = session_folder(tmp_path)
    par = d / 'parameters.m'
    par.write_text(par.read_text()
                   .replace('Layers(2).thickness_fit=0;',
                            'Layers(2).thickness_fit=1;')
                   .replace('Layers(2).msld_fit=[0,0,0];',
                            'Layers(2).msld_fit=[1,1,0];')
                   .replace('Pol_num=2;', 'Pol_num=3;'))
    (d / 'resolution.m').write_text('Sigma=Q.*0.01;\n')
    (d / 'rtheory2.dat').unlink()
    s = load_licorne_session(d)
    L2 = s['stack'].layers[2]
    assert L2.fit['thickness']['vary'] and L2.fit['MSLD_rho']['vary']
    assert not s['stack'].resolution['enabled']
    assert [t['channel'] for t in s['theory']] == ['+']
    notes = '\n'.join(s['notes'])
    assert 'angles are not carried over' in notes
    assert 'resolution is off' in notes
    # (0,0,-1) / (0,0,1): Licorne z, perpendicular to the first P
    assert 'Channel 3 left out' in notes


# -- Phase 3: axis map, normalisation, notes, MONO ---------------------------
def _axis_map(folder):
    return licorne_axis_map(read_parameters(folder / 'parameters.m'),
                            read_assignments(folder / 'parameters.m'),
                            read_table(folder / 'profile.dat'))


def test_axis_map_svd(tmp_path):
    """The rotation that puts the model's plane onto the film plane."""
    R = _axis_map(FIX1)                 # M along Licorne z, P along x
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)
    assert abs(np.linalg.det(R) - 1) < 1e-14
    assert np.allclose(R @ [1, 0, 0], [1, 0, 0], atol=1e-14)
    m = R @ licorne_direction(0, 0)
    assert abs(m[2]) < 1e-14 and abs(m[0]) < 1e-14
    for name in ('v127_chi3_137', 'v127_r2_6_508'):    # M in Licorne x-y
        R = _axis_map(DATA / name)
        assert np.allclose(R @ [1, 0, 0], [1, 0, 0], atol=1e-14)
        assert np.allclose(R, np.eye(3), atol=1e-14)
    # a third, non-coplanar direction: refused, or warned with axis_map
    p = tmp_path / 'parameters.m'
    p.write_text((FIX1 / 'parameters.m').read_text().replace(
        'Layers(4).msld=[3.09713e-007,0,0];',
        'Layers(4).msld=[3.09713e-007,90,90];'))
    with pytest.raises(ValueError, match='not coplanar'):
        load_licorne_model(p)
    with pytest.warns(UserWarning, match='out of the film plane'):
        load_licorne_model(p, axis_map=np.eye(3))


def test_small_tilt_dropped(tmp_path):
    """A 0.01 deg elevation left by a Licorne angle fit: imported in the
    film plane, with a note."""
    p = tmp_path / 'parameters.m'
    p.write_text((DATA / 'v127_chi3_137' / 'parameters.m').read_text()
                 .replace('Layers(3).msld=[2.60371e-06,9.83039,90];',
                          'Layers(3).msld=[2.60371e-06,9.83039,90.01];'))
    notes = []
    st = load_licorne_model(p, notes=notes)
    # the best plane shares the 0.01 deg between the vectors
    tilt = [float(m.group(1)) for m in
            (re.search(r'up to ([0-9.e-]+)°', n) for n in notes) if m]
    assert len(tilt) == 1 and 0.005 < tilt[0] <= 0.01
    assert all(l.MSLD_phi == 0 for l in st.layers)
    # the in-plane angles barely move
    ref = load_licorne_model(DATA / 'v127_chi3_137' / 'parameters.m')
    assert np.allclose([l.MSLD_theta for l in st.layers],
                       [l.MSLD_theta for l in ref.layers], atol=1e-4)


def test_fixture1_no_splitting():
    """Licorne computed R+ = R- for fixture 1 (M perpendicular to P); so
    does the import, channel by channel equal to the oracle."""
    st = load_licorne_model(FIX1 / 'parameters.m', FIX1 / 'profile.dat')
    st.licorne_renorm, st.licorne_outer = 'licorne', 'licorne'
    st.step_fallback = False
    st.build_sublayers()
    top = read_assignments(FIX1 / 'parameters.m')
    R = _axis_map(FIX1)
    layers, sub = oracle_inputs(FIX1)
    q = np.linspace(0.0106348, 0.24841, 204)
    out = []
    for pol, an, nf in zip(top['Polarization'][:2], top['Analysis'][:2],
                           top['Norm_factor']):
        name, (Pi, Pa) = licorne_channel(pol, an, R)
        ref = lic.licorne_R(q, np.zeros_like(q), layers, sub, pol, an,
                            norm=nf, background=1e-6)
        st.background = 1e-6
        out.append(st.reflectivities(q, [(np.array(Pi), None)],
                                     norms=[nf / 2])[:, 0])
        assert np.max(np.abs(out[-1] / ref - 1)) <= 1e-10
    assert np.max(np.abs(out[0] - out[1]) / out[0]) < 1e-12


def test_v127_end_to_end_unresolved():
    """The v127 channels (++, --, +-, analysed) equal the oracle; with the
    Licorne rounding of angles (LICORNE_DECIMALS) there is no 1e-9-turn
    floor left."""
    for name in ('v127_chi3_137', 'v127_r2_6_508'):
        d = DATA / name
        st = load_licorne_model(d / 'parameters.m', d / 'profile.dat')
        st.licorne_renorm, st.licorne_outer = 'licorne', 'licorne'
        st.step_fallback = False
        st.build_sublayers()
        top = read_assignments(d / 'parameters.m')
        layers, sub = oracle_inputs(d)
        q = np.linspace(0.0106348, 0.24841, 204)
        for pol, an, nf in zip(top['Polarization'][:3],
                               top['Analysis'][:3], top['Norm_factor']):
            _, (Pi, Pa) = licorne_channel(pol, an, _axis_map(d))
            ref = lic.licorne_R(q, np.zeros_like(q), layers, sub, pol, an,
                                norm=nf)
            R = st.reflectivities(q, [(np.array(Pi), np.array(Pa))],
                                  norms=[nf])[:, 0]
            assert np.max(np.abs(R / ref - 1)) <= 1e-10


def test_norm_factor(tmp_path):
    """Licorne's no-analyser channel is (R++ + R+-)/2 x Norm_factor:
    norm = Norm_factor / 2 without analyser, Norm_factor with one; a common
    norm becomes the stack's scale."""
    s = load_licorne_session(session_folder(tmp_path / 'a'))   # [2, 2, ...]
    assert s['stack'].scale == 1.0
    assert [c['norm'] for c in s['channels']] == [1.0, 1.0]
    d = session_folder(tmp_path / 'b', src=DATA / 'v127_r2_6_508', k_max=3)
    s = load_licorne_session(d)                        # [1.05] x 3, analysed
    assert s['stack'].scale == 1.05
    assert [c['norm'] for c in s['channels']] == [1.0] * 3
    assert [c['channel'] for c in s['channels']] == ['++', '--', '+-']
    d = session_folder(tmp_path / 'c')
    par = d / 'parameters.m'
    par.write_text(par.read_text().replace('Norm_factor=[2,2,1,1,1,1];',
                                           'Norm_factor=[2,1.8,1,1,1,1];'))
    s = load_licorne_session(d)
    assert s['stack'].scale == 1.0
    assert [c['norm'] for c in s['channels']] == [1.0, 0.9]


@pytest.mark.parametrize('edit, note', [
    (("Formalism='Supermatrix';", "Formalism='Parratt';"), 'Parratt'),
    (('Fraction=100;', 'Fraction=60;'), 'incoherent fraction'),
    (('Polarizer=0;', 'Polarizer=1;'), 'pol.dat'),
    (('Analyser=0;', 'Analyser=1;'), 'an.dat'),
    (('Q_mult=1;', 'Q_mult=1.01;'), 'Q_mult = 1.01'),
    (('Rexp_mult=[1,1,1,1,1,1];', 'Rexp_mult=[2,1,1,1,1,1];'),
     'Rexp_mult'),
    (("Layers(3).roughness_fun='tanh';", "Layers(3).roughness_fun='NC';"),
     "'NC' interfaces are sharp"),
])
def test_import_notes(tmp_path, edit, note):
    d = session_folder(tmp_path)
    par = d / 'parameters.m'
    text = par.read_text()
    assert edit[0] in text
    par.write_text(text.replace(*edit))
    s = load_licorne_session(d)
    assert any(note in n for n in s['notes'])
    if 'NC' in edit[1]:
        assert s['stack'].layers[3].roughness_model == 'none'
        par.write_text(par.read_text().replace(
            "Formalism='Supermatrix';", "Formalism='Parratt';"))
        with pytest.raises(NotImplementedError):
            load_licorne_session(d)


# Licorne's MONO resolution.m template (Licorne manual, resolution file)
MONO_SCRIPT = """\
% MONO: fixed wavelength, angle scan
Lambda=5; DLambda=0.01;
Theta=asin(Q*Lambda/4/pi);
DTheta=Q.*0; DTheta(Q > 0)=0.0007;
Sigma=Q.*sqrt((DTheta./Theta).^2+(DLambda/Lambda)^2);
"""


def test_mono_template(tmp_path):
    p = tmp_path / 'resolution.m'
    p.write_text(MONO_SCRIPT)
    r = read_resolution(p)
    assert r == pytest.approx({'enabled': True, 'mode': 'mono',
                               'wavelength': 5.0, 'dlambda_rel': 0.002,
                               'dtheta': 0.0007}, rel=1e-15)
    st = load_licorne_model(FIX1 / 'parameters.m')
    st.resolution.update(r)
    q = np.linspace(0.005, 0.3, 50)
    theta = np.arcsin(q * 5.0 / 4 / np.pi)
    expect = q * np.sqrt((0.0007 / theta)**2 + (0.01 / 5.0)**2)
    assert np.max(np.abs(st.resolution_sigma(q) / expect - 1)) < 1e-14
    p.write_text(MONO_SCRIPT.replace('DTheta(Q > 0)=0.0007;', ''))
    assert read_resolution(p) is None
    p.write_text(MONO_SCRIPT.replace('Theta=asin(Q*Lambda/4/pi);',
                                     'Theta=0.01;'))
    assert read_resolution(p) is None


def test_angles_from_parameters(tmp_path):
    """Exports before Licorne 1.2.7 write 0 for every angle in profile.dat;
    the angles come from parameters.m (rho and the NSLD from profile.dat)."""
    src = DATA / 'v127_chi3_137'
    d = tmp_path / 'old'
    d.mkdir()
    (d / 'parameters.m').write_text((src / 'parameters.m').read_text())
    rows = np.loadtxt(src / 'profile.dat', comments='#')
    rows[:, 5:7] = 0.0
    np.savetxt(d / 'profile.dat', rows, header='Depth Thickness Re_NSLD '
               'Im_NSLD MSLD_rho MSLD_phi MSLD_theta Roughness')
    a = load_licorne_model(src / 'parameters.m', src / 'profile.dat')
    b = load_licorne_model(d / 'parameters.m', d / 'profile.dat')
    assert a.to_dict() == b.to_dict()


TOF1_SCRIPT = """\
Theta=0.006;
DTheta=0.0004;
DLambda=0.01;
Lambda=4*pi*sin(Theta)./Q;
Sigma=Q.*sqrt((DTheta/Theta)^2+(DLambda./Lambda).^2);
"""


def test_one_angle_tof_and_default_fun(tmp_path):
    """Older Licorne's one-angle TOF resolution.m; without ResolutionFun
    (before 1.2.3) Licorne's convolution was mode 2."""
    d = session_folder(tmp_path)
    (d / 'resolution.m').write_text(TOF1_SCRIPT)
    par = d / 'parameters.m'
    par.write_text(par.read_text().replace('ResolutionFun=3;\n', ''))
    st = load_licorne_session(d)['stack']
    assert st.resolution['mode'] == 'tof' and st.resolution['enabled']
    assert st.resolution['tof_angles'] == [{'theta': 0.006,
                                            'dtheta': 0.0004, 'qmax': None}]
    assert st.resolution['tof_dlambda'] == 0.01
    assert st.resolution['scheme'] == 'licorne'
    assert st.resolution['licorne_fun'] == 2


def test_in_plane_angles_exact():
    """Licorne's theta = 90 (in plane) imports as an elevation of exactly 0,
    not cos(90 deg) ~ 6e-17 (it showed as ~1e-15 deg in the profile)."""
    for name in ('v127_chi3_137', 'v127_r2_6_508'):
        d = DATA / name
        st = load_licorne_model(d / 'parameters.m', d / 'profile.dat')
        assert all(l.MSLD_phi == 0.0 for l in st.layers)
    st = load_licorne_model(FIX1 / 'parameters.m', FIX1 / 'profile.dat')
    assert all(l.MSLD_phi == 0.0 for l in st.layers)
    assert {l.MSLD_theta for l in st.layers[1:-1]} == {0.25}


def test_axis_map_keeps_phi(tmp_path):
    '''The plane's normal may point either way (same reflectivity): the
    import picks the one that keeps the first magnetisation at +y.  With
    P along Licorne z and M at phi = 90, theta = 90 (along Licorne y),
    the in-plane angle is 90, not 270.'''
    p = tmp_path / 'parameters.m'
    p.write_text((FIX1 / 'parameters.m').read_text()
                 .replace('Polarization=[1,0,0; -1,0,0;',
                          'Polarization=[0,0,1; 0,0,-1;')
                 .replace('Layers(2).msld=[2.31522e-006,0,0];',
                          'Layers(2).msld=[2.31522e-006,90,90];')
                 .replace('Layers(4).msld=[3.09713e-007,0,0];',
                          'Layers(4).msld=[3.09713e-007,90,90];'))
    st = load_licorne_model(p)
    assert st.layers[2].MSLD_theta == 0.25 and st.layers[2].MSLD_phi == 0
    R = licorne_axis_map(read_parameters(p), read_assignments(p))
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])            # P -> sample x
    assert abs(np.linalg.det(R) - 1) < 1e-12


def test_generated_q_grid_noted(tmp_path):
    """q.dat generated in Licorne while rexp*.dat come from a measured
    file: Licorne paired the measured points with the generated Q row by
    row, so the import says their Q are wrong."""
    d = session_folder(tmp_path)
    assert not any('generated Q grid' in n
                   for n in load_licorne_session(d)['notes'])
    q = d / 'q.dat'
    q.write_text(q.read_text().replace('#"a.dat"', '#"<generated>"'))
    notes = load_licorne_session(d)['notes']
    assert sum('generated Q grid' in n and 'a.dat' in n for n in notes) == 2


# -- parameters.m settings Licorne applies on use (Q_mult, Rexp_mult, QLims,
#    ExpFit, ResolutionFun) ---------------------------------------------------
def edited_session(tmp_path, **lines):
    """session_folder with top-level lines of parameters.m replaced, e.g.
    Q_mult='1.01' (None: line removed)."""
    d = session_folder(tmp_path)
    p = d / 'parameters.m'
    text = p.read_text()
    for key, value in lines.items():
        text = re.sub(r'(?m)^%s=.*;\n' % key,
                      '' if value is None else '%s=%s;\n' % (key, value),
                      text)
    p.write_text(text)
    return d


def test_q_mult_rexp_mult(tmp_path):
    """q.dat and rexp are raw; Licorne multiplies on use."""
    d = edited_session(tmp_path, Q_mult='1.01',
                       Rexp_mult='[2,1,1,1,1,1]')
    s = load_licorne_session(d)
    Q = np.linspace(0.01, 0.2, 5)
    assert [(c['q_mult'], c['r_mult']) for c in s['channels']] == \
        [(1.01, 2.0), (1.01, 1.0)]
    assert np.array_equal(s['theory'][0]['Q'], np.array(
        ['%g' % q for q in Q], dtype=float) * 1.01)
    assert s['qrange'] == pytest.approx((0.0101, 0.202))    # QLims 0.22
    assert any('applied' in n for n in s['notes'])
    from tabs.experimental import file_ref, read_channel
    c = read_channel(file_ref(d / 'rexp1.dat', d), file_ref(d / 'q.dat', d),
                     1, str(d), q_mult=1.01, r_mult=2.0)
    raw = np.loadtxt(d / 'rexp1.dat', comments='#')
    order = np.argsort(np.loadtxt(d / 'q.dat', comments='#'))
    assert np.array_equal(c['R'], 2.0 * raw[order, 0])
    assert np.array_equal(c['dR'], 2.0 * raw[order, 1])
    assert np.allclose(c['Q'], 1.01 * np.loadtxt(d / 'q.dat',
                                                 comments='#')[order],
                       rtol=1e-15, atol=0)


def test_qlims_expfit_resolutionfun(tmp_path):
    d = edited_session(tmp_path, QLims='[0.05,0.15]', ExpFit='[1,0,1,1,1,1]',
                       ResolutionFun=None)
    s = load_licorne_session(d)
    assert s['qrange'] == (0.05, 0.15)
    assert [c['fit'] for c in s['channels']] == [True, False]
    assert any('not fitted (ExpFit = 0)' in n for n in s['notes'])
    assert s['stack'].resolution['licorne_fun'] == 2
    assert any('ResolutionFun 3' in n for n in s['notes'])


def test_import_into_the_tabs(tmp_path):
    """The import as the main window does it: mults, unfitted channel and
    fit range reach the Experimental and Simulation tabs, and survive a
    session round trip."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import main
    d = edited_session(tmp_path, Q_mult='1.01', Rexp_mult='[2,1,1,1,1,1]',
                       QLims='[0.05,0.15]', ExpFit='[1,0,1,1,1,1]')
    win = main.MainWindow()
    lic = load_licorne_session(d)
    assert win._restore(win._licorne_state(lic, str(d)), str(d)) == []
    exp, sim = win.experimental, win.simulation
    assert (sim.qmin.value(), sim.qmax.value()) == (0.05, 0.15)
    chans = exp.sets[0]['channels']
    raw = np.loadtxt(d / 'rexp1.dat', comments='#')
    assert np.allclose(np.sort(chans['+']['R']), np.sort(2.0 * raw[:, 0]))
    assert chans['+']['fit'] and not chans['-']['fit']
    inputs = exp._fit_inputs(exp.fit_report)
    assert [c['name'] for c in inputs[2]] == ['R+']
    state = exp.session_state(str(d))
    c = state['sets'][0]['channels']
    assert c['+']['q_mult'] == 1.01 and c['+']['r_mult'] == 2.0
    assert c['-']['fit'] is False
    win.close()
