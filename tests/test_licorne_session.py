"""Import of a saved Licorne session folder (model.licorne_io
load_licorne_session): channels, polarisation, fit bounds, background and the
TOF resolution script.

Run from the repo root: python -m pytest tests
"""
import pathlib
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
