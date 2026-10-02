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

from model.licorne_io import (licorne_channel, load_licorne_session,  # noqa: E402
                              read_assignments, read_resolution)

FIX1 = HERE / 'data' / 'licorne' / 'fixture1'
V127 = HERE / 'data' / 'licorne' / 'v127_chi3_137'

# Licorne's TOF resolution template, as saved in IPTS 27232 S1 best_103
TOF_SCRIPT = """\
Theta1=0.006;Theta2=0.01;Theta3=0.017;

DTheta1=0.0003;DTheta2=0.0005;DTheta3=0.00050;

Q1=0.04;Q2=0.12;QP1=Q<Q1;QP2=(Q>=Q1)&(Q<=Q2);QP3=Q>Q2;

DLambda=0.005; Lambda=Q.*0; Sigma=Q.*0;

Lambda(QP1)=4*pi*sin(Theta1)./Q(QP1);

Sigma(QP1)=Q(QP1).*sqrt((DTheta1/Theta1)^2+(DLambda./Lambda(QP1)).^2);
"""


def session_folder(tmp_path, n=5):
    """fixture1's model with n Q points, two channels and their theory."""
    d = tmp_path / 'best_1'
    d.mkdir()
    for f in ('parameters.m', 'profile.dat'):
        shutil.copy(FIX1 / f, d / f)
    Q = np.linspace(0.01, 0.2, n)
    (d / 'q.dat').write_text('#"a.dat" 0.01 0.2 %d\n' % n
                             + ''.join('%g\n' % q for q in Q))
    for k in (1, 2):
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
        ('+', [[1.0, 0, 0], [0, 0, 0]], 0)
    assert licorne_channel([0.98, 0, 0], [-0.98, 0, 0])[:2] == \
        ('+-', [[0.98, 0, 0], [-0.98, 0, 0]])
    name, pair, ax = licorne_channel([0, 0, -1], [0, 0, 1])
    assert (name, ax) == ('-+', 2) and pair == [[-1.0, 0, 0], [1.0, 0, 0]]
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
    assert any('not parallel' in n for n in s['notes'])


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
    assert 'Channel 3 left out' in notes            # (0,0,-1) / (0,0,1)
