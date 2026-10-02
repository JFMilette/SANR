"""End to end against Licorne: an imported session's Stack.reflectivities
against the oracle's licorne_R (tests/licorne_reference.py: expandrough ->
supermatrix -> spin_av -> resolut -> Norm_factor + Background), with
Licorne's own vectors, Norm_factor, Background and ResolutionFun 3 on
fixture 1's q.dat grid with the best_103 TOF resolution; and, when a full
Licorne session folder is available, against its rtheory*.dat.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import licorne_reference as lic                                 # noqa: E402
from licorne_inputs import oracle_inputs                        # noqa: E402
from model.licorne_io import (load_licorne_session,             # noqa: E402
                              read_assignments)
from model.polarisation import vectors_pair                     # noqa: E402
from test_licorne_session import TOF_SCRIPT, session_folder     # noqa: E402

DATA = HERE / 'data' / 'licorne'
FIXTURES = ['fixture1', 'v127_chi3_137', 'v127_r2_6_508']
# a full session folder exported by Licorne (q.dat, rexp*, rtheory*,
# resolution.m); see tests/data/licorne/README.md
SESSION = DATA / 'session1'


def imported(tmp_path, name):
    """load_licorne_session of fixture `name` on fixture 1's q.dat grid
    (204 points) with the best_103 TOF resolution.m."""
    top = read_assignments(DATA / name / 'parameters.m')
    d = session_folder(tmp_path, n=204, src=DATA / name,
                       k_max=int(top['Pol_num']))
    Q = np.linspace(0.0106348, 0.24841, 204)
    (d / 'q.dat').write_text('#"a.dat"\n' + ''.join('%.17g\n' % q
                                                    for q in Q))
    (d / 'resolution.m').write_text(TOF_SCRIPT + """
Lambda(QP2)=4*pi*sin(Theta2)./Q(QP2);
Lambda(QP3)=4*pi*sin(Theta3)./Q(QP3);
Sigma(QP2)=Q(QP2).*sqrt((DTheta2/Theta2)^2+(DLambda./Lambda(QP2)).^2);
Sigma(QP3)=Q(QP3).*sqrt((DTheta3/Theta3)^2+(DLambda./Lambda(QP3)).^2);
""")
    return load_licorne_session(d), top, Q


@pytest.mark.parametrize('name', FIXTURES)
def test_session_against_oracle(tmp_path, name):
    """Every channel of the imported session equals Licorne's computation
    (resolution, Norm_factor and Background included) to round-off."""
    s, top, Q = imported(tmp_path, name)
    st = s['stack']
    assert st.is_licorne_exact()
    assert st.resolution['scheme'] == 'licorne'
    assert st.resolution['licorne_fun'] == 3
    sigma = st.resolution_sigma(Q)
    layers, sub = oracle_inputs(DATA / name)
    assert len(s['channels']) == int(top['Pol_num'])
    for c in s['channels']:
        k = c['k'] - 1
        ref = lic.licorne_R(Q, sigma, layers, sub, top['Polarization'][k],
                            top['Analysis'][k], norm=top['Norm_factor'][k],
                            background=top['Background'], res_mode=3)
        R = st.reflectivities(Q, [vectors_pair(*c['pol'])],
                              norms=[c['norm']])[:, 0]
        assert np.max(np.abs(R / ref - 1)) <= 1e-10, c['channel']


def test_session_rtheory():
    """A Licorne session folder: SANR's curves on q.dat equal rtheory*.dat
    to the file's printed precision."""
    if not (SESSION / 'rtheory1.dat').exists():
        pytest.skip('needs a Licorne session export, see '
                    'tests/data/licorne/README.md')
    s = load_licorne_session(SESSION)
    st = s['stack']
    Q = np.loadtxt(SESSION / 'q.dat', comments='#', ndmin=2)[:, 0]
    for t in s['theory']:
        c = [c for c in s['channels'] if c['channel'] == t['channel']][0]
        R = st.reflectivities(Q, [vectors_pair(*c['pol'])],
                              norms=[c['norm']])[:, 0]
        # rtheory*.dat: 6 significant figures
        assert np.max(np.abs(R / t['R'] - 1)) <= 1e-5, t['channel']
