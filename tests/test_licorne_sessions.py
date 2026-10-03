"""Two full Licorne 1.4.2 session folders against their own rtheory*.dat
and Licorne's resolut, and the reduced chi^2 a FitProblem gives at the
session values (the reference numbers of benchmarks/README.md).

  session_A  HC-230104A_chi5_6 8Afinal: q.dat repeats Q = 0.0226139 and
             0.0617789 where the TOF angles are stitched
  session_B  Fit_thickness: an unfinished Licorne fit, a fixed benchmark

The folders are private (tests/data/private/, git-ignored, see
tests/data/licorne/README.md); the tests skip without them.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys
import warnings

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import licorne_reference as lic                                 # noqa: E402
from fit.problem import FitProblem                              # noqa: E402
from model.licorne_io import load_licorne_session               # noqa: E402
from model.licorne_resolution import kernel                     # noqa: E402
from model.polarisation import vectors_pair                     # noqa: E402

PRIVATE = HERE / 'data' / 'private'
# chi^2/N at the session values, measured when the sessions were added
CHI2 = {'session_A': 9.0249, 'session_B': 13.4354}
# session B's free parameters (its own fit flags are not a useful problem):
# thicknesses of layers 1-3, roughness of layers 1-3 and the substrate
VARY_B = [(1, 'thickness'), (2, 'thickness'), (3, 'thickness'),
          (1, 'roughness_sigma'), (2, 'roughness_sigma'),
          (3, 'roughness_sigma'), (4, 'roughness_sigma')]


def session(name):
    folder = PRIVATE / name
    if not (folder / 'rtheory1.dat').exists():
        pytest.skip('needs the private Licorne session %s, see '
                    'tests/data/licorne/README.md' % name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return folder, load_licorne_session(folder)


def problem(name, vary=None, **kw):
    """(FitProblem, session) of a session's own data, cost chi2; `vary`
    [(layer index, attr)] replaces the session's fit flags."""
    folder, s = session(name)
    st = s['stack']
    if vary:
        for layer in st.layers:
            for e in layer.fit.values():
                e['vary'] = False
        for i, attr in vary:
            st.layers[i].fit_entry(attr)['vary'] = True
    Q = np.loadtxt(folder / 'q.dat', comments='#', ndmin=2)[:, 0]
    data = []
    for c in s['channels']:
        E = np.loadtxt(c['path'], comments='#', ndmin=2)
        P = c['pol'][1]
        data.append(dict(Q=Q, R=E[:, 0], dR=E[:, 1], P0=np.array(c['pol'][0]),
                         P=np.array(P) if np.any(P) else None,
                         name='R' + c['channel'], norm=c['norm']))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return FitProblem(st, data, cost='chi2', **kw), s


@pytest.mark.parametrize('name', ['session_A', 'session_B'])
def test_session_reproduces_licorne(name):
    folder, s = session(name)
    st = s['stack']
    Q = np.loadtxt(folder / 'q.dat', comments='#', ndmin=2)[:, 0]
    sigma = st.resolution_sigma(Q)
    mode = int(st.resolution['licorne_fun'])
    R = np.random.default_rng(0).uniform(1e-6, 1.0, len(Q))
    ref = lic.resolut(R, Q, sigma, mode)
    assert np.max(np.abs(kernel(Q, sigma, mode) @ R - ref)
                  / np.abs(ref)) <= 1e-13
    assert s['theory']
    for t in s['theory']:
        c = [c for c in s['channels'] if c['channel'] == t['channel']][0]
        R = st.reflectivities(Q, [vectors_pair(*c['pol'])],
                              norms=[c['norm']])[:, 0]
        # rtheory*.dat: 6 significant figures
        assert np.max(np.abs(R / t['R'] - 1)) <= 1e-5, t['channel']
    p, _ = problem(name, VARY_B if name == 'session_B' else None)
    assert p(p.x0()) == pytest.approx(CHI2[name], abs=1e-4)


def test_session_A_has_repeated_Q():
    """The case B1 is about: A's q.dat repeats two points inside the grid."""
    folder, _ = session('session_A')
    Q = np.loadtxt(folder / 'q.dat', comments='#', ndmin=2)[:, 0]
    rep = Q[1:][np.diff(Q) == 0]
    assert np.allclose(rep, [0.0226139, 0.0617789], rtol=1e-5)


@pytest.mark.slow
def test_session_B_dream_defaults():
    """DREAM with sample()'s defaults (8 chains, snooker, tempered burn-in)
    on B's 7 parameters finds the minimum every start finds and converges
    (at the old defaults: chi^2/N 13.29, R-hat <= 1.09)."""
    from fit.dream.dream_fit import PosteriorProblem, sample
    p, _ = problem('session_B', VARY_B)
    pp = PosteriorProblem(p)
    r = sample(pp, n_gen=8000, seed=1, stop_when_converged=False)
    assert pp.chi2_per_point(r.best()[0]) <= 13.30
    assert r.rhat_history[-1][1].max() <= 1.1
