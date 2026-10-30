"""fit.npe.symmetry: the numerical symmetry probe (plans/npe_plan_v1.md
Y1).  Fast, without torch or sbi.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import model.stack as S                                     # noqa: E402
from fit.joint import JointProblem                          # noqa: E402
from fit.npe.symmetry import candidates, format_probe, probe  # noqa: E402
from fit.problem import FitProblem                          # noqa: E402
from model.polarisation import channel_pair, in_plane       # noqa: E402

Q = np.linspace(0.008, 0.15, 120)
FULL = ('++', '--', '+-', '-+')
HALF = ('+', '-')


def fit(st, channels, n=in_plane(0.0), free_rho=('Fe',)):
    """FitProblem of st's own channels; theta of every magnetic layer
    free over a whole turn, rho of the `free_rho` layers too."""
    st.build_sublayers()
    pairs = [channel_pair(c, n) for c in channels]
    R = st.reflectivities(Q, pairs, warn=False)
    for l in st.layers[1:-1]:
        if l.MSLD_rho:
            l.fit['MSLD_theta'] = {'vary': True, 'min': -0.5, 'max': 0.5}
            if l.name in free_rho:
                l.fit['MSLD_rho'] = {'vary': True, 'min': 1e-6, 'max': 8e-6}
    return FitProblem(st, [dict(Q=Q, R=R[:, j], dR=0.02 * R[:, j] + 1e-8,
                                P0=P0, P=P, name='R' + c)
                           for j, ((P0, P), c) in enumerate(zip(pairs,
                                                                channels))],
                      'chi2')


def verdicts(p):
    return {r['name']: r['verdict'] for r in probe(p, n_draws=6, rng=0)}


def test_collinear_full_analysis():
    st = S.Stack([S.Layer('front', 0, 0, 0, 0, 0),
                  S.Layer('Fe', 80, 8e-6, 0, 5e-6, 0.0, 4.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])
    v = verdicts(fit(st, FULL))
    assert v == {'Fe: θ → −θ': 'exact', 'Fe: θ → θ + ½': 'broken',
                 'Fe: ρ → −ρ, θ → θ + ½': 'exact'}


def test_magnetic_fronting_breaks_the_mirror():
    st = S.Stack([S.Layer('front', 0, 1e-6, 0, 0.4e-6, 0.1),
                  S.Layer('Fe', 80, 8e-6, 0, 5e-6, 0.0, 4.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])
    v = verdicts(fit(st, FULL, st.fronting_direction()))
    assert v['Fe: θ → −θ'] == 'broken'
    assert v['Fe: ρ → −ρ, θ → θ + ½'] == 'exact'


def noncollinear():
    return S.Stack([S.Layer('front', 0, 0, 0, 0, 0),
                    S.Layer('Fe', 60, 8e-6, 0, 5e-6, 0.1, 4.0),
                    S.Layer('Cu', 30, 6.5e-6, 0, 0, 0, 3.0),
                    S.Layer('Co', 50, 2.3e-6, 0, 4e-6, -0.15, 4.0),
                    S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])


def test_half_polarised_noncollinear():
    v = verdicts(fit(noncollinear(), HALF))
    assert v['all layers: θ → −θ'] == 'exact'
    assert v['Fe: θ → −θ'] == v['Co: θ → −θ'] == 'broken'
    assert v['all layers: θ → θ + ½'] == 'broken'
    assert v['Fe: ρ → −ρ, θ → θ + ½'] == 'exact'
    assert 'Co: ρ → −ρ, θ → θ + ½' not in v      # Co's rho is fixed


@pytest.mark.filterwarnings('ignore:Fe. MSLD')
def test_elevation_sign_and_joint():
    st = noncollinear()
    st.layers[1].MSLD_phi = 0.05
    st.layers[1].fit['MSLD_phi'] = {'vary': True, 'min': -0.2, 'max': 0.2}
    p = fit(st, FULL)
    assert verdicts(p)['Fe: φ → −φ'] == 'exact'
    # a joint fit's own copies move with their layer
    jp = JointProblem([('A', p), ('B', fit(noncollinear(), FULL))],
                      {'B': {'Fe.MSLD_theta': 'own'}})
    labels = [n for n, _ in candidates(jp)]
    assert 'Fe: θ → −θ' in labels
    rows = probe(jp, n_draws=3, rng=1)
    assert {r['name']: r['verdict'] for r in rows}[
        'all layers: θ → −θ'] == 'exact'
    assert 'exact' in format_probe(rows)


def test_nothing_magnetic():
    st = S.Stack([S.Layer('front', 0, 0, 0, 0, 0),
                  S.Layer('Fe', 80, 8e-6, 0, 0, 0, 4.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 4.0)])
    st.layers[1].fit['thickness'] = {'vary': True, 'min': 60, 'max': 100}
    p = fit(st, HALF)
    assert probe(p) == [] and 'nothing' in format_probe([])
