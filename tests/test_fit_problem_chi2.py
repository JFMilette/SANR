"""fit.problem chi2_per_point: the data-only chi^2/N beside the cost (cost
minus the priors' share), for one dataset and a joint fit, and the same
number in DREAM (plan chi2_column_plan_v1, T1 and T5).

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fit.dream.dream_fit import PosteriorProblem              # noqa: E402
from fit.joint import JointProblem                             # noqa: E402
from fit.problem import FitProblem                             # noqa: E402
from test_joint import RHO, data, stack                        # noqa: E402


def problem(priors, cost='chi2', joint=False):
    st = stack()
    if priors:
        fe = st.layers[1]
        fe.fit['thickness']['prior'] = {'kind': 'gauss', 'mu': 70.0,
                                        'tau': 2.0}
        fe.fit['MSLD_rho']['prior'] = {'kind': 'soft_box', 'a': 5.5e-6,
                                       'b': 6e-6, 's': 1e-7}
    if joint:
        return JointProblem([(n, FitProblem(st, data(r), cost))
                             for n, r in RHO.items()])
    return FitProblem(st, data(RHO['A']), cost)


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('priors', [False, True])
def test_chi2_from_the_cost(priors, joint):
    """cost - penalty / N is mean(residuals^2), to 1e-12; with priors it
    is below the cost."""
    p = problem(priors, joint=joint)
    rng = np.random.default_rng(0)
    for u in rng.uniform(0, 1, (5, len(p.params))):
        x = p.to_x(u)
        cost = p(u)
        direct = float(np.mean(p.residuals(x)**2))
        assert p.chi2_per_point(x, cost) == pytest.approx(direct, rel=1e-12)
        assert p.chi2_per_point(x) == pytest.approx(direct, rel=1e-12)
        if priors:
            assert cost - direct == pytest.approx(
                p.prior_penalty(x) / p.npoints, rel=1e-9)
            assert p.chi2_per_point(x, cost) < cost
        else:
            assert p.chi2_per_point(x, cost) == cost


@pytest.mark.parametrize('joint', [False, True])
def test_none_for_the_log_cost(joint):
    p = problem(True, cost='log', joint=joint)
    x = p.to_x(p.x0())
    assert p.chi2_per_point(x) is None
    assert p.chi2_per_point(x, p(p.x0())) is None


def test_dream_reports_the_same_chi2():
    """DREAM's chi^2/N of a state is the optimiser's at the same values
    (T5): the MAP or the median sent from DREAM compares directly."""
    p = problem(True)
    pp = PosteriorProblem(p, sigma_free=True)
    rng = np.random.default_rng(1)
    for u in rng.uniform(0, 1, (4, len(p.params))):
        x = p.to_x(u)
        assert pp.chi2_per_point(np.r_[x, 0.3]) == pytest.approx(
            p.chi2_per_point(x, p(u)), rel=1e-12)
