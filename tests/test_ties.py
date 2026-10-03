"""Ties (model.stack ties): one fitted value for the same attribute of
several layers, 'equal' or with Licorne's coupled-parameter mapping, kept
through fits, saving and layer edits.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import model.stack as S                                         # noqa: E402
from fit.de.de_fit import run_de                                # noqa: E402
from fit.dream.dream_fit import parameter_names                 # noqa: E402
from fit.problem import FitProblem                              # noqa: E402

X = np.array([1.0, 0, 0])
Q = np.linspace(0.01, 0.12, 60)


def bilayer():
    """vacuum / A 60 A / B 40 A / Si, NSLD of A and B fitted."""
    st = S.Stack([S.Layer('v', 0), S.Layer('A', 60, 4e-6, 0, 0, 0, 3.0),
                  S.Layer('B', 40, 3e-6, 0, 0, 0, 3.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0)])
    st.layers[1].fit['NSLD_real'] = {'vary': True, 'min': 1e-6, 'max': 7e-6}
    st.layers[2].fit['NSLD_real'] = {'vary': True, 'min': 2e-6, 'max': 6e-6}
    st.layers[1].fit['thickness'] = {'vary': True, 'min': 40, 'max': 80}
    return st


def test_equal_tie_in_a_fit():
    st = bilayer()
    st.tie([(1, 'NSLD_real'), (2, 'NSLD_real')])
    params = st.free_parameters()
    assert [p[:2] for p in params] == [(1, 'thickness'),
                                       (('tie', 0), 'NSLD_real')]
    assert params[1][2:] == (4e-6, 2e-6, 6e-6)       # bounds intersected
    assert parameter_names(st, params)[1] == 'tie(A.NSLD_real, B.NSLD_real)'
    truth = S.Stack.from_dict(st.to_dict())
    truth.set_tie(0, 5e-6)
    R = truth.reflectivities(Q, [(X, None)])[:, 0]
    p = FitProblem(st, [dict(Q=Q, R=R, dR=0.01 * R, P0=X, P=None)])
    x, cost, *_ = run_de(p, maxiter=40, seed=0)
    p.apply(x)
    a, b = p.stack.layers[1].NSLD_real, p.stack.layers[2].NSLD_real
    assert a == b and abs(a / 5e-6 - 1) < 1e-3 and cost < 1e-4


def test_licorne_mapping():
    st = bilayer()
    k = st.tie([(1, 'NSLD_real'), (2, 'NSLD_real')], 'licorne')
    v0, lo, hi = [4e-6, 3e-6], [1e-6, 2e-6], [7e-6, 6e-6]
    assert st.tie_bounds(k) == (sum(lo), sum(hi))
    for S_, want in ((sum(v0), v0), (sum(lo), lo), (sum(hi), hi)):
        st.set_tie(k, S_)
        got = [st.layers[i].NSLD_real for i in (1, 2)]
        assert np.allclose(got, want, rtol=1e-12, atol=0)
        assert st.tie_value(k) == pytest.approx(S_)
    # half way up: each member half way to its max
    st.set_tie(k, (sum(v0) + sum(hi)) / 2)
    assert np.allclose([st.layers[i].NSLD_real for i in (1, 2)],
                       [(4e-6 + 7e-6) / 2, (3e-6 + 6e-6) / 2])


def test_owner_round_trip_and_refusals():
    st = bilayer()
    st.tie([(1, 'NSLD_real'), (2, 'NSLD_real')])
    setattr(st.owner(('tie', 0)), 'NSLD_real', 5.5e-6)
    assert st.layers[2].NSLD_real == 5.5e-6
    back = S.Stack.from_dict(st.to_dict())
    assert back.ties == st.ties and back.free_parameters() == \
        st.free_parameters()
    with pytest.raises(ValueError, match='already tied'):
        st.tie([(1, 'NSLD_real'), (3, 'NSLD_real')])
    with pytest.raises(ValueError, match='one attribute'):
        st.tie([(1, 'thickness'), (2, 'NSLD_real')])
    st.untie(2, 'NSLD_real')
    assert st.ties == []


def test_layer_edits_follow():
    st = bilayer()
    st.layers.insert(3, S.Layer('C', 20, 1e-6))
    st.tie([(1, 'NSLD_real'), (2, 'NSLD_real'), (3, 'NSLD_real')])
    old = list(st.layers)
    st.layers[1], st.layers[2] = st.layers[2], st.layers[1]   # move A down
    st.reindex_ties(old)
    assert st.ties[0]['members'] == [[2, 'NSLD_real'], [1, 'NSLD_real'],
                                     [3, 'NSLD_real']]
    assert st.ties[0]['v0'] == [4e-6, 3e-6, 1e-6]
    old = list(st.layers)
    del st.layers[3]                                          # C goes
    st.reindex_ties(old)
    assert st.ties[0]['members'] == [[2, 'NSLD_real'], [1, 'NSLD_real']]
    old = list(st.layers)
    del st.layers[1]                                          # one left
    st.reindex_ties(old)
    assert st.ties == []


def test_simulation_tab_keeps_ties(monkeypatch):
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tabs import SimulationTab
    sim = SimulationTab()
    st = bilayer()
    st.tie([(1, 'NSLD_real'), (2, 'NSLD_real')])
    sim.restore_state(st.to_dict())
    assert sim.stack.ties == st.ties
    sim.list.setCurrentRow(1)
    assert sim.editor.rows['NSLD_real'][4].text().endswith('⛓')
    sim.list.setCurrentRow(1)
    sim.add_layer()                                  # a layer below A
    assert sim.stack.ties[0]['members'] == [[1, 'NSLD_real'],
                                            [3, 'NSLD_real']]
    sim.list.setCurrentRow(1)
    sim.editor._untie('NSLD_real')
    assert sim.stack.ties == []


# -- the sheet moment (T2) -------------------------------------------------------
def test_moment_mode():
    st = bilayer()
    A = st.layers[1]
    A.MSLD_rho = 2e-6
    A.fit['MSLD_rho'] = {'vary': True, 'min': 0.0, 'max': 4e-6}
    e = A.fit_entry('MSLD_moment')
    assert (e['min'], e['max']) == (0.0, 4e-6 * 80)
    e['vary'] = True
    attrs = [p[:2] for p in st.free_parameters()]
    assert (1, 'MSLD_moment') in attrs and (1, 'MSLD_rho') not in attrs
    assert A.MSLD_moment == pytest.approx(2e-6 * 60)
    # with the thickness tied to B's: the moment still uses the new t
    st.tie([(1, 'thickness'), (2, 'thickness')])
    st.layers[2].fit['thickness'] = {'vary': True, 'min': 20, 'max': 80}
    params = st.free_parameters()
    x = [{'MSLD_moment': 1.5e-4, 'thickness': 50.0}.get(p[1], p[2])
         for p in params]
    st.set_values(params, x)
    assert A.thickness == 50.0 and A.MSLD_rho == pytest.approx(3e-6)
    back = S.Stack.from_dict(st.to_dict())
    assert back.free_parameters() == st.free_parameters()


def test_moment_check_box():
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tabs import SimulationTab
    sim = SimulationTab()
    sim.list.setCurrentRow(1)
    ed = sim.editor
    ed.rows['MSLD_rho'][3].setChecked(True)
    ed.moment.setChecked(True)
    L = sim.stack.layers[1]
    assert L.fit['MSLD_moment']['vary'] and L.fit['MSLD_moment']['on']
    attrs = [p[1] for p in sim.stack.free_parameters()]
    assert 'MSLD_moment' in attrs and 'MSLD_rho' not in attrs
    ed.rows['MSLD_rho'][3].setChecked(False)          # rho not fitted
    assert not L.fit['MSLD_moment']['vary'] and L.fit['MSLD_moment']['on']
    sim.list.setCurrentRow(2)
    sim.list.setCurrentRow(1)
    assert ed.moment.isChecked()
