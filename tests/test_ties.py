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


# -- 'total': the members' total thickness kept --------------------------------
def trilayer():
    """vacuum / A 30 / B 15 / C 10 / Si."""
    return S.Stack([S.Layer('v', 0), S.Layer('A', 30, 4e-6, 0, 2e-6, 0, 3.0),
                    S.Layer('B', 15, 2e-6, 0, 0, 0, 3.0),
                    S.Layer('C', 10, 6e-6, 0, 0, 0, 3.0),
                    S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0)])


def test_total_tie_in_a_fit():
    """A and B split 45 A: the fit finds the split, the total never moves."""
    truth = trilayer()
    truth.layers[1].thickness, truth.layers[2].thickness = 33.0, 12.0
    R = truth.reflectivities(Q, [(X, X)])[:, 0]
    st = trilayer()
    st.layers[1].fit['thickness'] = {'vary': True, 'min': 0, 'max': 500}
    st.tie([(1, 'thickness'), (2, 'thickness')], 'total')
    params = st.free_parameters()
    assert params == [(('tie', 0, 0), 'thickness', 30, 0.0, 45.0)]
    assert parameter_names(st, params) == \
        ['A.thickness in total(A.thickness, B.thickness)']
    p = FitProblem(st, [dict(Q=Q, R=R, dR=0.01 * R, P0=X, P=X)])
    x, cost, *_ = run_de(p, maxiter=40, seed=0)
    p.apply(x)
    a, b = p.stack.layers[1].thickness, p.stack.layers[2].thickness
    assert abs(a - 33.0) < 1e-3 and a + b == pytest.approx(45.0, abs=1e-12)
    assert cost < 1e-6


def test_total_bounds_and_owner():
    st = trilayer()
    st.layers[3].fit['thickness'] = {'vary': True, 'min': 2.0, 'max': 14.0}
    st.tie([(1, 'thickness'), (2, 'thickness'), (3, 'thickness')], 'total')
    # C (the last) has 8 A to give and 4 A to take: shared by A and B
    assert st.total_bounds(0) == (55.0, [(28.0, 34.0), (13.0, 19.0)])
    rng = np.random.default_rng(0)
    for _ in range(50):
        for m, (lo, hi) in enumerate(st.total_bounds(0)[1]):
            setattr(st.owner(('tie', 0, m)), 'thickness', rng.uniform(lo, hi))
        c = st.layers[3].thickness
        assert 2.0 - 1e-9 <= c <= 14.0 + 1e-9
        assert sum(l.thickness for l in st.layers[1:4]) == \
            pytest.approx(55.0, abs=1e-9)
    # the member's own fit settings: a prior on its thickness applies
    st.layers[1].fit['thickness'] = {'vary': True, 'min': 0, 'max': 500,
                                     'prior': None}
    assert st.owner(('tie', 0, 0)).fit is st.layers[1].fit
    back = S.Stack.from_dict(st.to_dict())
    assert back.ties == st.ties and back.free_parameters() == \
        st.free_parameters()
    with pytest.raises(ValueError, match='thicknesses'):
        trilayer().tie([(1, 'NSLD_real'), (2, 'NSLD_real')], 'total')
    with pytest.raises(ValueError, match='no thickness'):
        trilayer().tie([(3, 'thickness'), (4, 'thickness')], 'total')


def test_simulation_tab_total_tie(monkeypatch):
    """Tie with… on a thickness offers the fixed total; the members are
    ordered top to bottom, so the deepest takes the rest."""
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tabs import SimulationTab
    sim = SimulationTab()
    sim.restore_state(trilayer().to_dict())
    sim.list.setCurrentRow(2)                                 # B

    def accept(dlg):
        for b in dlg.findChildren(QtWidgets.QCheckBox):
            b.setChecked(b.text() == 'A')
        [r for r in dlg.findChildren(QtWidgets.QRadioButton)
         if r.text().startswith('Fixed total')][0].setChecked(True)
        return 1
    monkeypatch.setattr(QtWidgets.QDialog, 'exec', accept)
    sim.editor._tie_with('thickness')
    assert sim.stack.ties[0]['mode'] == 'total'
    assert sim.stack.ties[0]['members'] == [[1, 'thickness'], [2, 'thickness']]
    lab = sim.editor.rows['thickness'][4]
    assert lab.text().endswith('⛓') and '45 Å' in lab.toolTip()
