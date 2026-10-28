"""Mixture of models (model.stack MIXTURE) and its window in the Simulation
tab: regions with 0, 1 or 2 extra layers mixed incoherently.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fit.problem import FitProblem                             # noqa: E402
from model.polarisation import default_vectors                  # noqa: E402
from model.stack import Layer, Stack                            # noqa: E402

Q = np.linspace(0.008, 0.18, 200)


def pair(ch='++'):
    P0, P = default_vectors()[ch]
    return np.asarray(P0, dtype=float), \
        None if P is None else np.asarray(P, dtype=float)


def stacks():
    """(two extra, one extra, none): the same film under 0-2 cap layers."""
    def film(caps):
        layers = [Layer('vacuum')]
        layers += [Layer(n, thickness=t, NSLD_real=v, roughness_sigma=3.0,
                         roughness_sublayer=10) for n, t, v in caps]
        layers += [Layer('Fe', thickness=90.0, NSLD_real=8e-6, MSLD_rho=4e-6,
                         roughness_sigma=4.0, roughness_sublayer=10),
                   Layer('Si', NSLD_real=2.07e-6, roughness_sigma=3.0,
                         roughness_sublayer=10)]
        st = Stack(layers)
        st.resolution.update(enabled=True, mode='tof', tof_dlambda=0.02,
                             tof_angles=[dict(theta=0.01, dtheta=2e-4,
                                              qmax=None)])
        st.build_sublayers()
        return st
    return (film([('cap1', 10.0, 3e-6), ('cap2', 15.0, 5e-6)]),
            film([('cap2', 15.0, 5e-6)]), film([]))


def mixed(w1=0.3, w2=0.2):
    a, b, c = stacks()
    a.mixture = [dict(name='one', weight=w1, fit=dict(vary=True, min=0.0,
                                                      max=1.0)),
                 dict(name='none', weight=w2, fit=dict(vary=True, min=0.0,
                                                       max=1.0))]
    a._mix_stacks = {'one': b, 'none': c}
    return a, b, c


def test_weighted_sum_of_the_models():
    a, b, c = stacks()
    for st in (a, b, c):
        st.scale, st.background = 0.9, 2e-7
    ra, rb, rc = (st.reflectivities(Q, [pair()])[:, 0] for st in (a, b, c))
    m, _, _ = mixed()
    m.scale, m.background = 0.9, 2e-7
    rm = m.reflectivities(Q, [pair()])[:, 0]
    want = 0.5 * ra + 0.3 * rb + 0.2 * rc     # scale, background once
    assert np.max(np.abs(rm / want - 1)) < 1e-12
    # weights summing above 1: divided by their sum, this model 0
    m.mixture[0]['weight'] = m.mixture[1]['weight'] = 0.8
    assert m.mixture_parts()[0] == 0.0
    rm = m.reflectivities(Q, [pair()])[:, 0]
    assert np.max(np.abs(rm / (0.5 * rb + 0.5 * rc) - 1)) < 1e-12


def test_ticked_parameters_are_shared_by_name():
    a, b, c = mixed()
    a.layers[3].fit_entry('thickness')['vary'] = True     # Fe
    a.layers[2].fit_entry('NSLD_real')['vary'] = True     # cap2
    a.layers[3].thickness, a.layers[2].NSLD_real = 70.0, 4e-6
    a.reflectivities(Q, [pair()])
    assert b.layers[2].thickness == c.layers[1].thickness == 70.0
    assert b.layers[1].NSLD_real == 4e-6                   # cap2 in b
    assert [p[:2] for p in a.free_parameters()] == [
        (2, 'NSLD_real'), (3, 'thickness'), (('mix', 0), 'weight'),
        (('mix', 1), 'weight')]


def test_fit_recovers_the_weights():
    truth, _, _ = mixed(0.35, 0.15)
    truth.layers[3].fit_entry('thickness')['vary'] = True
    R = truth.reflectivities(Q, [pair()])[:, 0]
    data = [dict(Q=Q, R=R, dR=0.02 * R, P0=pair()[0], P=pair()[1])]
    start, _, _ = mixed(0.1, 0.4)
    start.layers[3].thickness = 85.0
    start.layers[3].fit_entry('thickness').update(vary=True, min=80.0,
                                                  max=100.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = FitProblem(start, data)
    assert p.names == ['Fe.thickness', 'mixture one', 'mixture none']
    from scipy.optimize import minimize
    r = minimize(p, p.x0(), method='L-BFGS-B', bounds=[(0, 1)] * 3)
    x = p.to_x(r.x)
    assert x == pytest.approx([90.0, 0.35, 0.15], abs=2e-3)


def test_spread_or_mixture_not_both():
    a, b, c = mixed()
    data = [dict(Q=Q, R=np.full_like(Q, 1e-3), dR=np.full_like(Q, 1e-4),
                 P0=pair()[0], P=pair()[1])]
    b.thickness_spread = 0.05                # a mixed model's own: ignored
    b.thickness_spread_nodes = 7
    rm = a.reflectivities(Q, [pair()])
    b.thickness_spread = 0.0
    assert np.array_equal(a.reflectivities(Q, [pair()]), rm)
    a.thickness_spread = 0.03
    with pytest.raises(ValueError, match='cannot be used together'):
        a.reflectivities(Q, [pair()])
    a.thickness_spread = 0.0
    a.fit_entry('thickness_spread')['vary'] = True
    with pytest.raises(ValueError, match='cannot be used together'):
        FitProblem(a, data)


def test_entries_are_saved_not_the_models():
    a, _, _ = mixed()
    d = a.to_dict()
    assert [e['name'] for e in d['mixture']] == ['one', 'none']
    again = Stack.from_dict(d)
    assert again.mixture == a.mixture and again._mix_stacks == {}
    # unlinked: the stack alone, no weight to fit
    assert again.mixture_parts() == (1.0, [])
    assert not any(p[1] == 'weight' for p in again.free_parameters())


# -- the Simulation tab ---------------------------------------------------
QtWidgets = pytest.importorskip('PyQt6.QtWidgets')


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def three_models():
    """The tab with 'two extra' (shown), 'one extra', 'none', mixed."""
    from tabs.simulation import SimulationTab
    sim = SimulationTab()
    st = sim.stack
    st.layers.insert(1, Layer('cap2', thickness=15.0, NSLD_real=5e-6,
                              roughness_sigma=3.0, roughness_sublayer=10))
    st.layers.insert(1, Layer('cap1', thickness=10.0, NSLD_real=3e-6,
                              roughness_sigma=3.0, roughness_sublayer=10))
    sim.refresh_list(select=1)
    sim.recompute()
    sim._models[0]['name'] = 'two extra'
    for name in ('one extra', 'none'):
        sim.copy_model()
        st.layers.pop(1)
        sim.refresh_list(select=1)
        sim.recompute()
        sim._models[-1]['name'] = name
    sim._refresh_models()
    sim.model_box.setCurrentIndex(0)
    sim.show_spread_window()
    sim.lateral.button(2).click()             # Mixture of models
    for name, use, w, lo, hi, cb in sim._mix_rows:
        use.setChecked(True)
        w.setValue(0.3 if name == 'one extra' else 0.2)
        cb.setChecked(True)
    sim.recompute()
    return sim


def test_window_mixes_and_shares(app):
    sim = three_models()
    st = sim.stack
    assert [(e['name'], e['weight']) for e in st.mixture] == \
        [('one extra', 0.3), ('none', 0.2)]
    assert 'w₀ = 0.5' in sim.mix_w0.text()
    L2 = [l.name for l in st.layers].index('L2')
    box = dict((i, cb) for i, cb in sim._mix_layer_boxes)[L2]
    assert 'also in one extra, none' in box.text()
    box.setChecked(True)                      # the shared thickness
    assert st.layers[L2].fit['thickness']['vary']
    st.layers[L2].thickness = 111.0
    sim.recompute()
    for m in sim._models[1:]:
        assert [l['thickness'] for l in m['model']['layers']
                if l['name'] == 'L2'] == [111.0]
    sim.model_box.setCurrentIndex(2)           # 'none' shows it
    assert [l.thickness for l in sim.stack.layers if l.name == 'L2'] == \
        [111.0]
    assert sim.stack.mixture == []             # its own mixture: none
    sim.model_box.setCurrentIndex(0)
    assert len(sim.stack.mixture) == 2
    names = [p for p in FitProblem(sim.stack, [dict(
        Q=Q, R=np.full_like(Q, 1e-3), dR=np.full_like(Q, 1e-4),
        P0=pair()[0], P=pair()[1])]).names]
    assert names == ['L2.thickness', 'mixture one extra', 'mixture none']


def test_session_rename_and_delete(app, monkeypatch):
    from tabs.simulation import SimulationTab
    sim = three_models()
    state = sim.session_state()
    again = SimulationTab()
    again.restore_state(state)
    assert [e['name'] for e in again.stack.mixture] == ['one extra', 'none']
    again.recompute()
    assert set(again.stack._mix_stacks) == {'one extra', 'none'}
    # renaming a mixed model renames its entry
    again.model_box.setCurrentIndex(2)
    monkeypatch.setattr(QtWidgets.QInputDialog, 'getText',
                        lambda *a, **k: ('flat', True))
    again.rename_model()
    again.model_box.setCurrentIndex(0)
    assert [e['name'] for e in again.stack.mixture] == ['one extra', 'flat']
    # deleting one removes it from the mixture
    again.model_box.setCurrentIndex(1)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, 'question',
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
    again.delete_model()
    assert again.model_box.currentText() == 'two extra'
    assert [e['name'] for e in again.stack.mixture] == ['flat']
    assert again.mix_table.rowCount() == 1


def test_choosing_one_switches_the_other_off(app):
    sim = three_models()
    assert sim.lateral_mode() == 'mixture'
    assert sim.mix_box.isEnabled() and not sim.spread_box.isEnabled()
    sim.lateral.button(1).click()             # Thickness spread
    assert sim.stack.mixture == []
    assert not any(use.isChecked() for _, use, *_ in sim._mix_rows)
    assert sim.spread_box.isEnabled() and not sim.mix_box.isEnabled()
    sim.spread.setValue(4.0)
    sim.stack_rows['thickness_spread'][3].setChecked(True)
    sim.recompute()
    assert sim.stack.thickness_spread == 0.04
    sim.lateral.button(2).click()             # back to the mixture
    sim.recompute()
    assert sim.stack.thickness_spread == 0.0
    assert not sim.stack.fit['thickness_spread']['vary']
    # a model shown with a spread shows Thickness spread; neither: None
    sim.lateral.button(1).click()
    sim.spread.setValue(2.0)
    sim.recompute()
    sim.model_box.setCurrentIndex(1)
    assert sim.lateral_mode() == 'none'
    sim.model_box.setCurrentIndex(0)
    assert sim.lateral_mode() == 'spread'


@pytest.mark.parametrize('scheme', ['quadrature', 'licorne'])
def test_resolution_is_applied_after_the_mixing(scheme):
    """The ideal reflectivities are mixed, then the mix is averaged over
    this stack's resolution once (Q shift, scale, background after); the
    mixed models' own resolution is never used."""
    a, b, c = mixed()
    a.resolution.update(scheme=scheme, licorne_fun=3)
    a.q_shift, a.scale, a.background = 4e-4, 0.95, 3e-7
    pairs = [pair('++'), pair('--')]
    w0, parts = a.mixture_parts()

    def mixed_ideal(q):
        return w0 * a._trace(q, pairs) + sum(
            w * st._trace(q, pairs) for w, st in parts)
    want = 0.95 * a._resolve(Q, mixed_ideal, None, 4e-4) + 3e-7
    assert np.allclose(a.reflectivities(Q, pairs), want, rtol=1e-12, atol=0)
    # the mixed models' own resolution: ignored
    b.resolution.update(enabled=False)
    c.resolution['tof_dlambda'] = 0.2
    assert np.allclose(a.reflectivities(Q, pairs), want, rtol=1e-12, atol=0)
    # and the resolution really smears the mix: off, the curve differs
    a.resolution.update(enabled=False)
    assert not np.allclose(a.reflectivities(Q, pairs), want, rtol=1e-3)
