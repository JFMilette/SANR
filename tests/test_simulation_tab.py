"""Simulation tab: the Licorne-exact check box follows its settings.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
from model.stack import SMEARING_MODES                          # noqa: E402
from tabs.simulation import (RES_CONVOLUTIONS, SimulationTab,   # noqa: E402
                             res_convolution)


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def exact_tab():
    sim = SimulationTab()
    sim.lic_exact.setChecked(True)
    sim.recompute()
    assert sim.lic_exact.isChecked() and sim.stack.is_licorne_exact()
    return sim


def test_ticking_sets_every_setting(app):
    sim = exact_tab()
    st = sim.stack
    assert sim.rscheme.currentText() == 'Licorne'
    assert SMEARING_MODES[sim.msmear.currentIndex()] == 'step'
    assert RES_CONVOLUTIONS[sim.res_conv.currentIndex()][1] == 'licorne'
    assert (st.step_fallback, st.licorne_renorm, st.licorne_outer) == \
        (False, 'licorne', 'licorne')
    sim.lic_exact.setChecked(False)
    sim.recompute()
    assert not st.is_licorne_exact()
    assert RES_CONVOLUTIONS[sim.res_conv.currentIndex()][1] == 'quadrature'
    assert (st.step_fallback, st.licorne_renorm, st.licorne_outer) == \
        (True, 'manual', 'infinite')


@pytest.mark.parametrize('change', ['roughness', 'smearing', 'resolution'])
def test_leaving_licorne_unticks(app, change):
    """Any setting moved off Licorne's unticks the box at once (before the
    recompute) and after it; putting it back ticks it again."""
    sim = exact_tab()
    combo, other = {
        'roughness': (sim.rscheme, 0),                      # rms
        'smearing': (sim.msmear, SMEARING_MODES.index('vector')),
        'resolution': (sim.res_conv, 0),                    # quadrature
    }[change]
    back = combo.currentIndex()
    combo.setCurrentIndex(other)
    assert not sim.lic_exact.isChecked()
    sim.recompute()
    assert not sim.lic_exact.isChecked() and not sim.stack.is_licorne_exact()
    combo.setCurrentIndex(back)
    sim.recompute()
    assert sim.lic_exact.isChecked() and sim.stack.is_licorne_exact()


def test_roughness_scheme_from_the_user(app):
    """Choosing 'rms' in the combo (activated) also brings its default
    smearing; the box unticks."""
    sim = exact_tab()
    sim.rscheme.setCurrentIndex(0)
    sim.rscheme.activated.emit(0)
    assert not sim.lic_exact.isChecked()
    assert SMEARING_MODES[sim.msmear.currentIndex()] == 'vector'


def test_any_licorne_mode_is_exact(app):
    """The three Licorne convolutions are all Licorne's (ResolutionFun)."""
    sim = exact_tab()
    for i, (_, scheme, fun) in enumerate(RES_CONVOLUTIONS):
        sim.res_conv.setCurrentIndex(i)
        assert sim.lic_exact.isChecked() == (scheme == 'licorne')
    assert res_convolution({'scheme': 'licorne', 'licorne_fun': 2}) == 2


def test_hidden_setting_unticks(app):
    """A Licorne setting without a widget (e.g. from a session or the
    console) unticks the box at the next recompute."""
    sim = exact_tab()
    sim.stack.licorne_outer = 'infinite'
    sim.recompute()
    assert not sim.lic_exact.isChecked()


def test_session_restores_the_box(app):
    sim = exact_tab()
    state = sim.session_state()
    other = SimulationTab()
    assert not other.lic_exact.isChecked()
    other.restore_state(state)
    assert other.lic_exact.isChecked() and other.stack.is_licorne_exact()


def shown(sim):
    """The tab's settings: the model, minus the layers, and the combos."""
    sim.recompute()
    d = sim.stack.to_dict()
    del d['layers']
    return d, [w.currentIndex() for w in (sim.rscheme, sim.msmear,
                                          sim.res_conv)]


def test_untick_restores_the_settings(app):
    """Ticking then unticking goes back to where the tab was, also when it
    was not on the physical defaults (here Licorne's convolution)."""
    for conv in (0, 2):
        sim = SimulationTab()
        sim.res_conv.setCurrentIndex(conv)
        before = shown(sim)
        sim.lic_exact.setChecked(True)
        sim.recompute()
        sim.lic_exact.setChecked(False)
        assert shown(sim) == before and not sim.lic_exact.isChecked()


def test_session_forgets_the_settings(app):
    """A session loaded exact unticks to the physical defaults, not to the
    settings of the session before it."""
    sim = exact_tab()                     # remembers the default settings
    state = exact_tab().session_state()
    sim.restore_state(state)
    assert sim.lic_exact.isChecked()
    sim.lic_exact.setChecked(False)
    sim.recompute()
    assert sim.rscheme.currentText() == 'Licorne'     # kept: none to restore
    assert RES_CONVOLUTIONS[sim.res_conv.currentIndex()][1] == 'quadrature'


def test_angle_letters(app):
    """The GUI writes the in-plane angle (MSLD_theta) as φ and the
    out-of-plane one (MSLD_phi) as θ, Licorne's letters."""
    from tabs.simulation import (EDITOR_PARAMS, PARAM_DISPLAY,
                                 PROFILE_QUANTITIES, shown_names)
    labels = {p[0]: p[1] for p in EDITOR_PARAMS}
    assert labels['MSLD_theta'] == 'MSLD φ' and labels['MSLD_phi'] == 'MSLD θ'
    assert PARAM_DISPLAY['MSLD_theta'][0] == 'MSLD φ'
    assert [q[0] for q in PROFILE_QUANTITIES[2:]] == ['φ', 'θ']
    assert 'in plane' in PROFILE_QUANTITIES[2][1]
    assert shown_names('Fe.MSLD_theta, Co.MSLD_phi, stack.scale') == \
        'Fe.MSLD φ, Co.MSLD θ, stack.scale'


def test_interfaces_combo_and_crossing_warning(app):
    """A session saved before interface_order loads additive and is flagged
    where its interfaces cross (shaded on the profile); ordered clears it,
    is saved, and the combo only applies to the rms scheme."""
    import json
    import pyqtgraph as pg
    from model.stack import INTERFACE_ORDERS
    sim = SimulationTab()
    assert INTERFACE_ORDERS[sim.iorder.currentIndex()] == 'ordered'
    with open(HERE / 'data' / 'show_mistake.json') as f:
        sim.restore_state(json.load(f))
    sim.recompute()
    assert sim.stack.interface_order == 'additive'
    assert 'CrSb Mx -163 %' in sim.iorder_warn.text()
    assert 'CrSb Mx -163 %' in sim.status.text()
    shaded = [i for i in sim.prof_decor if isinstance(i, pg.LinearRegionItem)]
    assert len(shaded) == 1 and shaded[0].getRegion()[0] < 69.2 < \
        shaded[0].getRegion()[1]
    sim.iorder.setCurrentIndex(INTERFACE_ORDERS.index('ordered'))
    sim.recompute()
    assert sim.stack.interface_order == 'ordered'
    assert 'cross' not in sim.status.text()
    assert sim.iorder_warn.text() == '' and not any(
        isinstance(i, pg.LinearRegionItem) for i in sim.prof_decor)
    assert sim.session_state()['interface_order'] == 'ordered'
    sim.iorder.setCurrentIndex(INTERFACE_ORDERS.index('independent'))
    sim.recompute()
    assert sim.stack.interface_order == 'independent'
    assert sim.iorder_warn.text() == ''
    assert sim.session_state()['interface_order'] == 'independent'
    sim.rscheme.setCurrentIndex(1)                          # Licorne
    sim.recompute()
    assert not sim.iorder.isEnabled()


def test_picking_a_layer_opens_its_page(app):
    """A click in the layer list or in the profile opens the tool box on
    Layer properties; a reselect from code (after an edit made on another
    page) leaves the page alone."""
    from pyqtgraph.Qt import QtCore
    sim = SimulationTab()
    sim.recompute()
    sim.toolbox.setCurrentIndex(1)                  # General parameters
    sim.refresh_list(select=2)
    assert sim.toolbox.currentIndex() == 1
    sim.list.itemClicked.emit(sim.list.item(1))
    assert sim.toolbox.currentIndex() == 0

    sim.toolbox.setCurrentIndex(2)                  # Polarisation
    vb = sim.prof_plot.vb
    vb.setGeometry(QtCore.QRectF(0, 0, 400, 300))
    z = 0.5 * sum(sim._profile[1][1:3])             # inside layer 2
    pos = vb.mapViewToScene(QtCore.QPointF(z, 0))

    class Click:
        def double(self):
            return False

        def scenePos(self):
            return pos
    sim._on_profile_click(Click())
    assert sim.list.currentRow() == 2 and sim.toolbox.currentIndex() == 0


def test_new_layer_copies_the_layer_above(app):
    """+ adds below the selected layer a copy of it (own name, its fit
    bounds, nothing varied); below the fronting, the defaults."""
    sim = SimulationTab()
    sim.recompute()
    st = sim.stack
    sim.list.setCurrentRow(1)
    A = st.layers[1]
    A.thickness, A.NSLD_real, A.MSLD_rho = 123.0, 4.5e-6, 1.5e-6
    A.MSLD_sigma, A.roughness_model = 2.0, 'erf'
    A.fit_entry('thickness').update(vary=True, min=100.0, max=150.0)
    sim.add_layer()
    new = st.layers[2]
    assert sim.list.currentRow() == 2 and new is not A and new.name == 'L2'
    for a in ('thickness', 'NSLD_real', 'MSLD_rho', 'MSLD_sigma',
              'roughness_model', 'roughness_sigma', 'roughness_sublayer'):
        assert getattr(new, a) == getattr(A, a)
    assert new.fit['thickness'] == dict(vary=False, min=100.0, max=150.0)
    assert A.fit['thickness']['vary']                   # the original kept
    new.thickness = 7.0
    assert A.thickness == 123.0                         # not shared
    sim.list.setCurrentRow(0)                           # the fronting
    sim.add_layer()
    assert st.layers[1].thickness == 50.0 and \
        not any(e['vary'] for e in st.layers[1].fit.values())


def test_untick_all_fit(qtbot=None):
    """Untick all Fit: nothing varied afterwards (layers, general rows, a
    tie through its members), even once the boxes are read back; values and
    bounds kept; greyed out while a fit runs."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    from tabs import SimulationTab
    win = QtWidgets.QMainWindow()
    sim = SimulationTab()
    win.setCentralWidget(sim)
    st = sim.stack
    for i, attr in ((1, 'thickness'), (1, 'NSLD_real'), (2, 'thickness')):
        e = st.layers[i].fit_entry(attr)
        e.update(vary=True, min=1.0, max=99.0)
    st.ties = [dict(members=[[1, 'roughness_sigma'], [2, 'roughness_sigma']],
                    mode='equal')]
    st.layers[1].fit_entry('roughness_sigma')['vary'] = True
    sim._load_stack_fit()
    sim.stack_rows['scale'][3].setChecked(True)
    sim.stack_rows['background'][3].setChecked(True)
    sim.refresh_list(select=1)
    sim.recompute()
    assert len(st.free_parameters()) == 6
    sim.btn_untick.click()
    assert st.free_parameters() == []
    sim.recompute()                          # the boxes agree
    assert st.free_parameters() == []
    assert not any(cb.isChecked() for *_, cb in sim.stack_rows.values())
    fit_boxes = [row[3] for row in sim.editor.rows.values()]
    assert fit_boxes and not any(b.isChecked() for b in fit_boxes)
    assert st.layers[1].fit['thickness']['max'] == 99.0     # bounds kept
    assert 'Fit unticked on 6 parameters' in win.statusBar().currentMessage()
    sim.set_locked(True)
    assert not sim.btn_untick.isEnabled()
    sim.set_locked(False)


def test_tick_all_buttons(app):
    """Tick thickness / NSLD / roughness: Fit on that quantity of every
    layer it belongs to (thickness of the film layers only, NSLD of the
    film layers and the substrate, roughness of every interface but those with model 'none'), the editor
    showing it, bounds kept; greyed out while a fit runs."""
    from model.stack import Layer
    win = QtWidgets.QMainWindow()
    sim = SimulationTab()
    win.setCentralWidget(sim)
    st = sim.stack
    st.layers = [Layer('air', 0), Layer('a', 50, roughness_sigma=3),
                 Layer('b', 80, roughness_model='none'),
                 Layer('c', 20, roughness_sigma=2),
                 Layer('Si', 0, NSLD_real=2.07e-6, roughness_sigma=4)]
    st.layers[1].fit_entry('thickness').update(vary=True, max=99.0)
    sim.refresh_list(select=1)
    sim.btn_tick['thickness'].click()
    assert 'Fit ticked on 2 more thicknesses' in \
        win.statusBar().currentMessage()
    sim.btn_tick['NSLD_real'].click()
    sim.btn_tick['roughness_sigma'].click()
    free = {(i, a) for i, a, *_ in st.free_parameters()}
    assert free == {(1, 'thickness'), (2, 'thickness'), (3, 'thickness'),
                    (1, 'NSLD_real'), (2, 'NSLD_real'), (3, 'NSLD_real'),
                    (4, 'NSLD_real'),
                    (1, 'roughness_sigma'), (3, 'roughness_sigma'),
                    (4, 'roughness_sigma')}
    assert st.layers[1].fit['thickness']['max'] == 99.0     # bounds kept
    assert sim.editor.rows['thickness'][3].isChecked()
    assert sim.editor.rows['NSLD_real'][3].isChecked()
    assert st.tick_all('thickness') == 0                    # all ticked
    sim.set_locked(True)
    assert not any(b.isEnabled() for b in sim.btn_tick.values())
    sim.set_locked(False)


def test_models_switch_copy_new_delete_and_session(app, monkeypatch):
    sim = SimulationTab()
    st = sim.stack
    sim.list.setCurrentRow(2)
    sim.background.setValue(3e-7)
    sim.recompute()
    names = [l.name for l in st.layers]
    sim.copy_model()                         # a copy, shown
    assert sim.model_names() == ['Model 1', 'Model 1 copy']
    assert sim.stack is st and sim.list.currentRow() == 2
    st.layers[1].thickness = 33.0
    sim.background.setValue(1e-6)
    sim.refresh_list(select=1)
    sim.recompute()
    sim.new_model()                          # no layer, settings kept
    assert sim.model_box.currentText() == 'Model 3'
    assert st.background == 1e-6
    assert [l.name for l in st.layers] == ['vacuum', 'substrate']
    sim.list.setCurrentRow(0)
    sim.add_layer()                          # one film
    sim.refresh_list(select=1)
    sim.model_box.setCurrentIndex(0)         # back to the first, unchanged
    assert [l.name for l in st.layers] == names
    assert st.layers[1].thickness == 80.0 and st.background == 3e-7
    assert sim.list.currentRow() == 2
    sim.model_box.setCurrentIndex(1)
    assert st.layers[1].thickness == 33.0 and st.background == 1e-6
    # the session keeps every model and which one is shown
    state = sim.session_state()
    again = SimulationTab()
    again.restore_state(state)
    assert again.model_names() == sim.model_names()
    assert again.model_box.currentIndex() == 1
    assert again.stack.layers[1].thickness == 33.0
    again.model_box.setCurrentIndex(2)
    assert len(again.stack.layers) == 3
    # a session saved before the models: one model
    state.pop('simulation')['models']
    old = dict(state, simulation={'selected_layer': 1})
    again.restore_state(old)
    assert again.model_names() == ['Model 1']
    # delete the shown model: its neighbour is shown
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question',
                        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
    sim.delete_model()
    assert sim.model_names() == ['Model 1', 'Model 3']
    assert sim.model_box.currentText() == 'Model 1'
    assert st.layers[1].thickness == 80.0
    sim.set_locked(True)
    sim.model_box.setCurrentIndex(1)
    assert sim.model_box.currentIndex() == 0 and st.layers[1].thickness == 80.0
    sim.set_locked(False)


def test_main_window_tabs_and_empty_start(app):
    """Geometry first, Simulation shown first with no layer: the bare
    substrate computes without error, and a layer can be added."""
    import main
    win = main.MainWindow()
    tabs = win.tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == \
        ['Geometry', 'Simulation', 'Experimental data', 'Fit']
    assert tabs.currentWidget() is win.simulation
    sim = win.simulation
    assert [l.name for l in sim.stack.layers] == ['vacuum', 'substrate']
    sim.recompute()
    assert not sim.status.text().startswith('Error')
    assert sim.last_R is not None
    sim.list.setCurrentRow(0)
    sim.add_layer()
    sim.recompute()
    assert len(sim.stack.layers) == 3
    assert not sim.status.text().startswith('Error')


def test_plot_quantities_and_old_sessions(app):
    """Polarized with / without analysis, Non-polarized (R+ + R-)/2 with
    its error from both, and a session saved with the older list (a
    position) shows the same plot as before."""
    import numpy as np
    from tabs.experimental import dataset_series
    from tabs.simulation import (OLD_REFL_QUANTITIES, REFL_QUANTITIES,
                                 quantity_index)
    names = [q[0] for q in REFL_QUANTITIES]
    assert names[:3] == ['Polarized with analysis',
                         'Polarized without analysis', 'Non-polarized']
    Q = np.linspace(0.01, 0.1, 5)
    D = {'+': dict(Q=Q, R=np.full(5, 0.4), dR=np.full(5, 0.03),
                   dQ=0 * Q),
         '-': dict(Q=Q, R=np.full(5, 0.2), dR=np.full(5, 0.04),
                   dQ=0 * Q)}
    (series,), note = dataset_series(D, names.index('Non-polarized'))
    assert note == '' and series[0] == '(R+ + R−)/2'
    assert np.allclose(series[3], 0.3) and np.allclose(series[4], 0.025)
    only_up = {'+': D['+']}
    assert dataset_series(only_up, 2)[0] == []
    sim = SimulationTab()
    sim.refl_quantity.setCurrentIndex(names.index('Non-polarized'))
    sim.recompute()
    (curve, label, _, y), = sim._refl_shown
    R = sim.last_R
    assert label == '(R+ + R−)/2' and np.allclose(y, (R['+'] + R['-']) / 2)
    # sessions: by name now; an older one's position mapped by its name
    for k, old in enumerate(OLD_REFL_QUANTITIES):
        assert names[quantity_index({'quantity': k})] == old
    assert quantity_index({'quantity_name': 'SA SF', 'quantity': 0}) == \
        names.index('SA SF')
    state = sim.session_state()
    assert state['simulation']['display']['quantity_name'] == 'Non-polarized'
    fresh = SimulationTab()
    fresh.restore_state(state)
    assert fresh.refl_quantity.currentText() == 'Non-polarized'


def test_r_scale_is_one_or_the_other(app):
    """log R or R·Q⁴ (linear axis), never both; older sessions' check
    boxes map to one of them."""
    import numpy as np
    from tabs.simulation import saved_r_scale
    sim = SimulationTab()
    assert [sim.r_scale.itemText(i) for i in range(sim.r_scale.count())] \
        == ['log R', 'R·Q⁴']
    sim.recompute()
    assert sim._refl_log
    sim.r_scale.setCurrentIndex(1)
    sim.recompute()
    assert not sim._refl_log
    (_, _, Q, y), *_ = sim._refl_shown
    assert np.allclose(y, sim.last_R['++'] * Q**4, equal_nan=True)
    assert sim.refl_plot.getAxis('left').labelText == 'R·Q⁴ (Å⁻⁴)'
    state = sim.session_state()
    assert state['simulation']['display']['scale'] == 'rq4'
    fresh = SimulationTab()
    fresh.restore_state(state)
    assert fresh.r_scale.currentData() == 'rq4'
    for old, want in (({'logy': True, 'rq4': True}, 'rq4'),
                      ({'logy': False, 'rq4': True}, 'rq4'),
                      ({'logy': True, 'rq4': False}, 'log'),
                      ({'logy': False, 'rq4': False}, 'log'), ({}, 'log')):
        assert saved_r_scale(old) == want
    # an asymmetry: no R scale
    sim.refl_quantity.setCurrentIndex(3)
    sim.recompute()
    assert not sim.r_scale.isEnabled()


@pytest.mark.parametrize('scheme', ['rms', 'licorne'])
def test_negative_msld_is_m_reversed(scheme):
    """rho < 0 at phi is the same magnetisation as -rho at phi + 180 deg:
    with the 'vector' smearing (the components smeared) the same
    reflectivity, in both roughness schemes."""
    smearing = 'vector'
    import numpy as np
    from model.polarisation import default_vectors, vectors_pair
    from tabs.simulation import CHANNELS, make_stack
    pairs = [vectors_pair(*default_vectors()[c]) for c in CHANNELS]
    Q = np.linspace(0.005, 0.2, 300)
    R = []
    for sign in (-1, 1):
        st = make_stack()
        st.roughness_scheme, st.magnetic_smearing = scheme, smearing
        L = st.layers[1]                           # rho 1.5e-6 at 0.75 turn
        L.MSLD_rho = sign * 1.5e-6
        L.MSLD_theta = 0.25 if sign < 0 else 0.75
        st.build_sublayers()
        R.append(st.reflectivities(Q, pairs))
    assert np.allclose(R[0], R[1], rtol=1e-10, atol=1e-14)


def test_negative_msld_in_the_editor_and_fit(app):
    import warnings
    import numpy as np
    from fit.problem import FitProblem
    from model.stack import default_fit
    assert default_fit('MSLD_rho')['min'] < 0
    sim = SimulationTab()
    sim.list.setCurrentRow(1)
    box = sim.editor.rows['MSLD_rho'][0]
    box.setValue(-2.0)                              # 10^-6 A^-2
    sim.recompute()
    assert sim.stack.layers[1].MSLD_rho == pytest.approx(-2e-6)
    assert not sim.status.text().startswith('Error')
    # shown as entered: rho < 0 at L1's own angle (270 deg)
    z, curves = sim._profile[0], sim._profile[3]
    inside = (z > 35) & (z < 45)                    # L1's middle
    assert np.allclose(curves[1][inside], -2.0, atol=0.01)
    assert np.allclose(curves[2][inside], 270.0, atol=0.01)
    st = sim.stack
    for a in ('MSLD_rho', 'MSLD_theta'):
        st.layers[1].fit_entry(a)['vary'] = True
    st.layers[1].fit_entry('MSLD_rho').update(min=-5e-6, max=5e-6)
    data = [dict(Q=np.linspace(0.01, 0.1, 20), R=np.full(20, 1e-3),
                 dR=np.full(20, 1e-4), P0=np.array([1.0, 0, 0]), P=None)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        FitProblem(st, data)
    assert any('two equivalent minima' in str(x.message) for x in w)


def test_negative_msld_signed_in_step_smearing():
    """'step' smears rho as a signed number: from +rho to -rho it passes
    through 0 at the interface (an interface unlike +rho at phi + 180,
    whose |M| stays and whose angle jumps)."""
    import numpy as np
    from tabs.simulation import make_stack
    st = make_stack()
    st.magnetic_smearing = 'step'
    st.layers[2].MSLD_rho = -0.8e-6                 # L2 under L1 (+1.5e-6)
    st.build_sublayers()
    Z = st._interfaces()[1]                         # L1 / L2
    z = np.linspace(Z - 30, Z + 30, 601)
    rho = st.profile(z)[1]
    assert rho[0] == pytest.approx(1.5e-6, rel=1e-3)
    assert rho[-1] == pytest.approx(-0.8e-6, rel=1e-3)
    assert np.all(np.diff(rho) <= 1e-18)            # through 0, no jump


@pytest.mark.parametrize('scheme', ['rms', 'licorne'])
def test_vector_smearing_reports_negative_msld_as_entered(scheme):
    """'vector' smearing: a layer with rho < 0 at 0 deg shows rho < 0 at
    0 deg (not +|rho| at 180 deg) in its core, its slabs and across its
    interface with a +rho layer; the reflectivity is that of the vector."""
    import numpy as np
    from model.polarisation import default_vectors, vectors_pair
    from tabs.simulation import CHANNELS, make_stack
    st = make_stack()
    st.roughness_scheme, st.magnetic_smearing = scheme, 'vector'
    for l in st.layers:
        l.MSLD_theta = 0.0
    st.layers[2].MSLD_rho = -0.8e-6                 # L2, under L1 (+1.5e-6)
    st.build_sublayers()
    Z = st._interfaces()
    mid = 0.5 * (Z[1] + Z[2])                       # L2's middle
    rho, theta, phi = st.profile(np.array([mid]))[1:]
    assert rho[0] == pytest.approx(-0.8e-6, rel=1e-4)    # tanh tails
    assert theta[0] == 0 and phi[0] == 0
    _, arrays = st._slab_arrays
    assert np.all(arrays[2] == 0)                   # every slab at 0 deg
    assert arrays[1].min() == pytest.approx(-0.8e-6, rel=1e-3)
    z = np.linspace(Z[1] - 25, Z[1] + 25, 501)      # L1 / L2, through 0
    rho = st.profile(z)[1]
    assert np.all(np.diff(rho) <= 1e-18)
    assert np.all(st.profile(z)[2] == 0)
    # the same reflectivity as +0.8e-6 at 180 deg
    pairs = [vectors_pair(*default_vectors()[c]) for c in CHANNELS]
    Q = np.linspace(0.005, 0.2, 200)
    R = st.reflectivities(Q, pairs)
    st.layers[2].MSLD_rho, st.layers[2].MSLD_theta = 0.8e-6, 0.5
    st.build_sublayers()
    assert np.allclose(st.reflectivities(Q, pairs), R, rtol=1e-10,
                       atol=1e-14)


def test_right_axis_angles_or_roughness(app):
    """The profile's right axis: the angles, or a roughness stem at each
    interface (σ; σ_M where a layer has its own); the angle boxes shown in
    the first mode only, the σ boxes in the second; kept in the session."""
    import numpy as np
    sim = SimulationTab()
    sim.resize(1400, 900)
    sim.show()
    r = sim.prof_right
    assert r.mode() == 'angles'
    assert all(not cb.isHidden() for cb in sim.prof_boxes)
    assert all(cb.isHidden() for cb in r.boxes)
    st = sim.stack
    st.layers[2].roughness_model = 'none'           # L2's top: sharp
    st.layers[3].MSLD_sigma = 12.0                  # L3's own magnetic σ
    sim.recompute()
    r.set_mode('roughness')
    assert all(sim.prof_boxes[i].isHidden() for i in (2, 3))
    assert not sim.prof_boxes[0].isHidden()        # NSLD stays
    assert all(not cb.isHidden() for cb in r.boxes)
    assert not any(c.isVisible() for c in sim.prof_curves[2:])
    Z = st._interfaces()
    zs, ss = r.heads[0].getData()
    assert np.allclose(zs, Z)
    assert np.allclose(ss, [8.0, 0.0, 8.0, 4.0])     # L1, L2 (none), L3, Si
    assert np.allclose(r.heads[1].getData()[1], [12.0])
    assert sim.prof_plot.getAxis('right').labelText == 'roughness σ (Å)'
    title, rows = __import__('tabs.simulation', fromlist=['x']).profile_readout(
        sim, Z[2] + 0.5)
    assert [row[0] for row in rows][-2:] == ['σ', 'σ_M']
    r.boxes[1].setChecked(False)
    assert not r.heads[1].isVisible()
    state = sim.session_state()
    fresh = SimulationTab()
    fresh.restore_state(state)
    assert fresh.prof_right.mode() == 'roughness'
    assert [cb.isChecked() for cb in fresh.prof_right.boxes] == [True, False]
    r.set_mode('angles')
    assert sim.prof_plot.getAxis('right').labelText == 'MSLD φ, θ (deg)'
    assert not any(h.isVisible() for h in r.heads)


def test_right_axis_angles_or_roughness(app):
    """The profile's right axis: the angles, or a roughness stem at each
    interface (σ; σ_M where a layer has its own); the angle boxes shown in
    the first mode only, the σ boxes in the second; kept in the session,
    which loads into a tab showing other layers."""
    import numpy as np
    from tabs.simulation import empty_stack, profile_readout
    sim = SimulationTab()
    r = sim.prof_right
    assert r.mode() == 'angles'
    assert all(not cb.isHidden() for cb in sim.prof_boxes)
    assert all(cb.isHidden() for cb in r.boxes)
    st = sim.stack
    st.layers[2].roughness_model = 'none'           # L2's top: sharp
    st.layers[3].MSLD_sigma = 12.0                  # L3's own magnetic σ
    sim.recompute()
    r.set_mode('roughness')
    assert all(sim.prof_boxes[i].isHidden() for i in (2, 3))
    assert not sim.prof_boxes[0].isHidden()        # NSLD stays
    assert all(not cb.isHidden() for cb in r.boxes)
    assert not any(c.isVisible() for c in sim.prof_curves[2:])
    Z = st._interfaces()
    zs, ss = r.heads[0].getData()
    assert np.allclose(zs, Z)
    assert np.allclose(ss, [8.0, 0.0, 8.0, 4.0])     # L1, L2 (none), L3, Si
    assert np.allclose(r.heads[1].getData()[1], [12.0])
    assert sim.prof_plot.getAxis('right').labelText == 'roughness σ (Å)'
    _, rows = profile_readout(sim, Z[2] + 0.5)
    assert [row[0] for row in rows][-2:] == ['σ', 'σ_M']
    r.boxes[1].setChecked(False)
    assert not r.heads[1].isVisible()
    # a session with more layers than the tab shows (the load that failed)
    state = sim.session_state()
    fresh = SimulationTab(empty_stack())
    fresh.restore_state(state)
    assert fresh.prof_right.mode() == 'roughness'
    assert [cb.isChecked() for cb in fresh.prof_right.boxes] == [True, False]
    assert np.allclose(fresh.prof_right.heads[0].getData()[0], Z)
    r.set_mode('angles')
    assert sim.prof_plot.getAxis('right').labelText == 'MSLD φ, θ (deg)'
    assert not any(h.isVisible() for h in r.heads)


def test_selected_band_repaints(app):
    """Selecting another layer repaints the bands it re-tints at once: the
    screen matches a full repaint (setBrush alone left the old colours up,
    and the cursor then repainted them in strips)."""
    from types import SimpleNamespace

    import numpy as np
    from PyQt6 import QtCore, QtGui
    from tabs.simulation import LayerLabels, plot_widget

    w, p = plot_widget('z', 'SLD')
    w.resize(600, 400)
    sel = [1]
    labels = LayerLabels(p, lambda: sel[0])
    layers = [SimpleNamespace(name=n, thickness=t) for n, t in
              [('vacuum', 0), ('A', 20), ('B', 40), ('C', 30), ('sub', 0)]]
    labels.set_layers(layers, [0, 20, 60, 90], -20, 110)
    p.setXRange(-20, 110, padding=0)
    p.setYRange(0, 5, padding=0)
    w.show()

    def settle():
        for _ in range(3):
            app.processEvents()
            QtCore.QThread.msleep(10)

    def screen():
        img = app.primaryScreen().grabWindow(w.winId()).toImage()
        return img.convertToFormat(QtGui.QImage.Format.Format_RGB32)

    settle()
    try:
        for new in (3, 2):
            sel[0] = new
            labels.style()
            settle()
            shown = screen()
            w.viewport().update()
            settle()
            assert np.array_equal(shown.constBits().asstring(shown.sizeInBytes()),
                                  screen().constBits().asstring(
                                      shown.sizeInBytes()))
    finally:
        w.close()
