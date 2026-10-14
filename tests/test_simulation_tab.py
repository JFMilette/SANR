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
