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
