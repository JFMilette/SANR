"""DREAM page and window: the new sampler settings reach sample(), and the
window reports each chain's chi^2/N and chains in different minima.

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

QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
from tabs import ExperimentalTab, SimulationTab                 # noqa: E402
from fit.dream.dream_window import DreamWindow                  # noqa: E402
from fit.dream.dream_fit import sample                          # noqa: E402
from test_dream_fit import small_problem                        # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_page_defaults_and_options(app, monkeypatch):
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    Q = np.linspace(0.01, 0.1, 40)
    one = np.ones_like(Q)
    exp.sets = [dict(name='d', visible=True, channels={
        '+': dict(Q=Q, R=0.1 * one, dR=0.01 * one, dQ=0 * one,
                  theta=np.nan * one, path='x.dat')})]
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    assert exp.dream_rhat.value() == 1.1
    assert exp.dream_snooker.value() == 0.1 and exp.dream_temper.isChecked()
    exp.dream_start.setCurrentIndex(exp.dream_start.findData('current'))
    seen = {}
    monkeypatch.setattr(exp, '_launch_dream',
                        lambda problem, options, report: seen.update(options))
    exp.start_dream()
    assert seen['p_snooker'] == 0.1 and seen['x0'] == 'current'
    assert seen['beta_start'] == pytest.approx(1e-3)
    exp.dream_temper.setChecked(False)
    exp.start_dream()
    assert seen['beta_start'] == 1.0


def test_window_chain_chi2(app):
    pp = small_problem()
    res = sample(pp, n_gen=300, seed=0, stop_when_converged=False)
    win = DreamWindow()
    win.start(pp, 300, 'test')
    win.finish(res)
    chi2 = win.trace.chain_chi2
    assert len(chi2) == 8
    ref = [pp.chi2_per_point(x) for x in res.chains[-1]]
    assert np.allclose(chi2, ref, rtol=1e-8)
    win.trace.combo.setCurrentIndex(0)
    assert 'χ²/N per chain now' in win.trace.info.text()
    win.close()
