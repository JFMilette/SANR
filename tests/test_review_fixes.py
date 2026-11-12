"""Regressions of the November 2026 code review (each test names the
finding it guards): datasets during a fit, joint modes, a failed DREAM
start, saved fit problems, DREAM's R-hat and kept samples, the Simulation
tab (mixture priors, the Q range, switching models), data files (decimal
commas, byte order marks) and sessions (unsaved changes, failed loads,
the .json save).

Run from the repo root: python -m pytest tests
"""
import json
import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt6 import QtCore, QtWidgets                          # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def spin(ms=50):
    QtCore.QTimer.singleShot(ms, APP.quit)
    APP.exec()


def three_sets():
    """Simulation tab with layer 1's thickness varied, and an Experimental
    tab holding datasets a, b, c (R+)."""
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    Q = np.linspace(0.01, 0.2, 50)
    one = np.ones_like(Q)
    exp.sets = [dict(name=n, visible=True, channels={'+': dict(
        Q=Q, R=0.1 * one, dR=0.01 * one, dQ=0 * one, theta=np.nan * one,
        path='%s.dat' % n)}) for n in 'abc']
    exp.refresh_tree()
    return sim, exp


# -- the fit panel -------------------------------------------------------------
def test_datasets_kept_while_a_fit_runs(monkeypatch):
    """A running fit sends its result to its datasets by name: they can be
    neither renamed nor removed until it ends (a renamed one used to get a
    new, unused model)."""
    sim, exp = three_sets()
    exp._fit = object()                       # a fit going on
    exp.tree.topLevelItem(1).setText(0, 'bb')
    spin()
    assert [s['name'] for s in exp.sets][:3] == ['a', 'b', 'c']
    assert exp.tree.topLevelItem(1).text(0) == 'b'
    exp.tree.setCurrentItem(exp.tree.topLevelItem(2))
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', lambda *a: (
        QtWidgets.QMessageBox.StandardButton.Yes))
    exp.remove_selected()
    assert [s['name'] for s in exp.sets][:3] == ['a', 'b', 'c']
    exp._fit = None                           # ended: both allowed again
    exp.tree.topLevelItem(1).setText(0, 'bb')
    spin()
    assert [s['name'] for s in exp.sets][:3] == ['a', 'bb', 'c']


def test_joint_modes_kept_for_unticked_datasets():
    """A dataset's shared / own / fixed choices survive unticking it and
    ticking it again, and a rename while it is the only one ticked."""
    sim, exp = three_sets()
    p = exp.datasets_panel
    for r in range(3):
        p.list.item(r, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    spin()
    c = p.cells['c', 'L1.thickness']
    c.setCurrentIndex(c.findData('own'))
    spin()
    p.list.item(2, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    spin()
    p.list.item(2, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    spin()
    assert p.mode('c', 'L1.thickness') == 'own'
    # a alone, renamed, then joined again
    a = p.cells['a', 'L1.thickness']
    a.setCurrentIndex(a.findData('own'))
    spin()
    for r in (1, 2):
        p.list.item(r, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    spin()
    exp.tree.topLevelItem(0).setText(0, 'aa')
    spin()
    p.list.item(1, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    spin()
    assert p.mode('aa', 'L1.thickness') == 'own'


def test_failed_dream_start_changes_nothing():
    """A DREAM start that fails its checks (NPE seeds without a network)
    leaves Revert's values and the last run's problem as they were."""
    sim, exp = three_sets()
    exp.fit_set.setCurrentIndex(exp.fit_set.findText('a'))
    keep_start, keep_problem = object(), object()
    exp._fit_start, exp._dream_problem = keep_start, keep_problem
    exp.dream_start.setCurrentIndex(exp.dream_start.findData('npe'))
    exp._npe_seeds = None
    exp.start_dream()
    assert exp._dream is None
    assert exp._fit_start is keep_start
    assert exp._dream_problem is keep_problem


# -- saved fit problems ----------------------------------------------------------
def test_saved_problem_starts_where_the_fit_started():
    """problem_state saves the start values, not those the fit's stack
    holds after its last evaluation."""
    from fit.saved import problem_from_state, problem_state
    from test_joint import RHO, data, joint, stack
    from fit.problem import FitProblem
    p = FitProblem(stack(), data(RHO['A']), 'chi2')
    u0 = p.x0().copy()
    p(np.full(len(u0), 0.9))
    back = problem_from_state(problem_state(p))
    assert np.allclose(back.x0(), u0)
    assert back(u0) == p(u0)
    j = joint({'A': {'Fe.MSLD_rho': 'own'}})
    u0 = j.x0().copy()
    j(np.full(len(u0), 0.7))
    back = problem_from_state(problem_state(j))
    assert np.allclose(back.x0(), u0) and back(u0) == j(u0)


# -- DREAM -------------------------------------------------------------------------
def _gauss(x):
    return -0.5 * float(np.sum(np.asarray(x) ** 2))


def test_dream_final_rhat_and_kept_samples():
    """A run that ends between two R-hat generations has the R-hat of
    its last generation; the generation that ends the adaptation phase is
    not a posterior sample."""
    from fit.dream.dream import dream, kept_from
    calls = [0]

    def stop(*_):
        calls[0] += 1
        return calls[0] >= 250
    r = dream(_gauss, [(-5, 5)] * 2, n_gen=1000, n_chains=4, seed=0,
              callback=stop, stop_when_converged=False)
    assert r.generations[-1] == 250
    assert [g for g, _ in r.rhat_history][-1] == 250
    assert r.adapt_until not in r.generations[r.keep(0.0)]
    assert kept_from(np.array([0, 10]), 10, 0.0) == 11
    one = dream(_gauss, [(-5, 5)] * 2, n_gen=1, n_chains=4, seed=0)
    assert one.samples().shape[0] == 0         # all adaptation


def test_continued_rhat_curve_has_no_copy():
    """Continue: the live R-hat curve does not repeat the last R-hat of
    the run continued as a new point."""
    from fit.dream.dream_fit import PosteriorProblem
    from fit.dream.dream_window import DreamThread
    from test_dream_fit import small_problem
    pp = small_problem()
    assert isinstance(pp, PosteriorProblem)
    th = DreamThread(pp, n_gen=250, seed=0, stop_when_converged=False)
    th.run()
    first = th.result
    got = []
    th2 = DreamThread(pp, n_gen=320, resume=first,
                      stop_when_converged=False)
    th2.rhat.connect(lambda g, r, t: got.append(list(g)))
    th2.run()
    APP.processEvents()
    gens = got[-1] if got else []
    assert len(gens) == len(set(gens))
    assert all(g in [t for t, _ in th2.result.rhat_history] for g in gens)


# -- the Simulation tab ---------------------------------------------------------------
def test_mixture_edit_keeps_its_prior():
    from test_mixture import three_models
    sim = three_models()
    e = sim.stack.mixture[0]
    e['fit']['prior'] = {'kind': 'gauss', 'mu': 0.3, 'sigma': 0.05}
    name, use, w, lo, hi, cb = sim._mix_rows[0]
    w.setValue(0.25)                          # any edit of the table
    sim._mix_edited()
    assert sim.stack.mixture[0]['fit']['prior'] == \
        {'kind': 'gauss', 'mu': 0.3, 'sigma': 0.05}
    assert sim.stack.mixture[0]['weight'] == 0.25


def test_q_range_never_inverted():
    from tabs import SimulationTab
    sim = SimulationTab()
    sim.qmin.setValue(0.01)
    sim.qmax.setValue(0.2)
    sim.qmin.setValue(0.3)                    # above Q max: undone
    assert (sim.qmin.value(), sim.qmax.value()) == (0.01, 0.2)
    sim.qmax.setValue(0.005)                  # below Q min: undone
    assert (sim.qmin.value(), sim.qmax.value()) == (0.01, 0.2)
    sim.qmax.setValue(0.5)
    sim.qmin.setValue(0.3)
    assert (sim.qmin.value(), sim.qmax.value()) == (0.3, 0.5)


def test_model_switch_tells_the_other_tabs_once():
    """The other tabs hear of a model switch once the stack and the name
    are both the new model's."""
    from tabs import SimulationTab
    sim = SimulationTab()
    sim._add_model('Model 2', sim.stack.to_dict())
    sim.stack.layers[1].thickness = 123.0      # Model 2's own
    log = []
    sim.reflectanceChanged.connect(lambda: log.append(
        (sim.shown_model(), sim.stack.layers[1].thickness)))
    sim.switch_model(0)
    assert log and all(t != 123.0 for name, t in log if name == 'Model 1')
    assert all(t == 123.0 for name, t in log if name == 'Model 2')


# -- data files -------------------------------------------------------------------------
def test_decimal_commas_and_byte_order_marks():
    from tabs.experimental import clean_lines, decimal_comma, parse_data
    d, _ = parse_data(clean_lines('0,01 1,5e-3 1,0e-4\n0,02 1,2e-3 1e-4'), 0)
    assert np.allclose(d['Q'], [0.01, 0.02])
    assert np.allclose(d['R'], [1.5e-3, 1.2e-3])
    d, _ = parse_data(clean_lines('# x, y\n0,01;1,5e-3\n0,02;1,2E-03'), 0)
    assert np.allclose(d['R'], [1.5e-3, 1.2e-3])
    # comma-separated columns with decimal points: as before
    d, _ = parse_data(clean_lines('0.01,1.5e-3,1e-4\n0.02,1.2e-3,1e-4'), 0)
    assert np.allclose(d['Q'], [0.01, 0.02])
    assert not decimal_comma(['1,2', '3,4'])            # integers: columns
    for bom in ('﻿', 'ï»¿'):
        d, _ = parse_data(clean_lines(bom + '0.01 1.5e-3\n0.02 1e-3'), 0)
        assert len(d['Q']) == 2                          # first point kept


# -- sessions ------------------------------------------------------------------------------
@pytest.fixture
def window(monkeypatch):
    import main
    win = main.MainWindow()
    win.show()
    APP.processEvents()
    yield win
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', lambda *a, **k: (
        QtWidgets.QMessageBox.StandardButton.Discard))
    win.close()


def test_unsaved_changes_are_asked_about(window, monkeypatch, tmp_path):
    import main
    asked = []

    def answer(button):
        def question(parent, title, *a, **k):
            asked.append(title)
            return button
        return question
    B = QtWidgets.QMessageBox.StandardButton
    window.close()                            # nothing changed: no question
    assert asked == [] and not window.isVisible()
    window.show()
    window.simulation.background.setValue(1e-6)
    APP.processEvents()
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', answer(B.Cancel))
    window.close()
    assert asked == ['Quit SANR'] and window.isVisible()
    out = tmp_path / 's.json'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (
                            str(out), main.SAVE_FILTER.split(';;')[1])))
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', answer(B.Save))
    window.close()                            # saved, then closed
    assert out.is_file() and not window.isVisible()
    assert not (tmp_path / 's.json.part').exists()
    assert json.loads(out.read_text())['background'] == 1e-6


def test_open_session_does_not_ask(window, monkeypatch, tmp_path):
    """Opening a session replaces the one shown without asking about its
    unsaved changes: only quitting asks."""
    import main
    asked = []
    B = QtWidgets.QMessageBox.StandardButton

    def question(parent, title, *a, **k):
        asked.append(title)
        return B.Cancel
    out = tmp_path / 's.json'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (
                            str(out), main.SAVE_FILTER.split(';;')[1])))
    window.simulation.background.setValue(1e-6)
    APP.processEvents()
    assert window.save_session() and out.is_file()
    window.simulation.background.setValue(3e-6)        # not saved
    APP.processEvents()
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', question)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (str(out), '')))
    window.open_session()
    assert asked == []
    assert window.simulation.background.value() == pytest.approx(1e-6)
    window.simulation.background.setValue(4e-6)        # quitting still asks
    APP.processEvents()
    window.close()
    assert asked == ['Quit SANR'] and window.isVisible()


def test_failed_open_keeps_the_session(window, monkeypatch, tmp_path):
    import main
    window.simulation.background.setValue(2e-6)
    window.simulation.recompute()
    d = window.simulation.session_state()
    d['experimental'] = window.experimental.session_state(str(tmp_path))
    d['experimental']['sets'] = [{'broken': True}]
    d['background'] = 5e-6
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(d, default=main._plain))
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', lambda *a, **k: (
        QtWidgets.QMessageBox.StandardButton.Discard))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (str(bad), '')))
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical',
                        staticmethod(lambda *a, **k: errors.append(a)))
    window.open_session()
    assert errors                              # said so
    assert window.simulation.stack.background == 2e-6   # and put back
