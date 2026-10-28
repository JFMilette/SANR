"""fit.joint: one stack fitted to several datasets, each parameter shared,
own to a dataset or not fitted in it; DE and DREAM take the joint problem
as they take a FitProblem.

Run from the repo root: python -m pytest tests
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fit.de.de_fit import run_de                            # noqa: E402
from fit.dream.dream_fit import (PosteriorProblem, load_run,  # noqa: E402
                                 predictive_bands, profile_bands, sample,
                                 save_result)
from fit.joint import JointProblem, part_index, part_x, parts  # noqa: E402
from fit.problem import FitProblem                          # noqa: E402
from test_dream_fit import OLD, film, measure                # noqa: E402

RHO = {'A': 5e-6, 'B': 3e-6}          # Fe magnetisation of each dataset


def data(rho, channels=('+', '-')):
    st = film()
    st.layers[1].MSLD_rho = rho
    return measure(st, channels)


def stack():
    """Fe thickness and MSLD rho varied, starting off the truth (80 A)."""
    st = film()
    fe = st.layers[1]
    fe.thickness, fe.MSLD_rho = 75.0, 4e-6
    fe.fit['thickness'] = {'vary': True, 'min': 60, 'max': 100}
    fe.fit['MSLD_rho'] = {'vary': True, 'min': 1e-6, 'max': 8e-6}
    return st


def joint(modes=None, cost='chi2'):
    st = stack()
    return JointProblem([(n, FitProblem(st, data(r), cost))
                         for n, r in RHO.items()], modes)


def test_all_shared_is_one_fit_of_every_channel():
    jp = joint()
    one = FitProblem(stack(), data(RHO['A']) + data(RHO['B']), 'chi2')
    assert jp.names == one.names == ['Fe.thickness', 'Fe.MSLD_rho']
    assert jp.npoints == one.npoints and jp.channel_part == [0, 0, 1, 1]
    for u in (jp.x0(), np.full(2, 0.3)):
        assert jp(u) == pytest.approx(one(u), rel=1e-12)
    # the channels' points, for DREAM's bands and files
    for (_, i, *_), (_, j, *_) in zip(jp.channels, one.channels):
        assert np.array_equal(jp.Q[i], one.Q[j])


def test_own_and_fixed():
    jp = joint({'A': {'Fe.MSLD_rho': 'own'}, 'B': {'Fe.MSLD_rho': 'own',
                                                   'Fe.thickness': 'fixed'}})
    assert jp.names == ['Fe.thickness', 'Fe.MSLD_rho [A]',
                        'Fe.MSLD_rho [B]']
    assert jp.scopes == [None, 'A', 'B']
    x = [80.0, 5e-6, 3e-6]
    assert np.allclose(jp.part_x(x, 0), [80.0, 5e-6])
    # B keeps the thickness the fit started from
    assert np.allclose(jp.part_x(x, 1), [75.0, 3e-6])
    assert part_index(jp, 1) == [None, 2] and part_index(jp, 0) == [0, 1]
    # nothing at all fitted
    with pytest.raises(ValueError, match='no parameter'):
        joint({n: {'Fe.thickness': 'fixed', 'Fe.MSLD_rho': 'fixed'}
               for n in RHO})


def test_fixed_at_a_value_of_its_own():
    """A 'fixed' parameter takes the value given for that dataset; one
    given for a parameter that is fitted there is ignored; the values are
    one row per dataset, also when given as one row for all."""
    jp = JointProblem([(n, FitProblem(stack(), data(r)))
                       for n, r in RHO.items()],
                      {'A': {'Fe.thickness': 'fixed'},
                       'B': {'Fe.thickness': 'fixed'}},
                      values={'B': {'Fe.thickness': 70.0,
                                    'Fe.MSLD_rho': 9e-6}})
    assert jp.names == ['Fe.MSLD_rho']
    assert jp.fixed.shape == (2, 2)
    assert np.allclose(jp.part_x([4e-6], 0), [75.0, 4e-6])   # current
    assert np.allclose(jp.part_x([4e-6], 1), [70.0, 4e-6])   # its own
    with pytest.raises(ValueError, match='unknown parameter'):
        JointProblem([(n, FitProblem(stack(), data(r)))
                      for n, r in RHO.items()],
                     {'B': {'Fe.thickness': 'fixed'}},
                     values={'B': {'nope': 1.0}})
    one = JointProblem([(n, FitProblem(stack(), data(r)))
                        for n, r in RHO.items()],
                       {'B': {'Fe.thickness': 'fixed'}}, fixed=[60.0, 1e-6])
    assert np.allclose(one.fixed, [[60.0, 1e-6], [60.0, 1e-6]])


def test_single_problem_helpers():
    fp = FitProblem(stack(), data(5e-6), 'chi2')
    assert parts(fp) == [fp] and part_index(fp) == [0, 1]
    assert np.array_equal(part_x(fp, [1.0, 2.0, 3.0]), [1.0, 2.0])


def test_de_finds_each_dataset_its_own_value():
    """Two magnetisations, one thickness: each dataset's own rho and the
    shared thickness are found, where one shared rho cannot fit both."""
    modes = {n: {'Fe.MSLD_rho': 'own'} for n in RHO}
    for cost in ('chi2', 'log'):
        jp = joint(modes, cost)
        x, c, *_ = run_de(jp, maxiter=60, seed=1, tol=1e-8)
        assert x[0] == pytest.approx(80.0, abs=0.05)
        assert x[1] == pytest.approx(RHO['A'], rel=1e-3)
        assert x[2] == pytest.approx(RHO['B'], rel=1e-3)
        assert c < 1e-6
    shared = joint()
    _, c, *_ = run_de(shared, maxiter=60, seed=1)
    assert c > 1.0


def test_dream_save_load_and_bands(tmp_path):
    jp = joint({'B': {'Fe.MSLD_rho': 'own', 'Fe.thickness': 'fixed'}})
    pp = PosteriorProblem(jp, sigma_free=True)
    assert pp.names == ['Fe.thickness', 'Fe.MSLD_rho', 'Fe.MSLD_rho [B]',
                        'log10_sigma_scale']
    res = sample(pp, **OLD, n_gen=200, seed=0, stop_when_converged=False)
    save_result(tmp_path / 'j.npz', res, pp)
    back, problem, meta = load_run(tmp_path / 'j.npz')
    assert meta['datasets'] == ['A', 'B']
    assert problem.names == pp.names
    assert np.allclose(problem.fit.fixed, jp.fixed)
    x = res.best()[0]
    assert problem(x) == pytest.approx(pp(x), rel=1e-12)
    # each dataset's bands: its own channels and stack
    for k in (0, 1):
        b = predictive_bands(res, pp, n=20, part=k, replicates=2)
        assert [c['name'] for c in b['channels']] == ['channel 0',
                                                      'channel 1']
        assert np.array_equal(b['channels'][0]['R'],
                              jp.parts[k].channels[0][2])
        assert b['channels'][0]['bands'].shape == (5, len(
            b['channels'][0]['Q']))
    _, nuc_a, mag_a = profile_bands(res, pp, n=20, part=0)
    _, nuc_b, mag_b = profile_bands(res, pp, n=20, part=1)
    assert not np.allclose(mag_a, mag_b)


# -- GUI ------------------------------------------------------------------
APP = []                  # the QApplication, kept alive


def gui():
    """Simulation tab with layer 1's thickness and MSLD rho varied, and an
    Experimental tab holding datasets A and B (R+, R-)."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    APP.append(QtWidgets.QApplication.instance()
               or QtWidgets.QApplication([]))
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    for a in ('thickness', 'MSLD_rho'):
        sim.stack.layers[1].fit_entry(a)['vary'] = True
    sim.recompute()
    exp = ExperimentalTab(simulation=sim)
    Q = np.linspace(0.01, 0.1, 40)
    one = np.ones_like(Q)
    exp.sets = [dict(name=n, visible=True, channels={
        ch: dict(Q=Q, R=0.1 * one, dR=0.01 * one, dQ=0 * one,
                 theta=np.nan * one, path='%s%s.dat' % (n, ch))
        for ch in ('+', '-')}) for n in ('A', 'B')]
    exp.refresh_tree()
    return sim, exp


def choose_joint(exp, modes, values=None):
    from fit.panel import JOINT
    exp._joint = dict(sets=['A', 'B'], modes=modes)
    if values is not None:
        exp._joint['values'] = values
    exp._refresh_fit_sets()
    exp.fit_set.setCurrentIndex(exp.fit_set.findData(JOINT))


def test_dialog_selection():
    """A row per parameter: "All datasets" sets every ticked dataset, a
    dataset's column one of them (All then reads Mixed)."""
    gui()
    from PyQt6 import QtCore
    from fit.datasets_dialog import DatasetsDialog
    params = ['L.thickness', 'L.MSLD_rho']
    dlg = DatasetsDialog(['A', 'B', 'C'], ['A'], {}, params)
    assert dlg.chosen() == ['A'] and not dlg.table.isEnabled()
    dlg.list.item(1).setCheckState(QtCore.Qt.CheckState.Checked)
    assert dlg.table.isEnabled() and dlg.table.columnCount() == 4
    assert dlg.table.horizontalHeaderItem(3).text() == 'B'
    every = dlg.all_combos[1]
    every.setCurrentIndex(every.findData('own'))       # own in each
    assert [dlg.cells[n, 'L.MSLD_rho'].currentData() for n in 'AB'] == \
        ['own', 'own']
    assert dlg.selection() == {'sets': ['A', 'B'],
                               'modes': {'A': {'L.MSLD_rho': 'own'},
                                         'B': {'L.MSLD_rho': 'own'}},
                               'values': {}}
    a = dlg.cells['A', 'L.MSLD_rho']
    a.setCurrentIndex(a.findData('shared'))            # A alone
    assert every.currentData() == 'mixed'
    assert dlg.selection()['modes'] == {'B': {'L.MSLD_rho': 'own'}}
    # a third dataset joins: shared until set, so still mixed
    dlg.list.item(2).setCheckState(QtCore.Qt.CheckState.Checked)
    assert dlg.table.columnCount() == 5
    assert dlg.cells['B', 'L.MSLD_rho'].currentData() == 'own'
    assert dlg.all_combos[1].currentData() == 'mixed'
    dlg.all_combos[0].setCurrentIndex(dlg.all_combos[0].findData('fixed'))
    assert {dlg.mode(n, 'L.thickness') for n in 'ABC'} == {'fixed'}


def test_dialog_values_of_not_fitted():
    """Beside "Not fitted", a box in the Simulation tab's units: empty
    keeps the Simulation tab's value (shown greyed), a number is that
    dataset's value, "All datasets" writes every one, text is refused."""
    gui()
    from PyQt6 import QtCore
    from fit.datasets_dialog import DatasetsDialog
    params = ['L.thickness', 'L.MSLD_rho']
    info = {'L.thickness': (1.0, 'Å', 75.0),
            'L.MSLD_rho': (1e6, '10⁻⁶ Å⁻²', 2e-6)}
    dlg = DatasetsDialog(['A', 'B'], ['A', 'B'], {}, params, info=info,
                         values={'A': {'L.thickness': 60.0}})
    dlg.show()
    e = dlg.edits['B', 'L.MSLD_rho']
    assert not e.isVisible() and e.placeholderText() == '2'
    c = dlg.cells['B', 'L.MSLD_rho']
    c.setCurrentIndex(c.findData('fixed'))
    assert e.isVisible()
    e.setText('3.5')                                   # 10⁻⁶ Å⁻²
    e.editingFinished.emit()
    assert dlg.values['B']['L.MSLD_rho'] == pytest.approx(3.5e-6)
    # A's thickness value is kept but unused until it is not fitted
    assert dlg.selection()['values'] == {'B': {'L.MSLD_rho': 3.5e-6}}
    every = dlg.all_combos[0]
    every.setCurrentIndex(every.findData('fixed'))
    assert dlg.edits['A', 'L.thickness'].text() == '60'
    assert dlg.all_edits[0].isVisible() and dlg.all_edits[0].text() == ''
    dlg.all_edits[0].setText('80')
    dlg.all_edits[0].editingFinished.emit()
    assert [dlg.edits[n, 'L.thickness'].text() for n in 'AB'] == ['80'] * 2
    dlg.edits['A', 'L.thickness'].setText('')          # back to current
    dlg.edits['A', 'L.thickness'].editingFinished.emit()
    assert dlg.all_edits[0].text() == ''
    assert dlg.selection()['values'] == {
        'B': {'L.thickness': 80.0, 'L.MSLD_rho': pytest.approx(3.5e-6)}}
    seen = []
    import fit.datasets_dialog as D
    orig = D.QtWidgets.QMessageBox.information
    D.QtWidgets.QMessageBox.information = staticmethod(
        lambda *a: seen.append(a[2]))
    try:
        dlg.edits['B', 'L.thickness'].setText('abc')
        dlg.accept()
    finally:
        D.QtWidgets.QMessageBox.information = orig
    assert 'Not a number' in seen[0] and dlg.result() == 0
    assert 'border' in dlg.edits['B', 'L.thickness'].styleSheet()
    dlg.edits['B', 'L.thickness'].setText('81')        # typed, not entered
    dlg.accept()
    assert dlg.result() == 1 and dlg.values['B']['L.thickness'] == 81.0
    assert QtCore is not None


def test_joint_entry_problem_and_session(tmp_path):
    sim, exp = gui()
    pn = parameter_names_of(sim)
    choose_joint(exp, {'B': {pn[1]: 'own'}})
    assert exp.fit_set.currentText() == 'Joint: A + B'
    problem, inputs, *_ = exp._problem(exp.fit_report, 'chi2')
    assert isinstance(problem, JointProblem)
    assert problem.names == [pn[0], pn[1], pn[1] + ' [B]']
    assert [s['name'] for s, *_ in inputs] == ['A', 'B']
    # one dataset chosen: a plain fit of it
    exp.fit_set.setCurrentIndex(exp.fit_set.findText('B'))
    problem, *_ = exp._problem(exp.fit_report, 'chi2')
    assert isinstance(problem, FitProblem)
    # kept in the session, and through a rename
    exp.fit_set.setCurrentIndex(exp.fit_set.count() - 1)
    state = exp.session_state(str(tmp_path))
    assert state['fit']['joint']['sets'] == ['A', 'B']
    exp.restore_state(state, str(tmp_path))
    assert exp.fit_set.currentText() == 'Joint: A + B'
    exp._rename_fit_set('B', 'B2')
    exp.sets[1]['name'] = 'B2'
    exp.refresh_tree()
    assert exp.fit_set.currentText() == 'Joint: A + B2'
    assert exp._joint['modes'] == {'B2': {pn[1]: 'own'}}
    # a value for a parameter not fitted in B reaches the problem and the
    # session, and follows a rename
    exp.sets[1]['name'] = 'B'
    exp.refresh_tree()
    choose_joint(exp, {'B': {pn[0]: 'fixed'}}, {'B': {pn[0]: 42.0}})
    problem, *_ = exp._problem(exp.fit_report, 'chi2')
    assert problem.names == [pn[0], pn[1]]
    assert problem.part_x(problem.x0() * 0 + problem.lo, 1)[0] == 42.0
    state = exp.session_state(str(tmp_path))
    assert state['fit']['joint']['values'] == {'B': {pn[0]: 42.0}}
    exp.restore_state(state, str(tmp_path))
    exp._rename_fit_set('B', 'B3')
    assert exp._joint['values'] == {'B3': {pn[0]: 42.0}}
    exp.sets[1]['name'] = 'B3'
    exp.refresh_tree()
    # a dataset gone: no joint fit left
    del exp.sets[1]
    exp.refresh_tree()
    assert exp._joint is None and exp.fit_set.currentText() == 'A'


def parameter_names_of(sim):
    from fit.problem import parameter_names
    return parameter_names(sim.stack, sim.stack.free_parameters())


def test_windows_have_a_tab_per_dataset():
    sim, exp = gui()
    pn = parameter_names_of(sim)
    choose_joint(exp, {'B': {pn[1]: 'own', pn[0]: 'fixed'}})
    from fit.de.de_window import DEWindow
    from tabs.experimental import model_grid
    problem, inputs, qmin, qmax, _ = exp._problem(exp.fit_report, 'chi2')
    win = DEWindow()
    win.start(problem, [dict(name=s['name'], data={}, grids=model_grid(
        sim.stack, sim.last_Q, s['channels'], grid),
        vectors=dict(exp.sim_vectors(), **vec))
        for s, vec, data, grid in inputs], 'A + B', 'χ²')
    assert [win.tabs.tabText(k) for k in range(win.tabs.count())] == \
        ['A', 'B']
    assert win.views[0].drawn is not None and win.views[1].drawn is None
    x = np.array([90.0, 2e-6, 3e-6])
    win._append('1', 0.5, x)
    win.tabs.setCurrentIndex(1)              # drawn when shown
    view = win.views[1]
    assert view.stack.layers[1].MSLD_rho == pytest.approx(3e-6)
    # B's thickness is not fitted: the value the fit started from
    assert view.stack.layers[1].thickness == pytest.approx(problem.fixed[1, 0])
    assert win.params.topLevelItem(2).text(4) == 'B'
    sent = []
    win.sendRequested.connect(lambda *a: sent.append(a))
    win.running = False
    win._send()
    assert sent[0][2] == 1
    win.close()

    # DREAM: Summary, Traces, Corner, then one tab per dataset
    from fit.dream.dream_window import DreamWindow
    pp = PosteriorProblem(problem)
    res = sample(pp, **OLD, n_gen=120, seed=0, stop_when_converged=False)
    dw = DreamWindow()
    dw.start(pp, 120, 'joint')
    assert [dw.tabs.tabText(k) for k in range(dw.tabs.count())] == \
        ['Summary', 'Traces', 'Corner', 'A', 'B']
    assert not dw.part_box.isHidden()
    dw.finish(res)
    dw.tabs.setCurrentIndex(4)
    page, pred, prof, view, _ = dw.part_pages[1]
    assert page.currentIndex() == 1, page.widget(0).text()
    # one plot at a time: Reflectance first, then SLD, then SA if any
    plots = prof.parentWidget()
    assert view.currentText() == 'Reflectance' and \
        plots.currentWidget() is pred and pred.which == 'R'
    view.setCurrentText('SLD')
    assert plots.currentWidget() is prof
    sa = view.model().item(2).isEnabled()
    assert sa == pred.has_asymmetry()
    if sa:
        view.setCurrentText('SA')
        assert plots.currentWidget() is pred and pred.which == 'SA'
    got = []
    dw.boundsRequested.connect(lambda *a: got.append(a))
    dw.part_box.setCurrentIndex(1)
    dw._apply_bounds()
    assert got[0][2] == 1 and len(got[0][0]) == 3
    dw.close()


def test_each_dataset_its_own_dq_table():
    """The 'data' resolution mode with A's dQ chosen: in a joint fit B is
    fitted (points without a dQ, its curves) with its own dQ table."""
    sim, exp = gui()
    for n, dq in (('A', 1e-4), ('B', 5e-4)):
        s = next(t for t in exp.sets if t['name'] == n)
        for d in s['channels'].values():
            d['dQ'] = dq * np.ones_like(d['Q'])
    sim.set_data_dq('A', [0.01, 0.1], [1e-4, 1e-4])
    sim.recompute()
    choose_joint(exp, {})
    problem, *_ = exp._problem(exp.fit_report, 'chi2')
    a, b = (p.stack for p in problem.parts)
    assert a.resolution['data_set'] == 'A'
    assert b.resolution['data_set'] == 'B'
    assert b.resolution_sigma(np.array([0.05]))[0] == pytest.approx(5e-4)
    # the Simulation tab keeps A's
    assert sim.stack.resolution['data_set'] == 'A'
