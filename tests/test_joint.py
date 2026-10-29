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


def test_models_with_other_parameters():
    """Each dataset fitted with its own stack: parameters matched by name,
    a dataset without one shows '—' in the panel and is simply not fitted
    there; 'fixed' keeps that dataset's own value; a shared parameter takes
    its bounds and start from the dataset `sources` names."""
    a = stack()
    b = stack()
    b.layers[1].thickness = 85.0
    b.layers[1].fit['thickness'] = {'vary': True, 'min': 70, 'max': 95}
    b.layers[1].fit['MSLD_rho']['vary'] = False
    b.layers[1].fit['NSLD_real'] = {'vary': True, 'min': 7e-6, 'max': 9e-6}
    pa, pb = FitProblem(a, data(RHO['A'])), FitProblem(b, data(RHO['B']))
    jp = JointProblem([('A', pa), ('B', pb)])
    assert jp.names == ['Fe.thickness', 'Fe.MSLD_rho', 'Fe.NSLD_real']
    assert jp.index == [[0, 1], [0, 2]]
    # the shared thickness: A's bounds and start by default, B's on request
    assert jp.params[0][2:] == (75.0, 60, 100)
    jb = JointProblem([('A', pa), ('B', pb)], sources={'Fe.thickness': 'B'})
    assert jb.params[0][2:] == (85.0, 70, 95) and jb.origin[0] == (1, 0)
    # fixed in B: B's own value
    jf = JointProblem([('A', pa), ('B', pb)], {'B': {'Fe.thickness': 'fixed'}})
    assert jf.names == ['Fe.thickness', 'Fe.MSLD_rho', 'Fe.NSLD_real']
    assert np.allclose(jf.part_x([80.0, 5e-6, 8e-6], 1), [85.0, 8e-6])
    assert len(jf.fixed[0]) == 2 and len(jf.fixed[1]) == 2
    one = JointProblem([(n, FitProblem(stack(), data(r)))
                        for n, r in RHO.items()],
                       {'B': {'Fe.thickness': 'fixed'}}, fixed=[60.0, 1e-6])
    assert np.allclose(one.fixed, [[60.0, 1e-6], [60.0, 1e-6]])
    # through a session
    from fit.saved import problem_from_state, problem_state
    again = problem_from_state(problem_state(jb))
    assert again.names == jb.names and again.origin == jb.origin
    x = jb.to_u(jb.x0())
    assert again(x) == jb(x)


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


def choose_joint(exp, modes):
    from fit.panel import JOINT
    exp._joint = dict(sets=['A', 'B'], modes=modes)
    exp._refresh_fit_sets()
    exp.fit_set.setCurrentIndex(exp.fit_set.findData(JOINT))


def test_dialog_selection():
    """A row per parameter any ticked dataset's model has, its Fit tick
    asked of fit.panel; for ticked rows, "All datasets" sets every ticked
    dataset, a dataset's column one of them (All then reads Mixed), '—'
    where its model lacks the parameter; From lists the models sharing
    it."""
    gui()
    from PyQt6 import QtCore
    from fit.datasets_panel import SHOWN, DatasetsPanel
    params = ['L.thickness', 'L.MSLD_rho', 'L.NSLD_real']
    present = {'A': set(params[:2]), 'B': set(params), 'C': set(params)}
    dlg = DatasetsPanel()
    dlg.load(['A', 'B', 'C'], ['A'], ['M1', 'M2'], {'B': 'M2'}, params,
             present, {params[0]: True, params[1]: True, params[2]: None})
    assert dlg.chosen() == ['A'] and dlg.table.columnCount() == 2
    assert dlg.list.cellWidget(1, 1).currentText() == 'M2'
    assert dlg.list.cellWidget(0, 1).currentText() == SHOWN
    asked = []
    dlg.fitToggled.connect(lambda names, on: asked.append((names, on)))
    dlg.table.item(2, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    assert asked == [(['L.NSLD_real'], True)]       # partly: ticked all
    dlg.list.item(1, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    assert dlg.table.columnCount() == 6
    assert dlg.table.horizontalHeaderItem(5).text() == 'B'
    assert dlg.table.item(2, 4).text() == '—'       # A has no NSLD_real
    every = dlg.all_combos['L.MSLD_rho']
    every.setCurrentIndex(every.findData('own'))       # own in each
    assert [dlg.cells[n, 'L.MSLD_rho'].currentData() for n in 'AB'] == \
        ['own', 'own']
    assert dlg.selection() == {'sets': ['A', 'B'],
                               'modes': {'A': {'L.MSLD_rho': 'own'},
                                         'B': {'L.MSLD_rho': 'own'}},
                               'sources': {}}
    a = dlg.cells['A', 'L.MSLD_rho']
    a.setCurrentIndex(a.findData('shared'))            # A alone
    assert every.currentData() == 'mixed'
    assert dlg.selection()['modes'] == {'B': {'L.MSLD_rho': 'own'}}
    # the shared thickness: from either model
    f = dlg.from_combos['L.thickness']
    assert [f.itemText(k) for k in range(f.count())] == [SHOWN, 'M2']
    f.setCurrentText('M2')
    assert dlg.selection()['sources'] == {'L.thickness': 'M2'}
    # relinking B is reported
    got = []
    dlg.changed.connect(lambda: got.append(dlg.link('B')))
    dlg.list.cellWidget(1, 1).setCurrentText(SHOWN)
    assert got == [None]
    # the last ticked dataset stays ticked
    dlg.list.item(0, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    dlg.list.item(1, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert dlg.chosen() == ['B']


def test_fit_ticks_go_to_the_models():
    """The panel's Fit tick ticks the parameter in every model of the
    datasets fitted; Untick all unticks them; the Simulation tab has no Fit
    box of its own left."""
    from PyQt6 import QtCore
    sim, exp = gui()
    sim.copy_model()                        # 'Model 1 copy', shown
    exp.sets[1]['link'] = 'Model 1'
    choose_joint(exp, {})
    panel = exp.datasets_panel
    p = 'L1.NSLD_real'
    assert panel.varied[p] is False
    panel.table.item(panel.params.index(p), 0).setCheckState(
        QtCore.Qt.CheckState.Checked)
    APP[0].processEvents()
    assert sim.stack.layers[1].fit['NSLD_real']['vary']
    assert sim.stored_model('Model 1')['layers'][1]['fit']['NSLD_real'][
        'vary']
    assert panel.varied[p] is True
    exp._tick_requested('untick')
    APP[0].processEvents()
    assert not any(panel.varied.values())
    assert not sim.stack.free_parameters()
    assert all(cb.isHidden() for *_, cb, _ in sim.editor.rows.values())


def test_panel_follows_and_sets_the_dataset_box():
    """Ticking a second dataset makes the joint entry; picking one dataset
    in the Dataset box ticks it alone; picking the joint entry again
    brings its modes back."""
    from PyQt6 import QtCore
    from fit.panel import JOINT
    sim, exp = gui()
    exp.fit_set.setCurrentIndex(exp.fit_set.findText('A'))
    panel = exp.datasets_panel
    assert panel.chosen() == ['A']
    ticked = [n for n, v in panel.varied.items() if v]
    assert len(ticked) == 2
    panel.list.item(1, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    APP[0].processEvents()                  # B's parameters: reloaded
    assert exp.fit_set.currentData() == JOINT
    assert exp._joint['sets'] == ['A', 'B']
    p = ticked[1]
    c = panel.cells['B', p]
    c.setCurrentIndex(c.findData('own'))
    assert exp._joint['modes'] == {'B': {p: 'own'}}
    exp.fit_set.setCurrentIndex(exp.fit_set.findText('B'))
    assert panel.chosen() == ['B']
    exp.fit_set.setCurrentIndex(exp.fit_set.findData(JOINT))
    assert panel.chosen() == ['A', 'B']
    assert panel.cells['B', p].currentData() == 'own'
    # a parameter ticked Fit: a ticked row more
    sim.stack.layers[1].fit_entry('NSLD_real')['vary'] = True
    sim.recompute()
    assert len([n for n, v in panel.varied.items() if v]) == 3
    assert panel.cells['B', p].currentData() == 'own'


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
    # a link and a From are kept in the session
    sim.copy_model()
    exp.sets[1]['link'] = 'Model 1'
    exp._joint['sources'] = {pn[0]: 'Model 1'}
    state = exp.session_state(str(tmp_path))
    exp.restore_state(state, str(tmp_path))
    assert exp.sets[1]['link'] == 'Model 1'
    assert exp._joint['sources'] == {pn[0]: 'Model 1'}
    problem, *_ = exp._problem(exp.fit_report, 'chi2')
    assert [p.model_name for p in problem.parts] == ['Model 1 copy',
                                                     'Model 1']
    assert problem.sources == {pn[0]: 'B2'}
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
    assert view.stack.layers[1].thickness == pytest.approx(problem.fixed[1][0])
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


def test_send_to_a_model_per_dataset():
    """A result goes to the model named after each dataset: made (a copy
    of the model it was fitted with) when missing, overwritten when there;
    each dataset is then linked to it, so its next fit starts there.
    Revert removes what a send made and restores the links."""
    sim, exp = gui()
    pn = parameter_names_of(sim)
    choose_joint(exp, {'B': {pn[0]: 'own'}})
    problem, *_ = exp._problem(exp.fit_report, 'chi2')
    exp._set_fit_start(problem)
    x = np.array([90.0, 95.0, 2e-6])          # thickness, thickness [B], rho
    sent = exp._send_values(problem, x, 0, exp, 'test')
    assert sent == ['A', 'B'] and sim.model_names() == ['Model 1', 'A', 'B']
    assert sim.model_stack('A').layers[1].thickness == 90.0
    assert sim.model_stack('B').layers[1].thickness == 95.0
    assert sim.model_stack('B').layers[1].MSLD_rho == pytest.approx(2e-6)
    assert sim.stack.layers[1].thickness != 90.0      # Model 1 untouched
    assert [s.get('link') for s in exp.sets[:2]] == ['A', 'B']
    assert [s['name'] for s in exp.sets if s.get('live')] == [
        'Model 1 (simulation)', 'A (simulation)', 'B (simulation)']
    # the next fit starts from each dataset's own result; a send overwrites
    problem, *_ = exp._problem(exp.fit_report, 'chi2')
    assert np.allclose(problem.to_x(problem.x0()), x)
    exp._send_values(problem, x + [1, 1, 0], 0, exp, 'test')
    assert sim.model_names() == ['Model 1', 'A', 'B']
    assert sim.model_stack('B').layers[1].thickness == 96.0
    exp.revert_fit()
    assert sim.model_names() == ['Model 1']
    assert [s.get('link') for s in exp.sets[:2]] == [None, None]
    assert [s['name'] for s in exp.sets if s.get('live')] == [
        'Model 1 (simulation)']


def test_one_dataset_shows_its_model():
    """Fitting a dataset linked to a model the Simulation tab does not show
    shows it first; its result goes to the model named after it."""
    sim, exp = gui()
    sim.copy_model()                          # 'Model 1 copy', shown
    exp.sets[0]['link'] = 'Model 1'
    exp.fit_set.setCurrentIndex(exp.fit_set.findText('A'))
    problem, *_ = exp._problem(exp.fit_report, 'chi2')
    assert sim.shown_model() == 'Model 1' and problem.model_name == 'Model 1'
    exp._send_values(problem, problem.to_x(problem.x0()), 0, exp, 'test')
    assert sim.shown_model() == 'A'


def test_live_dataset_per_model():
    """A live dataset per model, last in the list, unticked when made;
    renamed and removed with its model; not removable by hand."""
    sim, exp = gui()
    live = lambda: [s['name'] for s in exp.sets if s.get('live')]
    assert live() == ['Model 1 (simulation)']
    sim.copy_model()
    assert live() == ['Model 1 (simulation)', 'Model 1 copy (simulation)']
    assert not exp.sets[-1]['visible'] and exp.sets[-1]['channels']
    sim.add_model('X', sim.model_dict('Model 1'))
    sim._models[-1]['name']                    # renamed through the box
    sim.modelRenamed.emit('X', 'Y')
    sim._models[-1]['name'] = 'Y'
    sim._refresh_models()
    assert live()[-1] == 'Y (simulation)'
    sim.remove_model('Y')
    assert live() == ['Model 1 (simulation)', 'Model 1 copy (simulation)']
    exp.new_set('C')
    assert [s['name'] for s in exp.sets][:3] == ['A', 'B', 'C']
    exp.tree.setCurrentItem(exp.tree.topLevelItem(len(exp.sets) - 1))
    exp.remove_selected()
    assert len(live()) == 2
