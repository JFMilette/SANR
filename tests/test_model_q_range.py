"""Each model's own Q range (tabs.simulation MODELS): Q min, Q max and Q
points kept per model through switching, sessions and Copy to… (general
parameters); every live curve on its model's grid, every dataset drawn and
fitted within its model's Q range.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

Q = np.linspace(0.01, 0.1, 20)                 # every dataset's points


@pytest.fixture(scope='module')
def qapp():
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def dataset(name):
    one = np.ones_like(Q)
    return dict(name=name, visible=True, channels={'+': dict(
        Q=Q, R=0.1 * one, dR=0.01 * one, dQ=0 * one, theta=np.nan * one,
        path='%s.dat' % name)})


def set_range(sim, qmin, qmax, nq):
    """The boxes (the shown model's Q range), recomputed at once."""
    sim.qmin.setValue(qmin)
    sim.qmax.setValue(qmax)
    sim.nq.setValue(nq)
    sim._timer.stop()
    sim.recompute()


def show(sim, name):
    sim.switch_model(sim.model_names().index(name))


def tabs():
    """Simulation tab (layer 1's thickness varied) and an Experimental tab
    with datasets A and B, each with its own model (DATASET MODELS)."""
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    sim.recompute()
    exp = ExperimentalTab(simulation=sim)
    exp.sets = [dataset('A'), dataset('B')] + \
        [s for s in exp.sets if s.get('live')]
    exp.refresh_tree()
    assert [s['link'] for s in exp.sets[:2]] == ['A', 'B']
    return sim, exp


def test_each_model_keeps_its_q_range(qapp):
    from tabs import SimulationTab
    sim = SimulationTab()
    first = sim.shown_model()
    sim.copy_model()                         # takes the shown one's range
    copy = sim.shown_model()
    assert sim.q_settings(first) == sim.q_settings(copy)
    set_range(sim, 0.002, 0.5, 123)
    show(sim, first)
    assert (sim.qmin.value(), sim.qmax.value(), sim.nq.value()) == \
        (0.0001, 0.25, 400)
    assert sim.q_settings(copy) == dict(qmin=0.002, qmax=0.5, nq=123)
    assert np.array_equal(sim.q_grid(copy), np.linspace(0.002, 0.5, 123))
    show(sim, copy)
    assert (sim.qmin.value(), sim.qmax.value(), sim.nq.value()) == \
        (0.002, 0.5, 123)
    assert np.array_equal(sim.last_Q, np.linspace(0.002, 0.5, 123))
    # a model added from another takes that one's range
    name = sim.add_model('X', sim.model_dict(first), sim.q_settings(first))
    assert sim.q_settings(name) == sim.q_settings(first)


def test_session_keeps_each_range_and_old_sessions_share_one(qapp):
    from tabs import SimulationTab
    sim = SimulationTab()
    first = sim.shown_model()
    sim.copy_model()
    set_range(sim, 0.002, 0.5, 123)
    state = sim.session_state()
    again = SimulationTab()
    again.restore_state(state)
    assert again.q_settings(first) == dict(qmin=0.0001, qmax=0.25, nq=400)
    assert again.q_settings(again.shown_model()) == \
        dict(qmin=0.002, qmax=0.5, nq=123)
    # saved before each model had its own: all take the session's range
    for m in state['simulation']['models']['list']:
        m.pop('q', None)
    old = SimulationTab()
    old.restore_state(state)
    for name in old.model_names():
        assert old.q_settings(name) == dict(qmin=0.002, qmax=0.5, nq=123)
    show(old, first)
    assert (old.qmin.value(), old.qmax.value(), old.nq.value()) == \
        (0.002, 0.5, 123)


def test_copy_to_with_general_parameters_copies_the_range(qapp):
    from tabs import SimulationTab
    sim = SimulationTab()
    first = sim.shown_model()
    sim.copy_model()
    set_range(sim, 0.002, 0.5, 123)
    sim.copy_parameters([first], ('values', 'bounds'))
    assert sim.q_settings(first) == dict(qmin=0.0001, qmax=0.25, nq=400)
    sim.copy_parameters([first], ('values', 'general'))
    assert sim.q_settings(first) == dict(qmin=0.002, qmax=0.5, nq=123)


def test_curves_and_plot_follow_each_models_range(qapp):
    sim, exp = tabs()
    show(sim, 'B')
    set_range(sim, 0.02, 0.06, 50)
    show(sim, 'A')
    set_range(sim, 0.0001, 0.25, 400)
    for s in exp.sets:
        s['visible'] = s.get('link') in ('A', 'B')
    exp._update_live()
    live = {s['link']: s for s in exp.sets if s.get('live')}
    assert np.array_equal(live['A']['channels']['+']['Q'], sim.last_Q)
    assert np.array_equal(live['B']['channels']['+']['Q'],
                          np.linspace(0.02, 0.06, 50))
    # each dataset drawn within its own model's range
    drawn = {name: np.asarray(q) for _, name, q, *_ in exp._shown}
    a = [q for n, q in drawn.items() if n.startswith('A —')]
    b = [q for n, q in drawn.items() if n.startswith('B —')]
    b_live = [q for n, q in drawn.items() if n.startswith('B (simulation)')]
    assert a and b and b_live
    assert all(np.array_equal(q, Q) for q in a)
    assert all(np.array_equal(q, Q[(Q >= 0.02) & (Q <= 0.06)]) for q in b)
    assert all(q.min() >= 0.02 and q.max() <= 0.06 for q in b_live)


def test_joint_fit_takes_each_datasets_range(qapp):
    from fit.joint import parts
    from fit.panel import JOINT
    sim, exp = tabs()
    show(sim, 'B')
    set_range(sim, 0.0001, 0.05, 400)
    show(sim, 'A')
    exp._joint = dict(sets=['A', 'B'], modes={})
    exp._refresh_fit_sets()
    exp.fit_set.setCurrentIndex(exp.fit_set.findData(JOINT))
    problem, inputs, qmin, qmax, _ = exp._problem(exp.fit_report, 'chi2')
    a, b = parts(problem)
    assert a.npoints == len(Q)
    assert b.npoints == int(np.sum(Q <= 0.05))
    assert (qmin, qmax) == (0.0001, 0.25)        # the span, for reports
    # one dataset: its own range
    exp.fit_set.setCurrentIndex(exp.fit_set.findText('B'))
    problem, *_ = exp._problem(exp.fit_report, 'chi2')
    assert problem.npoints == int(np.sum(Q <= 0.05))
