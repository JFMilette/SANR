"""A session keeps the ended fit and DREAM tabs (fit.panel SESSION): each
comes back with its problem, its whole history and its report, can send to
the Simulation tab and is not continued.

Run from the repo root: python -m pytest tests
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
from fit.joint import JointProblem                              # noqa: E402
from fit.problem import FitProblem                              # noqa: E402
from fit.saved import problem_from_state, problem_state         # noqa: E402
from tabs import ExperimentalTab, SimulationTab                 # noqa: E402

Q = np.linspace(0.01, 0.2, 80)


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def channel(scale):
    one = np.ones_like(Q)
    R = scale * np.exp(-Q / 0.03)
    return dict(Q=Q, R=R, dR=0.05 * R, dQ=0.02 * Q, theta=np.nan * one,
                path='x.dat')


def tab():
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    exp.sets = [dict(name='d', visible=True,
                     channels={'+': channel(1.0), '-': channel(0.8)})]
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    exp.fit_workers.setValue(1)
    tabs = QtWidgets.QTabWidget()
    tabs.addTab(QtWidgets.QWidget(), 'tool')
    exp.page_tabs = tabs
    return exp


def through_file(states):
    """As a session file would give them back."""
    return json.loads(json.dumps(states))


def test_problem_round_trip(app):
    """The same cost, parameters and points, joint fits included."""
    exp = tab()
    exp.fit_floor.setValue(2.0)
    problem, *_ = exp._problem(exp.fit_report, 'chi2', 'sum')
    again = problem_from_state(through_file(problem_state(problem)))
    x = problem.to_u(problem.x0()) * 0.9 + 0.05
    assert again.names == problem.names and again.target == 'sum'
    assert again.npoints == problem.npoints
    assert again(x) == problem(x)
    second = FitProblem(problem.stack,
                        [dict(e, R=1.1 * e['R'])
                         for e in problem.inputs['data']], 'chi2',
                        target='sum')
    joint = JointProblem([('a', problem), ('b', second)],
                         {'b': {problem.names[0]: 'own'}})
    again = problem_from_state(through_file(problem_state(joint)))
    x = joint.to_u(joint.x0()) * 0.8 + 0.1
    assert again.names == joint.names and again(x) == joint(x)


def wait_fit(app, exp):
    t0 = time.monotonic()
    while exp._fit is not None and time.monotonic() - t0 < 60:
        app.processEvents()


def test_fit_tab_round_trip(app):
    exp = tab()
    exp.lb_maxiter.setValue(20)
    exp.start_lbfgs()
    wait_fit(app, exp)
    win = exp.fit_window
    win.history.setCurrentItem(win.history.topLevelItem(1))
    states, running = exp.fit_tabs_state()
    assert running == [] and [d['kind'] for d in states] == ['fit']

    other = tab()
    assert other.restore_fit_tabs(through_file(states)) == []
    tabs = other.page_tabs
    assert tabs.count() == 2 and tabs.currentIndex() == 0
    back = tabs.widget(1)
    assert tabs.tabText(1) == exp.page_tabs.tabText(1)
    assert [e[0] for e in back.entries] == [e[0] for e in win.entries]
    assert [e[1] for e in back.entries] == [e[1] for e in win.entries]
    assert all(np.array_equal(a[2], b[2])
               for a, b in zip(back.entries, win.entries))
    assert back.history.currentItem() is back.history.topLevelItem(1)
    assert back.status.text() == win.status.text()
    assert back.history.headerItem().text(1) == \
        win.history.headerItem().text(1)
    assert not back.running and not back.btn_stop.isEnabled()
    assert not back.btn_continue.isEnabled() and back.btn_send.isEnabled()
    assert back is not other.fit_window
    # it sends its entry, and new tabs are numbered after it
    back.btn_send.click()
    assert other.simulation.stack.layers[1].thickness == \
        pytest.approx(win.entries[1][2][0])
    other.start_lbfgs()
    wait_fit(app, other)
    assert other.page_tabs.tabText(2).startswith('L-BFGS-B 2 · ')
    other.close_page(back)
    assert other.page_tabs.count() == 2


def test_running_tab_left_out(app):
    exp = tab()
    exp.lb_maxiter.setValue(20)
    exp.start_lbfgs()
    states, running = exp.fit_tabs_state()
    assert states == [] and running == [exp.fit_window.tab_name]
    wait_fit(app, exp)


def test_dream_tab_round_trip(app):
    from fit.dream.dream_fit import PosteriorProblem, sample
    exp = tab()
    fit, *_ = exp._problem(exp.dream_report, 'chi2', 'channels')
    pp = PosteriorProblem(fit)
    res = sample(pp, n_gen=200, seed=1, stop_when_converged=False)
    exp._make_dream_window(['d'])
    win = exp.dream_window
    win.start(pp, 200, 'header of the run', ['x.dat'], ['d'])
    win.finish(res)
    exp._reveal(win)
    states, running = exp.fit_tabs_state()
    assert running == [] and [d['kind'] for d in states] == ['dream']

    other = tab()
    assert other.restore_fit_tabs(through_file(states)) == []
    back = other.page_tabs.widget(1)
    assert other.page_tabs.tabText(1) == exp.page_tabs.tabText(1)
    assert np.array_equal(back.result.chains, res.chains)
    assert np.array_equal(back.result.logp, res.logp)
    assert back.header.text() == 'header of the run'
    assert back.problem.names == pp.names
    x = res.chains[-1, 0]
    assert back.problem(x) == pp(x)              # dQ and all, as it ran
    assert not back.btn_continue.isEnabled() and back.btn_save.isEnabled()
    assert back is not other.dream_window
