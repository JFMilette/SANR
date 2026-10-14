"""Line colours: a change reaches the Simulation and Experimental tabs and
the DE window as it is made.

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
import line_colours                                             # noqa: E402
from test_live_model import tab                                 # noqa: E402
from fit.de.de_window import DEWindow                           # noqa: E402
from fit.problem import FitProblem                              # noqa: E402
from tabs.experimental import model_grid                        # noqa: E402

NEW = '#123456'


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def defaults():
    yield
    line_colours.set_colours(line_colours.DEFAULTS)


def pens(items):
    """Colours of the lines and markers of pyqtgraph items."""
    out = set()
    for it in items:
        opts = getattr(it, 'opts', {})
        for key in ('pen', 'symbolPen'):
            p = opts.get(key)
            if p is not None and not isinstance(p, (str, tuple)):
                out.add(p.color().name())
    return out


def test_set_colours_keeps_valid_entries(defaults):
    line_colours.set_colours({'++': NEW, '--': 'not a colour', 'x': NEW})
    assert line_colours.get('++') == NEW
    assert line_colours.get('--') == line_colours.DEFAULTS['--']
    assert 'x' not in line_colours.colours()


def test_plots_follow_a_change(app, defaults):
    sim, exp = tab()
    exp.add_simulation()
    line_colours.set_colours({'++': NEW, '+': NEW})
    assert sim.refl_curves[0].opts['pen'].color().name() == NEW
    assert sim.hp_curves[0].opts['pen'].color().name() == NEW
    assert NEW in pens(c for c, _ in exp._items)
    t = exp.tree
    names = {t.topLevelItem(i).child(j).foreground(0).color().name()
             for i in range(t.topLevelItemCount())
             for j in range(t.topLevelItem(i).childCount())}
    assert NEW in names

    s, vec, data, qmin, qmax, grid = exp._fit_inputs(exp.fit_report)
    fp = FitProblem(sim.stack, data, 'chi2', qmin, qmax, grid)
    win = DEWindow(exp)
    grids = model_grid(sim.stack, sim.last_Q, s['channels'], grid)
    win.start(fp, [dict(name='d', data=s['channels'],
                        vectors=dict(exp.sim_vectors(), **vec),
                        grids=grids)], 'd', 'χ²')
    line_colours.set_colours({'+': '#abcdef'})
    assert '#abcdef' in pens(win.views[0]._refl_items)
    win.close()


def test_dataset_colours(app, defaults, monkeypatch):
    sim, exp = tab()
    exp.add_simulation()
    QtGui = pytest.importorskip('PyQt6.QtGui')
    picked = iter(['#aa0000', '#00bb00'])
    monkeypatch.setattr(QtWidgets.QColorDialog, 'getColor',
                        staticmethod(lambda *a: QtGui.QColor(next(picked))))
    exp.pick_colour(0)                       # dataset 'd': every channel
    assert exp.series_colour(0, '+') == exp.series_colour(0, '-') == '#aa0000'
    exp.pick_colour(0, '-')                  # then its R- alone
    assert exp.series_colour(0, '-') == '#00bb00'
    assert exp.series_colour(0, '+') == '#aa0000'
    assert exp.series_colour(0, None) == '#aa0000'           # asymmetry
    assert exp.series_colour(1, '+') == line_colours.get('+')  # other set
    drawn = pens(c for c, _ in exp._items)
    assert {'#aa0000', '#00bb00', line_colours.get('+')} <= drawn
    # saved with the session
    state = exp.session_state(str(HERE))
    assert state['sets'][0]['colour'] == '#aa0000'
    assert state['sets'][0]['channel_colours'] == {'-': '#00bb00'}
    exp.restore_state(state, str(HERE))
    assert exp.series_colour(0, '-') == '#00bb00'
    assert exp.series_colour(0, '+') == '#aa0000'
    exp.reset_colours(0, '-')
    assert exp.series_colour(0, '-') == '#aa0000'
    exp.reset_colours(0)
    assert exp.series_colour(0, '-') == line_colours.get('-')
