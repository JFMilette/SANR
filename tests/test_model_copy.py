"""Models: one made for each imported dataset, and Copy to… (the shown
model's parameters into others, tabs.simulation MODELS).

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


@pytest.fixture(scope='module')
def qapp():
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def dataset(name, link=None):
    import numpy as np
    Q = np.linspace(0.01, 0.1, 20)
    one = np.ones_like(Q)
    d = dict(name=name, visible=True, channels={'+': dict(
        Q=Q, R=0.1 * one, dR=0.01 * one, dQ=0 * one, theta=np.nan * one,
        path='%s.dat' % name)})
    if link:
        d['link'] = link
    return d


def test_every_dataset_gets_its_model(qapp):
    """A model per imported dataset, also for datasets already there (an
    older session), as a copy of the model each was linked to; an empty
    dataset or a frozen simulation has none."""
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    sim.copy_model()                        # 'Model 1 copy', shown
    sim.stack.layers[1].thickness = 77.0
    sim.recompute()
    exp.sets = [dataset('A'), dataset('B', link='Model 1'),
                dict(name='empty', visible=True, channels={}),
                dict(dataset('frozen'), line=True)]
    exp.refresh_tree()
    assert [s.get('link') for s in exp.sets[:2]] == ['A', 'B']
    names = sim.model_names()
    assert 'A' in names and 'B' in names
    assert 'empty' not in names and 'frozen' not in names
    assert sim.shown_model() == 'Model 1 copy'      # not shown
    thick = {m: sim.stored_model(m)['layers'][1]['thickness']
             for m in ('A', 'B', 'Model 1')}
    assert thick['A'] == 77.0                       # the shown model's
    assert thick['B'] == thick['Model 1']           # its link's
    # data imported later into the empty dataset: its model then
    exp.sets[2] = dataset('empty')
    exp.refresh_tree()
    assert exp.sets[2]['link'] == 'empty'
    n = len(sim.model_names())
    exp.refresh_tree()
    assert len(sim.model_names()) == n              # made once


def test_dataset_model_follows_its_dataset(qapp, monkeypatch):
    """Not renamed or deleted in the Simulation tab; renamed with its
    dataset; removed with it."""
    from PyQt6 import QtCore, QtWidgets
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    exp.sets = [dataset('A'), dataset('B')]
    exp.refresh_tree()
    sim.switch_model(sim.model_names().index('A'))
    assert not sim.model_btns['rename'].isEnabled()
    assert not sim.model_btns['delete'].isEnabled()
    sim.rename_model()
    sim.delete_model()
    assert 'A' in sim.model_names()
    sim.switch_model(sim.model_names().index('Model 1'))
    assert sim.model_btns['rename'].isEnabled()
    # renamed with its dataset
    top = exp.tree.topLevelItem(0)
    top.setText(0, 'A2')
    qapp.processEvents()
    assert exp.sets[0]['name'] == 'A2' and exp.sets[0]['link'] == 'A2'
    assert 'A2' in sim.model_names() and 'A' not in sim.model_names()
    # removed with it
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', staticmethod(
        lambda *a: QtWidgets.QMessageBox.StandardButton.Yes))
    exp.tree.setCurrentItem(exp.tree.topLevelItem(1))
    exp.remove_selected()
    assert 'B' not in sim.model_names()
    assert [s['name'] for s in exp.sets if not s.get('live')] == ['A2']


def test_copy_parameters_to_models(qapp):
    from tabs import SimulationTab
    sim = SimulationTab()
    sim.copy_model()                       # 'Model 1 copy', shown
    sim.copy_model()                       # 'Model 1 copy copy', shown
    st = sim.stack
    st.layers[1].thickness = 87.5
    e = st.layers[1].fit_entry('thickness')
    e.update(min=80.0, max=95.0, vary=True,
             prior={'kind': 'gauss', 'mu': 88.0, 'tau': 2.0})
    sim._load_stack_fit()
    sim.refresh_list(select=1)
    name = st.layers[1].name
    n, missing = sim.copy_parameters(['Model 1'])
    assert n > 0 and missing == {'Model 1': []}
    d = sim.stored_model('Model 1')
    lay = next(l for l in d['layers'] if l['name'] == name)
    assert lay['thickness'] == 87.5
    assert (lay['fit']['thickness']['min'],
            lay['fit']['thickness']['max']) == (80.0, 95.0)
    assert not lay['fit']['thickness'].get('vary')      # not asked
    assert 'prior' not in lay['fit']['thickness']
    before = next(l for l in sim.stored_model('Model 1 copy')['layers']
                  if l['name'] == name)['thickness']
    sim.copy_parameters(['Model 1 copy'], {'ticks', 'priors'})
    d = sim.stored_model('Model 1 copy')
    lay = next(l for l in d['layers'] if l['name'] == name)
    assert lay['fit']['thickness']['vary']
    assert lay['fit']['thickness']['prior']['mu'] == 88.0
    assert lay['thickness'] == before              # values not asked
