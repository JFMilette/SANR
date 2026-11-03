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


def test_import_makes_a_model(qapp):
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    i = exp.new_set('S1 4K 0.1T', refresh=False)
    exp._model_for_import(i)
    assert exp.sets[i]['link'] == 'S1 4K 0.1T'
    assert 'S1 4K 0.1T' in sim.model_names()
    assert sim.shown_model() == 'Model 1'          # not shown
    assert sim.stored_model('S1 4K 0.1T')['layers'] == \
        sim.model_dict('Model 1')['layers']
    # a model of that name already: linked, not made again
    j = exp.new_set('again', refresh=False)
    exp.sets[j]['name'] = 'S1 4K 0.1T'
    exp._model_for_import(j)
    assert sim.model_names().count('S1 4K 0.1T') == 1


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
