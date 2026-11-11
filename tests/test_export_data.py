"""Export… of the Experimental tab (tabs.export_data): file names, folders,
column names with units, errors scaled with each scale, the SLD profile
with its roughness columns.

Run from the repo root: python -m pytest tests
"""
import csv
import os
import pathlib
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

Q = np.linspace(0.01, 0.1, 20)


@pytest.fixture(scope='module')
def qapp():
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def channels(spins=('++', '+-', '-+', '--')):
    one = np.ones_like(Q)
    R = {'++': 0.5, '+-': 0.01, '-+': 0.02, '--': 0.3, '+': 0.4}
    return {ch: dict(Q=Q, R=R[ch] * one, dR=0.1 * R[ch] * one,
                     dQ=0.001 * one, theta=np.nan * one, path='%s.dat' % ch)
            for ch in spins}


def tabs():
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    sim.stack.layers[2].MSLD_sigma = 7.5         # a magnetic roughness
    sim.recompute()
    exp = ExperimentalTab(simulation=sim)
    exp.sets = [dict(name='A/1', visible=True, channels=channels()),
                dict(name='X', visible=True, channels=channels(('+',)))] + \
        [s for s in exp.sets if s.get('live')]
    exp.refresh_tree()
    return sim, exp


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def test_scaled_errors():
    from tabs.export_data import scaled
    q, y, dy = np.array([0.1, 0.2]), np.array([1e-2, -1.0]), \
        np.array([1e-3, 1.0])
    v, e = scaled(q, y, dy, 'log')
    assert v[0] == pytest.approx(-2.0) and \
        e[0] == pytest.approx(1e-3 / (1e-2 * np.log(10)))
    assert np.isnan(v[1]) and np.isnan(e[1])         # R <= 0: no log
    v, e = scaled(q, y, dy, 'rq4')
    assert np.allclose(v, y * q**4) and np.allclose(e, dy * q**4)
    v, e = scaled(q, y, dy, 'R')
    assert np.array_equal(v, y) and np.array_equal(e, dy)


def test_column_names_have_units():
    from tabs.export_data import series_columns
    s = [('R++', None, Q, Q, Q, Q)]
    assert series_columns(s, 'R')[0] == [
        'R++ data Q (Å⁻¹)', 'R++ data', 'R++ data error',
        'R++ data dQ (Å⁻¹)']
    assert series_columns(s, 'log')[0][1:3] == [
        'log10(R++ data)', 'log10(R++ data) error']
    assert series_columns(s, 'rq4')[0][1:3] == [
        'R++ data·Q⁴ (Å⁻⁴)', 'R++ data·Q⁴ error (Å⁻⁴)']
    assert series_columns([('SA half-polarized', None, Q, Q, Q, Q)],
                          None)[0][1] == 'SA half-polarized data'
    # a model curve: its Q and value only
    assert series_columns(s, 'log', 'model')[0] == [
        'R++ model Q (Å⁻¹)', 'log10(R++ model)']


def test_files_folders_and_values(qapp, tmp_path):
    from tabs.export_data import QUANTITY_TAGS, export_datasets, write_csv
    sim, exp = tabs()
    quantities = list(QUANTITY_TAGS)
    # one dataset: in the folder itself, named with the name given
    plans, notes = export_datasets(exp, str(tmp_path), [0],
                                   lambda s: 'run', quantities,
                                   ['R', 'log'])
    names = sorted(os.path.basename(p) for p, *_ in plans)
    assert names == sorted(
        ['run_SLD_profile.csv'] +
        ['run_%s_%s.csv' % (t, sc) for t in
         ('polarized', 'half_polarized', 'non_polarized')
         for sc in ('R', 'logR')] +
        ['run_SA_NSF.csv', 'run_SF_fraction.csv', 'run_SA_SF.csv',
         'run_SA_half_polarized.csv'])
    assert all(os.path.dirname(p) == str(tmp_path) for p, *_ in plans)
    assert not notes
    for p, h, c in plans:
        write_csv(p, h, c)
    head, rows = read(tmp_path / 'run_polarized_logR.csv')
    assert head[:4] == ['R++ data Q (Å⁻¹)', 'log10(R++ data)',
                        'log10(R++ data) error', 'R++ data dQ (Å⁻¹)']
    assert float(rows[0][1]) == pytest.approx(np.log10(0.5))
    assert float(rows[0][2]) == pytest.approx(0.1 / np.log(10))
    # then the model's four curves, Q and value each, on its own grid
    assert head[16:] == [x for ch in ('R++', 'R+-', 'R-+', 'R--') for x in (
        '%s model Q (Å⁻¹)' % ch, 'log10(%s model)' % ch)]
    live = next(s for s in exp.sets if s.get('live') and s['link'] == 'A/1')
    grid = sim.q_grid('A/1')
    grid = grid[(grid >= Q[0] - 1) & (grid <= sim.q_range('A/1')[1])]
    assert len(rows) == max(len(Q), len(grid))
    mq = [float(r[16]) for r in rows if r[16] != '']
    assert np.allclose(mq, grid)
    assert float(rows[0][17]) == pytest.approx(
        np.log10(live['channels']['++']['R'][0]))
    # half-polarized summed from the spin channels: R+ = R++ + R+-
    head, rows = read(tmp_path / 'run_half_polarized_R.csv')
    assert head[1] == 'R+ data' and float(rows[0][1]) == pytest.approx(0.51)
    head, rows = read(tmp_path / 'run_SA_NSF.csv')
    assert float(rows[0][1]) == pytest.approx(0.2 / 0.8)
    assert head[4:] == ['SA NSF model Q (Å⁻¹)', 'SA NSF model']
    # without the model: the data alone
    plans, _ = export_datasets(exp, str(tmp_path), [0], lambda s: 'run',
                               ['SA NSF'], ['R'], profile=False,
                               with_model=False)
    assert len(plans[0][1]) == 4
    # an X-ray-like set (R+ only): the model's R+ beside it
    plans, _ = export_datasets(exp, str(tmp_path), [1], lambda s: 'x',
                               [POLARIZED()], ['R'], profile=False)
    assert plans[0][1][4:] == ['R+ model Q (Å⁻¹)', 'R+ model']
    # several: a folder per dataset (its name made safe), the dataset's
    # name by default, what a dataset lacks noted
    plans, notes = export_datasets(
        exp, str(tmp_path), [0, 1], lambda s: s['name'].replace('/', '_'),
        ['SA NSF', POLARIZED()], ['rq4'], profile=False)
    where = {os.path.relpath(p, tmp_path) for p, *_ in plans}
    assert where == {os.path.join('A_1', 'A_1_SA_NSF.csv'),
                     os.path.join('A_1', 'A_1_polarized_RQ4.csv'),
                     os.path.join('X', 'X_polarized_RQ4.csv')}
    assert notes == ['X — SA NSF: needs R++, R--']


def POLARIZED():
    from tabs.simulation import POLARIZED as P
    return P


def test_q_range_as_plotted(qapp):
    from tabs.export_data import export_datasets
    sim, exp = tabs()
    sim.switch_model(sim.model_names().index('A/1'))
    sim.qmin.setValue(0.03)
    sim.qmax.setValue(0.07)
    sim._timer.stop()
    sim.recompute()
    keep = (Q >= 0.03) & (Q <= 0.07)
    for in_range, n in ((True, keep.sum()), (False, len(Q))):
        plans, _ = export_datasets(exp, '.', [0], lambda s: 'a',
                                   [POLARIZED()], ['R'], profile=False,
                                   in_range=in_range)
        (_, _, cols), = plans
        assert len(cols[0]) == n


def test_profile_with_roughness_columns(qapp):
    from tabs.export_data import profile_columns
    sim, _ = tabs()
    head, cols = profile_columns(sim.stack)
    assert head == [
        'z (Å)', 'NSLD Re (10⁻⁶ Å⁻²)', 'NSLD Im (10⁻⁶ Å⁻²)',
        'MSLD ρ (10⁻⁶ Å⁻²)', 'MSLD φ in plane (deg)',
        'MSLD θ out of plane (deg)', 'layer', 'interface z (Å)',
        'interface (above / below)', 'roughness σ (Å)',
        'magnetic roughness σ_M (Å)']
    st = sim.stack
    n = len(st.layers) - 1
    assert len(cols[7]) == n and np.allclose(cols[7], st._interfaces())
    sig = [l.roughness_sigma for l in st.layers[1:]]
    assert np.allclose(cols[9], sig)
    mag = list(cols[10])
    assert mag[1] == 7.5                        # layer 2's own
    assert all(m == s for k, (m, s) in enumerate(zip(mag, sig)) if k != 1)
    assert cols[8][0] == '%s / %s' % (st.layers[0].name, st.layers[1].name)
    # the profile in 10⁻⁶ Å⁻²: deep in the backing, its NSLD
    assert cols[1][-1] == pytest.approx(st.layers[-1].NSLD_real / 1e-6,
                                        rel=1e-6)


def test_dialog_builds(qapp):
    from tabs.export_data import ExportDialog
    _, exp = tabs()
    exp.tree.setCurrentItem(exp.tree.topLevelItem(0))
    dlg = ExportDialog(exp)
    assert dlg.chosen_sets() == [0] and dlg.ok.isEnabled()
    assert 'A_1_SLD_profile' in dlg.preview.text()
    dlg.name.setText('mine')
    assert 'mine_SLD_profile' in dlg.preview.text()
    for b in dlg.scales.values():
        b.setChecked(False)
    assert not dlg.ok.isEnabled()                # a reflectivity, no scale
    dlg.close()
