"""Resolution dialog: the spreads fitted to a dQ column, and Apply.

Run from the repo root: python -m pytest tests
"""
import copy
import os
import pathlib
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
from model.stack import DEFAULT_RESOLUTION, RESOLUTION_MODES    # noqa: E402
from tabs import ExperimentalTab, SimulationTab                 # noqa: E402
from tabs.resolution import fit_resolution                     # noqa: E402

Q = np.linspace(0.008, 0.2, 120)


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def res(mode, **kw):
    r = copy.deepcopy(DEFAULT_RESOLUTION)
    r.update(mode=mode, **kw)
    return r


def test_tof_spreads_recovered():
    true = res('tof', tof_dlambda=0.02)
    true['tof_angles'][1]['dtheta'] = 7e-4
    dQ = RESOLUTION_MODES['tof'](Q, true)
    start = res('tof', tof_dlambda=0.1)
    fit, misfit = fit_resolution(Q, dQ, start)
    assert fit['tof_dlambda'] == pytest.approx(0.02, rel=1e-6)
    assert [a['dtheta'] for a in fit['tof_angles']] == pytest.approx(
        [a['dtheta'] for a in true['tof_angles']], rel=1e-6)
    assert misfit < 1e-6
    # an angle with no data in its band keeps its dtheta
    low = Q[Q < 0.1]
    fit, _ = fit_resolution(low, RESOLUTION_MODES['tof'](low, true), start)
    assert fit['tof_angles'][2]['dtheta'] == start['tof_angles'][2]['dtheta']


def test_too_few_points():
    with pytest.raises(ValueError):
        fit_resolution(Q, np.zeros_like(Q), res('tof'))


def test_dialog_fits_and_applies(app):
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    true = res('tof', tof_dlambda=0.02, tof_angles=sim._tof_angles())
    true['tof_angles'][0]['dtheta'] = 4e-4
    one = np.ones_like(Q)
    exp.sets = [dict(name='meas', visible=True, channels={
        '+': dict(Q=Q, R=0.1 * one, dR=0.01 * one,
                  dQ=RESOLUTION_MODES['tof'](Q, true),
                  theta=np.nan * one, path='x.dat')})]
    exp.refresh_tree()
    sim._show_resolution()
    dlg = sim._res_dialog
    assert dlg.dataset.currentText() == 'meas'
    dlg.fit()
    assert dlg.btn_apply.isEnabled()
    dlg.apply()
    assert sim.res_on.isChecked()
    assert sim.res_tof_dl.value() == pytest.approx(0.02, abs=1e-4)  # A
    sim.recompute()
    got = sim.stack.resolution['tof_angles'][0]['dtheta']
    assert got == pytest.approx(4e-4, abs=1e-6)
    dlg.close()


# -- the 'data' mode: each point's own dQ ------------------------------------
def stitched():
    """One channel from 3 stitched angles: overlapping Q bands whose dQ do
    not join up (each angle its own dtheta)."""
    from model.stack import Stack
    from tabs.simulation import make_stack
    st = make_stack()
    st.build_sublayers()
    bands = [(0.008, 0.05, 0.006, 2e-4), (0.04, 0.12, 0.010, 6e-4),
             (0.10, 0.2, 0.017, 1.2e-3)]
    Q, dQ = [], []
    for lo, hi, th, dth in bands:
        q = np.linspace(lo, hi, 50)
        Q.append(q)
        one = [{'theta': th, 'dtheta': dth, 'qmax': None}]
        dQ.append(RESOLUTION_MODES['tof'](q, res('tof', tof_dlambda=0.01,
                                                 tof_angles=one)))
    Q, dQ = np.concatenate(Q), np.concatenate(dQ)
    P0 = np.array([0.0, 1.0, 0.0])
    R = st.reflectivities(Q, [(P0, None)], sigma=dQ)[:, 0]
    return st, dict(Q=Q, R=R, dR=0.02 * R, dQ=dQ, P0=P0, P=None, name='R+')


def test_explicit_sigma_matches_the_mode():
    from tabs.simulation import make_stack
    st = make_stack()
    st.build_sublayers()
    st.resolution.update(enabled=True, mode='tof')
    P = [(np.array([0.0, 1.0, 0.0]), None)]
    want = st.reflectivities(Q, P)
    got = st.reflectivities(Q, P, sigma=st.resolution_sigma(Q))
    assert np.allclose(got, want, rtol=1e-12)
    # the 'data' mode interpolates its table
    st.resolution.update(mode='data', data_q=[0.0, 0.2],
                         data_dq=[0.0, 2e-3])
    assert st.resolution_sigma([0.1]) == pytest.approx(1e-3)
    # a point's own dQ, the table where it has none
    assert st.data_sigma([0.1, 0.1], [5e-4, 0.0]) == \
        pytest.approx([5e-4, 1e-3])


def test_fit_uses_each_points_dq():
    from fit.problem import FitProblem
    st, d = stitched()
    st.layers[1].fit_entry('thickness')['vary'] = True
    st.resolution.update(enabled=True, mode='data', data_q=list(d['Q'][:50]),
                         data_dq=list(d['dQ'][:50]), data_set='x')
    fp = FitProblem(st, [d])
    assert fp.npoints == d['Q'].size                # the overlaps are kept
    assert fp(fp.x0()) == pytest.approx(0.0, abs=1e-20)
    # one smooth sigma(Q) for all three angles misses
    st.resolution['mode'] = 'tof'
    assert FitProblem(st, [d])(fp.x0()) > 1e-3


def test_dialog_use_data(app):
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    _, d = stitched()
    one = np.ones_like(d['Q'])
    exp.sets = [dict(name='meas', visible=True, channels={
        '+': dict(Q=d['Q'], R=d['R'], dR=d['dR'], dQ=d['dQ'],
                  theta=np.nan * one, path='x.dat')})]
    exp.refresh_tree()
    sim._show_resolution()
    dlg = sim._res_dialog
    dlg.use_data()
    sim.recompute()                       # the widgets reach the stack
    r = sim.stack.resolution
    assert sim.res_on.isChecked() and r['mode'] == 'data'
    assert r['data_set'] == 'meas' and 'data_fwhm' not in r
    assert len(r['data_q']) == len(set(d['Q']))     # a repeated Q averaged
    assert np.all(np.isfinite(sim.last_R['+']))
    dlg.fit()                                       # nothing to fit in 'data'
    assert dlg.fitted is None and 'fixed' in dlg.note.text()
    # the fit gets each point's dQ
    s, vec, data, *_ = exp._fit_inputs(exp.fit_report)
    assert np.array_equal(data[0]['dQ'], d['dQ'])
    # saved and restored with the stack
    from model.stack import Stack
    back = Stack.from_dict(sim.stack.to_dict())
    assert back.resolution['data_q'] == r['data_q']
    dlg.close()


def test_dq_table_averages_overlapping_angles():
    from tabs.resolution import dq_table
    # two angles in file order, overlapping on [0.04, 0.05]
    q1, q2 = np.linspace(0.01, 0.05, 5), np.linspace(0.04, 0.08, 5)
    Q = np.concatenate([q1, q2])
    dQ = np.concatenate([np.full(5, 1e-3), np.full(5, 3e-3)])
    q, s = dq_table({'+': (Q, dQ)})
    assert np.all(np.diff(q) > 0)
    assert s[q < 0.04] == pytest.approx(1e-3)
    assert s[(q >= 0.04) & (q <= 0.05)] == pytest.approx(2e-3)   # no zigzag
    assert s[q > 0.05] == pytest.approx(3e-3)


def test_dq_table_overlap_rules():
    from tabs.resolution import dq_table
    q1, q2 = np.linspace(0.01, 0.05, 5), np.linspace(0.04, 0.08, 5)
    Q = np.concatenate([q1, q2])
    dQ = np.concatenate([np.full(5, 1e-3), np.full(5, 3e-3)])
    both = {'+': (Q, dQ), '-': (Q, dQ)}          # R+ and R- of one scan
    over = lambda q: (q >= 0.04) & (q <= 0.05)    # noqa: E731
    q, s = dq_table(both, overlap='first')
    assert s[over(q)] == pytest.approx(1e-3)
    assert s[q > 0.05] == pytest.approx(3e-3)
    q, s = dq_table(both, overlap='last')
    assert s[over(q)] == pytest.approx(3e-3)
    assert s[q < 0.04] == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        dq_table(both, overlap='middle')


def test_dialog_overlap_redoes_the_table(app):
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    _, d = stitched()
    one = np.ones_like(d['Q'])
    exp.sets = [dict(name='meas', visible=True, channels={
        '+': dict(Q=d['Q'], R=d['R'], dR=d['dR'], dQ=d['dQ'],
                  theta=np.nan * one, path='x.dat')})]
    exp.refresh_tree()
    sim._show_resolution()
    dlg = sim._res_dialog
    dlg.use_data()
    mean = list(sim.stack.resolution['data_dq'])
    dlg.overlap.setCurrentIndex(2)                 # later angle: redone
    r = sim.stack.resolution
    assert r['data_overlap'] == 'last' and r['data_dq'] != mean
    q = np.asarray(r['data_q'])
    i = np.argmin(np.abs(q - 0.045))               # angles 1 and 2 overlap
    later = d['dQ'][50:100][np.argmin(np.abs(d['Q'][50:100] - q[i]))]
    assert r['data_dq'][i] == pytest.approx(later, rel=0.05)
    dlg.close()


def test_angles_told_apart_by_theta():
    """A REF_M file: sorted by Q, the angles' points interleaved where they
    overlap, one theta per angle."""
    from tabs.resolution import dq_table, split_angles
    q1, q2 = np.linspace(0.01, 0.05, 9), np.linspace(0.042, 0.08, 9)
    Q = np.concatenate([q1, q2])
    dQ = np.concatenate([0.02 * q1, 0.04 * q2])
    th = np.concatenate([np.full(9, 0.008), np.full(9, 0.019)])
    i = np.argsort(Q)
    Q, dQ, th = Q[i], dQ[i], th[i]
    assert len(split_angles(Q, dQ)) == 1                 # no Q falls back
    assert len(split_angles(Q, dQ, th)) == 2
    for rule, rel in (('first', 0.02), ('last', 0.04)):
        q, s = dq_table({'+': (Q, dQ, th)}, overlap=rule)
        over = (q >= 0.042) & (q <= 0.05)
        assert (s / q)[over] == pytest.approx(rel)        # no zigzag
    # an angle scan (theta different at every point) is not split by it
    assert len(split_angles(Q, dQ, np.linspace(0.005, 0.03, Q.size))) == 1


def test_old_mono_session_loads_as_tof_off():
    """The fixed-wavelength mode and the FWHM flag were removed: a session
    saved with them loads in the 'tof' mode, off, without their keys."""
    from model.stack import Stack
    from tabs.simulation import make_stack
    d = make_stack().to_dict()
    d['resolution'].update(enabled=True, mode='mono', wavelength=5.0,
                           dlambda_rel=0.002, dtheta=7e-4, data_fwhm=True)
    r = Stack.from_dict(d).resolution
    assert r['mode'] == 'tof' and not r['enabled']
    assert not {'wavelength', 'dlambda_rel', 'dtheta', 'data_fwhm'} & set(r)
