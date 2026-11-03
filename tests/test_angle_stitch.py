"""model.stack ANGLE (STITCH) FACTORS: per-angle stitch factors of a dataset
measured at several stitched angles, fitted like the scale
(plans/angle_scale_plan_v1.md, tests T1 - T11).

Run from the repo root: python -m pytest tests
"""
import base64
import glob
import json
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import model.stack as S                                 # noqa: E402
from fit.joint import JointProblem                      # noqa: E402
from fit.problem import FitProblem, angle_groups        # noqa: E402
from fit.saved import problem_from_state, problem_state  # noqa: E402
from model.polarisation import vectors_pair             # noqa: E402

PRIVATE = HERE / 'data' / 'private' / 'angle_scale'
PAIRS = {'+': ([1.0, 0, 0], [0, 0, 0]), '-': ([-1.0, 0, 0], [0, 0, 0])}
# three stitched angles, overlapping, sorted by Q as a REF_M export is
BANDS = [(0.008, 0.035, 0.008), (0.028, 0.076, 0.019), (0.068, 0.2, 0.046)]
BG = 2e-6


def film():
    """vacuum / magnetic film 120 A (M along x) / Si, a background."""
    st = S.Stack([S.Layer('vacuum', 0, 0, 0, 0, 0),
                  S.Layer('film', 120, 4e-6, 0, 1e-6, 0, 3.0),
                  S.Layer('Si', 0, 2.07e-6, 0, 0, 0, 3.0)])
    st.background = BG
    st.build_sublayers()
    st.layers[1].fit['thickness'] = {'vary': True, 'min': 100, 'max': 140}
    return st


def three_angles(n=(30, 30, 30), theta=True, channels='+-', jitter=1e-4):
    """R+ and R- of film() at three stitched angles, sorted by Q, each
    channel with its theta column (R-'s off by `jitter`, as one angle's
    spin states can be)."""
    Q = np.concatenate([np.linspace(a, b, k) for (a, b, _), k
                        in zip(BANDS, n)])
    th = np.concatenate([np.full(k, t) for (_, _, t), k in zip(BANDS, n)])
    i = np.argsort(Q, kind='stable')
    Q, th = Q[i], th[i]
    st = film()
    out = []
    for c in channels:
        P0, P = vectors_pair(*PAIRS[c])
        R = st.reflectivities(Q, [(P0, P)], warn=False)[:, 0]
        e = dict(Q=Q, R=R, dR=0.02 * R, P0=P0, P=P, name='R' + c)
        if theta:
            e['theta'] = th * (1 + (jitter if c == '-' else 0.0))
        out.append(e)
    return out


def angle_of_points(data):
    th = data[0]['theta']
    return np.searchsorted(np.unique(np.round(th, 6)), np.round(th, 6))


# -- D2: angle labels -------------------------------------------------------
def test_angles_found_and_spin_states_merged():
    data = three_angles()
    p = FitProblem(film(), data)
    assert p.n_angles == 3
    assert p.angle_theta == pytest.approx([0.008, 0.019, 0.046], rel=1e-3)
    for lab in p.angle_of:
        assert np.array_equal(lab, angle_of_points(data))
    # no theta in one channel, one angle, an angle scan: no angles
    d = three_angles()
    d[1].pop('theta')
    assert FitProblem(film(), d).angle_theta is None
    assert angle_groups([np.full(5, 0.01)]) is None
    assert angle_groups([np.linspace(0.005, 0.05, 40)]) is None


# -- T1: nothing changes without factors ------------------------------------
def test_model_bitwise_without_factors():
    with_th, without = three_angles(), three_angles(theta=False)
    a, b = FitProblem(film(), with_th), FitProblem(film(), without)
    x = [120.0]
    for m, n in zip(a.model(x), b.model(x)):
        assert np.array_equal(m, n)
    assert np.array_equal(a.residuals(x), b.residuals(x))


# -- T2, T3: the factors ----------------------------------------------------
@pytest.mark.parametrize('s2, s3', [(1.03, 1.0), (1.02, 0.97)])
def test_factors_chained_on_the_signal(s2, s3):
    data = three_angles()
    ref = FitProblem(film(), data).model([120.0])
    st = film()
    st.stitch_2, st.stitch_3 = s2, s3
    p = FitProblem(st, data)
    a = p.angle_factors()
    assert a == pytest.approx([1.0, s2, s2 * s3], rel=1e-15)
    if s3 != 1.0:
        assert a[2] == pytest.approx(0.9894)
    lab = angle_of_points(data)
    for m, r in zip(p.model([120.0]), ref):
        assert np.array_equal(m[lab == 0], r[lab == 0])
        for k in (1, 2):
            assert m[lab == k] == pytest.approx(
                BG + a[k] * (r[lab == k] - BG), rel=1e-13)


# -- T4: combined targets ---------------------------------------------------
@pytest.mark.parametrize('target', ['sum', 'asym'])
def test_targets_combine_the_factored_channels(target):
    data = three_angles()
    st = film()
    st.stitch_2, st.stitch_3 = 1.04, 0.95
    ch = FitProblem(st, data)
    p = FitProblem(st, data, target=target)
    ma, mb = ch.model([120.0])
    a, b, ia, ib, v, s = p.combined
    m = (ma[ia] + mb[ib]) / 2 if target == 'sum' else \
        (ma[ia] - mb[ib]) / (ma[ia] + mb[ib])
    assert p.residuals([120.0]) == pytest.approx((m - v) / s, rel=1e-12)


# -- T5: refused --------------------------------------------------------------
def test_varied_beyond_the_angles_refused():
    two = three_angles(n=(30, 30, 0))
    st = film()
    st.fit['stitch_3'] = dict(S.default_fit('stitch_3'), vary=True)
    with pytest.raises(ValueError, match='stitch_3 is fitted but the data '
                                         'have 2 angles'):
        FitProblem(st, two)
    st = film()
    st.fit['stitch_2'] = dict(S.default_fit('stitch_2'), vary=True)
    with pytest.raises(ValueError, match='stitch_2 is fitted but the data '
                                         'have no angles'):
        FitProblem(st, three_angles(theta=False))
    # a joint fit's dataset with fewer angles: fixed at 1, said
    st = film()
    st.stitch_3 = 1.1
    st.fit['stitch_3'] = dict(S.default_fit('stitch_3'), vary=True)
    with pytest.warns(UserWarning, match='stitch_3 is fixed at 1'):
        p = FitProblem(st, two, stitch_extra='fix')
    assert 'stack.stitch_3' not in p.names and p.stack.stitch_3 == 1.0


# -- T6: joint fits -----------------------------------------------------------
def joint_parts(n3=30):
    st = film()
    for a in ('stitch_2', 'stitch_3'):
        st.fit[a] = dict(S.default_fit(a), vary=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return [('A', FitProblem(st, three_angles())),
                ('B', FitProblem(st, three_angles(n=(30, 30, n3)),
                                 stitch_extra='fix'))]


def test_joint_stitch_factors_own_by_default():
    jp = JointProblem(joint_parts())
    for name in ('stack.stitch_2', 'stack.stitch_3'):
        assert '%s [A]' % name in jp.names and '%s [B]' % name in jp.names
        assert name not in jp.names
    assert 'film.thickness' in jp.names            # shared as before
    # 'shared' stays allowed, with a warning
    with pytest.warns(UserWarning, match='stitch_2 is shared'):
        jp = JointProblem(joint_parts(), modes={
            'A': {'stack.stitch_2': 'shared'},
            'B': {'stack.stitch_2': 'shared'}})
    assert 'stack.stitch_2' in jp.names
    # a 2-angle dataset: its stitch_3 fixed at 1
    jp = JointProblem(joint_parts(n3=0))
    assert 'stack.stitch_3 [A]' in jp.names
    assert 'stack.stitch_3 [B]' not in jp.names


def test_more_angles():
    """Five angles: a factor each after the first, chained."""
    q = [np.linspace(0.008 * 2**k, 0.016 * 2**k, 10) for k in range(5)]
    th = np.concatenate([np.full(10, 0.004 * 2**k) for k in range(5)])
    Q = np.concatenate(q)
    st = film()
    P0, P = vectors_pair(*PAIRS['+'])
    R = st.reflectivities(Q, [(P0, P)], warn=False)[:, 0]
    data = [dict(Q=Q, R=R, dR=0.02 * R, P0=P0, P=P, name='R+', theta=th)]
    ref = FitProblem(st, data).model([120.0])[0]
    for k in range(2, 6):
        setattr(st, 'stitch_%d' % k, 1.0 + k / 100)
    p = FitProblem(st, data)
    assert p.n_angles == 5
    a = np.cumprod([1.0, 1.02, 1.03, 1.04, 1.05])
    lab = np.repeat(np.arange(5), 10)
    assert p.model([120.0])[0] == pytest.approx(BG + a[lab] * (ref - BG),
                                                rel=1e-13)


# -- T7: saved state ----------------------------------------------------------
def test_saved_state_keeps_the_angles():
    st = film()
    st.stitch_2 = 1.03
    p = FitProblem(st, three_angles())
    q = problem_from_state(json.loads(json.dumps(problem_state(p))))
    assert q.angle_theta == pytest.approx(p.angle_theta)
    for a, b in zip(p.angle_of, q.angle_of):
        assert np.array_equal(a, b)
    assert q.stack.stitch_2 == 1.03
    assert np.array_equal(q.residuals([120.0]), p.residuals([120.0]))
    # a state written before theta was kept: no angles
    d = problem_state(p)
    for e in d['data']:
        e.pop('theta')
    assert problem_from_state(d).angle_theta is None
    # the stack: written always, absent -> 1
    sd = st.to_dict()
    assert sd['stitch_2'] == 1.03 and sd['stitch_4'] == 1.0
    for a in S.STITCH_PARAMS:
        sd.pop(a)
    assert S.Stack.from_dict(sd).stitch_2 == 1.0


# -- T11: priors ----------------------------------------------------------------
def test_prior_as_for_the_scale():
    from fit.dream.dream_fit import PosteriorProblem
    st = film()
    st.fit['stitch_2'] = dict(S.default_fit('stitch_2'), vary=True,
                              prior={'kind': 'gauss', 'mu': 1.0,
                                     'tau': 0.02})
    st.fit['scale'] = dict(S.default_fit('scale'), vary=True,
                           prior={'kind': 'gauss', 'mu': 1.0, 'tau': 0.02})
    p = FitProblem(st, three_angles())
    i, j = p.names.index('stack.stitch_2'), p.names.index('stack.scale')
    x = [120.0] * len(p.names)
    x[i] = x[j] = 1.0
    base = p.prior_penalty(x)
    x[i] = 1.04
    pen_stitch = p.prior_penalty(x) - base
    x[i], x[j] = 1.0, 1.04
    assert p.prior_penalty(x) - base == pytest.approx(pen_stitch)
    assert pen_stitch == pytest.approx(4.0)
    pp = PosteriorProblem(p)
    x[i], x[j] = 1.04, 1.0
    lp1 = pp.logprior(np.array(x))
    x[i], x[j] = 1.0, 1.04
    assert pp.logprior(np.array(x)) == pytest.approx(lp1)


# -- F1: DREAM summary ---------------------------------------------------------
def test_dream_summary_gives_the_cumulative_factors():
    from fit.dream.dream_fit import PosteriorProblem, stitch_summary
    st = film()
    for a in ('stitch_2', 'stitch_3'):
        st.fit[a] = dict(S.default_fit(a), vary=True)
    pp = PosteriorProblem(FitProblem(st, three_angles()))
    i, j = (pp.fit.names.index('stack.stitch_%d' % k) for k in (2, 3))

    class Result:
        def samples(self, discard):
            s = np.full((1000, len(pp.fit.names)), 120.0)
            s[:, i] = np.linspace(0.98, 1.02, 1000)
            s[:, j] = 1.0
            return s
    lines = stitch_summary(Result(), pp)
    assert lines[0].startswith('Cumulative angle factors')
    assert 'a2 = 1 [0.981' in lines[1] and 'a3 = 1 [0.981' in lines[1]


# -- U1: each dataset's own factors, in the Experimental tab -------------------
@pytest.fixture(scope='module')
def qapp():
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def stitch_tabs(qapp):
    """Simulation and Experimental tabs: datasets A, B (3 angles) and C (2
    angles), A on Model 1, B and C on a copy of it."""
    from fit.panel import JOINT
    from tabs import ExperimentalTab, SimulationTab
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    sim.copy_model()                        # 'Model 1 copy', shown

    def chans(n, path):
        return {e['name'][1:]: dict(e, dQ=0 * e['Q'], path=path)
                for e in three_angles(n=n)}
    exp.sets = [dict(name='A', visible=True, link='Model 1',
                     channels=chans((30, 30, 30), 'x.dat')),
                dict(name='B', visible=True, link='Model 1 copy',
                     channels=chans((30, 30, 30), 'x.dat')),
                dict(name='C', visible=True, link='Model 1 copy',
                     channels=chans((30, 30, 0), 'y.dat'))]
    exp.refresh_tree()
    exp._joint = dict(sets=['A', 'B', 'C'], modes={})
    exp._refresh_fit_sets()
    exp.fit_set.setCurrentIndex(exp.fit_set.findData(JOINT))
    return sim, exp


def test_dataset_stitch_line_and_editor(qapp, monkeypatch):
    from tabs.experimental import STITCH_KEY, StitchDialog, dataset_stitch
    sim, exp = stitch_tabs(qapp)
    # a Stitch line under each dataset with angles, its factors in it
    lines = {}
    for k in range(exp.tree.topLevelItemCount()):
        top = exp.tree.topLevelItem(k)
        for j in range(top.childCount()):
            it = top.child(j)
            if it.data(0, 0x0100)[1] == STITCH_KEY:
                lines[top.text(0)] = it.text(0)
    assert lines == {'A': 'Stitch  1, 1', 'B': 'Stitch  1, 1',
                     'C': 'Stitch  1'}
    assert list(dataset_stitch(exp.sets[2])) == ['stitch_2']
    # the editor: values and bounds, kept in the dataset and the session
    dlg = StitchDialog(exp.sets[0])
    dlg.boxes['stitch_2'][0].setValue(1.04)
    dlg.boxes['stitch_3'][1].setValue(0.9)
    assert dlg.values()['stitch_2']['value'] == pytest.approx(1.04)
    monkeypatch.setattr(StitchDialog, 'exec', lambda self: True)
    monkeypatch.setattr(StitchDialog, 'values',
                        lambda self: dict(stitch_2=dict(value=1.04, min=0.8,
                                                        max=1.25),
                                          stitch_3=dict(value=0.97, min=0.9,
                                                        max=1.1)))
    exp.edit_stitch(0)
    assert exp.sets[0]['stitch']['stitch_3']['min'] == 0.9
    state = exp.session_state(str(HERE))
    assert state['sets'][0]['stitch']['stitch_2']['value'] == 1.04
    assert 'Model 1' in sim.model_names()
    # the Simulation tab has no stitch rows any more
    assert not any(a.startswith('stitch') for a in sim.stack_rows)


def test_fit_takes_and_returns_each_datasets_factors(qapp):
    """Each dataset's stitch factors listed in the Fit tab by its own
    angles, ticked in every linked model; the fit starts from each
    dataset's values and bounds; a result sent goes back to the datasets,
    Revert restores them."""
    from PyQt6 import QtCore, QtWidgets
    sim, exp = stitch_tabs(qapp)
    exp.sets[0]['stitch'] = dict(stitch_2=dict(value=1.04, min=0.9,
                                               max=1.1))
    present = exp._datasets_state()[6]
    stitch = {n: [p for p in present[n] if 'stitch' in p] for n in 'ABC'}
    assert stitch == {'A': ['stack.stitch_2', 'stack.stitch_3'],
                      'B': ['stack.stitch_2', 'stack.stitch_3'],
                      'C': ['stack.stitch_2']}
    panel = exp.datasets_panel
    for p in ('stack.stitch_2', 'stack.stitch_3'):
        panel.table.item(panel.params.index(p), 0).setCheckState(
            QtCore.Qt.CheckState.Checked)
    qapp.processEvents()
    assert sim.stack.fit['stitch_2']['vary']
    assert sim.stored_model('Model 1')['fit']['stitch_3']['vary']
    got = exp._problem(QtWidgets.QLabel(), 'chi2')
    assert got is not None
    problem = got[0]
    names = [n for n in problem.names if 'stitch' in n]
    assert names == ['stack.stitch_2 [A]', 'stack.stitch_2 [B]',
                     'stack.stitch_2 [C]', 'stack.stitch_3 [A]',
                     'stack.stitch_3 [B]']
    j = problem.names.index('stack.stitch_2 [A]')
    assert problem.params[j][2:] == (1.04, 0.9, 1.1)    # A's own
    # a result goes back to each dataset; Revert restores them
    exp._set_fit_start(problem)
    x = [p[2] for p in problem.params]
    x[j] = 0.95
    x[problem.names.index('stack.stitch_3 [B]')] = 1.02
    exp._send_values(problem, x, 0, exp, 'Send')
    assert exp.sets[0]['stitch']['stitch_2']['value'] == 0.95
    assert exp.sets[1]['stitch']['stitch_3']['value'] == 1.02
    assert set(exp.sets[2]['stitch']) == {'stitch_2'}
    exp.revert_fit()
    assert exp.sets[0]['stitch']['stitch_2']['value'] == 1.04
    assert 'stitch' not in exp.sets[1]


# -- P1, T9: the QuickNXS header (private) ---------------------------------------
# the plan's sec. 1 table: angle 1 (plateau), steps 1->2, 2->3 (relative)
HEADER = {'47573': (0.0038, 0.0226, 0.0455), '47577': (0.0040, 0.0206, 0.0528),
          '47580': (0.0039, 0.0208, 0.0486), '47583': (0.0030, 0.0233, 0.0440),
          '47586': (0.0030, 0.0223, 0.0491), '47589': (0.0028, 0.0252, 0.0389)}
THETA = {'47573': (0.00770, 0.01906, 0.04564),
         '47577': (0.00800, 0.01917, 0.04583),
         '47580': (0.00793, 0.01914, 0.04578),
         '47583': (0.00692, 0.01821, 0.04467),
         '47586': (0.00695, 0.01813, 0.04477),
         '47589': (0.00653, 0.01759, 0.04422)}
needs_private = pytest.mark.skipif(
    not (PRIVATE / 'good_starting_point_1.json').exists(),
    reason='private data (tests/data/private/angle_scale)')


def private_files(state):
    return sorted(glob.glob(str(PRIVATE / ('REF_M_*_%s.dat' % state))))


@needs_private
def test_quicknxs_header():
    from tabs.experimental import parse_data, quicknxs_stitch_priors
    files = private_files('Off_Off')
    assert len(files) == 6
    for path in files:
        run = os.path.basename(path)[6:11]
        h = quicknxs_stitch_priors(path)
        plateau, *steps = HEADER[run]
        assert h['plateau'] == pytest.approx(plateau, abs=1e-4)
        assert h['steps'] == pytest.approx(steps, abs=1e-4)
        assert h['options']['total_reflectivity_q_cutoff'] == '0.016'
        assert h['options']['global_stitching'] == 'False'
        lines = open(path).read().splitlines()
        start = next(i for i, l in enumerate(lines)
                     if l and not l.startswith('#'))
        th = parse_data(lines, start)[0]['theta']
        assert angle_groups([th]) == pytest.approx(THETA[run], abs=1e-5)


@needs_private
def test_header_priors_per_dataset():
    from fit.panel import FitPanel
    path = private_files('Off_Off')[1]                 # 47577: S1 4K 0.1T
    s = {'channels': {'+': {'path': path}}}
    st = film()
    st.fit['scale'] = dict(S.default_fit('scale'),
                           prior={'kind': 'gauss', 'mu': 1.0, 'tau': 0.1})
    st.fit['stitch_2'] = dict(S.default_fit('stitch_2'), vary=True)
    with pytest.warns(UserWarning, match='QuickNXS header: stitch_2'):
        got = FitPanel._dataset_stack(s, st)
    assert got.fit['stitch_2']['prior']['tau'] == pytest.approx(0.0206,
                                                                abs=1e-4)
    assert got.fit['stitch_3']['prior']['tau'] == pytest.approx(0.0528,
                                                                abs=1e-4)
    assert got.fit['scale']['prior']['tau'] == 0.1     # its own kept
    assert 'prior' not in st.fit['stitch_2']           # a copy
    assert FitPanel._dataset_stack({'channels': {}}, st) is st


# -- T8: regression on the stored DREAM best state (private) --------------------
T8 = {'S1 4K 0.1T': (0.95, 1.01, 812.997, 639.448),
      'S1 4K 2T': (1.00, 1.01, 523.081, 522.235),
      'S1 4K 4.8T': (0.99, 0.99, 585.036, 572.005),
      'S1 10K 4.8T': (1.01, 1.01, 722.558, 703.638),
      'S1 20K 4.8T': (1.02, 0.94, 593.245, 555.353),
      'S1 250K 4.8T': (1.02, 1.05, 920.215, 835.767)}


@needs_private
def test_regression_good_starting_point():
    from fit.dream.dream_fit import load_run
    from fit.joint import part_x
    from tabs.experimental import parse_data
    sess = json.loads((PRIVATE / 'good_starting_point_1.json').read_text())
    tab = next(t for t in sess['fit_tabs'] if t['kind'] == 'dream')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res, prob, _ = load_run(base64.b64decode(tab['npz']))
    x = res.best()[0]
    state = problem_state(prob.fit)
    sets = {s['name']: s for s in sess['experimental']['sets']}
    for name, ps in state['parts']:
        for e in ps['data']:
            c = sets[name]['channels'][e['name'][1:]]
            path = PRIVATE / os.path.basename(c['file']['path'])
            d, _ = parse_data(path.read_text().splitlines(), c['start'])
            assert np.allclose(d['Q'], e['Q'])
            e['theta'] = d['theta'].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = problem_from_state(state)
    total = [0.0, 0.0]
    for k, (name, p) in enumerate(zip(fit.part_names, fit.parts)):
        s2, s3, at_ones, with_factors = T8[name]
        xs = part_x(fit, x, k)

        def chi2():
            return sum(np.sum(((m - R) / s)**2) for m, (_, _, R, s)
                       in zip(p.model(xs), p.channels))
        assert p.n_angles == 3
        c0 = chi2()
        p.stack.stitch_2, p.stack.stitch_3 = s2, s3
        c1 = chi2()
        assert c0 == pytest.approx(at_ones, abs=0.01)
        assert c1 == pytest.approx(with_factors, abs=0.01)
        total[0] += c0
        total[1] += c1
    assert total == pytest.approx([4157.132, 3828.445], abs=0.01)


# -- U3: data corrected by the factors ----------------------------------------
def test_stitch_corrected_data(qapp):
    """Each angle's signal divided by its factor, so that the data of a
    model with factors match the model without them; a dataset without
    angles, or factors at 1, is drawn as it is."""
    from tabs import ExperimentalTab, SimulationTab
    from tabs.experimental import stitch_corrected
    st = film()
    st.stitch_2, st.stitch_3 = 1.05, 0.9
    data = three_angles()
    p = FitProblem(st, data)
    measured = {e['name'][1:]: dict(e, R=m, dR=0.02 * m)
                for e, m in zip(data, p.model([120.0]))}
    ref = FitProblem(film(), data).model([120.0])
    values = dict(stitch_2=1.05, stitch_3=0.9)
    got = stitch_corrected(measured, values, BG)
    f = np.array([1.0, 1.05, 1.05 * 0.9])[angle_of_points(data)]
    for (ch, d), r in zip(got.items(), ref):
        assert d['R'] == pytest.approx(r, rel=1e-12)
        assert d['dR'] == pytest.approx(measured[ch]['dR'] / f, rel=1e-12)
    assert stitch_corrected(measured, {}, BG) is measured
    plain = {e['name'][1:]: e for e in three_angles(theta=False)}
    assert stitch_corrected(plain, values, BG) is plain
    # the tab: drawn corrected with the dataset's own factors, kept in
    # the session
    sim = SimulationTab()
    exp = ExperimentalTab(simulation=sim)
    chans = {ch: dict(d, dQ=0 * d['Q'], path='x.dat')
             for ch, d in measured.items()}
    exp.sets = [dict(name='meas', visible=True, channels=chans,
                     stitch=dict(stitch_2=dict(value=1.05, min=0.8,
                                               max=1.25)))]
    exp.refresh_tree()
    exp.show_corrected.setChecked(True)
    y = [item.yData for item, _ in exp._items]       # not log10
    Q = chans['+']['Q']
    keep = (Q >= sim.qmin.value()) & (Q <= sim.qmax.value())
    lab = angle_of_points(data)[keep]
    raw = chans['+']['R'][keep]
    bg = sim.stack.background                      # the linked model's
    assert any(np.allclose(v[lab == 1], bg + (raw[lab == 1] - bg) / 1.05,
                           rtol=1e-12) for v in y if len(v) == len(raw))
    assert exp.session_state(str(HERE))['display']['corrected'] is True

