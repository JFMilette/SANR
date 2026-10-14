"""fit.multistart: starts, local fits and the merging of their minima on a
toy problem with two minima; on the private Licorne sessions (slow) the
landscape of benchmarks/README.md.

Run from the repo root: python -m pytest tests   (-m slow for the sessions)
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fit.multistart import run_multistart, starting_points     # noqa: E402


class TwoMinima:
    """FitProblem-like: cost of u in [0, 1]^2 with a deep minimum at
    (0.8, 0.7) and a shallow one at (0.2, 0.3); x = 10 u."""

    params = [(1, 'a', 1.5, 0.0, 10.0), (1, 'b', 2.5, 0.0, 10.0)]
    lo, hi = np.zeros(2), np.full(2, 10.0)

    def x0(self):
        return np.array([0.15, 0.25])

    def to_x(self, u):
        return self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)

    def to_u(self, x):
        return np.clip((np.asarray(x) - self.lo) / (self.hi - self.lo), 0, 1)

    def __call__(self, u):
        u = np.asarray(u)
        a = np.sum((u - [0.8, 0.7]) ** 2) / 0.02
        b = np.sum((u - [0.2, 0.3]) ** 2) / 0.02
        return float(1.0 - np.exp(-a) - 0.5 * np.exp(-b) + 0.01 * u[0])


def test_starts():
    p = TwoMinima()
    U = starting_points(p, 12, seed=0)
    assert len(U) == 13 and np.array_equal(U[0], p.x0())
    assert len(np.unique(np.round(U, 9), axis=0)) == 13


def test_two_minima():
    p = TwoMinima()
    m = run_multistart(p, n_starts=16, seed=1)
    assert sum(r['hits'] for r in m) == 17
    assert [r['chi2'] for r in m] == sorted(r['chi2'] for r in m)
    best, other = m[0], m[1]
    assert np.allclose(best['u'], [0.8, 0.7], atol=0.01)
    assert np.allclose(best['x'], [8.0, 7.0], atol=0.1)
    # the current values sit in the shallow basin
    assert np.allclose(other['u'], [0.2, 0.3], atol=0.01)
    assert other['from_current'] and not best['from_current']
    assert not best['at_bound'].any()
    # far from both, the slope 0.01 u0 pulls u0 to 0: on a bound
    assert any(r['at_bound'][0] for r in m[2:])


def test_workers_and_cancel():
    p = TwoMinima()
    one = run_multistart(p, n_starts=8, seed=2)
    two = run_multistart(p, n_starts=8, seed=2, workers=2)
    assert [(r['hits'], round(r['chi2'], 9)) for r in one] == \
        [(r['hits'], round(r['chi2'], 9)) for r in two]
    seen = []
    m = run_multistart(p, n_starts=8, seed=2,
                       callback=lambda k, n: seen.append(k),
                       cancelled=lambda: len(seen) >= 3)
    assert sum(r['hits'] for r in m) == 3


def test_de_method():
    from test_dream_fit import small_problem
    fp = small_problem().fit
    m = run_multistart(fp, n_starts=2, method='de', seed=0,
                       de_options=dict(maxiter=5, popsize=5))
    assert sum(r['hits'] for r in m) == 3
    assert all(np.all((r['u'] >= 0) & (r['u'] <= 1)) for r in m)


@pytest.mark.slow
def test_session_B_landscape():
    """B: almost every start ends in the best minimum (measured 94 %)."""
    from test_licorne_sessions import VARY_B, problem
    p, _ = problem('session_B', VARY_B)
    m = run_multistart(p, n_starts=30, seed=0)
    best = m[0]['chi2']
    near = sum(r['hits'] for r in m if r['chi2'] < best + 1)
    assert best <= 13.33
    assert near >= 0.7 * 31, (near, [(r['chi2'], r['hits']) for r in m])


@pytest.mark.slow
def test_session_A_landscape(tmp_path):
    """A, untied (12 parameters): the best minimum is reached only from the
    session's own values (a benchmark: the table of minima is written for
    reading)."""
    from test_licorne_sessions import problem
    p, _ = problem('session_A', untie=True)
    m = run_multistart(p, n_starts=24, seed=0)
    lines = ['%10.4f %4d %5s %s' % (r['chi2'], r['hits'], r['from_current'],
                                    np.flatnonzero(r['at_bound']).tolist())
             for r in m]
    out = HERE.parent / 'benchmarks' / 'multistart_session_A.txt'
    out.write_text('chi2/N hits from_current at_bound\n' + '\n'.join(lines)
                   + '\n')
    assert m[0]['chi2'] <= 4.93 and m[0]['from_current']


def test_de_restarts():
    from fit.de.de_fit import run_de_restarts
    from fit.multistart import merge
    p = TwoMinima()
    seen = []
    (x, cost, gen, msg), ends = run_de_restarts(
        p, restarts=3, seed=0, maxiter=20, popsize=10, polish=True,
        callback=lambda x, c, g: seen.append(g))
    assert len(ends) == 3 and cost == min(c for _, _, c in ends)
    assert gen == len(seen) and seen == sorted(seen)
    assert msg.startswith('best of 3 runs')
    assert sum(m['hits'] for m in merge(p, ends)) == 3


def test_panel_table_apply_and_dream(monkeypatch):
    """The table's rows go to the Simulation tab and start DREAM."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tabs import ExperimentalTab, SimulationTab
    from fit.problem import FitProblem
    sim = SimulationTab()
    sim.stack.layers[1].fit_entry('thickness')['vary'] = True
    exp = ExperimentalTab(simulation=sim)
    Q = np.linspace(0.01, 0.1, 40)
    one = np.ones_like(Q)
    exp.sets = [dict(name='d', visible=True, channels={
        '+': dict(Q=Q, R=0.1 * one, dR=0.01 * one, dQ=0 * one,
                  theta=np.nan * one, path='x.dat')})]
    exp.refresh_tree()
    exp.fit_set.setCurrentIndex(0)
    s, vec, data, qmin, qmax, grid = exp._fit_inputs(exp.ms_report)
    fp = FitProblem(sim.stack, data, 'chi2', qmin, qmax, grid)
    m = run_multistart(fp, n_starts=4, seed=0)
    exp._ms_note = ''
    exp._ms_show(fp, m, 'multistart')
    assert exp.ms_table.rowCount() == len(m)
    exp.ms_table.selectRow(len(m) - 1)
    exp._ms_apply()
    assert sim.stack.layers[1].thickness == pytest.approx(m[-1]['x'][0])
    seen = {}
    monkeypatch.setattr(exp, '_launch_dream',
                        lambda problem, options, report: seen.update(options))
    # this flat data leaves chi^2/N > 2: the error scale is fitted too,
    # starting at the measured dR (log10 s = 0)
    assert exp.dream_sigma.isChecked()
    exp._ms_dream()
    assert seen['x0'].shape == (8, 2)
    assert np.allclose(seen['x0'][:, 0], m[-1]['x'][0],
                       atol=0.01 * (fp.hi[0] - fp.lo[0]))
    assert np.allclose(seen['x0'][:, 1], 0.0, atol=0.02)


@pytest.mark.slow
def test_session_A_de_restarts():
    """A: three short DE runs, every end listed, the result the best of
    them.  With tol 1e-6 and the polish, all three reach the session's
    4.927 (their populations hold its values); at tol 1e-3 one run
    stopped "successfully" at 10.28."""
    from fit.de.de_fit import run_de_restarts
    from fit.multistart import merge
    from test_licorne_sessions import problem
    p, _ = problem('session_A', untie=True)
    (x, cost, gen, msg), ends = run_de_restarts(
        p, restarts=3, seed=0, maxiter=50, popsize=5, tol=1e-6, init='sobol')
    m = merge(p, ends)
    assert sum(r['hits'] for r in m) == 3
    assert cost == m[0]['chi2'] and np.allclose(p.to_u(x), m[0]['u'])
    assert cost <= 4.93


def test_progress(monkeypatch):
    """Every start reports as it begins, runs and ends, in this process
    and from worker processes."""
    import fit.multistart as ms
    monkeypatch.setattr(ms, 'PROGRESS_EVERY', 0.0)
    p = TwoMinima()
    for workers in (1, 2):
        seen = []
        run_multistart(p, n_starts=4, seed=2, workers=workers,
                       progress=seen.append)
        for k in range(5):
            mine = [i['phase'] for i in seen if i['start'] == k]
            assert 'done' in mine
            if workers == 1:
                assert mine[0] == 'begin' and mine[-1] == 'done'
        # from the workers through the queue (one may come after its
        # start's end, or be left in the queue when the run ends)
        assert any(i['phase'] == 'begin' for i in seen)
        done = [i for i in seen if i['phase'] == 'done']
        assert len(done) == 5 and all(i['best'] is not None for i in done)
        if workers == 1:            # the patch does not reach the workers
            assert {'begin', 'lbfgs', 'done'} <= {i['phase'] for i in seen}
            it = [i for i in seen if i['phase'] == 'lbfgs']
            assert it[0]['step'] == 1 and it[0]['maxiter'] == 3000


def test_de_progress(monkeypatch):
    """A short DE reports its members as they are evaluated, generation by
    generation, then its polish."""
    import math
    import fit.multistart as ms
    from test_dream_fit import small_problem
    monkeypatch.setattr(ms, 'PROGRESS_EVERY', 0.0)
    seen = []
    fp = small_problem().fit
    run_multistart(fp, n_starts=1, method='de', seed=0,
                   de_options=dict(maxiter=5, init='sobol'),
                   progress=seen.append)
    phases = [i['phase'] for i in seen if i['start'] == 0]
    assert phases[0] == 'begin' and phases[-1] == 'done'
    de = [i for i in seen if i['phase'] == 'de' and i['start'] == 0]
    steps = [i['step'] for i in de]
    # the first population, generations 1-5, then 6 as the last ends
    assert steps == sorted(steps) and set(steps) == set(range(7))
    assert de[0]['maxiter'] == 5
    # the members of the first population, one by one
    first = [i['evals'] for i in de if i['step'] == 0]
    assert first == list(range(1, len(first) + 1))
    assert first[-1] == de[0]['size'] == \
        2 ** math.ceil(math.log2(15 * len(fp.params)))
    one = [i['evals'] for i in de if i['step'] == 1]
    assert one[-1] == de[0]['size']
    assert 'polish' in phases
