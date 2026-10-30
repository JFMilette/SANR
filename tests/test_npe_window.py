"""The NPE method of the fit box and its tab (fit.npe.npe_window): a run's
results shown, a cluster sent to the model, DREAM started from the seeds.
Fast with a stand-in run; the end-to-end training in the GUI is `slow`.

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
sys.path.insert(0, str(HERE))

QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
# the tabs first, as the application does (fit.dream.dream_window and
# tabs import each other)
from test_fit_panel import QA, app, tab                  # noqa: E402,F401
from fit.npe import available                                  # noqa: E402
from fit.npe.npe_window import NPERun, NPEWindow                # noqa: E402
from fit.npe.seed import Seeds                                  # noqa: E402

STATS = dict(round=1, n_sims=100, n_failed=2, proposal='prior',
             acceptance=1.0, over=0, epochs=3, best_validation_loss=-1.5,
             validation_loss=[0.5, -1.0, -1.5], training_loss=[0.6, -0.9,
                                                               -1.4],
             seconds=12.0)


class Result:
    """The parts of an NPEResult the tab shows."""

    def __init__(self, names):
        self.names = names
        self.meta = dict(rounds=[STATS], coarse=None, versions={}, git='',
                         seed=1)

    @property
    def rounds(self):
        return self.meta['rounds']


def fake_run(problem, n=8):
    fit = problem.fit
    rng = np.random.default_rng(0)
    u = rng.random((200, len(fit.params)))
    labels = (u[:, 0] > 0.5).astype(int)
    X = fit.to_x(u)
    clusters = [dict(label=k, weight=w, n=int((labels == k).sum()),
                     chains=c, best_logp=-10.0 - k,
                     chi2=problem.chi2_per_point(X[labels == k][0]),
                     best_x=X[labels == k][0], mean_x=X[labels == k][0],
                     at_bound=[])
                for k, w, c in ((0, 0.75, 6), (1, 0.25, 2))]
    x0 = np.vstack([X[labels == 0][:6], X[labels == 1][:2]])
    seeds = Seeds(x0, clusters, u, labels, -np.ones(200), np.zeros(200))
    return NPERun(Result(list(problem.names)), problem, seeds, [], None)


def npe_tab(app):
    exp = tab(QA, QA, licorne=False)
    exp.page_tabs = QtWidgets.QTabWidget()
    return exp


def test_npe_page_and_tab(app, monkeypatch):
    exp = npe_tab(app)
    assert 'npe' in exp._method_pages
    problem, *_ = exp._npe_problem()
    exp._make_npe_window(['d'])
    w = exp.npe_window
    assert w.tab_name.startswith('NPE ')
    w.start(problem, 'header', 300, ['d'])
    exp._show_npe_window()
    assert exp.page_tabs.indexOf(w) >= 0
    w.sims(1, 50, 100)
    assert w.bar.value() == 150
    w.round_done(STATS)
    run = fake_run(problem)
    exp._npe_done(run)
    assert w.table.rowCount() == 2 and w.cluster_box.count() == 2
    assert 'Clusters of 200' in w.report.toPlainText()
    assert exp._npe_seeds[0] == problem.names
    assert exp.dream_start.currentData() == 'npe'

    # a cluster to the model
    sent = []
    monkeypatch.setattr(exp, '_send_values',
                        lambda fit, x, part, parent, title: sent.append(x))
    w.cluster_box.setCurrentIndex(1)
    w.btn_send.click()
    assert np.array_equal(sent[0], run.seeds.clusters[1]['best_x'])

    # DREAM from the seeds: one chain per seed
    got = {}
    monkeypatch.setattr(exp, '_launch_dream',
                        lambda p, options, report: got.update(options))
    w.btn_dream.click()
    assert exp.fit_method.currentData() == 'dream'
    assert np.array_equal(got['x0'], run.seeds.x0)
    assert got['n_chains'] == 8

    # closed like the other tabs
    exp.close_page(w)
    assert exp.npe_window is None and exp.page_tabs.indexOf(w) < 0


def test_dream_start_needs_seeds_of_these_parameters(app, monkeypatch):
    exp = npe_tab(app)
    exp.dream_start.setCurrentIndex(exp.dream_start.findData('npe'))
    called = []
    monkeypatch.setattr(exp, '_launch_dream', lambda *a: called.append(a))
    exp.start_dream()
    assert not called and 'Train a network' in exp.dream_report.text()
    exp._npe_seeds = (['other'], np.zeros((8, 1)))
    exp.start_dream()
    assert not called


def test_window_failure_and_stop(app):
    w = NPEWindow()
    exp = npe_tab(app)
    problem, *_ = exp._npe_problem()
    w.start(problem, 'h', 10)
    assert w.running
    w.fail('stopped')
    assert not w.running and w.status.text() == 'Stopped.'


@pytest.mark.slow
@pytest.mark.skipif(not available(), reason='needs torch, sbi, sklearn')
def test_train_in_the_gui(app):
    """A tiny real run: Train, the thread ends, the seeds are there."""
    from PyQt6 import QtCore
    exp = npe_tab(app)
    exp.npe_rounds.setValue(2)
    exp.npe_sims.setValue(600)
    exp.npe_samples.setValue(500)
    exp.npe_seed.setValue(3)
    exp.npe_calibrate.setChecked(False)
    exp.start_npe()
    assert exp._npe is not None
    loop = QtCore.QEventLoop()
    exp._npe.finished.connect(loop.quit)
    QtCore.QTimer.singleShot(600_000, loop.quit)
    loop.exec()
    app.processEvents()
    w = exp.npe_window
    assert w.run is not None, w.status.text()
    assert len(exp._npe_seeds[1]) == 8 and w.table.rowCount() >= 1
    assert len(w.run.result.rounds) == 2


def test_npe_fits_the_noise_ticked_on_the_dream_page(app):
    """The DREAM page's error scale / floor ticks: NPE samples them too, so
    that its seeds start that DREAM."""
    from fit.dream.dream_fit import FLOOR_NAME, SIGMA_NAME
    exp = npe_tab(app)
    problem, *_ = exp._npe_problem()
    assert problem.n_extra == 0
    exp.dream_sigma.setChecked(True)
    exp.dream_floor.setChecked(True)
    problem, *_ = exp._npe_problem()
    assert problem.names[-2:] == [SIGMA_NAME, FLOOR_NAME]
