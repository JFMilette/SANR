"""
NPE window: a neural posterior estimation run (fit.npe) in a tab of the
main window, as the DREAM window is for DREAM (fit.dream.dream_window).

A run (NPEThread) probes the parameters' symmetries (fit.npe.symmetry),
chooses the training simulator (fit.npe.simulator choose_coarse, unless
coarse training is off), trains the network round after round
(fit.npe.train), checks the calibration of its amortised round 1 (when
asked), and seeds the DREAM chains with the exact posterior
(fit.npe.seed).  Re-seeding a network (the tab's "Re-seed", or a network
file saved without its seeds) only does the seeding.  Stop ends it at the
next simulation chunk or training batch.  The seeds, clusters and probe are
kept with the network (fit.npe.train NPEResult), so that "Load network…"
shows them as they were, at once.

  Report     what was done: the training simulator and its fidelity, each
             round (simulations, failures, proposal and its acceptance,
             epochs, best validation loss, time), the calibration, the
             symmetry probe, and the clusters as text
  Training   the loss -log q(theta | x) of every epoch (lower is better),
             one colour per round: on the 10 % of the simulations held
             out of the training (validation, solid line and filled
             points; it ends a round and picks its network, the best
             epoch's, starred) and on the 90 % trained on (training,
             dashed, hollow points); a dashed vertical line where each
             round starts; the round being trained drawn as it goes; the status line says its epoch, its loss and
             how many epochs have passed without a better one (the round
             ends after 20)
  Clusters   the candidate solutions (fit.npe.seed clusters): weight,
             chains, best chi^2/N, parameters at a bound, best values
  Marginals  one parameter at a time: the network's samples (a proposal,
             not the posterior) and each cluster's best value
  Constraints  per parameter, its 90 % interval over the bounds' (fit.npe.
             seed widths): the network's samples (bars) and, when the
             effective sample size is at least RELIABLE_ESS, the exact
             posterior's (the samples weighted exact / network, points)
  Network check  each sample's exact log posterior against the network's
             log density, by cluster, with the line of slope 1 a perfect
             network gives, and the effective sample size (fit.npe.seed
             importance)

"Load into model" sends the best state of the cluster chosen beside it to
the Simulation tab (in a joint fit, as the dataset chosen sees it); "DREAM
from seeds" samples the posterior with DREAM from the seeds, with the
DREAM page's settings (the chains: one per seed).  "Save network…" writes
the network, its seeds and clusters (fit.npe.train NPEResult.save, .npe)
for "Load network…" in the fit box; "Re-seed" draws new seeds from the
network with the samples beside it (this run's at first) and the NPE
page's Seeds and Seed (a new tab).  The network's samples are never a result: only DREAM's are.
Values are shown in the Simulation tab's units.  An NPE tab is not kept in
sessions.
"""

import os
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from fit.dream.dream_window import (CHAIN_COLOURS, SESSIONS_DIR, axis_label,
                                    display_info, duration, placeholder,
                                    plot_item, vbars)

DIALOG_OPTIONS = QtWidgets.QFileDialog.Option.DontUseNativeDialog
OK, BAD = '#4ade80', '#f87171'
LOSS_KEY = '#9aa0a6'              # the Training plot's legend key
RELIABLE_ESS = 100                # weighted widths shown from this ESS on
SPREAD = 3                        # Network check: log p / log q spreads


class NPERun:
    """What a run produced: the NPEResult, the PosteriorProblem, the seeds
    (fit.npe.seed Seeds), the symmetry probe and the calibration."""

    def __init__(self, result, problem, seeds, probe=(), calibration=None):
        self.result, self.problem, self.seeds = result, problem, seeds
        self.probe = list(probe)
        self.calibration = calibration


class NPEThread(QtCore.QThread):
    """Runs an NPE training and the seeding off the GUI thread."""

    stage = QtCore.pyqtSignal(str)                    # what it does now
    sims = QtCore.pyqtSignal(int, int, int)           # round, done, total
    epoch = QtCore.pyqtSignal(object)                 # a training's epoch
    roundDone = QtCore.pyqtSignal(object)             # a round's statistics
    coarse = QtCore.pyqtSignal(object)                # choose_coarse report
    done = QtCore.pyqtSignal(object)                  # NPERun
    failed = QtCore.pyqtSignal(str)

    def __init__(self, problem, *, result=None, coarse=True, threshold=0.3,
                 seed=None, n_workers=1, rounds=5, sims_per_round=20000,
                 n_samples=10000, n_chains=8, calibrate=True):
        super().__init__()
        self.problem, self.result = problem, result
        self.options = dict(coarse=coarse, threshold=threshold, seed=seed,
                            n_workers=n_workers, rounds=rounds,
                            sims_per_round=sims_per_round)
        self.n_samples, self.n_chains = n_samples, n_chains
        self.calibrate = calibrate
        self._stop = False

    def stop(self):
        self._stop = True

    def _callback(self, event, info):
        if event == 'fidelity':
            self.stage.emit(info)
        elif event == 'coarse':
            self.coarse.emit(info)
        elif event == 'simulate':
            self.sims.emit(*info)
        elif event == 'train':
            self.stage.emit('Round %d: training the network…' % (info + 1))
        elif event == 'epoch':
            self.epoch.emit(info)
        elif event == 'round':
            self.roundDone.emit(info)

    def run(self):
        try:
            from fit.npe.seed import seed_chains
            from fit.npe.symmetry import probe
            from fit.npe import train as npe_train
            o = self.options
            fit = self.problem.fit
            self.stage.emit('Symmetry probe…')
            rows = probe(fit, n_draws=10, rng=o['seed'])
            res, cal = self.result, None
            if res is None:
                res = npe_train.run_npe(
                    self.problem, coarse=o['coarse'], threshold=o['threshold'],
                    seed=o['seed'], n_workers=o['n_workers'],
                    rounds=o['rounds'], sims_per_round=o['sims_per_round'],
                    callback=self._callback, cancelled=lambda: self._stop)
                if self.calibrate and not self._stop:
                    self.stage.emit('Calibration of round 1…')
                    cal = npe_train.calibrate(res, res.simulator,
                                              seed=o['seed'],
                                              n_workers=o['n_workers'])
            else:
                cal = res.meta.get('calibration')
            self.stage.emit('Seeding: the exact posterior of the network\'s '
                            'samples…')
            seeds = seed_chains(
                res, self.problem, self.n_chains, rng=o['seed'],
                n_samples=self.n_samples, n_workers=o['n_workers'],
                progress=lambda d, n: self.stage.emit(
                    'Seeding: the exact posterior of %d / %d samples…'
                    % (d, n)),
                cancelled=lambda: self._stop)
            # kept with the network (Save network…, Load network…)
            res.seeds, res.meta['probe'] = seeds, rows
            res.seeding = dict(n_samples=self.n_samples,
                               n_chains=self.n_chains, seed=o['seed'])
        except Exception as exc:              # report, don't kill the app
            msg = 'stopped' if self._stop else \
                '%s: %s' % (type(exc).__name__, exc)
            self.failed.emit(msg)
            return
        self.done.emit(NPERun(res, self.problem, seeds, rows, cal))


def report_text(run):
    """The Report tab of a finished run."""
    from fit.npe.seed import format_clusters
    from fit.npe.simulator import describe_coarse
    from fit.npe.symmetry import format_probe
    meta = run.result.meta
    out = []
    if meta.get('coarse'):
        out.append('Training simulator: ' + describe_coarse(meta['coarse']))
    elif meta.get('rounds'):
        out.append('Training simulator: full fidelity (coarse training off)')
    if meta.get('embedding') and meta.get('flow'):
        (hidden, n_out), flow = meta['embedding'], meta['flow']
        out.append('Network: embedding %s → %d features, %s flow of %d '
                   'transforms × %d hidden units'
                   % (' × '.join(str(h) for h in hidden), n_out,
                      flow.get('model'), flow.get('num_transforms'),
                      flow.get('hidden_features')))
    if meta.get('rounds'):
        out.append('')
        out.append('%5s %7s %6s %9s %10s %7s %12s %9s'
                   % ('round', 'sims', 'failed', 'proposal', 'accepted',
                      'epochs', 'best loss', 'time'))
        for r in meta['rounds']:
            out.append('%5d %7d %6d %9s %10.3g %7d %12.5g %9s'
                       % (r['round'], r['n_sims'], r['n_failed'],
                          r['proposal'], r['acceptance'], r['epochs'],
                          r['best_validation_loss'],
                          duration(r['seconds'])))
            if r.get('over'):
                out.append('      the envelope\'s pool stayed small: its '
                           'draws are a rough truncated prior')
    cal = run.calibration
    out.append('')
    if cal is None:
        out.append('Calibration of round 1: not checked.')
    else:
        out.append('Calibration of round 1 (SBC, %d draws): %s'
                   % (cal['n'], 'passed' if cal['passed'] else
                      'FAILED — do not use it for scenario studies'))
        out.append('  KS p-values: ' + ', '.join(
            '%s %.3g' % (n, p) for n, p in zip(run.result.names,
                                               cal['ks_pvals'])))
    out.append('')
    out.append('Symmetry probe:')
    out.append(format_probe(run.probe))
    out.append('')
    sd = getattr(run.result, 'seeding', None) or {}
    out.append('Clusters of %d network samples (the chains start at the '
               'best ones; %d seeds, seed %s):'
               % (len(run.seeds.u), len(run.seeds.x0), sd.get('seed')))
    out.append(format_clusters(run.seeds))
    out.append('')
    out.append('The network only proposes: sample the posterior with '
               '"DREAM from seeds".')
    v = meta.get('versions', {})
    out.append('sbi %s, torch %s, git %s, seed %s'
               % (v.get('sbi'), v.get('torch'), (meta.get('git') or '')[:10],
                  meta.get('seed')))
    return '\n'.join(out)


class NPEWindow(QtWidgets.QWidget):
    """One NPE run: progress while it trains, its seeds after."""

    stopRequested = QtCore.pyqtSignal()
    # state x, dataset whose values are sent, what it is ('cluster 2')
    sendRequested = QtCore.pyqtSignal(object, int, str)
    dreamRequested = QtCore.pyqtSignal()
    reseedRequested = QtCore.pyqtSignal(int)          # samples

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle('Neural posterior estimation (NPE)')
        self.resize(1100, 820)
        self.problem = self.run = None
        self.infos = []
        self._rounds = []
        self._t0 = time.monotonic()
        self.total = 1

        self.header = QtWidgets.QLabel(wordWrap=True)
        self.bar = QtWidgets.QProgressBar()
        self.status = QtWidgets.QLabel(wordWrap=True)
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_stop.setToolTip('Stop at the next simulation chunk or '
                                 'training batch')
        self.btn_stop.clicked.connect(self.stopRequested)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.bar, 1)
        top.addWidget(self.btn_stop)

        self.tabs = QtWidgets.QTabWidget()
        self.report = QtWidgets.QPlainTextEdit(readOnly=True)
        self.report.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont))
        self.report.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.loss = pg.GraphicsLayoutWidget()
        self.loss_plot = plot_item('epoch (all rounds)',
                                   'loss  −log q(θ | x)   (lower is better)')
        # bottom left: the loss falls from round to round, leaving it free
        self.loss_plot.addLegend(offset=(10, -10))
        self.loss.addItem(self.loss_plot)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._table_picked)
        self.marg_combo = QtWidgets.QComboBox()
        self.marg_combo.currentIndexChanged.connect(self._draw_marginal)
        self.marg = pg.GraphicsLayoutWidget()
        self.marg_plot = plot_item('', 'samples')
        self.marg.addItem(self.marg_plot)
        marg = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(marg)
        v.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Parameter'))
        row.addWidget(self.marg_combo)
        row.addWidget(QtWidgets.QLabel(
            '<i>the network\'s samples: a proposal, not the posterior; '
            'lines: each cluster\'s best state</i>'))
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(self.marg, 1)
        # how much the data narrow each parameter (one bar each, scrolled)
        self.width_plot = plot_item('90 % interval / the bounds\' 90 %   '
                                    '(1: the data do not constrain it)')
        self.width_plot.invertY(True)
        self.width_plot.setMouseEnabled(x=False, y=False)
        self.width_view = pg.GraphicsLayoutWidget()
        self.width_view.addItem(self.width_plot)
        self.width_note = QtWidgets.QLabel(wordWrap=True)
        scroll = QtWidgets.QScrollArea(widgetResizable=True)
        scroll.setWidget(self.width_view)
        widths_page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(widths_page)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.width_note)
        v.addWidget(scroll, 1)
        # the network against the exact posterior, sample by sample
        self.check_plot = plot_item('log q   (the network)',
                                    'log p   (exact posterior, + a constant)')
        self.check_plot.addLegend(offset=(10, 10))
        self.check_view = pg.GraphicsLayoutWidget()
        self.check_view.addItem(self.check_plot)
        self.check_note = QtWidgets.QLabel(wordWrap=True)
        check_page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(check_page)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.check_note)
        v.addWidget(self.check_view, 1)
        self.pages = {}
        for name, w in (('Clusters', self.table), ('Marginals', marg),
                        ('Constraints', widths_page),
                        ('Network check', check_page)):
            page = QtWidgets.QStackedWidget()
            page.addWidget(placeholder('Shown when the seeds are ready.'))
            page.addWidget(w)
            self.pages[name] = page
        self.tabs.addTab(self.report, 'Report')
        self.tabs.addTab(self.loss, 'Training')
        self.tabs.addTab(self.pages['Clusters'], 'Clusters')
        self.tabs.addTab(self.pages['Marginals'], 'Marginals')
        self.tabs.addTab(self.pages['Constraints'], 'Constraints')
        self.tabs.addTab(self.pages['Network check'], 'Network check')

        self.btn_save = QtWidgets.QPushButton('Save network…')
        self.btn_save.clicked.connect(self._save)
        self.cluster_box = QtWidgets.QComboBox()
        self.cluster_box.setToolTip('The cluster whose best state "Load into '
                                    'model" sends')
        self.btn_send = QtWidgets.QPushButton('Load into model')
        self.btn_send.setToolTip('Send the best state of the cluster chosen '
                                 'beside it to the Simulation tab')
        self.btn_send.clicked.connect(self._send)
        self.btn_dream = QtWidgets.QPushButton('DREAM from seeds')
        self.btn_dream.setToolTip(
            'Sample the posterior with DREAM (the DREAM page\'s settings), '
            'one chain per seed:\nthe network proposes, DREAM decides')
        self.btn_dream.clicked.connect(self.dreamRequested)
        self.btn_reseed = QtWidgets.QPushButton('Re-seed')
        self.btn_reseed.setToolTip(
            'Draw new seeds from this network with the samples beside it and '
            'the NPE page\'s Seeds and Seed (the exact posterior is computed '
            'again), in a new tab')
        self.btn_reseed.clicked.connect(
            lambda: self.reseedRequested.emit(self.reseed_samples.value()))
        self.reseed_samples = QtWidgets.QSpinBox()
        self.reseed_samples.setRange(500, 1_000_000)
        self.reseed_samples.setSingleStep(1000)
        self.reseed_samples.setValue(10000)
        self.reseed_samples.setToolTip(
            'Network samples of a re-seed: their exact posterior is computed '
            '(one model evaluation each); the effective sample size grows '
            'about in proportion')
        self.part_box = QtWidgets.QComboBox()
        self.part_box.setToolTip('Joint fit: the dataset whose values "Load '
                                 'into model" writes')
        self.part_label = QtWidgets.QLabel('Dataset')
        self.note = QtWidgets.QLabel()
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.note, 1)
        bottom.addWidget(self.part_label)
        bottom.addWidget(self.part_box)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(QtWidgets.QLabel('Samples'))
        bottom.addWidget(self.reseed_samples)
        bottom.addWidget(self.btn_reseed)
        bottom.addSpacing(12)
        bottom.addWidget(self.cluster_box)
        bottom.addWidget(self.btn_send)
        bottom.addWidget(self.btn_dream)

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self.header)
        v.addLayout(top)
        v.addWidget(self.status)
        v.addWidget(self.tabs, 1)
        v.addLayout(bottom)

    # -- a run -------------------------------------------------------------
    def start(self, problem, header, total, datasets=()):
        """A new run of problem (a PosteriorProblem); total: the
        simulations it will make (0: a loaded network, seeding only)."""
        self.problem, self.run = problem, None
        self.infos = display_info(problem)
        names = list(getattr(problem.fit, 'part_names', None) or datasets
                     or ['Data'])
        self.part_box.clear()
        self.part_box.addItems(names)
        for w in (self.part_box, self.part_label):
            w.setVisible(len(names) > 1)
        self.header.setText(header)
        self.total = max(int(total), 1)
        self.bar.setRange(0, self.total if total else 0)
        self.bar.setValue(0)
        self.status.setText('Starting…')
        self.report.setPlainText('')
        self.note.setText('')
        self._rounds = []
        self.loss_plot.clear()
        self._t0 = time.monotonic()
        for page in self.pages.values():
            page.setCurrentIndex(0)
        self.btn_stop.setEnabled(True)
        for b in (self.btn_save, self.btn_send, self.btn_dream,
                  self.btn_reseed, self.reseed_samples):
            b.setEnabled(False)
        self.cluster_box.clear()
        self.tabs.setCurrentWidget(self.report)

    @property
    def running(self):
        return self.problem is not None and self.run is None and \
            self.btn_stop.isEnabled()

    def _elapsed(self):
        return duration(time.monotonic() - self._t0)

    def stage(self, text):
        self.status.setText('%s  (%s)' % (text, self._elapsed()))

    def coarse(self, report):
        from fit.npe.simulator import describe_coarse
        self.report.appendPlainText('Training simulator: '
                                    + describe_coarse(report))

    def sims(self, r, done, total):
        self.bar.setValue(min(r * total + done, self.total))
        self.status.setText('Round %d: %d / %d simulations  (%s)'
                            % (r + 1, done, total, self._elapsed()))

    def epoch_done(self, info):
        """An epoch of the round being trained: the status line and its
        loss curve so far."""
        v = info['validation_loss']
        self.status.setText(
            'Round %d: training, epoch %d, validation loss %.5g; best %s '
            '(stops after %d without improvement)  (%s)'
            % (info['round'], info['epoch'], v[-1],
               'this epoch' if not info['since_best'] else
               '%d epoch(s) ago' % info['since_best'], info['patience'],
               self._elapsed()))
        self._draw_loss(info)

    def round_done(self, stats):
        self._rounds.append(stats)
        self.report.appendPlainText(
            'Round %d: %d simulations (%d failed), proposal %s (accepted '
            '%.3g), %d epochs, best validation loss %.5g, %s'
            % (stats['round'], stats['n_sims'], stats['n_failed'],
               stats['proposal'], stats['acceptance'], stats['epochs'],
               stats['best_validation_loss'], duration(stats['seconds'])))
        self._draw_loss()

    def _draw_loss(self, current=None):
        """The finished rounds' losses, and the round being trained
        (current: an 'epoch' event of fit.npe.train) after them (see
        Training in the module docstring)."""
        p = self.loss_plot
        p.clear()
        grey = pg.mkPen(LOSS_KEY, width=1.5)
        # the legend's first rows say which line is which
        p.plot([], [], pen=grey, symbol='o', symbolSize=6, symbolPen=None,
               symbolBrush=LOSS_KEY,
               name='validation (held out): ends a round')
        p.plot([], [], pen=pg.mkPen(LOSS_KEY, width=1,
                                    style=QtCore.Qt.PenStyle.DashLine),
               symbol='o', symbolSize=5, symbolPen=LOSS_KEY,
               symbolBrush=None, name='training')
        p.plot([], [], pen=None, symbol='star', symbolSize=13,
               symbolPen=LOSS_KEY, symbolBrush=LOSS_KEY,
               name="best epoch: the round's network")
        start = 0
        for r in self._rounds + ([current] if current else []):
            c = CHAIN_COLOURS[(r['round'] - 1) % len(CHAIN_COLOURS)]
            v = np.asarray(r['validation_loss'], dtype=float)
            e = start + np.arange(1, len(v) + 1)
            if start:
                p.addItem(pg.InfiniteLine(
                    start + 0.5, angle=90, label='round %d' % r['round'],
                    labelOpts=dict(position=0.95, color=c),
                    pen=pg.mkPen(LOSS_KEY, width=1,
                                 style=QtCore.Qt.PenStyle.DotLine)))
            p.plot(e, v, pen=pg.mkPen(c, width=1.5), symbol='o',
                   symbolSize=6, symbolPen=None, symbolBrush=c,
                   name='round %d' % r['round'])
            p.plot(e, r['training_loss'],
                   pen=pg.mkPen(c, width=1,
                                style=QtCore.Qt.PenStyle.DashLine),
                   symbol='o', symbolSize=5, symbolPen=c, symbolBrush=None)
            if len(v):
                b = int(np.argmin(v))
                p.plot([e[b]], [v[b]], pen=None, symbol='star',
                       symbolSize=13, symbolPen=c, symbolBrush=c)
            start += len(v)

    def finish(self, run):
        """The run ended: its report, clusters and marginals."""
        self.run = run
        self.btn_stop.setEnabled(False)
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        if not self._rounds and run.result.rounds:
            self._rounds = list(run.result.rounds)
            self._draw_loss()
        self.report.setPlainText(report_text(run))
        cal = run.calibration
        best = min(c['chi2'] for c in run.seeds.clusters)
        self.status.setText(
            'Done in %s: %d cluster(s), best χ²/N %.5g; %d seeds for DREAM.'
            % (self._elapsed(), len(run.seeds.clusters), best,
               len(run.seeds.x0)))
        self.note.setText(
            '' if cal is None else
            '<span style="color:%s">round 1 %s</span>'
            % ((OK, 'calibrated') if cal['passed']
               else (BAD, 'not calibrated')))
        self._fill_table()
        self.marg_combo.blockSignals(True)
        self.marg_combo.clear()
        self.marg_combo.addItems([axis_label(i) for i in self.infos])
        self.marg_combo.blockSignals(False)
        self._draw_marginal()
        self._draw_widths()
        self._draw_check()
        for page in self.pages.values():
            page.setCurrentIndex(1)
        self.cluster_box.clear()
        self.cluster_box.addItems(
            ['cluster %d (χ²/N %.4g)' % (k, c['chi2'])
             for k, c in enumerate(run.seeds.clusters, 1)])
        n = (getattr(run.result, 'seeding', None) or {}).get('n_samples')
        if n:
            self.reseed_samples.setValue(int(n))
        for b in (self.btn_save, self.btn_send, self.btn_dream,
                  self.btn_reseed, self.reseed_samples):
            b.setEnabled(True)

    def fail(self, msg):
        self.btn_stop.setEnabled(False)
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.status.setText('Stopped.' if msg == 'stopped' else
                            '<span style="color:%s">NPE failed: %s</span>'
                            % (BAD, msg))
        self.btn_save.setEnabled(False)

    # -- results -----------------------------------------------------------
    def _fill_table(self):
        cl = self.run.seeds.clusters
        heads = ['#', 'weight', 'chains', 'χ²/N', 'at a bound'] + \
            [axis_label(i) for i in self.infos]
        self.table.clear()
        self.table.setColumnCount(len(heads))
        self.table.setRowCount(len(cl))
        self.table.setHorizontalHeaderLabels(heads)
        for k, c in enumerate(cl):
            vals = ['%d' % (k + 1), '%.3g' % c['weight'], '%d' % c['chains'],
                    '%.5g' % c['chi2'], ', '.join(c['at_bound']) or '—']
            vals += ['%.5g' % (v * s) for v, (_, s, _)
                     in zip(c['best_x'], self.infos)]
            for j, t in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(t)
                if j != 4:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight |
                        QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(k, j, item)
        self.table.resizeColumnsToContents()

    def _table_picked(self):
        rows = {i.row() for i in self.table.selectedItems()}
        if len(rows) == 1:
            self.cluster_box.setCurrentIndex(rows.pop())

    def _draw_marginal(self, *_):
        p = self.marg_plot
        p.clear()
        if self.run is None:
            return
        j = max(self.marg_combo.currentIndex(), 0)
        _, s, _ = self.infos[j]
        x = self.problem.to_x(self.run.seeds.u)[:, j] * s
        p.addItem(vbars(x))
        p.setLabel('bottom', axis_label(self.infos[j]))
        for k, c in enumerate(self.run.seeds.clusters):
            p.addItem(pg.InfiniteLine(
                c['best_x'][j] * s, angle=90,
                pen=pg.mkPen(CHAIN_COLOURS[k % len(CHAIN_COLOURS)],
                             width=1.5)))

    def _cluster_index(self):
        """Each sample's cluster, numbered as the table (by weight)."""
        seeds = self.run.seeds
        order = {c['label']: k for k, c in enumerate(seeds.clusters)}
        return np.array([order.get(int(l), -1) for l in seeds.labels])

    def _draw_widths(self):
        """Constraints: per parameter, the network samples' 90 % interval
        over the bounds' (bars) and, when the importance weights are
        reliable (ESS >= RELIABLE_ESS), the exact posterior's (points)."""
        from fit.npe.seed import importance, widths
        p = self.width_plot
        p.clear()
        seeds = self.run.seeds
        w, ess = importance(seeds)
        r = widths(seeds.u)
        n = len(r)
        y = np.arange(n)
        p.addItem(pg.BarGraphItem(x0=0, y=y, height=0.7, width=r, pen=None,
                                  brush=pg.mkBrush('#4c8dff')))
        p.addItem(pg.InfiniteLine(1.0, angle=90, pen=pg.mkPen(
            LOSS_KEY, width=1, style=QtCore.Qt.PenStyle.DashLine)))
        weighted = ess >= RELIABLE_ESS
        if weighted:
            p.addItem(pg.ScatterPlotItem(
                x=widths(seeds.u, w), y=y, symbol='o', size=8, pen=None,
                brush=pg.mkBrush('#f5a623')))
        left = p.getAxis('left')
        left.setTicks([[(k, i[0]) for k, i in enumerate(self.infos)]])
        font = QtGui.QFontMetrics(left.font() if left.style['tickFont']
                                  is None else left.style['tickFont'])
        left.setWidth(max(font.horizontalAdvance(i[0])
                          for i in self.infos) + 20)
        p.setXRange(0, max(1.05, float(r.max()) * 1.05), padding=0)
        p.setYRange(-0.6, n - 0.4, padding=0)
        self.width_view.setMinimumHeight(max(300, 20 * n + 80))
        self.width_note.setText(
            '<i>Bars: the network\'s samples.  %s  Short bar: the data '
            'determine the parameter; near 1: only its bounds hold it '
            '(two separate clusters also widen it).</i>'
            % ('Points: the exact posterior (the samples weighted by exact '
               '/ network).' if weighted else
               'The exact posterior is not shown: too few effective '
               'samples (%.0f, see Network check).' % ess))

    def _draw_check(self):
        """Network check: each sample's exact log posterior against the
        network's log density, by cluster; a perfect network puts them on
        a line of slope 1.  The effective sample size above."""
        from fit.npe.seed import importance
        p = self.check_plot
        p.clear()
        seeds = self.run.seeds
        lq = np.asarray(seeds.logq, float)
        lp = np.asarray(seeds.logp, float)
        k = self._cluster_index()
        for j in range(len(seeds.clusters)):
            on = k == j
            if on.any():
                p.addItem(pg.ScatterPlotItem(
                    x=lq[on], y=lp[on], size=4, pen=None,
                    brush=pg.mkBrush(CHAIN_COLOURS[j % len(CHAIN_COLOURS)]),
                    name='cluster %d' % (j + 1)))
        # the window: the bulk of the samples, not their far tails
        lo = [np.percentile(v, 1) for v in (lq, lp)]
        hi = [v.max() for v in (lq, lp)]
        c = float(np.median(lp - lq))
        a, b = min(lo[0], lo[1] - c), max(hi[0], hi[1] - c)
        p.addItem(pg.PlotDataItem(
            [a, b], [a + c, b + c], name='slope 1: a perfect network',
            pen=pg.mkPen(LOSS_KEY, width=1.5,
                         style=QtCore.Qt.PenStyle.DashLine)))
        for side, u, v in (('x', lo[0], hi[0]), ('y', lo[1], hi[1])):
            m = 0.05 * (v - u) + 1e-9
            getattr(p, 'set%sRange' % side.upper())(u - m, v + m, padding=0)
        _, ess = importance(seeds)
        n = len(lp)
        # spreads of log p and log q: far apart, the network has the
        # posterior's place but not its width
        iqr = lambda v: np.subtract(*np.percentile(v, [75, 25]))
        sp, sq = iqr(lp), iqr(lq)
        ratio = sp / sq if sp > 0 and sq > 0 else 1.0
        width = ''
        if ratio > SPREAD:
            width = ('  Here log p spreads over %.2g times the range of '
                     'log q: the posterior is much sharper than the '
                     'network, which only shows where it is.' % ratio)
        elif ratio < 1 / SPREAD:
            width = ('  Here log q spreads over %.2g times the range of '
                     'log p: the network is sharper than the posterior '
                     'and may miss part of it.' % (1 / ratio))
        self.check_note.setText(
            '<i>Effective sample size <b>%.0f</b> of %d (%.2g %%): the '
            'samples that count once weighted by exact / network.  Points '
            'along the line, a large ESS: the network matches the '
            'posterior.  A wide cloud, a small ESS: it only roughly does, '
            'and DREAM does the rest.%s</i>' % (ess, n, 100 * ess / n, width))

    def _send(self):
        k = self.cluster_box.currentIndex()
        if self.run is None or k < 0:
            return
        self.sendRequested.emit(self.run.seeds.clusters[k]['best_x'],
                                max(self.part_box.currentIndex(), 0),
                                'cluster %d' % (k + 1))

    def _save(self):
        if self.run is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save network', os.path.join(SESSIONS_DIR, 'network.npe'),
            'NPE network (*.npe);;All files (*)', options=DIALOG_OPTIONS)
        if not path:
            return
        try:
            self.run.result.save(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Save network',
                                           'Could not save:\n%s' % exc)
            return
        self.note.setText('Saved %s.' % os.path.basename(path))

