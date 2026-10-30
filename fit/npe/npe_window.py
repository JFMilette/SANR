"""
NPE window: a neural posterior estimation run (fit.npe) in a tab of the
main window, as the DREAM window is for DREAM (fit.dream.dream_window).

A run (NPEThread) probes the parameters' symmetries (fit.npe.symmetry),
chooses the training simulator (fit.npe.simulator choose_coarse, unless
coarse training is off), trains the network round after round
(fit.npe.train), checks the calibration of its amortised round 1 (when
asked), and seeds the DREAM chains with the exact posterior
(fit.npe.seed).  A network loaded from a file only does the seeding.  Stop
ends it at the next simulation chunk or training batch.

  Report     what was done: the training simulator and its fidelity, each
             round (simulations, failures, proposal and its acceptance,
             epochs, best validation loss, time), the calibration, the
             symmetry probe, and the clusters as text
  Training   the validation loss (solid) and training loss (dashed) of
             every epoch, one colour per round, updated after each round
  Clusters   the candidate solutions (fit.npe.seed clusters): weight,
             chains, best chi^2/N, parameters at a bound, best values
  Marginals  one parameter at a time: the network's samples (a proposal,
             not the posterior) and each cluster's best value

"Load into model" sends the best state of the cluster chosen beside it to
the Simulation tab (in a joint fit, as the dataset chosen sees it); "DREAM
from seeds" samples the posterior with DREAM from the seeds, with the
DREAM page's settings (the chains: one per seed).  "Save network…" writes
the network (fit.npe.train NPEResult.save, .npe) for "Load network…" in
the fit box.  The network's samples are never a result: only DREAM's are.
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
    out.append('Clusters of %d network samples (the chains start at the '
               'best ones):' % len(run.seeds.u))
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
        self.loss_plot = plot_item('epoch', 'loss (−log q)')
        self.loss_plot.addLegend(offset=(-10, 10))
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
        self.pages = {}
        for name, w in (('Clusters', self.table), ('Marginals', marg)):
            page = QtWidgets.QStackedWidget()
            page.addWidget(placeholder('Shown when the seeds are ready.'))
            page.addWidget(w)
            self.pages[name] = page
        self.tabs.addTab(self.report, 'Report')
        self.tabs.addTab(self.loss, 'Training')
        self.tabs.addTab(self.pages['Clusters'], 'Clusters')
        self.tabs.addTab(self.pages['Marginals'], 'Marginals')

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
        for b in (self.btn_save, self.btn_send, self.btn_dream):
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

    def round_done(self, stats):
        self._rounds.append(stats)
        self.report.appendPlainText(
            'Round %d: %d simulations (%d failed), proposal %s (accepted '
            '%.3g), %d epochs, best validation loss %.5g, %s'
            % (stats['round'], stats['n_sims'], stats['n_failed'],
               stats['proposal'], stats['acceptance'], stats['epochs'],
               stats['best_validation_loss'], duration(stats['seconds'])))
        self._draw_loss()

    def _draw_loss(self):
        self.loss_plot.clear()
        start = 0
        for r in self._rounds:
            c = CHAIN_COLOURS[(r['round'] - 1) % len(CHAIN_COLOURS)]
            e = start + np.arange(len(r['validation_loss']))
            self.loss_plot.plot(e, r['validation_loss'],
                                pen=pg.mkPen(c, width=1.5),
                                name='round %d' % r['round'])
            self.loss_plot.plot(e, r['training_loss'], pen=pg.mkPen(
                c, width=1, style=QtCore.Qt.PenStyle.DashLine))
            start += len(e)

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
        for page in self.pages.values():
            page.setCurrentIndex(1)
        self.cluster_box.clear()
        self.cluster_box.addItems(
            ['cluster %d (χ²/N %.4g)' % (k, c['chi2'])
             for k, c in enumerate(run.seeds.clusters, 1)])
        for b in (self.btn_save, self.btn_send, self.btn_dream):
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

