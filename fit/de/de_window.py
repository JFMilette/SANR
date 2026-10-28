"""
DE window: follows a running differential-evolution fit (fit.de.de_fit, run
in a DEThread) of the Experimental tab generation by generation, or a local
L-BFGS-B or least-squares fit (fit.de.de_fit.run_lbfgs, run_lsq) iteration
by iteration, one of scipy's other global methods (fit.de.global_fit) by
its own steps, or a
multistart (fit.multistart, run in a fit.panel.MultistartThread) start by
start.

The history lists the best cost of every generation (and the polished
result at the end); for a multistart, the distinct minima of the starts
ended so far, updated as each one ends (the entry selected stays selected,
the best otherwise), best first, with how many starts reached each (✓:
the current values' start among them).  Selecting an entry draws, in one tab per fitted
dataset (several in a joint fit, fit.joint), its depth profile (with its
sublayers) and its model against that dataset's fitted data, as any
quantity of the Simulation tab (reflectivities, half-polarized,
asymmetries; the model's channels outside the fit use the Simulation tab's
pairs), or the residuals (R_data - R_model) / sigma of every fitted point,
with the fit's own points and sigma (error floor included; dR, or |R|
without it, under the log R cost), so their mean square per channel is
the chi^2/N it adds; once the fit is done or stopped, "Send to Simulation" writes that
entry's parameters, as the dataset of the tab shown sees them, into the
Simulation tab.  Nothing reaches the Simulation tab otherwise.

Each tab keeps its own copy of the model, taken before the fit thread
starts, so drawing never touches the stack the fit is evaluating; only the
tab shown is drawn, the others when they are shown.
"""

import copy
import time
from html import escape as html_escape

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from fit.de.de_fit import run_de_restarts, run_lbfgs, run_lsq
from fit.de import global_fit
from fit.joint import JointProblem, part_x, parts
from fit.multistart import merge
import line_colours
import style
from tabs.simulation import (PARAM_DISPLAY, PROFILE_EXPORT_NAMES,
                             PROFILE_QUANTITIES, REFL_QUANTITIES, RIGHT,
                             Crosshair, LayerLabels, export_name, plot_widget,
                             profile_data, profile_layers, profile_readout)


AT_BOUND = QtGui.QColor('#f87171')
# the fit tabs' plot list: the Simulation tab's quantities, then this one
RESIDUALS = 'Residuals'
RESIDUAL_FORMULA = "(R_data − R_model) / σ"


def at_bound(v, lo, hi):
    """A value pinned at a bound usually means the bound is too tight."""
    return min(v - lo, hi - v) < 1e-3 * (hi - lo)


class DEThread(QtCore.QThread):
    """Runs fit.de.de_fit.run_de_restarts on a FitProblem off the GUI
    thread; with restarts > 1, `minima` gives the distinct ends of the runs
    (fit.multistart.merge) before finished_fit gives the best.  With
    method='lbfgs' or 'lsq', fit.de.de_fit.run_lbfgs or run_lsq instead
    (progress then counts iterations; nothing to continue); with a method
    of fit.de.global_fit, global_fit.run_global (its UNITS; `minima` gives
    SHGO's global and local minima before finished_fit)."""

    progress = QtCore.pyqtSignal(object, float, int)      # x, cost, generation
    status = QtCore.pyqtSignal(object)        # fit.de.de_fit STATUS info
    finished_fit = QtCore.pyqtSignal(object, float, int, str)
    minima = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, problem, method='de', **options):
        super().__init__()
        self.problem, self.options = problem, options
        self.method = method
        self._stop = False
        # where the run is (fit.de.de_fit CONTINUE): resume=state goes on
        self.state = {}

    def stop(self):
        self._stop = True

    def run(self):
        if self.method in ('lbfgs', 'lsq') + global_fit.METHODS:
            run = {'lbfgs': run_lbfgs, 'lsq': run_lsq}.get(
                self.method, lambda problem, **o: global_fit.run_global(
                    problem, self.method, minima=lambda ends: self.minima.emit(
                        merge(problem, ends)), **o))
            try:
                x, cost, it, msg = run(
                    self.problem, callback=self.progress.emit,
                    status=self.status.emit, cancelled=lambda: self._stop,
                    **self.options)
            except Exception as exc:              # report, don't kill the app
                self.failed.emit('%s: %s' % (type(exc).__name__, exc))
                return
            self.finished_fit.emit(x, cost, it, msg)
            return
        try:
            (x, cost, gen, msg), ends = run_de_restarts(
                self.problem, callback=self.progress.emit,
                status=self.status.emit, cancelled=lambda: self._stop,
                state=self.state, **self.options)
        except Exception as exc:                  # report, don't kill the app
            self.failed.emit('%s: %s' % (type(exc).__name__, exc))
            return
        if len(ends) > 1:
            self.minima.emit(merge(self.problem, ends))
        self.finished_fit.emit(x, cost, gen, str(msg))


class DEWindow(QtWidgets.QWidget):
    """History of one fit with, for each dataset, its profile and
    reflectivity plots."""

    stopRequested = QtCore.pyqtSignal()
    continueRequested = QtCore.pyqtSignal()
    # x, entry label, dataset (tab) whose values are sent
    sendRequested = QtCore.pyqtSignal(object, str, int)

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle('Fit')
        self.resize(1400, 850)
        self.problem = None
        self.views = []                # DatasetView of each dataset
        self.entries = []              # [(label, cost, x)]
        self.running = False
        self.method, self.unit = 'de', 'generation'     # set by start()
        self._x = None                 # parameters currently drawn

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_side())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._draw_view)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1040])
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(splitter)

    # -- layout -------------------------------------------------------------
    def _build_side(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        # what the running fit is doing (set_progress): changes every
        # generation even when the best cost does not
        self.progress = QtWidgets.QLabel()
        self.progress.setWordWrap(True)
        self.progress.setProperty('role', 'muted')
        v.addWidget(self.progress)

        self.history = QtWidgets.QTreeWidget()
        self.history.setColumnCount(2)
        self.history.setRootIsDecorated(False)
        self.history.setUniformRowHeights(True)
        self.history.setAlternatingRowColors(True)
        self.history.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.history.setToolTip('Best cost of every generation; select one '
                                'to draw it')
        self.history.currentItemChanged.connect(self._on_select)
        v.addWidget(self.history, 3)

        row = QtWidgets.QHBoxLayout()
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_stop.setToolTip('Stop the fit; the generations so far are '
                                 'kept')
        self.btn_send = QtWidgets.QPushButton('Send to Simulation')
        self.btn_send.setToolTip('Write the parameters of the selected entry '
                                 'into the Simulation tab (in a joint fit, '
                                 'as the dataset of the tab shown sees '
                                 'them)')
        self.btn_continue = QtWidgets.QPushButton('Continue')
        self.btn_continue.setToolTip(
            'Go on from where the fit ended: its last population, with the '
            'same parameters, bounds and data,\nfor up to Max iter more '
            'generations (the DE page\'s other settings as they are now)')
        self.btn_continue.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_continue.clicked.connect(self._continue)
        self.btn_send.clicked.connect(self._send)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_continue)
        row.addWidget(self.btn_send)
        v.addLayout(row)

        v.addWidget(QtWidgets.QLabel('Parameters of the selected entry:'))
        self.params = QtWidgets.QTreeWidget()
        self.params.setColumnCount(4)
        self.params.setHeaderLabels(['Layer', 'Parameter', 'Value', 'Unit'])
        self.params.setRootIsDecorated(False)
        self.params.setToolTip('Red: at a fit bound (widen it?)')
        v.addWidget(self.params, 2)
        return panel

    # -- fit lifecycle (called by the Experimental tab) ---------------------
    def start(self, problem, views, title, cost_label, polish=True,
              method='de'):
        """A new fit: problem is the FitProblem or fit.joint.JointProblem
        the thread will run (its datasets' stacks are copied now, before
        the thread starts); views one dict per dataset, in the problem's
        order: name, data (the fitted points {channel: dict(Q, R, dR,
        dQ)}), vectors (the {channel: [Pi, Pa]} of every model channel),
        grids (the {channel: Q} of the model curves,
        tabs.experimental.model_grid); polish whether the fit ends with a
        polish; method 'de', 'lbfgs' or 'lsq' (a local fit: iterations,
        no polish), one of fit.de.global_fit.METHODS (its own steps) or
        'multistart' (the minima, given by show_minima)."""
        self.problem = problem
        self.method = method
        self.unit = dict(global_fit.UNITS, lbfgs='iteration',
                         lsq='iteration', multistart='minimum').get(
                             method, 'generation')
        self.polish = polish and method == 'de'
        self.cost_label = cost_label
        self.tabs.blockSignals(True)
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()
        self.views = []
        for v, sub in zip(views, parts(problem)):
            # the fitted channels' fixed norms (Licorne's Norm_factor)
            view = DatasetView(copy.deepcopy(sub.stack), v['data'],
                               v['vectors'], v['grids'],
                               {name[1:]: n for (name, *_), n
                                in zip(sub.channels, sub.norms)},
                               fit=copy.deepcopy(sub))
            if getattr(sub, 'combined', None) is not None and \
                    sub.target == 'asym':
                # the fitted quantity: SA of the pair combined
                up = sub.channels[sub.combined[0]][0]
                view.quantity.setCurrentIndex(
                    [q[0] for q in REFL_QUANTITIES].index(
                        'SA half-polarized' if up == 'R+' else 'SA NSF'))
            self.views.append(view)
            self.tabs.addTab(view, v['name'])
        self.tabs.blockSignals(False)
        self.entries = []
        self.running = True
        self.setWindowTitle('Fit — %s' % title)
        self.history.blockSignals(True)
        self.history.clear()
        self.history.setHeaderLabels([self.unit.capitalize(), cost_label])
        self.history.setToolTip(
            'Distinct minima, best first, and how many starts ended in each '
            '(✓: the current values\' start did); select one to draw it'
            if method == 'multistart' else
            'Best cost of every %s; select one to draw it' % self.unit)
        self.history.blockSignals(False)
        joint = isinstance(problem, JointProblem)
        self.params.setColumnCount(5 if joint else 4)
        self.params.setHeaderLabels(['Layer', 'Parameter', 'Value', 'Unit']
                                    + (['Dataset'] if joint else []))
        self.status.setText('Fitting %d parameter(s) to %d points%s%s…'
                            % (len(problem.params), problem.npoints,
                               ' of %d datasets' % len(views) if joint
                               else '', {
                                   'lbfgs': ' (L-BFGS-B from the current '
                                            'values)',
                                   'lsq': ' (least squares from the '
                                          'current values)',
                                   'multistart': ' from many starts',
                                   **{m: ' (%s)' % label for m, label
                                      in global_fit.LABELS.items()}
                               }.get(method, '')))
        self.status.setStyleSheet('')
        self.progress.setText({
            'lbfgs': 'Cost at the current values…',
            'lsq': 'Cost at the current values…',
            'multistart': 'First start…',
            **{m: 'Cost at the current values…'
               for m in global_fit.METHODS}}.get(
                method, 'First generation (the starting population and one '
                        'generation)…'))
        self.progress.show()
        self._gen_times = []                  # time of each 'de' report
        # a multistart: (done, total), the running starts' last reports
        # and the ended ones' costs (set_starts, set_start_status)
        self._ms_count, self._ms_run, self._ms_ended = (0, 0), {}, {}
        self._ms_ends = []          # (start, u, cost) of the ended starts
        self._end_minima = None     # the minima a fit ends with (set_minima)
        self.btn_stop.setEnabled(True)
        self.btn_continue.setEnabled(False)
        self.btn_send.setEnabled(False)
        # the starting model until the first generation arrives: the
        # current values clipped into the bounds, as the fit starts from
        self._draw(problem.to_x(problem.x0()))

    def resume(self):
        """The fit goes on (Continue): same problem, history kept."""
        self.running = True
        self.status.setText('Continuing the fit…')
        self.status.setStyleSheet('')
        self.progress.setText('Evaluating the population again…')
        self.progress.show()
        self._gen_times = []
        self.btn_stop.setEnabled(True)
        self.btn_continue.setEnabled(False)
        self.btn_send.setEnabled(False)

    def add_generation(self, x, cost, gen):
        self._append('%d' % gen, cost, x)

    def set_progress(self, info):
        """A fit.de.de_fit STATUS report of the running fit."""
        if not self.running:
            return
        run = ('Run %d / %d — ' % (info['run'], info['runs'])
               if info.get('runs', 1) > 1 else '')
        c = self.cost_label
        if info['phase'] in ('lbfgs', 'lsq'):
            self.progress.setText('%s iteration %d / %d — %s %.6g' % (
                'L-BFGS-B' if info['phase'] == 'lbfgs' else 'Least-squares',
                info['iteration'], info['maxiter'], c, info['best']))
            return
        if info['phase'] == 'global':
            step = '%s %d%s' % (self.unit, info['step'],
                                ' / %d' % info['maxiter']
                                if info['maxiter'] else '')
            self.progress.setText('%s — %s%s — best %s %s' % (
                global_fit.LABELS[info['method']], step,
                ', %d evaluations' % info['evals']
                if info['evals'] is not None else '', c,
                '%.6g' % info['best'] if info['best'] is not None else '…'))
            return
        if info['phase'] == 'polish':
            self.progress.setText(
                '%sPolishing (L-BFGS-B)%s — %s %.6g' % (
                    run, ', iteration %d' % info['iteration']
                    if info['iteration'] else '…', c, info['best']))
            return
        t = self._gen_times
        t.append(time.monotonic())
        del t[:-11]
        rate = ' — %.2g s per generation' % ((t[-1] - t[0]) / (len(t) - 1)) \
            if len(t) > 1 else ''
        self.progress.setText(
            '%sGeneration %d / %d — best %s %.6g, population median %.6g<br>'
            'Spread of the population\'s %s %.2g (stops at ≤ tol %.2g)%s'
            % (run, info['generation'], info['maxiter'], c, info['best'],
               info['median'], c, info['spread'], info['tol'], rate))

    def set_minima(self, minima):
        """The minima a fit ends with (fit.multistart rows, best first:
        SHGO's global and local minima), shown by finish instead of the
        history."""
        self._end_minima = list(minima)

    def finish(self, x, cost, gen, msg):
        self.running = False
        self.progress.hide()
        stopped = msg == 'stopped'
        if self._end_minima and not stopped:
            m = self._end_minima
            self._fill_minima(m, lambda k, _: '%d — %s' % (
                k + 1, 'global' if k == 0 else 'local'))
            self.status.setText(
                '<b>Done</b> after %d %s(s) — %d minim%s: global %s %.6g%s'
                '<br><i>%s</i><br>Select a minimum and send it to the '
                'Simulation tab.' % (
                    gen, self.unit, len(m), 'um' if len(m) == 1 else 'a',
                    self.cost_label, m[0]['chi2'],
                    ', %d local' % (len(m) - 1) if len(m) > 1 else '',
                    html_escape(msg)))
            self.btn_stop.setEnabled(False)
            self.btn_send.setEnabled(self.history.currentItem() is not None)
            return
        # the polish (or a stop before the first generation) gives a result
        # that is not in the history yet.  Compared exactly: any tolerance
        # in physical units would hide small SLD moves (~1e-6 A^-2)
        new = not self.entries or cost != self.entries[-1][1] or \
            not np.array_equal(x, self.entries[-1][2])
        if new:
            self._append('stopped' if stopped else 'polished'
                         if self.method == 'de' else 'final', cost, x)
        note = ''
        if self.polish and not stopped:
            note = '<br>Polish: %s' % (
                '%s %.6g → %.6g' % (self.cost_label, self.entries[-2][1], cost)
                if new and len(self.entries) > 1 else
                'no improvement on the last generation')
        self.status.setText(
            '<b>%s</b> after %d %s(s) — best %.6g<br><i>%s</i>%s<br>'
            'Select an entry and send it to the Simulation tab.'
            % ('Stopped' if stopped else 'Done', gen, self.unit, cost,
               html_escape(msg), note))
        self.btn_stop.setEnabled(False)
        self.btn_send.setEnabled(self.history.currentItem() is not None)

    def set_starts(self, done, total):
        """How many starts of a running multistart have ended."""
        self._ms_count = (done, total)
        self._show_starts()

    def set_start_status(self, info):
        """A fit.multistart PROGRESS report of a running multistart."""
        k = info['start']
        if info['phase'] == 'done':
            self._ms_ended[k] = info['best']
            self._ms_run.pop(k, None)
            if self.running and info.get('u') is not None:
                # the list of minima as the ended starts give it so far
                self._ms_ends.append((k, np.asarray(info['u']),
                                      info['best']))
                self._fill_minima(merge(self.problem, self._ms_ends))
        elif k not in self._ms_ended:         # not a late report
            self._ms_run[k] = info
        self._show_starts()

    def _show_starts(self):
        """The ended starts and best cost, then where each running start
        is."""
        if not self.running:
            return
        c = self.cost_label
        lines = ['%d / %d starts done%s' % (self._ms_count + (
            ' — best %s %.6g' % (c, min(self._ms_ended.values()))
            if self._ms_ended else '',))]
        for k, i in sorted(self._ms_run.items()):
            what = {
                'begin': lambda: 'starting…',
                'de': lambda: '%s, %d / %d evaluated%s' % (
                    'generation %d / %d' % (min(i['step'], i['maxiter']),
                                            i['maxiter'])
                    if i['step'] else 'first population',
                    min(i['evals'], i['size']), i['size'],
                    ' — best %s %.6g' % (c, i['best'])
                    if i['best'] is not None else ''),
                'polish': lambda: 'polish (L-BFGS-B)%s — %s %.6g' % (
                    ', iteration %d' % i['step'] if i['step'] else '', c,
                    i['best']),
                'lbfgs': lambda: 'iteration %d — %s %.6g' % (
                    i['step'], c, i['best']),
                'lsq': lambda: 'least squares, iteration %d — %s %.6g' % (
                    i['step'], c, i['best'])}[i['phase']]()
            lines.append('%s: %s' % ('Current values' if k == 0 else
                                     'Start %d' % k, what))
        self.progress.setText('<br>'.join(lines))

    def show_minima(self, minima, stopped=False):
        """The end of a multistart: its minima (fit.multistart rows, best
        first) as the entries, the best one drawn."""
        self.running = False
        self.progress.hide()
        self._fill_minima(minima)
        hits = sum(m['hits'] for m in minima)
        self.status.setText(
            '<b>%s</b> — %d start(s) ended in %d distinct minim%s, best %s '
            '%.6g<br>Select a minimum and send it to the Simulation tab.' % (
                'Stopped' if stopped else 'Done', hits, len(minima),
                'um' if len(minima) == 1 else 'a', self.cost_label,
                minima[0]['chi2']) if minima else
            '<b>Stopped</b> before any start ended.')
        self.btn_stop.setEnabled(False)
        self.btn_send.setEnabled(self.history.currentItem() is not None)

    def _fill_minima(self, minima, label=None):
        """The entries: the minima (fit.multistart rows, best first), the
        one selected before kept selected (by its parameters), else the
        best; drawn once.  label(k, minimum): an entry's label (default:
        its starts)."""
        e = self._selected()
        keep = None if e is None or self.history.indexOfTopLevelItem(
            self.history.currentItem()) == 0 else e[2]
        self.history.blockSignals(True)
        self.history.clear()
        self.entries = []
        for k, m in enumerate(minima):
            self._append(label(k, m) if label else '%d — %d start%s%s' % (
                k + 1, m['hits'], '' if m['hits'] == 1 else 's',
                ' ✓' if m['from_current'] else ''), m['chi2'], m['x'])
        self.history.setCurrentItem(None)
        self.history.blockSignals(False)
        if minima:
            row = next((k for k, m in enumerate(minima) if keep is not None
                        and np.array_equal(m['x'], keep)), 0)
            self.history.setCurrentItem(self.history.topLevelItem(row))

    def fail(self, msg):
        self.running = False
        self.progress.hide()
        self.status.setText('Fit failed: %s' % html_escape(msg))
        self.status.setStyleSheet('color: %s' % style.ERROR)
        self.btn_stop.setEnabled(False)
        self.btn_send.setEnabled(self.history.currentItem() is not None)

    def _append(self, label, cost, x):
        h = self.history
        # follow the newest entry unless the user picked an older one
        cur = h.currentItem()
        follow = cur is None or h.indexOfTopLevelItem(cur) == len(self.entries) - 1
        self.entries.append((label, float(cost), np.array(x, dtype=float)))
        it = QtWidgets.QTreeWidgetItem([label, '%.6g' % cost])
        it.setTextAlignment(1, QtCore.Qt.AlignmentFlag.AlignRight
                            | QtCore.Qt.AlignmentFlag.AlignVCenter)
        h.addTopLevelItem(it)
        # mark the best entry so far
        best = int(np.argmin([e[1] for e in self.entries]))
        for i in range(h.topLevelItemCount()):
            font = h.topLevelItem(i).font(0)
            font.setBold(i == best)
            for c in range(2):
                h.topLevelItem(i).setFont(c, font)
        if follow:
            h.setCurrentItem(it)
            h.scrollToItem(it)

    # -- user actions -------------------------------------------------------
    def _stop(self):
        self.btn_stop.setEnabled(False)
        self.stopRequested.emit()

    def _continue(self):
        self.btn_continue.setEnabled(False)
        self.continueRequested.emit()

    def _selected(self):
        it = self.history.currentItem()
        return None if it is None else \
            self.entries[self.history.indexOfTopLevelItem(it)]

    def _on_select(self, *_):
        e = self._selected()
        self.btn_send.setEnabled(e is not None and not self.running)
        if e is not None:
            self._draw(e[2])

    def _send(self):
        e = self._selected()
        if e is not None and not self.running:
            self.sendRequested.emit(e[2], e[0], max(self.tabs.currentIndex(),
                                                    0))

    def closeEvent(self, ev):
        # closing only hides: the fit and its history carry on
        self.hide()
        ev.ignore()

    # -- drawing ------------------------------------------------------------
    def _draw(self, x):
        self._x = np.asarray(x, dtype=float)
        self._fill_params()
        self._draw_view()

    def _draw_view(self, *_):
        """Draw the tab shown, unless it already shows these parameters."""
        k = self.tabs.currentIndex()
        if self._x is None or not 0 <= k < len(self.views) or \
                self.views[k].drawn is self._x:
            return
        try:
            self.views[k].draw(self._x, lambda x, st: parts(
                self.problem)[k].apply(part_x(self.problem, x, k), st),
                part_x(self.problem, self._x, k))
        except Exception as exc:                      # keep the window alive
            self.status.setText('Cannot draw this entry: %s' % exc)
            self.status.setStyleSheet('color: %s' % style.ERROR)

    def _fill_params(self):
        t = self.params
        t.clear()
        layers = self.problem.stack.layers
        scopes = getattr(self.problem, 'scopes', None)
        for j, ((i, attr, _, lo, hi), v) in enumerate(
                zip(self.problem.params, self._x)):
            label, scale, unit = PARAM_DISPLAY[attr]
            row = ['(stack)' if i is None else self.problem.stack.entry_name(
                i) if isinstance(i, tuple) else layers[i].name, label,
                '%.6g' % (v * scale), unit]
            if scopes is not None:
                row.append(scopes[j] or 'shared')
            it = QtWidgets.QTreeWidgetItem(row)
            it.setTextAlignment(2, QtCore.Qt.AlignmentFlag.AlignRight
                                | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if at_bound(v, lo, hi):
                it.setForeground(2, AT_BOUND)
                it.setToolTip(2, 'At its fit bound [%.6g, %.6g]'
                              % (lo * scale, hi * scale))
            t.addTopLevelItem(it)
            if (i, attr) == (None, 'thickness_spread'):
                self._spread_rows(it, j, scopes)
        for c in range(t.columnCount()):
            t.resizeColumnToContents(c)

    def _spread_rows(self, it, j, scopes):
        """A thickness spread's row says its target and distribution, with
        what it means below it (Stack.spread_summary), at the entry drawn,
        as the dataset of its scope sees it (a shared one: the first)."""
        names = getattr(self.problem, 'part_names', None)
        k = names.index(scopes[j]) if scopes is not None and scopes[j] \
            else 0
        sub = parts(self.problem)[k]
        st = copy.deepcopy(sub.stack)
        sub.apply(part_x(self.problem, self._x, k), st)
        it.setText(0, st.spread_target_name())
        it.setText(1, 'Thickness spread (%s)' % {
            'gaussian': 'Gaussian', 'lognormal': 'log-normal'}[
                st.thickness_spread_dist])
        for label, v, unit in st.spread_summary():
            row = QtWidgets.QTreeWidgetItem(['', '  ' + label, '%.4g' % v,
                                             unit])
            row.setTextAlignment(2, QtCore.Qt.AlignmentFlag.AlignRight
                                 | QtCore.Qt.AlignmentFlag.AlignVCenter)
            for c in range(4):
                row.setForeground(c, QtGui.QColor(style.MUTED))
            self.params.addTopLevelItem(row)


class DatasetView(QtWidgets.QSplitter):
    """The depth profile and the model against the fitted data of one
    dataset, in a tab of the DE window."""

    def __init__(self, stack, data, vectors, grids, norms, fit=None):
        """stack: the view's own copy of the model; data, vectors, grids,
        norms: see DEWindow.start; fit: the view's own copy of its
        dataset's fit.problem.FitProblem, for the residuals."""
        super().__init__(QtCore.Qt.Orientation.Vertical)
        self.stack = stack
        self.data, self.vectors, self.model_Q = data, vectors, grids
        self.norms = norms
        self.fit = fit
        self.drawn = None              # parameters drawn last
        self._xp = None                # the fit's parameters drawn
        self._resid = None             # residuals of _xp, when first shown
        self._profile = None           # profile_data of the drawn stack
        self.addWidget(self._build_profile_plot())
        self.addWidget(self._build_reflectance_plot())
        self.setSizes([380, 470])

    def draw(self, values, apply, part_values=None):
        """The model with the stack's parameter values `values`, written
        into the view's stack by apply(values, stack); part_values: those
        of the view's fit problem, for the residuals."""
        self.drawn = values
        self._xp, self._resid = part_values, None
        apply(values, self.stack)
        self.stack.build_sublayers()
        self._draw_profile()
        # imported here: the Experimental tab imports this module
        from tabs.experimental import model_channels
        self._model = model_channels(self.stack, self.model_Q, self.vectors,
                                     self.norms)
        self._redraw_reflectance()

    def _build_profile_plot(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel('Depth profile:'))
        self.prof_boxes = []
        for name, _, col in PROFILE_QUANTITIES:
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet('color: %s' % col)
            cb.toggled.connect(self._show_profile_curves)
            self.prof_boxes.append(cb)
            bar.addWidget(cb)
        self.prof_show_smooth = QtWidgets.QCheckBox('smooth')
        self.prof_show_smooth.setChecked(True)
        self.prof_show_smooth.setToolTip('The continuous profile Stack.profile(z)')
        self.prof_show_smooth.toggled.connect(self._show_profile_curves)
        bar.addWidget(self.prof_show_smooth)
        self.prof_show_bars = QtWidgets.QCheckBox('sublayers')
        self.prof_show_bars.setChecked(True)
        self.prof_show_bars.setToolTip('The slabs the reflectivity is '
                                       'computed from')
        self.prof_show_bars.toggled.connect(self._show_profile_curves)
        bar.addWidget(self.prof_show_bars)
        bar.addStretch(1)
        v.addLayout(bar)

        w, p = plot_widget('depth z (Å)', 'SLD (10⁻⁶ Å⁻²)')
        self.prof_plot = p
        # angles on their own right axis, as in the Simulation tab
        self.prof_vb2 = pg.ViewBox()
        p.showAxis('right')
        p.scene().addItem(self.prof_vb2)
        p.getAxis('right').linkToView(self.prof_vb2)
        p.getAxis('right').setLabel('MSLD φ, θ (deg)')
        p.getAxis('right').enableAutoSIPrefix(False)
        self.prof_vb2.setXLink(p)

        def sync():
            self.prof_vb2.setGeometry(p.vb.sceneBoundingRect())
            self.prof_vb2.linkedViewChanged(p.vb, self.prof_vb2.XAxis)
        p.vb.sigResized.connect(sync)

        self.prof_bars, self.prof_curves = [], []
        for i, (_, _, col) in enumerate(PROFILE_QUANTITIES):
            bars = pg.BarGraphItem(x0=[], width=[], height=[],
                                   brush=pg.mkBrush(col + '33'),
                                   pen=pg.mkPen(col, width=0.6))
            curve = pg.PlotDataItem(pen=pg.mkPen(col, width=2))
            # the bars straight into the view box (the Matplotlib exporter
            # chokes on BarGraphItem); the curves where pyqtgraph's CSV
            # export finds them (PlotItem.items), as in the Simulation tab
            if i in RIGHT:
                self.prof_vb2.addItem(bars)
                self.prof_vb2.addItem(curve)
                p.items.append(curve)
            else:
                p.vb.addItem(bars)
                p.addItem(curve)
            export_name(curve, PROFILE_EXPORT_NAMES[i])
            self.prof_bars.append(bars)
            self.prof_curves.append(curve)
        p.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#4b4e55', width=0.7)))
        self.prof_lines = []                  # interface lines
        self.prof_layers = LayerLabels(p)     # band and name of each layer
        self.prof_cursor = Crosshair(w, p,
                                     lambda x: profile_readout(self, x))
        p.layer_at = lambda z: profile_layers(self, z)
        v.addWidget(w)
        return box

    def _build_reflectance_plot(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel('Reflectivity plot:'))
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItems([q[0] for q in REFL_QUANTITIES] + [RESIDUALS])
        self.quantity.setToolTip(
            'Fitted data (markers) and model (lines); Residuals: '
            '(R_data − R_model) / σ of each fitted point, with the fit\'s σ')
        self.quantity.currentIndexChanged.connect(self._redraw_reflectance)
        bar.addWidget(self.quantity)
        self.formula = QtWidgets.QLabel()
        self.formula.setProperty('role', 'muted')
        bar.addWidget(self.formula)
        bar.addStretch(1)
        self.logy = QtWidgets.QCheckBox('log R')
        self.logy.setChecked(True)
        self.rq4 = QtWidgets.QCheckBox('R·Q⁴')
        self.show_err = QtWidgets.QCheckBox('error bars')
        self.show_err.setChecked(True)
        for cb in (self.logy, self.rq4, self.show_err):
            cb.toggled.connect(self._redraw_reflectance)
            bar.addWidget(cb)
        bar.addWidget(line_colours.button())
        v.addLayout(bar)
        line_colours.notifier().changed.connect(self._redraw_reflectance)
        w, self.refl_plot = plot_widget('Q (Å⁻¹)', 'Reflectivity')
        self.refl_plot.addLegend(offset=(-10, 10))
        self.zero = pg.InfiniteLine(pos=0, angle=0,
                                    pen=pg.mkPen('#4b4e55', width=0.7))
        self.refl_plot.addItem(self.zero)
        self._refl_items = []                 # everything drawn, for clearing
        self._model = None                    # model channels, as self.data
        v.addWidget(w)
        self.refl_note = QtWidgets.QLabel()
        self.refl_note.setStyleSheet('color: %s' % style.ERROR)
        v.addWidget(self.refl_note)
        return box

    def _draw_profile(self):
        self._profile, self._on_bars = profile_data(self.stack)
        z, Z = self._profile[:2]
        p = self.prof_plot
        for ln in self.prof_lines:
            p.removeItem(ln)
        self.prof_lines = []
        for Zj in Z:
            ln = pg.InfiniteLine(pos=Zj, angle=90,
                                 pen=pg.mkPen('#62666d', width=1))
            ln.setZValue(-50)
            p.addItem(ln)
            self.prof_lines.append(ln)
        # names and the selected entry's thicknesses
        self.prof_layers.set_layers(self.stack.layers, Z, z[0], z[-1])
        self._show_profile_curves()
        p.setXRange(z[0], z[-1], padding=0)
        self.prof_layers.place()

    def _show_profile_curves(self, *_):
        """As the Simulation tab's _show_profile, without the layer bands."""
        if self.drawn is None:
            return
        z, _, e, curves, slabs = self._profile
        show_bars = self.prof_show_bars.isChecked()
        show_smooth = self.prof_show_smooth.isChecked()
        if show_bars:
            curves = [self._on_bars.get(i, c) for i, c in enumerate(curves)]
        for i, cb in enumerate(self.prof_boxes):
            on = cb.isChecked()
            self.prof_curves[i].setData(z, curves[i], connect='finite')
            self.prof_curves[i].setVisible(on and show_smooth)
            # a blank angle (NaN) is a bar of zero height
            self.prof_bars[i].setOpts(x0=e[:-1], width=np.diff(e),
                                      height=np.nan_to_num(slabs[i]), y0=0)
            self.prof_bars[i].setVisible(on and show_bars)
        p = self.prof_plot
        p.showAxis('right', any(self.prof_boxes[i].isChecked() for i in RIGHT))
        p.enableAutoRange(axis='y')
        self.prof_vb2.enableAutoRange(axis='y')
        self.prof_cursor.refresh()

    def _redraw_reflectance(self, *_):
        # imported here: the Experimental tab imports this module
        from tabs.experimental import dataset_series
        p = self.refl_plot
        for item in self._refl_items:
            p.removeItem(item)
        self._refl_items = []
        p.legend.clear()
        if self._model is None:
            return
        k = self.quantity.currentIndex()
        if k == len(REFL_QUANTITIES):
            self._draw_residuals()
            return
        self.show_err.setEnabled(True)
        name, label, formula = REFL_QUANTITIES[k]
        is_refl = label is None
        # log / R·Q⁴ only apply to raw reflectivities
        for w in (self.logy, self.rq4):
            w.setEnabled(is_refl)
        self.zero.setVisible(not is_refl)
        self.formula.setText(formula)
        log = is_refl and self.logy.isChecked()
        rq4 = is_refl and self.rq4.isChecked()
        p.setLabel('left', label or ('R·Q⁴ (Å⁻⁴)' if rq4 else 'Reflectivity'))
        p.setLogMode(x=False, y=log)

        data, note = dataset_series(self.data, k)
        # the raw reflectivities: only the fitted channels' model
        model = {c: d for c, d in self._model.items()
                 if k != 0 or c in self.data}
        for lab, col, Q, y, _, _ in dataset_series(model, k, line=True)[0]:
            y = y * Q**4 if rq4 else y
            if log:
                y = np.where(y > 0, y, np.nan)
            col = col or line_colours.get('asym')
            curve = pg.PlotDataItem(Q, y, pen=pg.mkPen(col, width=1.8),
                                    connect='finite')
            p.addItem(curve)
            self._refl_items.append(curve)
        for lab, col, Q, y, dy, _ in data:
            if rq4:
                y, dy = y * Q**4, dy * Q**4
            self._add_data(lab, col or line_colours.get('asym'), Q, y, dy,
                           log)
        self.refl_note.setText('Fitted data: %s' % note if note else '')
        self.refl_note.setVisible(bool(note))
        p.enableAutoRange()

    def residuals(self):
        """[(channel name, Q, (R_data - R_model) / sigma)] of each fitted
        channel at the parameters drawn, the fit's points and sigma (under
        the log R cost, which has none: dR, |R| where dR <= 0, with the
        error floor), computed once per entry drawn."""
        if self._resid is None and self.fit is not None and \
                self._xp is not None:
            f = self.fit
            out = []
            for m, (name, idx, R, sigma) in zip(f.model(self._xp),
                                                f.channels):
                if sigma is None:
                    d = self.data.get(name[1:], {})
                    dR = np.asarray(d.get('dR', np.abs(R)), dtype=float)
                    if dR.shape != R.shape:
                        dR = np.abs(R)
                    sigma = np.where(np.isfinite(dR) & (dR > 0), dR,
                                     np.abs(R))
                    if f.error_floor > 0:
                        sigma = np.hypot(sigma, f.error_floor * R)
                with np.errstate(divide='ignore', invalid='ignore'):
                    out.append((name, f.Q[idx], (R - m) / sigma))
            self._resid = out
        return self._resid or []

    def _draw_residuals(self):
        """Each fitted channel's residuals, in its colour, with its chi^2/N
        in the legend; lines at 0 and +-1 sigma."""
        p = self.refl_plot
        for w in (self.logy, self.rq4, self.show_err):
            w.setEnabled(False)
        self.formula.setText(RESIDUAL_FORMULA)
        p.setLabel('left', 'residual (σ)')
        p.setLogMode(x=False, y=False)
        self.zero.setVisible(True)
        for y in (-1.0, 1.0):
            ln = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen(
                style.MUTED, width=0.8, style=QtCore.Qt.PenStyle.DashLine))
            p.addItem(ln)
            self._refl_items.append(ln)
        res = self.residuals()
        for name, Q, r in res:
            col = line_colours.get(name[1:]) or line_colours.get('asym')
            ok = np.isfinite(r)
            pts = pg.PlotDataItem(Q[ok], r[ok], pen=pg.mkPen(col, width=0.6),
                                  symbol='o', symbolSize=4,
                                  symbolPen=pg.mkPen(col), symbolBrush=col)
            p.addItem(pts)
            p.legend.addItem(pts, '%s  χ²/N %.3g' % (
                name, float(np.mean(r[ok]**2)) if ok.any() else np.nan))
            self._refl_items.append(pts)
        self.refl_note.setText('' if res else
                               'No residuals: this fit has no fitted points '
                               'here.')
        self.refl_note.setVisible(not res)
        p.enableAutoRange()

    def _add_data(self, name, col, Q, y, dy, log):
        """Markers with error bars, as in the Experimental tab."""
        ok = np.isfinite(y) & ((y > 0) if log else True)
        Q, y = Q[ok], y[ok]
        dy = np.where(np.isfinite(dy[ok]), np.abs(dy[ok]), 0.0)
        curve = pg.PlotDataItem(Q, y, pen=None, symbol='o', symbolSize=5,
                                symbolPen=pg.mkPen(col), symbolBrush=col)
        # ErrorBarItem has no log mode: give it log10 coordinates directly
        if log:
            yl = np.log10(y)
            err = pg.ErrorBarItem(
                x=Q, y=yl, top=np.log10(y + dy) - yl,
                bottom=yl - np.log10(np.maximum(y - dy, 1e-3 * y)),
                beam=0, pen=pg.mkPen(col, width=0.8))
        else:
            err = pg.ErrorBarItem(x=Q, y=y, height=2 * dy, beam=0,
                                  pen=pg.mkPen(col, width=0.8))
        err.setVisible(self.show_err.isChecked())
        self.refl_plot.addItem(err)
        self.refl_plot.addItem(curve)
        self.refl_plot.legend.addItem(curve, name)
        self._refl_items += [err, curve]
