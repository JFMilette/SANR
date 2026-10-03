"""
The Fit tool box of the Experimental tab: a "Fit — general" page (dataset,
worker processes, Revert: shared by both methods), a "Fit — differential
evolution" page (fit.de, with restarts), a "Fit — multistart" page
(fit.multistart: local fits from many starts, and the table of the minima
they reach, which DE restarts also fill) and a "Fit — Bayesian (DREAM)"
page (fit.dream, which can start around a minimum of that table), with
everything they do: building the fit problem from the chosen dataset
and the Simulation tab, running it in a background thread, following it in
its window and sending results back to the Simulation tab.

FitPanel is a mixin of tabs.experimental.ExperimentalTab, which builds the
tool box with _build_fit_box() and provides the datasets: `sets`,
`simulation`, `_set_index`, `channel_vectors`, `sim_vectors`,
`add_simulation` and `_update_live`.

While a fit or a sampling runs (busy()), the Simulation tab is locked
(SimulationTab.set_locked): its layers can be looked at, not changed.
"""

import os
import warnings
from html import escape as html_escape

import numpy as np
from PyQt6 import QtCore, QtWidgets

from model.polarisation import vectors_pair
from tabs.simulation import SciSpinBox, dspin
from fit.problem import FitProblem
from fit.de.de_window import DEThread, DEWindow
from fit.dream.dream_fit import (PosteriorProblem, jittered_start, load_run,
                                 parameter_names, prior_tension)
from fit.dream.dream_window import DreamThread, DreamWindow
from fit.multistart import run_multistart

DIALOG_OPTIONS = QtWidgets.QFileDialog.Option.DontUseNativeDialog
FIT_COSTS = [('chi2', 'χ² (weighted by dR)'), ('log', 'log R (per decade)')]
# DREAM start (dream_fit.sample x0): label, key; 'multistart': around the
# minimum selected in the multistart table
DREAM_STARTS = [('Latin hypercube', None), ('DE population', 'de'),
                ('current values', 'current'),
                ('multistart minimum', 'multistart')]
MS_METHODS = [('L-BFGS-B', 'lbfgs'), ('short DE', 'de')]
# height (px) a tool-box page keeps however short the window; taller pages
# scroll
FIT_PAGE_MIN = 160
# chi^2/N of an optimum above which DREAM's error scale is ticked
SIGMA_SUGGEST = 2.0


class MultistartThread(QtCore.QThread):
    """Runs fit.multistart.run_multistart off the GUI thread."""

    progress = QtCore.pyqtSignal(int, int)            # starts done, total
    done = QtCore.pyqtSignal(object)                  # the minima
    failed = QtCore.pyqtSignal(str)

    def __init__(self, problem, **options):
        super().__init__()
        self.problem, self.options = problem, options
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            m = run_multistart(self.problem, cancelled=lambda: self._stop,
                               callback=self.progress.emit, **self.options)
        except Exception as exc:                  # report, don't kill the app
            self.failed.emit('%s: %s' % (type(exc).__name__, exc))
            return
        self.done.emit(m)


class FitPanel:
    """The Fit tool box and its actions (see the module docstring)."""

    @staticmethod
    def _grid(pairs, cols=2):
        """(label, widget) pairs in a grid, `cols` pairs per row."""
        g = QtWidgets.QGridLayout()
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        for k, (lab, w) in enumerate(pairs):
            r, c = divmod(k, cols)
            g.addWidget(QtWidgets.QLabel(lab), r, 2 * c)
            g.addWidget(w, r, 2 * c + 1)
            g.setColumnStretch(2 * c + 1, 1)
        return g

    @staticmethod
    def _page():
        """A tool-box page and its form layout: a flat group box at the top
        of the page, as in the Simulation tab."""
        box = QtWidgets.QGroupBox()
        box.setFlat(True)
        f = QtWidgets.QFormLayout(box)
        f.setVerticalSpacing(3)
        page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(box)
        pl.addStretch(1)
        return page, f

    def _build_fit_box(self):
        """Fit tool box: the settings both methods share, then one page per
        method."""
        self.fit_tools = QtWidgets.QToolBox()
        self.fit_set = QtWidgets.QComboBox()
        self.fit_set.setToolTip('Every channel of this dataset is fitted')
        self.fit_cost = QtWidgets.QComboBox()
        self.fit_cost.addItems([c[1] for c in FIT_COSTS])
        self.fit_cost.setToolTip(
            'χ²: residuals divided by dR (points without dR use a relative '
            'residual).\nlog R: residuals of log10 R, every decade weighted '
            'equally.')
        self.fit_maxiter = QtWidgets.QSpinBox()
        self.fit_maxiter.setRange(1, 100000)
        self.fit_maxiter.setValue(200)
        self.fit_maxiter.setToolTip('Maximum number of generations')
        self.fit_pop = QtWidgets.QSpinBox()
        self.fit_pop.setRange(5, 200)
        self.fit_pop.setValue(15)
        self.fit_pop.setToolTip('Population = this × number of free '
                                'parameters')
        self.fit_tol = SciSpinBox(1.0)
        self.fit_tol.setValue(1e-6)
        self.fit_tol.setToolTip(
            'Stop when the spread of the population\'s cost falls below '
            'tol × its mean.\nA large tol (1e-3) can stop "successfully" '
            'far from the minimum; 1e-6 runs to Max iter in practice.')
        self.fit_restarts = QtWidgets.QSpinBox()
        self.fit_restarts.setRange(1, 20)
        self.fit_restarts.setValue(1)
        self.fit_restarts.setToolTip(
            'Run the fit this many times (new random population each time) '
            'and keep the best;\nthe distinct minima they end in are listed '
            'on the multistart page.')
        # mutation F: a random value in [min, max] each generation (dithering,
        # scipy's default 0.5 – 1); min = max is a constant F
        self.fit_mut_lo = dspin(0, 1.99, 2, 0.05)
        self.fit_mut_hi = dspin(0, 1.99, 2, 0.05)
        self.fit_mut_lo.setValue(0.5)
        self.fit_mut_hi.setValue(1.0)
        for w in (self.fit_mut_lo, self.fit_mut_hi):
            w.setToolTip(
                'Mutation F: trial = best + F × (difference of two members).\n'
                'A new F is drawn in [min, max] every generation (dithering); '
                'min = max keeps it constant.\nLarger F searches wider but '
                'converges more slowly.')
        self.fit_workers = QtWidgets.QSpinBox()
        self.fit_workers.setRange(1, os.cpu_count() or 1)
        self.fit_workers.setValue(max(1, (os.cpu_count() or 2) // 2))
        self.fit_workers.setToolTip(
            'Processes evaluating each generation in parallel (1 = in this '
            'process).\nStarting them takes about a second; beyond the '
            'number of performance cores it no longer helps.')
        self.fit_floor = dspin(0.0, 50.0, 2, 0.5, ' %')
        self.fit_floor.setToolTip(
            'Relative error floor f: every point\'s error becomes '
            'sqrt(dR² + (f R)²),\nfor errors the counting statistics leave '
            'out (e.g. the normalisation between angles).  0 = off.\n'
            'DREAM can fit it instead ("fit error floor").')
        self.fit_polish = QtWidgets.QCheckBox('polish (L-BFGS-B)')
        self.fit_polish.setChecked(True)
        self.fit_polish.setToolTip('Refine the best member with a local '
                                   'gradient minimiser at the end')

        self.btn_revert = QtWidgets.QPushButton('Revert')
        self.btn_revert.setToolTip(
            'Restore the parameters and fit bounds from before the last fit '
            'result, MAP or posterior bounds were sent to the Simulation tab')
        self.btn_revert.clicked.connect(self.revert_fit)
        self.btn_revert.setEnabled(False)

        # -- shared by both methods
        page, f = self._page()
        f.addRow('Dataset', self.fit_set)
        f.addRow('Workers', self.fit_workers)
        f.addRow('Error floor', self.fit_floor)
        f.addRow(QtWidgets.QLabel(
            '<i>Every channel of the dataset is fitted between Q min and '
            'Q max of the Simulation tab, varying the parameters ticked Fit '
            'there (scale and background included); its Q points set the '
            'model curves of the fit windows.</i>', wordWrap=True))
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_revert)
        f.addRow(row)
        self._add_page(page, 'Fit — general')

        # -- differential evolution
        page, f = self._page()
        f.addRow('Cost', self.fit_cost)
        f.addRow(self._grid([('Max iter', self.fit_maxiter),
                             ('Pop', self.fit_pop), ('Tol', self.fit_tol),
                             ('Restarts', self.fit_restarts)]))
        mrow = QtWidgets.QHBoxLayout()
        mrow.addWidget(self.fit_mut_lo, 1)
        mrow.addSpacing(6)
        mrow.addWidget(QtWidgets.QLabel('–'))
        mrow.addSpacing(6)
        mrow.addWidget(self.fit_mut_hi, 1)
        f.addRow('Mutation', mrow)
        f.addRow('', self.fit_polish)

        row = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton('Fit')
        self.btn_fit.setToolTip('Fit in the fit window; stop it there')
        self.btn_window = QtWidgets.QPushButton('Window')
        self.btn_window.setToolTip('Show the window of the last fit')
        for b in (self.btn_fit, self.btn_window):
            row.addWidget(b)
        f.addRow(row)
        self.btn_fit.clicked.connect(self.start_fit)
        self.btn_window.clicked.connect(self._show_fit_window)
        self.btn_window.setEnabled(False)

        self.fit_report = QtWidgets.QLabel()
        self.fit_report.setWordWrap(True)
        self.fit_report.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(self.fit_report)
        self._fit = None               # running DEThread
        self._fit_start = None         # (params, x) before the last fit
        self._fit_warnings = ''        # html, shown above the fit report
        self._grid_choice = {}         # dataset -> channel of its Licorne Q
        self.fit_window = None         # DEWindow, made by the first fit
        self._add_page(page, 'Fit — differential evolution')

        # -- multistart
        self._add_page(self._build_multistart_page(), 'Fit — multistart')

        # -- DREAM
        self._add_page(self._build_dream_page(), 'Fit — Bayesian (DREAM)')
        self.fit_tools.setCurrentIndex(1)
        # every page title stays on screen: the pages scroll when the
        # window is too short for them (FIT_PAGE_MIN at least is kept)
        tabs = sum(b.sizeHint().height() + 2 for b in
                   self.fit_tools.findChildren(QtWidgets.QAbstractButton)
                   if b.parent() is self.fit_tools)
        self.fit_tools.setMinimumHeight(tabs + FIT_PAGE_MIN)
        self.fit_tools.setEnabled(self.simulation is not None)
        return self.fit_tools

    def _add_page(self, page, title):
        """A tool-box page in a scroll area (vertical only)."""
        area = QtWidgets.QScrollArea()
        area.setWidget(page)
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # its own size must not grow the tool box (the page scrolls instead)
        area.setMinimumHeight(FIT_PAGE_MIN)
        area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Ignored)
        self.fit_tools.addItem(area, title)

    def _build_multistart_page(self):
        """The multistart page: local fits from the current values and
        from points spread over the bounds (fit.multistart), and the table
        of the distinct minima they (or DE restarts) end in."""
        page, f = self._page()
        self.ms_starts = QtWidgets.QSpinBox()
        self.ms_starts.setRange(1, 1000)
        self.ms_starts.setValue(24)
        self.ms_starts.setToolTip(
            'Starts besides the current values, spread evenly over the '
            'bounds (Sobol points)')
        self.ms_method = QtWidgets.QComboBox()
        for label, key in MS_METHODS:
            self.ms_method.addItem(label, key)
        self.ms_method.setToolTip(
            'L-BFGS-B: a gradient descent from each start (fast).\n'
            'short DE: 100 generations of differential evolution holding '
            'the start, then the polish (slower, looks wider).')
        f.addRow(self._grid([('Starts', self.ms_starts),
                             ('Local fit', self.ms_method)]))
        row = QtWidgets.QHBoxLayout()
        self.btn_ms_run = QtWidgets.QPushButton('Run')
        self.btn_ms_run.setToolTip(
            'Fit from every start with the DE page\'s cost and the general '
            'page\'s workers')
        self.btn_ms_stop = QtWidgets.QPushButton('Stop')
        self.btn_ms_stop.setEnabled(False)
        row.addWidget(self.btn_ms_run)
        row.addWidget(self.btn_ms_stop)
        f.addRow(row)
        self.btn_ms_run.clicked.connect(self.start_multistart)
        self.btn_ms_stop.clicked.connect(self.stop_multistart)
        self.ms_table = QtWidgets.QTableWidget(0, 0)
        self.ms_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.ms_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.ms_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ms_table.verticalHeader().setVisible(False)
        self.ms_table.setMinimumHeight(90)
        self.ms_table.setToolTip(
            'Distinct minima, best first: cost, how many starts ended there, '
            'whether the current values did, the parameters held by a '
            'bound, then the values')
        f.addRow(self.ms_table)
        row = QtWidgets.QHBoxLayout()
        self.btn_ms_apply = QtWidgets.QPushButton('Apply to model')
        self.btn_ms_apply.setToolTip('Send the selected minimum to the '
                                     'Simulation tab (Revert undoes it)')
        self.btn_ms_dream = QtWidgets.QPushButton('Start DREAM here')
        self.btn_ms_dream.setToolTip(
            'Sample with DREAM from jittered copies of the selected minimum '
            '(the DREAM page\'s settings)')
        for b in (self.btn_ms_apply, self.btn_ms_dream):
            b.setEnabled(False)
            row.addWidget(b)
        f.addRow(row)
        self.btn_ms_apply.clicked.connect(self._ms_apply)
        self.btn_ms_dream.clicked.connect(self._ms_dream)
        self.ms_report = QtWidgets.QLabel(wordWrap=True)
        f.addRow(self.ms_report)
        self._ms = None                 # running MultistartThread
        self._ms_minima = []            # rows of the table
        self._ms_params = None          # [(layer, attr)] they belong to
        return page

    def start_multistart(self):
        sim = self.simulation
        if sim is None or self.busy():
            return
        inputs = self._fit_inputs(self.ms_report)
        if inputs is None:
            return
        s, vec, data, qmin, qmax, grid = inputs
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                problem = FitProblem(
                    sim.stack, data,
                    FIT_COSTS[self.fit_cost.currentIndex()][0], qmin, qmax,
                    grid, error_floor=self.fit_floor.value() / 100)
        except ValueError as exc:
            self.ms_report.setText(self._error(exc))
            return
        self._ms_note = ''.join(self._error(w.message) + '<br>'
                                for w in caught)
        self._ms_problem = problem
        self._fit_start = (problem.params, [p[2] for p in problem.params])
        self._fit_layers = [l.name for l in sim.stack.layers]
        self._ms = MultistartThread(
            problem, n_starts=self.ms_starts.value(),
            method=self.ms_method.currentData(),
            workers=self.fit_workers.value(),
            de_options=dict(popsize=self.fit_pop.value(), init='sobol'))
        self._ms.progress.connect(self._ms_progress)
        self._ms.done.connect(
            lambda m: self._ms_show(self._ms_problem, m, 'multistart'))
        self._ms.failed.connect(lambda msg: self.ms_report.setText(
            self._error('Multistart failed: %s' % msg)))
        self._ms.finished.connect(self._ms_cleanup)
        sim.set_locked(True)
        for b in (self.btn_fit, self.btn_sample, self.btn_revert,
                  self.btn_ms_run, self.btn_ms_apply, self.btn_ms_dream):
            b.setEnabled(False)
        self.btn_ms_stop.setEnabled(True)
        self.ms_report.setText(self._ms_note + 'Fitting from %d starts…'
                               % (self.ms_starts.value() + 1))
        self._ms.start()

    def stop_multistart(self):
        if self._ms is not None:
            self._ms.stop()
            self.btn_ms_stop.setEnabled(False)

    def _ms_progress(self, done, total):
        self.ms_report.setText(self._ms_note + '%d / %d starts done…'
                               % (done, total))

    def _ms_cleanup(self):
        self._ms.deleteLater()
        self._ms = None
        self.simulation.set_locked(False)
        for b in (self.btn_fit, self.btn_sample, self.btn_ms_run):
            b.setEnabled(True)
        self.btn_ms_stop.setEnabled(False)

    def _ms_show(self, problem, minima, source):
        """Fill the table with `minima` (fit.multistart rows) of problem's
        parameters, from `source` (multistart, DE restarts)."""
        self._ms_minima = list(minima)
        self._ms_params = [p[:2] for p in problem.params]
        self._ms_layers = [l.name for l in problem.stack.layers]
        self._ms_names = parameter_names(problem.stack, problem.params)
        t = self.ms_table
        cost = 'χ²/N' if problem.cost == 'chi2' else 'cost'
        heads = [cost, 'hits', 'current', 'on a bound'] + self._ms_names
        t.clear()
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(minima))
        for r, m in enumerate(minima):
            bound = [n for n, b in zip(self._ms_names, m['at_bound']) if b]
            cells = ['%.5g' % m['chi2'], '%d' % m['hits'],
                     '✓' if m['from_current'] else '',
                     ', '.join(bound)] + ['%.5g' % v for v in m['x']]
            for c, text in enumerate(cells):
                t.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        t.resizeColumnsToContents()
        if minima:
            t.selectRow(0)
        for b in (self.btn_ms_apply, self.btn_ms_dream):
            b.setEnabled(bool(minima))
        self._ms_start_item(bool(minima))
        if minima:
            self._suggest_sigma(problem, minima[0]['x'])
        hits = sum(m['hits'] for m in minima)
        note = self._ms_note if source == 'multistart' else ''
        self.ms_report.setText(
            note + '%s: %d end(s) in %d distinct minim%s, best %s %.5g.'
            % (source, hits, len(minima), 'um' if len(minima) == 1 else 'a',
               cost, minima[0]['chi2']) if minima
            else '%s: stopped before any start ended.' % source)

    def _suggest_sigma(self, problem, x):
        """After an optimum with chi^2/N > 2: tick "fit error scale" on the
        DREAM page and say why (the user can untick it)."""
        if problem.cost != 'chi2':
            return
        c = float(np.mean(problem.residuals(x)**2))
        if not c > SIGMA_SUGGEST:
            return
        self.dream_sigma.setChecked(True)
        self.dream_note.setText(
            '<i>χ²/N = %.3g at the optimum: the error bars underestimate the '
            'misfit by ×%.2g; the error scale is fitted.</i>'
            % (c, np.sqrt(c)))

    def _ms_start_item(self, on):
        """Enable the DREAM start 'multistart minimum'."""
        item = self.dream_start.model().item(
            self.dream_start.findData('multistart'))
        item.setEnabled(on)

    def _ms_selected(self):
        rows = self.ms_table.selectionModel().selectedRows() \
            if self.ms_table.selectionModel() else []
        if not rows or rows[0].row() >= len(self._ms_minima):
            return None
        return self._ms_minima[rows[0].row()]

    def _ms_apply(self):
        """Send the selected minimum to the Simulation tab."""
        m = self._ms_selected()
        if m is None or self.busy():
            return
        if [l.name for l in self.simulation.stack.layers] != \
                self._ms_layers or [p[:2] for p in self.simulation.stack
                                    .free_parameters()] != self._ms_params:
            self.ms_report.setText(self._error(
                'The layers or varied parameters of the Simulation tab '
                'changed since these minima were found.'))
            return
        params = self.simulation.stack.free_parameters()
        self.simulation.set_parameters(params, m['x'])
        self.ms_report.setText('Minimum %.5g sent to the Simulation tab.'
                               % m['chi2'])
        self.btn_revert.setEnabled(True)

    def _ms_dream(self):
        """Start DREAM around the selected minimum."""
        self.dream_start.setCurrentIndex(
            self.dream_start.findData('multistart'))
        self.fit_tools.setCurrentIndex(self.fit_tools.count() - 1)
        self.start_dream()

    def _build_dream_page(self):
        """The DREAM page of the fit tool box."""
        page, f = self._page()
        self.dream_gen = QtWidgets.QSpinBox()
        self.dream_gen.setRange(100, 1_000_000)
        self.dream_gen.setSingleStep(1000)
        self.dream_gen.setValue(30000)
        self.dream_gen.setToolTip(
            'Maximum number of generations; the first 10 % adapt the sampler '
            'and are discarded')
        self.dream_chains = QtWidgets.QSpinBox()
        self.dream_chains.setRange(0, 200)
        self.dream_chains.setSpecialValueText('auto')
        self.dream_chains.setToolTip(
            'Number of chains; auto = 8 with the archive, otherwise '
            'max(7, d + 1).\nWith the archive any number from 2 works (R-hat '
            'compares chains, so 1 can never converge), but 3 chains can all '
            'settle in one of two minima and look converged; without it at '
            'least 7 (2δ + 1).')
        self.dream_seed = QtWidgets.QSpinBox()
        self.dream_seed.setRange(-1, 999_999)
        self.dream_seed.setValue(-1)
        self.dream_seed.setSpecialValueText('random')
        self.dream_seed.setToolTip('Random seed: the same seed and settings '
                                   'give the same chains')
        self.dream_archive = QtWidgets.QCheckBox('archive (ZS)')
        self.dream_archive.setChecked(True)
        self.dream_archive.setToolTip(
            'DREAM(ZS): jumps drawn from an archive of past states; works '
            'with 2 chains or more (8 by default) and keeps every mode a '
            'chain has visited.\n'
            'Unchecked: classic DREAM with the outlier-chain test, at least 7 '
            'chains.')
        self._chains_last = 0
        self.dream_chains.valueChanged.connect(self._check_chains)
        self.dream_archive.toggled.connect(self._check_chains)
        self.dream_sigma = QtWidgets.QCheckBox('fit error scale')
        self.dream_sigma.setToolTip(
            'Sample a common factor 10^s on every dR, s in [-1, 1] '
            '(log10_sigma_scale)')
        self.dream_floor = QtWidgets.QCheckBox('fit error floor')
        self.dream_floor.setToolTip(
            'Sample a relative error floor f, log10 f in [-4, -0.5]: errors '
            'sqrt((s dR)² + (f R)²) (log10_error_floor);\nthe general page\'s '
            'fixed floor is then not used')
        self.dream_note = QtWidgets.QLabel(wordWrap=True)
        self.dream_stop = QtWidgets.QCheckBox('stop when converged')
        self.dream_stop.setChecked(True)
        self.dream_stop.setToolTip(
            'Stop once R-hat < the R̂ target for every parameter, plus a fifth '
            'of the generations (at most 5000) more')
        self.dream_rhat = QtWidgets.QDoubleSpinBox()
        self.dream_rhat.setRange(1.001, 2.0)
        self.dream_rhat.setDecimals(3)
        self.dream_rhat.setSingleStep(0.01)
        self.dream_rhat.setValue(1.1)
        self.dream_rhat.setToolTip(
            'Converged once the Gelman-Rubin R-hat of every parameter is '
            'below this.\nUsed to stop the run (stop when converged), for '
            'the convergence flag and for the "Not converged" warning.\n'
            '1.2 is Vrugt\'s default; 1.1 is stricter, 1.05 stricter still '
            'and needs longer runs.')
        self.dream_snooker = QtWidgets.QDoubleSpinBox()
        self.dream_snooker.setRange(0.0, 0.5)
        self.dream_snooker.setSingleStep(0.05)
        self.dream_snooker.setDecimals(2)
        self.dream_snooker.setValue(0.1)
        self.dream_snooker.setToolTip(
            'Probability of a snooker update (archive only): a chain jumps '
            'along the line through an archive point,\nwhich helps it move '
            'between distant modes.  0 = none.  The posterior stays exact.')
        self.dream_temper = QtWidgets.QCheckBox('tempered burn-in')
        self.dream_temper.setChecked(True)
        self.dream_temper.setToolTip(
            'During the adaptation phase the chains sample prior × '
            'likelihood^β, β rising from β start to 1,\nso they can cross '
            'between minima before settling; those generations are '
            'discarded anyway.')
        # %g notation, like the background (1e-06 .. 1)
        self.dream_beta = SciSpinBox(1.0)
        self.dream_beta.setMinimum(1e-6)
        self.dream_beta.setValue(1e-3)
        self.dream_beta.setToolTip(
            'β at the first generation, between 1e-6 and 1: 1e-3 flattens '
            'the log-likelihood a thousandfold.\nThe arrows step 10 times per '
            'decade.')
        self.dream_temper.toggled.connect(self.dream_beta.setEnabled)
        self.dream_start = QtWidgets.QComboBox()
        for label, key in DREAM_STARTS:
            self.dream_start.addItem(label, key)
        self._ms_start_item(False)
        self.dream_start.setToolTip(
            'Where the chains start: spread over the bounds (Latin '
            'hypercube), the final population of a DE fit with the Fit '
            'page\'s settings,\nor jittered copies of the current values '
            '(e.g. a fitted optimum)\nor of the minimum selected in the '
            'multistart table.  The adaptation phase is discarded either '
            'way.')
        f.addRow(self._grid([('Generations', self.dream_gen),
                             ('Chains', self.dream_chains),
                             ('Seed', self.dream_seed),
                             ('R̂ target', self.dream_rhat),
                             ('Snooker', self.dream_snooker),
                             ('β start', self.dream_beta)]))
        f.addRow('Start', self.dream_start)
        opts = QtWidgets.QGridLayout()
        opts.setContentsMargins(0, 0, 0, 0)
        for k, w in enumerate((self.dream_archive, self.dream_sigma,
                               self.dream_stop, self.dream_temper,
                               self.dream_floor)):
            opts.addWidget(w, k // 2, k % 2)
        f.addRow(opts)
        f.addRow(self.dream_note)
        self.dream_sigma.toggled.connect(
            lambda on: None if on else self.dream_note.setText(''))
        self.dream_archive.toggled.connect(self.dream_snooker.setEnabled)
        f.addRow(QtWidgets.QLabel(
            '<i>χ² likelihood with the measured dR, whatever the DE '
            'cost.</i>', wordWrap=True))
        row = QtWidgets.QHBoxLayout()
        self.btn_sample = QtWidgets.QPushButton('Sample')
        self.btn_sample.setToolTip('Sample the posterior of the parameters '
                                   'ticked Fit in the Simulation tab')
        self.btn_dream_window = QtWidgets.QPushButton('Window')
        self.btn_dream_window.setToolTip('Show the window of the last run')
        self.btn_dream_window.setEnabled(False)
        self.btn_dream_load = QtWidgets.QPushButton('Load chains…')
        self.btn_dream_load.setToolTip(
            'Open a run saved with "Save chains…" in the DREAM window')
        row.addWidget(self.btn_sample)
        row.addWidget(self.btn_dream_window)
        row.addWidget(self.btn_dream_load)
        f.addRow(row)
        self.btn_sample.clicked.connect(self.start_dream)
        self.btn_dream_load.clicked.connect(self.load_dream)
        self.btn_dream_window.clicked.connect(self._show_dream_window)
        self.dream_report = QtWidgets.QLabel(wordWrap=True)
        self.dream_report.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(self.dream_report)
        self._dream = None              # running DreamThread
        self._dream_problem = None      # PosteriorProblem of the last run
        self.dream_window = None
        return page

    def _refresh_fit_sets(self):
        """Datasets that can be fitted: static ones holding channels."""
        c = self.fit_set
        keep = c.currentText()
        c.blockSignals(True)
        c.clear()
        for s in self.sets:
            if s['channels'] and not s.get('line'):
                c.addItem(s['name'])
        i = c.findText(keep)
        c.setCurrentIndex(max(i, 0))
        c.blockSignals(False)

    def busy(self):
        """A fit, a multistart or a DREAM run is going on."""
        return self._fit is not None or self._dream is not None or \
            self._ms is not None

    def _fit_inputs(self, report):
        """(dataset, {channel: [Pi, Pa]}, data list, qmin, qmax, grid) of
        the Fit box's dataset for fit.problem.FitProblem, after pushing the
        Simulation tab's settings into its stack; None, with the reason in
        `report`, if there is nothing to fit (or no Q grid was chosen).
        grid: the Q points the Licorne resolution scheme convolves on
        (_licorne_grid), None for the quadrature."""
        i = self._set_index(self.fit_set.currentText())
        if i is None:
            report.setText(self._error('Import data to fit first.'))
            return None
        sim = self.simulation
        sim.recompute()                 # push the tab's settings into the stack
        s = self.sets[i]
        # a channel marked not fitted (Licorne ExpFit = 0) is only shown
        fitted = {ch: d for ch, d in s['channels'].items()
                  if d.get('fit', True)}
        if not fitted:
            report.setText(self._error('Every channel of %s is marked not '
                                       'fitted.' % s['name']))
            return None
        vec = {ch: self.channel_vectors(s, ch) for ch in fitted}
        data = [dict(zip(('P0', 'P'), vectors_pair(*vec[ch])), name='R' + ch,
                     Q=d['Q'], R=d['R'], dR=d['dR'], norm=d.get('norm', 1.0))
                for ch, d in fitted.items()]
        grid = None
        res = sim.stack.resolution
        if res.get('enabled') and res.get('scheme') == 'licorne':
            grid = self._licorne_grid(s)
            if grid is None:
                report.setText(self._error('No Q grid chosen: nothing was '
                                           'started.'))
                return None
        # the Simulation tab's Q range selects the fitted points
        return s, vec, data, sim.qmin.value(), sim.qmax.value(), grid

    def _licorne_grid(self, s):
        """The Q points of dataset s that Licorne's convolution runs on, as
        Licorne computes every channel on one q.dat: the channels' common Q
        if they share it, else those of the channel chosen in a small
        dialog (remembered per dataset); None if the dialog is
        cancelled."""
        grids = {ch: np.asarray(d['Q'], dtype=float)
                 for ch, d in s['channels'].items()}
        grids = {ch: q[np.isfinite(q)] for ch, q in grids.items()}
        first = next(iter(grids.values()))
        if all(np.array_equal(np.sort(q), np.sort(first))
               for q in grids.values()):
            return first
        names = list(grids)
        items = ['R%s — %d points, %.4g – %.4g Å⁻¹'
                 % (ch, len(q), q.min(), q.max()) for ch, q in grids.items()]
        last = self._grid_choice.get(s['name'])
        item, ok = QtWidgets.QInputDialog.getItem(
            self, 'Q grid for the Licorne resolution',
            'The channels of %s are not on the same Q points.\n'
            'Licorne\'s resolution is computed on one Q grid: whose?\n'
            'The other channels take the model interpolated at their points.'
            % s['name'], items,
            names.index(last) if last in names else 0, False)
        if not ok:
            return None
        ch = names[items.index(item)]
        self._grid_choice[s['name']] = ch
        return grids[ch]

    def start_fit(self):
        sim = self.simulation
        if sim is None or self.busy():
            return
        mut = (self.fit_mut_lo.value(), self.fit_mut_hi.value())
        if mut[0] > mut[1]:
            self.fit_report.setText(self._error(
                'Mutation: min must not exceed max.'))
            return
        inputs = self._fit_inputs(self.fit_report)
        if inputs is None:
            return
        s, vec, data, qmin, qmax, grid = inputs
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                problem = FitProblem(
                    sim.stack, data,
                    FIT_COSTS[self.fit_cost.currentIndex()][0], qmin, qmax,
                    grid, error_floor=self.fit_floor.value() / 100)
        except ValueError as exc:
            self.fit_report.setText(self._error(exc))
            return
        # e.g. rho and phi of one layer both varied: kept above every report
        self._fit_warnings = ''.join(self._error(w.message) + '<br>'
                                     for w in caught)
        self._fit_problem = problem
        self._fit_start = (problem.params, [p[2] for p in problem.params])
        # layer names at the start: a sent result only fits the same stack
        self._fit_layers = [l.name for l in sim.stack.layers]
        # the fitted points (Q range, R > 0 for log R) for the fit window
        shown = {}
        for ch, d in s['channels'].items():
            if not d.get('fit', True):
                continue
            ok = np.isfinite(d['Q']) & np.isfinite(d['R']) & \
                (d['Q'] >= qmin) & (d['Q'] <= qmax)
            if problem.cost == 'log':
                ok &= d['R'] > 0
            if ok.any():
                shown[ch] = {c: d[c][ok] for c in ('Q', 'R', 'dR', 'dQ')}
        # imported here: tabs.experimental imports this module
        from tabs.experimental import model_grid
        if self.fit_window is None:
            self.fit_window = DEWindow(self)
            self.fit_window.stopRequested.connect(self.stop_fit)
            self.fit_window.sendRequested.connect(self._send_fit)
        # the model's other channels use the Simulation tab's pairs
        self.fit_window.start(
            problem, shown, dict(self.sim_vectors(), **vec),
            model_grid(sim.stack, np.linspace(qmin, qmax, sim.nq.value()),
                       s['channels'], grid), s['name'],
            ('χ²' if problem.cost == 'chi2' else 'cost (log R)')
            + (' + prior' if any(problem.priors) else ''),
            polish=self.fit_polish.isChecked())
        self.btn_window.setEnabled(True)
        # a live simulation dataset, computed from the same pairs as the
        # fit, shows a result sent from the fit window against the data
        if not any(t.get('live') for t in self.sets):
            self.add_simulation()
        for t in self.sets:
            if t.get('live'):
                t['pairs'] = dict(vec)
        self._update_live()
        self._fit = DEThread(problem, maxiter=self.fit_maxiter.value(),
                              popsize=self.fit_pop.value(),
                              tol=self.fit_tol.value(),
                              mutation=mut[0] if mut[0] == mut[1] else mut,
                              polish=self.fit_polish.isChecked(),
                              workers=self.fit_workers.value(),
                              restarts=self.fit_restarts.value(),
                              atol=0.0, init='sobol')
        self._fit.minima.connect(
            lambda m, p=problem: self._ms_show(p, m, 'DE restarts'))
        self._fit.progress.connect(self._fit_progress)
        self._fit.progress.connect(self.fit_window.add_generation)
        self._fit.finished_fit.connect(self._fit_done)
        self._fit.failed.connect(self._fit_failed)
        self._fit.finished.connect(self._fit_cleanup)
        # a result is only sent to the stack the fit started from
        sim.set_locked(True)
        for b in (self.btn_fit, self.btn_sample, self.btn_revert,
                  self.btn_ms_run):
            b.setEnabled(False)
        self.fit_report.setText(
            self._fit_warnings + 'Fitting %d parameter(s) to %d points of %s '
            '(Q %.4g – %.4g Å⁻¹)…' % (len(problem.params), problem.npoints,
                                     s['name'], qmin, qmax))
        self._fit.start()

    def min_chains(self):
        """Fewest chains the chosen variant can run: 2 with the archive
        (R-hat needs two chains), 2 delta + 1 = 7 without it."""
        return 2 if self.dream_archive.isChecked() else 7

    def _check_chains(self, *_):
        """Keep the Chains box at auto (0) or at least min_chains(): a value
        in between snaps up to the minimum when it was reached going up, back
        to auto when going down."""
        box, lo = self.dream_chains, self.min_chains()
        k = box.value()
        if 0 < k < lo:
            box.blockSignals(True)
            box.setValue(0 if self._chains_last >= lo and k < self._chains_last
                         else lo)
            box.blockSignals(False)
        self._chains_last = box.value()

    def start_dream(self):
        sim = self.simulation
        if sim is None or self.busy():
            return
        inputs = self._fit_inputs(self.dream_report)
        if inputs is None:
            return
        s, vec, data, qmin, qmax, grid = inputs
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                problem = PosteriorProblem(
                    FitProblem(sim.stack, data, 'chi2', qmin, qmax, grid,
                               error_floor=self.fit_floor.value() / 100),
                    sigma_free=self.dream_sigma.isChecked(),
                    floor_free=self.dream_floor.isChecked())
        except ValueError as exc:
            self.dream_report.setText(self._error(exc))
            return
        note = ''.join(self._error(w.message) + '<br>' for w in caught)
        params = problem.fit.params
        self._dream_problem = problem
        self._fit_start = (params, [p[2] for p in params])
        self._fit_layers = [l.name for l in sim.stack.layers]
        n_gen = self.dream_gen.value()
        options = dict(
            n_gen=n_gen, archive=self.dream_archive.isChecked(),
            n_chains=self.dream_chains.value() or None,
            seed=None if self.dream_seed.value() < 0
            else self.dream_seed.value(),
            n_workers=self.fit_workers.value(),
            stop_when_converged=self.dream_stop.isChecked(),
            rhat_target=self.dream_rhat.value(),
            min_gen_after_converged=min(5000, n_gen // 5),
            p_snooker=self.dream_snooker.value(),
            beta_start=self.dream_beta.value()
            if self.dream_temper.isChecked() else 1.0,
            x0=self.dream_start.currentData())
        if options['x0'] == 'de':
            options['de_options'] = dict(
                maxiter=self.fit_maxiter.value(),
                popsize=self.fit_pop.value(), tol=self.fit_tol.value())
        elif options['x0'] == 'multistart':
            row = self._ms_selected()
            if row is None or self._ms_params != [
                    p[:2] for p in problem.fit.params]:
                self.dream_report.setText(self._error(
                    'Select a minimum in the multistart table of these '
                    'parameters first.'))
                return
            n = options['n_chains'] or (8 if options['archive']
                                        else max(7, problem.ndim + 1))
            options['x0'] = jittered_start(problem, n, options['seed'],
                                           row['x'])
        self._make_dream_window()
        header = ('%d parameter(s), %d points of %s (Q %.4g – %.4g Å⁻¹)'
                  % (problem.ndim, problem.fit.npoints, s['name'], qmin, qmax))
        self.dream_window.start(problem, n_gen, note + html_escape(header),
                                [d['path'] for d in s['channels'].values()])
        self._launch_dream(problem, options,
                           note + 'Sampling ' + html_escape(header) + '…')

    def _launch_dream(self, problem, options, report):
        """Run fit.dream.dream_fit.sample(problem, **options) in a DreamThread
        followed by the DREAM window."""
        self.dream_window.may_stop_early = bool(
            options.get('stop_when_converged'))
        self._dream = DreamThread(problem, **options)
        self._dream.progress.connect(self.dream_window.progress)
        self._dream.snapshot.connect(self.dream_window.snapshot)
        self._dream.done.connect(self._dream_done)
        self._dream.failed.connect(self._dream_failed)
        self._dream.finished.connect(self._dream_cleanup)
        if self.simulation is not None:
            self.simulation.set_locked(True)
        for b in (self.btn_fit, self.btn_sample, self.btn_revert,
                  self.btn_dream_load, self.btn_ms_run):
            b.setEnabled(False)
        self.dream_report.setText(report)
        self._dream.start()

    def continue_dream(self):
        """Go on with the stopped run shown in the DREAM window up to the
        generations it was started with, as if it had never stopped."""
        prev = self.dream_window.result
        if self.busy() or prev is None or prev.state is None or \
                not prev.state.get('cancelled'):
            return
        n_gen = int(prev.state['n_gen'])
        n_more = n_gen - int(prev.generations[-1])
        if n_more <= 0:
            return
        options = dict(n_gen=n_gen, resume=prev,
                       n_workers=self.fit_workers.value(),
                       stop_when_converged=self.dream_stop.isChecked(),
                       rhat_target=self.dream_rhat.value(),
                       min_gen_after_converged=min(5000, n_more // 5))
        self.dream_window.continue_start(n_gen)
        self._launch_dream(self._dream_problem, options,
                           'Continuing to generation %d…' % n_gen)

    def _make_dream_window(self):
        if self.dream_window is None:
            self.dream_window = DreamWindow(self)
            self.dream_window.stopRequested.connect(self.stop_dream)
            self.dream_window.sendRequested.connect(self._send_dream)
            self.dream_window.boundsRequested.connect(self._bounds_dream)
            self.dream_window.continueRequested.connect(self.continue_dream)
        self.btn_dream_window.setEnabled(True)

    def load_dream(self):
        """Show a run saved with "Save chains..." in the DREAM window.  Its
        MAP and bounds can be sent to the Simulation tab when the layers
        there have the names they had in the run."""
        if self.busy():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load chains', '', 'NumPy archive (*.npz);;All files (*)',
            options=DIALOG_OPTIONS)
        if not path:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')     # said when it ran
                result, problem, meta = load_run(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Load chains', 'Could not load %s:\n%s' % (path, exc))
            return
        self._dream_problem = problem
        self._fit_layers = [l.name for l in problem.fit.stack.layers]
        sim = self.simulation
        self._fit_start = None
        self.btn_revert.setEnabled(False)
        if sim is not None and \
                [l.name for l in sim.stack.layers] == self._fit_layers:
            sim.recompute()
            # what Revert restores: the Simulation tab's values and bounds now
            params = [(i, a, getattr(sim.stack.owner(i), a),
                       sim.stack.owner(i).fit_entry(a)['min'],
                       sim.stack.owner(i).fit_entry(a)['max'])
                      for i, a, *_ in problem.fit.params]
            self._fit_start = (params, [p[2] for p in params])
        self._make_dream_window()
        self.dream_window.show_loaded(
            result, problem, 'Loaded %s — %d parameter(s), %d points'
            % (html_escape(os.path.basename(path)), problem.ndim,
               problem.fit.npoints), meta.get('sources', []))
        self.dream_report.setText('Loaded %s.' % html_escape(
            os.path.basename(path)))

    def stop_dream(self):
        if self._dream is not None:
            self._dream.stop()
            self.dream_window.btn_stop.setEnabled(False)

    def _show_dream_window(self):
        if self.dream_window is not None:
            self.dream_window.show()
            self.dream_window.raise_()
            self.dream_window.activateWindow()

    def _dream_done(self, result):
        self.dream_window.finish(result)
        self.dream_report.setText(
            'Done after %d generation(s); see the DREAM window.'
            % result.generations[-1])

    def _dream_failed(self, msg):
        self.dream_window.fail(msg)
        self.dream_report.setText('Stopped.' if msg == 'stopped' else
                                  self._error('Sampling failed: %s' % msg))

    def _dream_cleanup(self):
        self._dream.deleteLater()
        self._dream = None
        self.simulation.set_locked(False)
        for b in (self.btn_fit, self.btn_sample, self.btn_dream_load,
                  self.btn_ms_run):
            b.setEnabled(True)

    def _send_dream(self, x):
        """Write a DREAM state (the MAP) into the Simulation tab."""
        if self._dream is not None:
            return
        if [l.name for l in self.simulation.stack.layers] != self._fit_layers:
            QtWidgets.QMessageBox.warning(
                self.dream_window, 'Load MAP into model',
                'The layers of the Simulation tab changed since the run '
                'started;\nthese parameters no longer match them.')
            return
        params = self._dream_problem.fit.params
        self.simulation.set_parameters(params, x[:len(params)])
        self.dream_report.setText('MAP sent to the Simulation tab.')
        self.btn_revert.setEnabled(True)

    def _bounds_dream(self, lo, hi):
        """Write posterior-based fit bounds into the Simulation tab."""
        if self._dream is not None:
            return
        if [l.name for l in self.simulation.stack.layers] != self._fit_layers:
            QtWidgets.QMessageBox.warning(
                self.dream_window, 'Apply ±σ to bounds',
                'The layers of the Simulation tab changed since the run '
                'started;\nthese parameters no longer match them.')
            return
        self.simulation.set_bounds(self._dream_problem.fit.params, lo, hi)
        self.dream_report.setText('Posterior bounds sent to the Simulation '
                                  'tab.')
        self.btn_revert.setEnabled(True)

    def stop_fit(self):
        if self._fit is not None:
            self._fit.stop()
            self.fit_window.btn_stop.setEnabled(False)

    def _show_fit_window(self):
        if self.fit_window is not None:
            self.fit_window.show()
            self.fit_window.raise_()
            self.fit_window.activateWindow()

    def _send_fit(self, x, label):
        """Write an entry of the fit window into the Simulation tab."""
        if self._fit is not None:
            return
        if [l.name for l in self.simulation.stack.layers] != self._fit_layers:
            QtWidgets.QMessageBox.warning(
                self.fit_window, 'Send to Simulation',
                'The layers of the Simulation tab changed since the fit '
                'started;\nthese parameters no longer match them.')
            return
        self.simulation.set_parameters(self._fit_problem.params, x)
        self.fit_report.setText(self._fit_header(
            'Sent <b>%s</b> to the Simulation tab' % html_escape(
                'generation ' + label if label.isdigit() else label)))
        self.btn_revert.setEnabled(True)

    def revert_fit(self):
        if self._fit_start is not None and not self.busy():
            params, x = self._fit_start
            self.simulation.set_parameters(params, x)
            # the bounds too (Apply ±σ to bounds changes them)
            self.simulation.set_bounds(params, [p[3] for p in params],
                                       [p[4] for p in params])
            self.fit_report.setText('Parameters and bounds restored to '
                                    'their values before the fit.')
            self.btn_revert.setEnabled(False)

    def _fit_progress(self, x, cost, gen):
        self.fit_report.setText(self._fit_header(
            'Generation %d — cost %.5g' % (gen, cost)))

    def _fit_done(self, x, cost, gen, msg):
        self.fit_window.finish(x, cost, gen, msg)
        self._suggest_sigma(self._fit_problem, x)
        tension = ''.join(self._error(t) + '<br>' for t in prior_tension(
            self._fit_problem, x, parameter_names(
                self._fit_problem.stack, self._fit_problem.params)))
        self.fit_report.setText(self._fit_header(
            '<b>%s</b> after %d generation(s) — cost %.5g<br><i>%s</i><br>%s'
            'Send an entry from the fit window to use it.'
            % ('Stopped' if msg == 'stopped' else 'Done', gen, cost,
               html_escape(msg), tension)))

    def _fit_failed(self, msg):
        self.fit_window.fail(msg)
        self.fit_report.setText(self._error('Fit failed: %s' % msg))

    def _fit_cleanup(self):
        self._fit.deleteLater()
        self._fit = None
        self.simulation.set_locked(False)
        for b in (self.btn_fit, self.btn_sample, self.btn_ms_run):
            b.setEnabled(True)

    def _fit_header(self, header):
        """The fit's warnings above a status line; the parameters are in
        the fit window."""
        return self._fit_warnings + header

    @staticmethod
    def _error(msg):
        return '<span style="color:#f87171">%s</span>' % html_escape(str(msg))

    def shutdown(self):
        """Stop a running fit; call before the application quits."""
        if self.fit_window is not None:     # it would keep the app alive
            self.fit_window.hide()
        if self.dream_window is not None:
            self.dream_window.hide()
        for t in (self._fit, self._dream, self._ms):
            if t is not None:
                t.stop()
                t.wait()
