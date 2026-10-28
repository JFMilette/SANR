"""
The Fit box of the Experimental tab: the settings every method shares
(dataset, cost, target, worker processes, error floor, Revert) and a Method
box, below which only the chosen method's options and buttons are shown
(_show_method): differential evolution (fit.de, with restarts), scipy's
basin hopping, dual annealing, SHGO and DIRECT (fit.de.global_fit, followed
in the same fit window as DE), local
L-BFGS-B (fit.de.de_fit.run_lbfgs: L-BFGS-B alone from the current values,
followed in the same fit window as DE), least squares
(fit.de.de_fit.run_lsq: trust-region least squares on the residuals from the
current values, likewise), multistart (fit.multistart: local fits from many
starts, followed in the fit window, which lists the minima they reach as
does the multistart table, both updated as each start ends; DE restarts
fill the table too) and Bayesian DREAM (fit.dream, which can start around a
minimum of that table), with everything they do: building the fit problem
from the chosen dataset and the Simulation tab, running it in a background
thread, following it in its window (a closable tab of the main window,
page_tabs) and sending results back to the Simulation tab.

JOINT FIT -- the Datasets to fit panel beside the fit box (the Fit tab's
right side, fit.datasets_panel, `datasets_panel`) picks the datasets
fitted, as it is ticked: one alone, or several fitted together with one
model, each parameter shared, own to a dataset or not fitted in it, at a
value typed for that dataset or the Simulation tab's (fit.joint).  The
Dataset box (fit_set, never shown) holds the choice: a dataset, or a
"Joint: …" entry for the selection (kept in the session); the fit windows
then show one tab per dataset.  A result sent to the Simulation tab
from a joint fit holds the values one dataset sees: that of the DE
window's tab shown (a multistart's minima too), of the dataset chosen in
the DREAM window, the first one's from the multistart table.

FitPanel is a mixin of tabs.experimental.ExperimentalTab, which builds the
fit box with _build_fit_box() and provides the datasets: `sets`,
`simulation`, `_set_index`, `channel_vectors`, `sim_vectors`,
`add_simulation` and `_update_live`.

While a fit or a sampling runs (busy()), the Simulation tab is locked
(SimulationTab.set_locked): its layers can be looked at, not changed.
"""

import copy
import os
import warnings
from html import escape as html_escape

import numpy as np
from PyQt6 import QtCore, QtWidgets

from model.polarisation import vectors_pair
from tabs.simulation import PARAM_DISPLAY, SciSpinBox, dspin
from fit.datasets_panel import DatasetsPanel
from fit.problem import FitProblem, parameter_names
from fit.de.de_fit import STRATEGIES
from fit.de.de_window import DEThread, DEWindow
from fit.de import global_fit
from fit.dream.dream_fit import (PosteriorProblem, jittered_start, load_run,
                                 prior_tension)
from fit.dream.dream_window import SESSIONS_DIR, DreamThread, DreamWindow
from fit.joint import JointProblem, part_index, part_x, parts
from fit.multistart import merge, run_multistart

DIALOG_OPTIONS = QtWidgets.QFileDialog.Option.DontUseNativeDialog
FIT_COSTS = [('chi2', 'χ² (weighted by dR)'), ('log', 'log R (per decade)')]
# what the cost compares (fit.problem TARGET); the label's end names it in
# the fit window
FIT_TARGETS = [('channels', 'every channel', ''),
               ('sum', '(R+ + R−)/2', ' of (R+ + R−)/2'),
               ('asym', 'spin asymmetry SA', ' of SA')]
# DREAM start (dream_fit.sample x0): label, key; 'multistart': around the
# minimum selected in the multistart table
DREAM_STARTS = [('Latin hypercube', None), ('DE population', 'de'),
                ('current values', 'current'),
                ('multistart minimum', 'multistart')]
# the Method box: key, label
FIT_METHODS = [('de', 'Differential evolution'),
               ('basinhopping', 'Basin hopping'),
               ('dual_annealing', 'Dual annealing'), ('shgo', 'SHGO'),
               ('direct', 'DIRECT'), ('lbfgs', 'Local (L-BFGS-B)'),
               ('lsq', 'Least squares'), ('multistart', 'Multistart'),
               ('dream', 'Bayesian (DREAM)')]
SHGO_SAMPLING = [('sobol', 'Sobol'), ('halton', 'Halton'),
                 ('simplicial', 'simplicial')]
MS_METHODS = [('L-BFGS-B', 'lbfgs'), ('least squares', 'lsq'),
              ('short DE', 'de')]
# height (px) the fit box keeps however short the window; taller content
# scrolls
FIT_PAGE_MIN = 160
# chi^2/N of an optimum above which DREAM's error scale is ticked
SIGMA_SUGGEST = 2.0
JOINT = 'joint'           # data of the Dataset box's joint-fit entry
# tab names (_tab_name): the method, a number, the dataset (cut to
# TAB_NAME_MAX characters) or how many datasets a joint fit has
TAB_METHODS = {'de': 'DE', 'lbfgs': 'L-BFGS-B', 'lsq': 'LSQ',
               'basinhopping': 'BH', 'dual_annealing': 'DA', 'shgo': 'SHGO',
               'direct': 'DIRECT',
               'multistart': 'Multistart',
               'dream': 'DREAM'}
TAB_NAME_MAX = 18


class MultistartThread(QtCore.QThread):
    """Runs fit.multistart.run_multistart off the GUI thread."""

    progress = QtCore.pyqtSignal(int, int)            # starts done, total
    status = QtCore.pyqtSignal(object)        # fit.multistart PROGRESS info
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
                               callback=self.progress.emit,
                               progress=self.status.emit, **self.options)
        except Exception as exc:                  # report, don't kill the app
            self.failed.emit('%s: %s' % (type(exc).__name__, exc))
            return
        self.done.emit(m)


class FitPanel:
    """The Fit box and its actions (see the module docstring)."""

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
        """A page of the fit box and its form layout: a flat group box, as
        in the Simulation tab."""
        box = QtWidgets.QGroupBox()
        box.setFlat(True)
        f = QtWidgets.QFormLayout(box)
        f.setVerticalSpacing(3)
        page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(box)
        return page, f

    def _build_fit_box(self):
        """Fit box: the settings every method shares and the method
        chosen, then that method's options."""
        self.fit_set = QtWidgets.QComboBox()
        self.fit_set.setToolTip('Every channel of this dataset is fitted '
                                '(of every dataset of a joint fit: tick '
                                'them on the right)')
        self._joint = None             # {'sets', 'modes'} of the joint fit
        # the datasets fitted together and their parameters' modes, live;
        # the Fit tab shows it beside the fit box
        self.datasets_panel = DatasetsPanel()
        self.datasets_panel.changed.connect(self._datasets_chosen)
        self._datasets_key = None      # what the panel was last loaded with
        self._choosing = False         # the panel itself is being applied
        self.fit_set.currentIndexChanged.connect(self._sync_datasets_panel)
        if self.simulation is not None:     # the parameters ticked Fit
            self.simulation.reflectanceChanged.connect(
                self._sync_datasets_panel)
        self.fit_cost = QtWidgets.QComboBox()
        self.fit_cost.addItems([c[1] for c in FIT_COSTS])
        self.fit_cost.setToolTip(
            'χ²: residuals divided by dR (points without dR use a relative '
            'residual).\nlog R: residuals of log10 R, every decade weighted '
            'equally.')
        self.fit_target = QtWidgets.QComboBox()
        for key, label, _ in FIT_TARGETS:
            self.fit_target.addItem(label, key)
        self.fit_target.setToolTip(
            'What every method but DREAM compares with the data (DREAM '
            'always samples every channel):\n'
            'every channel: each channel (R+, R−, or all four) on its own.\n'
            '(R+ + R−)/2: their mean point by point, where magnetism only '
            'enters at second order:\n  the NSLD, thicknesses and '
            'roughnesses, e.g. shared by datasets whose magnetism differs.\n'
            'spin asymmetry (R+ − R−)/(R+ + R−): first order in the MSLD, '
            'and the scale cancels:\n  the MSLD with the structure fixed.\n'
            'Both use R+ and R− (else R++ and R−−) at their common Q, errors '
            'propagated;\n  the spin-flip channels R+− and R−+ are still '
            'fitted on their own.')
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
            'in the multistart table.')
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
        self.fit_strategy = QtWidgets.QComboBox()
        for key, label in STRATEGIES:
            self.fit_strategy.addItem(label, key)
        self.fit_strategy.setToolTip(
            'How each trial member is built (fit.de.de_fit STRATEGY), from '
            'the greediest to the one that explores most:\n'
            'best1bin: best member + F × (difference of two members).  Fast, '
            'but with the current values near a minimum\n  the whole '
            'population collapses onto that minimum within a few '
            'generations.\n'
            'current-to-best1bin: each member moves part way towards the '
            'best.\n'
            'rand-to-best1bin: a random member, part way towards the best.\n'
            'rand1bin: random member + F × (difference of two members); no '
            'pull towards the best:\n  slower to converge, much wider '
            'search (to check whether a fit is stuck).\n'
            'Also used by the multistart\'s short DE and DREAM\'s DE '
            'start.')
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

        # -- shared by every method
        general, f = self._page()
        # the datasets fitted are ticked in the Datasets to fit panel; this
        # box, never shown, holds that choice for the fit and the session
        f.addRow('Dataset', self.fit_set)
        f.setRowVisible(self.fit_set, False)
        f.addRow('Cost', self.fit_cost)
        f.addRow('Fit to', self.fit_target)
        f.addRow('Workers', self.fit_workers)
        f.addRow('Error floor', self.fit_floor)
        f.addRow(QtWidgets.QLabel(
            '<i>Every channel of the datasets ticked on the right is fitted '
            'between Q min and Q max of the Simulation tab, varying the '
            'parameters ticked Fit there (scale and background included); '
            'their Q points set the model curves of the fit windows.  Fit '
            'to (R+ + R−)/2 or SA combines R+ and R− point by point.</i>',
            wordWrap=True))
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_revert)
        f.addRow(row)
        self.fit_method = QtWidgets.QComboBox()
        for key, label in FIT_METHODS:
            self.fit_method.addItem(label, key)
        self.fit_method.setToolTip(
            'The fitting method; its options and buttons are shown below.\n'
            'Differential evolution: a global search over the bounds.\n'
            'Basin hopping, dual annealing, SHGO, DIRECT: scipy\'s other '
            'global searches.\n'
            'Local (L-BFGS-B): a gradient descent from the current values.\n'
            'Least squares: trust-region least squares on the residuals '
            'from the current values.\n'
            'Multistart: local fits from the current values and from points '
            'spread over the bounds.\n'
            'Bayesian (DREAM): samples the posterior.')
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        f.addRow(line)
        f.addRow('Method', self.fit_method)

        # -- differential evolution
        page, f = self._page()
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
        f.addRow('Strategy', self.fit_strategy)
        f.addRow('', self.fit_polish)

        row = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton('Fit')
        self.btn_fit.setToolTip('Fit in the fit window; stop it there')
        self.btn_window = QtWidgets.QPushButton('Show')
        self.btn_window.setToolTip('Show the last fit')
        for b in (self.btn_fit, self.btn_window):
            row.addWidget(b)
        f.addRow(row)
        # not connected directly: clicked's `checked` would be the method
        self.btn_fit.clicked.connect(lambda: self.start_fit())
        self.btn_window.clicked.connect(self._show_fit_window)
        self.btn_window.setEnabled(False)

        self.fit_report = QtWidgets.QLabel()
        self.fit_report.setWordWrap(True)
        self.fit_report.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(self.fit_report)
        self._fit = None               # running DEThread
        self._de_state = None          # where the last DE ended (Continue)
        self._fit_ok = True            # the last DE did not fail
        self._fit_start = None         # (params, x) before the last fit
        self._fit_warnings = ''        # html, shown above the fit report
        self._grid_choice = {}         # dataset -> channel of its Licorne Q
        self.fit_window = None         # DEWindow, made by the first fit
        # the main window's QTabWidget: the fit and DREAM windows open there
        # as closable tabs (close_page); None: as windows of their own
        self.page_tabs = None
        self._n_tabs = 0               # fit and DREAM tabs opened so far
        self._fit_out = self.fit_report  # the report of the page that fitted

        # -- the methods' options, the chosen one's shown below the general
        # settings
        self._method_pages = {
            'de': page, 'lbfgs': self._build_local_page(),
            'lsq': self._build_lsq_page(),
            **self._build_global_pages(),
            'multistart': self._build_multistart_page(),
            'dream': self._build_dream_page()}
        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(general)
        for key, _ in FIT_METHODS:
            v.addWidget(self._method_pages[key])
        v.addStretch(1)
        self.fit_method.currentIndexChanged.connect(self._show_method)
        self._show_method()
        area = QtWidgets.QScrollArea()
        area.setWidget(content)
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # its own size must not grow the tab (it scrolls instead)
        area.setMinimumHeight(FIT_PAGE_MIN)
        area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Ignored)
        self.fit_tools = QtWidgets.QGroupBox('Fit')
        QtWidgets.QVBoxLayout(self.fit_tools).addWidget(area)
        self.fit_tools.setEnabled(self.simulation is not None)
        return self.fit_tools

    def _show_method(self, *_):
        """Only the chosen method's options under the general settings."""
        key = self.fit_method.currentData()
        for k, page in self._method_pages.items():
            page.setVisible(k == key)

    def _build_local_page(self):
        """The local page: L-BFGS-B alone from the current values, in the
        fit window (fit.de.de_fit.run_lbfgs)."""
        page, f = self._page()
        self.lb_maxiter = QtWidgets.QSpinBox()
        self.lb_maxiter.setRange(1, 100000)
        self.lb_maxiter.setValue(1000)
        self.lb_maxiter.setToolTip('Maximum number of L-BFGS-B iterations')
        f.addRow(self._grid([('Max iter', self.lb_maxiter)]))
        f.addRow(QtWidgets.QLabel(
            '<i>A gradient descent from the current values of the '
            'Simulation tab, within the fit bounds: fast, but it stops in '
            'the minimum nearest the start (refine a fit, or one done by '
            'hand).</i>', wordWrap=True))
        row = QtWidgets.QHBoxLayout()
        self.btn_lbfgs = QtWidgets.QPushButton('Fit')
        self.btn_lbfgs.setToolTip('Fit in the fit window; stop it there')
        self.btn_lbfgs_window = QtWidgets.QPushButton('Show')
        self.btn_lbfgs_window.setToolTip('Show the last fit')
        self.btn_lbfgs_window.setEnabled(False)
        row.addWidget(self.btn_lbfgs)
        row.addWidget(self.btn_lbfgs_window)
        f.addRow(row)
        self.btn_lbfgs.clicked.connect(self.start_lbfgs)
        self.btn_lbfgs_window.clicked.connect(self._show_fit_window)
        self.lb_report = QtWidgets.QLabel(wordWrap=True)
        self.lb_report.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(self.lb_report)
        return page

    def _build_lsq_page(self):
        """The least-squares page: fit.de.de_fit.run_lsq from the current
        values, in the fit window."""
        page, f = self._page()
        self.lsq_maxiter = QtWidgets.QSpinBox()
        self.lsq_maxiter.setRange(1, 100000)
        self.lsq_maxiter.setValue(1000)
        self.lsq_maxiter.setToolTip('Maximum number of least-squares '
                                    'iterations')
        f.addRow(self._grid([('Max iter', self.lsq_maxiter)]))
        f.addRow(QtWidgets.QLabel(
            '<i>Trust-region least squares (Gauss-Newton steps) on the '
            'residuals, from the current values of the Simulation tab, '
            'within the fit bounds: often fewer steps than L-BFGS-B, but it '
            'also stops in the minimum nearest the start.</i>',
            wordWrap=True))
        row = QtWidgets.QHBoxLayout()
        self.btn_lsq = QtWidgets.QPushButton('Fit')
        self.btn_lsq.setToolTip('Fit in the fit window; stop it there')
        self.btn_lsq_window = QtWidgets.QPushButton('Show')
        self.btn_lsq_window.setToolTip('Show the last fit')
        self.btn_lsq_window.setEnabled(False)
        row.addWidget(self.btn_lsq)
        row.addWidget(self.btn_lsq_window)
        f.addRow(row)
        self.btn_lsq.clicked.connect(self.start_lsq)
        self.btn_lsq_window.clicked.connect(self._show_fit_window)
        self.lsq_report = QtWidgets.QLabel(wordWrap=True)
        self.lsq_report.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(self.lsq_report)
        return page

    def _build_global_pages(self):
        """The pages of fit.de.global_fit's methods: {method: page}.  Each
        method's options widgets, by run_global keyword, are in
        _global_widgets[method] (read by _global_options)."""
        def spin(lo, hi, value, tip, step=1, special=None):
            w = QtWidgets.QSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(value)
            if special:
                w.setSpecialValueText(special)
            w.setToolTip(tip)
            return w

        def check(text, on, tip):
            w = QtWidgets.QCheckBox(text)
            w.setChecked(on)
            w.setToolTip(tip)
            return w

        T = SciSpinBox(1e6)
        T.setValue(1.0)
        T.setToolTip(
            'Metropolis temperature, in cost units (χ²/N for the χ² cost): '
            'a hop to a minimum this much worse is accepted with '
            'probability 1/e.')
        step = dspin(0.01, 1.0, 2, 0.05)
        step.setValue(0.1)
        step.setToolTip(
            'Largest random step of each parameter, as a fraction of its '
            'bounds (scipy adapts it to accept half the hops)')
        temp = dspin(0.02, 5e4, 2, 100)
        temp.setValue(5230.0)
        temp.setToolTip('Initial temperature of the annealing (scipy\'s '
                        'default 5230): higher visits farther')
        sampling = QtWidgets.QComboBox()
        for key, label in SHGO_SAMPLING:
            sampling.addItem(label, key)
        sampling.setToolTip(
            'How the sampling points are spread: Sobol or Halton sequences, '
            'or simplicial (the vertices of a triangulation of the box, '
            'slow beyond a few parameters)')
        # method: (options as (label, run_global keyword, widget); boxes
        # (keyword, check box); note)
        specs = {
            'basinhopping': (
                [('Hops', 'niter', spin(1, 100000, 100, 'Number of hops')),
                 ('T', 'T', T), ('Step', 'stepsize', step)], [],
                'From the current values: a random hop, then L-BFGS-B, '
                'the new minimum kept by a Metropolis test; finds the '
                'minima near the start one by one.'),
            'dual_annealing': (
                [('Max iter', 'maxiter',
                  spin(1, 100000, 1000, 'Number of annealing iterations')),
                 ('Initial T', 'initial_temp', temp)],
                [('local_search', check(
                    'local search (L-BFGS-B)', True,
                    'Refine each new minimum the annealing finds with '
                    'L-BFGS-B'))],
                'Generalised simulated annealing from the current values '
                'over the whole bounds.  The annealing runs one point at a '
                'time; the local search\'s gradients use the workers.'),
            'shgo': (
                [('Points', 'n', spin(4, 100000, 128,
                                      'Sampling points per iteration')),
                 ('Iterations', 'iters',
                  spin(1, 100, 1, 'Sampling iterations')),
                 ('Sampling', 'sampling_method', sampling)], [],
                'Simplicial homology global optimisation: samples the '
                'bounds, then runs L-BFGS-B from each candidate minimum.  '
                'Ends with the global and every local minimum found, in '
                'the fit window and the multistart table.  The sampling '
                'uses the workers.'),
            'direct': (
                [('Max evals', 'maxfun', spin(
                    0, 10_000_000, 0, 'Most cost evaluations; auto = '
                    '1000 per fitted parameter', step=1000, special='auto')),
                 ('Max iter', 'maxiter', spin(1, 100000, 1000,
                                              'Most iterations'))],
                [('locally_biased', check(
                    'locally biased (DIRECT-L)', True,
                    'DIRECT-L: refine the best boxes first (faster); '
                    'unticked, the original DIRECT (more global)'))],
                'DIviding RECTangles: deterministic, from the centre of '
                'the bounds, ignores the current values.  One process '
                '(Workers is not used).')}
        self._global_widgets, self._global_reports = {}, {}
        self._global_buttons, self._global_windows = [], []
        pages = {}
        for method, (rows, boxes, note) in specs.items():
            page, f = self._page()
            f.addRow(self._grid([(label, w) for label, _, w in rows]))
            for _, w in boxes:
                f.addRow('', w)
            f.addRow(QtWidgets.QLabel('<i>%s</i>' % note, wordWrap=True))
            row = QtWidgets.QHBoxLayout()
            run = QtWidgets.QPushButton('Fit')
            run.setToolTip('Fit in the fit window; stop it there')
            show = QtWidgets.QPushButton('Show')
            show.setToolTip('Show the last fit')
            show.setEnabled(False)
            row.addWidget(run)
            row.addWidget(show)
            f.addRow(row)
            run.clicked.connect(lambda _, m=method: self.start_fit(m))
            show.clicked.connect(self._show_fit_window)
            report = QtWidgets.QLabel(wordWrap=True)
            report.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            f.addRow(report)
            self._global_widgets[method] = dict(
                [(key, w) for _, key, w in rows] + boxes)
            self._global_reports[method] = report
            self._global_buttons.append(run)
            self._global_windows.append(show)
            pages[method] = page
        return pages

    def _global_options(self, method):
        """run_global's options of `method` from its page."""
        out = {}
        for key, w in self._global_widgets[method].items():
            if isinstance(w, QtWidgets.QCheckBox):
                out[key] = w.isChecked()
            elif isinstance(w, QtWidgets.QComboBox):
                out[key] = w.currentData()
            else:
                out[key] = w.value()
        if method == 'direct' and not out['maxfun']:
            out['maxfun'] = None                    # auto
        return out

    def _set_global_options(self, saved):
        """Inverse of _global_options for every method: saved {method:
        options} from a session (missing ones keep their value)."""
        for method, opts in (saved or {}).items():
            for key, v in opts.items():
                w = self._global_widgets.get(method, {}).get(key)
                if w is None:
                    continue
                if isinstance(w, QtWidgets.QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, QtWidgets.QComboBox):
                    w.setCurrentIndex(max(0, w.findData(v)))
                else:
                    w.setValue(0 if v is None else v)

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
            'least squares: trust-region least squares on the residuals '
            '(Gauss-Newton like, often fewer steps than L-BFGS-B).\n'
            'short DE: 100 generations of differential evolution holding '
            'the start, then the polish if it is ticked in the DE options\n'
            '(slower, looks wider).')
        f.addRow(self._grid([('Starts', self.ms_starts),
                             ('Local fit', self.ms_method)]))
        row = QtWidgets.QHBoxLayout()
        self.btn_ms_run = QtWidgets.QPushButton('Run')
        self.btn_ms_run.setToolTip(
            'Fit from every start with the general cost and '
            'workers')
        self.btn_ms_stop = QtWidgets.QPushButton('Stop')
        self.btn_ms_stop.setEnabled(False)
        self.btn_ms_window = QtWidgets.QPushButton('Show')
        self.btn_ms_window.setToolTip('Show the last fit')
        self.btn_ms_window.setEnabled(False)
        row.addWidget(self.btn_ms_run)
        row.addWidget(self.btn_ms_stop)
        row.addWidget(self.btn_ms_window)
        f.addRow(row)
        self.btn_ms_run.clicked.connect(self.start_multistart)
        self.btn_ms_stop.clicked.connect(self.stop_multistart)
        self.btn_ms_window.clicked.connect(self._show_fit_window)
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
            '(the DREAM options)')
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
        self._ms_shown = None           # the problem they belong to
        self._ms_names = []             # and its parameter names
        return page

    def start_multistart(self):
        sim = self.simulation
        if sim is None or self.busy():
            return
        built = self._problem(self.ms_report,
                              FIT_COSTS[self.fit_cost.currentIndex()][0])
        if built is None:
            return
        problem, inputs, qmin, qmax, self._ms_note = built
        self._ms_problem = problem
        self._set_fit_start(problem)
        # followed in the fit window, which sends a minimum as a fit's entry
        self._fit_problem, self._fit_out = problem, self.ms_report
        self._fit_warnings, self._de_state = self._ms_note, None
        self._open_fit_window(problem, inputs, qmin, qmax, 'multistart')
        self._ms = MultistartThread(
            problem, n_starts=self.ms_starts.value(),
            method=self.ms_method.currentData(),
            workers=self.fit_workers.value(),
            de_options=dict(popsize=self.fit_pop.value(), init='sobol',
                            strategy=self.fit_strategy.currentData(),
                            polish=self.fit_polish.isChecked()))
        self._ms.progress.connect(self._ms_progress)
        self._ms.progress.connect(self.fit_window.set_starts)
        self._ms.status.connect(self.fit_window.set_start_status)
        self._ms_ends = []              # (start, u, cost) of the ended starts
        self._ms.status.connect(self._ms_status)
        self.fit_window.set_starts(0, self.ms_starts.value() + 1)
        self._ms.done.connect(
            lambda m: self._ms_show(self._ms_problem, m, 'multistart'))
        self._ms.done.connect(lambda m, t=self._ms: self.fit_window.show_minima(
            m, stopped=t._stop))
        self._ms.failed.connect(lambda msg: self.ms_report.setText(
            self._error('Multistart failed: %s' % msg)))
        self._ms.failed.connect(self.fit_window.fail)
        self._ms.finished.connect(self._ms_cleanup)
        sim.set_locked(True)
        for b in (self.btn_fit, self.btn_lbfgs, self.btn_lsq,
                  *self._global_buttons, self.btn_sample,
                  self.btn_revert, self.btn_ms_run, self.btn_ms_apply,
                  self.btn_ms_dream):
            b.setEnabled(False)
        self.btn_ms_stop.setEnabled(True)
        self.ms_report.setText(self._ms_note + 'Fitting from %d starts…'
                               % (self.ms_starts.value() + 1))
        self._ms.start()

    def stop_multistart(self):
        if self._ms is not None:
            self._ms.stop()
            self.btn_ms_stop.setEnabled(False)
            self.fit_window.btn_stop.setEnabled(False)

    def _ms_progress(self, done, total):
        self.ms_report.setText(self._ms_note + '%d / %d starts done…'
                               % (done, total))

    def _ms_status(self, info):
        """A start of the running multistart ended: the table as the ended
        starts give it so far (its buttons wait for the end)."""
        if info['phase'] != 'done' or info.get('u') is None:
            return
        self._ms_ends.append((info['start'], np.asarray(info['u']),
                              info['best']))
        self._ms_fill(self._ms_problem,
                      merge(self._ms_problem, self._ms_ends))

    def _ms_cleanup(self):
        self._ms.deleteLater()
        self._ms = None
        self.simulation.set_locked(False)
        for b in (self.btn_fit, self.btn_lbfgs, self.btn_lsq,
                  *self._global_buttons, self.btn_sample,
                  self.btn_ms_run):
            b.setEnabled(True)
        self.btn_ms_stop.setEnabled(False)

    def _ms_show(self, problem, minima, source):
        """Fill the table with `minima` (fit.multistart rows) of problem's
        parameters, from `source` (multistart, DE restarts)."""
        self._ms_fill(problem, minima)
        for b in (self.btn_ms_apply, self.btn_ms_dream):
            b.setEnabled(bool(minima))
        self._ms_start_item(bool(minima))
        if minima:
            self._suggest_sigma(problem, minima[0]['x'])
        hits = sum(m['hits'] for m in minima)
        cost = 'χ²/N' if problem.cost == 'chi2' else 'cost'
        note = self._ms_note if source == 'multistart' else ''
        self.ms_report.setText(
            note + '%s: %d end(s) in %d distinct minim%s, best %s %.5g.'
            % (source, hits, len(minima), 'um' if len(minima) == 1 else 'a',
               cost, minima[0]['chi2']) if minima
            else '%s: stopped before any start ended.' % source)

    def _ms_fill(self, problem, minima):
        """The table's rows: `minima` of problem's parameters, the row
        selected before kept selected (by its parameters), else the
        best."""
        old = self._ms_selected()
        keep = None if old is None or old is self._ms_minima[0] \
            else old['x']
        self._ms_minima = list(minima)
        self._ms_shown = problem        # the problem of the table's minima
        self._ms_names = list(problem.names)
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
            t.selectRow(next((r for r, m in enumerate(minima)
                              if keep is not None
                              and np.array_equal(m['x'], keep)), 0))

    def _suggest_sigma(self, problem, x):
        """After an optimum with chi^2/N > 2: tick "fit error scale" on the
        DREAM options and say why (the user can untick it)."""
        if problem.cost != 'chi2' or problem.target != 'channels':
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
        """Send the selected minimum to the Simulation tab (the first
        dataset's values of a joint fit)."""
        m = self._ms_selected()
        if m is None or self.busy():
            return
        base = parts(self._ms_shown)[0]
        if [l.name for l in self.simulation.stack.layers] != \
                [l.name for l in base.stack.layers] or \
                [p[:2] for p in self.simulation.stack.free_parameters()] != \
                [p[:2] for p in base.params]:
            self.ms_report.setText(self._error(
                'The layers or varied parameters of the Simulation tab '
                'changed since these minima were found.'))
            return
        self.simulation.set_parameters(base.params,
                                       part_x(self._ms_shown, m['x']))
        joint = getattr(self._ms_shown, 'part_names', None)
        self.ms_report.setText(
            'Minimum %.5g sent to the Simulation tab%s.' % (
                m['chi2'], ' (the values of %s)' % html_escape(joint[0])
                if joint else ''))
        self.btn_revert.setEnabled(True)

    def _ms_dream(self):
        """Start DREAM around the selected minimum."""
        self.dream_start.setCurrentIndex(
            self.dream_start.findData('multistart'))
        self.fit_method.setCurrentIndex(self.fit_method.findData('dream'))
        self.start_dream()

    def _build_dream_page(self):
        """The DREAM options of the fit box."""
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
            'sqrt((s dR)² + (f R)²) (log10_error_floor);\nthe general '
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
        self.btn_dream_window = QtWidgets.QPushButton('Show')
        self.btn_dream_window.setToolTip('Show the last run')
        self.btn_dream_window.setEnabled(False)
        self.btn_dream_load = QtWidgets.QPushButton('Load chains…')
        self.btn_dream_load.setToolTip(
            'Open a run saved with "Save chains…" in the DREAM tab')
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

    def _fittable(self):
        """Names of the datasets that can be fitted: static ones holding
        channels."""
        return [s['name'] for s in self.sets
                if s['channels'] and not s.get('line')]

    def _refresh_fit_sets(self):
        """The Dataset box: every dataset that can be fitted, then the
        joint fit's entry (its datasets that are still there, if two or
        more)."""
        c = self.fit_set
        keep, joint = c.currentText(), c.currentData() == JOINT
        names = self._fittable()
        if self._joint:
            sets = [n for n in self._joint['sets'] if n in names]
            self._joint = dict(sets=sets, **{
                key: {n: m for n, m in (self._joint.get(key) or {}).items()
                      if n in sets} for key in ('modes', 'values')}) \
                if len(sets) > 1 else None
        c.blockSignals(True)
        c.clear()
        c.addItems(names)
        if self._joint:
            c.addItem('Joint: ' + ' + '.join(self._joint['sets']), JOINT)
        i = c.findData(JOINT) if joint and self._joint else c.findText(keep)
        c.setCurrentIndex(max(i, 0))
        c.blockSignals(False)
        self._sync_datasets_panel()

    def _fit_sets(self):
        """The datasets fitted: the Dataset box's one, or every dataset of
        the joint fit when its entry is chosen."""
        names = self._joint['sets'] if self._joint and \
            self.fit_set.currentData() == JOINT else \
            [self.fit_set.currentText()]
        found = [self._set_index(n) for n in names]
        return [self.sets[i] for i in found if i is not None]

    def _rename_fit_set(self, old, new):
        """Dataset `old` is now called `new`: the joint fit follows."""
        if self._joint and old in self._joint['sets']:
            self._joint['sets'] = [new if n == old else n
                                   for n in self._joint['sets']]
            for key in ('modes', 'values'):
                d = self._joint.get(key) or {}
                if old in d:
                    d[new] = d.pop(old)
        if old in self._grid_choice:
            self._grid_choice[new] = self._grid_choice.pop(old)

    def _sync_datasets_panel(self, *_):
        """The Datasets to fit panel from the Dataset box (its dataset, or
        the joint fit's), the datasets that can be fitted and the
        parameters ticked Fit in the Simulation tab; reloaded only when one
        of them changed, so a box being typed in is kept."""
        if self._choosing:
            return
        key, names, chosen, params, info = self._datasets_state()
        if key == self._datasets_key:
            return
        self._datasets_key = key
        joint = self._joint if len(chosen) > 1 else None
        self.datasets_panel.load(
            names, chosen, params,
            None if joint is None else joint.get('modes') or {},
            None if joint is None else joint.get('values') or {}, info)

    def _datasets_state(self):
        """(key, names, chosen, params, info) of _sync_datasets_panel."""
        names = self._fittable()
        chosen = [s['name'] for s in self._fit_sets()]
        params, info = [], {}
        if self.simulation is not None:
            st = self.simulation.stack
            free = st.free_parameters()
            params = parameter_names(st, free)
            # the value boxes of 'Not fitted': the Simulation tab's units
            info = {n: PARAM_DISPLAY[a][1:] + (v,)
                    for n, (_, a, v, *_) in zip(params, free)}
        key = (names, chosen, params, sorted(info.items()))
        return key, names, chosen, params, info

    def _datasets_chosen(self):
        """A tick, mode or value of the Datasets to fit panel: two datasets
        or more are the Dataset box's joint entry, one is that dataset."""
        sel = self.datasets_panel.selection()
        self._choosing = True
        try:
            if len(sel['sets']) > 1:
                self._joint = sel
                self._refresh_fit_sets()
                i = self.fit_set.findData(JOINT)
            else:
                i = self.fit_set.findText(sel['sets'][0])
            if i != self.fit_set.currentIndex():
                self.fit_set.setCurrentIndex(i)
        finally:
            self._choosing = False
        # the panel already shows this state: not reloaded
        self._datasets_key = self._datasets_state()[0]

    def busy(self):
        """A fit, a multistart or a DREAM run is going on."""
        return self._fit is not None or self._dream is not None or \
            self._ms is not None

    def _fit_inputs(self, report):
        """(dataset, {channel: [Pi, Pa]}, data list, qmin, qmax, grid) of
        the Fit box's (first) dataset for fit.problem.FitProblem, after
        pushing the Simulation tab's settings into its stack; None, with
        the reason in `report`, if there is nothing to fit (or no Q grid
        was chosen).  grid: the Q points the Licorne resolution scheme
        convolves on (_licorne_grid), None for the quadrature."""
        sets = self._fit_sets()
        if not sets:
            report.setText(self._error('Import data to fit first.'))
            return None
        sim = self.simulation
        sim.recompute()                 # push the tab's settings into the stack
        inputs = self._set_inputs(sets[0], report)
        if inputs is None:
            return None
        vec, data, grid = inputs
        # the Simulation tab's Q range selects the fitted points
        return sets[0], vec, data, sim.qmin.value(), sim.qmax.value(), grid

    def _set_inputs(self, s, report):
        """({channel: [Pi, Pa]}, data list, grid) of dataset s (see
        _fit_inputs), the Simulation tab's settings already in its
        stack."""
        sim = self.simulation
        # a channel marked not fitted (Licorne ExpFit = 0) is only shown
        fitted = {ch: d for ch, d in s['channels'].items()
                  if d.get('fit', True)}
        if not fitted:
            report.setText(self._error('Every channel of %s is marked not '
                                       'fitted.' % s['name']))
            return None
        vec = {ch: self.channel_vectors(s, ch) for ch in fitted}
        data = [dict(zip(('P0', 'P'), vectors_pair(*vec[ch])), name='R' + ch,
                     Q=d['Q'], R=d['R'], dR=d['dR'], dQ=d.get('dQ'),
                     norm=d.get('norm', 1.0))
                for ch, d in fitted.items()]
        grid = None
        res = sim.stack.resolution
        if res.get('enabled') and res.get('scheme') == 'licorne':
            grid = self._licorne_grid(s)
            if grid is None:
                report.setText(self._error('No Q grid chosen: nothing was '
                                           'started.'))
                return None
        return vec, data, grid

    def _problem(self, report, cost, target=None):
        """(problem, [(dataset, {channel: [Pi, Pa]}, data, grid)], qmin,
        qmax, warnings as html) of the fitted datasets with `cost` and
        `target` (fit.problem TARGET; None: the Fit to box's): a
        fit.problem.FitProblem of one dataset, a fit.joint.JointProblem of
        the joint fit's; None, with the reason in `report`, if nothing can
        be fitted."""
        sets = self._fit_sets()
        if not sets:
            report.setText(self._error('Import data to fit first.'))
            return None
        sim = self.simulation
        sim.recompute()                 # push the tab's settings into the stack
        inputs = []
        for s in sets:
            got = self._set_inputs(s, report)
            if got is None:
                return None
            inputs.append((s,) + got)
        qmin, qmax = sim.qmin.value(), sim.qmax.value()
        joint = self.fit_set.currentData() == JOINT
        if target is None:
            target = self.fit_target.currentData()
        fits, said = [], {}             # warning -> datasets that gave it
        for s, vec, data, grid in inputs:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fits.append((s['name'], FitProblem(
                        self._set_stack(s), data, cost, qmin, qmax, grid,
                        error_floor=self.fit_floor.value() / 100,
                        target=target)))
            except ValueError as exc:
                report.setText(self._error(
                    '%s: %s' % (s['name'], exc) if joint else exc))
                return None
            for w in caught:
                said.setdefault(str(w.message), []).append(s['name'])
        try:
            problem = JointProblem(fits, self._joint['modes'],
                                   values=self._joint.get('values')) \
                if joint else fits[0][1]
        except ValueError as exc:
            report.setText(self._error(exc))
            return None
        # e.g. rho and phi of one layer both varied, said once
        note = ''.join(self._error(
            msg if not joint or len(who) == len(fits)
            else '%s: %s' % (', '.join(who), msg)) + '<br>'
            for msg, who in said.items())
        return problem, inputs, qmin, qmax, note

    def _set_stack(self, s):
        """The stack dataset s is fitted with: the Simulation tab's, but in
        the 'data' resolution mode with the dQ table of s itself
        (tabs.resolution.dq_table, the overlap rule of the table chosen
        there), so that its points without a dQ and its curves in the fit
        window take its own dQ, not the chosen dataset's; the Simulation
        tab's stack itself if s is that dataset or has no dQ."""
        # imported here: tabs.resolution imports tabs.simulation
        from tabs.resolution import dq_table
        st = self.simulation.stack
        res = st.resolution
        if not (res.get('enabled') and res.get('mode') == 'data') or \
                res.get('data_set') == s['name']:
            return st
        chans = dict(self.dq_sets()).get(s['name'])
        if not chans:
            return st
        try:
            q, dq = dq_table(chans, res.get('data_overlap', 'mean'))
        except ValueError:
            return st
        st = copy.deepcopy(st)
        st.resolution.update(data_q=q.tolist(), data_dq=dq.tolist(),
                             data_set=s['name'])
        return st

    def _set_fit_start(self, problem):
        """What Revert restores: the stack's varied parameters as the fit
        of `problem` starts, and the layer names a result is sent to."""
        base = parts(problem)[0]
        self._fit_start = (base.params, [p[2] for p in base.params])
        self._fit_layers = [l.name for l in base.stack.layers]

    def _fitted_points(self, s, qmin, qmax, cost):
        """The points of dataset s that are fitted (Q range, R > 0 for log
        R), for the fit window."""
        shown = {}
        for ch, d in s['channels'].items():
            if not d.get('fit', True):
                continue
            ok = np.isfinite(d['Q']) & np.isfinite(d['R']) & \
                (d['Q'] >= qmin) & (d['Q'] <= qmax)
            if cost == 'log':
                ok &= d['R'] > 0
            if ok.any():
                shown[ch] = {c: d[c][ok] for c in ('Q', 'R', 'dR', 'dQ')}
        return shown

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

    def start_lbfgs(self):
        """A local fit: L-BFGS-B alone from the current values."""
        self.start_fit('lbfgs')

    def start_lsq(self):
        """A local fit: least squares from the current values."""
        self.start_fit('lsq')

    def start_fit(self, method='de'):
        """Fit with DE (method 'de'), L-BFGS-B alone ('lbfgs'), least
        squares ('lsq') or a method of fit.de.global_fit, followed in the
        fit window; its page's report says how it goes."""
        sim = self.simulation
        if sim is None or self.busy():
            return
        self._fit_out = out = dict(
            self._global_reports, lbfgs=self.lb_report,
            lsq=self.lsq_report).get(method, self.fit_report)
        mut = (self.fit_mut_lo.value(), self.fit_mut_hi.value())
        if method == 'de' and mut[0] > mut[1]:
            out.setText(self._error('Mutation: min must not exceed max.'))
            return
        built = self._problem(out, FIT_COSTS[self.fit_cost.currentIndex()][0])
        if built is None:
            return
        # e.g. rho and phi of one layer both varied: kept above every report
        problem, inputs, qmin, qmax, self._fit_warnings = built
        self._fit_problem = problem
        # a sent result only fits the stack with these layer names
        self._set_fit_start(problem)
        title = self._open_fit_window(problem, inputs, qmin, qmax, method)
        self._de_state = None
        self._fit_what = (title, qmin, qmax)
        self._launch_de(problem, mut, method=method)

    def _open_fit_window(self, problem, inputs, qmin, qmax, method):
        """Show problem's datasets (_problem's inputs) in the fit window
        for a run of `method` (DEWindow.start); returns its title."""
        sim = self.simulation
        # imported here: tabs.experimental imports this module
        from tabs.experimental import model_grid
        # as tabs, each fit gets a new one and the last one stays
        if self.fit_window is None or self.page_tabs is not None:
            self._retire(self.fit_window)
            w = self.fit_window = DEWindow(self)
            w.stopRequested.connect(self.stop_fit)
            w.continueRequested.connect(self.continue_fit)
            w.sendRequested.connect(
                lambda x, label, part, w=w: self._send_fit(x, label, part, w))
        self.fit_window.tab_name = self._tab_name(
            method, [s['name'] for s, *_ in inputs])
        # the model's other channels use the Simulation tab's pairs
        Q = np.linspace(qmin, qmax, sim.nq.value())
        views = [dict(name=s['name'],
                      data=self._fitted_points(s, qmin, qmax, problem.cost),
                      vectors=dict(self.sim_vectors(), **vec),
                      grids=model_grid(sim.stack, Q, s['channels'], grid))
                 for s, vec, data, grid in inputs]
        title = ' + '.join(s['name'] for s, *_ in inputs)
        self.fit_window.start(
            problem, views, title,
            ('χ²' if problem.cost == 'chi2' else 'cost (log R)')
            + dict((k, end) for k, _, end in FIT_TARGETS)[problem.target]
            + (' + prior' if any(problem.priors) else ''),
            polish=self.fit_polish.isChecked(), method=method)
        self._show_fit_window()
        for b in (self.btn_window, self.btn_lbfgs_window,
                  self.btn_lsq_window, *self._global_windows,
                  self.btn_ms_window):
            b.setEnabled(True)
        # a live simulation dataset, computed from the same pairs as the
        # fit, shows a result sent from the fit window against the data
        if not any(t.get('live') for t in self.sets):
            self.add_simulation()
        for t in self.sets:
            if t.get('live'):
                t['pairs'] = dict(inputs[0][1])
        self._update_live()
        return title

    def continue_fit(self):
        """Go on with the last DE fit from where it ended (fit.de.de_fit
        CONTINUE): its problem and population, the DE page's settings."""
        if self.simulation is None or self.busy() or not self._de_state:
            return
        mut = (self.fit_mut_lo.value(), self.fit_mut_hi.value())
        if mut[0] > mut[1]:
            self.fit_report.setText(self._error(
                'Mutation: min must not exceed max.'))
            self.fit_window.btn_continue.setEnabled(True)
            return
        self.fit_window.resume()
        self._fit_out = self.fit_report
        self._launch_de(self._fit_problem, mut, resume=self._de_state)

    def _launch_de(self, problem, mut, resume=None, method='de'):
        """Run DE (or L-BFGS-B alone, method 'lbfgs', least squares, 'lsq',
        or a method of fit.de.global_fit) on problem in a DEThread, wired
        to the fit window."""
        sim = self.simulation
        title, qmin, qmax = self._fit_what
        self._fit_ok = True
        if method in ('lbfgs', 'lsq'):
            self._fit = DEThread(problem, method=method,
                                 maxiter=(self.lb_maxiter if method == 'lbfgs'
                                          else self.lsq_maxiter).value(),
                                 workers=self.fit_workers.value())
        elif method in global_fit.METHODS:
            self._fit = DEThread(problem, method=method,
                                 workers=self.fit_workers.value(),
                                 **self._global_options(method))
        else:
            self._fit = DEThread(problem, maxiter=self.fit_maxiter.value(),
                                 popsize=self.fit_pop.value(),
                                 tol=self.fit_tol.value(),
                                 mutation=mut[0] if mut[0] == mut[1] else mut,
                                 polish=self.fit_polish.isChecked(),
                                 workers=self.fit_workers.value(),
                                 restarts=self.fit_restarts.value(),
                                 atol=0.0, init='sobol', resume=resume,
                                 strategy=self.fit_strategy.currentData())
        self._fit.minima.connect(
            lambda m, p=problem: self._ms_show(
                p, m, global_fit.LABELS.get(method, 'DE restarts')))
        if method in global_fit.METHODS:
            self._fit.minima.connect(self.fit_window.set_minima)
        self._fit.progress.connect(self._fit_progress)
        self._fit.progress.connect(self.fit_window.add_generation)
        self._fit.status.connect(self._fit_status)
        self._fit.status.connect(self.fit_window.set_progress)
        self._fit.finished_fit.connect(self._fit_done)
        self._fit.failed.connect(self._fit_failed)
        self._fit.finished.connect(self._fit_cleanup)
        # a result is only sent to the stack the fit started from
        sim.set_locked(True)
        for b in (self.btn_fit, self.btn_lbfgs, self.btn_lsq,
                  *self._global_buttons, self.btn_sample,
                  self.btn_revert, self.btn_ms_run):
            b.setEnabled(False)
        self._fit_out.setText(
            self._fit_warnings + '%s %d parameter(s) to %d points of %s '
            '(Q %.4g – %.4g Å⁻¹)…' % (
                'Fitting' if resume is None else 'Continuing: fitting',
                len(problem.params), problem.npoints, html_escape(title),
                qmin, qmax))
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
        built = self._problem(self.dream_report, 'chi2', 'channels')
        if built is None:
            return
        fit, inputs, qmin, qmax, note = built
        try:
            problem = PosteriorProblem(
                fit, sigma_free=self.dream_sigma.isChecked(),
                floor_free=self.dream_floor.isChecked())
        except ValueError as exc:
            self.dream_report.setText(self._error(exc))
            return
        self._dream_problem = problem
        self._set_fit_start(fit)
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
                popsize=self.fit_pop.value(), tol=self.fit_tol.value(),
                strategy=self.fit_strategy.currentData())
        elif options['x0'] == 'multistart':
            row = self._ms_selected()
            if row is None or self._ms_names != problem.fit.names:
                self.dream_report.setText(self._error(
                    'Select a minimum in the multistart table of these '
                    'parameters first.'))
                return
            n = options['n_chains'] or (8 if options['archive']
                                        else max(7, problem.ndim + 1))
            options['x0'] = jittered_start(problem, n, options['seed'],
                                           row['x'])
        self._make_dream_window([s['name'] for s, *_ in inputs])
        header = ('%d parameter(s), %d points of %s (Q %.4g – %.4g Å⁻¹)'
                  % (problem.ndim, problem.fit.npoints,
                     ' + '.join(s['name'] for s, *_ in inputs), qmin, qmax))
        self.dream_window.start(problem, n_gen, note + html_escape(header),
                                [d['path'] for s, *_ in inputs
                                 for d in s['channels'].values()],
                                [s['name'] for s, *_ in inputs])
        self._show_dream_window()
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
        for b in (self.btn_fit, self.btn_lbfgs, self.btn_lsq,
                  *self._global_buttons, self.btn_sample,
                  self.btn_revert, self.btn_dream_load, self.btn_ms_run):
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

    def _make_dream_window(self, names):
        """The DREAM window of a run of datasets `names` (or of a loaded
        file): as tabs, a new one each time, the last one staying."""
        if self.dream_window is None or self.page_tabs is not None:
            self._retire(self.dream_window)
            w = self.dream_window = DreamWindow(self)
            w.stopRequested.connect(self.stop_dream)
            w.continueRequested.connect(self.continue_dream)
            w.sendRequested.connect(
                lambda x, part, label, w=w: self._send_dream(x, part, w,
                                                             label))
            w.boundsRequested.connect(
                lambda lo, hi, part, w=w: self._bounds_dream(lo, hi, part, w))
        self.dream_window.tab_name = self._tab_name('dream', names)
        self.dream_window.setWindowTitle('Bayesian sampling (DREAM) — '
                                         + ' + '.join(names))
        self.btn_dream_window.setEnabled(True)

    def _tab_name(self, method, names):
        """Short name of a new fit or DREAM tab: e.g. 'DE 3 · film_5K' or
        'DREAM 4 · 3 datasets' (the full title is its tool tip)."""
        self._n_tabs += 1
        what = names[0] if len(names) == 1 else '%d datasets' % len(names)
        if len(what) > TAB_NAME_MAX:
            what = what[:TAB_NAME_MAX - 1].rstrip() + '…'
        return '%s %d · %s' % (TAB_METHODS[method], self._n_tabs, what)

    def _retire(self, w):
        """The fit or DREAM window w is no longer the newest: its entries
        can still be sent to the Simulation tab, but only the newest is
        stopped or continued."""
        if w is None:
            return
        w.stopRequested.disconnect()
        w.continueRequested.disconnect()
        w.btn_continue.setEnabled(False)
        w.btn_continue.setToolTip('Only the newest fit or run can be '
                                  'continued')

    def load_dream(self):
        """Show a run saved with "Save chains..." in the DREAM window.  Its
        MAP and bounds can be sent to the Simulation tab when the layers
        there have the names they had in the run."""
        if self.busy():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load chains', SESSIONS_DIR, 'NumPy archive (*.npz);;All files (*)',
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
                      for i, a, *_ in parts(problem.fit)[0].params]
            self._fit_start = (params, [p[2] for p in params])
        self._make_dream_window([os.path.basename(path)])
        self.dream_window.show_loaded(
            result, problem, 'Loaded %s — %d parameter(s), %d points'
            % (html_escape(os.path.basename(path)), problem.ndim,
               problem.fit.npoints), meta.get('sources', []),
            meta.get('datasets', []))
        self._show_dream_window()
        self.dream_report.setText('Loaded %s.' % html_escape(
            os.path.basename(path)))

    def stop_dream(self):
        if self._dream is not None:
            self._dream.stop()
            self.dream_window.btn_stop.setEnabled(False)

    def _show_dream_window(self):
        if self.dream_window is not None:
            self._reveal(self.dream_window)

    def _reveal(self, w):
        """Bring the fit or DREAM window w forward: as a tab of page_tabs
        (added the first time, named w.tab_name, its title as tool tip), or
        on its own."""
        tabs = self.page_tabs
        if tabs is None:
            w.show()
            w.raise_()
            w.activateWindow()
            return
        i = tabs.indexOf(w)
        if i < 0:
            i = tabs.addTab(w, '')
        tabs.setTabText(i, getattr(w, 'tab_name', w.windowTitle()))
        tabs.setTabToolTip(i, w.windowTitle())
        tabs.setCurrentIndex(i)

    def close_page(self, w):
        """The close button of tab w of page_tabs: the fit or DREAM tab is
        deleted with its history; not while its fit or run goes on."""
        if isinstance(w, DEWindow):
            running, what = w is self.fit_window and (
                self._fit is not None or self._ms is not None), 'fit'
        elif isinstance(w, DreamWindow):
            running, what = w is self.dream_window and \
                self._dream is not None, 'sampling'
        else:
            return
        if running:
            QtWidgets.QMessageBox.information(
                self, 'Close', 'Stop the running %s first.' % what)
            return
        buttons = ()
        if w is self.fit_window:
            self.fit_window = None
            buttons = (self.btn_window, self.btn_lbfgs_window,
                       self.btn_lsq_window, *self._global_windows,
                       self.btn_ms_window)
        elif w is self.dream_window:
            self.dream_window = None
            buttons = (self.btn_dream_window,)
        for b in buttons:
            b.setEnabled(False)
        self.page_tabs.removeTab(self.page_tabs.indexOf(w))
        w.deleteLater()

    def _dream_done(self, result):
        self.dream_window.finish(result)
        self.dream_report.setText(
            'Done after %d generation(s); see the DREAM tab.'
            % result.generations[-1])

    def _dream_failed(self, msg):
        self.dream_window.fail(msg)
        self.dream_report.setText('Stopped.' if msg == 'stopped' else
                                  self._error('Sampling failed: %s' % msg))

    def _dream_cleanup(self):
        self._dream.deleteLater()
        self._dream = None
        self.simulation.set_locked(False)
        for b in (self.btn_fit, self.btn_lbfgs, self.btn_lsq,
                  *self._global_buttons, self.btn_sample,
                  self.btn_dream_load, self.btn_ms_run):
            b.setEnabled(True)

    def _same_layers(self, parent, title, what='run', fit=None):
        """The Simulation tab still has the layers of the fit problem `fit`
        (default: of the last fit or run); a message box on `parent` if
        not."""
        layers = self._fit_layers if fit is None else \
            [l.name for l in parts(fit)[0].stack.layers]
        if [l.name for l in self.simulation.stack.layers] == layers:
            return True
        QtWidgets.QMessageBox.warning(
            parent, title, 'The layers of the Simulation tab changed since '
            'the %s started;\nthese parameters no longer match them.' % what)
        return False

    @staticmethod
    def _whose(problem, part):
        """' (the values of <dataset>)' for a joint fit, else ''."""
        names = getattr(problem, 'part_names', None)
        return ' (the values of %s)' % html_escape(names[part]) if names \
            else ''

    def _send_dream(self, x, part=0, w=None, label='MAP'):
        """Write a DREAM state of window w (default: the newest) into the
        Simulation tab: the values dataset `part` of a joint fit sees;
        label says which state ('MAP', 'chain 3')."""
        w = w or self.dream_window
        fit = w.problem.fit
        if self.busy() or not self._same_layers(w, 'Load into model',
                                                fit=fit):
            return
        self.simulation.set_parameters(parts(fit)[part].params,
                                       part_x(fit, x, part))
        self.dream_report.setText('%s sent to the Simulation tab%s.'
                                  % (label, self._whose(fit, part)))
        self.btn_revert.setEnabled(True)

    def _bounds_dream(self, lo, hi, part=0, w=None):
        """Write posterior-based fit bounds of window w (default: the
        newest) into the Simulation tab: those of the parameters fitted in
        dataset `part` of a joint fit."""
        w = w or self.dream_window
        fit = w.problem.fit
        if self.busy() or not self._same_layers(w, 'Apply ±σ to bounds',
                                                fit=fit):
            return
        rows = [(p, lo[j], hi[j]) for p, j in zip(parts(fit)[part].params,
                                                  part_index(fit, part))
                if j is not None]
        self.simulation.set_bounds([r[0] for r in rows],
                                   [r[1] for r in rows], [r[2] for r in rows])
        self.dream_report.setText('Posterior bounds sent to the Simulation '
                                  'tab%s.' % self._whose(fit, part))
        self.btn_revert.setEnabled(True)

    def stop_fit(self):
        """The fit window's Stop: the DE / L-BFGS-B fit or the multistart."""
        if self._fit is not None:
            self._fit.stop()
            self.fit_window.btn_stop.setEnabled(False)
        self.stop_multistart()

    def _show_fit_window(self):
        if self.fit_window is not None:
            self._reveal(self.fit_window)

    def _send_fit(self, x, label, part=0, w=None):
        """Write an entry of fit window w (default: the newest) into the
        Simulation tab: the values dataset `part` of a joint fit sees."""
        w = w or self.fit_window
        p = w.problem
        if self.busy() or not self._same_layers(w, 'Send to Simulation',
                                                'fit', fit=p):
            return
        self.simulation.set_parameters(parts(p)[part].params,
                                       part_x(p, x, part))
        self._fit_out.setText(self._fit_header(
            'Sent <b>%s</b> to the Simulation tab%s' % (html_escape(
                '%s %s' % (w.unit, label) if label[:1].isdigit()
                else label),
                self._whose(p, part))))
        self.btn_revert.setEnabled(True)

    def revert_fit(self):
        if self._fit_start is not None and not self.busy():
            params, x = self._fit_start
            self.simulation.set_parameters(params, x)
            # the bounds too (Apply ±σ to bounds changes them)
            self.simulation.set_bounds(params, [p[3] for p in params],
                                       [p[4] for p in params])
            self._fit_out.setText('Parameters and bounds restored to '
                                  'their values before the fit.')
            self.btn_revert.setEnabled(False)

    def _fit_progress(self, x, cost, gen):
        self._fit_out.setText(self._fit_header(
            '%s %d — cost %.5g' % (self.fit_window.unit.capitalize(), gen,
                                   cost)))

    def _fit_status(self, info):
        if info['phase'] == 'polish':
            self._fit_out.setText(self._fit_header(
                'Polishing (L-BFGS-B)%s — cost %.5g' % (
                    ', iteration %d' % info['iteration']
                    if info['iteration'] else '…', info['best'])))

    def _fit_done(self, x, cost, gen, msg):
        self.fit_window.finish(x, cost, gen, msg)
        self._suggest_sigma(self._fit_problem, x)
        tension = ''.join(self._error(t) + '<br>' for t in prior_tension(
            self._fit_problem, x))
        self._fit_out.setText(self._fit_header(
            '<b>%s</b> after %d %s(s) — cost %.5g<br><i>%s</i><br>%s'
            'Send an entry from the fit window to use it.'
            % ('Stopped' if msg == 'stopped' else 'Done', gen,
               self.fit_window.unit, cost, html_escape(msg), tension)))

    def _fit_failed(self, msg):
        self._fit_ok = False
        self.fit_window.fail(msg)
        self._fit_out.setText(self._error('Fit failed: %s' % msg))

    def _fit_cleanup(self):
        # where it ended, for Continue (not after a failure)
        self._de_state = dict(self._fit.state) if self._fit_ok else None
        self.fit_window.btn_continue.setEnabled(bool(self._de_state))
        self._fit.deleteLater()
        self._fit = None
        self.simulation.set_locked(False)
        for b in (self.btn_fit, self.btn_lbfgs, self.btn_lsq,
                  *self._global_buttons, self.btn_sample,
                  self.btn_ms_run):
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
