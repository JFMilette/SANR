"""
The Fit tool box of the Experimental tab: a "Fit — general" page (dataset,
worker processes, Revert: shared by both methods), a "Fit — differential
evolution" page (fit.de) and a "Fit — Bayesian (DREAM)" page (fit.dream),
with everything they do: building the fit problem from the chosen dataset
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
from tabs.simulation import dspin
from fit.problem import FitProblem
from fit.de.de_window import DEThread, DEWindow
from fit.dream.dream_fit import PosteriorProblem, load_run
from fit.dream.dream_window import DreamThread, DreamWindow

DIALOG_OPTIONS = QtWidgets.QFileDialog.Option.DontUseNativeDialog
FIT_COSTS = [('chi2', 'χ² (weighted by dR)'), ('log', 'log R (per decade)')]


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
        self.fit_tol = dspin(0, 1, 4, 0.001)
        self.fit_tol.setValue(0.01)
        self.fit_tol.setToolTip('Stop when the spread of the population\'s '
                                'cost falls below tol × its mean')
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
        f.addRow(QtWidgets.QLabel(
            '<i>Every channel of the dataset is fitted between Q min and '
            'Q max of the Simulation tab, varying the parameters ticked Fit '
            'there (scale and background included); its Q points set the '
            'model curves of the fit windows.</i>', wordWrap=True))
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_revert)
        f.addRow(row)
        self.fit_tools.addItem(page, 'Fit — general')

        # -- differential evolution
        page, f = self._page()
        f.addRow('Cost', self.fit_cost)
        f.addRow(self._grid([('Max iter', self.fit_maxiter),
                             ('Pop', self.fit_pop), ('Tol', self.fit_tol)]))
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
        self.fit_window = None         # DEWindow, made by the first fit
        self.fit_tools.addItem(page, 'Fit — differential evolution')

        # -- DREAM
        self.fit_tools.addItem(self._build_dream_page(),
                               'Fit — Bayesian (DREAM)')
        self.fit_tools.setCurrentIndex(1)
        # room for the tallest page: the tool box would scroll it otherwise
        tabs = sum(b.sizeHint().height() + 2 for b in
                   self.fit_tools.findChildren(QtWidgets.QAbstractButton)
                   if b.parent() is self.fit_tools)
        self.fit_tools.setMinimumHeight(tabs + max(
            self.fit_tools.widget(i).sizeHint().height()
            for i in range(self.fit_tools.count())))
        self.fit_tools.setEnabled(self.simulation is not None)
        return self.fit_tools

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
            'Number of chains; auto = 3 with the archive, otherwise '
            'max(7, d + 1).\nWith the archive any number from 2 works (R-hat '
            'compares chains, so 1 can never converge); without it at least '
            '7 (2δ + 1).')
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
            'with 2 chains or more (3 by default) and keeps every mode.\n'
            'Unchecked: classic DREAM with the outlier-chain test, at least 7 '
            'chains.')
        self._chains_last = 0
        self.dream_chains.valueChanged.connect(self._check_chains)
        self.dream_archive.toggled.connect(self._check_chains)
        self.dream_sigma = QtWidgets.QCheckBox('fit error scale')
        self.dream_sigma.setToolTip(
            'Sample a common factor 10^s on every dR, s in [-1, 1] '
            '(log10_sigma_scale)')
        self.dream_stop = QtWidgets.QCheckBox('stop when converged')
        self.dream_stop.setChecked(True)
        self.dream_stop.setToolTip(
            'Stop once R-hat < the R̂ target for every parameter, plus a fifth '
            'of the generations (at most 5000) more')
        self.dream_rhat = QtWidgets.QDoubleSpinBox()
        self.dream_rhat.setRange(1.001, 2.0)
        self.dream_rhat.setDecimals(3)
        self.dream_rhat.setSingleStep(0.01)
        self.dream_rhat.setValue(1.2)
        self.dream_rhat.setToolTip(
            'Converged once the Gelman-Rubin R-hat of every parameter is '
            'below this.\nUsed to stop the run (stop when converged), for '
            'the convergence flag and for the "Not converged" warning.\n'
            '1.2 is Vrugt\'s default; 1.1 or 1.05 is stricter and needs '
            'longer runs.')
        f.addRow(self._grid([('Generations', self.dream_gen),
                             ('Chains', self.dream_chains),
                             ('Seed', self.dream_seed),
                             ('R̂ target', self.dream_rhat)]))
        opts = QtWidgets.QGridLayout()
        opts.setContentsMargins(0, 0, 0, 0)
        for k, w in enumerate((self.dream_archive, self.dream_sigma,
                               self.dream_stop)):
            opts.addWidget(w, k // 2, k % 2)
        f.addRow(opts)
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
        """A fit or a DREAM run is going on."""
        return self._fit is not None or self._dream is not None

    def _fit_inputs(self, report):
        """(dataset, {channel: [Pi, Pa]}, data list, qmin, qmax) of the Fit
        box's dataset for fit.problem.FitProblem, after pushing the Simulation tab's
        settings into its stack; None, with the reason in `report`, if
        there is nothing to fit."""
        i = self._set_index(self.fit_set.currentText())
        if i is None:
            report.setText(self._error('Import data to fit first.'))
            return None
        sim = self.simulation
        sim.recompute()                 # push the tab's settings into the stack
        s = self.sets[i]
        vec = {ch: self.channel_vectors(s, ch) for ch in s['channels']}
        data = [dict(zip(('P0', 'P'), vectors_pair(*vec[ch])), name='R' + ch,
                     Q=d['Q'], R=d['R'], dR=d['dR'], norm=d.get('norm', 1.0))
                for ch, d in s['channels'].items()]
        # the Simulation tab's Q range selects the fitted points
        return s, vec, data, sim.qmin.value(), sim.qmax.value()

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
        s, vec, data, qmin, qmax = inputs
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                problem = FitProblem(
                    sim.stack, data,
                    FIT_COSTS[self.fit_cost.currentIndex()][0], qmin, qmax)
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
            ok = np.isfinite(d['Q']) & np.isfinite(d['R']) & \
                (d['Q'] >= qmin) & (d['Q'] <= qmax)
            if problem.cost == 'log':
                ok &= d['R'] > 0
            if ok.any():
                shown[ch] = {c: d[c][ok] for c in ('Q', 'R', 'dR', 'dQ')}
        if self.fit_window is None:
            self.fit_window = DEWindow(self)
            self.fit_window.stopRequested.connect(self.stop_fit)
            self.fit_window.sendRequested.connect(self._send_fit)
        # the model's other channels use the Simulation tab's pairs
        self.fit_window.start(
            problem, shown, dict(self.sim_vectors(), **vec),
            np.linspace(qmin, qmax, sim.nq.value()), s['name'],
            'χ²' if problem.cost == 'chi2' else 'cost (log R)',
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
                              workers=self.fit_workers.value())
        self._fit.progress.connect(self._fit_progress)
        self._fit.progress.connect(self.fit_window.add_generation)
        self._fit.finished_fit.connect(self._fit_done)
        self._fit.failed.connect(self._fit_failed)
        self._fit.finished.connect(self._fit_cleanup)
        # a result is only sent to the stack the fit started from
        sim.set_locked(True)
        self.btn_fit.setEnabled(False)
        self.btn_sample.setEnabled(False)
        self.btn_revert.setEnabled(False)
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
        s, vec, data, qmin, qmax = inputs
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                problem = PosteriorProblem(
                    FitProblem(sim.stack, data, 'chi2', qmin, qmax),
                    sigma_free=self.dream_sigma.isChecked())
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
            min_gen_after_converged=min(5000, n_gen // 5))
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
                  self.btn_dream_load):
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
        for b in (self.btn_fit, self.btn_sample, self.btn_dream_load):
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
        self.fit_report.setText(self._fit_header(
            '<b>%s</b> after %d generation(s) — cost %.5g<br><i>%s</i><br>'
            'Send an entry from the fit window to use it.'
            % ('Stopped' if msg == 'stopped' else 'Done', gen, cost,
               html_escape(msg))))

    def _fit_failed(self, msg):
        self.fit_window.fail(msg)
        self.fit_report.setText(self._error('Fit failed: %s' % msg))

    def _fit_cleanup(self):
        self._fit.deleteLater()
        self._fit = None
        self.simulation.set_locked(False)
        self.btn_fit.setEnabled(True)
        self.btn_sample.setEnabled(True)

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
        for t in (self._fit, self._dream):
            if t is not None:
                t.stop()
                t.wait()
