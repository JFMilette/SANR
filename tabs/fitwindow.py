"""
Fit window: follows a running fit of the Experimental tab generation by
generation.

The history lists the best cost of every generation (and the polished
result at the end).  Selecting an entry draws its depth profile (with its
sublayers) and its model against the fitted data, as any quantity of the
Simulation tab (reflectivities, half-polarized, asymmetries; the model's
channels outside the fit use the Simulation tab's pairs); once the fit is
done or
stopped, "Send to Simulation" writes that entry's parameters into the
Simulation tab.  Nothing reaches the Simulation tab otherwise.

The window keeps its own copy of the model, taken before the fit thread
starts, so drawing never touches the stack the fit is evaluating.
"""

import copy
from html import escape as html_escape

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from .simulation import (ASYM_COLOUR, EDITOR_PARAMS, PROFILE_QUANTITIES,
                         REFL_QUANTITIES, RIGHT, LayerLabels, plot_widget,
                         profile_data)


# attr -> (label, display scale, unit) for the parameter tables
PARAM_DISPLAY = {attr: (label.split(' (')[0], scale,
                        suffix.strip() or ('Å' if attr == 'thickness' else
                                           '10⁻⁶ Å⁻²'))
                 for attr, label, *_, suffix, scale in EDITOR_PARAMS}
AT_BOUND = QtGui.QColor('#f87171')


def at_bound(v, lo, hi):
    """A value pinned at a bound usually means the bound is too tight."""
    return min(v - lo, hi - v) < 1e-3 * (hi - lo)


class FitWindow(QtWidgets.QWidget):
    """History of one fit with its profile and reflectivity plots."""

    stopRequested = QtCore.pyqtSignal()
    sendRequested = QtCore.pyqtSignal(object, str)   # x, entry label

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle('Fit')
        self.resize(1400, 850)
        self.problem = None
        self.stack = None              # the window's own copy of the model
        self.data = {}                 # fitted points {channel: dict(Q, R, ...)}
        self.vectors = {}              # {channel: [Pi, Pa]} of the model
        self.entries = []              # [(label, cost, x)]
        self.running = False
        self._x = None                 # parameters currently drawn

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_side())
        plots = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        plots.addWidget(self._build_profile_plot())
        plots.addWidget(self._build_reflectance_plot())
        plots.setSizes([380, 470])
        splitter.addWidget(plots)
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
                                 'into the Simulation tab')
        self.btn_stop.clicked.connect(self._stop)
        self.btn_send.clicked.connect(self._send)
        row.addWidget(self.btn_stop)
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
        p.getAxis('right').setLabel('MSLD θ, φ (deg)')
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
            vb = self.prof_vb2 if i in RIGHT else p.vb
            vb.addItem(bars)
            vb.addItem(curve)
            self.prof_bars.append(bars)
            self.prof_curves.append(curve)
        p.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#4b4e55', width=0.7)))
        self.prof_lines = []                  # interface lines
        self.prof_layers = LayerLabels(p)     # band and name of each layer
        v.addWidget(w)
        return box

    def _build_reflectance_plot(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel('Reflectivity plot:'))
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItems([q[0] for q in REFL_QUANTITIES])
        self.quantity.setToolTip('Fitted data (markers) and model (lines)')
        self.quantity.currentIndexChanged.connect(self._redraw_reflectance)
        bar.addWidget(self.quantity)
        self.formula = QtWidgets.QLabel()
        self.formula.setStyleSheet('color: #9aa0a6')
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
        v.addLayout(bar)
        w, self.refl_plot = plot_widget('Q (Å⁻¹)', 'Reflectivity')
        self.refl_plot.addLegend(offset=(-10, 10))
        self.zero = pg.InfiniteLine(pos=0, angle=0,
                                    pen=pg.mkPen('#4b4e55', width=0.7))
        self.refl_plot.addItem(self.zero)
        self._refl_items = []                 # everything drawn, for clearing
        self._model = None                    # model channels, as self.data
        v.addWidget(w)
        self.refl_note = QtWidgets.QLabel()
        self.refl_note.setStyleSheet('color: #f87171')
        v.addWidget(self.refl_note)
        return box

    # -- fit lifecycle (called by the Experimental tab) ---------------------
    def start(self, problem, data, vectors, Q, title, cost_label,
              polish=True):
        """A new fit: problem is the FitProblem the thread will run (copy its
        stack now, before the thread starts), data the fitted points
        {channel: dict(Q, R, dR, dQ)}, vectors the {channel: [Pi, Pa]} of
        every model channel, Q the grid of the model curves, polish whether
        the fit ends with a polish."""
        self.problem = problem
        self.polish, self.cost_label = polish, cost_label
        self.stack = copy.deepcopy(problem.stack)
        self.data, self.vectors, self.model_Q = data, vectors, Q
        self.entries = []
        self.running = True
        self.setWindowTitle('Fit — %s' % title)
        self.history.blockSignals(True)
        self.history.clear()
        self.history.setHeaderLabels(['Generation', cost_label])
        self.history.blockSignals(False)
        self.status.setText('Fitting %d parameter(s) to %d points…'
                            % (len(problem.params), problem.npoints))
        self.status.setStyleSheet('')
        self.btn_stop.setEnabled(True)
        self.btn_send.setEnabled(False)
        # the starting model until the first generation arrives: the
        # current values clipped into the bounds, as the fit starts from
        self._draw(problem.to_x(problem.x0()))
        self.show()
        self.raise_()
        self.activateWindow()

    def add_generation(self, x, cost, gen):
        self._append('%d' % gen, cost, x)

    def finish(self, x, cost, gen, msg):
        self.running = False
        stopped = msg == 'stopped'
        # the polish (or a stop before the first generation) gives a result
        # that is not in the history yet.  Compared exactly: any tolerance
        # in physical units would hide small SLD moves (~1e-6 A^-2)
        new = not self.entries or cost != self.entries[-1][1] or \
            not np.array_equal(x, self.entries[-1][2])
        if new:
            self._append('stopped' if stopped else 'polished', cost, x)
        note = ''
        if self.polish and not stopped:
            note = '<br>Polish: %s' % (
                '%s %.6g → %.6g' % (self.cost_label, self.entries[-2][1], cost)
                if new and len(self.entries) > 1 else
                'no improvement on the last generation')
        self.status.setText(
            '<b>%s</b> after %d generation(s) — best %.6g<br><i>%s</i>%s<br>'
            'Select an entry and send it to the Simulation tab.'
            % ('Stopped' if stopped else 'Done', gen, cost, html_escape(msg),
               note))
        self.btn_stop.setEnabled(False)
        self.btn_send.setEnabled(self.history.currentItem() is not None)

    def fail(self, msg):
        self.running = False
        self.status.setText('Fit failed: %s' % html_escape(msg))
        self.status.setStyleSheet('color: #f87171')
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
            self.sendRequested.emit(e[2], e[0])

    def closeEvent(self, ev):
        # closing only hides: the fit and its history carry on
        self.hide()
        ev.ignore()

    # -- drawing ------------------------------------------------------------
    def _draw(self, x):
        self._x = np.asarray(x, dtype=float)
        try:
            self.problem.apply(self._x, self.stack)
            self.stack.build_sublayers()
            self._fill_params()
            self._draw_profile()
            # imported here: the Experimental tab imports this module
            from .experimental import simulation_channels
            self._model = simulation_channels(self.stack, self.model_Q,
                                              self.vectors)
            self._redraw_reflectance()
        except Exception as exc:                      # keep the window alive
            self.status.setText('Cannot draw this entry: %s' % exc)
            self.status.setStyleSheet('color: #f87171')

    def _fill_params(self):
        t = self.params
        t.clear()
        layers = self.stack.layers
        for (i, attr, _, lo, hi), v in zip(self.problem.params, self._x):
            label, scale, unit = PARAM_DISPLAY[attr]
            it = QtWidgets.QTreeWidgetItem(
                [layers[i].name, label, '%.6g' % (v * scale), unit])
            it.setTextAlignment(2, QtCore.Qt.AlignmentFlag.AlignRight
                                | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if at_bound(v, lo, hi):
                it.setForeground(2, AT_BOUND)
                it.setToolTip(2, 'At its fit bound [%.6g, %.6g]'
                              % (lo * scale, hi * scale))
            t.addTopLevelItem(it)
        for c in range(4):
            t.resizeColumnToContents(c)

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
        if self.stack is None:
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

    def _redraw_reflectance(self, *_):
        # imported here: the Experimental tab imports this module
        from .experimental import dataset_series
        p = self.refl_plot
        for item in self._refl_items:
            p.removeItem(item)
        self._refl_items = []
        p.legend.clear()
        if self._model is None:
            return
        k = self.quantity.currentIndex()
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
            curve = pg.PlotDataItem(Q, y, pen=pg.mkPen(col or ASYM_COLOUR,
                                                       width=1.8),
                                    connect='finite')
            p.addItem(curve)
            self._refl_items.append(curve)
        for lab, col, Q, y, dy, _ in data:
            if rq4:
                y, dy = y * Q**4, dy * Q**4
            self._add_data(lab, col or ASYM_COLOUR, Q, y, dy, log)
        self.refl_note.setText('Fitted data: %s' % note if note else '')
        self.refl_note.setVisible(bool(note))
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
