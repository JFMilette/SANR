"""
Simulation tab: build a Stack by hand and look at its profile and reflectivity.

    left  : general parameters, layer list (add / remove / reorder),
            property editor of the selected layer
    right : depth profile on top, reflectivity below, both wide.
            The profile overlays NSLD / MSLD (left axis) and the magnetic
            angle (right axis), each toggled by a checkbox:
            solid line = Stack.profile(z), bars = Stack.build_sublayers(),
            shaded bands = layers (selected one's name in bold, click a band
            to select it), grey verticals = nominal interfaces.

save_model / load_model write / read the stack and Q settings as JSON.

Every edit triggers a debounced recompute.
"""

import json
import re

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from model.stack import Layer, Stack


SLD_SCALE = 1e-6                      # SLDs are edited / plotted in 1e-6 A^-2
CHANNELS = ['++', '+-', '-+', '--']
AUTO_NAME = re.compile(r'L\d+')        # default layer names, renumbered
CH_COLOURS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
HP_COLOURS = ['#1f77b4', '#ff7f0e']    # half-polarized R↑, R↓
# combo label, axis label, colour
PROFILE_QUANTITIES = [
    ('NSLD', 'NSLD real (10⁻⁶ Å⁻²)', '#1f4e79'),
    ('MSLD', 'MSLD ρ (10⁻⁶ Å⁻²)', '#2a7f62'),
    ('Magnetic angle', 'MSLD φ (deg)', '#8b5a2b'),
]
# combo label, axis label (None = raw reflectivities), formula beside the combo.
# Channels indexed 0..3 = ↑↑ ↑↓ ↓↑ ↓↓ in the chosen basis.
REFL_QUANTITIES = [
    ('Reflectivity', None, ''),
    ('Half-polarized', None, 'R↑ = R↑↑ + R↑↓,   R↓ = R↓↓ + R↓↑'),
    ('SA NSF', 'SA_NSF', '(R↑↑ − R↓↓) / (R↑↑ + R↓↓)'),
    ('SF fraction', 'SF fraction', '(R↑↓ + R↓↑) / (R↑↑ + R↑↓ + R↓↑ + R↓↓)'),
    ('SA SF', 'SA_SF', '(R↑↓ − R↓↑) / (R↑↓ + R↓↑)'),
    ('SA half-polarized', 'SA',
     '(R↑ − R↓) / (R↑ + R↓),   R↑ = R↑↑ + R↑↓,  R↓ = R↓↓ + R↓↑'),
]
ASYM_COLOUR = '#6a3d9a'
# layer shading in the depth profile: alternating greys
BANDS = ['#00000000', '#0000000f']


def ratio(a, b):
    """a / b, NaN where b is not positive."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b > 0, a / b, np.nan)


def refl_quantity(name, R):
    """Asymmetry `name` of REFL_QUANTITIES from R = [R↑↑, R↑↓, R↓↑, R↓↓]."""
    uu, ud, du, dd = R
    if name == 'SA NSF':
        return ratio(uu - dd, uu + dd)
    if name == 'SF fraction':
        return ratio(ud + du, uu + ud + du + dd)
    if name == 'SA SF':
        return ratio(ud - du, ud + du)
    up, dn = uu + ud, dd + du
    return ratio(up - dn, up + dn)


def make_stack():
    """Default sample loaded when the tab opens."""
    vacuum = Layer('vacuum')

    L1 = Layer('L1', thickness=80.0, NSLD_real=4.0e-6,
               MSLD_rho=1.5e-6, MSLD_phi=0.75,
               roughness_sigma=8.0, roughness_model='tanh', roughness_sublayer=20)

    L2 = Layer('L2', thickness=120.0, NSLD_real=1.5e-6, NSLD_img=-2.0e-8,
               MSLD_rho=0.8e-6, MSLD_phi=0.5,
               roughness_sigma=5.0, roughness_model='tanh', roughness_sublayer=20)

    L3 = Layer('L3', thickness=60.0, NSLD_real=6.0e-6,
               MSLD_rho=0.0, MSLD_phi=0.0,
               roughness_sigma=8.0, roughness_model='tanh', roughness_sublayer=20)

    substrate = Layer('substrate', NSLD_real=2.07e-6,
                      roughness_sigma=4.0, roughness_model='tanh',
                      roughness_sublayer=20)

    return Stack([vacuum, L1, L2, L3, substrate])


def dspin(lo, hi, dec, step, suffix=''):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    return w


def plot_widget(xlabel, ylabel):
    w = pg.PlotWidget()
    p = w.getPlotItem()
    p.setLabel('bottom', xlabel)
    p.setLabel('left', ylabel)
    p.showGrid(x=True, y=True, alpha=0.25)
    p.getAxis('left').setWidth(70)
    p.getAxis('left').enableAutoSIPrefix(False)
    p.getAxis('bottom').enableAutoSIPrefix(False)
    return w, p


# ------------------------------------------------------------ cursor ----
class Crosshair(QtCore.QObject):
    """Vertical cursor following the mouse over `plot`, with a dot on each
    curve and a floating readout beside the pointer.

    readout(x) returns (header, rows) with rows = [(label, value, colour, y,
    viewbox)]; y is in the viewbox's own coordinates (NaN = no dot), and
    header None hides the cursor.  Everything is added straight to the
    viewboxes so the CSV exporter and autorange ignore it."""

    def __init__(self, widget, plot, readout):
        super().__init__(widget)
        self.plot, self.readout = plot, readout
        self._pos = None                        # last scene position
        vb = plot.vb
        self.line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(
            '#444444', width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self.text = pg.TextItem(anchor=(0, 1), color='#222222',
                                fill=pg.mkBrush(255, 255, 255, 225),
                                border=pg.mkPen('#999999'))
        for item in (self.line, self.text):
            item.setZValue(1000)
            vb.addItem(item, ignoreBounds=True)
        self.markers = {}                       # viewbox -> ScatterPlotItem
        self.hide()
        self._proxy = pg.SignalProxy(plot.scene().sigMouseMoved,
                                     rateLimit=60, slot=self._moved)
        widget.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Leave:
            self._pos = None
            self.hide()
        return False

    def hide(self):
        for item in [self.line, self.text] + list(self.markers.values()):
            item.setVisible(False)

    def refresh(self):
        """Redraw at the last mouse position (e.g. after the data changed)."""
        if self._pos is not None:
            self._update(self._pos)

    def _moved(self, evt):
        self._pos = evt[0]
        self._update(self._pos)

    def _marker(self, vb):
        if vb not in self.markers:
            m = pg.ScatterPlotItem(size=8, pen=pg.mkPen('w', width=1))
            m.setZValue(999)
            vb.addItem(m, ignoreBounds=True)
            self.markers[vb] = m
        return self.markers[vb]

    def _update(self, pos):
        vb = self.plot.vb
        if not vb.sceneBoundingRect().contains(pos):
            self.hide()
            return
        pt = vb.mapSceneToView(pos)
        x = pt.x()
        header, rows = self.readout(x)
        if header is None:
            self.hide()
            return

        html = ['<b>%s</b>' % header]
        dots = {m: ([], []) for m in self.markers}
        for label, value, col, y, box in rows:
            html.append('<span style="color:%s">■</span> %s: %s'
                        % (col, label, value))
            if np.isfinite(y):
                ys, cs = dots.setdefault(box, ([], []))
                ys.append(y)
                cs.append(pg.mkBrush(col))
        self.text.setHtml('<div style="font-size:9pt">%s</div>'
                          % '<br>'.join(html))
        for box, (ys, cs) in dots.items():
            m = self._marker(box)
            m.setData(x=[x] * len(ys), y=ys, brush=cs)
            m.setVisible(bool(ys))

        # keep the readout inside the view: flip to the other side of the
        # pointer past the middle of the range
        (x0, x1), (y0, y1) = vb.viewRange()
        self.text.setAnchor((1 if x > 0.5 * (x0 + x1) else 0,
                             0 if pt.y() > 0.5 * (y0 + y1) else 1))
        self.text.setPos(x, pt.y())
        self.line.setPos(x)
        self.line.setVisible(True)
        self.text.setVisible(True)


def fmt(v):
    return '—' if not np.isfinite(v) else '%.4g' % v


# ------------------------------------------------------- layer editor ----
class LayerEditor(QtWidgets.QGroupBox):
    """Form bound to one Layer; emits `changed` after writing into it."""

    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__('Layer properties')
        self.layer = None
        self._loading = False

        self.name = QtWidgets.QLineEdit()
        self.thickness = dspin(0.0, 1e5, 2, 1.0, ' Å')
        self.nsld_re = dspin(-100, 100, 4, 0.1)
        self.nsld_im = dspin(-100, 100, 5, 0.001)
        self.msld_rho = dspin(0, 100, 4, 0.1)
        self.msld_deg = dspin(-360, 360, 2, 5.0, ' °')
        self.sigma = dspin(0.0, 1e4, 2, 0.5, ' Å')
        self.model = QtWidgets.QComboBox()
        self.model.addItems(['tanh', 'erf'])
        self.nsub = QtWidgets.QSpinBox()
        self.nsub.setRange(1, 1000)
        self.nsub.setKeyboardTracking(False)

        unit = '(10⁻⁶ Å⁻²)'
        form = QtWidgets.QFormLayout(self)
        form.addRow('Name', self.name)
        form.addRow('Thickness', self.thickness)
        form.addRow('NSLD real %s' % unit, self.nsld_re)
        form.addRow('NSLD imag %s' % unit, self.nsld_im)
        form.addRow('MSLD ρ %s' % unit, self.msld_rho)
        form.addRow('MSLD φ', self.msld_deg)
        form.addRow(QtWidgets.QLabel('<i>Interface at top of this layer</i>'))
        form.addRow('Roughness σ', self.sigma)
        form.addRow('Roughness model', self.model)
        form.addRow('Sublayers', self.nsub)

        self.name.editingFinished.connect(self._store)
        for w in (self.thickness, self.nsld_re, self.nsld_im, self.msld_rho,
                  self.msld_deg, self.sigma, self.nsub):
            w.valueChanged.connect(self._store)
        self.model.currentIndexChanged.connect(self._store)
        self.setEnabled(False)

    def load(self, layer, is_fronting, is_backing):
        self.layer = layer
        self.setEnabled(layer is not None)
        if layer is None:
            return
        self._loading = True
        self.name.setText(layer.name)
        self.thickness.setValue(layer.thickness)
        self.nsld_re.setValue(layer.NSLD_real / SLD_SCALE)
        self.nsld_im.setValue(layer.NSLD_img / SLD_SCALE)
        self.msld_rho.setValue(layer.MSLD_rho / SLD_SCALE)
        self.msld_deg.setValue(layer.MSLD_phi * 360.0)
        self.sigma.setValue(layer.roughness_sigma)
        self.model.setCurrentText(layer.roughness_model)
        self.nsub.setValue(int(layer.roughness_sublayer))
        self._loading = False
        # semi-infinite media have no thickness; fronting has no top interface
        self.thickness.setEnabled(not (is_fronting or is_backing))
        for w in (self.sigma, self.model, self.nsub):
            w.setEnabled(not is_fronting)

    def _store(self):
        if self._loading or self.layer is None:
            return
        L = self.layer
        L.name = self.name.text()
        L.thickness = self.thickness.value()
        L.NSLD_real = self.nsld_re.value() * SLD_SCALE
        L.NSLD_img = self.nsld_im.value() * SLD_SCALE
        L.MSLD_rho = self.msld_rho.value() * SLD_SCALE
        L.MSLD_phi = self.msld_deg.value() / 360.0
        L.roughness_sigma = self.sigma.value()
        L.roughness_model = self.model.currentText()
        L.roughness_sublayer = self.nsub.value()
        self.changed.emit()


# ----------------------------------------------------- simulation tab ----
class SimulationTab(QtWidgets.QWidget):

    def __init__(self, stack=None, parent=None):
        super().__init__(parent)
        self.stack = stack if stack is not None else make_stack()
        self._profile = None        # (z, edges, [curves], [slab values])

        self._timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self.recompute)

        plots = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        plots.addWidget(self._build_profile_plot())
        plots.addWidget(self._build_reflectance_plot())
        plots.setSizes([400, 500])

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(plots)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1270])

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.refresh_list(select=1)
        self.recompute()

    # -- left column --------------------------------------------------------
    def _build_controls(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)

        # general parameters
        gen = QtWidgets.QGroupBox('General parameters')
        f = QtWidgets.QFormLayout(gen)
        self.qmin = dspin(1e-5, 10, 5, 0.001, ' Å⁻¹')
        self.qmax = dspin(1e-4, 10, 4, 0.01, ' Å⁻¹')
        self.nq = QtWidgets.QSpinBox()
        self.nq.setRange(2, 100000)
        self.nq.setKeyboardTracking(False)
        self.tail = dspin(0.5, 20, 2, 0.5, ' σ')
        self.qmin.setValue(1e-4)
        self.qmax.setValue(0.25)
        self.nq.setValue(400)
        self.tail.setValue(self.stack.tail)
        self.alpha = dspin(-360, 360, 2, 5.0, ' °')
        self.alpha.setValue(np.degrees(self.stack.alpha))
        self.alpha.setToolTip('Polarisation angle α mixing the spin channels '
                              'into the lab channels (++ +- -+ --)')

        self.basis = QtWidgets.QComboBox()
        self.basis.addItems(['lab (++ +- -+ --)', 'spin (uu ud du dd)'])
        self.logy = QtWidgets.QCheckBox('log R')
        self.logy.setChecked(True)
        self.rq4 = QtWidgets.QCheckBox('R·Q⁴')

        f.addRow('Q min', self.qmin)
        f.addRow('Q max', self.qmax)
        f.addRow('Q points', self.nq)
        f.addRow('Window tail', self.tail)
        f.addRow('Basis', self.basis)
        f.addRow('Polarisation α', self.alpha)
        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(self.logy)
        opts.addWidget(self.rq4)
        f.addRow('Display', opts)
        for w in (self.qmin, self.qmax, self.tail, self.alpha):
            w.valueChanged.connect(self.schedule)
        self.nq.valueChanged.connect(self.schedule)
        self.basis.currentIndexChanged.connect(self.schedule)
        self.logy.toggled.connect(self.schedule)
        self.rq4.toggled.connect(self.schedule)
        v.addWidget(gen)

        # layer list
        lay = QtWidgets.QGroupBox('Layers  (top → bottom)')
        lv = QtWidgets.QVBoxLayout(lay)
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._on_select)
        self.list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.list.model().rowsMoved.connect(self._on_rows_moved)
        lv.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton('+ Add')
        self.btn_del = QtWidgets.QPushButton('− Remove')
        self.btn_up = QtWidgets.QPushButton('▲')
        self.btn_dn = QtWidgets.QPushButton('▼')
        for b in (self.btn_add, self.btn_del, self.btn_up, self.btn_dn):
            row.addWidget(b)
        self.btn_add.clicked.connect(self.add_layer)
        self.btn_del.clicked.connect(self.remove_layer)
        self.btn_up.clicked.connect(lambda: self.move_layer(-1))
        self.btn_dn.clicked.connect(lambda: self.move_layer(+1))
        lv.addLayout(row)
        v.addWidget(lay, 1)

        self.editor = LayerEditor()
        self.editor.changed.connect(self._on_layer_edit)
        v.addWidget(self.editor)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        return panel

    # -- plots --------------------------------------------------------------
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
            cb.toggled.connect(self._show_profile)
            self.prof_boxes.append(cb)
            bar.addWidget(cb)
        self.prof_show_bars = QtWidgets.QCheckBox('sublayers')
        self.prof_show_bars.setChecked(True)
        self.prof_show_bars.toggled.connect(self._show_profile)
        bar.addWidget(self.prof_show_bars)
        bar.addStretch(1)
        v.addLayout(bar)

        w, self.prof_plot = plot_widget('depth z (Å)', 'SLD (10⁻⁶ Å⁻²)')
        p = self.prof_plot
        # SLDs share the left axis; the angle (deg) gets its own right axis
        self.prof_vb2 = pg.ViewBox()
        p.showAxis('right')
        p.scene().addItem(self.prof_vb2)
        p.getAxis('right').linkToView(self.prof_vb2)
        p.getAxis('right').setLabel(PROFILE_QUANTITIES[2][1])
        p.getAxis('right').enableAutoSIPrefix(False)
        self.prof_vb2.setXLink(p)

        def sync():
            self.prof_vb2.setGeometry(p.vb.sceneBoundingRect())
            self.prof_vb2.linkedViewChanged(p.vb, self.prof_vb2.XAxis)
        p.vb.sigResized.connect(sync)

        self.prof_bars, self.prof_curves = [], []
        for i, (name, _, col) in enumerate(PROFILE_QUANTITIES):
            bars = pg.BarGraphItem(x0=[], width=[], height=[],
                                   brush=pg.mkBrush(col + '33'),
                                   pen=pg.mkPen(col, width=0.6))
            curve = pg.PlotDataItem(pen=pg.mkPen(col, width=2))
            if i == 2:
                self.prof_vb2.addItem(bars)
                self.prof_vb2.addItem(curve)
                # the CSV exporter only walks PlotItem.items; register the
                # right-axis curve there too without adding it to p.vb
                p.items.append(curve)
            else:
                p.addItem(bars)
                p.addItem(curve)
            self.prof_bars.append(bars)
            self.prof_curves.append(curve)
        p.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#bbbbbb', width=0.7)))
        self.prof_decor = []                 # interface lines
        self.prof_bands = []                 # one shaded region per layer
        self.prof_labels = []                # layer names, pinned to the top
        p.vb.sigYRangeChanged.connect(self._place_layer_labels)
        p.scene().sigMouseClicked.connect(self._on_profile_click)
        self.prof_cursor = Crosshair(w, p, self._profile_readout)
        v.addWidget(w)
        return box

    def _build_reflectance_plot(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel('Reflectivity plot:'))
        self.refl_quantity = QtWidgets.QComboBox()
        self.refl_quantity.addItems([q[0] for q in REFL_QUANTITIES])
        self.refl_quantity.currentIndexChanged.connect(self.schedule)
        bar.addWidget(self.refl_quantity)
        self.refl_formula = QtWidgets.QLabel()
        self.refl_formula.setStyleSheet('color: #555555')
        bar.addWidget(self.refl_formula)
        bar.addStretch(1)
        v.addLayout(bar)

        w, self.refl_plot = plot_widget('Q (Å⁻¹)', 'Reflectivity')
        self.refl_plot.addLegend(offset=(-10, 10))
        self.refl_curves = [
            self.refl_plot.plot(pen=pg.mkPen(c, width=1.8), connect='finite')
            for c in CH_COLOURS]
        self.hp_curves = [
            self.refl_plot.plot(pen=pg.mkPen(c, width=1.8), connect='finite')
            for c in HP_COLOURS]
        self.asym_curve = self.refl_plot.plot(
            pen=pg.mkPen(ASYM_COLOUR, width=1.8), connect='finite')
        self.asym_zero = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#bbbbbb', width=0.7))
        self.refl_plot.addItem(self.asym_zero)
        self._refl_shown = []               # [(curve, name, Q, y)] on screen
        self._refl_log = False
        self.refl_cursor = Crosshair(w, self.refl_plot, self._refl_readout)
        v.addWidget(w)
        return box

    # -- layer list management ---------------------------------------------
    def refresh_list(self, select=None):
        self.list.blockSignals(True)
        self.list.clear()
        n = len(self.stack.layers)
        for i, l in enumerate(self.stack.layers):
            tag = ' [fronting]' if i == 0 else ' [backing]' if i == n - 1 else ''
            t = '' if tag else '  %.1f Å' % l.thickness
            self.list.addItem('%d  %s%s%s' % (i, l.name, t, tag))
            if tag:                          # fronting / backing stay put
                it = self.list.item(i)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled)
        self.list.blockSignals(False)
        if select is not None:
            self.list.setCurrentRow(max(0, min(select, n - 1)))
        self._on_select(self.list.currentRow())

    def _on_select(self, row):
        n = len(self.stack.layers)
        ok = 0 <= row < n
        self.editor.load(self.stack.layers[row] if ok else None,
                         row == 0, row == n - 1)
        interior = ok and 0 < row < n - 1
        self.btn_del.setEnabled(interior)
        self.btn_up.setEnabled(interior and row > 1)
        self.btn_dn.setEnabled(interior and row < n - 2)
        self._style_bands()

    def add_layer(self):
        row = self.list.currentRow()
        pos = min(max(row + 1, 1), len(self.stack.layers) - 1)
        self.stack.layers.insert(pos, Layer('L%d' % pos, thickness=50.0,
                                            NSLD_real=2.0e-6, roughness_sigma=3.0,
                                            roughness_sublayer=10))
        self._renumber()
        self.refresh_list(select=pos)
        self.schedule()

    def remove_layer(self):
        row = self.list.currentRow()
        if 0 < row < len(self.stack.layers) - 1:
            del self.stack.layers[row]
            self._renumber()
            self.refresh_list(select=row)
            self.schedule()

    def _renumber(self):
        """Rename auto-named interior layers ('L<n>') to match their position;
        manually named layers are left alone."""
        for i, l in enumerate(self.stack.layers[1:-1], start=1):
            if AUTO_NAME.fullmatch(l.name):
                l.name = 'L%d' % i

    def move_layer(self, d):
        row = self.list.currentRow()
        new = row + d
        if 0 < row < len(self.stack.layers) - 1 and \
                0 < new < len(self.stack.layers) - 1:
            L = self.stack.layers
            L[row], L[new] = L[new], L[row]
            self._renumber()
            self.refresh_list(select=new)
            self.schedule()

    def _on_rows_moved(self, _parent, start, _end, _dest_parent, dest):
        """Mirror a drag-and-drop in the list onto the stack; drops onto the
        fronting / backing positions are reverted."""
        L = self.stack.layers
        new = dest if dest < start else dest - 1
        if 0 < start < len(L) - 1 and 0 < new < len(L) - 1 and new != start:
            L.insert(new, L.pop(start))
            self._renumber()
            self.schedule()
        else:
            new = start
        # rebuild after Qt has finished the drop
        QtCore.QTimer.singleShot(0, lambda: self.refresh_list(select=new))

    def _on_layer_edit(self):
        row = self.list.currentRow()
        self.refresh_list(select=row)
        self.schedule()

    # -- computation --------------------------------------------------------
    def schedule(self, *_):
        self._timer.start()

    def recompute(self):
        try:
            self.stack.tail = self.tail.value()
            self.stack.alpha = np.radians(self.alpha.value())
            self.stack.build_sublayers()
            self._compute_profile()
            self._show_profile()
            self._draw_reflectance()
            self.status.setText('%d layers, %d sublayers' %
                                (len(self.stack.layers), len(self.stack.sublayers)))
            self.status.setStyleSheet('')
            self.prof_cursor.refresh()
            self.refl_cursor.refresh()
        except Exception as exc:                      # keep the UI alive
            self.status.setText('Error: %s' % exc)
            self.status.setStyleSheet('color: #b00020')

    def _compute_profile(self):
        st = self.stack
        Z = st._interfaces()
        lo = st.windows()[0][0]
        e = lo + np.concatenate([[0.0], np.cumsum([s.thickness
                                                   for s in st.sublayers])])
        pad = 0.35 * max(Z[-1], 1.0)
        z = np.linspace(min(e[0], Z[0] - pad) - 10,
                        max(e[-1], Z[-1] + pad) + 10, 4000)
        nsld, rho, phi = st.profile(z)
        curves = [nsld.real / SLD_SCALE, rho / SLD_SCALE, phi * 360.0]
        # fronting and backing are semi-infinite: draw them as one bar each
        # out to the edges of the plotted range
        e = np.concatenate([[z[0]], e, [z[-1]]])
        slabs_all = [st.fronting] + list(st.sublayers) + [st.backing]
        slabs = [np.array([s.NSLD_real / SLD_SCALE for s in slabs_all]),
                 np.array([s.MSLD_rho / SLD_SCALE for s in slabs_all]),
                 np.array([s.MSLD_phi * 360.0 for s in slabs_all])]
        self._profile = (z, Z, e, curves, slabs)

    def _show_profile(self, *_):
        """Draw the checked quantities from the cached profile."""
        if self._profile is None:
            return
        z, Z, e, curves, slabs = self._profile
        p = self.prof_plot
        show_bars = self.prof_show_bars.isChecked()

        for i, cb in enumerate(self.prof_boxes):
            on = cb.isChecked()
            self.prof_curves[i].setData(z, curves[i])
            self.prof_curves[i].setVisible(on)
            self.prof_bars[i].setOpts(x0=e[:-1], width=np.diff(e),
                                      height=slabs[i], y0=0)
            self.prof_bars[i].setVisible(on and show_bars)
        angle_on = self.prof_boxes[2].isChecked()
        p.showAxis('right', angle_on)
        p.showAxis('left', any(cb.isChecked() for cb in self.prof_boxes[:2]))

        for item in self.prof_decor + self.prof_bands + self.prof_labels:
            p.removeItem(item)
        self.prof_decor, self.prof_bands, self.prof_labels = [], [], []
        line = pg.mkPen('#888888', width=1)
        for Zj in Z:
            ln = pg.InfiniteLine(pos=Zj, angle=90, pen=line)
            ln.setZValue(-50)
            p.addItem(ln)
            self.prof_decor.append(ln)
        # layer j spans [bounds[j], bounds[j+1]]; the semi-infinite media
        # run to the edges of the plotted range
        bounds = np.concatenate([[z[0]], Z, [z[-1]]])
        n = len(self.stack.layers)
        for j, l in enumerate(self.stack.layers):
            band = pg.LinearRegionItem((bounds[j], bounds[j + 1]),
                                       movable=False, pen=pg.mkPen(None))
            band.setZValue(-100)
            p.addItem(band, ignoreBounds=True)
            self.prof_bands.append(band)
            text = l.name if j in (0, n - 1) else \
                '%s  (%.4g Å)' % (l.name, l.thickness)
            t = pg.TextItem(text, anchor=(0.5, 0))
            t.setPos(0.5 * (bounds[j] + bounds[j + 1]), 0)
            p.addItem(t, ignoreBounds=True)
            self.prof_labels.append(t)
        self._style_bands()
        p.setXRange(z[0], z[-1], padding=0)
        p.enableAutoRange(axis='y')
        self.prof_vb2.enableAutoRange(axis='y')
        self._place_layer_labels()

    def _style_bands(self):
        """Alternate grey shading per layer; bold the selected layer's name."""
        sel = self.list.currentRow()
        for j, (band, t) in enumerate(zip(self.prof_bands, self.prof_labels)):
            brush = pg.mkBrush(BANDS[j % 2])
            band.setBrush(brush)
            band.setHoverBrush(brush)
            font = t.textItem.font()
            font.setBold(j == sel)
            t.setFont(font)
            t.setColor('#222222' if j == sel else '#666666')

    def _place_layer_labels(self, *_):
        ytop = self.prof_plot.vb.viewRange()[1][1]
        for t in self.prof_labels:
            t.setPos(t.pos().x(), ytop)

    def _on_profile_click(self, ev):
        """Clicking inside a layer's band selects it in the list."""
        vb = self.prof_plot.vb
        if self._profile is None or ev.double() or \
                not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = vb.mapSceneToView(ev.scenePos()).x()
        self.list.setCurrentRow(int(np.searchsorted(self._profile[1], x)))

    def _profile_readout(self, x):
        """Cursor text for the depth profile: layer, depth, checked curves."""
        if self._profile is None:
            return None, []
        z, Z, e, curves, slabs = self._profile
        layer = self.stack.layers[int(np.searchsorted(Z, x))]
        rows = []
        for i, (name, _, col) in enumerate(PROFILE_QUANTITIES):
            if not self.prof_boxes[i].isChecked():
                continue
            y = np.interp(x, z, curves[i], left=np.nan, right=np.nan)
            unit = ' °' if i == 2 else ''
            if self.prof_show_bars.isChecked():
                k = min(max(int(np.searchsorted(e, x)) - 1, 0), len(slabs[i]) - 1)
                value = '%s%s  (sublayer %s)' % (fmt(y), unit, fmt(slabs[i][k]))
            else:
                value = fmt(y) + unit
            vb = self.prof_vb2 if i == 2 else self.prof_plot.vb
            rows.append((name, value, col, y, vb))
        return 'z = %.4g Å  —  %s' % (x, layer.name), rows

    def _refl_readout(self, x):
        """Cursor text for the reflectivity plot: Q and each visible curve."""
        rows = []
        for curve, name, Q, y in self._refl_shown:
            if not curve.isVisible() or not (Q[0] <= x <= Q[-1]):
                continue
            v = np.interp(x, Q, y)
            if self._refl_log:
                yv = np.log10(v) if v > 0 else np.nan
            else:
                yv = v
            col = curve.opts['pen'].color().name()
            rows.append((name, fmt(v), col, yv, self.refl_plot.vb))
        if not self._refl_shown:
            return None, []
        return 'Q = %.5g Å⁻¹' % x, rows

    def _draw_reflectance(self):
        qmin, qmax = self.qmin.value(), self.qmax.value()
        if qmax <= qmin:
            raise ValueError('Q max must exceed Q min')
        Q = np.linspace(qmin, qmax, self.nq.value())
        R = self.stack.calc_reflectance(Q)
        off = 4 if self.basis.currentIndex() == 0 else 0
        self.alpha.setEnabled(bool(off))            # α only mixes lab channels
        k = self.refl_quantity.currentIndex()
        name, label, formula = REFL_QUANTITIES[k]
        is_refl = label is None
        # log / R·Q⁴ only apply to raw reflectivities
        for w in (self.logy, self.rq4):
            w.setEnabled(is_refl)

        uu, ud, du, dd = R[:, off:off + 4].T
        if k == 0:
            shown = zip(self.refl_curves, (uu, ud, du, dd),
                        ['R' + c for c in CHANNELS] if off else
                        ['R_uu', 'R_ud', 'R_du', 'R_dd'])
        elif is_refl:
            shown = zip(self.hp_curves, (uu + ud, dd + du),
                        ['R+', 'R−'] if off else ['R_u', 'R_d'])
        else:
            shown = [(self.asym_curve, refl_quantity(name, (uu, ud, du, dd)),
                      name)]
        shown = list(shown)

        p = self.refl_plot
        legend = p.legend
        legend.clear()
        # clear unused curves rather than hide them, so visibility toggled
        # from the legend survives a change of plotted quantity
        used = {id(c) for c, _, _ in shown}
        for curve in self.refl_curves + self.hp_curves + [self.asym_curve]:
            if id(curve) not in used:
                curve.setData([], [])
        self.asym_zero.setVisible(not is_refl)
        self.refl_formula.setText(formula)

        if is_refl:
            log = self.logy.isChecked()
            p.setLabel('left', 'R·Q⁴ (Å⁻⁴)' if self.rq4.isChecked()
                       else 'Reflectivity')
        else:
            log = False
            p.setLabel('left', label)
        p.setLogMode(x=False, y=log)
        self._refl_log = log
        self._refl_shown = []
        # channel visibility is toggled by clicking the legend entries
        for curve, y, cname in shown:
            if is_refl and self.rq4.isChecked():
                y = y * Q**4
            if log:
                y = np.where(y > 0, y, np.nan)
            curve.setData(Q, y)
            legend.addItem(curve, cname)
            self._refl_shown.append((curve, cname, Q, y))
        p.enableAutoRange()

    # -- save / load --------------------------------------------------------
    def save_model(self, path):
        d = self.stack.to_dict()
        d['simulation'] = {'qmin': self.qmin.value(), 'qmax': self.qmax.value(),
                           'nq': self.nq.value(),
                           'basis': self.basis.currentIndex()}
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    def load_model(self, path):
        with open(path) as f:
            d = json.load(f)
        new = Stack.from_dict(d)                 # validate before touching UI
        self.stack.layers = new.layers
        self.stack.tail, self.stack.alpha = new.tail, new.alpha
        sim = d.get('simulation', {})
        widgets = [self.qmin, self.qmax, self.nq, self.tail, self.alpha,
                   self.basis]
        for w in widgets:
            w.blockSignals(True)
        self.qmin.setValue(sim.get('qmin', self.qmin.value()))
        self.qmax.setValue(sim.get('qmax', self.qmax.value()))
        self.nq.setValue(sim.get('nq', self.nq.value()))
        self.basis.setCurrentIndex(sim.get('basis', self.basis.currentIndex()))
        self.tail.setValue(new.tail)
        self.alpha.setValue(np.degrees(new.alpha))
        for w in widgets:
            w.blockSignals(False)
        self.refresh_list(select=1)
        self.recompute()
