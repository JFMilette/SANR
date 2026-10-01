"""
Simulation tab: build a Stack by hand and look at its profile and reflectivity.

    left  : general parameters beside the layer list (add / remove /
            reorder), property editor of the selected layer below
    right : depth profile on top, reflectivity below, both wide.
            The profile overlays NSLD / MSLD |M| (left axis) and the
            magnetic angles θ, φ (right axis), each toggled by a checkbox:
            solid line = Stack.profile(z), bars = Stack.build_sublayers(),
            shaded bands = layers (selected one's name in bold, click a band
            to select it), grey verticals = nominal interfaces.

save_model / load_model write / read the stack and Q settings as JSON.
The Polarisation box links each channel R++, R+-, R-+, R--, R+, R- to an
incident polarisation Pi and an analysed polarisation Pa (3-vectors in the
sample frame, length = efficiency, Pa = 0 for no analyser); the plot and
every dataset that does not set its own use these pairs.

Every edit triggers a debounced recompute.
"""

import html
import re
import warnings

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from model.polarisation import (CHANNEL_NAMES, default_vectors, in_plane,
                                vectors_pair)
from model.stack import (DEFAULT_SMEARING, ROUGHNESS_SCHEMES,
                         SMEARING_MODES, Layer, Stack, cos_turns, wrap_turns)


SLD_SCALE = 1e-6                      # SLDs are edited / plotted in 1e-6 A^-2
CHANNELS = ['++', '+-', '-+', '--']
AUTO_NAME = re.compile(r'L\d+')        # default layer names, renumbered
CH_COLOURS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
HP_COLOURS = ['#1f77b4', '#ff7f0e']    # half-polarized R↑, R↓
# combo label, axis label, colour
PROFILE_QUANTITIES = [
    ('NSLD', 'NSLD real (10⁻⁶ Å⁻²)', '#5b9bd5'),
    ('MSLD ρ', 'MSLD ρ = |M| (10⁻⁶ Å⁻²)', '#4fc08d'),
    ('θ', 'MSLD θ, in plane (deg)', '#e0a458'),
    ('φ', 'MSLD φ, out of plane (deg)', '#e36fa8'),
]
RIGHT = (2, 3)                        # PROFILE_QUANTITIES on the angle axis
# combo label, axis label (None = raw reflectivities), formula beside the combo.
# The channels are the Pi / Pa pairs of the Polarisation box.
REFL_QUANTITIES = [
    ('Reflectivity', None, ''),
    ('Half-polarized', None, 'R+, R− (Pa = 0: no analyser)'),
    ('SA NSF', 'SA_NSF', '(R++ − R−−) / (R++ + R−−)'),
    ('SF fraction', 'SF fraction', '(R+− + R−+) / (R++ + R+− + R−+ + R−−)'),
    ('SA SF', 'SA_SF', '(R+− − R−+) / (R+− + R−+)'),
    ('SA half-polarized', 'SA', '(R+ − R−) / (R+ + R−)'),
]
ASYM_COLOUR = '#b392f0'
# smallest font for the layer names in the depth profile
LABEL_MIN_PT = 6
# layer shading in the depth profile: alternating greys
BANDS = ['#00000000', '#ffffff0c']


def ratio(a, b):
    """a / b, NaN where b is not positive."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b > 0, a / b, np.nan)


def refl_quantity(name, R):
    """Asymmetry `name` of REFL_QUANTITIES from R = {channel: array}."""
    uu, ud, du, dd = (R[c] for c in CHANNELS)
    if name == 'SA NSF':
        return ratio(uu - dd, uu + dd)
    if name == 'SF fraction':
        return ratio(ud + du, uu + ud + du + dd)
    if name == 'SA SF':
        return ratio(ud - du, ud + du)
    return ratio(R['+'] - R['-'], R['+'] + R['-'])


def profile_data(st):
    """Depth profile of a stack with built sublayers, for plotting:
    ((z, Z, e, curves, slabs), on_bars).  curves are the continuous
    PROFILE_QUANTITIES at depths z, slabs their value in each slab between
    the edges e (fronting and backing included, out to the plotted range),
    Z the interfaces; on_bars[i] is angle curve i restricted to the slabs
    that have that angle."""
    Z = st._interfaces()
    lo = st.windows()[0][0]
    e = lo + np.concatenate([[0.0], np.cumsum([s.thickness
                                               for s in st.sublayers])])
    pad = 0.35 * max(Z[-1], 1.0)
    z = np.linspace(min(e[0], Z[0] - pad) - 10,
                    max(e[-1], Z[-1] + pad) + 10, 4000)
    nsld, rho, theta, phi = st.profile(z)
    # no angle where there is no magnetisation to point
    # (st.angle_defined): theta needs an in-plane part, phi any |M|
    mp = np.abs(rho * cos_turns(phi))
    curves = [nsld.real / SLD_SCALE, rho / SLD_SCALE,
              np.where(st.angle_defined(mp), theta * 360.0, np.nan),
              np.where(st.angle_defined(rho), phi * 360.0, np.nan)]
    angles_all = {2: theta * 360.0, 3: phi * 360.0}
    # fronting and backing are semi-infinite: draw them as one bar each
    # out to the edges of the plotted range
    e = np.concatenate([[z[0]], e, [z[-1]]])
    slabs_all = [st.fronting] + list(st.sublayers) + [st.backing]
    slab_rho = np.array([s.MSLD_rho for s in slabs_all])
    slab_mp = np.abs([Stack.inplane_rho(s) for s in slabs_all])
    slab_theta = wrap_turns([s.MSLD_theta for s in slabs_all]) * 360.0
    slab_phi = np.array([s.MSLD_phi for s in slabs_all]) * 360.0
    slabs = [np.array([s.NSLD_real / SLD_SCALE for s in slabs_all]),
             slab_rho / SLD_SCALE,
             np.where(st.angle_defined(slab_mp), slab_theta, np.nan),
             np.where(st.angle_defined(slab_rho), slab_phi, np.nan)]
    # with the bars shown, an angle curve is drawn over exactly the bars
    # that have that angle, so both start and stop at the same depths
    k = np.clip(np.searchsorted(e, z, side='right') - 1, 0, len(e) - 2)
    on_bars = {i: np.where(np.isfinite(slabs[i][k]), angles_all[i], np.nan)
               for i in RIGHT}
    return (z, Z, e, curves, slabs), on_bars


def make_stack():
    """Default sample loaded when the tab opens."""
    vacuum = Layer('vacuum')

    L1 = Layer('L1', thickness=80.0, NSLD_real=4.0e-6,
               MSLD_rho=1.5e-6, MSLD_theta=0.75,
               roughness_sigma=8.0, roughness_model='tanh', roughness_sublayer=20)

    L2 = Layer('L2', thickness=120.0, NSLD_real=1.5e-6, NSLD_img=-2.0e-8,
               MSLD_rho=0.8e-6, MSLD_theta=0.5,
               roughness_sigma=5.0, roughness_model='tanh', roughness_sublayer=20)

    L3 = Layer('L3', thickness=60.0, NSLD_real=6.0e-6,
               MSLD_rho=0.0, MSLD_theta=0.0,
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


class SciSpinBox(QtWidgets.QDoubleSpinBox):
    """Non-negative spin box shown in %g notation (e.g. 1e-07); the arrows
    step logarithmically, 10 steps per decade, and down from the smallest
    step goes to 0."""

    MIN_STEP = 1e-10

    def __init__(self, hi=1.0):
        super().__init__()
        self.setDecimals(20)             # keep full precision internally
        self.setRange(0.0, hi)
        self.setKeyboardTracking(False)

    def textFromValue(self, v):
        return '%g' % v

    def sizeHint(self):
        # Qt sizes from textFromValue(min / max) = '0' / '1'; size for the
        # widest text actually shown instead
        h = super().sizeHint()
        extra = self.fontMetrics().horizontalAdvance('-8.88888e-88') - \
            self.fontMetrics().horizontalAdvance('%g' % self.maximum())
        return QtCore.QSize(h.width() + max(extra, 0), h.height())

    def minimumSizeHint(self):
        return self.sizeHint()

    def valueFromText(self, text):
        return float(text)

    def validate(self, text, pos):
        State = QtGui.QValidator.State
        t = text.strip()
        try:
            v = float(t)
        except ValueError:
            # partial input such as '1e' or '1e-' is still being typed
            ok = re.fullmatch(r'[0-9]*\.?[0-9]*([eE][+-]?)?', t) is not None
            return State.Intermediate if ok else State.Invalid, text, pos
        ok = self.minimum() <= v <= self.maximum()
        return State.Acceptable if ok else State.Intermediate, text, pos

    def stepBy(self, n):
        v = self.value()
        if v <= 0:
            v = self.MIN_STEP if n > 0 else 0.0
        else:
            v *= 10 ** (0.1 * n)
            if v < self.MIN_STEP:
                v = 0.0
        self.setValue(min(v, self.maximum()))


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
            '#9aa0a6', width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self.text = pg.TextItem(anchor=(0, 1), color='#e3e5e8',
                                fill=pg.mkBrush(43, 45, 49, 235),
                                border=pg.mkPen('#55585f'))
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
            m = pg.ScatterPlotItem(size=8, pen=pg.mkPen('#1b1c1f', width=1))
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


class LayerLabels:
    """A shaded band and a name (with the thickness) per layer on a depth
    profile plot.  selected() is the row drawn in bold (-1 for none)."""

    def __init__(self, plot, selected=lambda: -1):
        self.plot, self.selected = plot, selected
        self.bands = []                      # one shaded region per layer
        self.labels = []                     # layer names, pinned to the top
        self.spans = []                      # (z0, z1, texts) of each label
        plot.vb.sigYRangeChanged.connect(self.place)
        plot.vb.sigXRangeChanged.connect(self.fit)
        plot.vb.sigResized.connect(self.fit)

    def set_layers(self, layers, Z, zlo, zhi):
        """One band per layer between the interfaces Z; the semi-infinite
        media run out to zlo / zhi, the edges of the plotted range."""
        p = self.plot
        for item in self.bands + self.labels:
            p.removeItem(item)
        self.bands, self.labels, self.spans = [], [], []
        # layer j spans [bounds[j], bounds[j+1]]
        bounds = np.concatenate([[zlo], Z, [zhi]])
        n = len(layers)
        for j, l in enumerate(layers):
            band = pg.LinearRegionItem((bounds[j], bounds[j + 1]),
                                       movable=False, pen=pg.mkPen(None))
            band.setZValue(-100)
            p.addItem(band, ignoreBounds=True)
            self.bands.append(band)
            # thickness goes on a second line so crowded stacks stay legible
            name = html.escape(l.name)
            texts = [name] if j in (0, n - 1) else \
                ['%s<br>%.4g Å' % (name, l.thickness), name]
            texts = ['<div align="center">%s</div>' % x for x in texts]
            t = pg.TextItem(html=texts[0], anchor=(0.5, 0))
            t.setPos(0.5 * (bounds[j] + bounds[j + 1]), 0)
            p.addItem(t, ignoreBounds=True)
            self.labels.append(t)
            self.spans.append((bounds[j], bounds[j + 1], texts))
        self.style()

    def style(self):
        """Alternate grey shading per layer; bold the selected layer's name."""
        sel = self.selected()
        for j, (band, t) in enumerate(zip(self.bands, self.labels)):
            brush = pg.mkBrush(BANDS[j % 2])
            band.setBrush(brush)
            band.setHoverBrush(brush)
            font = t.textItem.font()
            font.setBold(j == sel)
            t.setFont(font)
            t.setColor('#e3e5e8' if j == sel else '#9aa0a6')
        self.fit()

    def fit(self, *_):
        """Give every layer label the same size and layout: the largest that
        fits the narrowest band on screen, down to LABEL_MIN_PT, dropping the
        thickness line first if needed.  Bands too narrow even for that hide
        their label (except the selected layer's)."""
        labels, spans = self.labels, self.spans
        if not labels:
            return
        vb = self.plot.vb
        (x0, x1), _ = vb.viewRange()
        px = vb.width() / (x1 - x0) if x1 > x0 else 0   # pixels per Å
        base = QtGui.QFont().pointSizeF()
        sel = self.selected()
        margin = 2 * labels[0].textItem.document().documentMargin()

        def measure(t, text):
            """Text width at the base size, bold, without the margins."""
            font = t.textItem.font()
            font.setPointSizeF(base)
            font.setBold(True)
            t.setFont(font)
            t.setHtml(text)
            t.textItem.setTextWidth(-1)
            return t.textItem.document().idealWidth() - margin

        # per label: room on screen and width of each variant (0 = name +
        # thickness, 1 = name only; the media only have the name)
        rooms, widths = [], []
        for t, (z0, z1, texts) in zip(labels, spans):
            rooms.append((min(z1, x1) - max(z0, x0)) * px - 4 - margin)
            widths.append([measure(t, x) for x in texts])

        # the inner layers are all measured against the widest inner label,
        # so whether a label fits depends on its band width, not its name
        n = len(labels)
        widest = [max((widths[j][k] for j in range(1, n - 1)), default=0)
                  for k in range(2)]

        def fits(j, k):
            """Size at which variant k of label j fills its band."""
            w = widths[j][0] if j in (0, n - 1) else widest[k]
            return base * min(1.0, 0.95 * rooms[j] / w) if w > 0 else base

        # the inner layers set the common size; on screen ones only
        inner = [j for j in range(1, n - 1) if rooms[j] > 0] \
            or [j for j in range(n) if rooms[j] > 0]
        for k in range(2):
            size = min((fits(j, k) for j in inner), default=base)
            if size >= LABEL_MIN_PT:
                break
        size = max(size, LABEL_MIN_PT)

        for j, (t, (_, _, texts)) in enumerate(zip(labels, spans)):
            font = t.textItem.font()
            font.setPointSizeF(size)
            font.setBold(j == sel)
            t.setFont(font)
            t.setHtml(texts[min(k, len(texts) - 1)])
            # centring needs a fixed text width; refit it after the font change
            t.textItem.setTextWidth(-1)
            t.setTextWidth(t.textItem.document().idealWidth())
            t.setVisible(bool(fits(j, k) >= size * 0.999) or j == sel)

    def place(self, *_):
        """Pin the labels to the top of the view."""
        ytop = self.plot.vb.viewRange()[1][1]
        for t in self.labels:
            t.setPos(t.pos().x(), ytop)


# ------------------------------------------------------- layer editor ----
# attr, label, lo, hi, decimals, step, suffix, display = stored * scale
EDITOR_PARAMS = [
    ('thickness', 'Thickness (Å)', 0.0, 1e5, 4, 1.0, '', 1.0),
    ('NSLD_real', 'NSLD real', -100, 100, 6, 0.1, '', 1 / SLD_SCALE),
    ('NSLD_img', 'NSLD imag', -100, 100, 7, 0.001, '', 1 / SLD_SCALE),
    ('MSLD_rho', 'MSLD ρ', 0, 100, 6, 0.1, '', 1 / SLD_SCALE),
    ('MSLD_theta', 'MSLD θ', -360, 360, 4, 5.0, ' °', 360.0),
    ('MSLD_phi', 'MSLD φ', -90, 90, 4, 5.0, ' °', 360.0),
    ('roughness_sigma', 'Roughness σ', 0.0, 1e4, 4, 0.5, ' Å', 1.0),
]
OUT_OF_BOUNDS = 'QDoubleSpinBox { background: #4a1f22; border: 1px solid #f87171 }'
RES_MODES = ('mono', 'tof')     # res_mode combo order
# instrument setups of the fixed-angle mode: name -> (dlambda (A), angles)
RES_PRESETS = {
    'Mono (3 angles)': (0.005, [{'theta': 0.006, 'dtheta': 3e-4, 'qmax': 0.04},
                                {'theta': 0.010, 'dtheta': 5e-4, 'qmax': 0.12},
                                {'theta': 0.017, 'dtheta': 5e-4,
                                 'qmax': None}]),
    'ToF (1 angle)': (0.01, [{'theta': 0.010, 'dtheta': 3e-4, 'qmax': None}]),
}
SLIDER_STEPS = 1000             # slider resolution across [Min, Max]
NSUB_MAX = 1000                 # largest sublayer count
NSUB_DEFAULT_MAX = 50           # default top of the sublayer slider


PAIR_TIP = ('Pi: incident polarisation, Pa: analysed polarisation, as '
            '(x, y, z) in the sample frame\n(x, y in the film plane, z = film '
            'normal = Q).  |P| is the efficiency (≤ 1).\nPa = (0, 0, 0) means '
            'no analyser: every reflected spin is counted (R+, R−).\nOnly the '
            'in-plane part along a magnetic fronting\'s M survives in it.')
PAIR_COLUMNS = ['Pi x', 'Pi y', 'Pi z', 'Pa x', 'Pa y', 'Pa z']


class PairTable(QtWidgets.QTableWidget):
    """Incident (Pi) and analysed (Pa) polarisation of named channels, three
    components each (see PAIR_TIP).  With follow=True each row starts with a
    check box: ticked = use the Simulation tab's pair (vectors() gives None
    for that row)."""

    changed = QtCore.pyqtSignal()

    def __init__(self, names, follow=False):
        super().__init__(len(names), len(PAIR_COLUMNS) + bool(follow))
        self.names, self.follow = list(names), bool(follow)
        self._fallback = default_vectors()
        off = int(self.follow)
        self.setHorizontalHeaderLabels(
            (['Same as Simulation'] if follow else []) + PAIR_COLUMNS)
        self.setVerticalHeaderLabels(['R' + n for n in self.names])
        self.setToolTip(PAIR_TIP)
        self.spins, self.checks = [], []
        # values filled in by set_vectors, returned exactly (the boxes show
        # 4 decimals) until that box is edited: cos 0.7 stays cos 0.7
        self._exact = [[None] * len(PAIR_COLUMNS) for _ in self.names]
        for r in range(len(self.names)):
            row = []
            for c in range(len(PAIR_COLUMNS)):
                sb = dspin(-1, 1, 4, 0.1)
                sb.setButtonSymbols(
                    QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                sb.setFrame(False)
                sb.valueChanged.connect(
                    lambda _, r=r, c=c: self._edited(r, c))
                self.setCellWidget(r, c + off, sb)
                row.append(sb)
            self.spins.append(row)
            if follow:
                cb = QtWidgets.QCheckBox()
                cb.toggled.connect(lambda on, r=r: self._follow(r, on))
                self.setCellWidget(r, 0, cb)
                self.checks.append(cb)
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        if follow:
            self.horizontalHeader().setSectionResizeMode(
                0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setDefaultSectionSize(
            self.fontMetrics().height() + 8)
        self.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(self.horizontalHeader().sizeHint().height()
                            + len(self.names)
                            * self.verticalHeader().defaultSectionSize() + 4)
        self.set_vectors(default_vectors())

    def _edited(self, r, c):
        self._exact[r][c] = None
        self._check()
        self.changed.emit()

    def _row(self, r):
        return [w.value() if x is None else x
                for w, x in zip(self.spins[r], self._exact[r])]

    def _check(self):
        """Red for a vector longer than 1 (not a physical polarisation)."""
        for r, row in enumerate(self.spins):
            v = self._row(r)
            for vec, val in ((row[:3], v[:3]), (row[3:], v[3:])):
                bad = np.linalg.norm(val) > 1 + 1e-9
                for w in vec:
                    w.setStyleSheet(OUT_OF_BOUNDS if bad else '')
                    w.setToolTip('|P| > 1' if bad else '')

    def _follow(self, r, on):
        for w in self.spins[r]:
            w.setEnabled(not on)
        if on:
            self._fill(r, self._fallback[self.names[r]])
        self.changed.emit()

    def _fill(self, r, vec):
        vals = [float(v) for v in list(vec[0]) + list(vec[1])]
        for w, v in zip(self.spins[r], vals):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self._exact[r] = vals

    def set_vectors(self, vectors, fallback=None):
        """vectors: {channel: [Pi, Pa] or None (follow)}; fallback: the
        Simulation tab's vectors, shown on followed rows."""
        if fallback is not None:
            self._fallback = fallback
        for r, n in enumerate(self.names):
            vec = vectors.get(n)
            if self.follow:
                self.checks[r].blockSignals(True)
                self.checks[r].setChecked(vec is None)
                self.checks[r].blockSignals(False)
                for w in self.spins[r]:
                    w.setEnabled(vec is not None)
            self._fill(r, self._fallback[n] if vec is None else vec)
        self._check()
        self.changed.emit()

    def vectors(self):
        """{channel: [Pi, Pa]}, None for a followed row."""
        out = {}
        for r, n in enumerate(self.names):
            if self.follow and self.checks[r].isChecked():
                out[n] = None
            else:
                v = self._row(r)
                out[n] = [v[:3], v[3:]]
        return out


class LayerEditor(QtWidgets.QGroupBox):
    """Form bound to one Layer; emits `changed` after writing into it.

    Each fittable parameter has its value, fit bounds and a Fit check box
    (Layer.fit); the bounds are only editable while Fit is checked, and a
    varied value outside its bounds is shown in red. The slider under each
    row sweeps the value between Min and Max."""

    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__('Layer properties')
        self.layer = None
        self._loading = False
        self._sliding = False           # value change coming from a slider

        self.name = QtWidgets.QLineEdit()
        self.model = QtWidgets.QComboBox()
        self.model.addItems(['tanh', 'erf', 'none'])
        self.nsub = QtWidgets.QSpinBox()
        self.nsub.setRange(1, NSUB_MAX)
        self.nsub.setKeyboardTracking(False)
        # slider span only: an integer count is never fitted
        self.nsub_min, self.nsub_max = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for b in (self.nsub_min, self.nsub_max):
            b.setRange(1, NSUB_MAX)
            b.setKeyboardTracking(False)
            b.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            b.valueChanged.connect(self._sync_nsub_slider)
        self.nsub_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.nsub_slider.setToolTip('Sweep the sublayer count between Min '
                                    'and Max')
        self.nsub_slider.valueChanged.connect(self.nsub.setValue)
        self.nsub.valueChanged.connect(self._sync_nsub_slider)

        # attr -> (value, min, max, fit check box, row label)
        self.rows = {}
        for attr, label, lo, hi, dec, step, suffix, scale in EDITOR_PARAMS:
            val = dspin(lo, hi, dec, step, suffix)
            bmin, bmax = dspin(lo, hi, dec, step), dspin(lo, hi, dec, step)
            for b in (bmin, bmax):   # bounds rarely need arrows; save width
                b.setButtonSymbols(
                    QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            cb = QtWidgets.QCheckBox()
            cb.setToolTip('Vary %s in the fit, between Min and Max' % label)
            self.rows[attr] = (val, bmin, bmax, cb, QtWidgets.QLabel(label))
        self.sliders = {}
        for attr, *_ in EDITOR_PARAMS:
            sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sl.setRange(0, SLIDER_STEPS)
            sl.setToolTip('Sweep the value between Min and Max')
            sl.valueChanged.connect(
                lambda pos, a=attr: self._slider_moved(a, pos))
            self.sliders[attr] = sl
        self.rows['MSLD_theta'][4].setToolTip(
            'θ_M: in-plane angle of M from the sample x axis towards y '
            '(Majkrzak Fig. 1.14)')
        self.rows['MSLD_rho'][4].setToolTip(
            'ρ_M = |M|, including any out-of-plane part')
        self.rows['MSLD_phi'][4].setToolTip(
            'φ_M: elevation of M out of the film plane (+90° = +z). Only '
            'ρ cos φ is seen by neutrons (Halperin)')

        g = QtWidgets.QGridLayout(self)
        g.setColumnStretch(1, 1)
        g.setVerticalSpacing(2)
        g.addWidget(QtWidgets.QLabel('Name'), 0, 0)
        g.addWidget(self.name, 0, 1, 1, 4)
        for c, h in enumerate(['', 'Value', 'Min', 'Max', 'Fit']):
            lab = QtWidgets.QLabel('<b>%s</b>' % h)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            g.addWidget(lab, 1, c)
        g.addWidget(QtWidgets.QLabel('<i>SLDs in 10⁻⁶ Å⁻²</i>'), 2, 0, 1, 5)
        r = 3
        for attr, *_ in EDITOR_PARAMS:
            if attr == 'roughness_sigma':
                g.addWidget(QtWidgets.QLabel(
                    '<i>Interface at top of this layer</i>'), r, 0, 1, 5)
                r += 1
            val, bmin, bmax, cb, lab = self.rows[attr]
            for c, w in enumerate((lab, val, bmin, bmax)):
                g.addWidget(w, r, c)
            g.addWidget(cb, r, 4, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            g.addWidget(self.sliders[attr], r + 1, 1, 1, 3)
            r += 2
        g.addWidget(QtWidgets.QLabel('Roughness model'), r, 0)
        g.addWidget(self.model, r, 1)
        g.addWidget(QtWidgets.QLabel('Sublayers'), r + 1, 0)
        g.addWidget(self.nsub, r + 1, 1)
        g.addWidget(self.nsub_min, r + 1, 2)
        g.addWidget(self.nsub_max, r + 1, 3)
        g.addWidget(self.nsub_slider, r + 2, 1, 1, 3)

        self.name.editingFinished.connect(self._store)
        for attr, (val, bmin, bmax, cb, _) in self.rows.items():
            for w in (val, bmin, bmax):
                w.valueChanged.connect(self._store)
                w.valueChanged.connect(
                    lambda _, a=attr: self._sync_slider(a))
            cb.toggled.connect(self._store)
        for w in (self.nsub, self.nsub_min, self.nsub_max):
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
        for attr, *_, scale in EDITOR_PARAMS:
            val, bmin, bmax, cb, _ = self.rows[attr]
            e = layer.fit_entry(attr)
            val.setValue(getattr(layer, attr) * scale)
            bmin.setValue(e['min'] * scale)
            bmax.setValue(e['max'] * scale)
            cb.setChecked(e['vary'])
        self.model.setCurrentText(layer.roughness_model)
        n = int(layer.roughness_sublayer)
        e = layer.fit.get('roughness_sublayer') or \
            {'min': 1, 'max': NSUB_DEFAULT_MAX}
        self.nsub_min.setValue(int(e['min']))
        self.nsub_max.setValue(int(e['max']))
        self.nsub.setValue(n)
        self._sync_nsub_slider()
        self._loading = False
        for attr in self.sliders:
            self._sync_slider(attr)
        # semi-infinite media have no thickness; fronting has no top interface
        self._set_row_enabled('thickness', not (is_fronting or is_backing))
        self._set_row_enabled('roughness_sigma', not is_fronting)
        for w in (self.model, self.nsub, self.nsub_min, self.nsub_max,
                  self.nsub_slider):
            w.setEnabled(not is_fronting)
        self._check_bounds()

    def _set_row_enabled(self, attr, on):
        for w in self.rows[attr]:
            w.setEnabled(on)

    def _sync_nsub_slider(self, *_):
        """Span the sublayer slider over [Min, Max] and place it at the
        count, clamped to the ends."""
        sl = self.nsub_slider
        sl.blockSignals(True)
        sl.setRange(self.nsub_min.value(),
                    max(self.nsub_min.value(), self.nsub_max.value()))
        sl.setValue(self.nsub.value())
        sl.blockSignals(False)

    def _slider_moved(self, attr, pos):
        val, bmin, bmax, *_ = self.rows[attr]
        self._sliding = True
        val.setValue(bmin.value() + pos / SLIDER_STEPS *
                     (bmax.value() - bmin.value()))
        self._sliding = False

    def _sync_slider(self, attr):
        """Put the slider of `attr` where its value sits in [Min, Max]."""
        if self._sliding:
            return
        val, bmin, bmax, *_ = self.rows[attr]
        span = bmax.value() - bmin.value()
        pos = (val.value() - bmin.value()) / span if span > 0 else 0.0
        sl = self.sliders[attr]
        sl.blockSignals(True)
        sl.setValue(round(min(max(pos, 0.0), 1.0) * SLIDER_STEPS))
        sl.blockSignals(False)

    def _check_bounds(self):
        """Flag varied values that lie outside [Min, Max] or an empty range.
        The bounds stay editable when the parameter is fixed (they also span
        its slider)."""
        for attr, (val, bmin, bmax, cb, lab) in self.rows.items():
            row_on = lab.isEnabled()
            vary = cb.isChecked()
            bmin.setEnabled(row_on)
            bmax.setEnabled(row_on)
            self.sliders[attr].setEnabled(
                row_on and bmin.value() < bmax.value())
            bad = row_on and vary and not (
                bmin.value() <= val.value() <= bmax.value()
                and bmin.value() < bmax.value())
            val.setStyleSheet(OUT_OF_BOUNDS if bad else '')
            val.setToolTip('Outside the fit bounds' if bad else '')

    def _store(self):
        if self._loading or self.layer is None:
            return
        L = self.layer
        L.name = self.name.text()
        for attr, *_, scale in EDITOR_PARAMS:
            val, bmin, bmax, cb, _ = self.rows[attr]
            setattr(L, attr, val.value() / scale)
            L.fit[attr] = {'vary': cb.isChecked(),
                           'min': bmin.value() / scale,
                           'max': bmax.value() / scale}
        L.roughness_model = self.model.currentText()
        L.roughness_sublayer = self.nsub.value()
        L.fit['roughness_sublayer'] = {'vary': False,
                                       'min': self.nsub_min.value(),
                                       'max': self.nsub_max.value()}
        self._check_bounds()
        self.changed.emit()


# ----------------------------------------------------- simulation tab ----
class SimulationTab(QtWidgets.QWidget):

    # emitted after every successful recompute; read last_Q / last_R
    reflectanceChanged = QtCore.pyqtSignal()

    def __init__(self, stack=None, parent=None):
        super().__init__(parent)
        self.stack = stack if stack is not None else make_stack()
        self.last_Q = self.last_R = None     # latest Q, {channel: R}
        self._refl_note = ''                 # warning of the last reflectivity
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
        splitter.setSizes([460, 1140])

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
        self.qmin = dspin(1e-5, 10, 5, 0.001)
        self.qmax = dspin(1e-4, 10, 4, 0.01)
        self.nq = QtWidgets.QSpinBox()
        self.nq.setRange(2, 100000)
        self.nq.setKeyboardTracking(False)
        self.tail = dspin(0.5, 20, 2, 0.5, ' σ')
        self.qmin.setValue(1e-4)
        self.qmax.setValue(0.25)
        self.nq.setValue(400)
        self.tail.setValue(self.stack.tail)

        self.background = SciSpinBox()
        self.background.setValue(self.stack.background)
        self.background.setToolTip('Constant background added to every '
                                   'reflectivity channel (e.g. 1e-7)')

        self.q_fronting = QtWidgets.QCheckBox('Q in fronting')
        self.q_fronting.setChecked(self.stack.q_in_fronting)
        self.q_fronting.setToolTip(
            'Checked: Q is measured inside the fronting medium (beam enters '
            'through the substrate side), so each incident spin has its own '
            'vacuum Q.\nUnchecked: Q is the vacuum-referenced 2k0z, common to '
            'both spins.')

        self.msmear = QtWidgets.QComboBox()
        self.msmear.addItems(['vector (roughness)', 'angle (twist)',
                              'step (Licorne)'])
        self.msmear.setCurrentIndex(
            SMEARING_MODES.index(self.stack.magnetic_smearing))
        self.msmear.setToolTip(
            'How the magnetisation crosses a rough interface.\n'
            'vector: the components ρ cos θ, ρ sin θ are smeared (lateral '
            'average of a rough interface);\n  a non-magnetic neighbour only '
            'fades |M|, its θ has no effect.\n'
            'angle: ρ and θ are smeared separately, so M turns towards the '
            'next layer\'s θ\n  over the roughness width (magnetic twist); a '
            'non-magnetic layer\'s θ then matters; φ is interpolated the '
            'same way.\n'
            'step: ρ is smeared like the NSLD and each slab takes the whole '
            'angle of one layer\n  (the angle jumps at the interface; a '
            'non-magnetic layer lends no angle).')

        self.rscheme = QtWidgets.QComboBox()
        self.rscheme.addItems(['rms (erf / tanh)', 'Licorne'])
        self.rscheme.setCurrentIndex(
            ROUGHNESS_SCHEMES.index(self.stack.roughness_scheme))
        self.rscheme.setToolTip(
            'rms: additive erf / tanh steps of Gaussian width σ, windows of '
            '± tail·σ.\n'
            'Licorne: σ is Licorne\'s σ_L; window to the 97 % point, '
            'clipped at half thicknesses,\n  thin layers renormalised, jumps '
            'at the window edges (Licorne manual, App. 10.1).')

        # instrumental resolution, Gaussian in Q (see model.stack RESOLUTION)
        res = self.stack.resolution
        self.res_on = QtWidgets.QCheckBox('smear')
        self.res_on.setChecked(res['enabled'])
        self.res_on.setToolTip(
            'Average the reflectivity over a Gaussian in Q of standard '
            'deviation\nσ_Q = Q·√((Δθ/θ)² + (Δλ/λ)²).')
        self.res_mode = QtWidgets.QComboBox()
        self.res_mode.addItems(['fixed λ', 'fixed θ per Q band'])
        self.res_mode.setCurrentIndex(RES_MODES.index(res['mode']))
        self.res_mode.setToolTip(
            'fixed λ: one wavelength, angle scan, θ = asin(Qλ/4π) '
            '(Licorne resolution.m, MONO mode).\n'
            'fixed θ per Q band: one or more fixed angles, each used over a '
            'Q band, λ = 4π sin θ / Q\n(presets: Mono with 3 angles, ToF '
            'with 1).')
        self.res_lambda = dspin(0.1, 50, 3, 0.1, ' Å')
        self.res_lambda.setValue(res['wavelength'])
        self.res_lambda.setToolTip('Neutron wavelength λ')
        self.res_dlambda = dspin(0, 50, 3, 0.05, ' %')
        self.res_dlambda.setValue(100 * res['dlambda_rel'])
        self.res_dlambda.setToolTip('Relative wavelength spread Δλ/λ '
                                    '(standard deviation)')
        self.res_dtheta = dspin(0, 100, 3, 0.05, ' mrad')
        self.res_dtheta.setValue(1e3 * res['dtheta'])
        self.res_dtheta.setToolTip('Angular spread Δθ (standard deviation)')
        self.res_tof_dl = dspin(0, 1, 4, 0.001, ' Å')
        self.res_tof_dl.setValue(res['tof_dlambda'])
        self.res_tof_dl.setToolTip('Absolute wavelength spread Δλ '
                                   '(standard deviation), the same at every λ')
        self.res_tof = QtWidgets.QTableWidget(0, 3)
        self.res_tof.setHorizontalHeaderLabels(
            ['θ (mrad)', 'Δθ (mrad)', 'Q max (Å⁻¹)'])
        self.res_tof.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.res_tof.verticalHeader().setDefaultSectionSize(
            self.res_tof.fontMetrics().height() + 6)
        self.res_tof.setFixedHeight(5 * (self.res_tof.fontMetrics().height()
                                         + 6) + 4)
        self.res_tof.setToolTip(
            'One row per angle, in increasing Q.  Row i is used from the '
            'previous row\'s Q max up to its own;\nthe last row is '
            'open-ended.')
        self._set_tof_angles(res['tof_angles'])
        self.res_tof.itemChanged.connect(self.schedule)
        tof_add = QtWidgets.QPushButton('+')
        tof_del = QtWidgets.QPushButton('−')
        for b in (tof_add, tof_del):          # as narrow as the style allows
            b.setFixedWidth(b.sizeHint().width())
        tof_add.setToolTip('Add an angle after the last one')
        tof_del.setToolTip('Remove the selected angle')
        tof_add.clicked.connect(self._add_tof_angle)
        self.res_preset = QtWidgets.QComboBox()
        self.res_preset.addItems(['Preset…'] + list(RES_PRESETS))
        self.res_preset.setToolTip('Fill Δλ and the angle table with a '
                                   'known instrument setup')
        self.res_preset.activated.connect(self._apply_res_preset)
        tof_del.clicked.connect(self._del_tof_angle)

        f.setVerticalSpacing(3)
        qrange = QtWidgets.QHBoxLayout()
        qrange.addWidget(self.qmin, 1)
        qrange.addSpacing(6)
        qrange.addWidget(QtWidgets.QLabel('–'))
        qrange.addSpacing(6)
        qrange.addWidget(self.qmax, 1)
        f.addRow('Q (Å⁻¹)', qrange)
        f.addRow('Q points', self.nq)
        f.addRow('Window tail', self.tail)
        f.addRow('Background', self.background)
        f.addRow('Q reference', self.q_fronting)
        f.addRow('Roughness', self.rscheme)
        f.addRow('M smearing', self.msmear)
        resrow = QtWidgets.QHBoxLayout()
        resrow.addWidget(self.res_on)
        resrow.addSpacing(10)
        resrow.addWidget(self.res_mode, 1)
        f.addRow('Resolution', resrow)

        # one page of parameters per mode, in RES_MODES order
        mono = QtWidgets.QWidget()
        mrow = QtWidgets.QHBoxLayout(mono)
        mrow.setContentsMargins(0, 0, 0, 0)
        for lab, w in (('λ', self.res_lambda), ('Δλ/λ', self.res_dlambda),
                       ('Δθ', self.res_dtheta)):
            mrow.addSpacing(10)
            mrow.addWidget(QtWidgets.QLabel(lab))
            mrow.addSpacing(4)
            mrow.addWidget(w, 1)
        tof = QtWidgets.QWidget()
        tg = QtWidgets.QGridLayout(tof)
        tg.setContentsMargins(0, 0, 0, 0)
        tg.addWidget(QtWidgets.QLabel('Δλ'), 0, 0)
        tg.addWidget(self.res_tof_dl, 0, 1)
        tg.addWidget(tof_add, 0, 2)
        tg.addWidget(tof_del, 0, 3)
        tg.addWidget(self.res_preset, 0, 4)
        tg.addWidget(self.res_tof, 1, 0, 1, 5)
        tg.setColumnStretch(1, 1)
        self.res_pages = QtWidgets.QStackedWidget()
        self.res_pages.addWidget(mono)
        self.res_pages.addWidget(tof)
        f.addRow(self.res_pages)

        for w in (self.qmin, self.qmax, self.tail,
                  self.background, self.res_lambda, self.res_dlambda,
                  self.res_dtheta, self.res_tof_dl):
            w.valueChanged.connect(self.schedule)
        self.nq.valueChanged.connect(self.schedule)
        self.msmear.currentIndexChanged.connect(self.schedule)
        self.rscheme.activated.connect(self._scheme_changed)
        self.q_fronting.toggled.connect(self.schedule)
        self.res_on.toggled.connect(self._sync_resolution)
        self.res_mode.currentIndexChanged.connect(self._sync_resolution)
        self._sync_resolution()

        # layer list
        lay = QtWidgets.QGroupBox('Layers (top → bottom)')
        lv = QtWidgets.QVBoxLayout(lay)
        self.list = QtWidgets.QListWidget()
        # a few rows; the tool box below takes the rest of the height
        self.list.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                QtWidgets.QSizePolicy.Policy.Fixed)
        self.list.setFixedHeight(7 * self.list.fontMetrics().height() + 16)
        self.list.currentRowChanged.connect(self._on_select)
        self.list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.list.model().rowsMoved.connect(self._on_rows_moved)
        lv.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton('+')
        self.btn_del = QtWidgets.QPushButton('−')
        self.btn_up = QtWidgets.QPushButton('▲')
        self.btn_dn = QtWidgets.QPushButton('▼')
        for b, tip in ((self.btn_add, 'Add layer'),
                       (self.btn_del, 'Remove layer'),
                       (self.btn_up, 'Move layer up'),
                       (self.btn_dn, 'Move layer down')):
            b.setToolTip(tip)
            b.setMinimumWidth(0)
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                            QtWidgets.QSizePolicy.Policy.Fixed)
            row.addWidget(b)
        self.btn_add.clicked.connect(self.add_layer)
        self.btn_del.clicked.connect(self.remove_layer)
        self.btn_up.clicked.connect(lambda: self.move_layer(-1))
        self.btn_dn.clicked.connect(lambda: self.move_layer(+1))
        lv.addLayout(row)
        v.addWidget(lay)

        # polarisation of every channel: Pi, Pa pairs
        pol = QtWidgets.QGroupBox('Polarisation (Pi incident, Pa analysed)')
        pv = QtWidgets.QVBoxLayout(pol)
        self.pol = PairTable(CHANNEL_NAMES)
        self.pol.changed.connect(self.schedule)
        pv.addWidget(self.pol)
        fill = QtWidgets.QHBoxLayout()
        fill.addWidget(QtWidgets.QLabel('Fill ±P in plane at'))
        self.fill_angle = dspin(-360, 360, 2, 5.0, ' °')
        self.fill_angle.setToolTip('Angle from sample x towards y')
        fill.addWidget(self.fill_angle)
        b = QtWidgets.QPushButton('Fill')
        b.setToolTip('Set every row to ±P along this in-plane direction '
                     '(Pa = 0 for R+ / R−)')
        b.clicked.connect(lambda: self.pol.set_vectors(default_vectors(
            in_plane(np.radians(self.fill_angle.value())))))
        fill.addWidget(b)
        b = QtWidgets.QPushButton('Along fronting M')
        b.setToolTip('Set every row to ±P along the fronting\'s in-plane M '
                     '(sample x if the fronting is not magnetic)')
        b.clicked.connect(self._fill_along_fronting)
        fill.addWidget(b)
        fill.addStretch(1)
        pv.addLayout(fill)

        self.editor = LayerEditor()
        self.editor.changed.connect(self._on_layer_edit)

        # one collapsible page open at a time keeps the column short; QToolBox
        # scrolls a page that is taller than the room left
        self.toolbox = QtWidgets.QToolBox()
        width = 0
        for box, title in ((self.editor, 'Layer properties'),
                           (gen, 'General parameters'), (pol, 'Polarisation')):
            box.setTitle('')                  # the page tab carries it
            box.setFlat(True)
            page = QtWidgets.QWidget()          # box at the top, not centred
            pl = QtWidgets.QVBoxLayout(page)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.addWidget(box)
            pl.addStretch(1)
            self.toolbox.addItem(page, title)
            width = max(width, page.minimumSizeHint().width())
        # never narrower than a page plus a vertical scroll bar: no
        # horizontal scrolling
        self.toolbox.setMinimumWidth(
            width + self.style().pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent) + 4)
        v.addWidget(self.toolbox, 1)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        return panel

    def _fill_along_fronting(self):
        """Every row along the fronting's in-plane M (sample x if the
        fronting is not magnetic)."""
        m = self.stack.fronting_direction()
        self.pol.set_vectors(default_vectors(in_plane(0.0) if m is None
                                             else m))

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
        p.getAxis('right').setLabel('MSLD θ, φ (deg)')
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
            if i in RIGHT:
                self.prof_vb2.addItem(bars)
                self.prof_vb2.addItem(curve)
                # the CSV exporter only walks PlotItem.items; register the
                # right-axis curve there too without adding it to p.vb
                p.items.append(curve)
            else:
                # bars go straight into the viewbox: PlotItem.addItem would
                # list them in p.curves, and the Matplotlib exporter chokes
                # on BarGraphItem.getData() (x is None when built from x0)
                p.vb.addItem(bars)
                p.addItem(curve)
            self.prof_bars.append(bars)
            self.prof_curves.append(curve)
        p.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#4b4e55', width=0.7)))
        self.prof_decor = []                 # interface lines
        # band and name of each layer; the list is built further down
        self.prof_layers = LayerLabels(p, lambda: self.list.currentRow())
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
        self.refl_formula.setStyleSheet('color: #9aa0a6')
        bar.addWidget(self.refl_formula)
        bar.addStretch(1)
        self.logy = QtWidgets.QCheckBox('log R')
        self.logy.setChecked(True)
        self.rq4 = QtWidgets.QCheckBox('R·Q⁴')
        self.freeze_y = QtWidgets.QCheckBox('Freeze Y')
        self.freeze_y.setToolTip('Keep the current y range when the model '
                                 'changes (e.g. while dragging a slider)')
        for cb in (self.logy, self.rq4, self.freeze_y):
            cb.toggled.connect(self.schedule)
            bar.addWidget(cb)
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
            pos=0, angle=0, pen=pg.mkPen('#4b4e55', width=0.7))
        self.refl_plot.addItem(self.asym_zero)
        self._refl_shown = []               # [(curve, name, Q, y)] on screen
        self._refl_log = False
        self._refl_ykey = None              # what the y axis currently shows
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
        self.toolbox.setItemText(0, 'Layer properties — %s'
                                 % self.stack.layers[row].name if ok
                                 else 'Layer properties')
        interior = ok and 0 < row < n - 1
        self.btn_del.setEnabled(interior)
        self.btn_up.setEnabled(interior and row > 1)
        self.btn_dn.setEnabled(interior and row < n - 2)
        self.prof_layers.style()

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

    def set_parameters(self, params, x):
        """Write values x of Stack.free_parameters() entries `params` into
        the stack (e.g. from a fit) and refresh the editor and plots."""
        for (i, attr, *_), v in zip(params, x):
            setattr(self.stack.layers[i], attr, float(v))
        self.refresh_list(select=self.list.currentRow())
        self.schedule()

    def _on_layer_edit(self):
        row = self.list.currentRow()
        self.refresh_list(select=row)
        self.schedule()

    def _scheme_changed(self, i):
        """A new roughness scheme brings its default M smearing."""
        self.msmear.setCurrentIndex(
            SMEARING_MODES.index(DEFAULT_SMEARING[ROUGHNESS_SCHEMES[i]]))
        self.schedule()

    # -- computation --------------------------------------------------------
    def schedule(self, *_):
        # throttle, not debounce: a dragged slider still redraws as it moves
        if not self._timer.isActive():
            self._timer.start()

    def recompute(self):
        try:
            self.stack.tail = self.tail.value()
            self.stack.background = self.background.value()
            self.stack.q_in_fronting = self.q_fronting.isChecked()
            self.stack.magnetic_smearing = SMEARING_MODES[
                self.msmear.currentIndex()]
            self.stack.roughness_scheme = ROUGHNESS_SCHEMES[
                self.rscheme.currentIndex()]
            self.tail.setEnabled(self.stack.roughness_scheme == 'rms')
            mode = RES_MODES[self.res_mode.currentIndex()]
            self.stack.resolution.update(
                enabled=self.res_on.isChecked(), mode=mode,
                wavelength=self.res_lambda.value(),
                dlambda_rel=self.res_dlambda.value() / 100,
                dtheta=self.res_dtheta.value() / 1e3,
                tof_dlambda=self.res_tof_dl.value())
            if mode == 'tof':
                self.stack.resolution['tof_angles'] = self._tof_angles()
            self.stack.build_sublayers()
            self._compute_profile()
            self._show_profile()
            self._draw_reflectance()
            text = '%d layers, %d sublayers' % (len(self.stack.layers),
                                                len(self.stack.sublayers))
            if self._refl_note:
                text += ('<br><span style="color:#f87171">%s</span>'
                         % html.escape(self._refl_note))
            self.status.setText(text)
            self.status.setStyleSheet('')
            self.prof_cursor.refresh()
            self.refl_cursor.refresh()
            self.reflectanceChanged.emit()
        except Exception as exc:                      # keep the UI alive
            self.status.setText('Error: %s' % exc)
            self.status.setStyleSheet('color: #f87171')

    def _sync_resolution(self, *_):
        on = self.res_on.isChecked()
        self.res_mode.setEnabled(on)
        self.res_pages.setCurrentIndex(self.res_mode.currentIndex())
        self.res_pages.setEnabled(on)
        self.schedule()

    # -- TOF angle table (mrad in the table, rad in the model) ---------------
    def _set_tof_angles(self, angles):
        t = self.res_tof
        t.blockSignals(True)
        t.setRowCount(0)
        for a in angles:
            r = t.rowCount()
            t.insertRow(r)
            for c, v in enumerate((1e3 * a['theta'], 1e3 * a['dtheta'],
                                   a['qmax'])):
                t.setItem(r, c, QtWidgets.QTableWidgetItem(
                    '' if v is None else '%g' % v))
        self._mark_last_tof_row()
        t.blockSignals(False)

    def _mark_last_tof_row(self):
        """Every Q max is editable except the last row's (open-ended)."""
        t = self.res_tof
        editable = QtCore.Qt.ItemFlag.ItemIsEditable
        for r in range(t.rowCount()):
            it = t.item(r, 2)
            if it is None:
                it = QtWidgets.QTableWidgetItem('')
                t.setItem(r, 2, it)
            last = r == t.rowCount() - 1
            it.setFlags(it.flags() & ~editable if last else
                        it.flags() | editable)
            if last:
                it.setText('∞')
            elif it.text() == '∞':
                it.setText('')

    def _tof_angles(self):
        t = self.res_tof
        out = []
        for r in range(t.rowCount()):
            last = r == t.rowCount() - 1
            vals = []
            for c in range(2 if last else 3):
                it = t.item(r, c)
                try:
                    vals.append(float(it.text()))
                except (AttributeError, ValueError):
                    raise ValueError('TOF angle %d: %s is not a number'
                                     % (r + 1, t.horizontalHeaderItem(c)
                                        .text())) from None
            out.append({'theta': vals[0] / 1e3, 'dtheta': vals[1] / 1e3,
                        'qmax': None if last else vals[2]})
        return out

    def _apply_res_preset(self, i):
        if i <= 0:
            return
        dl, angles = RES_PRESETS[self.res_preset.itemText(i)]
        self.res_preset.setCurrentIndex(0)
        self.res_tof_dl.setValue(dl)          # schedules the recompute
        self._set_tof_angles(angles)
        self.schedule()

    def _add_tof_angle(self):
        t = self.res_tof
        t.blockSignals(True)
        r = t.rowCount()
        t.insertRow(r)
        if r:                                   # continue from the old last
            prev = [t.item(r - 1, c).text() for c in range(2)]
            t.item(r - 1, 2).setText('')
            for c, v in enumerate(prev):
                t.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self._mark_last_tof_row()
        t.blockSignals(False)
        if r:                                   # its Q max is now needed
            t.editItem(t.item(r - 1, 2))
        self.schedule()

    def _del_tof_angle(self):
        t = self.res_tof
        if t.rowCount() <= 1:
            return
        r = t.currentRow()
        t.blockSignals(True)
        t.removeRow(r if r >= 0 else t.rowCount() - 1)
        self._mark_last_tof_row()
        t.blockSignals(False)
        self.schedule()

    def _compute_profile(self):
        self._profile, self._on_bars = profile_data(self.stack)

    def _show_profile(self, *_):
        """Draw the checked quantities from the cached profile."""
        if self._profile is None:
            return
        z, Z, e, curves, slabs = self._profile
        p = self.prof_plot
        show_bars = self.prof_show_bars.isChecked()

        if show_bars:
            curves = [self._on_bars.get(i, c) for i, c in enumerate(curves)]
        for i, cb in enumerate(self.prof_boxes):
            on = cb.isChecked()
            self.prof_curves[i].setData(z, curves[i], connect='finite')
            self.prof_curves[i].setVisible(on)
            # a blank angle (NaN) is a bar of zero height
            self.prof_bars[i].setOpts(x0=e[:-1], width=np.diff(e),
                                      height=np.nan_to_num(slabs[i]), y0=0)
            self.prof_bars[i].setVisible(on and show_bars)
        p.showAxis('right', any(self.prof_boxes[i].isChecked() for i in RIGHT))
        p.showAxis('left', any(cb.isChecked() for i, cb in
                               enumerate(self.prof_boxes) if i not in RIGHT))

        for item in self.prof_decor:
            p.removeItem(item)
        self.prof_decor = []
        line = pg.mkPen('#62666d', width=1)
        for Zj in Z:
            ln = pg.InfiniteLine(pos=Zj, angle=90, pen=line)
            ln.setZValue(-50)
            p.addItem(ln)
            self.prof_decor.append(ln)
        self.prof_layers.set_layers(self.stack.layers, Z, z[0], z[-1])
        p.setXRange(z[0], z[-1], padding=0)
        p.enableAutoRange(axis='y')
        self.prof_vb2.enableAutoRange(axis='y')
        self.prof_layers.place()

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
        if self.prof_show_bars.isChecked():          # as drawn, see _show_profile
            curves = [self._on_bars.get(i, c) for i, c in enumerate(curves)]
        layer = self.stack.layers[int(np.searchsorted(Z, x))]
        rows = []
        for i, (name, _, col) in enumerate(PROFILE_QUANTITIES):
            if not self.prof_boxes[i].isChecked():
                continue
            y = np.interp(x, z, curves[i], left=np.nan, right=np.nan)
            unit = ' °' if i in RIGHT else ''
            if self.prof_show_bars.isChecked():
                k = min(max(int(np.searchsorted(e, x)) - 1, 0), len(slabs[i]) - 1)
                value = '%s%s  (sublayer %s)' % (fmt(y), unit, fmt(slabs[i][k]))
            else:
                value = fmt(y) + unit
            vb = self.prof_vb2 if i in RIGHT else self.prof_plot.vb
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
        vec = self.pol.vectors()
        pairs = [vectors_pair(*vec[c]) for c in CHANNEL_NAMES]
        # a magnetic fronting drops the part of P transverse to its M
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            R = self.stack.reflectivities(Q, pairs)
        self._refl_note = ' '.join(str(w.message) for w in caught)
        R = dict(zip(CHANNEL_NAMES, R.T))
        self.last_Q, self.last_R = Q, R
        k = self.refl_quantity.currentIndex()
        name, label, formula = REFL_QUANTITIES[k]
        is_refl = label is None
        # log / R·Q⁴ only apply to raw reflectivities
        for w in (self.logy, self.rq4):
            w.setEnabled(is_refl)

        if k == 0:
            shown = zip(self.refl_curves, (R[c] for c in CHANNELS),
                        ['R' + c for c in CHANNELS])
        elif is_refl:
            shown = zip(self.hp_curves, (R['+'], R['-']), ['R+', 'R−'])
        else:
            shown = [(self.asym_curve, refl_quantity(name, R), name)]
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
        # a frozen y range only makes sense for the same plotted quantity;
        # set before setData, which autoranges straight away
        ykey = (self.refl_quantity.currentIndex(), log, self.rq4.isChecked())
        if self.freeze_y.isChecked() and ykey == self._refl_ykey:
            p.disableAutoRange(axis='y')
            p.enableAutoRange(axis='x')
        else:
            p.enableAutoRange()
        self._refl_ykey = ykey
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

    # -- save / load --------------------------------------------------------
    def session_state(self):
        """The model (Stack.to_dict) plus the tab's own settings under
        'simulation': Q grid, polarisation and display."""
        self.recompute()                     # the widgets' values into the stack
        d = self.stack.to_dict()
        d['simulation'] = {
            'qmin': self.qmin.value(), 'qmax': self.qmax.value(),
            'nq': self.nq.value(), 'polarisation': self.pol.vectors(),
            'display': {
                'quantity': self.refl_quantity.currentIndex(),
                'logy': self.logy.isChecked(), 'rq4': self.rq4.isChecked(),
                'freeze_y': self.freeze_y.isChecked(),
                'profile': [cb.isChecked() for cb in self.prof_boxes],
                'sublayers': self.prof_show_bars.isChecked()},
            'selected_layer': self.list.currentRow()}
        return d

    def restore_state(self, d):
        """Inverse of session_state; a plain model file (no 'simulation')
        keeps the tab's settings."""
        new = Stack.from_dict(d)                 # validate before touching UI
        self.stack.layers = new.layers
        self.stack.tail = new.tail
        self.stack.background = new.background
        self.stack.q_in_fronting = new.q_in_fronting
        self.stack.resolution = new.resolution
        self.stack.magnetic_smearing = new.magnetic_smearing
        self.stack.roughness_scheme = new.roughness_scheme
        self.stack.licorne_renorm = new.licorne_renorm
        self.stack.step_fallback = new.step_fallback
        sim = d.get('simulation', {})
        widgets = [self.qmin, self.qmax, self.nq, self.tail, self.pol,
                   self.background, self.q_fronting, self.res_on,
                   self.msmear, self.rscheme, self.res_mode,
                   self.res_tof_dl,
                   self.res_lambda, self.res_dlambda, self.res_dtheta]
        for w in widgets:
            w.blockSignals(True)
        self.qmin.setValue(sim.get('qmin', self.qmin.value()))
        self.qmax.setValue(sim.get('qmax', self.qmax.value()))
        self.nq.setValue(sim.get('nq', self.nq.value()))
        self.tail.setValue(new.tail)
        self.pol.set_vectors(sim.get('polarisation', default_vectors()))
        self.background.setValue(new.background)
        self.q_fronting.setChecked(new.q_in_fronting)
        self.msmear.setCurrentIndex(
            SMEARING_MODES.index(new.magnetic_smearing))
        self.rscheme.setCurrentIndex(
            ROUGHNESS_SCHEMES.index(new.roughness_scheme))
        res = new.resolution
        self.res_on.setChecked(res['enabled'])
        self.res_lambda.setValue(res['wavelength'])
        self.res_dlambda.setValue(100 * res['dlambda_rel'])
        self.res_dtheta.setValue(1e3 * res['dtheta'])
        self.res_mode.setCurrentIndex(RES_MODES.index(res['mode']))
        self.res_tof_dl.setValue(res['tof_dlambda'])
        self._set_tof_angles(res['tof_angles'])
        disp = sim.get('display', {})
        display = [self.refl_quantity, self.logy, self.rq4, self.freeze_y,
                   self.prof_show_bars] + self.prof_boxes
        for w in display:
            w.blockSignals(True)
        self.refl_quantity.setCurrentIndex(disp.get(
            'quantity', self.refl_quantity.currentIndex()))
        for key, cb in (('logy', self.logy), ('rq4', self.rq4),
                        ('freeze_y', self.freeze_y),
                        ('sublayers', self.prof_show_bars)):
            cb.setChecked(disp.get(key, cb.isChecked()))
        for cb, on in zip(self.prof_boxes, disp.get('profile', [])):
            cb.setChecked(on)
        for w in widgets + display:
            w.blockSignals(False)
        self._sync_resolution()
        self.refresh_list(select=sim.get('selected_layer', 1))
        self.recompute()
