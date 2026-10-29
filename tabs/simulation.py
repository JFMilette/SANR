"""
Simulation tab: build a Stack by hand and look at its profile and reflectivity.

    left  : general parameters beside the layer list (add / remove /
            reorder), property editor of the selected layer below
    right : depth profile on top, reflectivity below, both wide.
            The profile overlays NSLD / MSLD |M| (left axis) and the
            magnetic angles φ (in plane) and θ (out of plane; Licorne's
            letters, see ANGLE_LETTERS) on the right axis, each toggled by a
            checkbox:
            solid line = Stack.profile(z), bars = Stack.build_sublayers(),
            shaded bands = layers (selected one's name in bold, click a band
            to select it), grey verticals = nominal interfaces.

save_model / load_model write / read the stack and Q settings as JSON.

MODELS -- the tab holds several models (Stack.to_dict each), the Model box
above the layer list showing one of them in self.stack: switching keeps the
shown one as it is and loads the other (restore_state, models=False), so
each keeps its layers, general parameters, resolution, fit settings and
selected layer; the polarisation, Q range and display are the tab's.  New
takes no layer (empty_stack) with the shown model's other settings,
Copy the whole model.  A session file is the shown model as before, the
others under 'simulation' -> 'models'; switching is off while a fit runs.

The thickness spread and the mixture of models (model.stack MIXTURE) have
their own window (General parameters -> Spread / mixture -> Edit…): the
spread's rows, a row per other model (used, weight, bounds, Fit) and a
thickness tick per layer (shared by every mixed model).  Each recompute
links the mixed models (Stack._mix_stacks, from their stored dicts) and
writes the values they share back into those dicts.
The Polarisation box links each channel R++, R+-, R-+, R--, R+, R- to an
incident polarisation Pi and an analysed polarisation Pa (3-vectors in the
sample frame, length = efficiency, Pa = 0 for no analyser); the plot and
every dataset that does not set its own use these pairs.

Every edit triggers a debounced recompute.
"""

import copy
import html
import re
import warnings

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from model.polarisation import (CHANNEL_NAMES, default_vectors, in_plane,
                                vectors_pair)
from model.stack import (DEFAULT_SMEARING, INTERFACE_ORDERS,
                         ROUGHNESS_SCHEMES, SMEARING_MODES, SPREAD_BOUND,
                         SPREAD_NODE_CHOICES, Layer, Stack, cos_turns,
                         default_fit, spread_factors, wrap_turns)
import line_colours
import plot_export                     # noqa: F401 (the CSV export's layers)
import style
from fit.priors import complete as complete_prior


SLD_SCALE = 1e-6                      # SLDs are edited / plotted in 1e-6 A^-2
CHANNELS = ['++', '+-', '-+', '--']
AUTO_NAME = re.compile(r'L\d+')        # default layer names, renumbered
# combo label, axis label, colour
PROFILE_QUANTITIES = [
    ('NSLD', 'NSLD real (10⁻⁶ Å⁻²)', '#5b9bd5'),
    ('MSLD ρ', 'MSLD ρ = |M| (10⁻⁶ Å⁻²)', '#4fc08d'),
    ('φ', 'MSLD φ, in plane (deg)', '#e0a458'),
    ('θ', 'MSLD θ, out of plane (deg)', '#e36fa8'),
]
RIGHT = (2, 3)                        # PROFILE_QUANTITIES on the angle axis
# column names of the profile curves in pyqtgraph's CSV export ('<name>_x'
# is the depth in A, '<name>_y' the value in the plot's units)
PROFILE_EXPORT_NAMES = ('NSLD', 'MSLD_rho', 'MSLD_phi_deg', 'MSLD_theta_deg')


def export_name(item, name):
    """Name a plot item for pyqtgraph's CSV export (its column headers)
    without giving it a legend entry: set after the item is added."""
    item.opts['name'] = name
# combo label, axis label (None = raw reflectivities), formula beside the combo.
# The channels are the Pi / Pa pairs of the Polarisation box.
POLARIZED = 'Polarized with analysis'          # R++, R+-, R-+, R--
HALF_POLARIZED = 'Polarized without analysis'  # R+, R-
UNPOLARIZED = 'Non-polarized'                  # (R+ + R-) / 2
REFL_QUANTITIES = [
    (POLARIZED, None, 'R++, R+−, R−+, R−−'),
    (HALF_POLARIZED, None, 'R+, R− (Pa = 0: no analyser)'),
    (UNPOLARIZED, None, '(R+ + R−) / 2'),
    ('SA NSF', 'SA_NSF', '(R++ − R−−) / (R++ + R−−)'),
    ('SF fraction', 'SF fraction', '(R+− + R−+) / (R++ + R+− + R−+ + R−−)'),
    ('SA SF', 'SA_SF', '(R+− − R−+) / (R+− + R−+)'),
    ('SA half-polarized', 'SA', '(R+ − R−) / (R+ + R−)'),
]
# the y scale of a reflectivity (r_scale_combo): one or the other
R_SCALES = [('log', 'log R'), ('rq4', 'R·Q⁴')]


def r_scale_combo():
    """The R scale box of a reflectivity plot: log R (log axis) or R·Q⁴
    (linear axis), never both."""
    c = QtWidgets.QComboBox()
    for key, label in R_SCALES:
        c.addItem(label, key)
    c.setToolTip('log R: the reflectivity on a log axis.\nR·Q⁴: the '
                 'reflectivity times Q⁴ on a linear axis (the Fresnel '
                 'decay taken out).')
    return c


def saved_r_scale(disp):
    """R_SCALES key of a saved display dict: its 'scale', or from the
    older 'logy' / 'rq4' check boxes (R·Q⁴ when it was ticked)."""
    if disp.get('scale') in dict(R_SCALES):
        return disp['scale']
    return 'rq4' if disp.get('rq4') else 'log'


def set_r_scale(combo, key):
    combo.setCurrentIndex(max(0, combo.findData(key)))


# the list before UNPOLARIZED, whose indices older sessions saved
OLD_REFL_QUANTITIES = [POLARIZED, HALF_POLARIZED, 'SA NSF', 'SF fraction',
                       'SA SF', 'SA half-polarized']


def quantity_index(disp, default=0):
    """REFL_QUANTITIES index of a saved display dict: its 'quantity_name',
    or its older 'quantity' index (OLD_REFL_QUANTITIES), else default."""
    names = [q[0] for q in REFL_QUANTITIES]
    if disp.get('quantity_name') in names:
        return names.index(disp['quantity_name'])
    k = disp.get('quantity')
    if isinstance(k, int) and 0 <= k < len(OLD_REFL_QUANTITIES):
        return names.index(OLD_REFL_QUANTITIES[k])
    return default
# smallest font for the layer names in the depth profile
LABEL_MIN_PT = 6
# an additive profile whose crossing interfaces push a quantity this far
# (% of the layer's value, Stack.interface_crossing) past the layer values
# is flagged next to the Interfaces combo and shaded on the profile
CROSSING_WARN = 1.0
# Stack.interface_crossing quantities as the profile names them
CROSSING_NAMES = {'NSLD_real': 'NSLD', 'NSLD_img': 'Im NSLD', 'M_x': 'Mx',
                  'M_y': 'My', 'M_z': 'Mz', 'MSLD_rho': 'ρ',
                  'MSLD_theta': 'φ', 'MSLD_phi': 'θ'}


def crossing_warnings(st):
    """The crossings of Stack.interface_crossing that matter when st's
    profile is additive (CROSSING_WARN), each with the quantity it pushes
    furthest and by how much: [(entry, quantity, %)]; [] otherwise."""
    if st.roughness_scheme != 'rms' or st.interface_order != 'additive':
        return []
    out = []
    for c in st.interface_crossing():
        q, v = max(c['excursion'].items(), key=lambda kv: abs(kv[1]))
        if abs(v) > CROSSING_WARN:
            out.append((c, q, v))
    return out


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
    """Example sample: three films on a substrate, the tab's model when it
    is built without one (the tests, selftest.py); SANR itself starts
    from empty_stack()."""
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


def empty_stack():
    """No layer: vacuum on the substrate of make_stack.  The model SANR
    starts with, and that of a New model."""
    st = make_stack()
    return Stack([st.layers[0], st.layers[-1]])


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
        self.line = pg.InfiniteLine(angle=90, movable=False)
        self.text = pg.TextItem(anchor=(0, 1))
        for item in (self.line, self.text):
            item.setZValue(1000)
            vb.addItem(item, ignoreBounds=True)
        self.markers = {}                       # viewbox -> ScatterPlotItem
        self.restyle()
        style.notifier().changed.connect(self.restyle)
        self.hide()
        self._proxy = pg.SignalProxy(plot.scene().sigMouseMoved,
                                     rateLimit=60, slot=self._moved)
        widget.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Leave:
            self._pos = None
            self.hide()
        return False

    def restyle(self, *_):
        """Colours of the current theme (style)."""
        self.line.setPen(pg.mkPen(style.MUTED, width=0.8,
                                  style=QtCore.Qt.PenStyle.DashLine))
        self.text.setColor(style.TEXT)
        fill = QtGui.QColor(style.SURFACE)
        fill.setAlpha(235)
        self.text.fill = pg.mkBrush(fill)
        self.text.border = pg.mkPen(style.BORDER_STRONG)
        self.text.update()
        for m in self.markers.values():
            m.setPen(pg.mkPen(style.PLOT_BG, width=1))
        self.refresh()

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
            m = pg.ScatterPlotItem(size=8, pen=pg.mkPen(style.PLOT_BG, width=1))
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


def profile_layers(view, z):
    """The name of the layer at each depth z of a profile plot (its CSV
    export's 'layer' column; view as for profile_readout)."""
    if view._profile is None:
        return [''] * len(z)
    return plot_export.layer_names(view._profile[1],
                                   [l.name for l in view.stack.layers])(z)


def profile_readout(view, x):
    """Cursor text for a depth profile plot: layer, depth, checked curves.
    view holds the plot as the Simulation tab does: stack, _profile and
    _on_bars (profile_data), prof_boxes, prof_show_smooth,
    prof_show_bars, prof_plot, prof_vb2 (the right axis) and prof_right
    (ProfileRightAxis)."""
    if view._profile is None:
        return None, []
    z, Z, e, curves, slabs = view._profile
    show_bars = view.prof_show_bars.isChecked()
    show_smooth = view.prof_show_smooth.isChecked()
    if show_bars:                                # as drawn, see _show_profile
        curves = [view._on_bars.get(i, c) for i, c in enumerate(curves)]
    layer = view.stack.layers[int(np.searchsorted(Z, x))]
    rows = []
    for i, (name, _, col) in enumerate(PROFILE_QUANTITIES):
        if not view.prof_right.angle_on(i) or not (show_smooth or show_bars):
            continue
        unit = ' °' if i in RIGHT else ''
        k = min(max(int(np.searchsorted(e, x)) - 1, 0), len(slabs[i]) - 1)
        if not show_smooth:                      # the bar's top
            y = slabs[i][k]
            value = '%s%s  (sublayer)' % (fmt(y), unit)
        else:
            y = np.interp(x, z, curves[i], left=np.nan, right=np.nan)
            if show_bars:
                value = '%s%s  (sublayer %s)' % (fmt(y), unit,
                                                 fmt(slabs[i][k]))
            else:
                value = fmt(y) + unit
        vb = view.prof_vb2 if i in RIGHT else view.prof_plot.vb
        rows.append((name, value, col, y, vb))
    return 'z = %.4g Å  —  %s' % (x, layer.name), rows + \
        view.prof_right.readout(x)


# RIGHT AXIS of a depth profile: the magnetisation angles (PROFILE_QUANTITIES
# RIGHT) or the roughness of each interface, a stem at its depth: σ, and σ_M
# where a layer has its own magnetic roughness (MSLD_sigma)
RIGHT_AXIS_MODES = (('angles', 'Angles'), ('roughness', 'Roughness'))
ROUGHNESS_QUANTITIES = [('σ', 'roughness σ', '#17becf'),
                        ('σ_M', 'magnetic σ', '#bcbd22')]


def roughness_stems(st):
    """([(Z_j, σ_j)], [(Z_j, σ_M,j)]) of every interface (the top of layer
    j + 1): its roughness_sigma (0 for a sharp, 'none' one), and its
    MSLD_sigma where the layer has one."""
    sig, mag = [], []
    for Zj, l in zip(st._interfaces(), st.layers[1:]):
        sig.append((Zj, 0.0 if l.roughness_model == 'none'
                    else max(float(l.roughness_sigma), 0.0)))
        if l.MSLD_sigma is not None:
            mag.append((Zj, max(float(l.MSLD_sigma), 0.0)))
    return sig, mag


class ProfileRightAxis:
    """The Right axis box of a depth profile (RIGHT AXIS) and the roughness
    stems it draws in the right view box; the angle check boxes stay the
    view's own (prof_boxes[RIGHT]), shown only in the 'angles' mode."""

    def __init__(self, view, changed):
        """view: has prof_plot, prof_vb2 and prof_boxes; changed: called
        when the mode or a σ box changes."""
        self.view = view
        self.combo = QtWidgets.QComboBox()
        for key, label in RIGHT_AXIS_MODES:
            self.combo.addItem(label, key)
        self.combo.setToolTip('What the right axis shows: the in-plane and '
                              'out-of-plane magnetisation angles, or the '
                              'roughness of each interface (Å)')
        self.combo.currentIndexChanged.connect(self._mode_changed)
        self.boxes, self.stems, self.heads = [], [], []
        for name, tip, col in ROUGHNESS_QUANTITIES:
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet('color: %s' % col)
            cb.setToolTip('The %s of each interface, at its depth' % tip)
            cb.toggled.connect(changed)
            self.boxes.append(cb)
            stem = pg.PlotDataItem(pen=pg.mkPen(col, width=2),
                                   connect='pairs')
            head = pg.ScatterPlotItem(size=7, pen=pg.mkPen(col),
                                      brush=pg.mkBrush(col))
            view.prof_vb2.addItem(stem)
            view.prof_vb2.addItem(head)
            self.stems.append(stem)
            self.heads.append(head)
        self._changed = changed
        self._points = ([], [])

    def widgets(self):
        """The box and the σ check boxes, for the profile's bar."""
        return [self.combo] + self.boxes

    def mode(self):
        return self.combo.currentData()

    def set_mode(self, key):
        self.combo.setCurrentIndex(max(0, self.combo.findData(key)))

    def _mode_changed(self, *_):
        self._show_boxes()
        self._changed()

    def _show_boxes(self):
        """The angle boxes in the 'angles' mode, the σ boxes otherwise."""
        angles = self.mode() == 'angles'
        for i in RIGHT:
            self.view.prof_boxes[i].setVisible(angles)
        for cb in self.boxes:
            cb.setVisible(not angles)

    def angle_on(self, i):
        """Profile quantity i is drawn (its box ticked, and on the right
        axis only in the 'angles' mode)."""
        return self.view.prof_boxes[i].isChecked() and \
            (i not in RIGHT or self.mode() == 'angles')

    def draw(self, st):
        """The stems of st, shown in the 'roughness' mode; the right axis
        labelled and shown when something is on it."""
        self._points = roughness_stems(st)
        rough = self.mode() == 'roughness'
        for cb, stem, head, pts in zip(self.boxes, self.stems, self.heads,
                                       self._points):
            z = np.array([p[0] for p in pts])
            s = np.array([p[1] for p in pts])
            stem.setData(np.repeat(z, 2), np.ravel(np.column_stack(
                [np.zeros_like(s), s])))
            head.setData(z, s)
            on = rough and cb.isChecked()
            stem.setVisible(on)
            head.setVisible(on)
        p = self.view.prof_plot
        p.getAxis('right').setLabel('roughness σ (Å)' if rough
                                    else 'MSLD φ, θ (deg)')
        p.showAxis('right', any(cb.isChecked() for cb in self.boxes) if rough
                   else any(self.view.prof_boxes[i].isChecked()
                            for i in RIGHT))

    def readout(self, x):
        """Cursor rows (as profile_readout's) of the interface nearest x."""
        if self.mode() != 'roughness':
            return []
        rows = []
        for cb, (name, _, col), pts in zip(self.boxes, ROUGHNESS_QUANTITIES,
                                           self._points):
            if not cb.isChecked() or not pts:
                continue
            Zj, s = min(pts, key=lambda p: abs(p[0] - x))
            rows.append((name, '%s Å  (interface at %.4g Å)' % (fmt(s), Zj),
                         col, s, self.view.prof_vb2))
        return rows

    def state(self):
        return {'right_axis': self.mode(),
                'roughness': [cb.isChecked() for cb in self.boxes]}

    def restore(self, disp):
        """The mode and σ boxes of a saved display dict, without drawing:
        the view redraws once its model is loaded."""
        for w in [self.combo] + self.boxes:
            w.blockSignals(True)
        self.set_mode(disp.get('right_axis', self.mode()))
        for cb, on in zip(self.boxes, disp.get('roughness', [])):
            cb.setChecked(on)
        for w in [self.combo] + self.boxes:
            w.blockSignals(False)
        self._show_boxes()


class LayerLabels:
    """A shaded band and a name (with the thickness) per layer on a depth
    profile plot.  selected() is the row highlighted (-1 for none): its name
    in bold, its band tinted and its interfaces drawn as bold lines in the
    theme's SELECT colour (violet, none of the curves' colours)."""

    def __init__(self, plot, selected=lambda: -1):
        self.plot, self.selected = plot, selected
        self.bands = []                      # one shaded region per layer
        self.labels = []                     # layer names, pinned to the top
        self.spans = []                      # (z0, z1, texts) of each label
        self.Z = []                          # the interfaces
        # the selected layer's top and bottom interfaces
        self.edges = [pg.InfiniteLine(angle=90, movable=False)
                      for _ in range(2)]
        for ln in self.edges:
            ln.setZValue(-40)                # over the grey interface lines
            ln.hide()
            plot.addItem(ln, ignoreBounds=True)
        plot.vb.sigYRangeChanged.connect(self.place)
        plot.vb.sigXRangeChanged.connect(self.fit)
        plot.vb.sigResized.connect(self.fit)
        style.notifier().changed.connect(self.style)

    def set_layers(self, layers, Z, zlo, zhi):
        """One band per layer between the interfaces Z; the semi-infinite
        media run out to zlo / zhi, the edges of the plotted range."""
        p = self.plot
        for item in self.bands + self.labels:
            p.removeItem(item)
        self.bands, self.labels, self.spans = [], [], []
        self.Z = list(Z)
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

    def style(self, *_):
        """Alternate grey shading per layer; the selected layer's name in
        bold, its band tinted and its interfaces in bold SELECT lines."""
        sel = self.selected()
        tint = QtGui.QColor(style.SELECT)
        tint.setAlpha(28)
        for j, (band, t) in enumerate(zip(self.bands, self.labels)):
            # alternating shading, the theme's band colour on odd layers
            brush = pg.mkBrush(tint if j == sel else
                               style.BAND if j % 2 else '#00000000')
            band.setBrush(brush)
            band.setHoverBrush(brush)
            # setBrush does not repaint: without this the old colour stays
            # on screen and only the strips the cursor crosses get the new
            band.update()
            font = t.textItem.font()
            font.setBold(j == sel)
            t.setFont(font)
            t.setColor(style.TEXT if j == sel else style.MUTED)
        # interface j - 1 is the top of layer j, interface j its bottom
        pen = pg.mkPen(style.SELECT, width=2.5)
        for ln, k in zip(self.edges, (sel - 1, sel)):
            on = sel >= 0 and 0 <= k < len(self.Z)
            if on:
                ln.setPos(self.Z[k])
                ln.setPen(pen)
            ln.setVisible(on)
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
    ('MSLD_rho', 'MSLD ρ', -100, 100, 6, 0.1, '', 1 / SLD_SCALE),
    ('MSLD_theta', 'MSLD φ', -360, 360, 4, 5.0, ' °', 360.0),
    ('MSLD_phi', 'MSLD θ', -90, 90, 4, 5.0, ' °', 360.0),
    ('roughness_sigma', 'Roughness σ', 0.0, 1e4, 4, 0.5, ' Å', 1.0),
    ('MSLD_sigma', 'Magnetic σ', 0.0, 1e4, 4, 0.5, ' Å', 1.0),
]
# attr -> (label, display scale, unit) for the parameter tables
# the GUI writes the magnetisation angles with Licorne's letters: the
# in-plane angle MSLD_theta as φ (Licorne's azimuth), the elevation
# MSLD_phi as θ (0 = in plane; Licorne's θ is measured from its z axis)
ANGLE_LETTERS = {'MSLD_theta': 'MSLD φ', 'MSLD_phi': 'MSLD θ'}


def shown_names(text):
    """text with the attribute names MSLD_theta / MSLD_phi written as the
    GUI shows them (ANGLE_LETTERS), e.g. 'Fe.MSLD_theta' -> 'Fe.MSLD φ'."""
    return re.sub(r'MSLD_(theta|phi)\b',
                  lambda m: ANGLE_LETTERS[m.group(0)], str(text))


PARAM_DISPLAY = {attr: (label.split(' (')[0], scale,
                        suffix.strip() or ('Å' if attr == 'thickness' else
                                           '10⁻⁶ Å⁻²'))
                 for attr, label, *_, suffix, scale in EDITOR_PARAMS}
PARAM_DISPLAY.update(scale=('Scale', 1.0, ''),
                     background=('Background', 1.0, ''),
                     q_shift=('Q shift', 1.0, 'Å⁻¹'),
                     thickness_spread=('Thickness spread', 100.0, '%'),
                     weight=('Weight', 1.0, ''))
# the stack rows shown in other units than stored: attr -> factor (shown =
# factor x stored)
STACK_ROW_FACTOR = {'thickness_spread': 100.0}
RES_MODES = ('tof', 'data')     # res_mode combo order
# res_conv combo: label, resolution['scheme'], resolution['licorne_fun']
RES_CONVOLUTIONS = [('quadrature', 'quadrature', None),
                    ('Licorne 1: simple mean', 'licorne', 1),
                    ('Licorne 2: rectangle rule', 'licorne', 2),
                    ('Licorne 3: midpoint rule', 'licorne', 3)]


def res_convolution(res):
    """Index in RES_CONVOLUTIONS of a resolution dict."""
    if res.get('scheme', 'quadrature') != 'licorne':
        return 0
    return [c[2] for c in RES_CONVOLUTIONS].index(res.get('licorne_fun', 3))
# instrument setups of the fixed-angle mode: name -> (dlambda (A), angles)
RES_PRESETS = {
    'ToF (3 angles)': (0.005, [{'theta': 0.006, 'dtheta': 3e-4, 'qmax': 0.04},
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
                    w.setStyleSheet(style.out_of_bounds() if bad else '')
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


def prior_text(spec, scale=1.0, unit=''):
    """Short description of a fit.priors spec in display units."""
    if spec is None:
        return 'no prior'
    u = (' ' + unit) if unit else ''
    if spec['kind'] == 'gauss':
        return 'prior %.6g ± %.3g%s' % (spec['mu'] * scale,
                                        spec['tau'] * abs(scale), u)
    return 'prior %.6g – %.6g%s (softness %.3g)' % (
        spec['a'] * scale, spec['b'] * scale, u, spec['s'] * abs(scale))


class PriorDialog(QtWidgets.QDialog):
    """Edit the prior of one fit parameter (fit.priors) in display units:
    none, Gaussian (expected ± tolerance, default 10 % of it) or soft box
    (lower, upper, softness, default 10 % of the width)."""

    KINDS = [('none', None), ('Gaussian: expected ± tolerance', 'gauss'),
             ('soft box: between lower and upper', 'soft_box')]

    def __init__(self, label, spec, scale=1.0, unit='', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Prior — %s' % label)
        self.setMinimumWidth(380)
        self.scale, self.spec = scale, spec
        self.kind = QtWidgets.QComboBox()
        for text, _ in self.KINDS:
            self.kind.addItem(text)
        self.edits = {}
        for key, tip in (('mu', 'expected value'),
                         ('tau', 'default 10 % of the expected value'),
                         ('a', 'lower'), ('b', 'upper'),
                         ('s', 'default 10 % of the width')):
            e = QtWidgets.QLineEdit()
            e.setValidator(QtGui.QDoubleValidator())
            e.setPlaceholderText(tip if key in ('tau', 's') else '')
            self.edits[key] = e
        f = QtWidgets.QFormLayout(self)
        f.addRow(QtWidgets.QLabel(
            '<i>What another measurement says %s should be%s; the fit '
            'bounds stay the physical limits.</i>'
            % (label, ' (in %s)' % unit if unit else ''), wordWrap=True))
        f.addRow('Kind', self.kind)
        for key, lab in (('mu', 'Expected'), ('tau', '±'), ('a', 'Lower'),
                         ('b', 'Upper'), ('s', 'Softness')):
            f.addRow(lab, self.edits[key])
        self.error = QtWidgets.QLabel()
        self.error.setStyleSheet('color: %s' % style.ERROR)
        f.addRow(self.error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        f.addRow(buttons)
        self._form = f
        if spec is not None:
            self.kind.setCurrentIndex([k for _, k in self.KINDS]
                                      .index(spec['kind']))
            for key, v in spec.items():
                if key in self.edits:
                    self.edits[key].setText('%.8g' % (v * scale))
        self.kind.currentIndexChanged.connect(self._show)
        self._show()

    def _show(self, *_):
        kind = self.KINDS[self.kind.currentIndex()][1]
        for key, e in self.edits.items():
            on = (kind == 'gauss' and key in ('mu', 'tau')) or \
                (kind == 'soft_box' and key in ('a', 'b', 's'))
            self._form.setRowVisible(e, on)
        # fit the rows now shown (grow for a kind's fields, shrink for none)
        self.layout().activate()
        self.resize(self.width(), self.sizeHint().height())

    def _accept(self):
        kind = self.KINDS[self.kind.currentIndex()][1]
        if kind is None:
            self.spec = None
            self.accept()
            return
        keys = ('mu', 'tau') if kind == 'gauss' else ('a', 'b', 's')
        spec = {'kind': kind}
        try:
            for key in keys:
                t = self.edits[key].text().strip()
                spec[key] = None if not t else float(t) / self.scale
            if kind == 'gauss' and spec['tau'] is not None:
                spec['tau'] = abs(spec['tau'])
            if kind == 'soft_box' and None in (spec['a'], spec['b']):
                raise ValueError('give lower and upper')
            if kind == 'gauss' and spec['mu'] is None:
                raise ValueError('give the expected value')
            self.spec = complete_prior(spec)
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        self.accept()


def prior_button():
    b = QtWidgets.QToolButton()
    b.setText('prior…')
    b.setAutoRaise(True)
    return b


def show_prior(button, spec, scale=1.0, unit=''):
    """The button of a parameter's prior: marked (●) when it has one."""
    button.setText('prior ●' if spec is not None else 'prior…')
    button.setToolTip(prior_text(spec, scale, unit) + ' — click to edit')


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
        self.stack, self.index = None, None     # for ties (load)
        self._loading = False
        self._sliding = False           # value change coming from a slider
        self._fronting = False          # the loaded layer is the fronting

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
        # right-click a row's name: tie it with other layers (model.stack
        # ties), or untie it
        for attr, row in self.rows.items():
            lab = row[4]
            lab.setContextMenuPolicy(
                QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            lab.customContextMenuRequested.connect(
                lambda pos, a=attr: self._tie_menu(a, pos))
        # attr -> button of its prior from other measurements (fit.priors)
        self.priors = {attr: prior_button() for attr, *_ in EDITOR_PARAMS}
        for attr, b in self.priors.items():
            b.clicked.connect(lambda _, a=attr: self._edit_prior(a))
        self.sliders = {}
        for attr, *_ in EDITOR_PARAMS:
            sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sl.setRange(0, SLIDER_STEPS)
            sl.setToolTip('Sweep the value between Min and Max')
            sl.valueChanged.connect(
                lambda pos, a=attr: self._slider_moved(a, pos))
            self.sliders[attr] = sl
        self.rows['MSLD_theta'][4].setToolTip(
            'φ: in-plane angle of M from the sample x axis towards y '
            '(Majkrzak Fig. 1.14)')
        self.rows['MSLD_rho'][4].setToolTip(
            'ρ_M: the magnetic SLD of M, including any out-of-plane part.\n'
            'Negative: M points opposite to φ, the same layer as +ρ at φ + '
            '180°.\nWith "step" or "angle" M smearing, ρ is smeared as a '
            'signed number:\nbetween +ρ and −ρ layers it passes through 0 '
            'at the interface.')
        self.rows['MSLD_phi'][4].setToolTip(
            'θ: elevation of M out of the film plane (0 = in plane, +90° = '
            '+z). Only ρ cos θ is seen by neutrons (Halperin)')
        # the magnetic profile's own roughness (model.stack MAGNETIC
        # ROUGHNESS); unticked, it follows Roughness σ
        self.msig_own = QtWidgets.QCheckBox('separate')
        self.msig_own.setToolTip(
            'Give the magnetic profile (ρ, φ, θ) its own roughness at this '
            'interface; the NSLD keeps Roughness σ.\nUnticked: the '
            'magnetic profile uses Roughness σ.  In the Licorne scheme '
            'the magnetic profile gets its own Licorne window\n(truncated, '
            'renormalised) — no longer a model Licorne itself can make.')
        self.msig_own.toggled.connect(self._msig_toggled)

        g = QtWidgets.QGridLayout(self)
        g.setColumnStretch(1, 1)
        g.setVerticalSpacing(2)
        g.addWidget(QtWidgets.QLabel('Name'), 0, 0)
        g.addWidget(self.name, 0, 1, 1, 4)
        # what is fitted is ticked in the Fit tab (fit.datasets_panel); the
        # hidden Fit boxes only mirror Layer.fit's 'vary' for _store
        for c, h in enumerate(['', 'Value', 'Min', 'Max', '']):
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
            cb.hide()
            g.addWidget(self.sliders[attr], r + 1, 1, 1, 3)
            g.addWidget(self.priors[attr], r + 1, 4)
            if attr == 'MSLD_sigma':
                g.addWidget(self.msig_own, r + 1, 0)
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

    def load(self, layer, is_fronting, is_backing, stack=None, index=None):
        self.layer = layer
        self.stack, self.index = stack, index
        self.setEnabled(layer is not None)
        if layer is None:
            return
        self._loading = True
        self.name.setText(layer.name)
        own = layer.MSLD_sigma is not None
        self.msig_own.setChecked(own)
        for attr, *_, scale in EDITOR_PARAMS:
            val, bmin, bmax, cb, _ = self.rows[attr]
            e = layer.fit_entry(attr)
            v = getattr(layer, attr)
            if attr == 'MSLD_sigma' and not own:
                v = layer.roughness_sigma
            val.setValue(v * scale)
            bmin.setValue(e['min'] * scale)
            bmax.setValue(e['max'] * scale)
            cb.setChecked(e['vary'])
            show_prior(self.priors[attr], e.get('prior'), scale,
                       PARAM_DISPLAY[attr][2])
            self._show_tie(attr)
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
        self._set_row_enabled('MSLD_sigma', not is_fronting and own)
        self.msig_own.setEnabled(not is_fronting)
        self._fronting = is_fronting
        for w in (self.model, self.nsub, self.nsub_min, self.nsub_max,
                  self.nsub_slider):
            w.setEnabled(not is_fronting)
        self._check_bounds()

    def _msig_toggled(self, own):
        """Separate magnetic σ on (it starts at Roughness σ) or off."""
        if self._loading or self.layer is None:
            return
        self.rows['MSLD_sigma'][0].setValue(
            self.rows['roughness_sigma'][0].value())
        self._set_row_enabled('MSLD_sigma', own and not self._fronting)
        self._store()

    def _set_row_enabled(self, attr, on):
        for w in self.rows[attr] + (self.priors[attr],):
            w.setEnabled(on)

    def _show_tie(self, attr):
        """Mark a tied row (⛓ and the other members in its tooltip)."""
        lab = self.rows[attr][4]
        label = [l for a, l, *_ in EDITOR_PARAMS if a == attr][0]
        k = None if self.stack is None else \
            self.stack.tie_of(self.index, attr)
        lab.setText(label + (' ⛓' if k is not None else ''))
        if k is None:
            tip = 'Right-click to tie with the same parameter of other layers'
        elif self.stack.ties[k]['mode'] == 'total':
            mem = self.stack.ties[k]['members']
            tip = ('Tied: %s — a fit keeps their total thickness (now '
                   '%.4g Å) and varies each; %s takes what the others leave '
                   '(right-click to untie)' % (
                       self.stack.tie_name(k),
                       sum(self.stack.layers[i].thickness for i, _ in mem),
                       self.stack.layers[mem[-1][0]].name))
        else:
            tip = ('Tied: %s, fitted as one parameter (right-click to untie)'
                   % self.stack.tie_name(k))
        lab.setToolTip(tip)

    def _tie_menu(self, attr, pos):
        if self.stack is None or self.index is None:
            return
        menu = QtWidgets.QMenu(self)
        k = self.stack.tie_of(self.index, attr)
        if k is not None:
            menu.addAction('Untie %s' % self.stack.tie_name(k),
                           lambda: self._untie(attr))
        else:
            menu.addAction('Tie with…', lambda: self._tie_with(attr))
        menu.exec(self.rows[attr][4].mapToGlobal(pos))

    def _tie_with(self, attr):
        """Choose the layers whose `attr` is fitted equal to this one's, or,
        for a thickness, whose total thickness with this one a fit keeps."""
        st, i = self.stack, self.index
        tied = st.tied()
        others = [j for j in range(len(st.layers)) if j != i and
                  st.fit_allowed(j, attr) and (j, attr) not in tied]
        if not others:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Tie %s.%s' % (st.layers[i].name, attr))
        v = QtWidgets.QVBoxLayout(dlg)
        equal = QtWidgets.QRadioButton(
            'Equal: fit these layers\' value equal to %s\'s, as one '
            'parameter' % st.layers[i].name)
        total = QtWidgets.QRadioButton(
            'Fixed total: a fit keeps the total thickness of %s and these '
            'layers\nat the value below and varies each; the deepest of them '
            'takes what the others leave' % st.layers[i].name)
        # the total, starting at the members' current sum; another value is
        # reached by changing the deepest member's thickness
        total_row = QtWidgets.QHBoxLayout()
        total_row.addWidget(QtWidgets.QLabel('Total thickness'))
        total_val = dspin(0.0, 1e6, 4, 1.0, ' Å')
        total_val.setToolTip(
            'The total thickness the fit keeps; set by changing the deepest '
            'member\'s thickness,\nso at least the sum of the others')
        total_row.addWidget(total_val)
        total_row.addStretch(1)
        edited = [False]                # the user's value, kept on toggles
        # an equal thickness tie rarely has overlapping bounds (a layer and
        # its interface): a fixed total is the default
        (total if attr == 'thickness' else equal).setChecked(True)
        if attr == 'thickness':
            v.addWidget(equal)
            v.addWidget(total)
            v.addLayout(total_row)
        else:
            v.addWidget(QtWidgets.QLabel(
                'Fit these layers\' value equal to %s\'s, as one parameter:'
                % st.layers[i].name))
        boxes = []
        for j in others:
            b = QtWidgets.QCheckBox(st.layers[j].name)
            v.addWidget(b)
            boxes.append((j, b))

        def members():
            return sorted([i] + [j for j, b in boxes if b.isChecked()])

        def update_total():
            """At least the sum of the members but the deepest; the current
            sum until the user types a value."""
            mem = members()
            rest = sum(st.layers[j].thickness for j in mem[:-1])
            total_val.blockSignals(True)
            total_val.setMinimum(rest)
            if not edited[0]:
                total_val.setValue(rest + st.layers[mem[-1]].thickness)
            total_val.blockSignals(False)
            total_val.setEnabled(total.isChecked())

        def user_edit(_):
            edited[0] = True
        total_val.valueChanged.connect(user_edit)
        total.toggled.connect(update_total)
        for _, b in boxes:
            b.toggled.connect(update_total)
        update_total()
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if not dlg.exec():
            return
        chosen = [j for j, b in boxes if b.isChecked()]
        if chosen:
            if total.isChecked():
                mem = members()
                total_val.interpretText()           # a value still typed in
                rest = sum(st.layers[j].thickness for j in mem[:-1])
                st.layers[mem[-1]].thickness = max(total_val.value() - rest,
                                                   0.0)
                st.tie([(j, attr) for j in mem], 'total')
            else:
                st.tie([(i, attr)] + [(j, attr) for j in chosen])
            self._show_tie(attr)
            self.changed.emit()

    def _untie(self, attr):
        self.stack.untie(self.index, attr)
        self._show_tie(attr)
        self.changed.emit()

    def _edit_prior(self, attr):
        if self.layer is None:
            return
        label, scale, unit = PARAM_DISPLAY[attr]
        e = self.layer.fit_entry(attr)
        dlg = PriorDialog('%s of %s' % (label, self.layer.name),
                          e.get('prior'), scale, unit, self)
        if dlg.exec():
            e['prior'] = dlg.spec
            show_prior(self.priors[attr], dlg.spec, scale, unit)
            self.changed.emit()

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
            val.setStyleSheet(style.out_of_bounds() if bad else '')
            val.setToolTip('Outside the fit bounds' if bad else '')

    def _store(self):
        if self._loading or self.layer is None:
            return
        L = self.layer
        L.name = self.name.text()
        for attr, *_, scale in EDITOR_PARAMS:
            val, bmin, bmax, cb, _ = self.rows[attr]
            setattr(L, attr, val.value() / scale)
            if attr == 'MSLD_sigma' and not self.msig_own.isChecked():
                L.MSLD_sigma = None
                val.blockSignals(True)          # shown following Roughness σ
                val.setValue(self.rows['roughness_sigma'][0].value())
                val.blockSignals(False)
                self._sync_slider(attr)
            # the prior (set in its own dialog) is kept
            L.fit[attr] = dict(L.fit.get(attr, {}), vary=cb.isChecked(),
                               min=bmin.value() / scale,
                               max=bmax.value() / scale)
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
    # the Model box changed (a model added, removed, renamed, shown or
    # written into); a rename also gives (old name, new name)
    modelsChanged = QtCore.pyqtSignal()
    modelRenamed = QtCore.pyqtSignal(str, str)

    def __init__(self, stack=None, parent=None):
        super().__init__(parent)
        self.stack = stack if stack is not None else make_stack()
        self.last_Q = self.last_R = None     # latest Q, {channel: R}
        self._refl_note = ''                 # warning of the last reflectivity
        # [(dataset name, {channel: (Q, dQ, theta)})] of the imported data
        # with a dQ column, for the resolution dialog (set by the
        # Experimental tab)
        self.dq_sets = lambda: []
        self._profile = None        # (z, edges, [curves], [slab values])
        self._crossings = []        # crossing_warnings of the profile
        self._locked = False        # a fit is running: see set_locked
        # MODELS: [{'name', 'model': Stack.to_dict, 'selected_layer'}], the
        # shown one's model None (it is self.stack)
        self._models = [dict(name='Model 1', model=None, selected_layer=1)]
        self._model_index = 0

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
        self._refresh_models()
        self._refresh_mixture_table()
        self.recompute()

    # -- left column --------------------------------------------------------
    def _build_controls(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)

        # general parameters
        gen = self.gen_box = QtWidgets.QGroupBox('General parameters')
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
        self.scale = dspin(0.0, 100.0, 4, 0.01)
        self.scale.setValue(self.stack.scale)
        self.scale.setToolTip('Factor on every reflectivity channel, applied '
                              'before the background')
        self.q_shift = dspin(-0.05, 0.05, 6, 0.0001)
        self.q_shift.setValue(self.stack.q_shift)
        self.q_shift.setToolTip(
            'Moves the model along Q (Å⁻¹): the reflectivity at a measured Q '
            'is the sample\'s at Q − shift,\nso a positive shift moves the '
            'curve to higher Q (e.g. a small misalignment of the Q axis).  '
            'The resolution keeps its σ_Q at the measured Q.')
        # THICKNESS SPREAD (model.stack), shown in %, stored as a fraction
        self.spread = dspin(0.0, 50.0, 2, 0.1)
        self.spread.setValue(100 * self.stack.thickness_spread)
        self.spread.setToolTip(
            'Lateral spread of the film thickness (σ/d, %).  Incoherent '
            'average over regions larger than the neutron coherence length: '
            'every layer together, or one layer alone (Spread below).\n'
            'Each node costs one more reflectivity: see nodes.')
        self.spread_target = QtWidgets.QComboBox()
        self.spread_target.setToolTip(
            'Which thicknesses vary laterally: every layer together, this '
            'layer alone (the layers below it then shift in depth), or\n'
            'several layers ticked below, all by the same factor.  A '
            'spread of part of the stack can trade off against\nthose '
            'layers\' roughness and their neighbours\' thicknesses: check '
            'with DREAM.')
        # the layers of a 'Several layers' target, one tick each
        self.spread_layers = QtWidgets.QListWidget()
        self.spread_layers.setToolTip(
            'The layers the spread scales, together: one thickness factor '
            'for all of them')
        self.spread_layers.setFixedHeight(
            5 * self.spread_layers.fontMetrics().height() + 12)
        self.spread_layers.itemChanged.connect(self.schedule)
        self.spread_dist = QtWidgets.QComboBox()
        for key, label in (('gaussian', 'Gaussian'),
                           ('lognormal', 'log-normal')):
            self.spread_dist.addItem(label, key)
        self.spread_dist.setToolTip(
            'Distribution of the thickness factor, mean 1 and standard '
            'deviation the spread in both.\nGaussian: up to about 19 % '
            '(thicknesses must stay positive).  log-normal: always positive, '
            'for large one-layer spreads.')
        self.spread_error = QtWidgets.QLabel(wordWrap=True)
        self.spread_error.setStyleSheet('color: %s' % style.ERROR)
        self.spread_error.hide()
        self.spread_nodes = QtWidgets.QComboBox()
        for n in SPREAD_NODE_CHOICES:
            self.spread_nodes.addItem(str(n), n)
        self.spread_nodes.setToolTip(
            'Gauss–Hermite nodes of the thickness average.  auto: from the '
            'spread, the film thickness and Q max (about 2e-3 relative '
            'error or better); a fit freezes it from the spread\'s upper '
            'bound so the cost does not jump.')
        # fit settings of the stack's own parameters (Stack.fit):
        # attr -> (value, min, max, Fit check box)
        self.stack_rows = {}
        for attr, val, make in (
                ('scale', self.scale, lambda: dspin(0.0, 100.0, 4, 0.01)),
                ('background', self.background, SciSpinBox),
                ('q_shift', self.q_shift,
                 lambda: dspin(-0.05, 0.05, 6, 0.0001)),
                ('thickness_spread', self.spread,
                 lambda: dspin(0.0, 50.0, 2, 0.1))):
            bmin, bmax = make(), make()
            for b in (bmin, bmax):
                b.setButtonSymbols(
                    QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            bmin.setToolTip('Fit minimum')
            bmax.setToolTip('Fit maximum')
            cb = QtWidgets.QCheckBox('Fit')
            cb.setToolTip('Vary %s in the fit, between the two bounds' % attr)
            self.stack_rows[attr] = (val, bmin, bmax, cb)
        self.stack_priors = {attr: prior_button() for attr in self.stack_rows}
        for attr, b in self.stack_priors.items():
            b.clicked.connect(lambda _, a=attr: self._edit_stack_prior(a))
        self._load_stack_fit()

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
            'vector: the components ρ cos φ, ρ sin φ are smeared (lateral '
            'average of a rough interface);\n  a non-magnetic neighbour only '
            'fades |M|, its φ has no effect.\n'
            'angle: ρ and φ are smeared separately, so M turns towards the '
            'next layer\'s φ\n  over the roughness width (magnetic twist); a '
            'non-magnetic layer\'s φ then matters; θ is interpolated the '
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

        self.iorder = QtWidgets.QComboBox()
        self.iorder.addItems(['additive', 'ordered (non-crossing)',
                              'independent (smooth)'])
        self.iorder.setCurrentIndex(
            INTERFACE_ORDERS.index(self.stack.interface_order))
        self.iorder.setToolTip(
            'How the rms interfaces combine (rms roughness only).\n'
            'additive: the erf / tanh steps add up.  Where a wider interface '
            'reaches past a sharper neighbour\n  they cross: a layer\'s '
            'occupancy goes negative, so ρ or the NSLD overshoot and in '
            'vector smearing\n  M points backwards (φ flips half a turn).  '
            'Sessions saved before this setting load as additive.\n'
            'ordered: each interface stays above the one grown before it '
            '(the substrate side), so every depth is a\n  weighted average '
            'of the layers; identical to additive where nothing crosses.  '
            'Where interfaces cross,\n  a layer\'s integrated moment / SLD '
            'gains the area additive made negative.\n'
            'independent: each interface fluctuates on its own, again kept '
            'above the ones below it; never negative\n  and smooth (no kink '
            'where interfaces cross), but a layer only a few σ thick gains '
            'material\n  even where nothing crosses (+2.5 % erf, +4 % tanh '
            'for t = 2σ).')
        self.iorder_warn = QtWidgets.QLabel()
        self.iorder_warn.setWordWrap(True)

        self.lic_exact = QtWidgets.QCheckBox(
            'exact (roughness, M smearing, resolution)')
        self.lic_exact.setChecked(self.stack.is_licorne_exact())
        self.lic_exact.setToolTip(
            'Every choice Licorne 1.4.2 makes, to reproduce its curves: '
            'Licorne roughness, step M smearing\n  without fallback, '
            'Licorne\'s 50-point window integral, its 1.5·t fronting / '
            'substrate pseudo-layers\n  and its resolution convolution on '
            'the data grid.  Unticking brings back the settings from before '
            'it was ticked\n  (the physical defaults if it was never off: '
            'exact integral, semi-infinite outer media,\n  quadrature).  '
            'Changing any of these unticks it.')
        self.lic_exact.toggled.connect(self._licorne_exact)

        # instrumental resolution, Gaussian in Q (see model.stack RESOLUTION)
        res = self.stack.resolution
        self.res_on = QtWidgets.QCheckBox('enable')
        self.res_on.setChecked(res['enabled'])
        self.res_on.setToolTip(
            'Average the reflectivity over a Gaussian in Q of standard '
            'deviation\nσ_Q = Q·√((Δθ/θ)² + (Δλ/λ)²).')
        self.res_mode = QtWidgets.QComboBox()
        self.res_mode.addItems(['fixed θ per Q band', 'from data dQ'])
        self.res_mode.setCurrentIndex(RES_MODES.index(res['mode']))
        self.res_mode.setToolTip(
            'fixed θ per Q band: time of flight at one or more fixed angles, '
            'each used over a Q band,\nλ = 4π sin θ / Q (presets: 3 angles '
            'or 1).\n'
            'from data dQ: the measured dQ — a fit averages each data point '
            'over its own dQ (stitched angles\nas measured); curves '
            'elsewhere interpolate the dQ of the dataset chosen with '
            'Show….')
        self.res_conv = QtWidgets.QComboBox()
        self.res_conv.addItems([c[0] for c in RES_CONVOLUTIONS])
        self.res_conv.setCurrentIndex(res_convolution(res))
        self.res_conv.setToolTip(
            'How the reflectivity is averaged over σ_Q.\n'
            'quadrature: a Gaussian average, independent of the Q points '
            'computed.\n'
            'Licorne: Licorne\'s own sum over the computed Q points '
            'themselves (its ResolutionFun),\n  unnormalised, constant tails '
            'past the ends, so the result depends on the Q grid:\n'
            '  1 simple mean: plain average of the points within ±σ/2;\n'
            '  2 rectangle rule: Gaussian-weighted sum within ±3σ, each point '
            'weighted by its spacing\n    towards the centre point (Licorne '
            'before 1.2.3);\n'
            '  3 midpoint rule: the same, each point weighted by half the '
            'distance between its neighbours.\n'
            'The fit uses the data Q; the curve here uses this tab\'s Q grid, '
            'so it can differ from Licorne\'s.')
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
        for attr, label in (('scale', 'Scale'), ('background', 'Background'),
                            ('q_shift', 'Q shift (Å⁻¹)')):
            f.addRow(label, self._stack_row(attr))
        # THICKNESS SPREAD and MIXTURE: their own window
        srow = QtWidgets.QHBoxLayout()
        self.spread_summary = QtWidgets.QLabel()
        self.spread_summary.setProperty('role', 'muted')
        srow.addWidget(self.spread_summary, 1)
        b = QtWidgets.QPushButton('Edit…')
        b.setToolTip('Thickness spread and mixture of models')
        b.clicked.connect(self.show_spread_window)
        srow.addWidget(b)
        f.addRow('Spread / mixture', srow)
        self._build_spread_window()
        f.addRow('Q reference', self.q_fronting)
        f.addRow('Licorne', self.lic_exact)
        f.addRow('Roughness', self.rscheme)
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(self.iorder)
        orow.addWidget(self.iorder_warn, 1)
        f.addRow('Interfaces', orow)
        f.addRow('M smearing', self.msmear)
        resrow = QtWidgets.QHBoxLayout()
        resrow.addWidget(self.res_on)
        resrow.addSpacing(10)
        resrow.addWidget(self.res_mode, 1)
        resrow.addWidget(self.res_conv, 1)
        res_show = QtWidgets.QPushButton('Show…')
        res_show.setToolTip('Plot σ_Q against Q, compare it with the dQ '
                            'column of the imported data and fit the '
                            'spreads to it')
        res_show.clicked.connect(self._show_resolution)
        resrow.addWidget(res_show)
        f.addRow('Resolution', resrow)

        # one page of parameters per mode, in RES_MODES order
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
        data = QtWidgets.QWidget()
        drow = QtWidgets.QHBoxLayout(data)
        drow.setContentsMargins(0, 0, 0, 0)
        self.res_data_info = QtWidgets.QLabel()
        self.res_data_info.setProperty('role', 'muted')
        self.res_data_info.setWordWrap(True)
        drow.addWidget(self.res_data_info, 1)
        res_data_pick = QtWidgets.QPushButton('Choose…')
        res_data_pick.setToolTip('Pick the dataset whose dQ (σ) the '
                                 'curves use')
        res_data_pick.clicked.connect(self._show_resolution)
        drow.addWidget(res_data_pick)
        self.res_pages = QtWidgets.QStackedWidget()
        self.res_pages.addWidget(tof)
        self.res_pages.addWidget(data)
        f.addRow(self.res_pages)

        for w in (self.qmin, self.qmax, self.tail,
                  self.background, self.res_tof_dl):
            w.valueChanged.connect(self.schedule)
        for val, bmin, bmax, cb in self.stack_rows.values():
            for w in (val, bmin, bmax):
                w.valueChanged.connect(self.schedule)
            cb.toggled.connect(self.schedule)
        self.spread_nodes.currentIndexChanged.connect(self.schedule)
        self.spread_target.currentIndexChanged.connect(
            self._spread_target_changed)
        self.spread_dist.currentIndexChanged.connect(self._spread_dist_changed)
        self.nq.valueChanged.connect(self.schedule)
        self.msmear.currentIndexChanged.connect(self.schedule)
        self.iorder.currentIndexChanged.connect(self.schedule)
        self.rscheme.activated.connect(self._scheme_changed)
        # the Licorne-exact box follows its settings at once
        for w in (self.rscheme, self.msmear, self.res_conv):
            w.currentIndexChanged.connect(self._sync_licorne_exact)
        self.q_fronting.toggled.connect(self.schedule)
        self.res_on.toggled.connect(self._sync_resolution)
        self.res_mode.currentIndexChanged.connect(self._sync_resolution)
        self.res_conv.currentIndexChanged.connect(self.schedule)
        self._sync_resolution()

        # layer list
        lay = QtWidgets.QGroupBox('Layers (top → bottom)')
        lv = QtWidgets.QVBoxLayout(lay)
        # MODELS: the model shown and the buttons that add / remove one
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Model'))
        self.model_box = QtWidgets.QComboBox()
        self.model_box.setToolTip(
            'The model shown; each keeps its own layers, general parameters, '
            'resolution and fit settings.\nPolarisation, the Q range and the '
            'plots are shared.')
        self.model_box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                     QtWidgets.QSizePolicy.Policy.Fixed)
        self.model_box.currentIndexChanged.connect(self.switch_model)
        row.addWidget(self.model_box, 1)
        self.model_btns = {}
        for key, label, tip, slot in (
                ('new', 'New', 'New model: no layer (vacuum on the '
                 'substrate), this model\'s other settings', self.new_model),
                ('copy', 'Copy', 'New model: a copy of this one',
                 self.copy_model),
                ('rename', 'Rename…', 'Rename this model', self.rename_model),
                ('delete', 'Delete', 'Delete this model', self.delete_model)):
            b = self.model_btns[key] = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        lv.addLayout(row)
        self.list = QtWidgets.QListWidget()
        # a few rows; the tool box below takes the rest of the height
        self.list.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                QtWidgets.QSizePolicy.Policy.Fixed)
        self.list.setFixedHeight(7 * self.list.fontMetrics().height() + 16)
        self.list.currentRowChanged.connect(self._on_select)
        # picking a layer (click, or arrow keys in the list) opens its page
        self.list.itemClicked.connect(self._show_layer_page)
        self.list.currentRowChanged.connect(
            lambda _: self.list.hasFocus() and self._show_layer_page())
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
        self.btn_untick = QtWidgets.QPushButton('Untick all Fit')
        self.btn_untick.setToolTip(
            'Untick Fit on every parameter of every layer and of the general '
            'parameters (ties follow);\nvalues and fit bounds are kept')
        self.btn_untick.clicked.connect(self.untick_all)
        # tick one quantity on every layer it belongs to (Stack.tick_all)
        self.btn_tick = {}
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_untick)
        for attr, label, tip in (
                ('thickness', 'Tick thickness',
                 'Tick Fit on the thickness of every film layer'),
                ('NSLD_real', 'Tick NSLD',
                 'Tick Fit on the real NSLD of every film layer and of the '
                 'substrate (not the fronting)'),
                ('roughness_sigma', 'Tick roughness',
                 'Tick Fit on the roughness σ of every interface, the '
                 'substrate\'s included (not those with roughness model '
                 '"none")')):
            b = self.btn_tick[attr] = QtWidgets.QPushButton(label)
            b.setToolTip(tip + ';\nvalues and fit bounds are kept, as when '
                         'ticked by hand')
            b.clicked.connect(lambda _, a=attr: self.tick_all(a))
            row.addWidget(b)
        for b in [self.btn_untick] + list(self.btn_tick.values()):
            b.hide()                    # the Fit tab has its own
            b.setMinimumWidth(0)
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                            QtWidgets.QSizePolicy.Policy.Fixed)
        lv.addLayout(row)
        v.addWidget(lay)

        # polarisation of every channel: Pi, Pa pairs
        pol = self.pol_box = QtWidgets.QGroupBox(
            'Polarisation (Pi incident, Pa analysed)')
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

    def _stack_row(self, attr):
        """Value, min – max, Fit and prior of a stack parameter."""
        val, bmin, bmax, cb = self.stack_rows[attr]
        row = QtWidgets.QHBoxLayout()
        row.addWidget(val, 3)
        row.addWidget(bmin, 2)
        row.addWidget(QtWidgets.QLabel('–'))
        row.addWidget(bmax, 2)
        row.addWidget(cb)
        cb.hide()                       # ticked in the Fit tab
        row.addWidget(self.stack_priors[attr])
        return row

    def _build_spread_window(self):
        """The Thickness spread and mixture window (non-modal): the spread's
        rows, the models mixed with this one and the thicknesses they share
        (model.stack THICKNESS SPREAD, MIXTURE)."""
        win = self.spread_window = QtWidgets.QDialog(self)
        win.setWindowTitle('Thickness spread and mixture')
        outer = QtWidgets.QVBoxLayout(win)
        panel = self.spread_panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

        # one or the other (model.stack MIXTURE)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Lateral variation:'))
        self.lateral = QtWidgets.QButtonGroup(self)
        for k, (key, label) in enumerate((
                ('none', 'None'), ('spread', 'Thickness spread'),
                ('mixture', 'Mixture of models'))):
            b = QtWidgets.QRadioButton(label)
            b.setProperty('mode', key)
            self.lateral.addButton(b, k)
            row.addWidget(b)
        row.addStretch(1)
        self.lateral.button(0).setChecked(True)
        self.lateral.idClicked.connect(self._lateral_chosen)
        v.addLayout(row)

        box = self.spread_box = QtWidgets.QGroupBox('Thickness spread')
        f = QtWidgets.QFormLayout(box)
        f.addRow('σ/d (%)', self._stack_row('thickness_spread'))
        nrow = QtWidgets.QHBoxLayout()
        nrow.addWidget(self.spread_target, 3)
        nrow.addWidget(self.spread_dist, 2)
        nrow.addWidget(QtWidgets.QLabel('nodes'))
        nrow.addWidget(self.spread_nodes, 1)
        f.addRow('Spread', nrow)
        f.addRow('', self.spread_layers)
        f.addRow('', self.spread_error)
        v.addWidget(box)

        box = self.mix_box = QtWidgets.QGroupBox('Mixture of models')
        mv = QtWidgets.QVBoxLayout(box)
        note = QtWidgets.QLabel(
            'R = w₀ R(this model) + Σ w R(model): regions larger than the '
            'coherence length, e.g. 0, 1 or 2 extra layers.  The other '
            'models are those of the Model box; each keeps its own layers '
            '(no thickness spread), this model\'s scale, background, Q shift '
            'and resolution apply once.')
        note.setWordWrap(True)
        note.setProperty('role', 'muted')
        mv.addWidget(note)
        t = self.mix_table = QtWidgets.QTableWidget(0, 6)
        t.setHorizontalHeaderLabels(['Use', 'Model', 'Weight', 'Min', 'Max',
                                     'Fit'])
        t.verticalHeader().hide()
        t.setColumnHidden(5, True)      # a weight is ticked in the Fit tab
        t.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        mv.addWidget(t)
        self.mix_w0 = QtWidgets.QLabel()
        mv.addWidget(self.mix_w0)
        v.addWidget(box)

        box = self.shared_box = QtWidgets.QGroupBox('Shared parameters')
        lv = QtWidgets.QVBoxLayout(box)
        note = QtWidgets.QLabel(
            'Every parameter of this model ticked Fit (in the Fit tab) is '
            'shared: the mixed models take its value, layers matched by '
            'name (a model without that layer skips it).  Show the model '
            'with the most layers.  A layer\'s thickness ticked there is '
            'one value for all models.')
        note.setWordWrap(True)
        note.setProperty('role', 'muted')
        lv.addWidget(note)
        self.mix_layers = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self.mix_layers).setContentsMargins(0, 0, 0, 0)
        self._mix_layer_key = None
        lv.addWidget(self.mix_layers)
        self.mix_layers.hide()          # ticked in the Fit tab
        v.addWidget(box)
        v.addStretch(1)

        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(win.close)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        outer.addLayout(row)
        win.resize(580, 780)
        self._mix_cache = {}        # name -> (its stored dict, linked Stack)

    def lateral_mode(self):
        return self.lateral.checkedButton().property('mode')

    def _lateral_chosen(self, k):
        """None, the thickness spread or the mixture: the other one is
        switched off (spread 0 and not fitted; no model mixed in)."""
        mode = self.lateral.button(k).property('mode')
        if mode != 'spread':
            val, _, _, cb = self.stack_rows['thickness_spread']
            val.setValue(0.0)
            cb.setChecked(False)
        if mode != 'mixture' and self.stack.mixture:
            self.stack.mixture = []
            self._refresh_mixture_table()
        self._show_lateral()
        self.schedule()

    def _show_lateral(self):
        mode = self.lateral_mode()
        self.spread_box.setEnabled(mode == 'spread')
        self.mix_box.setEnabled(mode == 'mixture')
        self.shared_box.setEnabled(mode == 'mixture')

    def _refresh_lateral(self, keep=True):
        """The choice from the stack: a spread or a mixture shows it;
        neither keeps the choice shown (keep) or shows None."""
        st = self.stack
        mode = 'mixture' if st.mixture else 'spread' if (
            st.thickness_spread or
            st.fit.get('thickness_spread', {}).get('vary')) else None
        if mode is None:
            mode = self.lateral_mode() if keep else 'none'
        for b in self.lateral.buttons():
            if b.property('mode') == mode:
                b.setChecked(True)
        self._show_lateral()

    def show_spread_window(self):
        self._refresh_mixture_table()
        self.spread_window.show()
        self.spread_window.raise_()
        self.spread_window.activateWindow()

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
        right = QtWidgets.QHBoxLayout()      # filled once prof_vb2 exists
        self.prof_show_smooth = QtWidgets.QCheckBox('smooth')
        self.prof_show_smooth.setChecked(True)
        self.prof_show_smooth.setToolTip('The continuous profile Stack.profile(z)')
        self.prof_show_smooth.toggled.connect(self._show_profile)
        self.prof_show_bars = QtWidgets.QCheckBox('sublayers')
        self.prof_show_bars.setChecked(True)
        self.prof_show_bars.toggled.connect(self._show_profile)
        for i, cb in enumerate(self.prof_boxes):
            (right if i in RIGHT else bar).addWidget(cb)
        bar.addSpacing(10)
        bar.addWidget(QtWidgets.QLabel('Right axis:'))
        bar.addLayout(right)
        bar.addSpacing(10)
        bar.addWidget(self.prof_show_smooth)
        bar.addWidget(self.prof_show_bars)
        bar.addStretch(1)
        v.addLayout(bar)

        w, self.prof_plot = plot_widget('depth z (Å)', 'SLD (10⁻⁶ Å⁻²)')
        w.setToolTip('Nominal thicknesses; the thickness spread is not drawn')
        p = self.prof_plot
        # SLDs share the left axis; the angle (deg) gets its own right axis
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
            export_name(curve, PROFILE_EXPORT_NAMES[i])
            self.prof_bars.append(bars)
            self.prof_curves.append(curve)
        # RIGHT AXIS: angles or roughness, its box before the angle boxes
        self.prof_right = ProfileRightAxis(self, self._show_profile)
        right.insertWidget(0, self.prof_right.combo)
        for cb in self.prof_right.boxes:
            right.addWidget(cb)
        self.prof_right._show_boxes()
        p.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#4b4e55', width=0.7)))
        self.prof_decor = []                 # interface lines
        # band and name of each layer; the list is built further down
        self.prof_layers = LayerLabels(p, lambda: self.list.currentRow())
        p.scene().sigMouseClicked.connect(self._on_profile_click)
        self.prof_cursor = Crosshair(
            w, p, lambda x: profile_readout(self, x))
        p.layer_at = lambda z: profile_layers(self, z)
        v.addWidget(w)
        return box

    def _build_reflectance_plot(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel('Plot:'))
        self.refl_quantity = QtWidgets.QComboBox()
        self.refl_quantity.addItems([q[0] for q in REFL_QUANTITIES])
        self.refl_quantity.currentIndexChanged.connect(self.schedule)
        bar.addWidget(self.refl_quantity)
        self.refl_formula = QtWidgets.QLabel()
        self.refl_formula.setProperty('role', 'muted')
        bar.addWidget(self.refl_formula)
        bar.addStretch(1)
        self.r_scale = r_scale_combo()
        self.r_scale.currentIndexChanged.connect(self.schedule)
        bar.addWidget(self.r_scale)
        self.freeze_y = QtWidgets.QCheckBox('Freeze Y')
        self.freeze_y.setToolTip('Keep the current y range when the model '
                                 'changes (e.g. while dragging a slider)')
        self.freeze_y.toggled.connect(self.schedule)
        bar.addWidget(self.freeze_y)
        bar.addWidget(line_colours.button())
        v.addLayout(bar)

        w, self.refl_plot = plot_widget('Q (Å⁻¹)', 'Reflectivity')
        self.refl_plot.addLegend(offset=(-10, 10))
        self.refl_curves = [self.refl_plot.plot(connect='finite')
                            for _ in CHANNELS]
        self.hp_curves = [self.refl_plot.plot(connect='finite')
                          for _ in range(2)]
        self.asym_curve = self.refl_plot.plot(connect='finite')
        self.unpol_curve = self.refl_plot.plot(connect='finite')
        self._line_colours()
        line_colours.notifier().changed.connect(self._line_colours)
        self.asym_zero = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#4b4e55', width=0.7))
        self.refl_plot.addItem(self.asym_zero)
        self._refl_shown = []               # [(curve, name, Q, y)] on screen
        self._refl_log = False
        self._refl_ykey = None              # what the y axis currently shows
        self.refl_cursor = Crosshair(w, self.refl_plot, self._refl_readout)
        v.addWidget(w)
        return box

    def _line_colours(self):
        """Pens of the reflectivity curves from line_colours."""
        for curve, key in zip(self.refl_curves + self.hp_curves
                              + [self.asym_curve, self.unpol_curve],
                              CHANNELS + ['+', '-', 'asym', 'unpol']):
            curve.setPen(pg.mkPen(line_colours.get(key), width=1.8))
        for sample, _ in self.refl_plot.legend.items:
            sample.update()

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
        self._refresh_spread_targets()      # names and positions may change
        if select is not None:
            self.list.setCurrentRow(max(0, min(select, n - 1)))
        self._on_select(self.list.currentRow())

    def _on_select(self, row):
        n = len(self.stack.layers)
        ok = 0 <= row < n
        self.editor.load(self.stack.layers[row] if ok else None,
                         row == 0, row == n - 1, self.stack, row)
        self.toolbox.setItemText(0, 'Layer properties — %s'
                                 % self.stack.layers[row].name if ok
                                 else 'Layer properties')
        interior = ok and 0 < row < n - 1 and not self._locked
        self.btn_del.setEnabled(interior)
        self.btn_up.setEnabled(interior and row > 1)
        self.btn_dn.setEnabled(interior and row < n - 2)
        if self._locked:                 # shown, greyed, not editable
            self.editor.setEnabled(False)
        self.prof_layers.style()

    def _show_layer_page(self, *_):
        """Open the tool box on the layer's page (from General parameters or
        Polarisation) when the user picks a layer.  Not done in _on_select,
        which also runs after edits made on those pages."""
        self.toolbox.setCurrentIndex(0)

    def set_locked(self, locked):
        """While a fit or sampling runs: nothing that changes the model can
        be edited (layer values, general parameters, polarisation, adding /
        removing / moving layers), but layers can still be selected and
        their values read, the tool-box pages switched and the plots used."""
        self._locked = bool(locked)
        for w in (self.gen_box, self.pol_box, self.btn_add, self.btn_untick,
                  *self.btn_tick.values(), self.model_box,
                  *self.model_btns.values(), self.spread_panel):
            w.setEnabled(not self._locked)
        if not self._locked:
            self._refresh_models()           # Delete needs two models
        self.list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop if self._locked
            else QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self._on_select(self.list.currentRow())   # editor, move buttons

    def untick_all(self):
        """The Untick all Fit button (Stack.untick_all)."""
        if self._locked:
            return
        self.recompute()                   # the general rows' ticks first
        n = self.stack.untick_all()
        self._load_stack_fit()
        self.refresh_list(select=self.list.currentRow())
        self.schedule()
        msg = 'Fit unticked on %d parameter%s.' % (n, '' if n == 1 else 's')
        win = self.window()
        if isinstance(win, QtWidgets.QMainWindow):
            win.statusBar().showMessage(msg, 8000)

    def tick_all(self, attr):
        """A Tick … button (Stack.tick_all)."""
        if self._locked:
            return
        n = self.stack.tick_all(attr)
        self.refresh_list(select=self.list.currentRow())
        self.schedule()
        msg = 'Fit ticked on %d more %s.' % (n, {
            'thickness': 'thickness' if n == 1 else 'thicknesses',
            'NSLD_real': 'NSLD' if n == 1 else 'NSLDs',
            'roughness_sigma': 'roughness' if n == 1 else 'roughnesses'}[attr])
        win = self.window()
        if isinstance(win, QtWidgets.QMainWindow):
            win.statusBar().showMessage(msg, 8000)

    def add_layer(self):
        row = self.list.currentRow()
        pos = min(max(row + 1, 1), len(self.stack.layers) - 1)
        old = list(self.stack.layers)
        above = old[pos - 1]
        if pos > 1:
            # a copy of the layer above, its fit bounds kept but nothing
            # varied: a new layer adds no fit parameters
            layer = copy.deepcopy(above)
            layer.name = 'L%d' % pos
            layer.fit = {a: dict(e, vary=False) for a, e in above.fit.items()}
        else:                            # below the fronting: the defaults
            layer = Layer('L%d' % pos, thickness=50.0, NSLD_real=2.0e-6,
                          roughness_sigma=3.0, roughness_sublayer=10)
        self.stack.layers.insert(pos, layer)
        self.stack.reindex_ties(old)
        self._reindex_spread(old)
        self._renumber()
        self.refresh_list(select=pos)
        self.schedule()

    def remove_layer(self):
        row = self.list.currentRow()
        if 0 < row < len(self.stack.layers) - 1:
            old = list(self.stack.layers)
            del self.stack.layers[row]
            self.stack.reindex_ties(old)
            self._reindex_spread(old)
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
            old = list(L)
            L[row], L[new] = L[new], L[row]
            self.stack.reindex_ties(old)
            self._reindex_spread(old)
            self._renumber()
            self.refresh_list(select=new)
            self.schedule()

    def _on_rows_moved(self, _parent, start, _end, _dest_parent, dest):
        """Mirror a drag-and-drop in the list onto the stack; drops onto the
        fronting / backing positions are reverted."""
        L = self.stack.layers
        new = dest if dest < start else dest - 1
        if 0 < start < len(L) - 1 and 0 < new < len(L) - 1 and new != start:
            old = list(L)
            L.insert(new, L.pop(start))
            self.stack.reindex_ties(old)
            self._reindex_spread(old)
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
            setattr(self.stack.owner(i), attr, float(v))
        self._load_stack_fit()             # recompute reads these boxes
        self._refresh_mixture_table()
        self.refresh_list(select=self.list.currentRow())
        self.schedule()

    def _bound_limits(self):
        """attr -> (lo, hi): the bounds the editor's boxes can hold."""
        limits = {attr: (a / scale, b / scale)
                  for attr, _, a, b, *_, scale in EDITOR_PARAMS}
        limits.update(scale=(self.scale.minimum(), self.scale.maximum()),
                      background=(self.background.minimum(),
                                  self.background.maximum()),
                      q_shift=(self.q_shift.minimum(),
                               self.q_shift.maximum()),
                      thickness_spread=(self.spread.minimum() / 100,
                                        self.spread.maximum() / 100),
                      weight=(0.0, 1.0))
        return limits

    def _write_bounds(self, stack, params, lo, hi):
        limits = self._bound_limits()
        for (i, attr, *_), a, b in zip(params, lo, hi):
            lim_lo, lim_hi = sorted(limits[attr])
            e = stack.owner(i).fit_entry(attr)
            e['min'] = float(min(max(a, lim_lo), lim_hi))
            e['max'] = float(min(max(b, lim_lo), lim_hi))

    def set_bounds(self, params, lo, hi):
        """Write fit bounds [lo, hi] of Stack.free_parameters() entries
        `params`, clipped to what the editor's boxes can hold, and refresh
        the editor."""
        self._write_bounds(self.stack, params, lo, hi)
        self._load_stack_fit()
        self._refresh_mixture_table()
        self.refresh_list(select=self.list.currentRow())
        self.schedule()

    def _load_stack_fit(self):
        """Show the stack's own parameters (scale, background, Q shift,
        thickness spread and its nodes) and their fit settings."""
        for attr, (val, bmin, bmax, cb) in self.stack_rows.items():
            e = self.stack.fit_entry(attr)
            k = STACK_ROW_FACTOR.get(attr, 1.0)
            for w in (val, bmin, bmax, cb):
                w.blockSignals(True)
            val.setValue(k * getattr(self.stack, attr))
            bmin.setValue(k * e['min'])
            bmax.setValue(k * e['max'])
            cb.setChecked(e['vary'])
            for w in (val, bmin, bmax, cb):
                w.blockSignals(False)
            show_prior(self.stack_priors[attr], e.get('prior'),
                       *PARAM_DISPLAY[attr][1:])
        c = self.spread_nodes
        c.blockSignals(True)
        c.setCurrentIndex(max(0, c.findData(self.stack.thickness_spread_nodes)))
        if c.currentData() != self.stack.thickness_spread_nodes:
            # a count not among the choices (e.g. a session's 13): kept
            c.addItem(str(self.stack.thickness_spread_nodes),
                      self.stack.thickness_spread_nodes)
            c.setCurrentIndex(c.count() - 1)
        c.blockSignals(False)
        c = self.spread_dist
        c.blockSignals(True)
        c.setCurrentIndex(max(0, c.findData(self.stack.thickness_spread_dist)))
        c.blockSignals(False)
        self._refresh_spread_targets()

    def _refresh_spread_targets(self):
        """The Spread target box: 'Whole stack', then each interior layer
        by name ('#i name' when names repeat), then 'Several layers' with
        its ticks below; the stack's target selected."""
        c, L = self.spread_target, self.stack.layers
        names = [l.name for l in L]
        label = ['#%d %s' % (i, n) if names.count(n) > 1 else n
                 for i, n in enumerate(names)]
        t = self.stack.thickness_spread_target
        c.blockSignals(True)
        c.clear()
        c.addItem('Whole stack', 'stack')
        for i in range(1, len(L) - 1):
            c.addItem(label[i], i)
        c.addItem('Several layers', 'several')
        c.setCurrentIndex(max(0, c.findData(
            'several' if isinstance(t, tuple) else t)))
        c.blockSignals(False)
        self._fill_spread_layers(t if isinstance(t, tuple) else ())

    def _fill_spread_layers(self, ticked):
        """The ticks of 'Several layers': each interior layer, those in
        `ticked` ticked; shown only for that target."""
        lw, L = self.spread_layers, self.stack.layers
        names = [l.name for l in L]
        lw.blockSignals(True)
        lw.clear()
        for i in range(1, len(L) - 1):
            it = QtWidgets.QListWidgetItem(
                '#%d %s' % (i, names[i]) if names.count(names[i]) > 1
                else names[i])
            it.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if i in ticked
                             else QtCore.Qt.CheckState.Unchecked)
            lw.addItem(it)
        lw.blockSignals(False)
        lw.setVisible(self.spread_target.currentData() == 'several')

    def _spread_target_changed(self, *_):
        """Picking 'Several layers' starts from the current target's
        layers ticked, so the model does not change until a tick does."""
        if self.spread_target.currentData() == 'several':
            self._fill_spread_layers(self.stack._spread_layers())
        else:
            self.spread_layers.hide()
        self.schedule()

    def _spread_target(self):
        """The target the Spread box and ticks say; None: no layer ticked."""
        t = self.spread_target.currentData() or 'stack'
        if t != 'several':
            return t
        lw = self.spread_layers
        ticked = tuple(
            lw.item(r).data(QtCore.Qt.ItemDataRole.UserRole)
            for r in range(lw.count())
            if lw.item(r).checkState() == QtCore.Qt.CheckState.Checked)
        return ticked or None

    def _reindex_spread(self, old):
        """After layers were added, removed or moved: the spread follows its
        layer; if that layer is gone it is reset, and the status bar says
        so (Stack.reindex_spread_target)."""
        if self.stack.reindex_spread_target(old):
            self._load_stack_fit()
            msg = ('The thickness spread\'s layer was deleted: the spread is '
                   'reset (whole stack, 0 %, not fitted).')
            win = self.window()
            if isinstance(win, QtWidgets.QMainWindow):
                win.statusBar().showMessage(msg, 15000)
            else:
                self.status.setText(msg)

    def _spread_dist_changed(self, *_):
        """The Gaussian's bounds kept within its default upper bound; the
        log-normal's widened from the Gaussian's default."""
        dist = self.spread_dist.currentData()
        _, bmin, bmax, _ = self.stack_rows['thickness_spread']
        top = 100 * SPREAD_BOUND[dist]
        if dist == 'gaussian':
            for b in (bmin, bmax):
                b.setValue(min(b.value(), top))
        elif bmax.value() == 100 * SPREAD_BOUND['gaussian']:
            bmax.setValue(top)
        self.schedule()

    def _edit_stack_prior(self, attr):
        e = self.stack.fit_entry(attr)
        label, scale, unit = PARAM_DISPLAY[attr]
        dlg = PriorDialog(label, e.get('prior'), scale, unit, self)
        if dlg.exec():
            e['prior'] = dlg.spec
            show_prior(self.stack_priors[attr], dlg.spec, scale, unit)
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

    def _shown_licorne_exact(self):
        """Whether the combos and the stack's other Licorne settings (no
        widget) are all Stack.LICORNE_EXACT."""
        st = self.stack
        scheme, fun = RES_CONVOLUTIONS[self.res_conv.currentIndex()][1:]
        shown = {'roughness_scheme':
                 ROUGHNESS_SCHEMES[self.rscheme.currentIndex()],
                 'magnetic_smearing': SMEARING_MODES[self.msmear.currentIndex()],
                 'resolution.scheme': scheme}
        return all(shown[k] == v if k in shown else st._setting(k) == v
                   for k, v in st.LICORNE_EXACT.items())

    def _sync_licorne_exact(self, *_):
        """Tick the Licorne-exact box exactly when every Licorne setting
        holds; never runs _licorne_exact."""
        self.lic_exact.blockSignals(True)
        self.lic_exact.setChecked(self._shown_licorne_exact())
        self.lic_exact.blockSignals(False)

    def _licorne_exact(self, on):
        """Licorne-exact on / off (Stack.set_licorne_exact); the combos
        follow."""
        self.stack.set_licorne_exact(on)
        res = self.stack.resolution
        for w in (self.rscheme, self.msmear, self.res_conv):
            w.blockSignals(True)
        self.rscheme.setCurrentIndex(
            ROUGHNESS_SCHEMES.index(self.stack.roughness_scheme))
        self.msmear.setCurrentIndex(
            SMEARING_MODES.index(self.stack.magnetic_smearing))
        self.res_conv.setCurrentIndex(res_convolution(res))
        for w in (self.rscheme, self.msmear, self.res_conv):
            w.blockSignals(False)
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
            self.stack.scale = self.scale.value()
            self.stack.q_shift = self.q_shift.value()
            self.stack.thickness_spread = self.spread.value() / 100
            self.stack.thickness_spread_nodes = \
                self.spread_nodes.currentData()
            target = self._spread_target()
            if target is not None:           # none ticked: kept as it was
                self.stack.set_spread_target(target)
            self.stack.thickness_spread_dist = self.spread_dist.currentData()
            try:
                if target is None:
                    raise ValueError('Several layers: tick at least one; '
                                     'the spread keeps its previous layers.')
                spread_factors(self.stack.thickness_spread,
                               self.stack.thickness_spread_dist, 5)
                self.spread_error.hide()
            except ValueError as exc:       # a note, not a dialog
                self.spread_error.setText(str(exc))
                self.spread_error.show()
            for attr, (_, bmin, bmax, cb) in self.stack_rows.items():
                k = STACK_ROW_FACTOR.get(attr, 1.0)
                self.stack.fit[attr] = dict(
                    self.stack.fit.get(attr, {}), vary=cb.isChecked(),
                    min=bmin.value() / k, max=bmax.value() / k)
            self.stack.q_in_fronting = self.q_fronting.isChecked()
            self.stack.magnetic_smearing = SMEARING_MODES[
                self.msmear.currentIndex()]
            self.stack.roughness_scheme = ROUGHNESS_SCHEMES[
                self.rscheme.currentIndex()]
            self.stack.interface_order = INTERFACE_ORDERS[
                self.iorder.currentIndex()]
            self.tail.setEnabled(self.stack.roughness_scheme == 'rms')
            self.iorder.setEnabled(self.stack.roughness_scheme == 'rms')
            mode = RES_MODES[self.res_mode.currentIndex()]
            self.stack.resolution.update(
                enabled=self.res_on.isChecked(), mode=mode,
                tof_dlambda=self.res_tof_dl.value())
            scheme, fun = RES_CONVOLUTIONS[self.res_conv.currentIndex()][1:]
            self.stack.resolution['scheme'] = scheme
            if fun is not None:
                self.stack.resolution['licorne_fun'] = fun
            if mode == 'tof':
                self.stack.resolution['tof_angles'] = self._tof_angles()
            self._sync_licorne_exact()
            self._link_mixture()
            self.stack.build_sublayers()
            self._compute_profile()
            self._show_crossings()
            self._show_profile()
            self._draw_reflectance()
            self._store_partners()
            self._refresh_mixture_info()
            text = '%d layers, %d sublayers' % (len(self.stack.layers),
                                                len(self.stack.sublayers))
            if self._refl_note:
                text += ('<br><span style="color:%s">%s</span>'
                         % (style.ERROR, html.escape(self._refl_note)))
            if self._crossings:
                text += ('<br><span style="color:%s">%s (General parameters '
                         '→ Interfaces)</span>'
                         % (style.ERROR, html.escape(self._crossing_text())))
            res = self.stack.resolution
            if res['enabled'] and res.get('scheme') == 'licorne':
                text += ('<br><i>Licorne convolution: this curve depends on '
                         'its Q grid; compare with data on the data\'s Q '
                         '(the Experimental tab\'s live curves are)</i>')
            self.status.setText(text)
            self.status.setStyleSheet('')
            self.prof_cursor.refresh()
            self.refl_cursor.refresh()
            self.reflectanceChanged.emit()
        except Exception as exc:                      # keep the UI alive
            self.status.setText('Error: %s' % exc)
            self.status.setStyleSheet('color: %s' % style.ERROR)

    def _sync_resolution(self, *_):
        on = self.res_on.isChecked()
        self.res_mode.setEnabled(on)
        self.res_conv.setEnabled(on)
        self.res_pages.setCurrentIndex(self.res_mode.currentIndex())
        self.res_pages.setEnabled(on)
        res = self.stack.resolution
        n = len(res.get('data_q', ()))
        self.res_data_info.setText(
            'dQ of %s (%d Q, overlap: %s); a fit uses each point\'s '
            'own dQ, and each dataset\'s own table' % (res.get('data_set') or '?', n,
                        {'first': 'earlier angle', 'last': 'later angle'}.get(
                            res.get('data_overlap'), 'mean')) if n else
            'No dQ yet: choose a dataset')
        self.schedule()

    def resolution_shown(self):
        """Copy of the resolution parameters the widgets show (mode
        included), whether the resolution is on or not."""
        res = copy.deepcopy(self.stack.resolution)
        res.update(mode=RES_MODES[self.res_mode.currentIndex()],
                   tof_dlambda=self.res_tof_dl.value())
        if res['mode'] == 'tof':
            res['tof_angles'] = self._tof_angles()
        return res

    def set_resolution_spreads(self, res):
        """The spreads of res (Δλ and the angles' Δθ) into the widgets, and
        the resolution on."""
        self.res_tof_dl.setValue(res['tof_dlambda'])
        self._set_tof_angles(res['tof_angles'])
        self.res_mode.setCurrentIndex(RES_MODES.index(res['mode']))
        self.res_on.setChecked(True)
        self.schedule()

    def set_data_dq(self, name, Q, sigma, overlap='mean'):
        """The 'data' mode with the dQ table (Q, sigma_Q) of dataset
        `name` (overlap: the rule the table was made with,
        tabs.resolution.dq_table), and the resolution on."""
        Q, sigma = np.asarray(Q, dtype=float), np.asarray(sigma, dtype=float)
        q, inv = np.unique(Q, return_inverse=True)    # a repeated Q: the mean
        dq = np.bincount(inv, sigma) / np.bincount(inv)
        self.stack.resolution.update(data_q=q.tolist(), data_dq=dq.tolist(),
                                     data_set=name,
                                     data_overlap=overlap)
        self.res_mode.setCurrentIndex(RES_MODES.index('data'))
        self.res_on.setChecked(True)
        self._sync_resolution()

    def _show_resolution(self):
        # imported here: tabs.resolution takes plot_widget from this module
        from tabs.resolution import ResolutionDialog
        dlg = getattr(self, '_res_dialog', None)
        if dlg is None:
            dlg = self._res_dialog = ResolutionDialog(self, self)
        dlg.refresh_datasets()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

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
        self._crossings = crossing_warnings(self.stack)

    def _crossing_text(self):
        """'Additive interfaces cross: CrSb Mx -163 %' (see CROSSING_WARN)."""
        names = [l.name for l in self.stack.layers]
        return 'Additive interfaces cross: ' + ', '.join(
            '%s %s %+.0f %%' % (names[c['host']], CROSSING_NAMES[q], v)
            for c, q, v in self._crossings)

    def _show_crossings(self):
        """The warning beside the Interfaces combo (see CROSSING_WARN)."""
        names = [l.name for l in self.stack.layers]
        tips = []
        for c, q, v in self._crossings:
            host = names[c['host']]
            tips.append(
                '%s: the interfaces of %s cross (its occupancy goes down to '
                '%.3g at %.1f Å); the additive %s goes %+.0f %% of %s\'s '
                'value past the layer values over %.1f – %.1f Å (shaded).'
                % (host, names[c['layer']], c['occupancy'], c['z'],
                   CROSSING_NAMES[q], v, host, *c['zrange']))
        self.iorder_warn.setText(
            '<span style="color:%s">⚠ %s</span>'
            % (style.ERROR, html.escape(self._crossing_text()))
            if self._crossings else '')
        self.iorder_warn.setToolTip(
            '\n'.join(tips + ['Interfaces ordered keeps every quantity '
                              'within the layer values.']) if tips else '')

    def _show_profile(self, *_):
        """Draw the checked quantities from the cached profile."""
        if self._profile is None:
            return
        z, Z, e, curves, slabs = self._profile
        p = self.prof_plot
        show_bars = self.prof_show_bars.isChecked()
        show_smooth = self.prof_show_smooth.isChecked()

        if show_bars:
            curves = [self._on_bars.get(i, c) for i, c in enumerate(curves)]
        for i, cb in enumerate(self.prof_boxes):
            on = self.prof_right.angle_on(i)
            self.prof_curves[i].setData(z, curves[i], connect='finite')
            self.prof_curves[i].setVisible(on and show_smooth)
            # a blank angle (NaN) is a bar of zero height
            self.prof_bars[i].setOpts(x0=e[:-1], width=np.diff(e),
                                      height=np.nan_to_num(slabs[i]), y0=0)
            self.prof_bars[i].setVisible(on and show_bars)
        self.prof_right.draw(self.stack)
        p.showAxis('left', any(cb.isChecked() for i, cb in
                               enumerate(self.prof_boxes) if i not in RIGHT))

        for item in self.prof_decor:
            p.removeItem(item)
        self.prof_decor = []
        shade = QtGui.QColor(style.ERROR)
        shade.setAlpha(45)
        for c, _, _ in self._crossings:
            region = pg.LinearRegionItem(c['zrange'], movable=False,
                                         brush=shade, pen=pg.mkPen(None))
            region.setZValue(-60)
            p.addItem(region, ignoreBounds=True)
            self.prof_decor.append(region)
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
        self._show_layer_page()

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
        # log R / R·Q⁴ only apply to raw reflectivities
        self.r_scale.setEnabled(is_refl)
        rq4 = is_refl and self.r_scale.currentData() == 'rq4'

        if name == POLARIZED:
            shown = zip(self.refl_curves, (R[c] for c in CHANNELS),
                        ['R' + c for c in CHANNELS])
        elif name == HALF_POLARIZED:
            shown = zip(self.hp_curves, (R['+'], R['-']), ['R+', 'R−'])
        elif name == UNPOLARIZED:
            shown = [(self.unpol_curve, 0.5 * (R['+'] + R['-']),
                      '(R+ + R−)/2')]
        else:
            shown = [(self.asym_curve, refl_quantity(name, R), name)]
        shown = list(shown)

        p = self.refl_plot
        legend = p.legend
        legend.clear()
        # clear unused curves rather than hide them, so visibility toggled
        # from the legend survives a change of plotted quantity
        used = {id(c) for c, _, _ in shown}
        for curve in self.refl_curves + self.hp_curves + [self.asym_curve,
                                                         self.unpol_curve]:
            if id(curve) not in used:
                curve.setData([], [])
        self.asym_zero.setVisible(not is_refl)
        self.refl_formula.setText(formula)

        if is_refl:
            log = not rq4
            p.setLabel('left', 'R·Q⁴ (Å⁻⁴)' if rq4 else 'Reflectivity')
        else:
            log = False
            p.setLabel('left', label)
        # a frozen y range only makes sense for the same plotted quantity;
        # set before setData, which autoranges straight away
        ykey = (self.refl_quantity.currentIndex(), log, rq4)
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
            if rq4:
                y = y * Q**4
            if log:
                y = np.where(y > 0, y, np.nan)
            curve.setData(Q, y)
            legend.addItem(curve, cname)
            self._refl_shown.append((curve, cname, Q, y))

    # -- models -------------------------------------------------------------
    def _refresh_models(self):
        """The Model box from self._models, its current item the shown one;
        modelsChanged tells the Experimental tab (a dataset per model)."""
        b = self.model_box
        b.blockSignals(True)
        b.clear()
        b.addItems([m['name'] for m in self._models])
        b.setCurrentIndex(self._model_index)
        b.blockSignals(False)
        self.model_btns['delete'].setEnabled(
            len(self._models) > 1 and not self._locked)
        self.modelsChanged.emit()

    def model_names(self):
        return [m['name'] for m in self._models]

    def shown_model(self):
        """The name of the model the tab shows."""
        return self._models[self._model_index]['name']

    def stored_model(self, name):
        """The stored dict (Stack.to_dict) of model `name`, None for the
        shown one (its stack is the tab's) or an unknown name."""
        return next((m['model'] for m in self._models if m['name'] == name),
                    None)

    def model_stack(self, name):
        """The Stack of model `name` as a fit or a curve takes it: the
        tab's own for the shown model, else a new one from its stored dict
        with the models of its mixture linked (Stack._mix_stacks); None for
        an unknown name."""
        if name == self.shown_model():
            return self.stack
        d = self.stored_model(name)
        if d is None:
            return None
        st = Stack.from_dict(copy.deepcopy(d))
        dicts = {m['name']: m['model'] for m in self._models
                 if m['model'] is not None}
        dicts[self.shown_model()] = self.stack.to_dict()
        st._mix_stacks = {e['name']: Stack.from_dict(dicts[e['name']])
                          for e in st.mixture if e['name'] in dicts}
        return st

    def _edit_stored(self, name, edit):
        """edit(stack) on a Stack of stored model `name`, written back as a
        new dict (so that whatever caches the old one sees the change)."""
        m = next(m for m in self._models if m['name'] == name)
        st = Stack.from_dict(copy.deepcopy(m['model']))
        edit(st)
        m['model'] = st.to_dict()
        self.modelsChanged.emit()

    def set_model_parameters(self, name, params, x):
        """set_parameters into model `name`, shown or stored."""
        if name == self.shown_model():
            self.set_parameters(params, x)
            return

        def edit(st):
            for (i, attr, *_), v in zip(params, x):
                setattr(st.owner(i), attr, float(v))
        self._edit_stored(name, edit)

    def set_model_varied(self, name, edit):
        """edit(stack) changes the Fit ticks of model `name` (e.g.
        Stack.set_varied, tick_all, untick_all), shown or stored; the
        shown one's editor and boxes follow."""
        if name != self.shown_model():
            self._edit_stored(name, edit)
            return
        edit(self.stack)
        self._load_stack_fit()
        self._refresh_mixture_table()
        self.refresh_list(select=self.list.currentRow())
        self.schedule()

    def set_model_bounds(self, name, params, lo, hi):
        """set_bounds into model `name`, shown or stored."""
        if name == self.shown_model():
            self.set_bounds(params, lo, hi)
            return
        self._edit_stored(name, lambda st: self._write_bounds(
            st, params, lo, hi))

    def _unique_name(self, base):
        names = set(self.model_names())
        if base not in names:
            return base
        k = 2
        while '%s (%d)' % (base, k) in names:
            k += 1
        return '%s (%d)' % (base, k)

    def _shown_model(self):
        """(model dict, its 'simulation' settings without 'models') of the
        shown model, the widgets' values included."""
        d = self.session_state()
        sim = d.pop('simulation')
        sim.pop('models', None)
        return copy.deepcopy(d), sim

    def switch_model(self, i):
        """Show model i: the shown one is kept as it is (MODELS)."""
        if i == self._model_index or not 0 <= i < len(self._models) or \
                self._locked:
            self._refresh_models()
            return
        d, sim = self._shown_model()
        cur, new = self._models[self._model_index], self._models[i]
        state = copy.deepcopy(new['model'])
        state['simulation'] = dict(sim, selected_layer=new['selected_layer'])
        try:
            self.restore_state(state, models=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Switch model', 'Could not load %s:\n%s'
                % (new['name'], exc))
            d['simulation'] = sim
            self.restore_state(d, models=False)
            self._refresh_models()
            return
        cur.update(model=d, selected_layer=sim['selected_layer'])
        new.update(model=None)
        self._model_index = i
        self._refresh_models()
        self._refresh_mixture_table()
        self._refresh_lateral(keep=False)
        self.recompute()                     # this model's mixture linked

    def _add_model(self, name, d, selected_layer=1):
        """Append model dict d as `name` and show it."""
        self._models.append(dict(name=self._unique_name(name), model=d,
                                 selected_layer=selected_layer))
        self.switch_model(len(self._models) - 1)

    def new_model(self):
        """No layer (empty_stack) with the shown model's other
        settings, as Import profile replaces only the layers."""
        if self._locked:
            return
        d, _ = self._shown_model()
        d['layers'] = empty_stack().to_dict()['layers']
        d['ties'] = []
        d['thickness_spread_target'] = 'stack'
        self._add_model('Model %d' % (len(self._models) + 1), d)

    def copy_model(self):
        if self._locked:
            return
        d, sim = self._shown_model()
        self._add_model(self._models[self._model_index]['name'] + ' copy', d,
                        sim['selected_layer'])

    def rename_model(self):
        if self._locked:
            return
        m = self._models[self._model_index]
        name, ok = QtWidgets.QInputDialog.getText(
            self, 'Rename model', 'Name:', text=m['name'])
        name = name.strip()
        if not ok or not name or name == m['name']:
            return
        old, m['name'] = m['name'], self._unique_name(name)
        self._rename_in_mixtures(old, m['name'])
        self.modelRenamed.emit(old, m['name'])
        self._refresh_models()
        self._refresh_mixture_table()

    def delete_model(self):
        """Delete the shown model (after asking) and show its neighbour."""
        if self._locked or len(self._models) < 2:
            return
        name = self._models[self._model_index]['name']
        if QtWidgets.QMessageBox.question(
                self, 'Delete model', 'Delete the model %s?' % name) != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.remove_model(name)

    def remove_model(self, name):
        """Delete model `name` without asking (the shown one: its neighbour
        is shown first); never the last one."""
        names = self.model_names()
        if name not in names or len(names) < 2:
            return
        gone = names.index(name)
        if gone == self._model_index:
            self.switch_model(gone - 1 if gone > 0 else 1)
            if self._model_index == gone:      # could not switch
                return
        del self._models[gone]
        if self._model_index > gone:
            self._model_index -= 1
        self._rename_in_mixtures(name, None)
        self._refresh_models()
        self._refresh_mixture_table()
        self.schedule()

    def add_model(self, name, d):
        """Append model dict d (Stack.to_dict) as `name`, not shown."""
        self._models.append(dict(name=self._unique_name(name), model=d,
                                 selected_layer=1))
        self._refresh_models()
        self._refresh_mixture_table()
        return self._models[-1]['name']

    def model_dict(self, name):
        """A copy of model `name` as a dict (Stack.to_dict), the shown one
        with its widgets' values."""
        if name == self.shown_model():
            return self._shown_model()[0]
        return copy.deepcopy(self.stored_model(name))

    # -- mixture (model.stack MIXTURE) -------------------------------------
    def _refresh_mixture_table(self):
        """One row per other model: used, its weight, bounds and Fit."""
        t = self.mix_table
        entries = {e['name']: e for e in self.stack.mixture}
        others = [m['name'] for k, m in enumerate(self._models)
                  if k != self._model_index]
        for r in range(t.rowCount()):         # gone at once, not later
            for c in range(t.columnCount()):
                w = t.cellWidget(r, c)
                if w is not None:
                    w.hide()
        t.setRowCount(0)
        t.setRowCount(len(others))
        self._mix_rows = []
        for r, name in enumerate(others):
            e = entries.get(name)
            fit = e['fit'] if e else default_fit('weight')
            use = QtWidgets.QCheckBox()
            use.setChecked(e is not None)
            use.setStyleSheet('margin-left: 6px')
            w, lo, hi = (dspin(0.0, 1.0, 4, 0.01) for _ in range(3))
            w.setValue(e['weight'] if e else 0.0)
            lo.setValue(fit['min'])
            hi.setValue(fit['max'])
            cb = QtWidgets.QCheckBox()
            cb.setChecked(bool(fit.get('vary')))
            for c, widget in enumerate((use, None, w, lo, hi, cb)):
                if widget is None:
                    it = QtWidgets.QTableWidgetItem(name)
                    it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                    t.setItem(r, c, it)
                else:
                    t.setCellWidget(r, c, widget)
            for widget in (w, lo, hi):
                widget.valueChanged.connect(self._mix_edited)
            for widget in (use, cb):
                widget.toggled.connect(self._mix_edited)
            self._mix_rows.append((name, use, w, lo, hi, cb))
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._refresh_lateral()
        self._refresh_mixture_info()

    def _mix_edited(self, *_):
        """The table into Stack.mixture, in the Model box's order."""
        mix = []
        for name, use, w, lo, hi, cb in self._mix_rows:
            for widget in (w, lo, hi, cb):
                widget.setEnabled(use.isChecked())
            if use.isChecked():
                mix.append(dict(name=name, weight=w.value(), fit=dict(
                    vary=cb.isChecked(), min=lo.value(), max=hi.value())))
        self.stack.mixture = mix
        self.schedule()

    def _refresh_mixture_info(self):
        """The weight w0 of this model, the shared-thickness ticks and the
        summary in General parameters."""
        for name, use, w, lo, hi, cb in getattr(self, '_mix_rows', []):
            for widget in (w, lo, hi, cb):
                widget.setEnabled(use.isChecked())
        st = self.stack
        w0, parts = st.mixture_parts()
        name = self._models[self._model_index]['name']
        self.mix_w0.setText('Weight of %s (this model): w₀ = %.4g' % (name, w0)
                            if parts else 'No other model mixed in.')
        self._refresh_mixture_layers()
        bits = []
        if st.thickness_spread:
            bits.append('%.3g %% (%s, %s)' % (
                100 * st.thickness_spread, st.spread_target_name(),
                st.thickness_spread_dist))
        if st.mixture:
            bits.append('mixture of %d models' % (len(st.mixture) + 1))
        self.spread_summary.setText('; '.join(bits) or 'none')

    def _refresh_mixture_layers(self):
        """A tick per film layer: its thickness Fit, and which mixed
        models have a layer of that name."""
        mixed = {n: st for n, st in self.stack._mix_stacks.items()
                 if any(e['name'] == n for e in self.stack.mixture)}
        rows = []
        for i, l in enumerate(self.stack.layers[1:-1], 1):
            also = [n for n, st in mixed.items()
                    if any(x.name == l.name for x in st.layers)]
            rows.append((i, l.name, tuple(also)))
        lay = self.mix_layers.layout()
        if rows != self._mix_layer_key:
            self._mix_layer_key = rows
            while lay.count():
                w = lay.takeAt(0).widget()
                if w is not None:
                    w.hide()
                    w.deleteLater()
            self._mix_layer_boxes = []
            for i, name, also in rows:
                cb = QtWidgets.QCheckBox('%s thickness%s' % (
                    name, '  (also in %s)' % ', '.join(also) if also else
                    '  (only in this model)' if mixed else ''))
                cb.toggled.connect(lambda on, i=i: self._shared_toggled(i, on))
                lay.addWidget(cb)
                self._mix_layer_boxes.append((i, cb))
        for i, cb in self._mix_layer_boxes:
            cb.blockSignals(True)
            cb.setChecked(bool(self.stack.layers[i].fit.get(
                'thickness', {}).get('vary')))
            cb.blockSignals(False)

    def _shared_toggled(self, i, on):
        if self._locked:
            return
        self.stack.layers[i].fit_entry('thickness')['vary'] = on
        self._on_select(self.list.currentRow())     # the editor's Fit box
        self.schedule()

    def _link_mixture(self):
        """Stack._mix_stacks: a Stack per mixed model, from its stored
        dict, rebuilt only when that dict was replaced."""
        stored = {m['name']: m['model'] for m in self._models
                  if m['model'] is not None}
        links = {}
        for e in self.stack.mixture:
            d = stored.get(e['name'])
            if d is None:
                continue
            c = self._mix_cache.get(e['name'])
            if c is None or c[0] is not d:
                c = self._mix_cache[e['name']] = (d, Stack.from_dict(d))
            links[e['name']] = c[1]
        self.stack._mix_stacks = links

    def _store_partners(self):
        """The shared values the mixed models took (Stack.sync_mixture)
        into their stored dicts, in place: switching to one shows them."""
        layers = self.stack.mixture_shared()
        names = {n for n, _ in layers}
        for name, st in self.stack._mix_stacks.items():
            d = self._mix_cache[name][0]
            for ld, l in zip(d['layers'], st.layers):
                if l.name in names:
                    for n, attr in layers:
                        if n == l.name:
                            ld[attr] = getattr(l, attr)

    def _rename_in_mixtures(self, old, new):
        """Every model's mixture entry naming `old` (None: removed)."""
        lists = [self.stack.mixture] + [m['model'].setdefault('mixture', [])
                                        for m in self._models
                                        if m['model'] is not None]
        for mix in lists:
            for e in list(mix):
                if e['name'] == old:
                    if new is None:
                        mix.remove(e)
                    else:
                        e['name'] = new
        self._mix_cache.clear()

    # -- save / load --------------------------------------------------------
    def session_state(self):
        """The shown model (Stack.to_dict) plus the tab's own settings under
        'simulation': Q grid, polarisation, display and the other models
        (MODELS)."""
        self.recompute()                     # the widgets' values into the stack
        d = self.stack.to_dict()
        d['simulation'] = {
            'qmin': self.qmin.value(), 'qmax': self.qmax.value(),
            'nq': self.nq.value(), 'polarisation': self.pol.vectors(),
            'display': {
                'quantity_name': self.refl_quantity.currentText(),
                'scale': self.r_scale.currentData(),
                'freeze_y': self.freeze_y.isChecked(),
                'profile': [cb.isChecked() for cb in self.prof_boxes],
                **self.prof_right.state(),
                'sublayers': self.prof_show_bars.isChecked(),
                'smooth': self.prof_show_smooth.isChecked()},
            'selected_layer': self.list.currentRow(),
            'models': {'current': self._model_index,
                       'list': copy.deepcopy(self._models)}}
        return d

    def restore_state(self, d, models=True):
        """Inverse of session_state; a plain model file (no 'simulation')
        keeps the tab's settings and replaces only the shown model, a session
        saved before MODELS becomes the only model.  models=False: only the
        shown model, whatever d holds (switch_model)."""
        new = Stack.from_dict(d)                 # validate before touching UI
        sim = d.get('simulation')
        models = None if not models or sim is None else sim.get(
            'models', {'current': 0, 'list': [dict(
                name='Model 1', model=None,
                selected_layer=sim.get('selected_layer', 1))]})
        if models is not None:
            cur, lst = int(models['current']), copy.deepcopy(models['list'])
            if not 0 <= cur < len(lst) or lst[cur]['model'] is not None:
                raise ValueError('the session\'s models are inconsistent')
            for m in lst:
                if m['model'] is not None:
                    Stack.from_dict(m['model'])
            self._models, self._model_index = lst, cur
            self._refresh_models()
        self.stack.layers = new.layers
        self.stack.tail = new.tail
        self.stack.background = new.background
        self.stack.scale = new.scale
        self.stack.q_shift = new.q_shift
        self.stack.thickness_spread = new.thickness_spread
        self.stack.thickness_spread_nodes = new.thickness_spread_nodes
        self.stack.thickness_spread_target = new.thickness_spread_target
        self.stack.thickness_spread_dist = new.thickness_spread_dist
        self.stack.fit = new.fit
        self.stack.q_in_fronting = new.q_in_fronting
        self.stack.resolution = new.resolution
        self.stack.magnetic_smearing = new.magnetic_smearing
        self.stack.roughness_scheme = new.roughness_scheme
        self.stack.interface_order = new.interface_order
        self.stack.licorne_renorm = new.licorne_renorm
        self.stack.licorne_outer = new.licorne_outer
        self.stack.step_fallback = new.step_fallback
        self.stack.ties = new.ties
        self.stack.mixture = new.mixture
        # unticking Licorne-exact must not bring back the previous session's
        # settings
        self.stack._before_exact = new._before_exact
        sim = d.get('simulation', {})
        widgets = [self.qmin, self.qmax, self.nq, self.tail, self.pol,
                   self.background, self.q_fronting, self.res_on,
                   self.msmear, self.rscheme, self.iorder, self.res_mode,
                   self.res_conv, self.res_tof_dl]
        for w in widgets:
            w.blockSignals(True)
        self.qmin.setValue(sim.get('qmin', self.qmin.value()))
        self.qmax.setValue(sim.get('qmax', self.qmax.value()))
        self.nq.setValue(sim.get('nq', self.nq.value()))
        self.tail.setValue(new.tail)
        self.pol.set_vectors(sim.get('polarisation', default_vectors()))
        self._load_stack_fit()
        self.q_fronting.setChecked(new.q_in_fronting)
        self.msmear.setCurrentIndex(
            SMEARING_MODES.index(new.magnetic_smearing))
        self.rscheme.setCurrentIndex(
            ROUGHNESS_SCHEMES.index(new.roughness_scheme))
        self.iorder.setCurrentIndex(
            INTERFACE_ORDERS.index(new.interface_order))
        res = new.resolution
        self.res_on.setChecked(res['enabled'])
        self.res_mode.setCurrentIndex(RES_MODES.index(res['mode']))
        self.res_conv.setCurrentIndex(res_convolution(res))
        self.res_tof_dl.setValue(res['tof_dlambda'])
        self._set_tof_angles(res['tof_angles'])
        disp = sim.get('display', {})
        display = [self.refl_quantity, self.r_scale, self.freeze_y,
                   self.prof_show_bars, self.prof_show_smooth] + self.prof_boxes
        for w in display:
            w.blockSignals(True)
        self.refl_quantity.setCurrentIndex(quantity_index(
            disp, self.refl_quantity.currentIndex()))
        if disp:
            set_r_scale(self.r_scale, saved_r_scale(disp))
        for key, cb in (('freeze_y', self.freeze_y),
                        ('sublayers', self.prof_show_bars),
                        ('smooth', self.prof_show_smooth)):
            cb.setChecked(disp.get(key, cb.isChecked()))
        for cb, on in zip(self.prof_boxes, disp.get('profile', [])):
            cb.setChecked(on)
        if disp:
            self.prof_right.restore(disp)
        for w in widgets + display:
            w.blockSignals(False)
        self._sync_resolution()
        self.refresh_list(select=sim.get('selected_layer', 1))
        self._mix_cache.clear()
        self._refresh_mixture_table()
        self._refresh_lateral(keep=False)
        self.recompute()
