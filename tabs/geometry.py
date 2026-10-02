"""
Geometry tab: a 3-D view (pyqtgraph.opengl) of what a measurement sees, for
one layer of the Simulation tab's stack and one of its channels.

    sample    the film plane x-y (sample frame of model.stack ANGLES), z the
              film normal, along Q
    beam      incoming k_i and outgoing k_f (specular), at a drawn incidence
              angle (exaggerated: real angles are a few degrees at most)
              and an azimuth in the film plane, with the incident
              polarisation Pi on k_i and the analysed one Pa on k_f
              (length = efficiency; no Pa arrow without an analyser) and
              the polarisation axis (along Pi) through the origin
    M         the layer's magnetisation, its in-plane projection (the only
              part the neutrons see, model.stack HALPERIN) and that
              projection split along P (non-spin-flip: splits R++ / R--)
              and perpendicular to P (spin-flip: R+- / R-+ go as its
              square), with its angles as arcs: phi in the film plane from
              x (MSLD_theta) and theta, either SANR's elevation out of the
              film plane (MSLD_phi, 0 = in plane) or, with 'Licorne
              angles', Licorne's polar angle from z: 90 deg - elevation
              (90 = in plane), as in Licorne's msld = [rho, phi, theta]
              when the import kept Licorne's axes (model.licorne_io ANGLES)

The specular reflectivity does not depend on the beam azimuth (only M
relative to P enters the transfer matrix); 90 deg, the default, puts the
scattering plane in y-z with P along x perpendicular to it, the usual guide
field.  The layer follows the Simulation tab's selection both ways; the view
is redrawn after every recompute there.
"""

import html

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtGui, QtWidgets

from model.polarisation import CHANNEL_NAMES
from model.stack import cos_turns, sin_turns
from style import MUTED, PLOT_BG
from tabs.simulation import PROFILE_QUANTITIES, SLD_SCALE

BEAM = '#d0d4da'
PI_COLOUR, PA_COLOUR = '#4c8dff', '#b392f0'
M_COLOUR = PROFILE_QUANTITIES[1][2]
THETA_COLOUR, PHI_COLOUR = PROFILE_QUANTITIES[2][2], PROFILE_QUANTITIES[3][2]
PAR_COLOUR, PERP_COLOUR = '#f2c94c', '#f87171'
PLATE = (0.55, 0.58, 0.63, 0.22)
PLATE_HALF, PLATE_DEPTH = 3.0, 0.25    # scene units
BEAM_LEN, P_LEN, M_LEN = 4.6, 1.3, 2.3
CHANNEL_NOTES = {
    '++': 'non-spin-flip: nuclear + M∥',
    '--': 'non-spin-flip: nuclear − M∥',
    '+-': 'spin-flip: |M⊥|²',
    '-+': 'spin-flip: |M⊥|²',
    '+': 'no analyser: R++ + R+− (nuclear + M∥, plus |M⊥|²)',
    '-': 'no analyser: R−− + R−+ (nuclear − M∥, plus |M⊥|²)',
}


def rgba(colour, alpha=1.0):
    c = pg.mkColor(colour)
    return (c.redF(), c.greenF(), c.blueF(), alpha)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def _place(item, start, d):
    """Rotate item's +z onto the unit vector d, then move it to start."""
    axis = np.cross([0.0, 0.0, 1.0], d)
    if np.linalg.norm(axis) < 1e-9:
        if d[2] < 0:
            item.rotate(180, 1, 0, 0)
    else:
        item.rotate(np.degrees(np.arccos(np.clip(d[2], -1, 1))), *axis)
    item.translate(*start)


def arrow(view, start, vec, colour, radius=0.035, head=0.22):
    """Solid arrow from start along vec (a mesh: GL line widths above 1 are
    not available everywhere)."""
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return
    d = vec / length
    head = min(head, 0.5 * length)
    c = rgba(colour)
    shaft = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(
        rows=1, cols=16, radius=[radius, radius], length=length - head),
        color=c, smooth=True, shader='shaded')
    _place(shaft, np.asarray(start, float), d)
    tip = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(
        rows=1, cols=20, radius=[2.6 * radius, 0.0], length=head),
        color=c, smooth=True, shader='shaded')
    _place(tip, np.asarray(start, float) + d * (length - head), d)
    view.addItem(shaft)
    view.addItem(tip)


def dashed(view, a, b, colour, alpha=0.8, dash=0.12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.linalg.norm(b - a) / dash), 1)
    t = np.linspace(0, 1, n + 1)
    pts = a + np.outer(t, b - a)
    pts = pts[: len(pts) // 2 * 2]              # pairs: dash, gap, dash...
    if len(pts):
        view.addItem(gl.GLLinePlotItem(pos=pts, mode='lines',
                                       color=rgba(colour, alpha),
                                       antialias=True))


def line(view, pts, colour, alpha=1.0):
    view.addItem(gl.GLLinePlotItem(pos=np.asarray(pts, float),
                                   mode='line_strip',
                                   color=rgba(colour, alpha), antialias=True))


def label(view, pos, text, colour, size=11):
    font = QtGui.QFont()
    font.setPointSize(size)
    view.addItem(gl.GLTextItem(pos=np.asarray(pos, float), text=text,
                               color=pg.mkColor(colour), font=font))


def plate_mesh():
    """The sample: a flat box, top face at z = 0."""
    h, d = PLATE_HALF, PLATE_DEPTH
    v = np.array([[x, y, z] for z in (-d, 0.0) for y in (-h, h)
                  for x in (-h, h)])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 5, 7], [4, 7, 6],
                  [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 3, 7], [1, 7, 5]])
    return gl.MeshData(vertexes=v, faces=f)


def fmt(v):
    return '%.3g' % v if abs(v) >= 5e-4 or v == 0 else '%.2e' % v


def vec_text(v):
    return '(%s)' % ', '.join('%.3g' % (c + 0.0) for c in v)


class GeometryTab(QtWidgets.QWidget):

    def __init__(self, simulation, parent=None):
        super().__init__(parent)
        self.sim = simulation
        self._dirty = True

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(PLOT_BG)

        self.layer = QtWidgets.QComboBox()
        self.layer.setToolTip('Follows the layer selected in the Simulation '
                              'tab, and selects it there')
        self.layer.activated.connect(self._layer_picked)
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(['R' + c for c in CHANNEL_NAMES])
        self.channel.setToolTip('Pi and Pa of this channel, from the '
                                'Simulation tab\'s Polarisation box')
        self.channel.currentIndexChanged.connect(self.redraw)
        self.azimuth = QtWidgets.QDoubleSpinBox(
            minimum=-360, maximum=360, decimals=1, singleStep=15, value=90,
            suffix=' °', wrapping=True)
        self.azimuth.setToolTip(
            'Direction of the beam in the film plane, from sample x towards '
            'y.  The specular reflectivity does not depend on it (only M '
            'relative to P counts); 90° puts the scattering plane in y-z, '
            'with P along x perpendicular to it (the usual guide field).')
        self.azimuth.valueChanged.connect(self.redraw)
        self.incidence = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal,
                                           minimum=2, maximum=45, value=12)
        self.incidence.setToolTip('Drawn angle of incidence, exaggerated: '
                                  'real angles are a few degrees at most')
        self.incidence.valueChanged.connect(self.redraw)
        self.show_split = QtWidgets.QCheckBox('M ∥ P and M ⊥ P', checked=True)
        self.show_angles = QtWidgets.QCheckBox('angles φ, θ', checked=True)
        self.show_labels = QtWidgets.QCheckBox('labels', checked=True)
        self.licorne_angles = QtWidgets.QCheckBox('Licorne angles')
        self.licorne_angles.setToolTip(
            'θ as in Licorne\'s msld = [ρ, φ, θ]: the polar angle from the '
            'film normal z,\n  θ = 90° − elevation (90° = in plane), instead '
            'of the elevation out of the plane (0 = in plane).\n'
            'φ (in plane, from x) is the same in both.  The values equal '
            'Licorne\'s file when the import kept\n  Licorne\'s axes (M in '
            'Licorne\'s x-y plane, P along x); otherwise Licorne\'s axes were '
            'rotated.')
        for cb in (self.show_split, self.show_angles, self.show_labels,
                   self.licorne_angles):
            cb.toggled.connect(self.redraw)

        views = QtWidgets.QHBoxLayout()
        for text, elev, daz in (('3-D', 24, -60), ('Top', 90, -90),
                                ('Scattering plane', 0, -90),
                                ('Along beam', 8, 180)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _, e=elev, a=daz: self.set_camera(e, a))
            views.addWidget(b)

        form = QtWidgets.QFormLayout()
        form.addRow('Layer', self.layer)
        form.addRow('Channel', self.channel)
        form.addRow('Beam azimuth', self.azimuth)
        form.addRow('Incidence (drawn)', self.incidence)
        shows = QtWidgets.QVBoxLayout()
        for cb in (self.show_split, self.show_angles, self.show_labels,
                   self.licorne_angles):
            shows.addWidget(cb)
        form.addRow('Show', shows)

        self.readout = QtWidgets.QLabel(wordWrap=True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        side = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(side)
        sv.addLayout(form)
        sv.addWidget(QtWidgets.QLabel('View'))
        sv.addLayout(views)
        sv.addSpacing(8)
        sv.addWidget(self.readout, 1)
        side.setMinimumWidth(380)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(side)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1180])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.set_camera(24, -60)
        simulation.reflectanceChanged.connect(self._changed)
        simulation.list.currentRowChanged.connect(self._changed)

    # -- syncing -------------------------------------------------------------
    def _changed(self, *_):
        """Redraw now if shown, else when next shown."""
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._dirty:
            self.redraw()

    def _layer_picked(self, row):
        self.sim.list.setCurrentRow(row)

    def _sync_layers(self):
        layers = self.sim.stack.layers
        n = len(layers)
        names = ['%d  %s%s' % (i, l.name, ' [fronting]' if i == 0 else
                               ' [backing]' if i == n - 1 else '')
                 for i, l in enumerate(layers)]
        self.layer.blockSignals(True)
        if names != [self.layer.itemText(i)
                     for i in range(self.layer.count())]:
            self.layer.clear()
            self.layer.addItems(names)
        self.layer.setCurrentIndex(max(self.sim.list.currentRow(), 0))
        self.layer.blockSignals(False)

    def set_camera(self, elevation, d_azimuth):
        """Camera at `elevation`, at `d_azimuth` from the beam's azimuth."""
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(0, 0, 0), distance=10.5,
            elevation=elevation, azimuth=self.azimuth.value() + d_azimuth)

    # -- drawing -------------------------------------------------------------
    def redraw(self, *_):
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        self._sync_layers()
        st = self.sim.stack
        k = max(self.layer.currentIndex(), 0)
        L = st.layers[k]
        ch = CHANNEL_NAMES[self.channel.currentIndex()]
        Pi, Pa = (np.asarray(v, float) for v in self.sim.pol.vectors()[ch])
        labels = self.show_labels.isChecked()

        v = self.view
        v.clear()
        self._scene(v, labels)

        # beam: k_i comes down onto the origin, k_f leaves it, specular
        a = np.radians(self.azimuth.value())
        al = np.radians(self.incidence.value())
        u = np.array([np.cos(a), np.sin(a), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        ki = np.cos(al) * u - np.sin(al) * z
        kf = np.cos(al) * u + np.sin(al) * z
        arrow(v, -BEAM_LEN * ki, BEAM_LEN * ki * 0.97, BEAM, radius=0.045)
        arrow(v, [0, 0, 0], BEAM_LEN * kf, BEAM, radius=0.045)
        arrow(v, [0, 0, 0], 1.4 * z, MUTED, radius=0.025, head=0.16)
        pi_at, pa_at = -0.6 * BEAM_LEN * ki, 0.6 * BEAM_LEN * kf
        arrow(v, pi_at, P_LEN * Pi, PI_COLOUR, radius=0.04)
        if np.any(Pa):
            arrow(v, pa_at, P_LEN * Pa, PA_COLOUR, radius=0.04)
        p = unit(Pi)
        if p is not None:
            dashed(v, -3.4 * p, 3.4 * p, PI_COLOUR, 0.55)
        if labels:
            label(v, -1.02 * BEAM_LEN * ki, 'k_i', BEAM)
            label(v, 1.04 * BEAM_LEN * kf, 'k_f', BEAM)
            label(v, 1.5 * z, 'Q', MUTED)
            label(v, pi_at + P_LEN * Pi + 0.1, 'Pi', PI_COLOUR)
            if np.any(Pa):
                label(v, pa_at + P_LEN * Pa + 0.1, 'Pa', PA_COLOUR)
            if p is not None:
                label(v, 3.5 * p, 'P axis', PI_COLOUR, 9)

        # magnetisation, scaled to the largest |M| of the stack
        rho = L.MSLD_rho
        ref = max((abs(l.MSLD_rho) for l in st.layers), default=0.0)
        th, ph = L.MSLD_theta, L.MSLD_phi
        m_hat = np.array([cos_turns(ph) * cos_turns(th),
                          cos_turns(ph) * sin_turns(th), sin_turns(ph)])
        m_ip = m_hat * np.array([1.0, 1.0, 0.0])
        s = M_LEN * abs(rho) / ref if ref > 0 else 0.0
        sgn = np.sign(rho) if rho else 1.0
        M, Mip = s * sgn * m_hat, s * sgn * m_ip
        if s > 0:
            arrow(v, [0, 0, 0], M, M_COLOUR, radius=0.05, head=0.28)
            if abs(M[2]) > 1e-3:
                dashed(v, [0, 0, 0], Mip, M_COLOUR, 0.9)
                dashed(v, Mip, M, M_COLOUR, 0.5)
            if labels:
                label(v, 1.08 * M + [0, 0, 0.1], 'M', M_COLOUR, 12)
            par = np.dot(Mip, p) * p if p is not None else np.zeros(3)
            perp = Mip - par if p is not None else np.zeros(3)
            # one part zero: the other is M's in-plane projection already
            if self.show_split.isChecked() and \
                    min(np.linalg.norm(par), np.linalg.norm(perp)) > 1e-3:
                arrow(v, [0, 0, 0], par, PAR_COLOUR, radius=0.035)
                arrow(v, [0, 0, 0], perp, PERP_COLOUR, radius=0.035)
                dashed(v, par, Mip, PERP_COLOUR, 0.45)
                dashed(v, perp, Mip, PAR_COLOUR, 0.45)
                if labels:
                    label(v, 1.1 * par + [0, 0, 0.08], 'M∥', PAR_COLOUR)
                    label(v, 1.1 * perp + [0, 0, 0.08], 'M⊥', PERP_COLOUR)
            if self.show_angles.isChecked():
                self._arcs(v, M, labels)
        self.readout.setText(self._readout(st, k, ch, Pi, Pa))

    def _scene(self, v, labels):
        """Sample plate, its grid and the sample axes."""
        v.addItem(gl.GLMeshItem(meshdata=plate_mesh(), color=PLATE,
                                smooth=False, glOptions='translucent'))
        grid = gl.GLGridItem(color=(154, 160, 166, 50))
        grid.setSize(2 * PLATE_HALF, 2 * PLATE_HALF)
        grid.setSpacing(0.5, 0.5)
        v.addItem(grid)
        for e, name in ((np.eye(3)[0], 'x'), (np.eye(3)[1], 'y'),
                        (np.eye(3)[2], 'z')):
            far = (PLATE_HALF + 0.6) if name != 'z' else 3.0
            line(v, [[0, 0, 0], far * e], MUTED, 0.6)
            if labels:
                label(v, (far + 0.15) * e, name, MUTED, 12)

    def _arcs(self, v, M, labels):
        """phi in the film plane from x and theta, up from the film plane or
        (Licorne angles) down from z, of the drawn M (rho < 0 turns it by
        180 deg)."""
        t = np.arctan2(M[1], M[0])
        f = np.arctan2(M[2], np.hypot(M[0], M[1]))
        if self.licorne_angles.isChecked():
            self._polar_arc(v, M, t, labels)
            f = 0.0                     # no elevation arc
        r = 0.75
        s = np.linspace(0, t, 40)
        line(v, np.c_[r * np.cos(s), r * np.sin(s), np.zeros_like(s)],
             THETA_COLOUR)
        if labels and abs(t) > 0.05:
            label(v, [1.1 * r * np.cos(t / 2), 1.1 * r * np.sin(t / 2), 0.05],
                  'φ', THETA_COLOUR)
        if abs(f) > 1e-3:
            d = np.array([np.cos(t), np.sin(t), 0.0])
            r = 1.05
            s = np.linspace(0, f, 40)
            pts = r * (np.outer(np.cos(s), d) + np.outer(np.sin(s), [0, 0, 1]))
            line(v, pts, PHI_COLOUR)
            if labels:
                label(v, 1.12 * pts[20], 'θ', PHI_COLOUR)

    def _polar_arc(self, v, M, t, labels):
        """Licorne's theta: from the z axis down to M, in the vertical plane
        that holds M."""
        polar = np.arccos(np.clip(M[2] / np.linalg.norm(M), -1, 1))
        if polar < 1e-3:
            return
        d = np.array([np.cos(t), np.sin(t), 0.0])
        r = 1.05
        s = np.linspace(0, polar, 40)
        pts = r * (np.outer(np.sin(s), d) + np.outer(np.cos(s), [0, 0, 1]))
        line(v, pts, PHI_COLOUR)
        dashed(v, [0, 0, 0], [0, 0, r], PHI_COLOUR, 0.6)
        if labels:
            label(v, 1.12 * pts[20], 'θ (Licorne)', PHI_COLOUR)

    def _readout(self, st, k, ch, Pi, Pa):
        L = st.layers[k]
        rho = L.MSLD_rho / SLD_SCALE
        th, ph = L.MSLD_theta * 360, L.MSLD_phi * 360
        ip = rho * cos_turns(L.MSLD_phi)
        mz = rho * sin_turns(L.MSLD_phi)
        mip = ip * np.array([cos_turns(L.MSLD_theta),
                             sin_turns(L.MSLD_theta), 0.0])
        lic = self.licorne_angles.isChecked()
        if lic:
            angles = ('Licorne msld = [ρ, φ, θ] = [%s, %.4g°, %.4g°]'
                      '  (θ from z: 90° = in plane)'
                      % (fmt(rho), (th + 180) % 360 - 180, 90 - ph))
        else:
            angles = ('ρ = |M| = %s,  φ = %.4g° (in plane),  θ = %.4g° (out '
                      'of plane)' % (fmt(rho), th, ph))
        rows = ['<b>%s</b>  (layer %d)' % (html.escape(L.name), k), angles,
                '<span style="color:%s">Seen by the neutrons (in plane): '
                'ρ %s θ = %s</span>' % (M_COLOUR, 'sin' if lic else 'cos',
                                       fmt(ip))]
        if abs(mz) > 1e-9:
            rows.append('M<sub>z</sub> = ρ %s θ = %s: not seen (B<sub>z</sub> '
                        'is continuous across the surface)'
                        % ('cos' if lic else 'sin', fmt(mz)))
        rows.append('')
        rows.append('<b>R%s</b>: %s' % (ch, CHANNEL_NOTES[ch]))
        rows.append('<span style="color:%s">Pi = %s, |Pi| = %.3g</span>'
                    % (PI_COLOUR, vec_text(Pi), np.linalg.norm(Pi)))
        rows.append('<span style="color:%s">Pa = %s</span>'
                    % (PA_COLOUR, vec_text(Pa) + ', |Pa| = %.3g'
                       % np.linalg.norm(Pa) if np.any(Pa)
                       else '0 (no analyser)'))
        p = unit(Pi)
        if p is not None:
            par = float(np.dot(mip, p))
            perp = float(np.linalg.norm(mip - par * p))
            rows.append('<span style="color:%s">M∥ = %s along P '
                        '(non-spin-flip: R++ / R−− differ by it)</span>'
                        % (PAR_COLOUR, fmt(par)))
            rows.append('<span style="color:%s">M⊥ = %s in plane across P '
                        '(spin-flip: R+− and R−+ ∝ M⊥²)</span>'
                        % (PERP_COLOUR, fmt(perp)))
            if ip:
                ang = np.degrees(np.arccos(np.clip(
                    np.dot(mip / abs(ip), p), -1, 1)))
                rows.append('angle between in-plane M and P: %.4g°' % ang)
            q = unit(Pa)
            if q is not None and abs(abs(np.dot(p, q)) - 1) > 1e-6:
                rows.append('Pa is not along Pi: this channel mixes '
                            'non-spin-flip and spin-flip.')
            f = st.fronting_direction()
            if f is not None and np.linalg.norm(np.cross(p, f)) > 1e-6:
                rows.append('<span style="color:#f87171">Magnetic fronting: '
                            'only the part of P along its M survives.</span>')
        rows.append('')
        rows.append('<span style="color:%s">SLDs in 10⁻⁶ Å⁻². Arrows: M '
                    'scaled to the largest |M| of the stack, P to its '
                    'efficiency.</span>' % MUTED)
        return '<br>'.join(rows)
