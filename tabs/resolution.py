"""
Resolution dialog of the Simulation tab ("Show…" beside Resolution): sigma_Q
of the resolution parameters against Q, the dQ column of an imported dataset
beside it, and either the parameters fitted to that dQ or the dQ itself
("Use data dQ": the 'data' mode of model.stack RESOLUTION, in which a fit
averages each point over its own dQ and the dataset's dQ is the table for
curves at other Q).

FIT -- fit_resolution takes sigma_Q^2 / Q^2 = (dtheta/theta)^2 +
(dlambda/lambda)^2 (model.stack RESOLUTION), which is linear in the squared
spreads, and solves it by non-negative least squares over the data points
for the dtheta^2 of each angle of the 'tof' table (its Q band, as
model.stack._sigma_tof) and dlambda^2, (dlambda/lambda)^2 = dlambda^2 Q^2 /
(4 pi sin theta)^2; an angle with no data in its band keeps its dtheta.
Every point weighs the same in sigma_Q^2 / Q^2.  The dQ column is a standard
deviation; points with dQ <= 0 or Q <= 0 are left out.
"""

import copy

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from scipy.optimize import nnls

from model.stack import RESOLUTION_MODES
import style

DATA_COLOUR, MODEL_COLOUR, FIT_COLOUR = '#4c8dff', '#e0a458', '#4fc08d'


def sigma_q(Q, res):
    """sigma_Q of resolution dict res at Q > 0, on or not."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return RESOLUTION_MODES[res['mode']](np.asarray(Q, dtype=float), res)


def _tof_band(Q, angles):
    """Index of the angle used at each Q (model.stack._sigma_tof)."""
    edges = np.array([a['qmax'] for a in angles[:-1]], dtype=float)
    k = np.searchsorted(edges, Q, side='right')
    if len(edges) >= 2:
        k = np.where(Q == edges[-1], k - 1, k)
    return k


def fit_resolution(Q, dQ, res):
    """Resolution dict res with its spreads fitted to the points (Q, dQ)
    (see FIT), and the rms relative misfit of sigma_Q.  Raises ValueError
    when too few points are usable."""
    if res['mode'] == 'data':
        raise ValueError('the resolution is the data dQ: choose fixed θ per '
                         'Q band to fit spreads')
    if res['mode'] != 'tof':
        raise ValueError('unknown resolution mode %r' % (res['mode'],))
    Q, dQ = np.asarray(Q, dtype=float), np.asarray(dQ, dtype=float)
    ok = np.isfinite(Q) & np.isfinite(dQ) & (Q > 0) & (dQ > 0)
    Q, sig = Q[ok], dQ[ok]
    if Q.size < 2:
        raise ValueError('needs at least 2 points with dQ > 0')
    y = (sig / Q)**2
    new = copy.deepcopy(res)
    angles = new['tof_angles']
    theta = np.array([a['theta'] for a in angles], dtype=float)
    if np.any(theta <= 0):
        raise ValueError('TOF angles must be > 0')
    k = _tof_band(Q, angles)
    used = [j for j in range(len(angles)) if np.any(k == j)]
    cols = [(k == j) / theta[j]**2 for j in used]
    cols.append((Q / (4 * np.pi * np.sin(theta[k])))**2)
    x, _ = nnls(np.column_stack(cols).astype(float), y)
    for j, v in zip(used, x[:-1]):
        angles[j]['dtheta'] = float(np.sqrt(v))
    new['tof_dlambda'] = float(np.sqrt(x[-1]))
    model = RESOLUTION_MODES[new['mode']](Q, new)
    misfit = float(np.sqrt(np.mean((model / sig - 1)**2)))
    return new, misfit


# overlap rule of dq_table: combo label -> key
OVERLAPS = [('overlap: mean', 'mean'), ('overlap: earlier angle', 'first'),
            ('overlap: later angle', 'last')]


# a theta column with at most this many distinct values is one value per
# stitched angle (TOF: one fixed angle per run); more is an angle scan
MAX_ANGLES = 12


def split_angles(Q, dQ, theta=None):
    """[(Q, dQ)] of each stitched angle of one channel, each by Q: by its
    theta column when that holds one value per angle (REF_M files, sorted
    by Q with the angles' points interleaved where they overlap), else
    where Q falls back in file order (one angle if the file is sorted)."""
    if theta is not None:
        t = np.round(np.asarray(theta, dtype=float), 6)
        vals = np.unique(t[np.isfinite(t)])
        if 1 < vals.size <= MAX_ANGLES and np.all(np.isfinite(t)):
            out = []
            for v in vals:
                m = t == v
                i = np.argsort(Q[m], kind='stable')
                out.append((Q[m][i], dQ[m][i]))
            return out
    cuts = np.flatnonzero(np.diff(Q) <= 0) + 1
    return list(zip(np.split(Q, cuts), np.split(dQ, cuts)))


def dq_table(chans, overlap='mean'):
    """(Q, sigma_Q) table of a dataset's {channel: (Q, dQ[, theta])} for
    the 'data' mode: each channel is cut into its stitched angles
    (split_angles) and each angle is interpolated on its own.  Where angles
    overlap, rather than a zigzag between their interleaved points, the
    table takes (overlap) their 'mean', the 'first' angle (the one starting
    at the lowest Q) or the 'last' (starting at the highest Q); angles that
    start at the same Q (R+ and R- of one scan) are averaged.  A file
    sorted by Q without a theta column has a single angle per channel: the
    rule cannot apply."""
    if overlap not in dict((k, k) for _, k in OVERLAPS):
        raise ValueError('unknown overlap rule %r' % (overlap,))
    segs = []
    for Q, dQ, *theta in chans.values():
        Q, dQ = np.asarray(Q, dtype=float), np.asarray(dQ, dtype=float)
        ok = np.isfinite(Q) & np.isfinite(dQ) & (Q > 0) & (dQ > 0)
        th = np.asarray(theta[0], dtype=float)[ok] if theta and \
            theta[0] is not None else None
        Q, dQ = Q[ok], dQ[ok]
        segs += [(q, d) for q, d in split_angles(Q, dQ, th) if q.size]
    if not segs:
        raise ValueError('no point with dQ > 0')
    grid = np.unique(np.concatenate([q for q, _ in segs]))
    # (angle, grid point): its sigma there, NaN outside its Q range
    V = np.full((len(segs), grid.size), np.nan)
    for k, (q, d) in enumerate(segs):
        inside = (grid >= q[0]) & (grid <= q[-1])
        V[k, inside] = np.interp(grid[inside], q, d)
    cover = np.isfinite(V)
    if overlap != 'mean':
        start = np.array([q[0] for q, _ in segs])[:, None]
        key = np.where(cover, start, np.inf if overlap == 'first' else -np.inf)
        pick = key.min(0) if overlap == 'first' else key.max(0)
        cover &= np.isclose(start, pick[None, :], rtol=0, atol=1e-12)
    return grid, np.where(cover, V, 0).sum(0) / cover.sum(0)


def describe(res):
    """The spreads of res as text, in the units of the Simulation tab."""
    return 'Δλ = %.3g Å, Δθ = %s mrad' % (
        res['tof_dlambda'],
        ', '.join('%.3g' % (1e3 * a['dtheta']) for a in res['tof_angles']))


class ResolutionDialog(QtWidgets.QDialog):
    """sigma_Q of the Simulation tab's parameters (redrawn as they change),
    a dataset's dQ, and the parameters fitted to it, applied with Apply."""

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.fitted = None                # resolution dict from the last fit
        self._sets_known = False          # refresh_datasets has run
        self._message = ''                # what the last action did
        self.setWindowTitle('Resolution')
        v = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        self.yaxis = QtWidgets.QComboBox()
        self.yaxis.addItems(['σ_Q / Q (%)', 'σ_Q (Å⁻¹)'])
        bar.addWidget(self.yaxis)
        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel('Data dQ:'))
        self.dataset = QtWidgets.QComboBox()
        self.dataset.setToolTip('The dQ column of an imported dataset '
                                '(every channel of it)')
        bar.addWidget(self.dataset, 1)
        self.btn_fit = QtWidgets.QPushButton('Fit to data dQ')
        self.btn_fit.setToolTip(
            'Fit Δλ and each angle\'s Δθ to the dQ (an angle with no points '
            'keeps its Δθ)')
        bar.addWidget(self.btn_fit)
        self.btn_use = QtWidgets.QPushButton('Use data dQ')
        self.btn_use.setToolTip(
            'Take the resolution from the measured dQ (mode "from data dQ"):'
            '\na fit averages each data point over its own dQ, so stitched '
            'angles whose dQ do not join up\nare each taken as measured; '
            'curves at other Q interpolate this dataset\'s dQ')
        bar.addWidget(self.btn_use)
        self.overlap = QtWidgets.QComboBox()
        self.overlap.addItems([label for label, _ in OVERLAPS])
        self.overlap.setToolTip(
            'Where two stitched angles overlap in Q, the dQ the curves use '
            '(the dQ table): the mean of both,\nthe earlier angle\'s (lower '
            'Q) or the later angle\'s.  The fit always uses each point\'s '
            'own dQ.\nAngles are told apart by the θ column (one value per '
            'angle), else where Q falls back in file order.')
        bar.addWidget(self.overlap)
        v.addLayout(bar)

        w, self.plot = _plot()
        self.plot.addLegend(offset=(-10, 10))
        v.addWidget(w, 1)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        v.addWidget(self.note)

        row = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton('Apply')
        self.btn_apply.setToolTip('Put the fitted spreads in the Simulation '
                                  'tab and turn the resolution on')
        self.btn_apply.setEnabled(False)
        close = QtWidgets.QPushButton('Close')
        row.addStretch(1)
        row.addWidget(self.btn_apply)
        row.addWidget(close)
        v.addLayout(row)

        self.yaxis.currentIndexChanged.connect(self.redraw)
        self.dataset.currentIndexChanged.connect(self._forget_fit)
        self.btn_fit.clicked.connect(self.fit)
        self.btn_use.clicked.connect(self.use_data)
        self.overlap.currentIndexChanged.connect(self._overlap_changed)
        self.btn_apply.clicked.connect(self.apply)
        close.clicked.connect(self.close)
        sim.reflectanceChanged.connect(self.redraw)
        self.resize(760, 480)
        self.refresh_datasets()

    # -- data ----------------------------------------------------------------
    def refresh_datasets(self):
        """The datasets with a dQ column (SimulationTab.dq_sets)."""
        res = self.sim.stack.resolution
        if self._sets_known:
            name = self.dataset.currentText()
        else:                             # first time: the dQ in use, if any
            name = res.get('data_set', '')
            keys = [k for _, k in OVERLAPS]
            self.overlap.blockSignals(True)
            self.overlap.setCurrentIndex(
                keys.index(res.get('data_overlap', 'mean')))
            self.overlap.blockSignals(False)
        self._sets_known = True
        self._sets = self.sim.dq_sets()
        self.dataset.blockSignals(True)
        self.dataset.clear()
        self.dataset.addItems([n for n, _ in self._sets] or ['(none)'])
        i = self.dataset.findText(name)
        self.dataset.setCurrentIndex(max(i, 0))
        self.dataset.blockSignals(False)
        ok = bool(self._sets)
        for w in (self.dataset, self.btn_fit, self.btn_use,
                  self.overlap):
            w.setEnabled(ok)
        self._forget_fit()

    def data(self):
        """(Q, sigma_Q) of the chosen dataset, all channels, by Q; None if
        there is none."""
        if not self._sets:
            return None
        chans = self._sets[self.dataset.currentIndex()][1]
        Q = np.concatenate([c[0] for c in chans.values()])
        dQ = np.concatenate([c[1] for c in chans.values()])
        ok = np.isfinite(Q) & np.isfinite(dQ) & (Q > 0) & (dQ > 0)
        i = np.argsort(Q[ok])
        return Q[ok][i], dQ[ok][i]

    def current(self):
        """The Simulation tab's resolution parameters (shown mode)."""
        return self.sim.resolution_shown()

    # -- actions -------------------------------------------------------------
    def _forget_fit(self, *_):
        self.fitted = None
        self._message = ''
        self.btn_apply.setEnabled(False)
        self.redraw()

    def fit(self):
        d = self.data()
        try:
            self.fitted, misfit = fit_resolution(d[0], d[1], self.current())
        except ValueError as exc:
            self.fitted = None
            self._message = '<span style="color:%s">Cannot fit: %s</span>' % (
                style.ERROR, exc)
        else:
            self._message = 'Fitted: %s — rms misfit of σ_Q %.2g %%%s' % (
                describe(self.fitted), 100 * misfit, '' if misfit < 0.05 else
                ' (large: is the angle table right?)')
        self.btn_apply.setEnabled(self.fitted is not None)
        self.redraw()

    def use_data(self):
        if not self._sets:
            return
        name, chans = self._sets[self.dataset.currentIndex()]
        overlap = OVERLAPS[self.overlap.currentIndex()][1]
        try:
            q, sigma = dq_table(chans, overlap)
        except ValueError:
            return
        self.sim.set_data_dq(name, q, sigma, overlap)
        self.fitted = None
        self.btn_apply.setEnabled(False)
        self._message = ('Resolution from the dQ of %s: a fit averages '
                         'each point over its own dQ' % name)
        self.redraw()

    def _overlap_changed(self, *_):
        """A new overlap rule redoes the table in use, if it is this
        dataset's."""
        res = self.sim.stack.resolution
        if self._sets and self.sim.res_mode.currentText() == 'from data dQ' \
                and res.get('data_set') == self.dataset.currentText():
            self.use_data()

    def apply(self):
        if self.fitted is not None:
            self.sim.set_resolution_spreads(self.fitted)

    # -- drawing -------------------------------------------------------------
    def redraw(self, *_):
        p = self.plot
        p.clear()
        p.legend.clear()
        rel = self.yaxis.currentIndex() == 0
        p.setLabel('left', 'σ_Q / Q (%)' if rel else 'σ_Q (Å⁻¹)')

        def y(Q, s):
            return 100 * s / Q if rel else s

        # sigma_Q / Q grows as 1 / Q at small Q: the data's Q range, else the
        # Simulation tab's from a 40th of its Q max
        d = self.data()
        hi = self.sim.qmax.value()
        lo = max(self.sim.qmin.value(), hi / 40)
        if d is not None and d[0].size:
            lo, hi = d[0][0], d[0][-1]
            p.addItem(pg.PlotDataItem(
                d[0], y(*d), pen=None, symbol='o', symbolSize=4,
                symbolPen=None, symbolBrush=DATA_COLOUR,
                name='data dQ'))
        Q = np.linspace(lo, hi, 600)
        notes = [self._message] if self._message else []
        try:
            res = self.current()
            p.addItem(pg.PlotDataItem(
                Q, y(Q, sigma_q(Q, res)), pen=pg.mkPen(MODEL_COLOUR, width=2),
                name='current: dQ of %s' % res.get('data_set')
                if res['mode'] == 'data' else 'current parameters'))
            if not self.sim.res_on.isChecked():
                notes.append('the resolution is off in the Simulation tab')
        except ValueError as exc:
            notes.append('current parameters: %s' % exc)
        if self.fitted is not None:
            p.addItem(pg.PlotDataItem(
                Q, y(Q, sigma_q(Q, self.fitted)),
                pen=pg.mkPen(FIT_COLOUR, width=2,
                             style=QtCore.Qt.PenStyle.DashLine),
                name='fitted'))
        if not self._sets:
            notes.append('no imported dataset has a dQ column')
        self.note.setText('<br>'.join(n[:1].upper() + n[1:] for n in notes))
        p.enableAutoRange()


def _plot():
    # imported here: tabs.simulation imports this module
    from tabs.simulation import plot_widget
    return plot_widget('Q (Å⁻¹)', 'σ_Q / Q (%)')
