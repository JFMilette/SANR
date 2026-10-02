"""
DREAM window: progress of a fit.dream.dream_fit.sample run (generation, acceptance,
max R-hat), Stop, and its results, all drawn with pyqtgraph:

  Summary     the warnings of fit.dream.dream_fit.posterior_warnings, then the
              summary table; the convergence flag and the number of
              warnings are shown under the tabs
  Traces      one parameter (or log p) at a time, chosen in a combo: every
              chain against generation, the adaptation phase shaded, and
              the histogram of the kept samples beside it.  Updated live
              while the run goes on (from a thinned copy of the chains).
  Corner      the correlation matrix of all parameters (click a cell to pick
              a pair, hover to read it) and the pair chosen: 2-D histogram
              with both marginals.  Updated live (provisional, from the
              thinned chains, at most every CORNER_EVERY s while shown) once
              the adaptation phase is over; redrawn from the full chains at
              the end
  Predictive  data with the 68 / 95 % bands and median of the model R of
              200 posterior draws, and the spin asymmetry if R+ / R- (or
              R++ / R--) share their Q points
  Profile     68 / 95 % bands and median of the slab NSLD and in-plane
              MSLD of 200 draws

Values are shown in the Simulation tab's units (SLDs in 1e-6 A^-2, angles
in degrees).  Predictive and Profile are computed when first shown.
"Load MAP into model" sends the best state visited to the Simulation tab;
"Apply ±σ to bounds" sets Min / Max of every sampled parameter there to
median -/+ k sigma (16 / 84 % widths); "Save chains..." writes the run
with fit.dream.dream_fit.save_result.
"""

import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from fit.dream.dream_fit import (SIGMA_NAME, format_summary, posterior_warnings,
                         predictive_bands, profile_bands, sample, save_result,
                         summary_table)
from tabs.simulation import (CH_COLOURS, HP_COLOURS, PARAM_DISPLAY,
                             PROFILE_QUANTITIES, shown_names)

OK, BAD = '#4ade80', '#f87171'
PROGRESS_EVERY = 0.2                     # s between progress signals
TRACE_ROWS = 4000                        # live trace: thinned to this many
CORNER_EVERY = 1.0                       # s between live corner redraws
CHAIN_COLOURS = ['#5b9bd5', '#e0a458', '#4fc08d', '#e36fa8', '#b392f0',
                 '#d95f02', '#66a61e', '#e6ab02', '#9aa0a6', '#1b9e77']
CHANNEL_COLOURS = {'R++': CH_COLOURS[0], 'R+-': CH_COLOURS[1],
                   'R-+': CH_COLOURS[2], 'R--': CH_COLOURS[3],
                   'R+': HP_COLOURS[0], 'R-': HP_COLOURS[1]}
NSLD_COLOUR, MSLD_COLOUR = PROFILE_QUANTITIES[0][2], PROFILE_QUANTITIES[1][2]
# 2-D histograms: plot background -> accent -> near white
DENSITY = pg.ColorMap([0.0, 0.35, 0.75, 1.0],
                      ['#1b1c1f', '#2a3b5c', '#4c8dff', '#dce8ff'])
CORRELATION = pg.colormap.get('CET-D1A')
ADAPT_BRUSH = pg.mkBrush(154, 160, 166, 40)
# speed for the time left: over the last minute (progress signals are
# 0.2 s apart); the cost of a generation hardly changes along a run, so a
# long window only smooths out changes of the machine's load
RATE_POINTS = 300


def duration(seconds):
    """'2 h 05 min', '3 min 20 s' or '45 s'."""
    s = int(round(max(seconds, 0)))
    if s >= 3600:
        return '%d h %02d min' % (s // 3600, s % 3600 // 60)
    if s >= 60:
        return '%d min %02d s' % (s // 60, s % 60)
    return '%d s' % s


def plot_item(x='', y=''):
    """A PlotItem in the app's style (SI prefixes off: units are given)."""
    p = pg.PlotItem()
    p.showGrid(x=True, y=True, alpha=0.25)
    for side, label in (('bottom', x), ('left', y)):
        p.setLabel(side, label)
        p.getAxis(side).enableAutoSIPrefix(False)
    return p


def display_info(problem):
    """[(label, scale, unit)] of each sampled parameter: name, factor from
    stored to shown value, unit (as in the Simulation tab)."""
    out = []
    for (_, attr, *_), name in zip(problem.fit.params, problem.names):
        _, scale, unit = PARAM_DISPLAY[attr]
        out.append((shown_names(name), scale, unit))
    if problem.sigma_free:
        out.append((SIGMA_NAME, 1.0, ''))
    return out


def axis_label(info):
    name, _, unit = info
    return '%s (%s)' % (name, unit) if unit else name


def hbars(values, bins=50, colour='#4c8dff'):
    """Horizontal histogram of values (bars along x, bins along y)."""
    counts, edges = np.histogram(values, bins=bins)
    return pg.BarGraphItem(x0=0, y0=edges[:-1], width=counts,
                           height=np.diff(edges), pen=None,
                           brush=pg.mkBrush(colour))


def vbars(values, bins=50, colour='#4c8dff'):
    counts, edges = np.histogram(values, bins=bins)
    return pg.BarGraphItem(x0=edges[:-1], y0=0, height=counts,
                           width=np.diff(edges), pen=None,
                           brush=pg.mkBrush(colour))


def placeholder(text):
    lab = QtWidgets.QLabel(text)
    lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    lab.setStyleSheet('color: #9aa0a6')
    return lab


# ------------------------------------------------------------ thread ----
class DreamThread(QtCore.QThread):
    """Runs fit.dream.dream_fit.sample off the GUI thread."""

    # generation, acceptance of the last 100 generations, max R-hat or nan
    progress = QtCore.pyqtSignal(int, float, float)
    # thinned chains so far: generations (n,), X (n, N, d), logp (n, N),
    # last R-hat (d,) or None
    snapshot = QtCore.pyqtSignal(object, object, object, object)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, problem, **options):
        super().__init__()
        self.problem, self.options = problem, options
        self._stop = False
        self.result = None

    def stop(self):
        self._stop = True

    def run(self):
        last = {'X': None, 't': 0.0}
        moved = []
        # a thinned copy of the chains for the live trace: every `stride`
        # generations, halved (stride doubled) when it reaches TRACE_ROWS
        hist = {'g': [], 'X': [], 'lp': [], 'stride': 1}
        prev = self.options.get('resume')
        if prev is not None:          # continuing: start from its chains
            g = prev.generations
            while len(g) // hist['stride'] >= TRACE_ROWS // 2:
                hist['stride'] *= 2
            # multiples of the stride from one stride on, like the
            # callback's, so halving keeps them evenly spaced
            keep = (g % hist['stride'] == 0) & (g > 0)
            hist.update(g=list(g[keep]), X=list(prev.chains[keep]),
                        lp=list(prev.logp[keep]))

        def callback(t, X, logp, rhat):
            if last['X'] is not None:
                moved.append(np.mean(np.any(X != last['X'], axis=1)))
            last['X'] = X
            if t % hist['stride'] == 0:
                hist['g'].append(t)
                hist['X'].append(X)
                hist['lp'].append(logp)
                if len(hist['g']) >= TRACE_ROWS:
                    for k in ('g', 'X', 'lp'):
                        hist[k] = hist[k][1::2]
                    hist['stride'] *= 2
            now = time.monotonic()
            if now - last['t'] > PROGRESS_EVERY:
                last['t'] = now
                self.progress.emit(
                    t, float(np.mean(moved[-100:])) if moved else 0.0,
                    float(np.max(rhat)) if rhat is not None else np.nan)
                self.snapshot.emit(np.array(hist['g']), np.array(hist['X']),
                                   np.array(hist['lp']),
                                   None if rhat is None else np.array(rhat))
            return self._stop

        try:
            self.result = sample(self.problem, callback=callback,
                                 **self.options)
        except Exception as exc:              # report, don't kill the app
            msg = 'stopped' if self._stop else \
                '%s: %s' % (type(exc).__name__, exc)
            self.failed.emit(msg)
            return
        self.done.emit(self.result)


# ------------------------------------------------------------ traces ----
class TracePanel(QtWidgets.QWidget):
    """One parameter's chains against generation, and its histogram."""

    def __init__(self):
        super().__init__()
        self.combo = QtWidgets.QComboBox()
        self.combo.setToolTip('Parameter to show (↑ / ↓ to step through)')
        self.combo.setMinimumContentsLength(24)
        self.combo.currentIndexChanged.connect(self._draw)
        self.info = QtWidgets.QLabel()
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel('Parameter'))
        for step, glyph, tip in ((-1, '◀', 'Previous parameter'),
                                 (+1, '▶', 'Next parameter')):
            b = QtWidgets.QToolButton()
            b.setText(glyph)
            b.setToolTip(tip + ' (wraps around)')
            b.setAutoRepeat(True)          # hold to run through them
            b.clicked.connect(lambda _, k=step: self.step_parameter(k))
            if step < 0:
                bar.addWidget(b)
                bar.addWidget(self.combo)
            else:
                bar.addWidget(b)
        bar.addSpacing(12)
        bar.addWidget(self.info, 1)

        self.glw = pg.GraphicsLayoutWidget()
        self.p = plot_item('generation')
        self.h = plot_item('count')
        self.h.setYLink(self.p)
        self.h.hideAxis('left')
        self.glw.addItem(self.p, 0, 0)
        self.glw.addItem(self.h, 0, 1)
        self.glw.ci.layout.setColumnStretchFactor(0, 5)
        self.glw.ci.layout.setColumnStretchFactor(1, 1)
        self.adapt = pg.LinearRegionItem(movable=False, brush=ADAPT_BRUSH,
                                         pen=pg.mkPen(None))
        self.adapt.setZValue(-10)
        self.p.addItem(self.adapt)
        self.curves, self.bars = [], None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(bar)
        v.addWidget(self.glw, 1)
        self.clear()

    def step_parameter(self, k):
        """Show the k-th next entry of the combo, wrapping around."""
        n = self.combo.count()
        if n:
            self.combo.setCurrentIndex((self.combo.currentIndex() + k) % n)

    def clear(self):
        self.infos, self.data = [], None
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.blockSignals(False)
        self._draw()

    def set_parameters(self, infos, adapt_until):
        """The sampled parameters (display_info) and the last generation
        of the adaptation phase."""
        self.infos, self.adapt_until = infos, adapt_until
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem('log p')
        for k, info in enumerate(infos, 1):
            self.combo.addItem('%d · %s' % (k, info[0]))
        self.combo.setCurrentIndex(1 if infos else 0)
        self.combo.blockSignals(False)
        self.adapt.setRegion((0, adapt_until))

    def set_data(self, gens, X, logp, rhat):
        """Chains so far (or the final ones): gens (n,), X (n, N, d),
        logp (n, N), rhat (d,) or None."""
        if len(gens) == 0:
            return
        self.data = (np.asarray(gens), np.asarray(X), np.asarray(logp), rhat)
        self._draw()

    def _draw(self, *_):
        k = self.combo.currentIndex()
        if self.data is None or k < 0:
            for c in self.curves:
                c.setData([], [])
            if self.bars is not None:
                self.h.removeItem(self.bars)
                self.bars = None
            self.info.setText('')
            return
        g, X, logp, rhat = self.data
        if k == 0:
            Y, label = logp, 'log p'
        else:
            name, scale, unit = self.infos[k - 1]
            Y, label = X[:, :, k - 1] * scale, axis_label(self.infos[k - 1])
        N = Y.shape[1]
        while len(self.curves) < N:
            c = pg.PlotDataItem(pen=pg.mkPen(
                CHAIN_COLOURS[len(self.curves) % len(CHAIN_COLOURS)],
                width=1))
            c.setDownsampling(auto=True, method='peak')
            c.setClipToView(True)
            self.p.addItem(c)
            self.curves.append(c)
        for j, c in enumerate(self.curves):
            if j < N:
                c.setData(g, np.where(np.isfinite(Y[:, j]), Y[:, j], np.nan),
                          connect='finite')
            else:
                c.setData([], [])
        self.p.setLabel('left', label)
        # kept samples: after the adaptation phase and the first half
        keep = g >= max(self.adapt_until, g[-1] / 2)
        vals = Y[keep].ravel()
        vals = vals[np.isfinite(vals)]
        if self.bars is not None:
            self.h.removeItem(self.bars)
            self.bars = None
        if vals.size > 1 and np.ptp(vals) > 0:
            self.bars = hbars(vals)
            self.h.addItem(self.bars)
        text = ''
        if vals.size:
            q = np.percentile(vals, [16, 50, 84])
            text = 'median %.6g  (−%.3g / +%.3g)' % (q[1], q[1] - q[0],
                                                     q[2] - q[1])
        if k > 0 and rhat is not None:
            text += '    R̂ = %.3f' % rhat[k - 1]
        self.info.setText(text)
        if k == 0:                        # the first log p are far below
            fin = logp[np.isfinite(logp)]
            later = logp[len(logp) // 10:]
            later = later[np.isfinite(later)]
            if fin.size and later.size:
                lo = np.percentile(later, 1)
                self.p.setYRange(lo - 0.05 * (fin.max() - lo + 1),
                                 fin.max() + 0.05 * (fin.max() - lo + 1))
        else:
            self.p.enableAutoRange(axis='y')


# ------------------------------------------------------------ corner ----
class CornerPanel(QtWidgets.QWidget):
    """Correlation matrix of every parameter, and one chosen pair."""

    def __init__(self):
        super().__init__()
        self.samples = None
        self.infos = []

        # left: the matrix
        self.mw = pg.GraphicsLayoutWidget()
        self.mp = plot_item()
        self.mp.showGrid(x=False, y=False)
        self.mp.setAspectLocked(True)
        self.mp.invertY(True)
        self.mp.setMenuEnabled(False)
        self.mp.vb.setMouseEnabled(False, False)
        self.mw.addItem(self.mp)
        self.mimg = pg.ImageItem(axisOrder='row-major')
        self.mimg.setLookupTable(CORRELATION.getLookupTable(nPts=256))
        self.mimg.setLevels((-1, 1))
        self.mp.addItem(self.mimg)
        self.sel = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
        self.sel.setPen(pg.mkPen('#e3e5e8', width=2))
        self.mp.addItem(self.sel)
        bar = pg.ColorBarItem(values=(-1, 1), colorMap=CORRELATION,
                              interactive=False, width=12)
        bar.setImageItem(self.mimg)
        self.mw.addItem(bar)
        self.mread = QtWidgets.QLabel(
            'Click a cell to show that pair; hover to read it.')
        self.mread.setWordWrap(True)
        self.mw.scene().sigMouseMoved.connect(self._hover)
        self.mw.scene().sigMouseClicked.connect(self._click)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QtWidgets.QLabel('<b>Correlation matrix</b>'))
        lv.addWidget(self.mw, 1)
        lv.addWidget(self.mread)

        # right: the pair
        self.cx, self.cy = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for c in (self.cx, self.cy):
            c.setMinimumContentsLength(16)
            c.currentIndexChanged.connect(self._draw_pair)
        self.pread = QtWidgets.QLabel()
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel('X'))
        bar.addWidget(self.cx, 1)
        bar.addSpacing(8)
        bar.addWidget(QtWidgets.QLabel('Y'))
        bar.addWidget(self.cy, 1)
        self.pw = pg.GraphicsLayoutWidget()
        self.top = plot_item()
        self.main = plot_item()
        self.side = plot_item()
        self.top.setXLink(self.main)
        self.side.setYLink(self.main)
        self.top.hideAxis('bottom')
        self.side.hideAxis('left')
        self.pw.addItem(self.top, 0, 0)
        self.pw.addItem(self.main, 1, 0)
        self.pw.addItem(self.side, 1, 1)
        lay = self.pw.ci.layout
        lay.setColumnStretchFactor(0, 4)
        lay.setColumnStretchFactor(1, 1)
        lay.setRowStretchFactor(0, 1)
        lay.setRowStretchFactor(1, 4)
        self.dimg = pg.ImageItem(axisOrder='col-major')
        self.dimg.setLookupTable(DENSITY.getLookupTable(nPts=256))
        self.main.addItem(self.dimg)
        self.medx = pg.InfiniteLine(angle=90, pen=pg.mkPen(
            '#e3e5e8', width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self.medy = pg.InfiniteLine(angle=0, pen=pg.mkPen(
            '#e3e5e8', width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        for ln in (self.medx, self.medy):
            self.main.addItem(ln, ignoreBounds=True)
        self.hist_items = []
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addLayout(bar)
        rv.addWidget(self.pw, 1)
        rv.addWidget(self.pread)

        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([380, 620])
        self.note = QtWidgets.QLabel()
        self.note.setStyleSheet('color: #9aa0a6')
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.note)
        v.addWidget(split, 1)

    def clear(self):
        """Forget the samples: the next ones pick the pair afresh."""
        self.samples, self.infos = None, []
        self.note.setText('')

    def set_result(self, result, infos, discard=0.5):
        """The posterior samples of a finished run."""
        s = result.samples(discard)
        g0 = discard * result.generations[-1] if discard < 1 else discard
        self.set_samples(s, infos, 'Final: %d samples, from generation %d '
                         'on.' % (len(s), max(g0, result.adapt_until)))

    def set_samples(self, s, infos, note):
        """Samples s (M, d) in stored units; `note` above the plots.  The
        pair shown is kept when the parameters stay the same, else the most
        correlated pair is chosen."""
        same = self.samples is not None and \
            [i[0] for i in infos] == [i[0] for i in self.infos]
        self.note.setText(note)
        self.infos = infos
        self.samples = s * np.array([i[1] for i in infos])
        d = s.shape[1]
        with np.errstate(invalid='ignore', divide='ignore'):
            C = np.atleast_2d(np.corrcoef(s.T)) if d > 1 else np.ones((1, 1))
        self.C = np.nan_to_num(C)
        self.mimg.setImage(self.C, levels=(-1, 1))
        self.mimg.setRect(QtCore.QRectF(0, 0, d, d))
        ticks = [[(k + 0.5, str(k + 1)) for k in range(d)]]
        for side in ('bottom', 'left'):
            self.mp.getAxis(side).setTicks(ticks)
        self.mp.setRange(xRange=(0, d), yRange=(0, d), padding=0.02)
        if same:                          # live update: keep the user's pair
            self._draw_pair()
            return
        names = ['%d · %s' % (k + 1, i[0]) for k, i in enumerate(infos)]
        # the most correlated pair first
        if d > 1:
            A = np.abs(self.C) - 2 * np.eye(d)
            i, j = np.unravel_index(np.argmax(A), A.shape)
        else:
            i = j = 0
        for c, sel in ((self.cx, j), (self.cy, i)):
            c.blockSignals(True)
            c.clear()
            c.addItems(names)
            c.setCurrentIndex(int(sel))
            c.blockSignals(False)
        self._draw_pair()

    def _cell(self, scene_pos):
        if self.samples is None or \
                not self.mp.sceneBoundingRect().contains(scene_pos):
            return None
        p = self.mp.vb.mapSceneToView(scene_pos)
        d = self.samples.shape[1]
        j, i = int(np.floor(p.x())), int(np.floor(p.y()))
        return (i, j) if 0 <= i < d and 0 <= j < d else None

    def _hover(self, pos):
        cell = self._cell(pos)
        if cell is None:
            return
        i, j = cell
        self.mread.setText('%s  ×  %s:  r = %+.3f'
                           % (self.infos[j][0], self.infos[i][0],
                              self.C[i, j]))

    def _click(self, ev):
        cell = self._cell(ev.scenePos())
        if cell is None:
            return
        i, j = cell
        for c, k in ((self.cx, j), (self.cy, i)):
            c.blockSignals(True)
            c.setCurrentIndex(k)
            c.blockSignals(False)
        self._draw_pair()

    def _draw_pair(self, *_):
        for p, it in self.hist_items:
            p.removeItem(it)
        self.hist_items = []
        if self.samples is None:
            return
        j, i = self.cx.currentIndex(), self.cy.currentIndex()
        if i < 0 or j < 0:
            return
        self.sel.setRect(j, i, 1, 1)
        x, y = self.samples[:, j], self.samples[:, i]
        self.main.setLabel('bottom', axis_label(self.infos[j]))
        self.main.setLabel('left', axis_label(self.infos[i]))
        bins = 60
        xr = (x.min(), x.max()) if np.ptp(x) > 0 else (x[0] - 1, x[0] + 1)
        yr = (y.min(), y.max()) if np.ptp(y) > 0 else (y[0] - 1, y[0] + 1)
        H, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
        self.dimg.setImage(H, levels=(0, max(H.max(), 1)))
        self.dimg.setRect(QtCore.QRectF(xe[0], ye[0], xe[-1] - xe[0],
                                        ye[-1] - ye[0]))
        self.medx.setValue(np.median(x))
        self.medy.setValue(np.median(y))
        for p, it in ((self.top, vbars(x, bins)), (self.side, hbars(y, bins))):
            p.addItem(it)
            self.hist_items.append((p, it))
        self.main.autoRange(padding=0)
        r = self.C[i, j]
        qx, qy = np.percentile(x, [16, 50, 84]), np.percentile(y, [16, 50, 84])
        self.pread.setText(
            'r = %+.3f     X: %.6g (−%.3g / +%.3g)     Y: %.6g (−%.3g / +%.3g)'
            % (r, qx[1], qx[1] - qx[0], qx[2] - qx[1],
               qy[1], qy[1] - qy[0], qy[2] - qy[1]))


# ------------------------------------------------- predictive / profile ----
# fill alphas (of 255) on the dark plot background
A_PRED, A_95, A_68 = 55, 110, 170


def band(plot, x, lo, hi, colour, alpha, log=False, edges=False):
    """Filled band between lo and hi, with thin edge lines if `edges`.
    FillBetweenItem ignores the plot's log mode, so on a log axis
    (log=True) its edges are PlotCurveItems in log10 coordinates, like the
    error bars."""
    if log:
        lo, hi = np.log10(lo), np.log10(hi)
    c = QtGui.QColor(colour)
    c.setAlpha(170 if edges else 0)
    pen = pg.mkPen(c, width=1) if edges else pg.mkPen(None)
    a = pg.PlotCurveItem(x, lo, pen=pen)
    b = pg.PlotCurveItem(x, hi, pen=pen)
    c.setAlpha(alpha)
    fill = pg.FillBetweenItem(a, b, brush=pg.mkBrush(c))
    for it in (fill, a, b):
        plot.addItem(it)
    return [fill, a, b]


class PredictivePanel(QtWidgets.QWidget):
    """Data with, for posterior draws, the 95 % band of replicated data
    (model + measurement noise, light, edged), the 68 / 95 % bands of the
    model alone (darker) and its median.  Each kind of band can be hidden
    with its check box."""

    KEY = ('Light band with edges: 95 % of replicated data (model + noise '
           'dR).   Darker bands: model 95 % / 68 % (parameter uncertainty).  '
           'Line: median.')

    def __init__(self):
        super().__init__()
        self.view = pg.GraphicsLayoutWidget()
        self.items = {'predictive': [], 'model': []}   # band items per kind
        self.boxes = {}
        row = QtWidgets.QHBoxLayout()
        for kind, text, tip in (
                ('predictive', 'replicated data 95 %',
                 'Light band with edges: 95 % of replicated data '
                 '(model + measurement noise dR)'),
                ('model', 'model 68 / 95 %',
                 'Darker bands: 68 / 95 % of the model alone (parameter '
                 'uncertainty)')):
            cb = QtWidgets.QCheckBox(text, checked=True, toolTip=tip)
            cb.toggled.connect(lambda on, k=kind: self._show(k, on))
            self.boxes[kind] = cb
            row.addWidget(cb)
        row.addStretch(1)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(row)
        v.addWidget(self.view, 1)

    def _show(self, kind, on):
        for it in self.items[kind]:
            it.setVisible(on)

    def _band(self, kind, *args, **kw):
        self.items[kind] += band(*args, **kw)

    def show_bands(self, b):
        view = self.view
        view.clear()
        self.items = {'predictive': [], 'model': []}
        view.addLabel(self.KEY, row=0, col=0, color='#9aa0a6', size='9pt')
        p = plot_item('', 'R')
        p.setLogMode(y=True)
        p.addLegend(offset=(-10, 10))
        view.addItem(p, 1, 0)
        for c in b['channels']:
            col = CHANNEL_COLOURS.get(c['name'], '#9aa0a6')
            Q, R, dR = c['Q'], c['R'], c['dR']
            lo95, lo68, med, hi68, hi95 = np.maximum(c['bands'], 1e-300)
            # a replicate below 0 cannot be drawn on a log axis: cap the
            # band 3 decades under the median, like the error bars
            plo, phi = c['predictive'][0], c['predictive'][4]
            plo = np.maximum(plo, 1e-3 * med)
            self._band('predictive', p, Q, plo, np.maximum(phi, plo), col,
                       A_PRED, log=True, edges=True)
            self._band('model', p, Q, lo95, hi95, col, A_95, log=True)
            self._band('model', p, Q, lo68, hi68, col, A_68, log=True)
            ok = R > 0
            yl = np.log10(R[ok])
            # ErrorBarItem has no log mode: log10 coordinates, as in the
            # Experimental tab
            p.addItem(pg.ErrorBarItem(
                x=Q[ok], y=yl, top=np.log10(R[ok] + dR[ok]) - yl,
                bottom=yl - np.log10(np.maximum(R[ok] - dR[ok],
                                                1e-3 * R[ok])),
                beam=0, pen=pg.mkPen(col, width=0.8)))
            p.addItem(pg.PlotDataItem(Q[ok], R[ok], pen=None, symbol='o',
                                      symbolSize=4, symbolPen=None,
                                      symbolBrush=col, name=c['name']))
            p.addItem(pg.PlotDataItem(Q, med, pen=pg.mkPen(col, width=1.5)))
        a = b['asymmetry']
        if a is None:
            p.setLabel('bottom', 'Q (Å⁻¹)')
            self._apply_boxes()
            return
        s = plot_item('Q (Å⁻¹)', 'spin asymmetry')
        s.setXLink(p)
        view.addItem(s, 2, 0)
        view.ci.layout.setRowStretchFactor(1, 3)
        view.ci.layout.setRowStretchFactor(2, 2)
        lo95, lo68, med, hi68, hi95 = a['bands']
        plo, phi = a['predictive'][0], a['predictive'][4]
        self._band('predictive', s, a['Q'], plo, phi, '#4c8dff', A_PRED,
                   edges=True)
        self._band('model', s, a['Q'], lo95, hi95, '#4c8dff', A_95)
        self._band('model', s, a['Q'], lo68, hi68, '#4c8dff', A_68)
        self._apply_boxes()
        ok = np.isfinite(a['A'])
        s.addItem(pg.ErrorBarItem(x=a['Q'][ok], y=a['A'][ok],
                                  height=2 * a['dA'][ok], beam=0,
                                  pen=pg.mkPen('#9aa0a6', width=0.8)))
        s.addItem(pg.PlotDataItem(a['Q'][ok], a['A'][ok], pen=None,
                                  symbol='o', symbolSize=4, symbolPen=None,
                                  symbolBrush='#9aa0a6'))
        s.addItem(pg.PlotDataItem(a['Q'], med, pen=pg.mkPen('#4c8dff',
                                                            width=1.5)))
        # noisy points at small R have huge asymmetry errors: frame the
        # bands (an asymmetry lies in [-1, 1]) rather than the error bars
        fin = np.concatenate([plo[np.isfinite(plo)], phi[np.isfinite(phi)],
                              med[np.isfinite(med)]])
        if fin.size:
            lo, hi = max(fin.min(), -1.5), min(fin.max(), 1.5)
            pad = 0.08 * max(hi - lo, 0.1)
            s.setYRange(lo - pad, hi + pad, padding=0)


    def _apply_boxes(self):
        for kind, cb in self.boxes.items():
            self._show(kind, cb.isChecked())


class ProfilePanel(pg.GraphicsLayoutWidget):
    """68 / 95 % bands and median of the slab SLDs of posterior draws."""

    def show_bands(self, z, nuc, mag):
        self.clear()
        p = plot_item('depth z (Å)', 'SLD (10⁻⁶ Å⁻²)')
        p.addLegend(offset=(-10, 10))
        self.addItem(p)
        for bands, col, name in ((nuc, NSLD_COLOUR, 'NSLD'),
                                 (mag, MSLD_COLOUR,
                                  'MSLD in plane (ρ cos θ)')):
            lo95, lo68, med, hi68, hi95 = bands / 1e-6
            band(p, z, lo95, hi95, col, A_95)
            band(p, z, lo68, hi68, col, A_68)
            p.addItem(pg.PlotDataItem(z, med, pen=pg.mkPen(col, width=1.5),
                                      name=name))


# ------------------------------------------------------------ window ----
class DreamWindow(QtWidgets.QWidget):
    """One DREAM run: progress while it samples, its results after."""

    stopRequested = QtCore.pyqtSignal()
    sendRequested = QtCore.pyqtSignal(object)        # MAP state x
    boundsRequested = QtCore.pyqtSignal(object, object)   # min, max
    continueRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle('Bayesian sampling (DREAM)')
        self.resize(1100, 820)
        self.problem = self.result = None
        self.sources = []
        self._live, self._corner_t = None, 0.0   # chains so far, last redraw
        self._previous = None              # the run being continued
        self.may_stop_early = False        # stop when converged is on
        self._reset_clock()

        self.header = QtWidgets.QLabel(wordWrap=True)
        self.bar = QtWidgets.QProgressBar()
        self.status = QtWidgets.QLabel()
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_stop.setToolTip('Stop and keep the generations done so far')
        self.btn_stop.clicked.connect(self.stopRequested)
        self.btn_continue = QtWidgets.QPushButton('Continue')
        self.btn_continue.clicked.connect(self.continueRequested)
        self.btn_continue.setEnabled(False)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.bar, 1)
        top.addWidget(self.btn_continue)
        top.addWidget(self.btn_stop)

        self.converged = QtWidgets.QLabel()

        self.tabs = QtWidgets.QTabWidget()
        self.summary = QtWidgets.QPlainTextEdit(readOnly=True)
        self.summary.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont))
        self.summary.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.trace = TracePanel()
        self.corner = CornerPanel()
        self.predictive = PredictivePanel()
        self.profile = ProfilePanel()
        # result tabs: a page holding either a placeholder or the panel
        self.pages = {}
        self.tabs.addTab(self.summary, 'Summary')
        self.tabs.addTab(self.trace, 'Traces')
        for name, panel in (('Corner', self.corner),
                            ('Predictive', self.predictive),
                            ('Profile', self.profile)):
            page = QtWidgets.QStackedWidget()
            page.addWidget(placeholder('Shown when the run ends.'))
            page.addWidget(panel)
            self.pages[name] = page
            self.tabs.addTab(page, name)
        self._drawn = set()
        self.tabs.currentChanged.connect(self._draw_tab)

        self.btn_save = QtWidgets.QPushButton('Save chains…')
        self.btn_save.clicked.connect(self._save)
        self.btn_map = QtWidgets.QPushButton('Load MAP into model')
        self.btn_map.setToolTip('Send the best state visited to the '
                                'Simulation tab')
        self.btn_map.clicked.connect(
            lambda: self.sendRequested.emit(self.result.best()[0]))
        self.bounds_k = QtWidgets.QDoubleSpinBox()
        self.bounds_k.setRange(0.1, 10.0)
        self.bounds_k.setSingleStep(0.5)
        self.bounds_k.setDecimals(1)
        self.bounds_k.setValue(1.0)
        self.bounds_k.setPrefix('× ')
        self.bounds_k.setToolTip('Width of the new bounds in posterior σ '
                                 '(1σ leaves out about 32 % of the '
                                 'posterior, 2σ about 5 %)')
        self.btn_bounds = QtWidgets.QPushButton('Apply ±σ to bounds')
        self.btn_bounds.setToolTip(
            'Set Min / Max of every sampled parameter in the Simulation tab '
            'to median − k·(−σ) and median + k·(+σ) (16 / 84 % widths); '
            'Revert restores the bounds from before the run')
        self.btn_bounds.clicked.connect(self._apply_bounds)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.converged, 1)
        bottom.addWidget(self.bounds_k)
        bottom.addWidget(self.btn_bounds)
        bottom.addSpacing(12)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_map)

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self.header)
        v.addLayout(top)
        v.addWidget(self.status)
        v.addWidget(self.tabs, 1)
        v.addLayout(bottom)

    def start(self, problem, n_gen, header, sources=()):
        self.problem, self.result, self.sources = problem, None, list(sources)
        self.infos = display_info(problem)
        self.header.setText(header)
        self.bar.setRange(0, n_gen)
        self.bar.setValue(0)
        self.status.setText('Starting…')
        self._reset_clock()
        self.converged.setText('')
        self.summary.setPlainText('')
        self.trace.clear()
        self.trace.set_parameters(self.infos, max(n_gen // 10, 1))
        self.corner.clear()
        self._live, self._corner_t = None, 0.0
        for page in self.pages.values():
            page.setCurrentIndex(0)
            page.widget(0).setText('Shown when the run ends.')
        self._drawn = set()
        self.tabs.setCurrentWidget(self.trace)
        self.btn_stop.setEnabled(True)
        for b in (self.btn_save, self.btn_map, self.btn_bounds,
                  self.btn_continue):
            b.setEnabled(False)
        self._previous = None
        self.show()
        self.raise_()

    def continue_start(self, n_gen):
        """The shown run goes on up to generation n_gen: back to the
        running state, keeping its traces (extended live) and summary."""
        self._previous, self.result = self.result, None
        last = int(self._previous.generations[-1])
        self.bar.setRange(0, n_gen)
        self.bar.setValue(last)
        self.status.setText('Continuing from generation %d…' % last)
        self._reset_clock()
        self.summary.setPlainText(
            'Continuing: the summary below is from generation %d and is '
            'updated when the run ends.\n\n' % last
            + self.summary.toPlainText())
        for name, page in self.pages.items():
            if name != 'Corner':
                page.setCurrentIndex(0)
                page.widget(0).setText('Shown when the run ends.')
        self._drawn = set()
        self._live, self._corner_t = None, 0.0
        self.btn_stop.setEnabled(True)
        for b in (self.btn_save, self.btn_map, self.btn_bounds,
                  self.btn_continue):
            b.setEnabled(False)

    def show_loaded(self, result, problem, header, sources=()):
        """A saved run (fit.dream.dream_fit.load_run), shown as a finished one."""
        self.start(problem, int(result.generations[-1]), header, sources)
        self.finish(result)
        self.status.setText('Saved run: %d generation(s), %d evaluations.'
                            % (result.generations[-1], result.n_evals))

    def _reset_clock(self):
        """Start timing a run (or a continuation) now."""
        self._t_start = time.monotonic()
        self._rate = deque(maxlen=RATE_POINTS)    # (time, generation)

    def progress(self, gen, acc, rhat):
        self.bar.setValue(gen)
        now = time.monotonic()
        self._rate.append((now, gen))
        text = 'Generation %d / %d — acceptance %.3f — max R̂ %s' % (
            gen, self.bar.maximum(), acc,
            'not yet (adaptation)' if np.isnan(rhat) else '%.3f' % rhat)
        text += ' — elapsed %s' % duration(now - self._t_start)
        (t0, g0), (t1, g1) = self._rate[0], self._rate[-1]
        if g1 > g0 and t1 > t0:      # speed over the last minute
            left = (self.bar.maximum() - gen) / ((g1 - g0) / (t1 - t0))
            text += ', about %s left%s' % (
                duration(left), ''
                if self.may_stop_early else '')
        self.status.setText(text)

    def snapshot(self, gens, X, logp, rhat):
        if self.result is not None:            # a late one after the end
            return
        self.trace.set_data(gens, X, logp, rhat)
        self._live = (gens, X)
        if self.tabs.currentWidget() is self.pages['Corner'] and \
                time.monotonic() - self._corner_t > CORNER_EVERY:
            self._live_corner()

    def _live_corner(self):
        """Provisional corner of the thinned chains so far: generations
        after the adaptation phase and in the second half, as at the end."""
        if self._live is None or self.result is not None:
            return
        gens, X = self._live
        adapt = self.trace.adapt_until
        keep = gens >= max(adapt, gens[-1] / 2)
        page = self.pages['Corner']
        if keep.sum() < 2 or gens[-1] <= adapt:
            page.widget(0).setText(
                'Shown once the adaptation phase is over (generation %d).'
                % adapt)
            page.setCurrentIndex(0)
            return
        self._corner_t = time.monotonic()
        s = X[keep].reshape(-1, X.shape[2])
        self.corner.set_samples(
            s, self.infos, 'Provisional — generation %d: %d samples from the '
            'chains so far (thinned); redrawn from the full chains at the '
            'end.' % (gens[-1], len(s)))
        page.setCurrentIndex(1)

    def finish(self, result):
        self.result, self._previous = result, None
        self.btn_stop.setEnabled(False)
        last = int(result.generations[-1])
        self.bar.setValue(last)
        self.status.setText('Ended after %d generation(s), %d evaluations, '
                            'in %s.' % (last, result.n_evals, duration(
                                time.monotonic() - self._t_start)))
        rhat = result.rhat_history[-1][1] if result.rhat_history else None
        self.trace.adapt_until = result.adapt_until
        self.trace.adapt.setRegion((0, result.adapt_until))
        self.trace.set_data(result.generations, result.chains, result.logp,
                            rhat)
        self.btn_save.setEnabled(True)
        st = result.state or {}
        can = bool(st.get('cancelled')) and st.get('n_gen', 0) > st['t']
        self.btn_continue.setEnabled(can)
        self.btn_continue.setToolTip(
            'Go on with this stopped run up to the %d generations it was '
            'started with, as if it had never stopped' % st['n_gen'] if can
            else 'Only a run stopped with Stop can be continued')
        if last <= result.adapt_until:
            msg = ('Stopped within the adaptation phase (the first %d '
                   'generations): no posterior sample.' % result.adapt_until)
            self.summary.setPlainText(msg)
            for page in self.pages.values():
                page.widget(0).setText(msg)
            return
        # warnings in the Summary, not above the tabs: a long list would
        # shrink the plots
        # the run's own convergence target (1.2 for runs saved without one)
        target = float((result.state or {}).get('rhat_target', 1.2))
        warn = posterior_warnings(result, self.problem.bounds,
                                  rhat_max=target)
        head = ''.join(['Warnings:\n'] + ['  • %s\n' % w for w in warn]
                       + ['\n']) if warn else ''
        self.summary.setPlainText(
            shown_names(head + format_summary(result, self.problem)))
        ok = rhat is not None and bool(np.all(rhat < target))
        text = '<span style="color:%s">● %s</span>' % (
            (OK, 'Converged: R̂ &lt; %g for every parameter' % target) if ok
            else (BAD, 'Not converged (R̂ target %g)' % target))
        if warn:
            text += ('&nbsp;&nbsp;&nbsp;<span style="color:%s">%d warning%s '
                     '— see Summary</span>'
                     % (BAD, len(warn), '' if len(warn) == 1 else 's'))
        self.converged.setText(text)
        self.btn_map.setEnabled(True)
        self.btn_bounds.setEnabled(True)
        for page in self.pages.values():
            page.widget(0).setText('Computing…')
        self._draw_tab(self.tabs.currentIndex())

    def fail(self, msg):
        self.btn_stop.setEnabled(False)
        if self._previous is not None:       # a failed continue: back to it
            self.finish(self._previous)
            self._previous = None
        self.status.setText('Stopped.' if msg == 'stopped'
                            else 'Failed: %s' % msg)

    def _draw_tab(self, i):
        name = self.tabs.tabText(i)
        if name == 'Corner' and self.result is None:   # still running
            self._live_corner()
            return
        if name not in self.pages or name in self._drawn or \
                self.result is None or \
                self.result.generations[-1] <= self.result.adapt_until:
            return
        page = self.pages[name]
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor)
        try:
            if name == 'Corner':
                self.corner.set_result(self.result, self.infos)
            elif name == 'Predictive':
                self.predictive.show_bands(
                    predictive_bands(self.result, self.problem))
            else:
                self.profile.show_bands(
                    *profile_bands(self.result, self.problem))
        except Exception as exc:
            page.widget(0).setText('Cannot draw: %s' % exc)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        page.setCurrentIndex(1)
        self._drawn.add(name)

    def _apply_bounds(self):
        """median -/+ k sigma of each model parameter (not the error scale)
        as new fit bounds."""
        rows, _ = summary_table(self.result, self.problem)
        k = self.bounds_k.value()
        n = len(self.problem.fit.params)
        self.boundsRequested.emit(
            [r['median'] - k * r['minus'] for r in rows[:n]],
            [r['median'] + k * r['plus'] for r in rows[:n]])

    # -- saving -------------------------------------------------------------
    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save chains', 'dream_run.npz', 'NumPy archive (*.npz)',
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if not path:
            return
        try:
            save_result(path, self.result, self.problem, self.sources)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Save chains',
                                           'Could not save %s:\n%s'
                                           % (path, exc))
