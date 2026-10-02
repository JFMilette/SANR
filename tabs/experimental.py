"""
Experimental tab: import measured reflectivity channels and plot them with
the same quantities as the simulation tab, with error bars.

Each file holds one channel (R+, R-, R++, R+-, R-+ or R--) as columns
    Qz (1/A)   R (a.u.)   [dR (a.u.)   dQz (1/A)   theta]
or, when the Q values are kept in a separate file (one Q per line, e.g.
q.dat next to rexp1.dat / rtheory1.dat), as columns
    R (a.u.)   [dR (a.u.)   dQz (1/A)   theta]
with one row per Q value.  Missing dR / dQz are taken as zero.
Importing opens the file text in a dialog with line numbers; the user picks
the line where the data start, where Q comes from and which channel the file
is.

POLARISATION -- every channel is a pair of an incident polarisation Pi and
an analysed polarisation Pa (3-vectors in the sample frame, length =
efficiency, Pa = (0, 0, 0) for no analyser, as for R+ / R-).  An imported
channel either has its own pair (set in the import dialog or with
"Polarisation…" on its dataset) or follows the Simulation tab's
Polarisation box.

The "+ Simulation" button adds a live dataset holding the simulation tab's
channels R++, R+-, R-+, R--, R+, R- (background included, once each) on its
Q grid, with the Simulation tab's pairs unless it has its own.  Starting a
fit gives every live dataset the fitted channels' pairs, so what is drawn
against the data is exactly what is fitted.  It is recomputed whenever the
simulation changes, drawn as lines without error bars, and can be frozen
into a static copy from its context menu; the frozen copy is still drawn as
lines.

Only points between the Simulation tab's Q min and Q max are plotted, the
range that is also fitted.

Asymmetries combine channels measured on different Q grids: every channel is
linearly interpolated (value and error) onto the Q points of the first
channel in the formula, over the range they share.  Errors are propagated
assuming independent channels.  R+ / R- are used as imported when present,
otherwise built as R+ = R++ + R+-, R- = R-- + R-+ (a simulated dataset
always holds R+ / R- themselves, with the background counted once).

The Fit tool box (fit.panel.FitPanel, a mixin of this tab) fits the
Simulation tab's model to every channel of one dataset: by differential
evolution (fit.de, followed in fit.de.de_window) or by sampling its
posterior with DREAM (fit.dream, followed in fit.dream.dream_window).
"""

import os
import re

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from fit.panel import FitPanel
from model.polarisation import CHANNEL_NAMES, default_vectors, vectors_pair
from .simulation import (CH_COLOURS, HP_COLOURS, REFL_QUANTITIES, Crosshair,
                         PairTable, fmt, plot_widget)


DATA_FILTER = 'Data files (*.dat *.txt *.csv *.refl);;All files (*)'
DIALOG_OPTIONS = QtWidgets.QFileDialog.Option.DontUseNativeDialog
# import order in the channel combo; colours follow the simulation tab
DATA_CHANNELS = ['+', '-', '++', '-+', '+-', '--']
CHANNEL_COLOURS = {'++': CH_COLOURS[0], '+-': CH_COLOURS[1],
                   '-+': CH_COLOURS[2], '--': CH_COLOURS[3],
                   '+': HP_COLOURS[0], '-': HP_COLOURS[1]}
PLOT_ORDER = ['++', '+-', '-+', '--', '+', '-']
SPLIT = re.compile(r'[\s,;]+')
# where the Q values of an imported file come from
Q_IN_FILE, Q_SEPARATE = 0, 1
# a separate Q file found next to the data file is used automatically
Q_FILE_NAMES = ('q.dat', 'q.txt', 'Q.dat', 'Q.txt')


def parse_row(line):
    """Floats of one data line, or None if it is not numeric."""
    parts = [p for p in SPLIT.split(line.strip()) if p]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def guess_start(lines, ncols=(4,)):
    """First line (0-based) that parses as at least ncols[0] numbers, else
    ncols[1], ...; comment lines are never chosen."""
    for n in ncols:
        for i, line in enumerate(lines):
            if line.lstrip().startswith('#'):
                continue
            row = parse_row(line)
            if row is not None and len(row) >= n:
                return i
    return 0


def first_row(lines, start):
    """Numbers of the first data line at or after `start`, or None."""
    for line in lines[start:]:
        s = line.strip()
        if s and not s.startswith('#'):
            return parse_row(s)
    return None


def read_q_file(path):
    """Q values of a separate Q file: first number of each numeric line,
    comment / text lines skipped."""
    with open(path, errors='replace') as f:
        lines = f.read().splitlines()
    Q = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        row = parse_row(s)
        if row:
            Q.append(row[0])
    if not Q:
        raise ValueError('no numeric values in %s' % os.path.basename(path))
    return np.array(Q, dtype=float)


def find_q_file(path):
    """A Q file next to the data file `path`, or None."""
    folder = os.path.dirname(path)
    for name in Q_FILE_NAMES:
        q = os.path.join(folder, name)
        if os.path.isfile(q) and not os.path.samefile(q, path):
            return q
    return None


def parse_data(lines, start, Q=None):
    """Parse lines[start:] into a dict of Q, R, dR, dQ, theta arrays, sorted
    by Q.  With Q=None the columns are Q R [dR dQ theta]; otherwise they are
    R [dR dQ theta] and row i belongs to Q[i].  Missing dR / dQ are zero.
    Blank / comment lines are skipped; returns (data, n_skipped)."""
    keys = ['Q', 'R', 'dR', 'dQ', 'theta']
    if Q is not None:
        keys = keys[1:]
    need = 2 if Q is None else 1
    rows, skipped = [], 0
    for line in lines[start:]:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        row = parse_row(s)
        if row is None or len(row) < need:
            skipped += 1
            continue
        row = row[:len(keys)]
        rows.append(row + [np.nan] * (len(keys) - len(row)))
    if not rows:
        raise ValueError('no data rows with at least %d numeric column%s'
                         % (need, 's' if need > 1 else ''))
    a = np.array(rows, dtype=float)
    if Q is not None:
        if len(Q) != len(a):
            raise ValueError('%d data rows but %d Q values in the Q file'
                             % (len(a), len(Q)))
        a = np.column_stack([Q, a])
    # absent errors / resolution count as zero, absent theta stays NaN
    for j in (2, 3):
        a[:, j] = np.where(np.isnan(a[:, j]), 0.0, a[:, j])
    a = a[np.argsort(a[:, 0])]
    return {k: a[:, i] for i, k in enumerate(['Q', 'R', 'dR', 'dQ',
                                              'theta'])}, skipped


def on_grid(Q, d):
    """Channel d (R, dR) interpolated onto Q, NaN outside its range."""
    kw = dict(left=np.nan, right=np.nan)
    return (np.interp(Q, d['Q'], d['R'], **kw),
            np.interp(Q, d['Q'], d['dR'], **kw))


def asym(a, da, b, db):
    """(a - b) / (a + b) and its error."""
    s = a + b
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(s > 0, (a - b) / s, np.nan)
        e = np.where(s > 0, 2 * np.hypot(b * da, a * db) / s**2, np.nan)
    return v, e


# ------------------------------------------------------ import dialog ----
class LineNumberArea(QtWidgets.QWidget):

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QtCore.QSize(self.editor.number_width(), 0)

    def paintEvent(self, ev):
        self.editor.paint_numbers(ev)


class NumberedText(QtWidgets.QPlainTextEdit):
    """Read-only text view with a line-number gutter; the chosen start line
    is highlighted and clicking a line emits `lineClicked` (0-based)."""

    lineClicked = QtCore.pyqtSignal(int)

    def __init__(self, text):
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont))
        self.setPlainText(text)
        self.start = 0
        self.gutter = LineNumberArea(self)
        self.blockCountChanged.connect(self._update_width)
        self.updateRequest.connect(self._update_gutter)
        self._update_width()

    def number_width(self):
        digits = len(str(max(1, self.blockCount())))
        return 16 + self.fontMetrics().horizontalAdvance('9') * digits

    def _update_width(self, *_):
        self.setViewportMargins(self.number_width(), 0, 0, 0)

    def _update_gutter(self, rect, dy):
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        cr = self.contentsRect()
        self.gutter.setGeometry(QtCore.QRect(cr.left(), cr.top(),
                                             self.number_width(), cr.height()))

    def paint_numbers(self, ev):
        p = QtGui.QPainter(self.gutter)
        # the gutter does not inherit the editor's fixed font under the app
        # stylesheet; number_width() is measured with it, so draw with it
        p.setFont(self.font())
        p.fillRect(ev.rect(), QtGui.QColor('#2b2d31'))
        block = self.firstVisibleBlock()
        top = self.blockBoundingGeometry(block).translated(
            self.contentOffset()).top()
        h = self.fontMetrics().height()
        while block.isValid() and top <= ev.rect().bottom():
            n = block.blockNumber()
            if block.isVisible() and top + h >= ev.rect().top():
                p.setPen(QtGui.QColor('#f87171' if n == self.start
                                      else '#62666d'))
                p.drawText(0, int(top), self.gutter.width() - 8, h,
                           QtCore.Qt.AlignmentFlag.AlignRight, str(n + 1))
            top += self.blockBoundingRect(block).height()
            block = block.next()

    def set_start(self, line):
        """Highlight line `line` (0-based) and everything after it."""
        self.start = line
        sel = []
        block = self.document().findBlockByNumber(line)
        if block.isValid():
            first = QtWidgets.QTextEdit.ExtraSelection()
            first.format.setBackground(QtGui.QColor('#5a4a1c'))
            first.format.setProperty(
                QtGui.QTextFormat.Property.FullWidthSelection, True)
            first.cursor = QtGui.QTextCursor(block)
            sel.append(first)
            rest = QtWidgets.QTextEdit.ExtraSelection()
            rest.format.setBackground(QtGui.QColor('#34301f'))
            cur = QtGui.QTextCursor(block)
            cur.movePosition(QtGui.QTextCursor.MoveOperation.NextBlock)
            cur.movePosition(QtGui.QTextCursor.MoveOperation.End,
                             QtGui.QTextCursor.MoveMode.KeepAnchor)
            rest.cursor = cur
            sel.append(rest)
            self.setTextCursor(QtGui.QTextCursor(block))
            self.centerCursor()
        self.setExtraSelections(sel)
        self.gutter.update()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.lineClicked.emit(self.cursorForPosition(
                ev.position().toPoint()).blockNumber())


class ImportDialog(QtWidgets.QDialog):
    """Show a data file, let the user pick the first data line, where Q comes
    from and the channel; `data` / `q_path` hold the result after accept()."""

    def __init__(self, path, parent=None, sets=(), current=None, taken=None,
                 q_path=None, sim_vectors=None):
        super().__init__(parent)
        self.setWindowTitle('Import — %s' % os.path.basename(path))
        with open(path, errors='replace') as f:
            text = f.read()
        self.lines = text.splitlines()
        self.path = path
        self.data = None
        self.q_path = None
        self._q_cache = (None, None)        # (path, Q array or error str)

        self.view = NumberedText(text)
        self.start = QtWidgets.QSpinBox()
        self.start.setRange(1, max(1, len(self.lines)))
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(['R' + c for c in DATA_CHANNELS])
        # dataset: pick an existing one or type a new name
        self.taken = taken or {}          # set name -> channels already in it
        self.dataset = QtWidgets.QComboBox()
        self.dataset.setEditable(True)
        self.dataset.addItems(list(sets))
        self.dataset.setCurrentText(current or 'Set %d' % (len(sets) + 1))
        self.dataset.currentTextChanged.connect(self._default_channel)
        self._default_channel(self.dataset.currentText())
        self.preview = QtWidgets.QLabel()
        self.preview.setWordWrap(True)
        # polarisation of the channel: the Simulation tab's pair, or its own
        self.sim_vectors = sim_vectors or default_vectors()
        self.pair_box = QtWidgets.QVBoxLayout()
        self.pair = None
        self.channel.currentIndexChanged.connect(self._rebuild_pair)
        self._rebuild_pair()

        # Q source: first column of this file, or a separate one-column file
        self.q_mode = QtWidgets.QComboBox()
        self.q_mode.addItems(['Column 1 of this file', 'Separate Q file'])
        self.q_edit = QtWidgets.QLineEdit()
        self.q_edit.setPlaceholderText('file with one Q value per line')
        self.q_browse = QtWidgets.QPushButton('Browse…')
        q_row = QtWidgets.QHBoxLayout()
        q_row.addWidget(self.q_mode)
        q_row.addWidget(self.q_edit, 1)
        q_row.addWidget(self.q_browse)
        self.columns = QtWidgets.QLabel()

        form = QtWidgets.QFormLayout()
        form.addRow('Data start at line', self.start)
        form.addRow('Q values', q_row)
        form.addRow('Dataset', self.dataset)
        form.addRow('Channel', self.channel)
        form.addRow('Polarisation', self.pair_box)
        form.addRow(self.columns)
        form.addRow(self.preview)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self.view, 1)
        v.addLayout(form)
        v.addWidget(buttons)

        start = guess_start(self.lines, (4, 1))
        # one or two columns (R [dR]) means Q lives elsewhere: use the Q file
        # next to the data, else the one used for the previous import
        row = first_row(self.lines, start)
        sibling = find_q_file(path)
        if row is not None and len(row) <= 2 and (sibling or q_path):
            self.q_mode.setCurrentIndex(Q_SEPARATE)
        self.q_edit.setText(sibling or q_path or '')

        self.start.valueChanged.connect(self._on_start)
        self.view.lineClicked.connect(lambda n: self.start.setValue(n + 1))
        self.q_mode.currentIndexChanged.connect(self._on_start)
        self.q_edit.editingFinished.connect(self._on_start)
        self.q_browse.clicked.connect(self._browse_q)
        self.start.setValue(start + 1)
        self._on_start()
        self.resize(820, 700)

    def _browse_q(self):
        folder = os.path.dirname(self.q_edit.text() or self.path)
        q, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Q values', folder, DATA_FILTER, options=DIALOG_OPTIONS)
        if q:
            self.q_edit.setText(q)
            self.q_mode.setCurrentIndex(Q_SEPARATE)
            self._on_start()

    def _load_q(self):
        """Q array of the chosen Q file (cached), or an error message."""
        q = self.q_edit.text().strip()
        if not q:
            return 'choose the file holding the Q values'
        if self._q_cache[0] != q:
            try:
                self._q_cache = (q, read_q_file(q))
            except (OSError, ValueError) as exc:
                self._q_cache = (q, 'Q file: %s' % exc)
        return self._q_cache[1]

    def _on_start(self, *_):
        line = self.start.value() - 1
        self.view.set_start(line)
        separate = self.q_mode.currentIndex() == Q_SEPARATE
        self.q_edit.setEnabled(separate)
        self.columns.setText(
            '<i>Columns: %sR (a.u.), [dR (a.u.), dQz (Å⁻¹), θ]%s. Click a '
            'line in the text to start the data there.</i>'
            % ('' if separate else 'Qz (Å⁻¹), ',
               ' — one row per Q value' if separate else ''))
        self.data = self.q_path = None
        try:
            Q = None
            if separate:
                Q = self._load_q()
                if isinstance(Q, str):
                    raise ValueError(Q)
            self.data, skipped = parse_data(self.lines, line, Q)
        except ValueError as exc:
            self.data = None
            self.preview.setText('<span style="color:#f87171">%s</span>' % exc)
            self.ok.setEnabled(False)
            return
        if separate:
            self.q_path = self.q_edit.text().strip()
        d = self.data
        msg = '%d points, Q = %.4g … %.4g Å⁻¹' % (len(d['Q']), d['Q'][0],
                                                   d['Q'][-1])
        if not np.any(d['dR']):
            msg += ', no dR column (errors = 0)'
        if skipped:
            msg += ('  <span style="color:#f87171">(%d non-numeric lines '
                    'skipped)</span>' % skipped)
        self.preview.setText(msg)
        self.ok.setEnabled(True)

    def _default_channel(self, name):
        """Preselect the first channel the chosen dataset does not have."""
        have = self.taken.get(name, ())
        # a set that already has spin channels most likely wants the others
        order = PLOT_ORDER if any(len(c) == 2 for c in have) else DATA_CHANNELS
        free = [c for c in order if c not in have]
        if free:
            self.channel.setCurrentIndex(DATA_CHANNELS.index(free[0]))

    def selected_channel(self):
        return DATA_CHANNELS[self.channel.currentIndex()]

    def _rebuild_pair(self, *_):
        """One Pi / Pa row for the selected channel, following the
        Simulation tab until unticked."""
        if self.pair is not None:
            self.pair_box.removeWidget(self.pair)
            self.pair.deleteLater()
        ch = self.selected_channel()
        self.pair = PairTable([ch], follow=True)
        self.pair.set_vectors({ch: None}, fallback=self.sim_vectors)
        self.pair_box.addWidget(self.pair)

    def selected_vectors(self):
        """[Pi, Pa] of the channel, or None to follow the Simulation tab."""
        return self.pair.vectors()[self.selected_channel()]

    def selected_dataset(self):
        return self.dataset.currentText().strip() or 'Set'


# ---------------------------------------------- derived quantities ----
# D below is one dataset's channels: {channel: dict(Q, R, dR, dQ, theta, path)}
def half_polarized(D, sign):
    """R+ or R- (sign '+' / '-') as (Q, R, dR, dQ): imported, or summed from
    the NSF + SF channels on the NSF channel's Q grid; None if unavailable."""
    if sign in D:
        d = D[sign]
        return d['Q'], d['R'], d['dR'], d['dQ']
    nsf, sf = ('++', '+-') if sign == '+' else ('--', '-+')
    if nsf in D and sf in D:
        d = D[nsf]
        b, db = on_grid(d['Q'], D[sf])
        return d['Q'], d['R'] + b, np.hypot(d['dR'], db), d['dQ']
    return None


def asymmetry(D, name):
    """(Q, value, error, dQ) of an asymmetry, or a str naming what's missing."""
    if name == 'SA half-polarized':
        up, dn = half_polarized(D, '+'), half_polarized(D, '-')
        if up is None or dn is None:
            return 'needs R+ and R− (or R++, R+-, R-+, R--)'
        Q = up[0]
        b, db = on_grid(Q, dict(Q=dn[0], R=dn[1], dR=dn[2]))
        return (Q,) + asym(up[1], up[2], b, db) + (up[3],)
    need = {'SA NSF': ['++', '--'], 'SA SF': ['+-', '-+'],
            'SF fraction': ['+-', '-+', '++', '--']}[name]
    missing = [c for c in need if c not in D]
    if missing:
        return 'needs ' + ', '.join('R' + c for c in missing)
    Q, dQ = D[need[0]]['Q'], D[need[0]]['dQ']
    vals = [on_grid(Q, D[c]) for c in need]
    if name != 'SF fraction':
        (a, da), (b, db) = vals
        return (Q,) + asym(a, da, b, db) + (dQ,)
    (ud, dud), (du, ddu), (uu, duu), (dd, ddd) = vals
    N, M = ud + du, uu + dd
    dN, dM = np.hypot(dud, ddu), np.hypot(duu, ddd)
    T = N + M
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(T > 0, N / T, np.nan)
        e = np.where(T > 0, np.hypot(M * dN, N * dM) / T**2, np.nan)
    return Q, v, e, dQ


def dataset_series(D, k, line=False):
    """Series of REFL_QUANTITIES[k] for one dataset as
    ([(label, colour, Q, y, dy, dQ)], note); colour None = per-dataset.
    A simulated (line) dataset's R+ / R- only show as half-polarized."""
    name, label, _ = REFL_QUANTITIES[k]
    if k == 0:
        hide = {'+', '-'} if line and len(D) > 2 else set()
        return [('R' + ch, CHANNEL_COLOURS[ch], D[ch]['Q'], D[ch]['R'],
                 D[ch]['dR'], D[ch]['dQ']) for ch in PLOT_ORDER
                if ch in D and ch not in hide], ''
    if label is None:
        out = []
        for sign, col, lab in (('+', HP_COLOURS[0], 'R+'),
                               ('-', HP_COLOURS[1], 'R−')):
            hp = half_polarized(D, sign)
            if hp is not None:
                out.append((lab, col) + hp)
        return out, '' if out else 'needs R+ / R− or the four spin channels'
    res = asymmetry(D, name)
    if isinstance(res, str):
        return [], res
    return [(name, None) + res], ''


def simulation_channels(stack, Q, vectors):
    """Channels dict of the stack's reflectivity on Q, one channel per
    {name: [Pi, Pa]} of vectors (see model.polarisation.vectors_pair); zero errors."""
    names = [c for c in PLOT_ORDER if c in vectors]
    # a transverse P is reported by the Simulation tab and the fit
    R = stack.reflectivities(Q, [vectors_pair(*vectors[c]) for c in names],
                             warn=False)
    zero = np.zeros_like(Q)
    return {ch: dict(Q=Q, R=R[:, j], dR=zero, dQ=zero,
                     theta=np.full_like(Q, np.nan), path='simulation')
            for j, ch in enumerate(names)}


def vec_text(vec):
    Pi, Pa = vec
    fmt3 = lambda v: '(%s)' % ', '.join('%.3g' % c for c in v)
    return 'Pi = %s, Pa = %s' % (fmt3(Pi), 'none' if not np.any(Pa)
                                 else fmt3(Pa))


class PairDialog(QtWidgets.QDialog):
    """Edit the Pi / Pa pairs of a dataset's channels; a ticked row follows
    the Simulation tab."""

    def __init__(self, name, names, own, sim, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Polarisation — %s' % name)
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            '<i>Pi incident, Pa analysed polarisation (x, y, z) in the sample '
            'frame; |P| = efficiency; Pa = 0: no analyser.</i>',
            wordWrap=True))
        self.table = PairTable(names, follow=True)
        self.table.set_vectors(own, fallback=sim)
        v.addWidget(self.table)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)
        self.resize(640, self.sizeHint().height())

    def vectors(self):
        return self.table.vectors()


# ------------------------------------------------------- data tree ----
ROLE = QtCore.Qt.ItemDataRole.UserRole          # (set index, channel or None)
SYMBOLS = ['o', 's', 't', 'd', 'star', 't1', 'p', 'h', '+', 'x']
# marker shown beside each dataset name in the tree
SYMBOL_GLYPH = {'o': '●', 's': '■', 't': '▼', 'd': '◆', 'star': '★',
                't1': '▲', 'p': '⬟', 'h': '⬢', '+': '+', 'x': '×'}
SET_COLOURS = ['#b392f0', '#1b9e77', '#d95f02', '#e7298a', '#66a61e',
               '#e6ab02', '#d4a55a', '#5aa9e6']


def glyph_icon(glyph):
    """Small icon showing a dataset's marker glyph."""
    pix = QtGui.QPixmap(16, 16)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pix)
    p.setPen(QtGui.QColor('#e3e5e8'))
    p.drawText(pix.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, glyph)
    p.end()
    return QtGui.QIcon(pix)


class DataTree(QtWidgets.QTreeWidget):
    """Datasets (checkable, renamable) holding channel items; dragging a
    channel onto another dataset emits `moveRequested(src, channel, dst)`.
    The tab owns the data and rebuilds the tree afterwards."""

    moveRequested = QtCore.pyqtSignal(int, str, int)

    def __init__(self):
        super().__init__()
        self.setHeaderHidden(True)
        # the column is as wide as the longest line (long file names), and
        # the tree scrolls sideways instead of eliding it
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)

    def dropEvent(self, ev):
        src = self.currentItem()
        dst = self.itemAt(ev.position().toPoint())
        ev.setDropAction(QtCore.Qt.DropAction.IgnoreAction)
        ev.accept()                          # never let Qt move items itself
        if src is None or dst is None or src.data(0, ROLE)[1] is None:
            return
        s, ch = src.data(0, ROLE)
        d = dst.data(0, ROLE)[0]
        if d != s:
            # after Qt has finished the drag
            QtCore.QTimer.singleShot(
                0, lambda: self.moveRequested.emit(s, ch, d))


# ------------------------------------------------------ session ----
def file_ref(path, base):
    """A file as saved in a session: absolute, and relative to the session
    folder `base` so a moved folder still finds it."""
    return {'path': os.path.abspath(path),
            'relative': os.path.relpath(path, base)}


def find_file(ref, base):
    """Path of a saved file_ref: where it was, else relative to `base`."""
    for p in (ref['path'], os.path.join(base, ref['relative'])):
        if os.path.isfile(p):
            return p
    raise OSError('not found: %s' % ref['path'])


def read_channel(ref, q_ref, start, base):
    """An imported channel read again, as the import dialog read it."""
    path = find_file(ref, base)
    q_path = find_file(q_ref, base) if q_ref else None
    with open(path, errors='replace') as f:
        lines = f.read().splitlines()
    Q = read_q_file(q_path) if q_path else None
    data, _ = parse_data(lines, start, Q)
    return dict(data, path=path, q_path=q_path, start=start)


# ------------------------------------------------- experimental tab ----
class ExperimentalTab(FitPanel, QtWidgets.QWidget):

    def __init__(self, simulation=None, parent=None):
        super().__init__(parent)
        # [dict(name, visible, channels={channel: dict(Q, R, dR, dQ, ...)},
        #       live, line)]; live sets mirror the simulation tab, line sets
        #       (live or frozen simulations) are drawn as lines
        self.sets = []
        self.simulation = simulation
        self.last_q_path = None       # separate Q file of the last import

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_plot())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1270])

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        if simulation is not None:
            simulation.reflectanceChanged.connect(self._update_live)
            # the plotted Q range follows the Simulation tab's
            simulation.qmin.valueChanged.connect(self.redraw)
            simulation.qmax.valueChanged.connect(self.redraw)
        self.refresh_tree()
        self.redraw()

    # -- left column --------------------------------------------------------
    def _build_controls(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)

        box = QtWidgets.QGroupBox('Experimental data')
        bv = QtWidgets.QVBoxLayout(box)
        self.tree = DataTree()
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._update_buttons)
        self.tree.moveRequested.connect(self.move_channel)
        self.tree.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        bv.addWidget(self.tree)
        bv.addWidget(QtWidgets.QLabel(
            '<i>Tick datasets to plot them together. Drag a channel onto '
            'another dataset to regroup; double-click a dataset to rename.'
            '</i>', wordWrap=True))
        row = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton('Import…')
        self.btn_new = QtWidgets.QPushButton('+ Dataset')
        self.btn_sim = QtWidgets.QPushButton('+ Simulation')
        self.btn_sim.setToolTip('Add a dataset that follows the reflectance '
                                'of the Simulation tab')
        self.btn_sim.setEnabled(self.simulation is not None)
        self.btn_remove = QtWidgets.QPushButton('− Remove')
        for b in (self.btn_import, self.btn_new, self.btn_sim,
                  self.btn_remove):
            row.addWidget(b)
        bv.addLayout(row)
        self.btn_import.clicked.connect(self.import_data)
        self.btn_new.clicked.connect(lambda: self.new_set())
        self.btn_sim.clicked.connect(self.add_simulation)
        self.btn_remove.clicked.connect(self.remove_selected)
        v.addWidget(box, 1)

        disp = QtWidgets.QGroupBox('Display')
        f = QtWidgets.QFormLayout(disp)
        self.logy = QtWidgets.QCheckBox('log R')
        self.logy.setChecked(True)
        self.rq4 = QtWidgets.QCheckBox('R·Q⁴')
        self.show_err = QtWidgets.QCheckBox('error bars')
        self.show_err.setChecked(True)
        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(self.logy)
        opts.addWidget(self.rq4)
        opts.addWidget(self.show_err)
        f.addRow('Scale', opts)
        for w in (self.logy, self.rq4, self.show_err):
            w.toggled.connect(self.redraw)
        v.addWidget(disp)
        v.addWidget(self._build_fit_box())

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        return panel

    def _build_plot(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel('Reflectivity plot:'))
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItems([q[0] for q in REFL_QUANTITIES])
        self.quantity.currentIndexChanged.connect(self.redraw)
        bar.addWidget(self.quantity)
        self.formula = QtWidgets.QLabel()
        self.formula.setStyleSheet('color: #9aa0a6')
        bar.addWidget(self.formula)
        bar.addStretch(1)
        v.addLayout(bar)

        w, self.plot = plot_widget('Qz (Å⁻¹)', 'Reflectivity')
        self.plot.addLegend(offset=(-10, 10))
        self.zero = pg.InfiniteLine(pos=0, angle=0,
                                    pen=pg.mkPen('#4b4e55', width=0.7))
        self.plot.addItem(self.zero)
        self._items = []            # [(curve, errorbar)] currently drawn
        self._shown = []            # [(curve, name, Q, y, dy)] for the cursor
        self._log = False
        self.cursor = Crosshair(w, self.plot, self._readout)
        v.addWidget(w)
        return box

    # -- tree ---------------------------------------------------------------
    def refresh_tree(self, select=None):
        """Rebuild the tree from self.sets; select = (set, channel or None)."""
        t = self.tree
        t.blockSignals(True)
        t.clear()
        current = None
        for i, s in enumerate(self.sets):
            top = QtWidgets.QTreeWidgetItem([s['name']])
            top.setIcon(0, glyph_icon(SYMBOL_GLYPH[SYMBOLS[i % len(SYMBOLS)]]))
            top.setData(0, ROLE, (i, None))
            live = s.get('live', False)
            flags = (top.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                     | QtCore.Qt.ItemFlag.ItemIsEditable
                     | QtCore.Qt.ItemFlag.ItemIsDropEnabled) \
                & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled
            if live:                          # its channels come from the model
                flags &= ~QtCore.Qt.ItemFlag.ItemIsDropEnabled
                top.setToolTip(0, 'Live: follows the Simulation tab')
            top.setFlags(flags)
            top.setCheckState(0, QtCore.Qt.CheckState.Checked if s['visible']
                              else QtCore.Qt.CheckState.Unchecked)
            font = top.font(0)
            font.setBold(True)
            font.setItalic(live)
            top.setFont(0, font)
            t.addTopLevelItem(top)
            if select == (i, None):
                current = top
            for ch in PLOT_ORDER:
                if ch not in s['channels']:
                    continue
                d = s['channels'][ch]
                it = QtWidgets.QTreeWidgetItem(['R%-3s %s  (%d pts)' % (
                    ch, os.path.basename(d['path']), len(d['Q']))])
                it.setData(0, ROLE, (i, ch))
                it.setFlags((it.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled)
                            & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled)
                if live:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled)
                it.setForeground(0, QtGui.QColor(CHANNEL_COLOURS[ch]))
                tip = d['path'] if not d.get('q_path') else \
                    '%s\nQ: %s' % (d['path'], d['q_path'])
                own = self._own_vectors(s, ch)
                tip += '\n%s%s' % (vec_text(self.channel_vectors(s, ch)),
                                   '' if own is not None else
                                   ' (Simulation tab)')
                it.setToolTip(0, tip)
                top.addChild(it)
                if select == (i, ch):
                    current = it
            top.setExpanded(True)
        t.blockSignals(False)
        if current is not None:
            t.setCurrentItem(current)
        self._update_buttons()
        self._refresh_fit_sets()

    def _selected(self):
        it = self.tree.currentItem()
        return it.data(0, ROLE) if it is not None else None

    def _update_buttons(self, *_):
        sel = self._selected()
        self.btn_remove.setEnabled(sel is not None and not (
            sel[1] is not None and self.sets[sel[0]].get('live')))

    def _on_item_changed(self, item, _col):
        i, ch = item.data(0, ROLE)
        if ch is not None:
            return
        s = self.sets[i]
        s['visible'] = item.checkState(0) == QtCore.Qt.CheckState.Checked
        name = item.text(0).strip()
        if name:
            s['name'] = name
        # rebuild outside the itemChanged handler
        QtCore.QTimer.singleShot(0, lambda: self.refresh_tree((i, None)))
        self.redraw()

    def _context_menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None:
            return
        i, ch = it.data(0, ROLE)
        menu = QtWidgets.QMenu(self)
        live = self.sets[i].get('live', False)
        if ch is None:
            menu.addAction('Rename', lambda: self.tree.editItem(it, 0))
            menu.addAction('Polarisation…', lambda: self.edit_polarisation(i))
            if live:
                menu.addAction('Freeze as static copy',
                               lambda: self.freeze_simulation(i))
            else:
                menu.addAction('Import into this dataset…',
                               lambda: self.import_data(self.sets[i]['name']))
        elif not live:
            move = menu.addMenu('Move to')
            for j, s in enumerate(self.sets):
                if j != i:
                    move.addAction(s['name'],
                                   lambda j=j: self.move_channel(i, ch, j))
            move.addAction('New dataset', lambda: self.move_channel(
                i, ch, self.new_set(refresh=False)))
        menu.addAction('Remove', self.remove_selected)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # -- polarisation ---------------------------------------------------------
    def sim_vectors(self):
        """{channel: [Pi, Pa]} of the Simulation tab's Polarisation box."""
        if self.simulation is None:
            return default_vectors()
        return self.simulation.pol.vectors()

    def _own_vectors(self, s, ch):
        """[Pi, Pa] a channel of dataset s sets itself, or None."""
        if s.get('live'):
            return (s.get('pairs') or {}).get(ch)
        return s['channels'][ch].get('pol') if ch in s['channels'] else None

    def channel_vectors(self, s, ch):
        """[Pi, Pa] used for channel ch of dataset s: its own, else the
        Simulation tab's."""
        own = self._own_vectors(s, ch)
        return own if own is not None else self.sim_vectors()[ch]

    def live_vectors(self, s):
        return {c: self.channel_vectors(s, c) for c in CHANNEL_NAMES}

    def edit_polarisation(self, i):
        s = self.sets[i]
        names = CHANNEL_NAMES if s.get('live') else \
            [c for c in PLOT_ORDER if c in s['channels']]
        if not names:
            QtWidgets.QMessageBox.information(
                self, 'Polarisation', 'Import a channel into %s first.'
                % s['name'])
            return
        dlg = PairDialog(s['name'], names,
                         {c: self._own_vectors(s, c) for c in names},
                         self.sim_vectors(), self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        vec = dlg.vectors()
        if s.get('live'):
            s['pairs'] = vec
            self._update_live()
        else:
            for c in names:
                s['channels'][c]['pol'] = vec[c]
        self.refresh_tree((i, None))

    # -- data management ----------------------------------------------------
    def new_set(self, name=None, refresh=True):
        """Append an empty dataset, return its index."""
        names = {s['name'] for s in self.sets}
        if name is None:
            n = len(self.sets) + 1
            while 'Set %d' % n in names:
                n += 1
            name = 'Set %d' % n
        self.sets.append(dict(name=name, visible=True, channels={}))
        if refresh:
            self.refresh_tree((len(self.sets) - 1, None))
        return len(self.sets) - 1

    def _set_index(self, name):
        """Index of the static (non-live) dataset called `name`."""
        for i, s in enumerate(self.sets):
            if s['name'] == name and not s.get('live'):
                return i
        return None

    def _unique_name(self, base):
        names = {s['name'] for s in self.sets}
        name, n = base, 2
        while name in names:
            name, n = '%s %d' % (base, n), n + 1
        return name

    def add_simulation(self):
        """Add a live dataset mirroring the simulation tab's reflectance."""
        sim = self.simulation
        if sim is None:
            return
        if sim.last_R is None:
            sim.recompute()
        if sim.last_R is None:
            QtWidgets.QMessageBox.warning(
                self, 'Simulation', 'The simulation has no valid '
                'reflectance yet:\n%s' % sim.status.text())
            return
        s = dict(name=self._unique_name('Simulation'), visible=True,
                 live=True, line=True)
        s['channels'] = simulation_channels(sim.stack, sim.last_Q,
                                            self.live_vectors(s))
        self.sets.append(s)
        self.refresh_tree((len(self.sets) - 1, None))
        self.redraw()

    def freeze_simulation(self, i):
        """Static copy of live dataset i, e.g. to compare two models."""
        s = self.sets[i]
        self.sets.append(dict(
            name=self._unique_name(s['name'] + ' (frozen)'), visible=True,
            line=True,
            channels={ch: dict(d, path='simulation (frozen)',
                               pol=self.channel_vectors(s, ch))
                      for ch, d in s['channels'].items()}))
        self.refresh_tree((len(self.sets) - 1, None))
        self.redraw()

    def _update_live(self):
        sim = self.simulation
        live = [s for s in self.sets if s.get('live')]
        if not live or sim.last_R is None:
            return
        for s in live:
            s['channels'] = simulation_channels(sim.stack, sim.last_Q,
                                                self.live_vectors(s))
        sel = self._selected()
        self.refresh_tree(sel)                # point counts may have changed
        self.redraw()

    def import_data(self, dataset=None):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Import experimental data', '', DATA_FILTER,
            options=DIALOG_OPTIONS)
        if not path:
            return
        if not isinstance(dataset, str):            # default: selected set
            sel = self._selected()
            dataset = self.sets[sel[0]]['name'] if sel else None
        try:
            static = [s for s in self.sets if not s.get('live')]
            if dataset not in {s['name'] for s in static}:
                dataset = None
            dlg = ImportDialog(
                path, self, [s['name'] for s in static], dataset,
                {s['name']: set(s['channels']) for s in static},
                self.last_q_path, self.sim_vectors())
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Import', 'Could not read %s:\n%s' % (path, exc))
            return
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted or \
                dlg.data is None:
            return
        ch, name = dlg.selected_channel(), dlg.selected_dataset()
        i = self._set_index(name)
        if i is not None and not self._confirm_replace(i, ch):
            return
        if i is None:
            i = self.new_set(name, refresh=False)
        self.sets[i]['channels'][ch] = dict(dlg.data, path=path,
                                            q_path=dlg.q_path,
                                            start=dlg.start.value() - 1,
                                            pol=dlg.selected_vectors())
        if dlg.q_path:
            self.last_q_path = dlg.q_path
        self.refresh_tree((i, ch))
        self.redraw()

    def _confirm_replace(self, i, ch):
        s = self.sets[i]
        if ch not in s['channels']:
            return True
        return QtWidgets.QMessageBox.question(
            self, 'Experimental data',
            '%s already has R%s (%s).\nReplace it?'
            % (s['name'], ch, os.path.basename(s['channels'][ch]['path']))) \
            == QtWidgets.QMessageBox.StandardButton.Yes

    def move_channel(self, src, ch, dst):
        if self.sets[src].get('live') or self.sets[dst].get('live'):
            self.refresh_tree((src, ch))
            return
        if src == dst or not self._confirm_replace(dst, ch):
            self.refresh_tree((src, ch))
            return
        self.sets[dst]['channels'][ch] = self.sets[src]['channels'].pop(ch)
        self.refresh_tree((dst, ch))
        self.redraw()

    def remove_selected(self):
        sel = self._selected()
        if sel is None:
            return
        i, ch = sel
        if ch is not None and self.sets[i].get('live'):
            return                       # the simulation would restore it
        if ch is not None:
            del self.sets[i]['channels'][ch]
            self.refresh_tree((i, None))
        else:
            s = self.sets[i]
            if s['channels'] and QtWidgets.QMessageBox.question(
                    self, 'Experimental data',
                    'Remove %s and its %d channel(s)?'
                    % (s['name'], len(s['channels']))) != \
                    QtWidgets.QMessageBox.StandardButton.Yes:
                return
            del self.sets[i]
            self.refresh_tree((min(i, len(self.sets) - 1), None))
        self.redraw()

    # -- session ------------------------------------------------------------
    def session_state(self, base):
        """Plain-JSON state of the tab for a session file in folder `base`:
        the datasets (imported channels by file, data start line and Q file,
        re-read on restore; frozen simulations by value), display and fit
        settings."""
        sets = []
        for s in self.sets:
            e = dict(name=s['name'], visible=s['visible'],
                     live=bool(s.get('live')), line=bool(s.get('line')))
            if s.get('live'):
                e['pairs'] = s.get('pairs')
            else:
                e['channels'] = {}
                for ch, d in s['channels'].items():
                    if 'start' in d:            # imported from a file
                        c = dict(file=file_ref(d['path'], base),
                                 q_file=file_ref(d['q_path'], base)
                                 if d.get('q_path') else None,
                                 start=d['start'])
                    else:                       # computed, no file to re-read
                        c = dict(values={k: d[k] for k in ('Q', 'R', 'dR',
                                                           'dQ')},
                                 path=d['path'])
                    c['pol'] = d.get('pol')
                    e['channels'][ch] = c
            sets.append(e)
        return dict(
            sets=sets,
            display=dict(quantity=self.quantity.currentIndex(),
                         logy=self.logy.isChecked(), rq4=self.rq4.isChecked(),
                         errors=self.show_err.isChecked()),
            fit=dict(dataset=self.fit_set.currentText(),
                     cost=self.fit_cost.currentIndex(),
                     maxiter=self.fit_maxiter.value(),
                     mutation=[self.fit_mut_lo.value(),
                               self.fit_mut_hi.value()],
                     population=self.fit_pop.value(), tol=self.fit_tol.value(),
                     polish=self.fit_polish.isChecked(),
                     workers=self.fit_workers.value()))

    def restore_state(self, state, base):
        """Inverse of session_state, after the Simulation tab is restored
        (live datasets are computed from it).  Returns the channels that
        could not be read, as text."""
        missing = []
        self.sets = []
        for e in state.get('sets', []):
            s = dict(name=e['name'], visible=e['visible'], channels={})
            if e.get('live'):
                s.update(live=True, line=True, pairs=e.get('pairs'))
                self.sets.append(s)
                continue
            if e.get('line'):
                s['line'] = True
            for ch, c in e.get('channels', {}).items():
                if 'values' in c:
                    d = {k: np.asarray(v, dtype=float)
                         for k, v in c['values'].items()}
                    d.update(theta=np.full_like(d['Q'], np.nan),
                             path=c['path'])
                else:
                    try:
                        d = read_channel(c['file'], c.get('q_file'),
                                         c['start'], base)
                    except (OSError, ValueError) as exc:
                        missing.append('%s — R%s: %s' % (e['name'], ch, exc))
                        continue
                s['channels'][ch] = dict(d, pol=c.get('pol'))
            self.sets.append(s)
        sim = self.simulation
        for s in self.sets:
            if s.get('live') and sim is not None and sim.last_R is not None:
                s['channels'] = simulation_channels(sim.stack, sim.last_Q,
                                                    self.live_vectors(s))

        disp, fit = state.get('display', {}), state.get('fit', {})
        widgets = (self.quantity, self.logy, self.rq4, self.show_err)
        for w in widgets:
            w.blockSignals(True)
        self.quantity.setCurrentIndex(disp.get('quantity', 0))
        self.logy.setChecked(disp.get('logy', True))
        self.rq4.setChecked(disp.get('rq4', False))
        self.show_err.setChecked(disp.get('errors', True))
        for w in widgets:
            w.blockSignals(False)
        self.refresh_tree()
        if fit:
            self.fit_set.setCurrentIndex(max(0, self.fit_set.findText(
                fit.get('dataset', ''))))
            self.fit_cost.setCurrentIndex(fit.get('cost', 0))
            self.fit_maxiter.setValue(fit.get('maxiter', 200))
            lo, hi = fit.get('mutation', (0.5, 1.0))
            self.fit_mut_lo.setValue(lo)
            self.fit_mut_hi.setValue(hi)
            self.fit_pop.setValue(fit.get('population', 15))
            self.fit_tol.setValue(fit.get('tol', 0.01))
            self.fit_polish.setChecked(fit.get('polish', True))
            self.fit_workers.setValue(fit.get('workers',
                                              self.fit_workers.value()))
        self._fit_start = None
        self.btn_revert.setEnabled(False)
        self.redraw()
        return missing

    # -- drawing ------------------------------------------------------------
    def redraw(self, *_):
        p = self.plot
        for curve, err in self._items:
            p.removeItem(curve)
            if err is not None:
                p.removeItem(err)
        self._items, self._shown = [], []
        p.legend.clear()

        k = self.quantity.currentIndex()
        name, label, formula = REFL_QUANTITIES[k]
        is_refl = label is None
        for w in (self.logy, self.rq4):
            w.setEnabled(is_refl)
        self.zero.setVisible(not is_refl)
        self.formula.setText(formula)

        if is_refl:
            log = self.logy.isChecked()
            rq4 = self.rq4.isChecked()
            p.setLabel('left', 'R·Q⁴ (Å⁻⁴)' if rq4 else 'Reflectivity')
        else:
            log = rq4 = False
            p.setLabel('left', label)
        p.setLogMode(x=False, y=log)
        self._log = log

        # colour = channel, marker = dataset; single-series quantities
        # (asymmetries) are coloured per dataset instead
        shown = [(i, s) for i, s in enumerate(self.sets)
                 if s['visible'] and s['channels']]
        prefix = len(shown) > 1
        notes = []
        for i, s in shown:
            series, note = dataset_series(s['channels'], k,
                                          s.get('line', False))
            if note:
                notes.append('%s: %s' % (s['name'], note))
            symbol = SYMBOLS[i % len(SYMBOLS)]
            for sname, col, Q, y, dy, dQ in series:
                if self.simulation is not None:
                    keep = (Q >= self.simulation.qmin.value()) & \
                        (Q <= self.simulation.qmax.value())
                    Q, y, dy, dQ = Q[keep], y[keep], dy[keep], dQ[keep]
                if col is None:
                    col = SET_COLOURS[i % len(SET_COLOURS)]
                if rq4:
                    y, dy = y * Q**4, dy * Q**4
                lab = '%s — %s' % (s['name'], sname) if prefix else sname
                self._add_series(lab, col, symbol, Q, y, dy, dQ, log,
                                 line=s.get('line', s.get('live', False)))

        n = sum(len(s['channels']) for s in self.sets)
        if not n:
            self.status.setText('No data loaded — use Import…')
        elif notes:
            self.status.setText('<span style="color:#f87171">%s — %s</span>'
                                % (name, '<br>'.join(notes)))
        else:
            self.status.setText('%d channel(s) in %d dataset(s), %d plotted'
                                % (n, len(self.sets), len(shown)))
        p.enableAutoRange()
        self.cursor.refresh()

    def _add_series(self, name, col, symbol, Q, y, dy, dQ, log, line=False):
        """Markers with error bars, or a plain line (simulated data)."""
        if line:
            # keep gaps (NaN) as breaks in the line
            yy = np.where(np.isfinite(y) & ((y > 0) if log else True), y, np.nan)
            curve = pg.PlotDataItem(Q, yy, pen=pg.mkPen(col, width=1.8),
                                    connect='finite')
            self.plot.addItem(curve)
            self.plot.legend.addItem(curve, name)
            self._items.append((curve, None))
            ok = np.isfinite(yy)
            self._shown.append((curve, name, Q[ok], yy[ok],
                                np.zeros(ok.sum())))
            return
        ok = np.isfinite(y) & (y > 0 if log else True)
        Q, y, dy, dQ = Q[ok], y[ok], dy[ok], dQ[ok]
        dy = np.where(np.isfinite(dy), np.abs(dy), 0.0)
        dQ = np.where(np.isfinite(dQ), np.abs(dQ), 0.0)
        curve = pg.PlotDataItem(Q, y, pen=None, symbol=symbol, symbolSize=6,
                                symbolPen=pg.mkPen(col), symbolBrush=col)
        # ErrorBarItem has no log mode: give it log10 coordinates directly,
        # with the asymmetric bars that a symmetric dR becomes in log space
        if log:
            yl = np.log10(y)
            top = np.log10(y + dy) - yl
            # R - dR <= 0 would reach -inf: cap the bar at 3 decades
            bottom = yl - np.log10(np.maximum(y - dy, 1e-3 * y))
            err = pg.ErrorBarItem(x=Q, y=yl, top=top, bottom=bottom,
                                  left=dQ, right=dQ, beam=0,
                                  pen=pg.mkPen(col, width=0.8))
        else:
            err = pg.ErrorBarItem(x=Q, y=y, height=2 * dy, width=2 * dQ,
                                  beam=0, pen=pg.mkPen(col, width=0.8))
        self.plot.addItem(err)
        self.plot.addItem(curve)
        self.plot.legend.addItem(curve, name)
        err.setVisible(self.show_err.isChecked())
        # legend clicks toggle the curve; its error bars follow
        curve.visibleChanged.connect(
            lambda c=curve, e=err: e.setVisible(
                c.isVisible() and self.show_err.isChecked()))
        self._items.append((curve, err))
        self._shown.append((curve, name, Q, y, dy))

    def _readout(self, x):
        """Cursor text: nearest data point of each visible series."""
        if not self._shown:
            return None, []
        rows = []
        for curve, name, Q, y, dy in self._shown:
            if not curve.isVisible() or not len(Q) or not (Q[0] <= x <= Q[-1]):
                continue
            i = int(np.argmin(np.abs(Q - x)))
            yv = np.log10(y[i]) if self._log else y[i]
            pen = curve.opts['symbolPen'] if curve.opts['symbol'] else \
                curve.opts['pen']
            col = pg.mkPen(pen).color().name()
            rows.append(('%s @ %.5g' % (name, Q[i]),
                         '%s ± %s' % (fmt(y[i]), fmt(dy[i])),
                         col, yv, self.plot.vb))
        return 'Q = %.5g Å⁻¹' % x, rows
