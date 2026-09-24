"""
Experimental tab: import measured reflectivity channels and plot them with
the same quantities as the simulation tab, with error bars.

Each file holds one channel (R+, R-, R++, R+-, R-+ or R--) as columns
    Qz (1/A)   R (a.u.)   dR (a.u.)   dQz (1/A)   [theta]
Importing opens the file text in a dialog with line numbers; the user picks
the line where the data start and which channel the file is.

Asymmetries combine channels measured on different Q grids: every channel is
linearly interpolated (value and error) onto the Q points of the first
channel in the formula, over the range they share.  Errors are propagated
assuming independent channels.  R+ / R- are used as imported when present,
otherwise built as R+ = R++ + R+-, R- = R-- + R-+.
"""

import os
import re

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from .simulation import (CH_COLOURS, HP_COLOURS, ASYM_COLOUR,
                         REFL_QUANTITIES, Crosshair, fmt, plot_widget)


DATA_FILTER = 'Data files (*.dat *.txt *.csv *.refl);;All files (*)'
DIALOG_OPTIONS = QtWidgets.QFileDialog.Option.DontUseNativeDialog
# import order in the channel combo; colours follow the simulation tab
DATA_CHANNELS = ['+', '-', '++', '-+', '+-', '--']
CHANNEL_COLOURS = {'++': CH_COLOURS[0], '+-': CH_COLOURS[1],
                   '-+': CH_COLOURS[2], '--': CH_COLOURS[3],
                   '+': HP_COLOURS[0], '-': HP_COLOURS[1]}
PLOT_ORDER = ['++', '+-', '-+', '--', '+', '-']
SPLIT = re.compile(r'[\s,;]+')


def parse_row(line):
    """Floats of one data line, or None if it is not numeric."""
    parts = [p for p in SPLIT.split(line.strip()) if p]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def guess_start(lines):
    """First line (0-based) that parses as at least 4 numbers."""
    for i, line in enumerate(lines):
        row = parse_row(line)
        if row is not None and len(row) >= 4:
            return i
    return 0


def parse_data(lines, start):
    """Parse lines[start:] into a dict of Q, R, dR, dQ, theta arrays, sorted
    by Q.  Blank / comment lines are skipped; returns (data, n_skipped)."""
    rows, skipped = [], 0
    for line in lines[start:]:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        row = parse_row(s)
        if row is None or len(row) < 4:
            skipped += 1
            continue
        rows.append(row[:5] + [np.nan] * (5 - len(row[:5])))
    if not rows:
        raise ValueError('no data rows with at least 4 numeric columns')
    a = np.array(rows, dtype=float)
    a = a[np.argsort(a[:, 0])]
    keys = ['Q', 'R', 'dR', 'dQ', 'theta']
    return {k: a[:, i] for i, k in enumerate(keys)}, skipped


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
        return 12 + self.fontMetrics().horizontalAdvance('9') * digits

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
        p.fillRect(ev.rect(), QtGui.QColor('#eeeeee'))
        block = self.firstVisibleBlock()
        top = self.blockBoundingGeometry(block).translated(
            self.contentOffset()).top()
        h = self.fontMetrics().height()
        while block.isValid() and top <= ev.rect().bottom():
            n = block.blockNumber()
            if block.isVisible() and top + h >= ev.rect().top():
                p.setPen(QtGui.QColor('#b00020' if n == self.start
                                      else '#888888'))
                p.drawText(0, int(top), self.gutter.width() - 5, h,
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
            first.format.setBackground(QtGui.QColor('#ffe08a'))
            first.format.setProperty(
                QtGui.QTextFormat.Property.FullWidthSelection, True)
            first.cursor = QtGui.QTextCursor(block)
            sel.append(first)
            rest = QtWidgets.QTextEdit.ExtraSelection()
            rest.format.setBackground(QtGui.QColor('#fff6d6'))
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
    """Show a data file, let the user pick the first data line and the
    channel; `data` / `channel` hold the result after accept()."""

    def __init__(self, path, parent=None, sets=(), current=None, taken=None):
        super().__init__(parent)
        self.setWindowTitle('Import — %s' % os.path.basename(path))
        with open(path, errors='replace') as f:
            text = f.read()
        self.lines = text.splitlines()
        self.data = None

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

        form = QtWidgets.QFormLayout()
        form.addRow('Data start at line', self.start)
        form.addRow('Dataset', self.dataset)
        form.addRow('Channel', self.channel)
        form.addRow(QtWidgets.QLabel(
            '<i>Columns: Qz (Å⁻¹), R (a.u.), dR (a.u.), dQz (Å⁻¹), '
            '[θ]. Click a line in the text to start the data there.</i>'))
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

        self.start.valueChanged.connect(self._on_start)
        self.view.lineClicked.connect(lambda n: self.start.setValue(n + 1))
        self.start.setValue(guess_start(self.lines) + 1)
        self._on_start()
        self.resize(820, 680)

    def _on_start(self, *_):
        line = self.start.value() - 1
        self.view.set_start(line)
        try:
            self.data, skipped = parse_data(self.lines, line)
        except ValueError as exc:
            self.data = None
            self.preview.setText('<span style="color:#b00020">%s</span>' % exc)
            self.ok.setEnabled(False)
            return
        d = self.data
        msg = '%d points, Q = %.4g … %.4g Å⁻¹' % (len(d['Q']), d['Q'][0],
                                                   d['Q'][-1])
        if skipped:
            msg += ('  <span style="color:#b00020">(%d non-numeric lines '
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


def dataset_series(D, k):
    """Series of REFL_QUANTITIES[k] for one dataset as
    ([(label, colour, Q, y, dy, dQ)], note); colour None = per-dataset."""
    name, label, _ = REFL_QUANTITIES[k]
    if k == 0:
        return [('R' + ch, CHANNEL_COLOURS[ch], D[ch]['Q'], D[ch]['R'],
                 D[ch]['dR'], D[ch]['dQ']) for ch in PLOT_ORDER if ch in D], ''
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


# ------------------------------------------------------- data tree ----
ROLE = QtCore.Qt.ItemDataRole.UserRole          # (set index, channel or None)
SYMBOLS = ['o', 's', 't', 'd', 'star', 't1', 'p', 'h', '+', 'x']
# marker shown beside each dataset name in the tree
SYMBOL_GLYPH = {'o': '●', 's': '■', 't': '▼', 'd': '◆', 'star': '★',
                't1': '▲', 'p': '⬟', 'h': '⬢', '+': '+', 'x': '×'}
SET_COLOURS = ['#6a3d9a', '#1b9e77', '#d95f02', '#e7298a', '#66a61e',
               '#e6ab02', '#a6761d', '#1f78b4']


def glyph_icon(glyph):
    """Small icon showing a dataset's marker glyph."""
    pix = QtGui.QPixmap(16, 16)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pix)
    p.setPen(QtGui.QColor('#333333'))
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


# ------------------------------------------------- experimental tab ----
class ExperimentalTab(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        # [dict(name, visible, channels={channel: dict(Q, R, dR, dQ, ...)})]
        self.sets = []

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_plot())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1270])

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
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
        self.btn_remove = QtWidgets.QPushButton('− Remove')
        for b in (self.btn_import, self.btn_new, self.btn_remove):
            row.addWidget(b)
        bv.addLayout(row)
        self.btn_import.clicked.connect(self.import_data)
        self.btn_new.clicked.connect(lambda: self.new_set())
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
        f.addRow('Scale', opts)
        f.addRow('', self.show_err)
        for w in (self.logy, self.rq4, self.show_err):
            w.toggled.connect(self.redraw)
        v.addWidget(disp)

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
        self.formula.setStyleSheet('color: #555555')
        bar.addWidget(self.formula)
        bar.addStretch(1)
        v.addLayout(bar)

        w, self.plot = plot_widget('Qz (Å⁻¹)', 'Reflectivity')
        self.plot.addLegend(offset=(-10, 10))
        self.zero = pg.InfiniteLine(pos=0, angle=0,
                                    pen=pg.mkPen('#bbbbbb', width=0.7))
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
            top.setFlags((top.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                          | QtCore.Qt.ItemFlag.ItemIsEditable
                          | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
                         & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled)
            top.setCheckState(0, QtCore.Qt.CheckState.Checked if s['visible']
                              else QtCore.Qt.CheckState.Unchecked)
            font = top.font(0)
            font.setBold(True)
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
                it.setForeground(0, QtGui.QColor(CHANNEL_COLOURS[ch]))
                it.setToolTip(0, d['path'])
                top.addChild(it)
                if select == (i, ch):
                    current = it
            top.setExpanded(True)
        t.blockSignals(False)
        if current is not None:
            t.setCurrentItem(current)
        self._update_buttons()

    def _selected(self):
        it = self.tree.currentItem()
        return it.data(0, ROLE) if it is not None else None

    def _update_buttons(self, *_):
        self.btn_remove.setEnabled(self._selected() is not None)

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
        if ch is None:
            menu.addAction('Rename', lambda: self.tree.editItem(it, 0))
            menu.addAction('Import into this dataset…',
                           lambda: self.import_data(self.sets[i]['name']))
        else:
            move = menu.addMenu('Move to')
            for j, s in enumerate(self.sets):
                if j != i:
                    move.addAction(s['name'],
                                   lambda j=j: self.move_channel(i, ch, j))
            move.addAction('New dataset', lambda: self.move_channel(
                i, ch, self.new_set(refresh=False)))
        menu.addAction('Remove', self.remove_selected)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

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
        for i, s in enumerate(self.sets):
            if s['name'] == name:
                return i
        return None

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
            dlg = ImportDialog(
                path, self, [s['name'] for s in self.sets], dataset,
                {s['name']: set(s['channels']) for s in self.sets})
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
        self.sets[i]['channels'][ch] = dict(dlg.data, path=path)
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

    # -- drawing ------------------------------------------------------------
    def redraw(self, *_):
        p = self.plot
        for curve, err in self._items:
            p.removeItem(curve)
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
            series, note = dataset_series(s['channels'], k)
            if note:
                notes.append('%s: %s' % (s['name'], note))
            symbol = SYMBOLS[i % len(SYMBOLS)]
            for sname, col, Q, y, dy, dQ in series:
                if col is None:
                    col = SET_COLOURS[i % len(SET_COLOURS)]
                if rq4:
                    y, dy = y * Q**4, dy * Q**4
                lab = '%s — %s' % (s['name'], sname) if prefix else sname
                self._add_series(lab, col, symbol, Q, y, dy, dQ, log)

        n = sum(len(s['channels']) for s in self.sets)
        if not n:
            self.status.setText('No data loaded — use Import…')
        elif notes:
            self.status.setText('<span style="color:#b00020">%s — %s</span>'
                                % (name, '<br>'.join(notes)))
        else:
            self.status.setText('%d channel(s) in %d dataset(s), %d plotted'
                                % (n, len(self.sets), len(shown)))
        p.enableAutoRange()
        self.cursor.refresh()

    def _add_series(self, name, col, symbol, Q, y, dy, dQ, log):
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
            col = curve.opts['symbolPen'].color().name()
            rows.append(('%s @ %.5g' % (name, Q[i]),
                         '%s ± %s' % (fmt(y[i]), fmt(dy[i])),
                         col, yv, self.plot.vb))
        return 'Q = %.5g Å⁻¹' % x, rows
