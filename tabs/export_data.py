"""
Export… of the Experimental tab: datasets written as CSV files.

WHAT -- for each dataset picked (measured, a live simulation or a frozen
copy), any of
  the SLD profile of the model it is linked to (a frozen copy: its own),
      `<name>_SLD_profile.csv`: the depth profile as the Simulation tab
      plots it (z, NSLD Re / Im, MSLD rho, phi, theta, and the layer at each
      depth, as its pyqtgraph CSV export), then a table of the interfaces:
      their depth, the two layers, the roughness sigma and the magnetic
      roughness sigma_M (MSLD_sigma, else sigma: the width the magnetic
      profile really has, model.stack MAGNETIC ROUGHNESS);
  the seven quantities of the plot (tabs.simulation REFL_QUANTITIES), each
      `<name>_<tag>.csv` (QUANTITY_TAGS, e.g. SA_half_polarized); the three
      reflectivities (polarized with analysis, without, non-polarized) once
      per scale picked, `<name>_<tag>_<scale>.csv` (SCALES: R, logR, RQ4).
The values and errors are the plot's (tabs.experimental dataset_series:
the half-polarized R+ / R- summed from the spin channels when not
imported, the asymmetries with their propagated errors), optionally only
the points within the Q range of the dataset's model, as plotted.

COLUMNS -- per measured curve of the file (e.g. R++, R+-, R-+, R--): its Q,
its value, the value's error and its dQ, each named after the curve with
its unit:
  'R++ data Q (Å⁻¹)', 'R++ data', 'R++ data error', 'R++ data dQ (Å⁻¹)'  R
  'log10(R++ data)', 'log10(R++ data) error'                          logR
  'R++ data·Q⁴ (Å⁻⁴)', 'R++ data·Q⁴ error (Å⁻⁴)'                       RQ4
then, for a measured dataset (unless unticked), the curves of its model as
the plot shows them (its live simulation dataset, on the model's own Q
points): 'R++ model Q (Å⁻¹)', 'R++ model' (or its log10 / ·Q⁴), with no
error or dQ.  A simulated dataset exported itself has 'model' columns.
The errors are those of the plot (one standard deviation) scaled with the
value: d(log10 R) = dR / (R ln 10) (log10 R and its error empty where
R <= 0), d(R Q⁴) = dR Q⁴ (Q taken as exact).  The asymmetries (SA, the SF
fraction) are ratios: one scale, no unit.  Curves of one file on
different Q points are side by side, the shorter ones padded with empty
cells.  UTF-8 with a byte-order mark, so that Excel reads Å and ⁻¹.

FILES -- one dataset: its files in the folder chosen; several: one
sub-folder per dataset, named after it.  <name> is the name typed in the
dialog, else the dataset's name; characters a file name cannot hold
become '_'.
"""

import csv
import itertools
import os
import re

import numpy as np
from PyQt6 import QtCore, QtWidgets

from model.stack import Stack
from .simulation import (HALF_POLARIZED, POLARIZED, REFL_QUANTITIES,
                         UNPOLARIZED, profile_data, roughness_stems)

# REFL_QUANTITIES name -> file tag
QUANTITY_TAGS = {
    POLARIZED: 'polarized',
    HALF_POLARIZED: 'half_polarized',
    UNPOLARIZED: 'non_polarized',
    'SA NSF': 'SA_NSF',
    'SF fraction': 'SF_fraction',
    'SA SF': 'SA_SF',
    'SA half-polarized': 'SA_half_polarized',
}
# the reflectivities (the others are ratios: no scale)
REFLECTIVITIES = (POLARIZED, HALF_POLARIZED, UNPOLARIZED)
# scale key -> (file tag, label in the dialog)
SCALES = {'R': ('R', 'R'), 'log': ('logR', 'log₁₀ R'),
          'rq4': ('RQ4', 'R·Q⁴')}
PROFILE_TAG = 'SLD_profile'
INV_A = 'Å⁻¹'
SLD_UNIT = '10⁻⁶ Å⁻²'


def safe_name(text):
    """text as a file or folder name: characters Windows refuses as '_'."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(text)).strip().rstrip('.')
    return s or 'dataset'


def curve_label(label):
    """A curve's plot label in a column name: plain minus signs."""
    return str(label).replace('−', '-')


def scaled(Q, y, dy, scale):
    """(value, error) of reflectivity y ± dy in `scale` (COLUMNS)."""
    Q, y, dy = (np.asarray(a, dtype=float) for a in (Q, y, dy))
    if scale == 'log':
        with np.errstate(divide='ignore', invalid='ignore'):
            ok = y > 0
            v = np.where(ok, np.log10(np.where(ok, y, 1.0)), np.nan)
            e = np.where(ok, dy / (np.where(ok, y, 1.0) * np.log(10)),
                         np.nan)
        return v, e
    if scale == 'rq4':
        q4 = Q**4
        return y * q4, dy * q4
    return y, dy


def series_columns(series, scale=None, kind='data'):
    """(header, columns) of a file's curves [(label, colour, Q, y, dy, dQ)]
    (tabs.experimental dataset_series) in `scale` ('R', 'log', 'rq4'; None
    for a ratio), see COLUMNS.  kind 'data': Q, value, error, dQ, each
    named '<curve> data …'; 'model' (a computed curve): Q and value only,
    '<curve> model …'."""
    header, cols = [], []
    for label, _, Q, y, dy, dQ in series:
        lab = '%s %s' % (curve_label(label), kind)
        if scale in ('log', 'rq4'):
            v, e = scaled(Q, y, dy, scale)
            names = (('log10(%s)' % lab, 'log10(%s) error' % lab)
                     if scale == 'log' else
                     ('%s·Q⁴ (Å⁻⁴)' % lab, '%s·Q⁴ error (Å⁻⁴)' % lab))
        else:
            v, e = np.asarray(y, dtype=float), np.asarray(dy, dtype=float)
            names = (lab, '%s error' % lab)
        if kind == 'model':
            header += ['%s Q (%s)' % (lab, INV_A), names[0]]
            cols += [np.asarray(Q, dtype=float), v]
            continue
        header += ['%s Q (%s)' % (lab, INV_A), names[0], names[1],
                   '%s dQ (%s)' % (lab, INV_A)]
        cols += [np.asarray(Q, dtype=float), v, e,
                 np.asarray(dQ, dtype=float)]
    return header, cols


def _crop(series, qrange):
    """series within qrange (None: as they are), empty curves left out."""
    if qrange is None:
        return series
    lo, hi = qrange
    out = []
    for lab, col, Q, y, dy, dQ in series:
        keep = (Q >= lo) & (Q <= hi)
        if keep.any():
            out.append((lab, col, Q[keep], y[keep], dy[keep], dQ[keep]))
    return out


def profile_columns(stack):
    """(header, columns) of a model's SLD profile and its interfaces (WHAT);
    builds its sublayers."""
    st = stack
    st.build_sublayers()
    (z, Z, _, curves, _), _ = profile_data(st)
    nsld_re, rho, phi_inplane, theta_out, nsld_im = curves
    names = [l.name for l in st.layers]
    layer = [names[i] for i in np.searchsorted(np.asarray(Z, dtype=float), z)]
    sig, mag = roughness_stems(st)
    mag = dict(mag)
    zi = [Zj for Zj, _ in sig]
    header = ['z (Å)', 'NSLD Re (%s)' % SLD_UNIT, 'NSLD Im (%s)' % SLD_UNIT,
              'MSLD ρ (%s)' % SLD_UNIT, 'MSLD φ in plane (deg)',
              'MSLD θ out of plane (deg)', 'layer',
              'interface z (Å)', 'interface (above / below)',
              'roughness σ (Å)', 'magnetic roughness σ_M (Å)']
    cols = [z, nsld_re, nsld_im, rho, phi_inplane, theta_out, layer,
            np.asarray(zi, dtype=float),
            ['%s / %s' % (names[j], names[j + 1]) for j in range(len(sig))],
            np.asarray([s for _, s in sig], dtype=float),
            np.asarray([mag.get(Zj, s) for Zj, s in sig], dtype=float)]
    return header, cols


def _cell(v):
    """A text cell as it is, a number with 10 significant digits."""
    if isinstance(v, str):
        return v
    v = float(v)
    return 'nan' if np.isnan(v) else '%.10g' % v


def write_csv(path, header, cols):
    """Columns of different lengths side by side, padded with ''."""
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in itertools.zip_longest(*cols, fillvalue=''):
            w.writerow([_cell(v) for v in row])


def dataset_files(s, prefix, quantities, scales, profile_stack=None,
                  qrange=None, model=None):
    """[(file name, header, columns)] of dataset s (WHAT, FILES), and
    [notes] on what it lacks.  quantities: REFL_QUANTITIES names; scales:
    SCALES keys for the reflectivities; profile_stack: the Stack whose
    profile to write (None: none); qrange: (Q min, Q max) to keep, None
    all; model: the channels of its model's curve (the live simulation
    dataset), written beside the data in every file (None: none).  A
    simulated dataset's own curves are 'model' columns."""
    from .experimental import dataset_series
    names = [q[0] for q in REFL_QUANTITIES]
    line = s.get('line', s.get('live', False))
    kind = 'model' if line else 'data'
    out, notes = [], []
    if profile_stack is not None:
        out.append(('%s_%s.csv' % (prefix, PROFILE_TAG),)
                   + profile_columns(profile_stack))
    for name in quantities:
        k = names.index(name)
        series, note = dataset_series(s['channels'], k, line,
                                      lambda key: None)
        series = _crop(series, qrange)
        if not series:
            notes.append('%s: %s' % (name, note or 'no point in the Q range'))
            continue
        # the model's curves that the data have (e.g. R+ alone for an X-ray
        # set): taken as the data's are, then matched by their label
        have = {lab for lab, *_ in series}
        curve = [] if model is None else [
            c for c in _crop(dataset_series(model, k, False,
                                            lambda key: None)[0], qrange)
            if c[0] in have]

        def table(sc):
            h, c = series_columns(series, sc, kind)
            if curve:
                hm, cm = series_columns(curve, sc, 'model')
                h, c = h + hm, c + cm
            return h, c
        tag = QUANTITY_TAGS[name]
        if name in REFLECTIVITIES:
            for sc in scales:
                out.append(('%s_%s_%s.csv' % (prefix, tag, SCALES[sc][0]),)
                           + table(sc))
        else:
            out.append(('%s_%s.csv' % (prefix, tag),) + table(None))
    return out, notes


class ExportDialog(QtWidgets.QDialog):
    """Export…: the datasets, what to write, the name and the folder."""

    def __init__(self, exp, folder=''):
        super().__init__(exp)
        self.exp, self.folder = exp, folder
        self.setWindowTitle('Export datasets')
        v = QtWidgets.QVBoxLayout(self)

        v.addWidget(QtWidgets.QLabel('Datasets:'))
        self.sets = QtWidgets.QListWidget()
        sel = exp._selected()
        for i, s in enumerate(exp.sets):
            if not s.get('channels'):
                continue
            it = QtWidgets.QListWidgetItem(s['name'])
            it.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            on = sel is not None and sel[0] == i
            it.setCheckState(QtCore.Qt.CheckState.Checked if on
                             else QtCore.Qt.CheckState.Unchecked)
            self.sets.addItem(it)
        self.sets.itemChanged.connect(self._update)
        v.addWidget(self.sets, 1)

        box = QtWidgets.QGroupBox('Save')
        g = QtWidgets.QGridLayout(box)
        self.profile = QtWidgets.QCheckBox('SLD profile of its model')
        self.profile.setChecked(True)
        self.profile.setToolTip(
            'z, NSLD Re / Im, MSLD ρ, φ, θ and the layer at each depth, as '
            'the Simulation tab\'s profile;\nthen each interface with its '
            'roughness σ and magnetic roughness σ_M.')
        g.addWidget(self.profile, 0, 0, 1, 2)
        self.quantities = {}
        for r, (name, _, formula) in enumerate(REFL_QUANTITIES, 1):
            b = self.quantities[name] = QtWidgets.QCheckBox(name)
            b.setToolTip('%s — file …_%s' % (formula, QUANTITY_TAGS[name]))
            b.setChecked(name == POLARIZED)
            b.toggled.connect(self._update)
            g.addWidget(b, (r - 1) % 4 + 1, (r - 1) // 4)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Reflectivity as:'))
        self.scales = {}
        for key, (tag, label) in SCALES.items():
            b = self.scales[key] = QtWidgets.QCheckBox(label)
            b.setChecked(key == 'R')
            b.setToolTip('File …_%s; the errors scaled with the values' % tag)
            b.toggled.connect(self._update)
            row.addWidget(b)
        row.addStretch(1)
        g.addLayout(row, 5, 0, 1, 2)
        self.in_range = QtWidgets.QCheckBox(
            'only the points within each model\'s Q range (as plotted)')
        self.in_range.setChecked(True)
        g.addWidget(self.in_range, 6, 0, 1, 2)
        self.with_model = QtWidgets.QCheckBox(
            'with the model curve beside the data (its own Q column)')
        self.with_model.setChecked(True)
        self.with_model.setToolTip(
            'A measured dataset\'s files also hold the curve of the model it '
            'is linked to, as plotted\n(its "(simulation)" dataset, on the '
            'model\'s Q points): columns "… model".')
        g.addWidget(self.with_model, 7, 0, 1, 2)
        self.profile.toggled.connect(self._update)
        v.addWidget(box)

        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText('the dataset\'s name')
        self.name.setToolTip(
            'Start of every file name, e.g. name_SA_half_polarized.csv;\n'
            'empty: each dataset\'s name.  Several datasets go in a folder '
            'each, named after it.')
        self.name.textChanged.connect(self._update)
        form.addRow('Name', self.name)
        v.addLayout(form)
        self.preview = QtWidgets.QLabel(wordWrap=True)
        self.preview.setProperty('role', 'muted')
        v.addWidget(self.preview)

        self.bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.ok = self.bb.addButton(
            'Choose folder and save…',
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        v.addWidget(self.bb)
        self.resize(560, 560)
        self._update()

    def chosen_sets(self):
        return [self.sets.item(r).data(QtCore.Qt.ItemDataRole.UserRole)
                for r in range(self.sets.count())
                if self.sets.item(r).checkState()
                == QtCore.Qt.CheckState.Checked]

    def chosen_quantities(self):
        return [n for n, b in self.quantities.items() if b.isChecked()]

    def chosen_scales(self):
        return [k for k, b in self.scales.items() if b.isChecked()]

    def prefix(self, s):
        return safe_name(self.name.text().strip() or s['name'])

    def _update(self, *_):
        refl = any(self.quantities[n].isChecked() for n in REFLECTIVITIES)
        for b in self.scales.values():
            b.setEnabled(refl)
        sets = self.chosen_sets()
        what = self.profile.isChecked() or bool(self.chosen_quantities())
        ok = bool(sets) and what and (not refl or bool(self.chosen_scales()))
        self.ok.setEnabled(ok)
        if not sets:
            text = 'Tick the datasets to save.'
        elif not what:
            text = 'Tick what to save.'
        elif refl and not self.chosen_scales():
            text = 'Tick at least one scale for the reflectivities.'
        else:
            s = self.exp.sets[sets[0]]
            ex = ['%s_%s' % (self.prefix(s), QUANTITY_TAGS[n]
                             + ('_' + SCALES[self.chosen_scales()[0]][0]
                                if n in REFLECTIVITIES else ''))
                  for n in self.chosen_quantities()][:2]
            if self.profile.isChecked():
                ex.insert(0, '%s_%s' % (self.prefix(s), PROFILE_TAG))
            text = 'e.g. %s.csv' % ', '.join(ex[:2])
            text += (' — in the folder chosen' if len(sets) == 1 else
                     ' — in one sub-folder per dataset, named after it')
        self.preview.setText(text)


def model_curve(exp, s):
    """The channels of the curve of measured dataset s's model: its live
    simulation dataset's, as plotted (its model's Q grid and norms); None
    for a simulated dataset or without one."""
    if s.get('line') or s.get('live') or exp.simulation is None:
        return None
    link = exp._link(s)
    live = next((t for t in exp.sets
                 if t.get('live') and t.get('link') == link), None)
    if live is None:
        return None
    return exp._live_channels(live) or None


def export_datasets(exp, folder, indices, prefix_of, quantities, scales,
                    profile=True, in_range=True, with_model=True):
    """Write the files of datasets exp.sets[indices] (FILES) under folder:
    ([(path, header, columns)], [notes]).  prefix_of(s): the name of its
    files; with_model: a measured dataset's files also hold its model's
    curve (model_curve)."""
    sim = exp.simulation
    many = len(indices) > 1
    plans, notes = [], []
    for i in indices:
        s = exp.sets[i]
        st = None
        if profile:
            if s.get('model'):                       # a frozen copy
                st = Stack.from_dict(s['model'])
            elif sim is not None:
                st = sim.model_stack(exp._link(s))
        qrange = sim.q_range(exp._link(s)) if in_range and sim else None
        curve = model_curve(exp, s) if with_model else None
        files, lacks = dataset_files(s, prefix_of(s), quantities, scales,
                                     st, qrange, curve)
        notes += ['%s — %s' % (s['name'], n) for n in lacks]
        where = os.path.join(folder, safe_name(s['name'])) if many else folder
        plans += [(os.path.join(where, f), h, c) for f, h, c in files]
    return plans, notes


def run_export(exp):
    """The Export… button: the dialog, the folder, the files."""
    from .experimental import DIALOG_OPTIONS
    dlg = ExportDialog(exp)
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        exp, 'Folder to save in', getattr(exp, 'export_dir', '') or '',
        DIALOG_OPTIONS | QtWidgets.QFileDialog.Option.ShowDirsOnly)
    if not folder:
        return
    exp.export_dir = folder
    try:
        plans, notes = export_datasets(
            exp, folder, dlg.chosen_sets(), dlg.prefix,
            dlg.chosen_quantities(), dlg.chosen_scales(),
            dlg.profile.isChecked(), dlg.in_range.isChecked(),
            dlg.with_model.isChecked())
    except Exception as exc:                        # noqa: BLE001
        QtWidgets.QMessageBox.critical(exp, 'Export', str(exc))
        return
    if not plans:
        QtWidgets.QMessageBox.information(
            exp, 'Export', 'Nothing to save.\n\n' + '\n'.join(notes))
        return
    there = [p for p, *_ in plans if os.path.exists(p)]
    if there and QtWidgets.QMessageBox.question(
            exp, 'Export', '%d file(s) already exist, e.g.\n%s\n\nReplace '
            'them?' % (len(there), there[0])) != \
            QtWidgets.QMessageBox.StandardButton.Yes:
        return
    try:
        for path, header, cols in plans:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_csv(path, header, cols)
    except OSError as exc:
        QtWidgets.QMessageBox.critical(exp, 'Export',
                                       'Could not write:\n%s' % exc)
        return
    text = 'Saved %d file(s) in\n%s' % (len(plans), folder)
    if notes:
        text += '\n\nNot saved (the dataset lacks it):\n' + '\n'.join(notes)
    QtWidgets.QMessageBox.information(exp, 'Export', text)
