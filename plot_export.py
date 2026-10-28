"""
pyqtgraph's "CSV of original plot data" export (right-click on a plot ->
Export), with one addition: a plot that has a `layer_at` attribute, a
function of depths z returning the name of the layer each one is in (the
SLD profile plots), gets a last column 'layer' naming the layer of every
row, at the depth of the first exported curve.  Every other plot exports
exactly as pyqtgraph's own exporter does.  Importing this module puts it
in pyqtgraph's list in place of the original.
"""

import csv
import itertools

import numpy as np
from pyqtgraph import ErrorBarItem
from pyqtgraph.exporters import CSVExporter
from pyqtgraph.exporters.Exporter import Exporter


class LayerCSVExporter(CSVExporter):
    """CSVExporter, plus the 'layer' column of the profile plots."""

    def export(self, fileName=None):
        layer_at = getattr(self.item, 'layer_at', None)
        if layer_at is None or fileName is None:
            return super().export(fileName)
        for item in self.item.items:
            if isinstance(item, ErrorBarItem):
                self._exportErrorBarItem(item)
            elif hasattr(item, 'implements') and item.implements('plotData'):
                self._exportPlotDataItem(item)
        if self.data:
            z = np.asarray(self.data[0][0], dtype=float)
            self.header.append('layer')
            self.data.append([list(layer_at(z))])
        sep = ',' if self.params['separator'] == 'comma' else '\t'
        columns = [column for dataset in self.data for column in dataset]
        with open(fileName, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.header)
            for row in itertools.zip_longest(*columns, fillvalue=''):
                writer.writerow([
                    v if isinstance(v, str) else np.format_float_positional(
                        v, precision=self.params['precision'])
                    for v in row])
        self.header.clear()
        self.data.clear()


def layer_names(Z, names):
    """layer_at for interfaces Z (between names[i] and names[i + 1]): the
    name of the layer at each depth, as the profile cursor reads it."""
    Z, names = np.asarray(Z, dtype=float), list(names)
    return lambda z: [names[i] for i in np.searchsorted(Z, z)]


# in place of pyqtgraph's own, at the same place in its list
_list = Exporter.Exporters
if CSVExporter in _list:
    _list[_list.index(CSVExporter)] = LayerCSVExporter
elif LayerCSVExporter not in _list:
    _list.append(LayerCSVExporter)
