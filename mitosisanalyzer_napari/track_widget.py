from typing import List
from functools import lru_cache
import logging
import numpy as np
import os
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
import pandas as pd
from pathlib import Path
from scipy import fft
from scipy.signal.windows import blackmanharris
from scipy.signal import correlate

from magicgui import magic_factory
from magicgui.widgets import create_widget
from napari.types import LayerDataTuple, ImageData
from napari.layers import Image

from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QFileDialog,
    QMessageBox,
)
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from napari_matplotlib.scatter import FeaturesScatterWidget

from mitosisanalyzer.calc import (
    oscillation,
    closest_point,
    extend_line,
    zero_crossings,
    dominant_freq,
    fft_freq,
    autocorr_freq,
)

REF_AXIS_LAYER = "Reference Axis"
POLE_LAYER = "Spindle Poles"
FRAME_COL = "Frame"
POLE_ONE_X_COL = "Pole 1,x (pixel)"
POLE_ONE_Y_COL = "Pole 1,y (pixel)"
POLE_TWO_X_COL = "Pole 2,x (pixel)"
POLE_TWO_Y_COL = "Pole 2,y (pixel)"


@lru_cache(maxsize=16)
def read_df(path):
    return pd.read_csv(path)


# def calc_ref_axis(ref_x1, ref_y1, ref_x2, ref_y2):
#    ref_xy1 = np.array([ref_x1, ref_y1])
#    ref_xy2 = np.array([ref_x2, ref_y2])
#    ref_axis = ref_xy2 - ref_xy1
#    return ref_axis, ref_xy1, ref_xy2


def create_line_shapes(x1, y1, x2, y2, n=1, xlims=(0, 512)):
    if xlims is not None:
        # extend line with xlims as endpoints
        (x1, y1), (x2, y2) = extend_line(x1, y1, x2, y2, xlims=xlims)
    # create multi-dimensional array [n, [p1, p2], [frame,y,x]]
    return np.array(
        [
            np.array(((frame, y1, x1), (frame, y2, x2)))
            for frame in np.arange(0.0, n, 1.0)
        ]
    ).reshape((n, 2, 3))


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, precision=3):
        super(TableModel, self).__init__()
        self._data = data
        self._precision = precision

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (float)):
                return f"{{:{5+self._precision}.{self._precision}f}}".format(value)
            elif isinstance(value, (int)):
                return f"{value:5}"
            else:
                return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])


class TrackEditorWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._point_layer = None
        self._axis_layer = None
        self._export_file = None
        self._summary_df = pd.DataFrame(
            {
                "Pole 1": [np.nan, np.nan, np.nan, np.nan],
                "Pole 2": [np.nan, np.nan, np.nan, np.nan],
            },
            index=[
                "Osc. zero-crossings",
                "Osc. frequency (FFT)",
                "Osc. frequency (autocorr.)",
                "Osc. dominant frequency",
            ],
        )

        # Create a button
        export_btn = QPushButton("Export to CSV")
        # Connect the click event to a function
        export_btn.clicked.connect(self._on_export)

        # create new widget with create_widget and type annotation
        self._layer_select = create_widget(annotation=Image)
        self._layer_select.changed.connect(self._on_image_changed)
        self._csv_file = create_widget(
            label="CSV file", annotation=Path, options={"filter": "*.csv"}
        )
        self._csv_file.changed.connect(self._on_csv_changed)
        self._ref_frame = create_widget(
            label="Ref frame",
            annotation=int,
            widget_type="IntSlider",
            options={"min": 0, "max": 0},
        )
        self._ref_frame.changed.connect(self._on_ref_frame_changed)
        # self._table = create_widget(label="Frequency", annotation=Table, value=self._frequencies)
        self._table = QTableView()
        self._table_model = TableModel(self._summary_df)
        self._table.setModel(self._table_model)

        # The `layer_select` widgets `reset_choices` method has to be connected to viewer.layers.events
        layers_events = self._viewer.layers.events
        layers_events.inserted.connect(self._layer_select.reset_choices)
        layers_events.removed.connect(self._layer_select.reset_choices)
        layers_events.reordered.connect(self._layer_select.reset_choices)

        self.setLayout(QVBoxLayout())
        # add it to the layout
        self.layout().addWidget(self._layer_select.native)
        self.layout().addWidget(self._csv_file.native)
        self.layout().addWidget(self._table)
        self.layout().addWidget(self._ref_frame.native)
        self.layout().addWidget(export_btn)

    def _on_image_changed(self):
        print(self._layer_select.value.scale)
        if self._point_layer is not None and self._layer_select is not None:
            self._point_layer.scale = self._layer_select.value.scale
            self._axis_layer.scale = self._layer_select.value.scale
        self._viewer.reset_view()

    def _on_csv_changed(self):
        self._df = read_df(self._csv_file.value)
        path = os.path.splitext(str(self._csv_file.value))
        self._export_file = path[0] + "-curated" + path[1]
        print(self._df.columns.values)
        no_frames = len(self._df)
        self._ref_frame.value = 0
        self._ref_frame.max = no_frames - 1

        p1 = self._df[[FRAME_COL, POLE_ONE_Y_COL, POLE_ONE_X_COL]].values
        p2 = self._df[[FRAME_COL, POLE_TWO_Y_COL, POLE_TWO_X_COL]].values
        # zero-index frame coordinates and combine
        p1[:, 0] = p1[:, 0] - 1
        p2[:, 0] = p2[:, 0] - 1
        pole_data = np.concatenate((p1, p2), axis=0)
        ref_frame = self._ref_frame.value
        line_data = create_line_shapes(
            p1[ref_frame][2],
            p1[ref_frame][1],
            p2[ref_frame][2],
            p2[ref_frame][1],
            no_frames,
        )

        pole_amps = oscillation(
            p1[ref_frame, -1:0:-1], p2[ref_frame, -1:0:-1], pole_data[:, -1:0:-1]
        )
        # pole_velocities = np.concatenate(
        #    (
        #        velocity(np.concatenate((p1[0:-1, -1:0:-1], p1[1:, -1:0:-1]), axis=1)),
        #        velocity(np.concatenate((p2[0:-1, -1:0:-1], p2[1:, -1:0:-1]), axis=1)),
        #    ),
        #    axis=0,
        # )
        # print(pole_velocities)

        pole_anot = p1.shape[0] * ["1"] + p2.shape[0] * ["2"]
        pole_colors = p1.shape[0] * ["cyan"] + p2.shape[0] * ["yellow"]
        frame_str = [f"{int(row[0]):04d}" for row in pole_data]
        features = {
            "Pole": pole_anot,
            "Osc. Amplitude": pole_amps,
            #    "Velocity": pole_velocities,
            "Frame": pole_data[:, 0],
            "Angle": np.concatenate(
                (self._df["angle"].values, self._df["angle"].values), axis=0
            ),
            "Embryo center (pixel)": np.concatenate(
                (
                    self._df["Embryo center (pixel)"].values,
                    self._df["Embryo center (pixel)"].values,
                ),
                axis=0,
            ),
        }
        scale = (
            self._layer_select.value.scale
            if self._layer_select.value is not None
            else None
        )
        self._axis_layer = self._viewer.add_shapes(
            line_data, name=REF_AXIS_LAYER, scale=scale, shape_type="line", edge_width=1
        )
        self._point_layer = self._viewer.add_points(
            pole_data,
            name=POLE_LAYER,
            scale=scale,
            border_color=pole_colors,
            face_color=["#ffffff00"] * len(pole_colors),
            features=features,
        )
        self._recalculate(
            self._point_layer, axis_layer=self._axis_layer, refresh_all=False
        )

        @self._point_layer.events.data.connect
        def _on_data(data_event):
            print(f"_on_data: event")

        @self._point_layer.events.set_data.connect
        def _on_set_data(data_event):
            l = data_event.source
            print(f"_on_set_data: {l.name}")
            # recalculate(layer=l, points=l.selected_data)

        @self._point_layer.mouse_drag_callbacks.append
        def _on_points_moved(layer, event):
            # on click
            points = layer.selected_data
            print(len(points), "currently selected points", event.source)
            yield

            # on move
            while event.type == "mouse_move":
                yield

            # on release
            sel_points = layer.selected_data
            refresh_all = self._viewer.dims.current_step[0] == self._ref_frame.value
            print(
                "mouse released",
                len(sel_points),
                f"currently selected points, points={sel_points}, refresh_all={refresh_all}",
            )
            self._recalculate(
                self._point_layer, axis_layer=self._axis_layer, refresh_all=refresh_all
            )

    def _recalculate(self, point_layer, axis_layer=None, refresh_all=False):
        n = len(self._df)
        ref_p1, ref_p2 = self._get_ref_points(
            self._point_layer, self._ref_frame.value, len(self._df)
        )
        pole_data = point_layer.data[:, -1:0:-1]
        pole_amps = oscillation(
            ref_p1[-1:0:-1],
            ref_p2[-1:0:-1],
            pole_data,
            pixel_res=self._layer_select.value.scale[-1],
        )
        # closest_points = [
        #    closest_point(
        #        x=p[2],
        #        y=p[1],
        #        z=p[0],
        #        line_start=ref_p1[-1:0:-1],
        #        line_end=ref_p2[-1:0:-1],
        #        # pixel_res=self._layer_select.value.scale[-1],
        #    )
        #    for p in point_layer.data
        # ]
        sel_data = self._point_layer.selected_data.copy()
        if refresh_all:
            point_layer.selected_data = list(range(len(point_layer.data)))
        point_layer.features["Osc. Amplitude"] = pole_amps
        # point_layer.features["Closest Point"] = closest_points
        # closest_points = np.array(closest_points)
        if axis_layer is not None:
            # print(
            #    f"point_layer.data.shape={point_layer.data.shape}, closest_points.shape={closest_points.shape}"
            # )
            # print(
            #    f"point_layer.data[0]={point_layer.data[0]}, closest_points[0]={closest_points[0]}"
            # )
            # vectors = np.stack((closest_points, point_layer.data), axis=1)
            # print(vectors.shape)
            # print(vectors)
            reflines = create_line_shapes(ref_p1[2], ref_p1[1], ref_p2[2], ref_p2[1], n)
            axis_layer.data = reflines
            # axis_layer.add(vectors, shape_type=["line"] * len(vectors))
        if refresh_all:
            point_layer.data = point_layer.data.copy()
            point_layer.selected_data = sel_data
        c_step = self._viewer.dims.current_step
        n_step = list(c_step)
        if c_step[0] != 0:
            n_step[0] = 0
        else:
            n_step[0] = len(self._df) - 1
        self._viewer.dims.current_step = n_step
        self._viewer.dims.current_step = c_step

        pole1_amps = pole_amps[0 : len(self._df)]
        pole2_amps = pole_amps[len(self._df) : -1]
        self._summary_df["Pole 1"] = [
            zero_crossings(pole1_amps),
            fft_freq(pole1_amps),
            autocorr_freq(pole1_amps),
            dominant_freq(pole1_amps),
        ]
        self._summary_df["Pole 2"] = [
            zero_crossings(pole2_amps),
            fft_freq(pole2_amps),
            autocorr_freq(pole2_amps),
            dominant_freq(pole2_amps),
        ]
        self._table_model = TableModel(self._summary_df)
        self._table.setModel(self._table_model)

    def _get_ref_points(self, point_layer, ref_frame, n):
        pole_data = point_layer.data
        p1 = pole_data[ref_frame]
        p2 = pole_data[ref_frame + n]
        return p1, p2

    def _on_ref_frame_changed(self):
        ref_frame = self._ref_frame.value
        print(f"TrackWidget:_on_ref_frame_changed: {ref_frame}")
        self._recalculate(
            self._point_layer, axis_layer=self._axis_layer, refresh_all=True
        )

    def _on_export(self):
        fpath, _ = QFileDialog.getSaveFileName(self, "Export as CSV", self._export_file)
        if fpath:
            self._export_file
            try:
                self._df.to_csv(fpath, index=False)
            except Exception as e:
                QMessageBox.warning(self._viewer.window.qt_viewer, "Error", e)
