import numpy as np

from mitosisanalyzer_napari._widget import (
    TrackEditorWidget,
)


def random_image(w=1024, h=1024):
    return np.random.random((h, w))


def test_widget_editor_widget():
    viewer = make_napari_viewer()
    layer = viewer.add_image(random_image)
    my_widget = TrackEditorWidget(viewer)
