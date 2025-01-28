try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._widget import (
    TrackEditorWidget,
)

__all__ = ("TrackEditorWidget",)

from napari.utils.notifications import show_info


def show_about_message():
    show_info(
        "MitoAnalyzer provides tools to track spindle pole movements in mitotoc cells over time."
    )
