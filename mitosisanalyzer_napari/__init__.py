from napari.utils.notifications import show_info


def show_about_message():
    show_info(
        "MitoAnalyzer provides tools to track spindle pole movements in mitotoc cells over time."
    )
