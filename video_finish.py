"""Abre la herramienta de videos ya revelados, sin la ventana de captura."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from video_finish_window import VideoFinishWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = VideoFinishWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
