"""Confirmación de exportación: preview por archivo, nombre de salida y FPS."""

from __future__ import annotations

from pathlib import Path

import bayer_render
import cv2
import numpy as np
import qoi_utils
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QSpinBox,
    QAbstractItemView, QWidget,
)

FIXED_W, FIXED_H = 2840, 2200
DEFAULT_FPS = 18

_EXT_MAP = {
    "prores": ".mov",
    "prores_hq": ".mov",
    "cineform": ".mov",
    "hevc": ".mp4",
    "h264": ".mp4",
    "ffv1": ".mkv",
}


def default_output_name(stem: str, fmt: str) -> str:
    if fmt == "tiff_seq":
        return f"{stem}_TIFF_SEQ"
    if fmt == "dng":
        return f"{stem}_DNG_SEQ"
    if fmt == "tif":
        return stem
    ext = _EXT_MAP.get(fmt, ".mp4")
    return f"{stem}_{fmt}{ext}"


def default_settings_for_file(filename: str, collection_meta: dict, fmt: str = "prores") -> dict:
    info = collection_meta.get(filename, {})
    try:
        fps = int(info.get("fps", DEFAULT_FPS))
    except (TypeError, ValueError):
        fps = DEFAULT_FPS
    stem = Path(filename).stem
    return {"fps": fps, "output_name": default_output_name(stem, fmt)}


def file_pixel_format(filename: str, collection_meta: dict) -> str:
    info = collection_meta.get(filename, {})
    return info.get("pixel_format", "bayer")


def count_frames(path: Path, pixel_fmt: str, w=FIXED_W, h=FIXED_H) -> int:
    fsize = path.stat().st_size
    if pixel_fmt == "qoi_rgb":
        return len(qoi_utils.build_frame_index(str(path)))
    if pixel_fmt in ("rgb", "qoi_rgb"):
        fb = w * h * 3
    else:
        fb = int(w * h * 1.5)
    return max(1, fsize // fb) if fb else 1


def render_bayer_frame_rgb(packed: bytes, w=FIXED_W, h=FIXED_H, downscale=1) -> np.ndarray:
    """Pipeline base compartido preview (HQ debayer + gamma 2.2 LUT)."""
    bgr = bayer_render.render_capture_view(packed, w, h, downscale=downscale, to_bgr=True)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_frame_rgb(
    path: Path, frame_idx: int, pixel_fmt: str,
    w=FIXED_W, h=FIXED_H, downscale=2,
) -> np.ndarray | None:
    try:
        if pixel_fmt == "qoi_rgb":
            index = qoi_utils.build_frame_index(str(path))
            if not index:
                return None
            idx = max(0, min(frame_idx, len(index) - 1))
            offset, fsz = index[idx]
            qoi_data = qoi_utils.read_frame_at(str(path), offset, fsz)
            return qoi_utils.decode_qoi(qoi_data, w, h).copy()

        if pixel_fmt == "rgb":
            fb = w * h * 3
            with open(path, "rb") as f:
                f.seek(frame_idx * fb)
                raw = f.read(fb)
            if len(raw) < fb:
                return None
            return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()

        fb = int(w * h * 1.5)
        with open(path, "rb") as f:
            f.seek(frame_idx * fb)
            packed = f.read(fb)
        if len(packed) < fb:
            return None
        return render_bayer_frame_rgb(packed, w, h, downscale=downscale)
    except Exception as e:
        print(f"WARN read_frame_rgb {path.name} #{frame_idx}: {e}")
        return None


def rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def make_thumbnail(path: Path, pixel_fmt: str) -> QPixmap | None:
    n = count_frames(path, pixel_fmt)
    idx = min(100, max(0, n - 1))
    rgb = read_frame_rgb(path, idx, pixel_fmt, downscale=8)
    if rgb is None:
        return None
    small = cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA)
    return rgb_to_qpixmap(small)


class ExportConfirmDialog(QDialog):
    """Confirmación previa a exportación: miniatura, nombre de salida y FPS por archivo."""

    def __init__(
        self,
        parent,
        filenames: list[str],
        root: str,
        collection: str,
        collection_meta: dict,
        fmt: str,
    ):
        super().__init__(parent)
        self.root = Path(root)
        self.collection = collection
        self.collection_meta = collection_meta or {}
        self.filenames = list(filenames)
        self.fmt = fmt
        self._output_edits: dict[str, QLineEdit] = {}
        self._fps_spins: dict[str, QSpinBox] = {}

        self.setWindowTitle("Confirmar exportación")
        self.setMinimumSize(820, 480)
        self.resize(900, 560)

        layout = QVBoxLayout(self)
        info = QLabel(
            "Revisa el nombre de salida y los FPS de cada archivo antes de exportar. "
            "La imagen se procesa con el ISP de la cámara (Arena), sin ajustes extra de tono."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 9pt;")
        layout.addWidget(info)

        self.table = QTableWidget(len(self.filenames), 4)
        self.table.setHorizontalHeaderLabels(["", "Archivo origen", "Nombre de salida", "FPS"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(3, 72)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setIconSize(QSize(160, 120))

        for row, fn in enumerate(self.filenames):
            self.table.setRowHeight(row, 128)
            path = self.root / self.collection / fn
            pf = file_pixel_format(fn, self.collection_meta)
            thumb = make_thumbnail(path, pf)
            thumb_item = QTableWidgetItem()
            thumb_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if thumb is not None:
                thumb_item.setIcon(QIcon(thumb))
            self.table.setItem(row, 0, thumb_item)

            src_item = QTableWidgetItem(fn)
            src_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, src_item)

            defaults = default_settings_for_file(fn, self.collection_meta, self.fmt)
            out_edit = QLineEdit(defaults["output_name"])
            out_edit.setPlaceholderText("Nombre de salida")
            self._output_edits[fn] = out_edit
            self.table.setCellWidget(row, 2, out_edit)

            fps_spin = QSpinBox()
            fps_spin.setRange(1, 120)
            fps_spin.setValue(int(defaults["fps"]))
            self._fps_spins[fn] = fps_spin
            fps_wrap = QWidget()
            fps_lay = QHBoxLayout(fps_wrap)
            fps_lay.setContentsMargins(4, 0, 4, 0)
            fps_lay.addWidget(fps_spin)
            self.table.setCellWidget(row, 3, fps_wrap)

        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_export = QPushButton("Exportar")
        btn_export.setStyleSheet(
            "background-color: #0078d7; color: white; font-weight: bold; padding: 8px 20px;"
        )
        btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_export)
        layout.addLayout(btn_row)

    def _on_export(self):
        for fn in self.filenames:
            name = self._output_edits[fn].text().strip()
            if not name:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Nombre vacío", f"Indica un nombre de salida para «{fn}».")
                return
        self.accept()

    def get_all_settings(self) -> dict[str, dict]:
        out = {}
        for fn in self.filenames:
            out[fn] = {
                "output_name": self._output_edits[fn].text().strip(),
                "fps": int(self._fps_spins[fn].value()),
            }
        return out
