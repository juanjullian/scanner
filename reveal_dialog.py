"""Pantalla opcional de revelado para ProRes 4444 XQ."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

import reveal_pipeline

FIXED_W = 2840
FIXED_H = 2200
_PREVIEW_W = 1100


def _is_bayer(filename: str, meta: dict) -> bool:
    return meta.get(filename, {}).get("pixel_format", "bayer") not in ("rgb", "qoi_rgb")


def _is_bw(filename: str, meta: dict) -> bool:
    return "Blanco" in meta.get(filename, {}).get("type", "")


class RevealDialog(QDialog):
    def __init__(self, parent, folder: Path, filenames: list[str], collection_meta: dict, settings: dict, start_name: str | None):
        super().__init__(parent)
        self.setWindowTitle("Ajustes de revelado")
        self.resize(1280, 760)
        self._folder = Path(folder)
        self._filenames = list(filenames)
        self._meta = collection_meta or {}
        self._settings = reveal_pipeline.coerce_reveal(settings)
        self._lin_prev = None
        self._sigma_scale = 1.0
        self._loading = False

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._paint_preview)
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(120)
        self._frame_timer.timeout.connect(self._load_frame)

        root = QHBoxLayout(self)
        self.preview = QLabel("Elige una película Bayer.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setStyleSheet("background: #111; color: #aaa;")
        root.addWidget(self.preview, 1)

        side = QVBoxLayout()
        side.addWidget(QLabel("Película de referencia"))
        self.combo = QComboBox()
        self.combo.addItems(self._filenames)
        if start_name in self._filenames:
            self.combo.setCurrentText(start_name)
        side.addWidget(self.combo)

        self.lbl_frame = QLabel("Cuadro")
        side.addWidget(self.lbl_frame)
        self.sld_frame = QSlider(Qt.Orientation.Horizontal)
        self.sld_frame.setRange(0, 0)
        self.sld_frame.valueChanged.connect(self._on_frame_label)
        self.sld_frame.sliderReleased.connect(self._load_frame)
        side.addWidget(self.sld_frame)

        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet("color: #888; font-size: 9pt;")
        side.addWidget(self.lbl_note)

        self._rows = {}
        self._add_slider(side, "gain_r", "Ganancia rojo", 50, 250, 100, "Multiplica el rojo. El verde queda en 1.")
        self._add_slider(side, "gain_b", "Ganancia azul", 50, 250, 100, "Multiplica el azul. El verde queda en 1.")
        self._add_slider(side, "ev", "Exposición (EV)", -200, 200, 100, "Pasos de diafragma, antes de la curva.")
        self._add_slider(side, "shadow_b", "Sombras", 10, 400, 10, "Un número más alto abre las sombras. El blanco del sensor sigue en el blanco.")
        self._add_slider(side, "shadow_lift", "Levante de negro", 0, 200, 1000, "Sube el negro sin mover el blanco.")
        self._add_slider(side, "midtone", "Detalle de medios tonos", 0, 150, 100, "Contraste de frecuencias medias, sin afilar el grano fino.")
        side.addStretch(1)

        buttons = QHBoxLayout()
        btn_reset = QPushButton("Restablecer")
        btn_reset.clicked.connect(self._reset)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Listo")
        btn_ok.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; padding: 6px 16px;")
        btn_ok.clicked.connect(self.accept)
        buttons.addWidget(btn_reset)
        buttons.addStretch(1)
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_ok)
        side.addLayout(buttons)

        wrap = QWidget()
        wrap.setLayout(side)
        wrap.setFixedWidth(340)
        root.addWidget(wrap)

        self._load_sliders()
        if start_name in self._filenames:
            self.combo.setCurrentText(start_name)
        self.combo.currentTextChanged.connect(self._on_film)
        QTimer.singleShot(0, self._on_film)

    def settings(self) -> dict:
        return self._collect()

    def _add_slider(self, layout, key, title, lo, hi, scale, tip):
        title_lbl = QLabel(title)
        title_lbl.setToolTip(tip)
        value_lbl = QLabel("")
        value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        head = QHBoxLayout()
        head.addWidget(title_lbl)
        head.addWidget(value_lbl)
        layout.addLayout(head)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setToolTip(tip)
        slider.valueChanged.connect(self._schedule)
        layout.addWidget(slider)
        self._rows[key] = (slider, value_lbl, scale)

    def _load_sliders(self):
        for key, (slider, _lbl, scale) in self._rows.items():
            slider.blockSignals(True)
            slider.setValue(int(round(self._settings[key] * scale)))
            slider.blockSignals(False)
        self._refresh_labels()

    def _refresh_labels(self):
        values = self._collect()
        for key, (_slider, lbl, _scale) in self._rows.items():
            number = values[key]
            if key == "ev":
                lbl.setText(f"{number:+.2f}")
            elif key == "shadow_b":
                lbl.setText(f"{number:.0f}")
            else:
                lbl.setText(f"{number:.2f}")

    def _collect(self) -> dict:
        raw = {}
        for key, (slider, _lbl, scale) in self._rows.items():
            raw[key] = slider.value() / float(scale)
        return reveal_pipeline.coerce_reveal(raw)

    def _schedule(self, *_args):
        self._refresh_labels()
        self._timer.start()

    def _reset(self):
        self._settings = reveal_pipeline.default_reveal()
        self._load_sliders()
        self._paint_preview()

    def _on_film(self, *_args):
        name = self.combo.currentText()
        if not name:
            return
        path = self._folder / name
        if not path.is_file() or not _is_bayer(name, self._meta):
            self._lin_prev = None
            self.sld_frame.setRange(0, 0)
            self.lbl_note.setText("Esta película no es Bayer. Elige otra para ajustar el revelado.")
            self.preview.setText("Sin mosaico para revelar.")
            self.preview.setPixmap(QPixmap())
            return
        frame_bytes = int(FIXED_W * FIXED_H * 1.5)
        count = path.stat().st_size // frame_bytes
        if count <= 0:
            self._lin_prev = None
            self.preview.setText("El archivo no tiene cuadros.")
            return
        self.sld_frame.blockSignals(True)
        self.sld_frame.setRange(0, count - 1)
        self.sld_frame.setValue(min(100, count - 1))
        self.sld_frame.blockSignals(False)
        note = "Los ajustes se guardan para todas las películas del ProRes 4444 XQ y del HEVC lossless."
        if _is_bw(name, self._meta):
            note = "Esta película es blanco y negro. " + note
        self.lbl_note.setText(note)
        self._load_frame()

    def _on_frame_label(self, value: int):
        self.lbl_frame.setText(f"Cuadro {value + 1}")
        if not self.sld_frame.isSliderDown():
            self._frame_timer.start()

    def _load_frame(self):
        name = self.combo.currentText()
        path = self._folder / name
        if not path.is_file() or not _is_bayer(name, self._meta):
            return
        frame_bytes = int(FIXED_W * FIXED_H * 1.5)
        index = int(self.sld_frame.value())
        self.lbl_frame.setText(f"Cuadro {index + 1}")
        try:
            with open(path, "rb") as handle:
                handle.seek(index * frame_bytes)
                packed = handle.read(frame_bytes)
            if len(packed) < frame_bytes:
                self.preview.setText("Cuadro incompleto.")
                return
            lin = reveal_pipeline.debayer_linear(packed, FIXED_W, FIXED_H)
            preview_w = min(_PREVIEW_W, FIXED_W)
            self._sigma_scale = preview_w / float(FIXED_W)
            small_h = max(1, int(round(FIXED_H * self._sigma_scale)))
            self._lin_prev = cv2.resize(lin, (preview_w, small_h), interpolation=cv2.INTER_AREA)
        except Exception as exc:
            self._lin_prev = None
            self.preview.setText(f"No se pudo leer el cuadro.\n{exc}")
            return
        self._paint_preview()

    def _paint_preview(self):
        if self._lin_prev is None:
            return
        proc = reveal_pipeline.apply_reveal(self._lin_prev, self._collect(), self._sigma_scale)
        if _is_bw(self.combo.currentText(), self._meta):
            proc = reveal_pipeline.to_bw(proc)
        u8 = np.clip(proc * np.float32(255.0) + np.float32(0.5), 0, 255).astype(np.uint8)
        u8 = np.ascontiguousarray(u8)
        height, width = u8.shape[:2]
        image = QImage(u8.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(image)
        self.preview.setPixmap(pix.scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
