"""Exportación en un solo diálogo: formato, borradores, avance y tiempo estimado."""
from __future__ import annotations

import copy
import re
import tempfile
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from export_grade import (
    TEST_EXPORT_FRAME_COUNT,
    TEST_EXPORT_SKIP_FRAMES,
    ExportGradeDialog,
    count_frames,
    default_grade_settings,
    default_output_name,
    default_settings_for_file,
    file_pixel_format,
    geometry_has_changes,
    make_thumbnail,
    match_video_ext,
)

BAYER_GROUPS = [
    ("RAW", [("dng", "DNG Raw Sequence (16-bit)")]),
    ("Secuencia de imágenes", [("exr", "EXR lineal (half)")]),
    ("Revelado Rec.709", [
        ("prores_4444", "ProRes 422 HQ"),
        ("hevc_nvenc", "HEVC 4:4:4 lossless NVENC"),
    ]),
    ("Video con gamma", [
        ("prores", "ProRes 4444"),
        ("cineform", "GoPro CineForm"),
        ("prores_hq", "ProRes 422 HQ"),
    ]),
    ("Comprimido de entrega", [
        ("hevc", "HEVC 10-bit 4:4:4"),
        ("av1", "AV1 NVENC 10-bit"),
        ("h264", "H.264 (proxy)"),
    ]),
]

ISP_GROUPS = [
    ("Desde el ISP de la cámara", [
        ("tiff_seq", "TIFF 8-bit, fiel al ISP"),
        ("prores", "ProRes 4444"),
        ("prores_hq", "ProRes 422 HQ"),
        ("hevc", "HEVC 10-bit 4:4:4"),
        ("av1", "AV1 NVENC 10-bit"),
        ("h264", "H.264 (proxy)"),
    ]),
]

REVEAL_FORMATS = frozenset({"prores_4444", "hevc_nvenc"})
SHARP_FORMATS = frozenset({"prores", "cineform", "prores_hq", "hevc", "av1", "h264"})
SEQUENCE_FORMATS = frozenset({"dng", "exr", "tiff_seq", "tif"})
SHARP_CHOICES = [
    ("0,0", "DCB puro"),
    ("0.8,1.5", "Suave (S:0.8 / A:1.5)"),
    ("1.3,1.5", "Medio (S:1.3 / A:1.5)"),
    ("2.0,2.5", "Grueso (S:2.0 / A:2.5)"),
]

_SPEED_RE = re.compile(
    r"Frame:\s*(\d+)\s*/\s*(\d+).*?Vel:\s*([0-9]+(?:[.,][0-9]+)?)\s*fps",
    re.IGNORECASE,
)

_HINTS = {
    "dng": "Secuencia DNG de 16 bits, sin revelado. Cada archivo sale en su carpeta.",
    "exr": (
        "Secuencia OpenEXR half en RGB lineal: 1.0 es el clip del sensor. "
        "Mismo debayer que el visor HQ, sin gamma ni gradación."
    ),
    "tiff_seq": "TIFF de 8 bits, fiel al ISP de la cámara. Sin revelado ni gradación.",
    "prores_4444": (
        "ProRes 422 HQ con el revelado Rec.709, 10 bits 4:2:2. "
        "Abre los ajustes en cada archivo antes de exportar."
    ),
    "hevc_nvenc": (
        "HEVC 4:4:4 lossless por la GPU, con el mismo revelado. "
        "Abre los ajustes en cada archivo antes de exportar."
    ),
}


def format_options(fmt: str, source_is_processed: bool) -> dict:
    reveal = fmt in REVEAL_FORMATS and not source_is_processed
    sharp = fmt in SHARP_FORMATS and not source_is_processed
    grade = fmt in SHARP_FORMATS
    test = grade or fmt in {"exr", "prores_4444", "hevc_nvenc"}
    fps = fmt not in SEQUENCE_FORMATS
    return {"reveal": reveal, "sharp": sharp, "grade": grade, "test": test, "fps": fps}


def default_sharp(fmt: str) -> str:
    if format_options(fmt, False)["sharp"]:
        return "2.0,2.5"
    return "0,0"


def parse_export_speed(text: str):
    match = _SPEED_RE.search(text or "")
    if not match:
        return None
    speed = float(match.group(3).replace(",", "."))
    return int(match.group(1)), int(match.group(2)), speed


def estimate_remaining_seconds(speed, current, file_frames, following):
    if speed is None or speed <= 0:
        return None, None
    left = max(0, int(file_frames) - int(current))
    file_eta = left / float(speed)
    total_eta = (left + max(0, int(following))) / float(speed)
    return file_eta, total_eta


def format_duration(seconds) -> str:
    if seconds is None:
        return "—"
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total} s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {secs} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def draft_for_file(filename: str, collection_meta: dict, fmt: str) -> dict:
    base = default_settings_for_file(filename, collection_meta, fmt)
    info = collection_meta.get(filename, {})
    saved = info.get("export_by_format") if isinstance(info.get("export_by_format"), dict) else {}
    entry = saved.get(fmt) if isinstance(saved, dict) else None
    output_name = default_output_name(Path(filename).stem, fmt)
    grade = base["grade"]
    test_export = False
    sharp = default_sharp(fmt)
    if isinstance(entry, dict):
        if entry.get("output_name"):
            output_name = match_video_ext(str(entry["output_name"]), fmt)
        if isinstance(entry.get("grade"), dict) and entry["grade"]:
            grade = copy.deepcopy(entry["grade"])
        if "test_export" in entry:
            test_export = bool(entry["test_export"])
        if entry.get("sharp"):
            sharp = str(entry["sharp"])
    if fmt == "prores_4444" and Path(output_name).name == f"{Path(filename).stem}_prores_4444.mov":
        output_name = default_output_name(Path(filename).stem, fmt)
    return {
        "output_name": output_name,
        "grade": grade,
        "test_export": test_export,
        "sharp": sharp,
        "fps": int(base["fps"]),
    }


def draft_to_meta(draft: dict) -> dict:
    grade = draft.get("grade")
    if not isinstance(grade, dict):
        grade = default_grade_settings()
    return {
        "output_name": str(draft.get("output_name") or ""),
        "grade": copy.deepcopy(grade),
        "test_export": bool(draft.get("test_export")),
        "sharp": str(draft.get("sharp") or "0,0"),
    }


def _check_image_url() -> str:
    """Palomita blanca para el indicador. El de Windows, dentro de la tabla, sale rojo."""
    path = Path(tempfile.gettempdir()) / "scanner_export_check.png"
    if not path.exists():
        image = QImage(22, 22, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        pen = QPen(QColor("white"))
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(5, 11, 9, 15)
        painter.drawLine(9, 15, 17, 6)
        painter.end()
        image.save(str(path))
    return path.as_posix()


def _dialog_stylesheet() -> str:
    check = _check_image_url()
    return f"""
        QDialog {{
            background: #161616;
            color: #e8e8e8;
        }}
        QLabel {{
            color: #e8e8e8;
            font-size: 10pt;
            background: transparent;
        }}
        QFrame#options {{
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 6px;
        }}
        QLineEdit, QSpinBox, QComboBox {{
            background: #2a2a2a;
            color: #f2f2f2;
            border: 1px solid #4a4a4a;
            border-radius: 4px;
            padding: 4px 8px;
            min-height: 22px;
        }}
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
            border: 1px solid #0078d7;
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            background: #3a3a3a;
            border: none;
            width: 16px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 22px;
        }}
        QComboBox QAbstractItemView {{
            background: #2a2a2a;
            color: #f2f2f2;
            selection-background-color: #0078d7;
            selection-color: #ffffff;
            outline: none;
        }}
        QCheckBox {{
            spacing: 8px;
            color: #e8e8e8;
            font-size: 10pt;
            background: transparent;
        }}
        QCheckBox::indicator, QTableWidget::indicator, QTableView::indicator {{
            width: 22px;
            height: 22px;
            border: 2px solid #9a9a9a;
            border-radius: 4px;
            background: #1a1a1a;
        }}
        QCheckBox::indicator:hover, QTableWidget::indicator:hover, QTableView::indicator:hover {{
            border-color: #6eb5ff;
        }}
        QCheckBox::indicator:checked, QTableWidget::indicator:checked, QTableView::indicator:checked {{
            background: #0078d7;
            border-color: #0078d7;
            image: url("{check}");
        }}
        QTableWidget {{
            background: #141414;
            color: #f0f0f0;
            gridline-color: #2e2e2e;
            border: 1px solid #333;
            alternate-background-color: #1b1b1b;
            selection-background-color: #1d3a52;
            selection-color: #f0f0f0;
        }}
        QHeaderView::section {{
            background: #222;
            color: #ddd;
            border: none;
            border-bottom: 1px solid #333;
            border-right: 1px solid #2a2a2a;
            padding: 8px 6px;
            font-weight: 600;
        }}
        QProgressBar {{
            border: 1px solid #333;
            border-radius: 3px;
            background: #2a2a2a;
            color: #f0f0f0;
            text-align: center;
            font-size: 8pt;
        }}
        QProgressBar::chunk {{
            background: #0078d7;
            border-radius: 2px;
        }}
        QPushButton {{
            background: #2a2a2a;
            color: #f0f0f0;
            border: 1px solid #4a4a4a;
            border-radius: 4px;
            padding: 6px 14px;
        }}
        QPushButton:hover {{
            background: #333;
            border-color: #0078d7;
        }}
        QPushButton:disabled {{
            color: #777;
            background: #222;
        }}
    """


def _fill_format_combo(combo: QComboBox, groups) -> None:
    combo.clear()
    for title, items in groups:
        combo.addItem(title)
        header = combo.model().item(combo.count() - 1)
        header.setEnabled(False)
        header.setForeground(QBrush(QColor("#8ab4f8")))
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        for fmt_id, label in items:
            combo.addItem("    " + label, fmt_id)


def _grade_status_text(grade: dict) -> str:
    if grade.get("enabled"):
        exposure = float(grade.get("exposure", 1.0))
        extra = f", exp={exposure:.2f}×" if abs(exposure - 1.0) > 1e-3 else ""
        return f"Activa (γ={grade['gamma']:.2f}{extra})"
    if geometry_has_changes(grade):
        return "Encuadre guardado"
    return "Sin ajustes"


class ExportSessionDialog(QDialog):
    """Tabla de exportación: formato, opciones, borradores y avance en el mismo lugar."""

    def __init__(self, parent, filenames, root, collection, collection_meta, manager):
        super().__init__(parent)
        self.root = Path(root)
        self.collection = collection
        self.collection_meta = collection_meta or {}
        self.manager = manager
        self.filenames = list(filenames)
        self.source_is_processed = bool(self.filenames) and all(
            file_pixel_format(fn, self.collection_meta) in ("rgb", "qoi_rgb")
            for fn in self.filenames
        )
        self._drafts: dict[str, dict] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._reveal_seen: set[str] = set()
        self._grade_seen: set[str] = set()
        self._frame_counts: dict[str, int] = {}
        self._applying = False
        self._exporting = False
        self._job_names: list[str] = []
        self._done: set[str] = set()
        self._speed = None
        self._active = None
        self._current = 0
        self._fmt = "tiff_seq" if self.source_is_processed else "dng"

        self.setWindowTitle("Exportar")
        self.setMinimumSize(1020, 640)
        self.resize(1180, 760)
        self.setStyleSheet(_dialog_stylesheet())

        from reveal_pipeline import coerce_reveal
        host = self.parent()
        saved = host._persist.get("reveal") if host is not None and hasattr(host, "_persist") else None
        self._reveal = coerce_reveal(saved)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._flush)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        if self.source_is_processed:
            banner = QLabel("Fuente: RGB procesado por el ISP de la cámara (ya tiene balance, gamma y nitidez).")
            banner.setStyleSheet("background: #1a237e; color: #90caf9; padding: 6px; border-radius: 4px;")
            layout.addWidget(banner)

        options = QFrame()
        options.setObjectName("options")
        options_lay = QVBoxLayout(options)
        options_lay.setContentsMargins(12, 10, 12, 12)
        options_lay.setSpacing(8)
        options_lay.addWidget(QLabel("Formato de salida"))
        self.combo_fmt = QComboBox()
        groups = ISP_GROUPS if self.source_is_processed else BAYER_GROUPS
        _fill_format_combo(self.combo_fmt, groups)
        options_lay.addWidget(self.combo_fmt)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("color: #aaa; font-size: 10pt;")
        options_lay.addWidget(self.lbl_hint)

        self.sharp_row = QWidget()
        sharp_lay = QHBoxLayout(self.sharp_row)
        sharp_lay.setContentsMargins(0, 0, 0, 0)
        sharp_lay.addWidget(QLabel("Perfil de nitidez"))
        self.combo_sharp = QComboBox()
        for value, label in SHARP_CHOICES:
            self.combo_sharp.addItem(label, value)
        sharp_lay.addWidget(self.combo_sharp, 1)
        options_lay.addWidget(self.sharp_row)

        self.reveal_row = QWidget()
        reveal_lay = QVBoxLayout(self.reveal_row)
        reveal_lay.setContentsMargins(0, 0, 0, 0)
        self.lbl_reveal = QLabel("")
        self.lbl_reveal.setWordWrap(True)
        reveal_lay.addWidget(self.lbl_reveal)
        reveal_note = QLabel(
            "Cada archivo guarda su propio revelado. "
            "El color, la perforación y el recorte también."
        )
        reveal_note.setWordWrap(True)
        reveal_note.setStyleSheet("color: #999; font-size: 10pt;")
        reveal_lay.addWidget(reveal_note)
        options_lay.addWidget(self.reveal_row)
        self._refresh_reveal_label()

        self._default_output_dir = self.root / self.collection
        self._output_dir = self._default_output_dir
        dir_row = QHBoxLayout()
        self.lbl_output_dir = QLabel(f"Carpeta de salida: {self._output_dir}")
        self.lbl_output_dir.setWordWrap(True)
        self.btn_dir = QPushButton("Elegir carpeta")
        self.btn_dir.clicked.connect(self._pick_output_dir)
        self.btn_dir_reset = QPushButton("Junto al archivo")
        self.btn_dir_reset.clicked.connect(self._reset_output_dir)
        dir_row.addWidget(self.lbl_output_dir, 1)
        dir_row.addWidget(self.btn_dir)
        dir_row.addWidget(self.btn_dir_reset)
        options_lay.addLayout(dir_row)
        layout.addWidget(options)

        pick_row = QHBoxLayout()
        self.btn_all = QPushButton("Marcar todos")
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none = QPushButton("Desmarcar todos")
        self.btn_none.clicked.connect(lambda: self._set_all(False))
        self.btn_analyze = QPushButton("Analizar estabilización")
        self.btn_analyze.setToolTip(
            "Corre el análisis solo en los archivos marcados para exportar. "
            "Avisa si alguno no tiene el recuadro."
        )
        self.btn_analyze.clicked.connect(self._analyze_selected)
        self.btn_fps_all = QPushButton("FPS a todos")
        self.btn_fps_all.setToolTip("Copia el FPS al resto de los archivos.")
        self.btn_fps_all.clicked.connect(self._fps_to_all)
        self._fps_anchor = None
        pick_row.addWidget(self.btn_all)
        pick_row.addWidget(self.btn_none)
        pick_row.addWidget(self.btn_analyze)
        pick_row.addWidget(self.btn_fps_all)
        pick_row.addStretch()
        layout.addLayout(pick_row)

        headers = ["", "Archivo", "Nombre de salida", "FPS", "Seteo", "Prueba", "Tiempo"]
        self.table = QTableWidget(len(self.filenames), len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 176)
        self.table.setColumnWidth(3, 72)
        self.table.setColumnWidth(4, 230)
        self.table.setColumnWidth(5, 88)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnWidth(6, 110)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)

        self._edits: dict[str, QLineEdit] = {}
        self._fps: dict[str, QSpinBox] = {}
        self._tests: dict[str, QCheckBox] = {}
        self._grade_labels: dict[str, QLabel] = {}
        self._badge_lay: dict[str, QVBoxLayout] = {}
        self._badge_host: dict[str, QWidget] = {}
        self._grade_btns: dict[str, QPushButton] = {}
        self._reveal_btns: dict[str, QPushButton] = {}
        self._bars: dict[str, QProgressBar] = {}
        self._thumbs: dict[str, QLabel] = {}
        self._times: dict[str, QLabel] = {}
        self._rows: dict[str, int] = {}

        for row, fn in enumerate(self.filenames):
            self._rows[fn] = row
            self.table.setRowHeight(row, 168)
            path = self.root / self.collection / fn
            pixel_fmt = file_pixel_format(fn, self.collection_meta)
            try:
                self._frame_counts[fn] = count_frames(path, pixel_fmt)
            except Exception:
                self._frame_counts[fn] = 1
            self.table.setCellWidget(row, 0, self._build_thumb(path, pixel_fmt, fn))

            src = QTableWidgetItem(fn)
            src.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            src.setForeground(QBrush(QColor("#f0f0f0")))
            src.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 1, src)

            edit = QLineEdit()
            edit.setPlaceholderText("Nombre de salida")
            edit.textChanged.connect(lambda text, name=fn: self._on_name(name, text))
            self._edits[fn] = edit
            self.table.setCellWidget(row, 2, edit)

            spin = QSpinBox()
            spin.setRange(1, 120)
            spin.setValue(int(draft_for_file(fn, self.collection_meta, self._fmt)["fps"]))
            spin.valueChanged.connect(lambda _value, name=fn: self._on_fps(name))
            self._fps[fn] = spin
            self.table.setCellWidget(row, 3, self._wrap(spin))

            self.table.setCellWidget(row, 4, self._build_settings(fn))

            chk = QCheckBox()
            chk.setFixedSize(36, 36)
            chk.setToolTip(
                f"Exportar {TEST_EXPORT_FRAME_COUNT} cuadros "
                f"(desde el #{TEST_EXPORT_SKIP_FRAMES + 1})"
            )
            chk.toggled.connect(lambda _state, name=fn: self._on_test(name))
            self._tests[fn] = chk
            self.table.setCellWidget(row, 5, self._wrap(chk, center=True))

            time_lbl = QLabel(self._frames_text(fn))
            time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            time_lbl.setStyleSheet("color: #e8e8e8; font-size: 10pt; background: transparent;")
            self._times[fn] = time_lbl
            self.table.setCellWidget(row, 6, time_lbl)
            for col in range(self.table.columnCount()):
                cell = self.table.cellWidget(row, col)
                if cell is not None:
                    self._bind_row_menu(cell, fn)

        layout.addWidget(self.table, 1)

        copy_row = QHBoxLayout()
        self.chk_copy_grade = QCheckBox("Copiar gradación del archivo ajustado a todos")
        self.chk_copy_test = QCheckBox(
            f"Marcar prueba ({TEST_EXPORT_FRAME_COUNT} cuadros) en todos"
        )
        copy_row.addWidget(self.chk_copy_grade)
        copy_row.addWidget(self.chk_copy_test)
        copy_row.addStretch()
        layout.addLayout(copy_row)

        self.lbl_status = QLabel("Tiempo total: —")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #f0f0f0; font-size: 11pt; font-weight: 600;")
        layout.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_close = QPushButton("Cerrar")
        self.btn_close.clicked.connect(self.reject)
        self.btn_export = QPushButton("Exportar")
        self.btn_export.setStyleSheet(
            "background-color: #0078d7; color: white; font-weight: bold; padding: 8px 20px;"
        )
        self.btn_export.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self.btn_close)
        btn_row.addWidget(self.btn_export)
        layout.addLayout(btn_row)

        self.combo_sharp.currentIndexChanged.connect(self._on_sharp)
        preferred = None
        if host is not None and hasattr(host, "_persist"):
            preferred = host._persist.get("export_format")
            if preferred == "prores_xq":
                preferred = "prores_4444"
        self._applying = True
        self.combo_fmt.setCurrentIndex(self._index_for(preferred))
        self._fmt = self.combo_fmt.currentData() or self._fmt
        self._apply_format()
        self._applying = False
        self.combo_fmt.currentIndexChanged.connect(self._on_format)
        jobs = getattr(parent, "adjust_jobs", None)
        if jobs is not None:
            jobs.progress.connect(self._on_adjust_progress)
            jobs.settled.connect(self._on_adjust_settled)

    def _wrap(self, widget, center=False) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(4, 0, 4, 0)
        if center:
            lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(widget)
        return box

    def _build_thumb(self, path: Path, pixel_fmt: str, filename: str) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        image = QLabel()
        image.setFixedSize(160, 120)
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setStyleSheet("background: #111;")
        try:
            thumb = make_thumbnail(path, pixel_fmt)
        except Exception as exc:
            print(f"WARN thumbnail {path.name}: {exc}")
            thumb = None
        if thumb is not None:
            image.setPixmap(thumb)
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFixedHeight(14)
        bar.setFormat("%p%")
        self._bars[filename] = bar
        self._thumbs[filename] = image
        lay.addWidget(image)
        lay.addWidget(bar)
        return box

    def _build_settings(self, filename: str) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(2, 4, 2, 4)
        lay.setSpacing(2)
        grade_lbl = QLabel("Sin ajustes")
        grade_lbl.setStyleSheet("color: #bbb; font-size: 9pt; background: transparent;")
        grade_lbl.setWordWrap(True)
        badge_host = QWidget()
        badge_host.setFixedHeight(22)
        badge_lay = QHBoxLayout(badge_host)
        badge_lay.setContentsMargins(0, 0, 0, 0)
        badge_lay.setSpacing(4)
        self._badge_lay[filename] = badge_lay
        self._badge_host[filename] = badge_host
        btn_grade = QPushButton("Ajustar gradación…")
        btn_grade.clicked.connect(lambda _checked=False, name=filename: self._open_grade(name))
        btn_reveal = QPushButton("Ajustes…")
        btn_reveal.clicked.connect(lambda _checked=False, name=filename: self._open_reveal(name))
        self._grade_labels[filename] = grade_lbl
        self._grade_btns[filename] = btn_grade
        self._reveal_btns[filename] = btn_reveal
        lay.addWidget(grade_lbl)
        lay.addWidget(badge_host)
        lay.addWidget(btn_grade)
        lay.addWidget(btn_reveal)
        return box

    def _index_for(self, preferred) -> int:
        fallback = 0
        found = False
        for index in range(self.combo_fmt.count()):
            data = self.combo_fmt.itemData(index)
            if not data:
                continue
            if not found:
                fallback = index
                found = True
            if data == preferred:
                return index
        return fallback

    def _hint_for(self, fmt: str) -> str:
        if fmt in _HINTS:
            return _HINTS[fmt]
        opts = format_options(fmt, self.source_is_processed)
        if self.source_is_processed and opts["grade"]:
            return "El ISP de la cámara ya reveló. La gradación es opcional."
        if opts["sharp"]:
            return "Video con gamma. El perfil de nitidez y la gradación son opcionales."
        return "Revisa el nombre de salida antes de exportar."

    def _draft(self, fmt: str, filename: str) -> dict:
        bucket = self._drafts.setdefault(fmt, {})
        if filename not in bucket:
            bucket[filename] = draft_for_file(filename, self.collection_meta, fmt)
        return bucket[filename]

    def _mark(self, filename: str):
        if self._applying:
            return
        self._dirty.add((self._fmt, filename))
        self._save_timer.start()

    def _on_name(self, filename: str, text: str):
        if self._applying:
            return
        self._draft(self._fmt, filename)["output_name"] = text
        self._mark(filename)

    def _on_fps(self, filename: str):
        if self._applying:
            return
        self._fps_anchor = filename
        self._draft(self._fmt, filename)["fps"] = int(self._fps[filename].value())
        self._mark(filename)

    def _fps_to_all(self):
        if not self.filenames:
            return
        anchor = self._fps_anchor if self._fps_anchor in self._fps else self.filenames[0]
        value = int(self._fps[anchor].value())
        for filename in self.filenames:
            spin = self._fps[filename]
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
            self._draft(self._fmt, filename)["fps"] = value
            self._mark(filename)

    def _on_test(self, filename: str):
        if self._applying:
            return
        self._draft(self._fmt, filename)["test_export"] = self._tests[filename].isChecked()
        self._mark(filename)
        self._times[filename].setText(self._frames_text(filename))

    def _on_sharp(self):
        if self._applying:
            return
        value = self.combo_sharp.currentData() or "0,0"
        for filename in self.filenames:
            self._draft(self._fmt, filename)["sharp"] = value
            self._mark(filename)

    def _set_sharp_combo(self, value: str):
        index = self.combo_sharp.findData(value)
        if index < 0:
            index = self.combo_sharp.findData("2.0,2.5")
        self.combo_sharp.setCurrentIndex(max(0, index))

    def _apply_format(self):
        fmt = self._fmt
        opts = format_options(fmt, self.source_is_processed)
        self.lbl_hint.setText(self._hint_for(fmt))
        self.sharp_row.setVisible(opts["sharp"])
        self.reveal_row.setVisible(opts["reveal"])
        self.table.setColumnHidden(3, not opts["fps"])
        self.btn_fps_all.setVisible(opts["fps"])
        self.table.setColumnHidden(4, not (opts["grade"] or opts["reveal"]))
        self.table.setColumnHidden(5, not opts["test"])
        self.chk_copy_grade.setVisible(opts["grade"])
        self.chk_copy_test.setVisible(opts["test"])
        self.btn_analyze.setVisible(opts["reveal"])
        if opts["reveal"]:
            self.table.horizontalHeaderItem(4).setText("Ajustes")
        elif opts["grade"]:
            self.table.horizontalHeaderItem(4).setText("Gradación")
        was = self._applying
        self._applying = True
        try:
            sharp = default_sharp(fmt)
            if self.filenames:
                sharp = self._draft(fmt, self.filenames[0]).get("sharp") or sharp
            self._set_sharp_combo(sharp)
            for filename in self.filenames:
                draft = self._draft(fmt, filename)
                self._edits[filename].setText(draft["output_name"])
                self._tests[filename].setChecked(bool(draft.get("test_export")))
                if opts["reveal"]:
                    self._grade_labels[filename].setVisible(False)
                    self._badge_host[filename].setVisible(True)
                    self._paint_file_badges(filename)
                else:
                    self._badge_host[filename].setVisible(False)
                    self._grade_labels[filename].setText(_grade_status_text(draft["grade"]))
                    self._grade_labels[filename].setVisible(opts["grade"])
                self._grade_btns[filename].setVisible(opts["grade"])
                self._reveal_btns[filename].setVisible(opts["reveal"])
                self._times[filename].setText(self._frames_text(filename))
        finally:
            self._applying = was

    def _on_format(self):
        fmt = self.combo_fmt.currentData()
        if not fmt or self._applying or self._exporting:
            return
        self._flush()
        self._fmt = fmt
        self._apply_format()
        host = self.parent()
        if host is not None and hasattr(host, "_persist"):
            host._persist["export_format"] = fmt
            if hasattr(host, "save_config"):
                host.save_config()

    def _flush(self):
        self._save_timer.stop()
        if self.manager is None or not self._dirty:
            self._dirty.clear()
            return
        pending = list(self._dirty)
        self._dirty.clear()
        for fmt, filename in pending:
            draft = self._drafts.get(fmt, {}).get(filename)
            if not draft:
                continue
            info = self.manager.get_file_info(self.collection, filename)
            by_format = dict(info.get("export_by_format") or {})
            by_format[fmt] = draft_to_meta(draft)
            fps = int(self._fps[filename].value())
            self.manager.set_file_info(
                self.collection, filename, export_by_format=by_format, fps=fps,
            )
            meta = self.collection_meta.setdefault(filename, {})
            meta["export_by_format"] = by_format
            meta["fps"] = fps

    def _file_mode(self, filename: str) -> str:
        info = self.collection_meta.get(filename, {})
        return "BW" if "Blanco" in info.get("type", "Color") else "COLOR"

    def _open_grade(self, filename: str):
        path = self.root / self.collection / filename
        pixel_fmt = file_pixel_format(filename, self.collection_meta)
        current = self._draft(self._fmt, filename)["grade"]
        dialog = ExportGradeDialog(self, path, pixel_fmt, self._file_mode(filename), current)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_grade = dialog.get_settings()
        self._store_grade(filename, new_grade)
        self._grade_seen.add(filename)
        if self.chk_copy_grade.isChecked():
            for other in self.filenames:
                if other != filename:
                    self._store_grade(other, copy.deepcopy(new_grade))

    def _store_grade(self, filename: str, grade: dict):
        draft = self._draft(self._fmt, filename)
        draft["grade"] = copy.deepcopy(grade)
        self._grade_labels[filename].setText(_grade_status_text(grade))
        self._mark(filename)

    def _bind_row_menu(self, widget, filename: str):
        targets = [widget, *widget.findChildren(QWidget)]
        for target in targets:
            target.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            target.customContextMenuRequested.connect(
                lambda pos, name=filename, host=target: self._show_adjust_menu(name, host.mapToGlobal(pos))
            )

    def _table_menu(self, pos):
        row = self.table.rowAt(pos.y())
        item = self.table.item(row, 1) if row >= 0 else None
        if item is None:
            return
        self._show_adjust_menu(item.text(), self.table.viewport().mapToGlobal(pos))

    def _show_adjust_menu(self, filename: str, global_pos):
        others = [name for name in self.filenames if name != filename]
        menu = QMenu(self)
        action = menu.addAction("Copiar estos ajustes a los demás clips")
        action.setEnabled(bool(others))
        chosen = menu.exec(global_pos)
        if chosen is action:
            self._copy_adjust_to_others(filename, others)

    def _copy_adjust_to_others(self, filename: str, others: list[str]):
        from film_finish import copy_adjust, has_adjust, load
        src = self.root / self.collection / filename
        if not has_adjust(load(src)):
            QMessageBox.information(
                self, "Sin ajustes",
                "Este clip no tiene revelado, color, recorte ni zona de perforación para copiar.",
            )
            return
        answer = QMessageBox.question(
            self, "Copiar ajustes",
            f"Se copian el revelado, el color, el recorte y la zona de perforación de {filename} "
            f"a los otros {len(others)} clips. El desplazamiento ya medido de cada uno no se toca.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        copied = 0
        for name in others:
            if copy_adjust(src, self.root / self.collection / name):
                copied += 1
        self._refresh_finish_labels()
        self.lbl_status.setText(f"Ajustes copiados a {copied} clips.")

    def _open_reveal(self, filename: str):
        from reveal_dialog import RevealDialog
        dialog = RevealDialog(
            self, self.root / self.collection, self.filenames,
            self.collection_meta, self._reveal, filename,
        )
        accepted = bool(dialog.exec())
        self._refresh_finish_labels()
        if not accepted:
            return
        reference = dialog.reference_name()
        if reference:
            self._reveal_seen.add(reference)
        self._refresh_reveal_label()

    def _refresh_reveal_label(self):
        from reveal_pipeline import coerce_reveal
        saved = coerce_reveal(self._reveal)
        self.lbl_reveal.setText(
            "Cada archivo guarda su revelado. Los que todavía no tienen usan el general: "
            f"rojo {saved['gain_r']:.2f}, azul {saved['gain_b']:.2f}, "
            f"EV {saved['ev']:+.2f}, gamma {saved['gamma']:.2f}."
        )

    def _refresh_finish_labels(self):
        if not format_options(self._fmt, self.source_is_processed)["reveal"]:
            return
        for filename in self.filenames:
            self._paint_file_badges(filename)

    def _paint_file_badges(self, filename: str, percent=None):
        layout = self._badge_lay.get(filename)
        if layout is None:
            return
        import film_finish
        from reveal_dialog import _paint_badges
        state = film_finish.review_state(self.root / self.collection / filename)
        if percent is not None and state.get("stab") != "listo":
            state["stab"] = "corriendo"
            state["percent"] = int(percent)
        _paint_badges(layout, film_finish.review_badges(state))

    def _on_adjust_progress(self, name: str, done: int, total: int, _state: str):
        if name not in self._badge_lay or not total:
            return
        self._paint_file_badges(name, int(round(100 * done / total)))

    def _on_adjust_settled(self, name: str, _ok: bool, _message: str):
        if name not in self._badge_lay:
            return
        self._paint_file_badges(name)

    def _set_all(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            if item is not None:
                item.setCheckState(state)

    def checked_filenames(self) -> list[str]:
        chosen = []
        for filename in self.filenames:
            item = self.table.item(self._rows[filename], 1)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                chosen.append(filename)
        return chosen

    def _frames_text(self, filename: str) -> str:
        return f"{self._export_frames(filename)} cuadros"

    def _export_frames(self, filename: str) -> int:
        opts = format_options(self._fmt, self.source_is_processed)
        if opts["test"] and self._tests[filename].isChecked():
            return TEST_EXPORT_FRAME_COUNT
        return max(1, int(self._frame_counts.get(filename, 1)))

    def _set_output_dir(self, folder: Path):
        self._output_dir = folder
        self.lbl_output_dir.setText(f"Carpeta de salida: {folder}")

    def _pick_output_dir(self):
        chosen = QFileDialog.getExistingDirectory(self, "Carpeta de salida", str(self._output_dir))
        if chosen:
            self._set_output_dir(Path(chosen))

    def _reset_output_dir(self):
        self._set_output_dir(self._default_output_dir)

    def _perfo_gaps(self, selected: list[str]):
        import film_finish
        missing = []
        pending = []
        ready = []
        for name in selected:
            raw = self.root / self.collection / name
            roi = (film_finish.load(raw).get("perfo") or {}).get("roi")
            if not roi:
                missing.append(name)
                continue
            if film_finish.perfo_is_done(raw, roi):
                ready.append(name)
            else:
                pending.append((name, [int(v) for v in roi]))
        return pending, missing, ready

    def _queue_perfo(self, pending) -> str:
        host = self.parent()
        jobs = getattr(host, "adjust_jobs", None) if host is not None else None
        if jobs is None:
            QMessageBox.warning(self, "Sin cola", "No pude encolar el análisis.")
            return "No pude encolar el análisis."
        import film_finish
        queued = 0
        busy = 0
        for name, roi in pending:
            raw = self.root / self.collection / name
            data = film_finish.load(raw)
            perfo = data.get("perfo") or {}
            ref = film_finish.perfo_ref(perfo)
            if ref is None:
                side = film_finish.read_perfo(raw) or {}
                if side.get("status") == "done":
                    ref = film_finish.perfo_ref(side)
            if ref is None:
                ref = 0
            data["perfo"] = {
                "enabled": True,
                "roi": list(roi),
                "ref": film_finish.perfo_ref(perfo),
                "horizontal": film_finish.perfo_horizontal(perfo),
            }
            film_finish.save(raw, data)
            if jobs.enqueue(raw, roi, ref_frame=ref):
                queued += 1
                self._paint_file_badges(name, 0)
            else:
                busy += 1
        if queued and busy:
            return f"Encolé {queued} análisis. {busy} ya estaban en curso."
        if queued:
            return f"Encolé {queued} análisis de estabilización."
        if busy:
            return "El análisis ya está en curso."
        return "No encolé ningún análisis."

    @staticmethod
    def _name_block(names: list[str]) -> str:
        shown = list(names[:8])
        text = "\n".join(shown)
        extra = len(names) - len(shown)
        if extra > 0:
            text += f"\n…y {extra} más"
        return text

    def _analyze_selected(self):
        selected = self.checked_filenames()
        if not selected:
            QMessageBox.warning(self, "Sin archivos", "Marca al menos un archivo.")
            return
        pending, missing, ready = self._perfo_gaps(selected)
        if missing:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Sin recuadro")
            box.setText("Estos archivos no tienen el recuadro de estabilización:\n\n" + self._name_block(missing))
            if not pending:
                box.setInformativeText("Márcalo en Ajustes. Sin recuadro no hay análisis.")
                box.exec()
                return
            box.setInformativeText("Esos se saltan. Puedo analizar los que sí tienen recuadro.")
            go = box.addButton("Analizar el resto", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is not go:
                return
        if not pending:
            QMessageBox.information(
                self, "Estabilización",
                "Los archivos marcados ya tienen el análisis listo." if ready else "No hay nada que analizar.",
            )
            return
        self.lbl_status.setText(self._queue_perfo(pending))

    def _confirm_stabilization(self, selected: list[str]) -> bool:
        if not format_options(self._fmt, self.source_is_processed)["reveal"]:
            return True
        pending, missing, _ready = self._perfo_gaps(selected)
        if not pending and not missing:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Estabilización")
        lines = []
        if pending:
            lines.append("Falta el análisis de estabilización en:\n\n" + self._name_block([name for name, _roi in pending]))
        if missing:
            lines.append("Sin recuadro, no se puede analizar:\n\n" + self._name_block(missing))
        box.setText("\n\n".join(lines))
        volver = box.addButton("Volver", QMessageBox.ButtonRole.RejectRole)
        go = box.addButton("Exportar igual", QMessageBox.ButtonRole.AcceptRole)
        analyze = None
        if pending:
            box.setInformativeText("Puedo correr el análisis antes, o exportar con lo que hay.")
            analyze = box.addButton("Analizar primero", QMessageBox.ButtonRole.ActionRole)
            box.setDefaultButton(analyze)
        else:
            box.setInformativeText("Puedes volver a marcar el recuadro, o exportar igual.")
            box.setDefaultButton(volver)
        box.exec()
        chosen = box.clickedButton()
        if analyze is not None and chosen is analyze:
            note = self._queue_perfo(pending)
            self.lbl_status.setText(note + " Cuando terminen, vuelve a exportar.")
            return False
        return chosen is go

    def _confirm_seen(self, selected: list[str]) -> bool:
        opts = format_options(self._fmt, self.source_is_processed)
        if opts["reveal"]:
            missing = [name for name in selected if name not in self._reveal_seen]
            what = "los ajustes"
        elif opts["grade"]:
            missing = [name for name in selected if name not in self._grade_seen]
            what = "la gradación"
        else:
            return True
        if not missing:
            return True
        listed = "\n".join(missing)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Seteo sin abrir")
        box.setText(f"No abriste {what} de estos archivos:\n\n{listed}")
        box.setInformativeText("Puedes volver a la tabla o exportar igual.")
        box.addButton("Volver", QMessageBox.ButtonRole.RejectRole)
        go = box.addButton("Exportar igual", QMessageBox.ButtonRole.AcceptRole)
        box.exec()
        return box.clickedButton() is go

    def _build_jobs(self, selected: list[str]):
        opts = format_options(self._fmt, self.source_is_processed)
        sharp = self.combo_sharp.currentData() if opts["sharp"] else "0,0"
        jobs = []
        for filename in selected:
            draft = self._draft(self._fmt, filename)
            settings = {
                "output_name": self._edits[filename].text().strip(),
                "output_dir": str(self._output_dir),
                "fps": int(self._fps[filename].value()),
                "grade": copy.deepcopy(draft["grade"]),
            }
            if opts["test"]:
                settings["test_export"] = self._tests[filename].isChecked()
            path = self.root / self.collection / filename
            jobs.append((path, self._fmt, sharp or "0,0", settings))
        return jobs

    def _set_locked(self, locked: bool):
        self._exporting = locked
        self.combo_fmt.setEnabled(not locked)
        self.combo_sharp.setEnabled(not locked)
        self.btn_export.setEnabled(not locked)
        self.btn_dir.setEnabled(not locked)
        self.btn_dir_reset.setEnabled(not locked)
        self.btn_all.setEnabled(not locked)
        self.btn_none.setEnabled(not locked)
        self.btn_analyze.setEnabled(not locked)
        self.chk_copy_grade.setEnabled(not locked)
        self.chk_copy_test.setEnabled(not locked)
        for filename in self.filenames:
            self._edits[filename].setEnabled(not locked)
            self._fps[filename].setEnabled(not locked)
            self._tests[filename].setEnabled(not locked)
            self._grade_btns[filename].setEnabled(not locked)
            self._reveal_btns[filename].setEnabled(not locked)
        self.btn_close.setText("Cancelar exportación" if locked else "Cerrar")

    def _on_export_clicked(self):
        if self._exporting:
            return
        if self.chk_copy_test.isVisible() and self.chk_copy_test.isChecked() and self.filenames:
            state = self._tests[self.filenames[0]].isChecked()
            for filename in self.filenames:
                self._tests[filename].setChecked(state)
        selected = self.checked_filenames()
        if not selected:
            QMessageBox.warning(self, "Sin archivos", "Marca al menos un archivo.")
            return
        for filename in selected:
            if not self._edits[filename].text().strip():
                QMessageBox.warning(
                    self, "Nombre vacío", f"Indica un nombre de salida para «{filename}».",
                )
                return
        if not self._confirm_stabilization(selected):
            return
        if not self._confirm_seen(selected):
            return
        self._flush()
        jobs = self._build_jobs(selected)
        host = self.parent()
        if host is None or not hasattr(host, "begin_export_session"):
            return
        self._job_names = list(selected)
        self._done.clear()
        self._speed = None
        self._active = None
        self._current = 0
        for filename in self.filenames:
            self._bars[filename].setValue(0)
            self._bars[filename].setFormat("%p%")
            self._times[filename].setText(self._frames_text(filename))
        self.lbl_status.setText("Tiempo total: —")
        self._set_locked(True)
        if not host.begin_export_session(self, jobs):
            self._set_locked(False)

    def note_file_started(self, filename: str):
        self._active = filename
        self._current = 0
        bar = self._bars.get(filename)
        if bar is not None:
            bar.setValue(0)
            bar.setFormat("%p%")
        self.lbl_status.setText(f"Exportando {filename}…  Tiempo total: —")

    def note_progress(self, filename: str, percent: int):
        if not self._exporting:
            return
        bar = self._bars.get(filename)
        if bar is not None:
            bar.setValue(max(0, min(100, int(percent))))

    def note_thumb(self, filename: str, data: bytes):
        if not self._exporting:
            return
        label = self._thumbs.get(filename)
        if label is None or not data:
            return
        image = QImage.fromData(data, "JPG")
        if image.isNull():
            return
        label.setPixmap(QPixmap.fromImage(image).scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def note_info(self, filename: str, text: str):
        if not self._exporting:
            return
        parsed = parse_export_speed(text)
        if parsed is None:
            return
        current, _total, speed = parsed
        self._active = filename
        self._current = current
        self._speed = speed
        self._paint_etas()

    def _paint_etas(self):
        if not self._speed or not self._active:
            return
        following = 0
        passed = False
        for filename in self._job_names:
            if filename == self._active:
                passed = True
                continue
            if passed and filename not in self._done:
                following += self._export_frames(filename)
        file_eta, total_eta = estimate_remaining_seconds(
            self._speed, self._current, self._export_frames(self._active), following,
        )
        speed_txt = f"{self._speed:.1f}".replace(".", ",")
        self.lbl_status.setText(
            f"{self._active}: cuadro {self._current} · {speed_txt} fps · "
            f"este archivo {format_duration(file_eta)} · total {format_duration(total_eta)}"
        )
        full_share = None
        if self._speed > 0:
            full_share = {
                filename: self._export_frames(filename) / self._speed
                for filename in self._job_names
                if filename != self._active and filename not in self._done
            }
        if self._active not in self._done:
            self._times[self._active].setText(format_duration(file_eta))
        for filename, seconds in (full_share or {}).items():
            self._times[filename].setText(format_duration(seconds))

    def note_file_finished(self, filename: str, ok: bool, message: str):
        self._done.add(filename)
        bar = self._bars.get(filename)
        label = self._times.get(filename)
        if ok:
            if bar is not None:
                bar.setValue(100)
            if label is not None:
                label.setText("Listo")
        else:
            if bar is not None:
                bar.setFormat("Error")
                bar.setValue(0)
            if label is not None:
                label.setText("Error")
                label.setToolTip(message or "")
        self._active = None
        self._current = 0

    def note_queue_finished(self):
        self._set_locked(False)
        errors = [
            filename for filename in self._job_names
            if self._times.get(filename) is not None and self._times[filename].text() == "Error"
        ]
        if errors:
            self.lbl_status.setText("La cola terminó con errores.")
            QMessageBox.warning(
                self, "Cola terminada",
                "Estos archivos no se exportaron:\n\n" + "\n".join(errors),
            )
        else:
            self.lbl_status.setText("Cola terminada.")
            QMessageBox.information(self, "Fin", "Cola terminada.")

    def note_cancelled(self):
        self._set_locked(False)
        self.lbl_status.setText("Exportación cancelada.")

    def reject(self):
        if self._exporting:
            host = self.parent()
            if host is not None and hasattr(host, "on_batch_cancel"):
                host.on_batch_cancel()
            return
        self._flush()
        super().reject()

    def closeEvent(self, event):
        if self._exporting:
            event.ignore()
            return
        self._flush()
        super().closeEvent(event)
