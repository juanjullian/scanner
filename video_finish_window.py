"""Ventana para color, estabilización y recorte de videos ya revelados."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSlider, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import film_finish
import gammac
from export_session import estimate_remaining_seconds, format_duration
from reveal_dialog import FrameView, TurnControl, _GENTLE, _NEUTRAL, _paint_badges
from timeline_bar import TimelineBar, _timeline_button
from video_export import export_clip, output_path
from video_session import (
    empty_session, is_video, load_session, match_folder, meaningful_adjust,
    missing_files, save_session, session_path, same_parent,
)
from video_source import (
    Cancelled, GrayCropStream, PreviewReader, VideoError, VideoInfo, probe,
    read_gray_crop, read_rgb,
)

_PREVIEW_W = 1100
_TASKS = ("Estabilización", "Color", "Recorte")


@dataclass
class Clip:
    path: str
    fps: float = 24.0
    fps_text: str = "24/1"
    fps_edited: bool = False
    width: int = 0
    height: int = 0
    frames: int = 0
    linked: bool = False
    adjust: dict = field(default_factory=film_finish.empty)

    def info(self) -> VideoInfo | None:
        if not self.linked or self.width < 2 or self.frames < 1:
            return None
        return VideoInfo(Path(self.path), self.width, self.height, self.fps, self.fps_text, self.frames)


class ClipList(QListWidget):
    files_dropped = pyqtSignal(list)
    remove_pressed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.remove_pressed.emit()
            return
        super().keyPressEvent(event)


class _AnalyzeThread(QThread):
    progress = pyqtSignal(str, int, int)
    finished_one = pyqtSignal(str, bool, str, int)

    def __init__(self, jobs: list[tuple]):
        super().__init__()
        self.jobs = jobs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from perfo_lock import PerfoError, _clip_rect, _search_rect, analyze_ordered

        for path, rect, ref, width, height, frames, fps in self.jobs:
            if self._cancel:
                break
            target = Path(path)
            try:
                ident = _clip_rect(rect, width, height)
                search = _search_rect(ident, width, height)
                film_finish.write_perfo_status(
                    target, "running", f"0 / {frames}", roi=list(rect), ref=int(ref),
                )
                last = 0.0

                def on_progress(done, total, _target=target, _rect=rect, _ref=ref):
                    nonlocal last
                    self.progress.emit(str(_target), int(done), int(total))
                    now = time.time()
                    if now - last > 2 or done == total:
                        film_finish.write_perfo_status(
                            _target, "running", f"{done} / {total}", roi=list(_rect), ref=int(_ref),
                        )
                        last = now

                def sample(index, _target=target, _search=search, _fps=fps):
                    return read_gray_crop(_target, _search, int(index), float(_fps))

                stream = GrayCropStream(target, search, lambda: self._cancel)
                result = analyze_ordered(
                    sample, stream, int(frames), rect, int(width), int(height), int(ref), on_progress,
                )
                if stream.code not in (None, 0):
                    raise VideoError(stream.error or f"ffmpeg terminó con código {stream.code}.")
                measured = len(result.get("tx") or [])
                note = "Lista"
                if measured != int(frames):
                    note = f"Lista. Medí {measured} cuadros; el archivo anunciaba {frames}."
                film_finish.write_perfo_status(target, "done", note, result)
                self.finished_one.emit(str(target), True, note, measured)
            except Cancelled:
                film_finish.write_perfo_status(target, "error", "Cancelado.")
                self.finished_one.emit(str(target), False, "Cancelado.", 0)
                break
            except (PerfoError, VideoError) as exc:
                film_finish.write_perfo_status(target, "error", str(exc))
                self.finished_one.emit(str(target), False, str(exc), 0)
            except Exception as exc:
                film_finish.write_perfo_status(target, "error", str(exc))
                self.finished_one.emit(str(target), False, str(exc), 0)


class _ExportThread(QThread):
    frame = pyqtSignal(str, int, int)
    file_done = pyqtSignal(str, bool, str)

    def __init__(self, jobs: list[tuple[str, str, str, str]]):
        super().__init__()
        self.jobs = jobs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        for source, dst, codec, fps_text in self.jobs:
            if self._cancel:
                self.file_done.emit(source, False, "Cancelado.")
                break
            try:
                export_clip(
                    Path(source), Path(dst), codec, fps_text, lambda: self._cancel,
                    lambda done, total, _source=source: self.frame.emit(_source, int(done), int(total)),
                )
                self.file_done.emit(source, True, "")
            except Cancelled:
                self.file_done.emit(source, False, "Cancelado.")
                break
            except Exception as exc:
                self.file_done.emit(source, False, str(exc))


class VideoFinishWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tratar videos")
        self.resize(1480, 900)
        self._session_file = session_path()
        self._session = load_session(self._session_file)
        self._clips: dict[str, Clip] = {}
        self._current: str | None = None
        self._adjust = film_finish.empty()
        self._filling = False
        self._task = 0
        self._segment = -1
        self._roi = None
        self._roi_ref = None
        self._crop = None
        self._shifts = None
        self._pad = 0
        self._preview = None
        self._preview_wh = (1, 1)
        self._sigma = 1.0
        self._full_w = 1
        self._full_h = 1
        self._count = 1
        self._playing = False
        self._play_recipe = None
        self._draft_id = None
        self._csliders = {}
        self._reader = PreviewReader()
        self._analyze: _AnalyzeThread | None = None
        self._analyze_queue: list[tuple] = []
        self._export: _ExportThread | None = None
        self._export_rows: dict[str, int] = {}
        self._export_started = 0.0
        self._export_file_started = 0.0
        self._export_following: dict[str, int] = {}
        self._export_engine = "ProRes NV"
        self._export_after_analyze: list | None = None
        self._bars: dict[str, QProgressBar] = {}

        host = QWidget()
        self.setCentralWidget(host)
        root = QVBoxLayout(host)
        body = QHBoxLayout()
        root.addLayout(body, 1)

        left = QVBoxLayout()
        hint = QLabel("Suelta videos .mov, .mp4 o .mkv.")
        hint.setWordWrap(True)
        left.addWidget(hint)
        self.list = ClipList()
        self.list.files_dropped.connect(self._add_paths)
        self.list.remove_pressed.connect(self._remove_selected)
        self.list.currentRowChanged.connect(self._on_row)
        self.list.model().rowsMoved.connect(lambda *_args: self._save_session())
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        btn_remove = QPushButton("Quitar")
        btn_remove.clicked.connect(self._remove_selected)
        btn_up = QPushButton("Subir")
        btn_up.clicked.connect(lambda: self._move(-1))
        btn_down = QPushButton("Bajar")
        btn_down.clicked.connect(lambda: self._move(1))
        row.addWidget(btn_remove)
        row.addWidget(btn_up)
        row.addWidget(btn_down)
        left.addLayout(row)
        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(280)
        body.addWidget(left_wrap)

        self.view = FrameView()
        self.view.set_empty("Suelta un video en la lista.")
        self.view._play.setToolTip(
            "Reproduce el video. Si la perforación está medida, "
            "a la izquierda queda el original y a la derecha el estabilizado."
        )
        self.view.rect_ready.connect(self._on_rect)
        self.view.play_clicked.connect(self._toggle_play)
        preview_col = QWidget()
        preview_lay = QVBoxLayout(preview_col)
        preview_lay.setContentsMargins(0, 0, 0, 0)
        preview_lay.addWidget(self.view, 1)
        self._timeline_host = QWidget()
        time_lay = QVBoxLayout(self._timeline_host)
        time_lay.setContentsMargins(0, 2, 0, 0)
        time_lay.setSpacing(2)
        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(4)
        tools.addStretch(1)
        self.bar = TimelineBar()
        self.bar.head_moved.connect(self._on_timeline_head)
        self.bar.picked.connect(self._on_timeline_pick)
        self.bar.edited.connect(self._on_timeline_edit)
        self.bar.setToolTip(
            "La línea blanca es el cuadro. Arrastra sobre un tramo para partirlo "
            "y arrastra el borde para correrlo cuadro a cuadro. "
            "Ctrl y la rueda acercan. Shift y las flechas mueven el borde de a un cuadro."
        )
        self.btn_zoom_out = _timeline_button("−", "Alejar. También Ctrl y la rueda.", lambda: self.bar.zoom_by(1 / 1.4))
        self.btn_zoom_in = _timeline_button("+", "Acercar. También Ctrl y la rueda.", lambda: self.bar.zoom_by(1.4))
        self.btn_zoom_fit = _timeline_button("Todo", "Ver el rollo entero.", self.bar.fit_film)
        self.btn_split = _timeline_button("Partir", "Corta el tramo donde está la línea. Las dos partes salen iguales.", self._split_at_head)
        self.btn_merge = _timeline_button("Unir", "Funde el tramo con el anterior. El anterior conserva su ajuste.", self._merge_selected)
        for button in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_fit, self.btn_split, self.btn_merge):
            tools.addWidget(button)
        time_lay.addLayout(tools)
        time_lay.addWidget(self.bar, 1)
        preview_lay.addWidget(self._timeline_host)
        self._crop_i = 0

        side = QVBoxLayout()
        self.lbl_name = QLabel("Ningún video")
        self.lbl_name.setWordWrap(True)
        side.addWidget(self.lbl_name)
        self.lbl_frame = QLabel("Cuadro")
        side.addWidget(self.lbl_frame)
        self.sld_frame = QSlider(Qt.Orientation.Horizontal)
        self.sld_frame.setRange(0, 0)
        self.sld_frame.valueChanged.connect(self._on_frame_label)
        self.sld_frame.sliderPressed.connect(self._stop_play)
        self.sld_frame.sliderReleased.connect(lambda: self._load_frame())
        side.addWidget(self.sld_frame)
        self._badge_host = QWidget()
        self._badge_host.setFixedHeight(22)
        self._badge_lay = QHBoxLayout(self._badge_host)
        self._badge_lay.setContentsMargins(0, 0, 0, 0)
        self._badge_lay.setSpacing(4)
        side.addWidget(self._badge_host)

        tasks = QHBoxLayout()
        self._task_btns = []
        for index, title in enumerate(_TASKS):
            button = QPushButton(title)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, number=index: self._set_task(number))
            self._task_btns.append(button)
            tasks.addWidget(button)
        side.addLayout(tasks)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_stab())
        self.stack.addWidget(self._build_color())
        self.stack.addWidget(self._build_crop())
        side.addWidget(self.stack, 1)
        side_wrap = QWidget()
        side_wrap.setLayout(side)
        side_wrap.setMinimumWidth(320)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(6)
        split.setChildrenCollapsible(False)
        split.addWidget(preview_col)
        split.addWidget(side_wrap)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([900, 340])
        body.addWidget(split, 1)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Salida"))
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItem("ProRes 422 HQ", "prores_4444")
        self.combo_fmt.addItem("HEVC 4:4:4 lossless", "hevc_nvenc")
        self.combo_fmt.setToolTip(
            "ProRes 422 HQ se codifica en la tarjeta con ProRes NV, "
            "igual que la exportación del raw. HEVC lossless usa NVENC."
        )
        bar.addWidget(self.combo_fmt, 1)
        bar.addWidget(QLabel("FPS"))
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setDecimals(3)
        self.spin_fps.setSingleStep(1)
        self.spin_fps.valueChanged.connect(self._on_fps)
        bar.addWidget(self.spin_fps)
        self.btn_export = QPushButton("Exportar")
        self.btn_export.clicked.connect(self._export_clicked)
        bar.addWidget(self.btn_export)
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self._cancel_work)
        self.btn_cancel.setVisible(False)
        bar.addWidget(self.btn_cancel)
        root.addLayout(bar)
        self.lbl_status = QLabel("Los ajustes se guardan al lado de cada video.")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Archivo", "Avance", "Tiempo"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(220)
        self.table.setVisible(False)
        root.addWidget(self.table)

        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(80)
        self._frame_timer.timeout.connect(lambda: self._load_frame())
        self._measure_timer = QTimer(self)
        self._measure_timer.setSingleShot(True)
        self._measure_timer.setInterval(200)
        self._measure_timer.timeout.connect(self._commit_slider_measure)
        self._session_timer = QTimer(self)
        self._session_timer.setSingleShot(True)
        self._session_timer.setInterval(400)
        self._session_timer.timeout.connect(self._write_session)

        self.setStyleSheet(_style())
        self._set_task(0)
        self._load_saved_list()
        QTimer.singleShot(0, self._offer_relink)

    def closeEvent(self, event):
        self._stop_play()
        self._save_current()
        self._analyze_queue.clear()
        self._write_session()
        if self._analyze is not None and self._analyze.isRunning():
            self._analyze.cancel()
            self._analyze.wait(1500)
        if self._export is not None and self._export.isRunning():
            self._export.cancel()
            self._export.wait(1500)
        self._reader.close()
        super().closeEvent(event)

    def _build_stab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Dibuja la zona alrededor de una perforación, sin meter el cuadro. "
            "El cuadro en el que dibujas queda como referencia de toda la estabilización. "
            "Al reproducir, la izquierda es el original y la derecha el estabilizado."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.chk_perfo = QCheckBox("Aplicar estabilización al exportar")
        self.chk_perfo.toggled.connect(lambda _on: self._save_current())
        layout.addWidget(self.chk_perfo)
        self.chk_horizontal = QCheckBox("Eje horizontal")
        self.chk_horizontal.setChecked(True)
        self.chk_horizontal.setToolTip(
            "Activado corrige de lado y de arriba abajo. "
            "Desactivado deja solo el movimiento vertical. No vuelve a analizar."
        )
        self.chk_horizontal.toggled.connect(self._on_horizontal)
        layout.addWidget(self.chk_horizontal)
        self.lbl_perfo = QLabel("Sin zona.")
        self.lbl_perfo.setWordWrap(True)
        layout.addWidget(self.lbl_perfo)
        row = QHBoxLayout()
        btn = QPushButton("Analizar este archivo")
        btn.clicked.connect(self._analyze_one)
        btn_all = QPushButton("Encolar la lista")
        btn_all.clicked.connect(self._analyze_all)
        row.addWidget(btn)
        row.addWidget(btn_all)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _build_color(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "El área tiene que dejar afuera el borde y las perforaciones. "
            "Abajo, el rollo parte en un solo tramo. Partir lo corta en la línea "
            "y el borde se corre cuadro a cuadro."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.combo_grade = QComboBox()
        self.combo_grade.addItem("Color", "color")
        self.combo_grade.addItem("Blanco y negro", "bw")
        self.combo_grade.currentIndexChanged.connect(self._on_grade_mode)
        layout.addWidget(self.combo_grade)
        sat_head = QHBoxLayout()
        sat_title = QLabel("Saturación")
        self.lbl_sat = QLabel("100")
        self.lbl_sat.setAlignment(Qt.AlignmentFlag.AlignRight)
        sat_head.addWidget(sat_title)
        sat_head.addWidget(self.lbl_sat)
        layout.addLayout(sat_head)
        self.sld_sat = QSlider(Qt.Orientation.Horizontal)
        self.sld_sat.setRange(0, 100)
        self.sld_sat.setValue(100)
        self.sld_sat.valueChanged.connect(self._on_sat)
        layout.addWidget(self.sld_sat)
        btn_measure = QPushButton("Medir")
        btn_measure.clicked.connect(self._measure_segment)
        layout.addWidget(btn_measure)
        self.lbl_color = QLabel("Dibuja el área. El tramo de abajo parte cubriendo todo el rollo.")
        self.lbl_color.setWordWrap(True)
        form = QVBoxLayout()
        form.addWidget(QLabel("Anclaje"))
        self.combo_lock = QComboBox()
        self.combo_lock.addItem("Verde", 1)
        self.combo_lock.addItem("Rojo", 0)
        self.combo_lock.addItem("Azul", 2)
        self.combo_lock.addItem("Promedio", -2)
        self.combo_lock.addItem("Mediana", -3)
        self.combo_lock.currentIndexChanged.connect(self._touch_segment)
        form.addWidget(self.combo_lock)
        form.addWidget(QLabel("Escala"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItem("Por canal", 2)
        self.combo_scale.addItem("Rango común", 1)
        self.combo_scale.addItem("Sin estirar", 0)
        self.combo_scale.currentIndexChanged.connect(self._touch_segment)
        form.addWidget(self.combo_scale)
        self._add_color_slider(form, "lo_th", "Umbral bajo", 0, 200, 100, False, "loTh, en porcentaje.")
        self._add_color_slider(form, "hi_th", "Umbral alto", 0, 200, 100, False, "hiTh, en porcentaje.")
        self._add_color_slider(form, "gam_max", "Tope de gamma", 10, 100, 10, False, "GamMax. 2,5 es suave.")
        self._add_color_slider(form, "strength", "Fuerza", 0, 100, 100, True, "0,50 deja la mitad del cuadro original.")
        self._add_color_slider(form, "rng_lim", "Rango mínimo", 1, 40, 1, False, "RngLim.")
        self._add_color_slider(form, "omin", "Negro de salida", 0, 80, 1, True, "omin.")
        self._add_color_slider(form, "omax", "Blanco de salida", 175, 255, 1, True, "omax.")
        self.chk_dark = QCheckBox("Proteger sombras")
        self.chk_dark.toggled.connect(self._on_apply_knob)
        form.addWidget(self.chk_dark)
        self._add_color_slider(form, "dark", "Nivel de sombras", 0, 120, 1, True, "DarkProtect.")
        self.chk_high = QCheckBox("Proteger luces")
        self.chk_high.toggled.connect(self._on_apply_knob)
        form.addWidget(self.chk_high)
        self._add_color_slider(form, "high", "Nivel de luces", 140, 255, 1, True, "HighProtect.")
        self._add_color_slider(form, "red_mul", "Multiplicador rojo", 25, 400, 100, False, "RedMul.")
        self._add_color_slider(form, "grn_mul", "Multiplicador verde", 25, 400, 100, False, "GrnMul.")
        self._add_color_slider(form, "blu_mul", "Multiplicador azul", 25, 400, 100, False, "BluMul.")
        btn_preset = QPushButton("Dejar este ajuste como predeterminado")
        btn_preset.clicked.connect(self._save_preset)
        form.addWidget(btn_preset)
        form.addStretch(1)
        inner = QWidget()
        inner.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        self._gammac_box = QWidget()
        box = QVBoxLayout(self._gammac_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.lbl_color)
        box.addWidget(scroll, 1)
        layout.addWidget(self._gammac_box, 1)
        self._apply_preset_to_controls(self._preset())
        return page

    def _build_crop(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "A la izquierda, el cuadro con el recorte. A la derecha, lo que se codifica. "
            "El primer tramo fija la proporción y el tamaño del video. "
            "Los siguientes se mueven y se agrandan o achican parejos, sin deformar."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.chk_crop = QCheckBox("Aplicar este recorte al exportar")
        self.chk_crop.toggled.connect(self._on_crop_toggle)
        layout.addWidget(self.chk_crop)
        self.chk_flip = QCheckBox("Voltear horizontal")
        self.chk_flip.setToolTip(
            "El volteo es de este tramo. Se aplica después del recorte, "
            "sobre lo que se exporta, y esta vista lo muestra."
        )
        self.chk_flip.toggled.connect(self._on_flip)
        layout.addWidget(self.chk_flip)
        self.lbl_crop = QLabel("Sin recorte.")
        self.lbl_crop.setWordWrap(True)
        layout.addWidget(self.lbl_crop)
        self.turn = TurnControl()
        self.turn.changed.connect(self._on_turn)
        layout.addWidget(self.turn)
        layout.addStretch(1)
        return page

    def _load_saved_list(self):
        order = []
        for item in self._session.get("files") or []:
            path = str(item.get("path") or "")
            if not path or path in self._clips:
                continue
            clip = _clip_from_item(item)
            if clip.linked:
                try:
                    _apply_probe(clip, probe(Path(path)))
                    clip.adjust = film_finish.load(Path(path))
                except VideoError:
                    clip.linked = False
            self._clips[path] = clip
            order.append(path)
        self._fill_list(order, select=order[0] if order else None)

    def _offer_relink(self):
        missing = [self._clips[item["path"]] for item in missing_files(self._session) if item.get("path") in self._clips]
        if not missing:
            return
        self._relink_dialog(missing)

    def _relink_dialog(self, missing: list[Clip]):
        folder = same_parent([{"path": clip.path} for clip in missing])
        lines = "\n".join(Path(clip.path).name for clip in missing[:8])
        extra = "" if len(missing) <= 8 else f"\n… y {len(missing) - 8} más"
        text = "Estos videos ya no están donde quedaron guardados:\n" + lines + extra
        box = QMessageBox(self)
        box.setWindowTitle("Volver a enlazar")
        box.setText(text)
        pick_file = box.addButton("Buscar archivo…", QMessageBox.ButtonRole.ActionRole)
        pick_folder = box.addButton("Buscar carpeta…", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Dejar así", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is pick_file:
            if len(missing) == 1:
                self._relink_one(missing[0])
                return
            for clip in missing:
                if not Path(clip.path).is_file():
                    self._relink_one(clip)
            return
        if clicked is pick_folder:
            start = str(folder) if folder else str(Path.home())
            chosen = QFileDialog.getExistingDirectory(self, "Carpeta donde están los videos", start)
            if not chosen:
                return
            pairs = match_folder([{"path": clip.path} for clip in missing], Path(chosen))
            self._apply_folder_relink(pairs)

    def _relink_one(self, clip: Clip):
        chosen, _filt = QFileDialog.getOpenFileName(
            self, f"Dónde está {Path(clip.path).name}", str(Path(clip.path).parent),
            "Video (*.mov *.mp4 *.mkv)",
        )
        if not chosen:
            return
        self._apply_relink(clip.path, Path(chosen), None)

    def _apply_folder_relink(self, pairs: dict[str, Path]):
        conflicts = []
        for old, new in pairs.items():
            clip = self._clips.get(old)
            if clip and meaningful_adjust(clip.adjust) and meaningful_adjust(film_finish.load(new)):
                conflicts.append(old)
        choice = None
        if conflicts:
            choice = self._ask_adjust("Algunos archivos ya tienen ajustes al lado. ¿Cuáles dejo en todos?")
        for old, new in pairs.items():
            self._apply_relink(old, new, choice)
        if not pairs:
            self.lbl_status.setText("En esa carpeta no encontré los mismos nombres.")

    def _apply_relink(self, old: str, new: Path, choice: str | None):
        clip = self._clips.get(old)
        if clip is None or not new.is_file() or not is_video(new):
            return
        try:
            info = probe(new)
        except VideoError as exc:
            QMessageBox.warning(self, "Video", str(exc))
            return
        disk = film_finish.load(new)
        session_adjust = clip.adjust if isinstance(clip.adjust, dict) else film_finish.empty()
        if meaningful_adjust(session_adjust) and meaningful_adjust(disk):
            pick = choice or self._ask_adjust(f"«{new.name}» ya tiene ajustes al lado. ¿Cuáles dejo?")
            if pick == "session":
                film_finish.save(new, session_adjust)
                kept = session_adjust
            else:
                kept = disk
        elif meaningful_adjust(session_adjust):
            film_finish.save(new, session_adjust)
            kept = session_adjust
        else:
            kept = disk
        updated = Clip(
            path=str(new),
            fps=clip.fps if clip.fps_edited else info.fps,
            fps_text=f"{clip.fps:.6f}" if clip.fps_edited else info.fps_text,
            fps_edited=clip.fps_edited,
            width=info.width,
            height=info.height,
            frames=info.frames,
            linked=True,
            adjust=json.loads(json.dumps(kept)),
        )
        del self._clips[old]
        self._clips[str(new)] = updated
        if self._current == old:
            self._current = str(new)
        order = [str(new) if item == old else item for item in self._order()]
        self._fill_list(order, select=self._current)
        self._save_session()

    def _ask_adjust(self, text: str) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("Ajustes")
        box.setText(text)
        session_btn = box.addButton("Los de esta sesión", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Los del archivo", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is session_btn:
            return "session"
        return "disk"

    def _add_paths(self, paths: list[str]):
        added = None
        skipped = []
        order = self._order()
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    if is_video(child):
                        added = self._add_one(child, order) or added
                continue
            if not is_video(path):
                skipped.append(path.name)
                continue
            added = self._add_one(path, order) or added
        if skipped:
            self.lbl_status.setText("Solo entran .mov, .mp4 y .mkv.")
        if added:
            self._fill_list(order, select=added)
            self._save_session()

    def _add_one(self, path: Path, order: list[str]) -> str | None:
        key = str(path)
        if key in self._clips and self._clips[key].linked:
            return key
        try:
            info = probe(path)
        except VideoError as exc:
            self.lbl_status.setText(str(exc))
            return None
        clip = Clip(path=key, linked=True, adjust=film_finish.load(path))
        _apply_probe(clip, info)
        self._clips[key] = clip
        if key not in order:
            order.append(key)
        return key

    def _remove_selected(self):
        rows = sorted({index.row() for index in self.list.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._save_current()
        order = self._order()
        for row in rows:
            if 0 <= row < len(order):
                gone = order.pop(row)
                self._clips.pop(gone, None)
                if self._current == gone:
                    self._current = None
        select = order[min(rows[-1], len(order) - 1)] if order else None
        self._fill_list(order, select=select)
        self._save_session()

    def _move(self, delta: int):
        row = self.list.currentRow()
        order = self._order()
        new = row + delta
        if row < 0 or not (0 <= new < len(order)):
            return
        order[row], order[new] = order[new], order[row]
        self._fill_list(order, select=order[new])
        self._save_session()

    def _order(self) -> list[str]:
        paths = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item is not None:
                paths.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return paths

    def _fill_list(self, order: list[str], select: str | None):
        self._filling = True
        self.list.clear()
        chosen = -1
        for path in order:
            clip = self._clips.get(path)
            if clip is None:
                continue
            item = QListWidgetItem(_clip_label(clip))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.list.addItem(item)
            if path == select:
                chosen = self.list.count() - 1
        self._filling = False
        if chosen >= 0:
            self.list.setCurrentRow(chosen)
        elif self.list.count():
            self.list.setCurrentRow(0)
        else:
            self._current = None
            self._show_empty()

    def _on_row(self, row: int):
        if self._filling:
            return
        path = None
        item = self.list.item(row) if row >= 0 else None
        if item is not None:
            path = str(item.data(Qt.ItemDataRole.UserRole))
        if path == self._current:
            return
        self._save_current()
        self._stop_play()
        self._current = path
        self._load_current()

    def _show_empty(self):
        self.lbl_name.setText("Ningún video")
        self.view.clear()
        self.stack.setEnabled(False)
        self.sld_frame.setEnabled(False)
        _paint_badges(self._badge_lay, [])

    def _load_current(self):
        clip = self._clips.get(self._current or "")
        self._reader.close()
        self._preview = None
        self._play_recipe = None
        self._draft_id = None
        if clip is None or not clip.linked:
            self._show_empty()
            if clip is not None:
                self.lbl_name.setText(f"Sin enlace · {Path(clip.path).name}")
                self.lbl_status.setText("Este video no está. Vuelve a enlazarlo para verlo.")
            return
        self.stack.setEnabled(True)
        self.sld_frame.setEnabled(True)
        self._full_w = clip.width
        self._full_h = clip.height
        self._count = max(1, clip.frames)
        self.lbl_name.setText(Path(clip.path).name)
        self.spin_fps.blockSignals(True)
        self.spin_fps.setValue(float(clip.fps))
        self.spin_fps.blockSignals(False)
        self._adjust = film_finish.load(Path(clip.path))
        clip.adjust = json.loads(json.dumps(self._adjust))
        self._load_adjust()
        self.sld_frame.blockSignals(True)
        self.sld_frame.setRange(0, self._count - 1)
        self.sld_frame.setValue(min(100, self._count - 1))
        self.sld_frame.blockSignals(False)
        self._load_frame()

    def _load_adjust(self):
        perfo = self._adjust.get("perfo") or {}
        self._roi = tuple(perfo["roi"]) if perfo.get("roi") else None
        self._roi_ref = film_finish.perfo_ref(perfo)
        self.chk_perfo.blockSignals(True)
        self.chk_perfo.setChecked(bool(perfo.get("enabled")) and self._roi is not None)
        self.chk_perfo.blockSignals(False)
        self.chk_horizontal.blockSignals(True)
        self.chk_horizontal.setChecked(film_finish.perfo_horizontal(perfo))
        self.chk_horizontal.blockSignals(False)
        self._crop_i = 0
        self._segment = 0
        self._sync_crop_view()
        self.chk_crop.blockSignals(True)
        self.chk_crop.setChecked(bool(self._adjust.get("crop")))
        self.chk_crop.blockSignals(False)
        self._expand_single_frame_segment()
        self._load_shifts()
        self._sync_segment_controls()
        self._sync_grade_controls()
        self._sync_turn()
        self._refresh_bars()
        self._refresh_review()

    def _save_current(self):
        clip = self._clips.get(self._current or "")
        if clip is None or not clip.linked or not isinstance(self._adjust, dict):
            return
        self._adjust["perfo"] = {
            "enabled": self.chk_perfo.isChecked() and self._roi is not None,
            "roi": list(self._roi) if self._roi else None,
            "ref": self._roi_ref,
            "horizontal": self.chk_horizontal.isChecked(),
        }
        self._adjust["segments"] = self._segments_for_save()
        if self._adjust.get("crop_spans"):
            spans = self._crops_for_save()
            self._adjust["crop_spans"] = spans
            master = spans[0].get("crop") if spans else None
            self._adjust["crop"] = list(master) if self.chk_crop.isChecked() and master else None
            self._adjust["flip_h"] = bool(spans[0].get("flip_h")) if spans else False
        elif not self.chk_crop.isChecked():
            self._adjust["crop"] = None
            self._adjust["flip_h"] = self.chk_flip.isChecked()
        self._adjust["turn"] = self.turn.degrees()
        film_finish.save(Path(clip.path), self._adjust)
        clip.adjust = json.loads(json.dumps(self._adjust))
        self._session_timer.start()
        self._refresh_review()
        self._refresh_label(clip.path)

    def _on_horizontal(self, _on):
        self._save_current()
        self._paint_preview()

    def _save_session(self):
        self._session_timer.start()

    def _write_session(self):
        files = []
        for path in self._order():
            clip = self._clips.get(path)
            if clip is not None:
                files.append(_clip_to_item(clip))
        self._session["files"] = files
        self._session["version"] = 1
        save_session(self._session_file, self._session)

    def _refresh_label(self, path: str):
        clip = self._clips.get(path)
        if clip is None:
            return
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText(_clip_label(clip))

    def _on_fps(self, value: float):
        clip = self._clips.get(self._current or "")
        if clip is None or self._filling:
            return
        clip.fps = float(value)
        clip.fps_edited = True
        clip.fps_text = f"{clip.fps:.6f}"
        self._refresh_label(clip.path)
        self._save_session()

    def _set_task(self, index: int):
        self._task = index
        self.stack.setCurrentIndex(index)
        for number, button in enumerate(self._task_btns):
            button.setChecked(number == index)
        if hasattr(self, "_timeline_host"):
            self._timeline_host.setVisible(index in (1, 2))
        self._paint_preview()

    def _on_frame_label(self, value: int):
        self.lbl_frame.setText(f"Cuadro {value + 1}")
        self._follow_span()
        self._refresh_bars()
        if not self.sld_frame.isSliderDown() and not self._playing:
            self._frame_timer.start()

    def _load_frame(self):
        clip = self._clips.get(self._current or "")
        info = clip.info() if clip else None
        if info is None:
            return
        index = int(self.sld_frame.value())
        if not self._playing:
            self.lbl_frame.setText(f"Cuadro {index + 1}")
        try:
            image = self._reader.frame(info, index, _PREVIEW_W)
        except VideoError as exc:
            self.lbl_status.setText(str(exc))
            return
        self._preview = image
        self._preview_wh = (int(image.shape[1]), int(image.shape[0]))
        self._sigma = self._preview_wh[0] / float(self._full_w)
        self._paint_preview()

    def _paint_preview(self, *_args):
        if self._preview is None:
            return
        base = self._preview
        index = int(self.sld_frame.value())
        mode, sat = self._grade_setting()
        recipe = self._recipe_for_preview(base, index) if mode == "color" and self._task == 1 else None
        graded = gammac.apply(base, recipe) if recipe else base
        graded = gammac.saturate(graded, sat)
        plain = gammac.saturate(base, sat)
        playing = self._playing
        captions = None
        if self._task == 0:
            original = _to_u8(plain)
            if self._shifts:
                shown = original
                side = _to_u8(self._warp_preview(plain))
                captions = ("Original", "Estabilizado")
            else:
                shown = original
                side = None
            overlay = None if playing else self._preview_rect(self._roi)
            color, allow = "#ff9f1a", not playing
        elif self._task == 1:
            shown_img = self._warp_preview(graded) if playing else graded
            shown = _to_u8(shown_img)
            overlay = None if playing else self._preview_rect(self._segment_rect())
            color, side, allow = "#ffcc00", None, not playing
        else:
            self._sync_crop_view()
            shown_img = self._warp_preview(graded)
            if playing and self.chk_crop.isChecked():
                rect = self._preview_rect(self._crop)
                if rect and rect[2] > 2 and rect[3] > 2:
                    x, y, width, height = rect
                    shown_img = shown_img[y:y + height, x:x + width]
                if self._flip_on():
                    shown_img = _hflip(shown_img)
                shown = _to_u8(shown_img)
                overlay, side, allow, aspect = None, None, False, None
            else:
                side = self._crop_side(shown_img)
                overlay = self._preview_rect(self._shown_crop())
                if self._flip_on():
                    shown_img = _hflip(shown_img)
                shown = _to_u8(shown_img)
                allow = True
                aspect = self._crop_aspect()
            color = "#4cc9f0"
        if self._task != 2:
            aspect = None
        self.view.set_view(shown, overlay, color, side, allow, captions, aspect=aspect)
        self._refresh_bars()
        self._refresh_crop_label()

    def _toggle_play(self):
        if self._count <= 1 or self._preview is None:
            return
        if self._playing:
            self._stop_play()
            return
        self._playing = True
        self.view.set_playing(True)
        self._show_play_state(int(self.sld_frame.value()))
        self._paint_preview()
        QTimer.singleShot(0, self._advance)

    def _stop_play(self, *_args):
        if not self._playing and self.view._play._playing:
            self.view.set_playing(False)
        if not self._playing:
            return
        self._playing = False
        self.view.set_playing(False)
        self.lbl_status.setText("Pausa.")
        self._load_frame()

    def _show_play_state(self, index: int):
        if self._shifts:
            stab = "con estabilización. A la izquierda el original y a la derecha el tratado."
        else:
            stab = "sin estabilización. No hay desplazamiento medido: vuelve a analizar la perforación."
        self.lbl_frame.setText(f"Reproduciendo · cuadro {index + 1}")
        self.lbl_status.setText(f"Reproduciendo, {stab}")

    def _advance(self):
        if not self._playing:
            return
        started = time.perf_counter()
        nxt = int(self.sld_frame.value()) + 1
        if nxt >= self._count:
            nxt = 0
        self.sld_frame.blockSignals(True)
        self.sld_frame.setValue(nxt)
        self.sld_frame.blockSignals(False)
        self._show_play_state(nxt)
        self._load_frame()
        if not self._playing:
            return
        clip = self._clips.get(self._current or "")
        fps = clip.fps if clip and clip.fps > 0 else 18
        spent = (time.perf_counter() - started) * 1000.0
        wait = max(0, int(round(1000.0 / fps - spent)))
        QTimer.singleShot(wait, self._advance)

    def _on_rect(self, rect):
        full = self._full_rect(rect)
        if full is None:
            return
        if self._task == 0:
            self._roi = full
            self._roi_ref = int(self.sld_frame.value())
            self.chk_perfo.setChecked(True)
            self.lbl_perfo.setText(
                f"Zona {full[2]}×{full[3]} en {full[0]}, {full[1]}."
                f" Referencia: cuadro {self._roi_ref + 1}."
            )
        elif self._task == 2:
            if self._flip_on():
                full = film_finish.mirror_x(full, self._full_w)
            self._apply_crop_rect(full)
        else:
            segment = self._ensure_segment()
            segment["rect"] = list(full)
            segment["dirty"] = True
            segment.pop("recipe", None)
            self._play_recipe = None
            self._draft_id = None
            self.lbl_color.setText("Vista previa del área. Medir la guarda para la exportación.")
        self._save_current()
        self._paint_preview()

    def _on_crop_toggle(self, _on: bool):
        self._save_current()
        self._paint_preview()

    def _analyze_one(self):
        clip = self._clips.get(self._current or "")
        if clip is None or not clip.linked:
            return
        if not self._roi:
            self.lbl_perfo.setText("Dibuja primero la zona de la perforación.")
            return
        if self._roi_ref is None:
            self._roi_ref = int(self.sld_frame.value())
        self.chk_perfo.setChecked(True)
        self._save_current()
        self._enqueue(clip, self._roi, self._saved_ref(clip))

    def _analyze_all(self):
        self._save_current()
        queued = 0
        for path in self._order():
            clip = self._clips.get(path)
            if clip is None or not clip.linked:
                continue
            if path == self._current and self._roi:
                zone = self._roi
            else:
                zone = (film_finish.load(Path(path)).get("perfo") or {}).get("roi")
            if not zone:
                continue
            if film_finish.perfo_is_done(Path(path), zone, clip.frames):
                continue
            if self._enqueue(clip, tuple(zone), self._saved_ref(clip)):
                queued += 1
        if queued == 0:
            self.lbl_perfo.setText("No queda nada por medir. Los que ya están listos se dejan.")

    def _enqueue(self, clip: Clip, zone, ref: int) -> bool:
        if film_finish.perfo_is_done(Path(clip.path), zone, clip.frames):
            if clip.path == self._current:
                self.lbl_perfo.setText("Este archivo ya está medido.")
            return False
        data = film_finish.load(Path(clip.path))
        perfo = data.get("perfo") or {}
        perfo["enabled"] = True
        perfo["roi"] = list(zone)
        known = self._roi_ref if clip.path == self._current else film_finish.perfo_ref(perfo)
        if known is not None:
            perfo["ref"] = int(known)
        data["perfo"] = perfo
        film_finish.save(Path(clip.path), data)
        if clip.path == self._current:
            self._adjust["perfo"] = data["perfo"]
            self._roi_ref = int(ref)
        job = (clip.path, tuple(zone), int(ref), clip.width, clip.height, clip.frames, clip.fps)
        self._analyze_queue.append(job)
        if clip.path == self._current:
            self.lbl_perfo.setText("En la cola.")
        self._kick_analyze()
        return True

    def _kick_analyze(self):
        if self._analyze is not None and self._analyze.isRunning():
            return
        if not self._analyze_queue:
            self._maybe_export_after_analyze()
            if (self._export is None or not self._export.isRunning()) and not self._export_after_analyze:
                self.btn_cancel.setVisible(False)
            return
        jobs = self._analyze_queue
        self._analyze_queue = []
        self._analyze = _AnalyzeThread(jobs)
        self._analyze.progress.connect(self._on_analyze_progress)
        self._analyze.finished_one.connect(self._on_analyze_done)
        self._analyze.finished.connect(self._kick_analyze)
        self._analyze.start()
        self.btn_cancel.setVisible(True)

    def _on_analyze_progress(self, path: str, done: int, total: int):
        if path == self._current and total:
            self.lbl_perfo.setText(f"En curso, {done} de {total}.")
            self._refresh_review(int(round(100 * done / total)))
        if self._export_after_analyze and total and path in self._bars:
            self._bars[path].setValue(int(round(100 * done / total)))
            row = self._export_rows.get(path)
            if row is not None:
                self.table.item(row, 2).setText(f"Midiendo {done}/{total}")
            self.lbl_status.setText(
                f"Midiendo la estabilización de {Path(path).name}, {done} de {total}. Después exporto."
            )

    def _on_analyze_done(self, path: str, ok: bool, message: str, measured: int):
        clip = self._clips.get(path)
        if ok and clip is not None and measured > 0 and measured != clip.frames:
            clip.frames = measured
            self._refresh_label(path)
            if path == self._current:
                self._count = measured
                self.sld_frame.setRange(0, max(0, measured - 1))
        if path == self._current:
            self._load_shifts()
            if not ok:
                self.lbl_perfo.setText(message or "Sin perforación.")
            elif message and message != "Lista":
                self.lbl_status.setText(message)
            self._paint_preview()
            self._refresh_review()
        if self._export_after_analyze and not ok:
            row = self._export_rows.get(path)
            if row is not None:
                self.table.item(row, 2).setText(message or "Sin perforación.")
        if self._analyze is None or not self._analyze.isRunning():
            if not self._analyze_queue and not self._export_after_analyze and (
                self._export is None or not self._export.isRunning()
            ):
                self.btn_cancel.setVisible(False)

    def _load_shifts(self):
        self._shifts = None
        self._pad = 0
        clip = self._clips.get(self._current or "")
        if clip is None or not clip.linked:
            return
        data = film_finish.read_perfo(Path(clip.path)) or {}
        saved = data.get("roi")
        same = self._roi and saved and tuple(int(v) for v in saved) == tuple(int(v) for v in self._roi)
        if data.get("status") == "done" and data.get("tx") and same and len(data["tx"]) == clip.frames:
            self._shifts = (data["tx"], data["ty"])
            self._pad = int(data.get("pad") or 0)
        self._say_perfo(data.get("status") if same or not saved else None, data.get("message", ""))

    def _say_perfo(self, status=None, message=""):
        if self._roi is None:
            self.lbl_perfo.setText("Sin zona.")
            return
        ref = f" Referencia: cuadro {self._roi_ref + 1}." if self._roi_ref is not None else ""
        if status == "done":
            text = f"Lista. Margen negro de {self._pad} px.{ref}"
            if message and message != "Lista":
                text = message + f" Margen negro de {self._pad} px.{ref}"
        elif status == "running":
            text = f"En curso. {message}{ref}"
        elif status == "error":
            text = (message or "Sin perforación.") + ref
        else:
            text = f"Zona guardada. Falta analizar.{ref}"
        self.lbl_perfo.setText(text)

    def _refresh_review(self, percent=None):
        clip = self._clips.get(self._current or "")
        if clip is None or not clip.linked:
            _paint_badges(self._badge_lay, [])
            return
        state = film_finish.review_state(Path(clip.path), clip.frames)
        if percent is not None and state.get("stab") != "listo":
            state["stab"] = "corriendo"
            state["percent"] = int(percent)
        _paint_badges(self._badge_lay, film_finish.review_badges(state))

    def _grade_setting(self) -> tuple[str, float]:
        grade = self._adjust.get("grade") if isinstance(self._adjust, dict) else None
        if isinstance(grade, dict) and grade.get("mode") in ("color", "bw"):
            mode = grade["mode"]
            try:
                sat = float(grade.get("sat", 0.0 if mode == "bw" else 1.0))
            except (TypeError, ValueError):
                sat = 0.0 if mode == "bw" else 1.0
            return mode, min(1.0, max(0.0, sat))
        return "color", 1.0

    def _sync_grade_controls(self):
        had = isinstance(self._adjust.get("grade"), dict) and self._adjust["grade"].get("mode") in ("color", "bw")
        mode, sat = self._grade_setting()
        self.combo_grade.blockSignals(True)
        self.sld_sat.blockSignals(True)
        self.combo_grade.setCurrentIndex(1 if mode == "bw" else 0)
        self.sld_sat.setValue(int(round(sat * 100)))
        self.combo_grade.blockSignals(False)
        self.sld_sat.blockSignals(False)
        self.lbl_sat.setText(str(self.sld_sat.value()))
        self._gammac_box.setVisible(mode == "color")
        if not had and self._current:
            self._adjust["grade"] = {"mode": mode, "sat": sat}
            self._save_current()

    def _store_grade(self):
        mode = self.combo_grade.currentData() or "color"
        self._adjust["grade"] = {"mode": mode, "sat": self.sld_sat.value() / 100.0}
        self.lbl_sat.setText(str(self.sld_sat.value()))
        self._gammac_box.setVisible(mode == "color")
        self._save_current()
        self._paint_preview()

    def _on_grade_mode(self, *_args):
        mode = self.combo_grade.currentData() or "color"
        self.sld_sat.blockSignals(True)
        self.sld_sat.setValue(0 if mode == "bw" else 100)
        self.sld_sat.blockSignals(False)
        self._store_grade()

    def _on_sat(self, *_args):
        self._store_grade()

    def _preset(self) -> dict:
        saved = self._session.get("gammac") if isinstance(self._session.get("gammac"), dict) else {}
        out = dict(_GENTLE)
        out["lock"] = 1
        out["scale"] = 2
        out["rect"] = None
        for key, value in saved.items():
            if key in out or key in ("lock", "scale", "rect"):
                out[key] = value
        return out

    def _apply_preset_to_controls(self, preset: dict):
        self.combo_lock.blockSignals(True)
        self.combo_scale.blockSignals(True)
        self.combo_lock.setCurrentIndex(max(0, self.combo_lock.findData(int(preset.get("lock", 1)))))
        self.combo_scale.setCurrentIndex(max(0, self.combo_scale.findData(int(preset.get("scale", 2)))))
        self.combo_lock.blockSignals(False)
        self.combo_scale.blockSignals(False)
        self._set_color_values(preset)

    def _save_preset(self):
        values = self._color_values()
        values["lock"] = int(self.combo_lock.currentData())
        values["scale"] = int(self.combo_scale.currentData())
        segment = self._selected()
        values["rect"] = list(segment["rect"]) if segment and segment.get("rect") else None
        self._session["gammac"] = values
        self._save_session()
        where = " y la zona" if values["rect"] else ""
        self.lbl_color.setText(f"Quedó como base para los videos que vengan{where}.")

    def _add_color_slider(self, layout, key, title, lo, hi, scale, apply_knob, tip):
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
        slider.valueChanged.connect(self._on_apply_knob if apply_knob else self._touch_segment)
        layout.addWidget(slider)
        self._csliders[key] = (slider, value_lbl, scale)

    def _set_color_values(self, values: dict):
        for key, (slider, _lbl, scale) in self._csliders.items():
            slider.blockSignals(True)
            slider.setValue(int(round(float(values.get(key, _NEUTRAL[key])) * scale)))
            slider.blockSignals(False)
        self.chk_dark.blockSignals(True)
        self.chk_high.blockSignals(True)
        self.chk_dark.setChecked(bool(values.get("dark_on", False)))
        self.chk_high.setChecked(bool(values.get("high_on", False)))
        self.chk_dark.blockSignals(False)
        self.chk_high.blockSignals(False)
        self._refresh_color_labels()

    def _refresh_color_labels(self):
        for key, (slider, lbl, scale) in self._csliders.items():
            number = slider.value() / float(scale)
            lbl.setText(f"{int(round(number))}" if scale == 1 else f"{number:.2f}")

    def _color_values(self) -> dict:
        values = {}
        for key, (slider, _lbl, scale) in self._csliders.items():
            number = slider.value() / float(scale)
            values[key] = int(round(number)) if scale == 1 else float(number)
        values["dark_on"] = self.chk_dark.isChecked()
        values["high_on"] = self.chk_high.isChecked()
        if values["omin"] > values["omax"]:
            values["omin"], values["omax"] = values["omax"], values["omin"]
        return values

    def _measure_kwargs(self) -> dict:
        values = self._color_values()
        return {
            "lock": int(self.combo_lock.currentData()),
            "scale": int(self.combo_scale.currentData()),
            "lo_th": float(values["lo_th"]),
            "hi_th": float(values["hi_th"]),
            "muls": (float(values["red_mul"]), float(values["grn_mul"]), float(values["blu_mul"])),
            "rng_lim": int(values["rng_lim"]),
            "gam_max": float(values["gam_max"]),
            "strength": float(values["strength"]),
            "omin": float(values["omin"]) / 255.0,
            "omax": float(values["omax"]) / 255.0,
            "dark": (float(values["dark"]) / 255.0) if values["dark_on"] else None,
            "high": (float(values["high"]) / 255.0) if values["high_on"] else None,
        }

    def _write_knobs(self, segment: dict):
        segment.update(self._color_values())
        segment["lock"] = int(self.combo_lock.currentData())
        segment["scale"] = int(self.combo_scale.currentData())

    def _selected(self):
        items = self._adjust.get("segments") or []
        if self._segment < 0 or self._segment >= len(items):
            return None
        return items[self._segment]

    def _segment_rect(self):
        segment = self._selected()
        if not segment or not segment.get("rect"):
            return None
        return tuple(segment["rect"])

    def _ensure_segment(self):
        spans = self._covered_segments()
        frame = int(self.sld_frame.value())
        self._segment = 0
        for index, item in enumerate(spans):
            if int(item["in"]) <= frame <= int(item["out"]):
                self._segment = index
                break
        self._adjust["segments"] = spans
        self._sync_segment_controls()
        return spans[self._segment]

    def _blank_segment(self, start: int, end: int) -> dict:
        preset = self._preset()
        segment = {"in": int(start), "out": int(end), "sample": int(start)}
        segment["rect"] = list(preset["rect"]) if preset.get("rect") else None
        for key in _NEUTRAL:
            segment[key] = preset.get(key, _GENTLE.get(key, _NEUTRAL[key]))
        segment["lock"] = int(preset.get("lock", 1))
        segment["scale"] = int(preset.get("scale", 2))
        return segment

    def _covered_segments(self) -> list:
        return film_finish.cover(self._adjust.get("segments") or [], self._count, self._blank_segment)

    def _blank_crop(self, start: int, end: int) -> dict:
        stored = self._stored_crop_spans()
        master = stored[0] if stored else None
        crop = list(master["crop"]) if master and master.get("crop") else None
        flip = bool(master.get("flip_h")) if master else False
        return {"in": int(start), "out": int(end), "crop": crop, "flip_h": flip}

    def _stored_crop_spans(self) -> list:
        spans = [item for item in (self._adjust.get("crop_spans") or []) if isinstance(item, dict)]
        if spans:
            return spans
        if self._adjust.get("crop") or self._adjust.get("flip_h"):
            return [{
                "in": 0,
                "out": max(0, self._count - 1),
                "crop": self._adjust.get("crop"),
                "flip_h": bool(self._adjust.get("flip_h")),
            }]
        return []

    def _covered_crops(self) -> list:
        return film_finish.cover(self._stored_crop_spans(), self._count, self._blank_crop)

    def _segments_for_save(self) -> list:
        items = self._adjust.get("segments") or []
        if len(items) == 1:
            segment = items[0]
            last = max(0, self._count - 1)
            bare = int(segment.get("in", 0)) == 0 and int(segment.get("out", -1)) == last
            if bare and not segment.get("rect") and not (segment.get("recipe") or {}).get("ok"):
                return []
        return items

    def _crops_for_save(self) -> list:
        spans = [item for item in (self._adjust.get("crop_spans") or []) if isinstance(item, dict)]
        if not spans:
            return []
        if len(spans) == 1 and not spans[0].get("crop") and not spans[0].get("flip_h"):
            return []
        return spans

    def _on_timeline_head(self, frame: int):
        self._stop_play()
        frame = int(frame)
        if int(self.sld_frame.value()) != frame:
            self.sld_frame.setValue(frame)
        self._follow_span()
        self._frame_timer.start()

    def _on_timeline_pick(self, index: int):
        if self._task == 2:
            self._crop_i = int(index)
            self._sync_crop_view()
        else:
            self._segment = int(index)
            self._sync_segment_controls()
        spans = self._covered_crops() if self._task == 2 else self._covered_segments()
        if 0 <= index < len(spans):
            start, end = int(spans[index]["in"]), int(spans[index]["out"])
            frame = int(self.sld_frame.value())
            if frame < start or frame > end:
                self.sld_frame.setValue(start)
                self._load_frame()
        self._refresh_bars()

    def _on_timeline_edit(self, spans, selected: int):
        if self._task == 2:
            self._adjust["crop_spans"] = list(spans)
            self._crop_i = int(selected)
            self.chk_crop.setChecked(any(item.get("crop") for item in spans))
            self._sync_crop_view()
        else:
            self._adjust["segments"] = list(spans)
            self._segment = int(selected)
            self._play_recipe = None
            self._draft_id = None
            self._sync_segment_controls()
        self._save_current()
        self._paint_preview()

    def _split_at_head(self):
        if self._task == 2:
            spans = self._covered_crops()
            new, selected, changed = film_finish.split_at(spans, int(self.sld_frame.value()), self._count)
            if not changed:
                self.lbl_crop.setText("La línea está en el borde del tramo. Métela adentro para partir.")
                return
            self._adjust["crop_spans"] = new
            self._crop_i = selected
        else:
            spans = self._covered_segments()
            new, selected, changed = film_finish.split_at(spans, int(self.sld_frame.value()), self._count)
            if not changed:
                self.lbl_color.setText("La línea está en el borde del tramo. Métela adentro para partir.")
                return
            self._adjust["segments"] = new
            self._segment = selected
            self._sync_segment_controls()
        self._save_current()
        self._paint_preview()

    def _merge_selected(self):
        if self._task == 2:
            spans = self._covered_crops()
            new, selected, changed = film_finish.merge_span(spans, self._crop_i)
            if not changed:
                self.lbl_crop.setText("Hay un solo tramo: cubre el rollo entero.")
                return
            self._adjust["crop_spans"] = new
            self._crop_i = selected
        else:
            spans = self._covered_segments()
            new, selected, changed = film_finish.merge_span(spans, self._segment)
            if not changed:
                self.lbl_color.setText("Hay un solo tramo: cubre el rollo entero.")
                return
            self._adjust["segments"] = new
            self._segment = selected
            self._play_recipe = None
            self._draft_id = None
            self._sync_segment_controls()
        self._save_current()
        self._paint_preview()

    def _follow_span(self):
        frame = int(self.sld_frame.value())
        if self._task == 2:
            spans = self._covered_crops()
            for index, span in enumerate(spans):
                if int(span["in"]) <= frame <= int(span["out"]) and index != self._crop_i:
                    self._crop_i = index
                    self._sync_crop_view()
                    return
        elif self._task == 1:
            spans = self._covered_segments()
            for index, span in enumerate(spans):
                if int(span["in"]) <= frame <= int(span["out"]) and index != self._segment:
                    self._segment = index
                    if self._adjust.get("segments"):
                        self._sync_segment_controls()
                    return

    def _apply_crop_rect(self, full):
        spans = self._covered_crops()
        if not spans:
            return
        index = self._crop_i if 0 <= self._crop_i < len(spans) else 0
        master = spans[0].get("crop")
        if index == 0 or not master:
            rect = film_finish.snap_crop(full, self._full_w, self._full_h)
            if rect is None:
                self.lbl_crop.setText("Ese recorte no entra. Márcalo de nuevo.")
                return
            film_finish.refit_crops(spans, master, rect, self._full_w, self._full_h)
        else:
            aspect = float(master[2]) / float(master[3])
            rect = film_finish.fit_aspect(full, aspect, self._full_w, self._full_h)
            if rect is None:
                return
            spans[index]["crop"] = list(rect)
        self._adjust["crop_spans"] = spans
        self._crop_i = index
        self.chk_crop.blockSignals(True)
        self.chk_crop.setChecked(True)
        self.chk_crop.blockSignals(False)
        self._sync_crop_view()
        self._save_current()

    def _crop_aspect(self):
        spans = self._covered_crops()
        if self._crop_i <= 0 or not spans or not spans[0].get("crop"):
            return None
        crop = spans[0]["crop"]
        if crop[3] <= 0:
            return None
        return float(crop[2]) / float(crop[3])

    def _sync_crop_view(self):
        spans = self._covered_crops()
        if not spans:
            self._crop = None
            return
        index = self._crop_i if 0 <= self._crop_i < len(spans) else 0
        self._crop_i = index
        crop = spans[index].get("crop")
        self._crop = tuple(crop) if crop else None
        if hasattr(self, "chk_flip"):
            self.chk_flip.blockSignals(True)
            self.chk_flip.setChecked(bool(spans[index].get("flip_h")))
            self.chk_flip.blockSignals(False)

    def _expand_single_frame_segment(self):
        items = self._adjust["segments"]
        if len(items) != 1 or int(items[0]["in"]) != int(items[0]["out"]):
            return
        items[0]["in"] = 0
        items[0]["out"] = max(0, self._count - 1)

    def _sync_segment_controls(self):
        segment = self._selected()
        if segment is None:
            self._apply_preset_to_controls(self._preset())
            return
        values = dict(_NEUTRAL)
        values.update({key: segment[key] for key in _NEUTRAL if key in segment})
        self.combo_lock.blockSignals(True)
        self.combo_scale.blockSignals(True)
        self.combo_lock.setCurrentIndex(max(0, self.combo_lock.findData(int(segment.get("lock", 1)))))
        self.combo_scale.setCurrentIndex(max(0, self.combo_scale.findData(int(segment.get("scale", 2)))))
        self.combo_lock.blockSignals(False)
        self.combo_scale.blockSignals(False)
        self._set_color_values(values)

    def _touch_segment(self, *_args):
        self._refresh_color_labels()
        segment = self._selected()
        if segment is None or not segment.get("rect"):
            self.lbl_color.setText("Dibuja el área. Ahí el slider vuelve a medir.")
            self._paint_preview()
            return
        self._write_knobs(segment)
        frame = int(self.sld_frame.value())
        if int(segment["in"]) <= frame <= int(segment["out"]):
            segment["dirty"] = True
            self._play_recipe = None
            self._draft_id = None
            self._paint_preview()
        self._measure_timer.start()

    def _on_apply_knob(self, *_args):
        self._touch_segment()

    def _commit_slider_measure(self):
        self._run_measure(allow_sample=True)

    def _measure_segment(self):
        self._measure_timer.stop()
        self._run_measure(allow_sample=False)

    def _run_measure(self, allow_sample: bool):
        segment = self._selected()
        if segment is None or not segment.get("rect"):
            self.lbl_color.setText("Primero dibuja el área y marca el tramo.")
            return
        frame = int(self.sld_frame.value())
        if not (int(segment["in"]) <= frame <= int(segment["out"])):
            sample = segment.get("sample")
            if allow_sample and sample is not None and int(segment["in"]) <= int(sample) <= int(segment["out"]):
                frame = int(sample)
            else:
                self.lbl_color.setText("El cuadro tiene que estar dentro del tramo.")
                return
        clip = self._clips.get(self._current or "")
        if clip is None or not clip.linked:
            return
        try:
            revealed = read_rgb(
                Path(clip.path), frame, clip.fps, clip.width, clip.height, full=True,
            )
        except VideoError as exc:
            self.lbl_color.setText(str(exc))
            return
        self._write_knobs(segment)
        recipe = gammac.measure(revealed, segment["rect"], **self._measure_kwargs())
        segment["recipe"] = recipe
        segment["sample"] = frame
        segment["dirty"] = False
        self._play_recipe = None
        self._draft_id = None
        if int(segment["in"]) == int(segment["out"]) and len(self._adjust["segments"]) == 1:
            segment["in"] = 0
            segment["out"] = max(0, self._count - 1)
        self.lbl_color.setText(recipe.get("message") or "Medido.")
        self._save_current()
        self._refresh_bars()
        self._paint_preview()

    def _recipe_for_preview(self, revealed, index: int):
        owner = None
        for item in self._adjust["segments"]:
            if int(item["in"]) <= index <= int(item["out"]):
                owner = item
                break
        if owner and owner.get("recipe") and not owner.get("dirty"):
            return owner["recipe"]
        selected = self._selected()
        if self._task == 1 and selected is owner and selected and selected.get("rect"):
            if self._draft_id == id(selected) and self._play_recipe is not None:
                return self._play_recipe
            preview_rect = self._preview_rect(selected["rect"])
            if preview_rect:
                recipe = gammac.measure(revealed, preview_rect, **self._measure_kwargs())
                if recipe.get("ok"):
                    self._play_recipe = recipe
                    self._draft_id = id(selected)
                    return recipe
            return self._play_recipe
        return None

    def _turn_degrees(self) -> float:
        if self._task != 2:
            return 0.0
        try:
            return float(self._adjust.get("turn") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _spin_center(self, width: int, height: int) -> tuple[float, float]:
        if self._crop:
            x, y, rect_w, rect_h = self._crop
            sx = width / float(self._full_w)
            sy = height / float(self._full_h)
            return ((x + rect_w / 2.0) * sx, (y + rect_h / 2.0) * sy)
        return (width / 2.0, height / 2.0)

    def _warp_preview(self, image):
        tx, ty = 0.0, 0.0
        if self._shifts:
            index = int(self.sld_frame.value())
            if index < len(self._shifts[0]):
                tx = self._shifts[0][index] * self._sigma
                ty = self._shifts[1][index] * self._sigma
        if not self.chk_horizontal.isChecked():
            tx = 0.0
        degrees = self._turn_degrees()
        if abs(tx) < 1e-3 and abs(ty) < 1e-3 and abs(degrees) < 0.05:
            return image
        height, width = image.shape[:2]
        matrix = film_finish.sampling_matrix(tx, ty, degrees, self._spin_center(width, height))
        return cv2.warpAffine(
            image, matrix, (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def _crop_side(self, image):
        rect = self._preview_rect(self._crop) if self.chk_crop.isChecked() else None
        if not rect:
            return None
        x, y, width, height = rect
        crop = image[y:y + height, x:x + width]
        if self._flip_on():
            crop = _hflip(crop)
        return _to_u8(crop)

    def _flip_on(self) -> bool:
        return self.chk_flip.isChecked()

    def _shown_crop(self):
        if not self._crop or not self._flip_on():
            return self._crop
        return film_finish.mirror_x(self._crop, self._full_w)

    def _on_flip(self, *_args):
        spans = self._covered_crops()
        if spans:
            index = min(max(self._crop_i, 0), len(spans) - 1)
            spans[index]["flip_h"] = self.chk_flip.isChecked()
            self._adjust["crop_spans"] = spans
        self._save_current()
        self._paint_preview()

    def _preview_rect(self, rect):
        if not rect or self._full_w < 2:
            return None
        sx = self._preview_wh[0] / float(self._full_w)
        sy = self._preview_wh[1] / float(self._full_h)
        x = int(round(rect[0] * sx))
        y = int(round(rect[1] * sy))
        width = max(1, int(round(rect[2] * sx)))
        height = max(1, int(round(rect[3] * sy)))
        return (
            min(x, self._preview_wh[0] - 1),
            min(y, self._preview_wh[1] - 1),
            min(width, self._preview_wh[0] - x),
            min(height, self._preview_wh[1] - y),
        )

    def _full_rect(self, rect):
        if self._preview_wh[0] < 2:
            return None
        sx = self._full_w / float(self._preview_wh[0])
        sy = self._full_h / float(self._preview_wh[1])
        return (
            int(round(rect[0] * sx)),
            int(round(rect[1] * sy)),
            max(8, int(round(rect[2] * sx))),
            max(8, int(round(rect[3] * sy))),
        )

    def _refresh_bars(self):
        if not hasattr(self, "bar"):
            return
        if self._task == 2:
            spans, selected = self._covered_crops(), self._crop_i
        else:
            spans, selected = self._covered_segments(), self._segment
        self.bar.set_bar(self._count, spans, selected, int(self.sld_frame.value()))

    def _refresh_crop_label(self):
        if self._crop is None:
            turn = self.turn.degrees()
            text = "Dibuja el recorte sobre la imagen. El primer tramo fija la proporción."
            if abs(turn) >= 0.05:
                text += " Giro " + f"{turn:.1f}°".replace(".", ",") + "."
            if self.chk_flip.isChecked():
                text += " Volteado horizontal."
            self.lbl_crop.setText(text)
            return
        if self._crop_i <= 0:
            role = " Este tramo fija la proporción y el tamaño del video."
        else:
            role = " Misma proporción que el primero: se mueve y se escala parejo."
        text = f"Recorte {self._crop[2]}×{self._crop[3]} desde {self._crop[0]}, {self._crop[1]}.{role}"
        if self._pad:
            text += f" La perforación deja {self._pad} px de negro."
        turn = self.turn.degrees()
        if abs(turn) >= 0.05:
            text += " Giro " + f"{turn:.1f}°".replace(".", ",") + "."
        if self.chk_flip.isChecked():
            text += " Volteado horizontal."
        self.lbl_crop.setText(text)

    def _sync_turn(self):
        try:
            degrees = float(self._adjust.get("turn") or 0.0)
        except (TypeError, ValueError):
            degrees = 0.0
        self.turn.set_degrees(degrees)

    def _on_turn(self, degrees: float):
        self._adjust["turn"] = round(float(degrees), 1)
        self._save_current()
        self._paint_preview()

    def _export_clicked(self):
        if self._export is not None and self._export.isRunning():
            self.lbl_status.setText("Ya hay una exportación en curso.")
            return
        self._save_current()
        linked = [self._clips[path] for path in self._order() if self._clips.get(path) and self._clips[path].linked]
        if not linked:
            self.lbl_status.setText("No hay videos enlazados para exportar.")
            return
        if not self._confirm_export(linked):
            return
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de salida", str(Path(linked[0].path).parent))
        if not folder:
            return
        codec = self.combo_fmt.currentData()
        self._export_engine = "ProRes NV" if codec == "prores_4444" else "NVENC"
        taken: set[str] = set()
        jobs = []
        pending = []
        for clip in linked:
            dst = output_path(Path(folder), Path(clip.path), codec, taken)
            fps_text = clip.fps_text if not clip.fps_edited else f"{clip.fps:.6f}"
            jobs.append((clip.path, str(dst), codec, fps_text))
            needed = self._stab_job(clip)
            if needed is not None:
                pending.append(needed)
        self._fill_export_table(jobs, {clip.path for clip, _zone, _ref in pending})
        if not pending:
            self._start_export(jobs)
            return
        self._export_after_analyze = jobs
        self.btn_export.setEnabled(False)
        self.btn_cancel.setVisible(True)
        names = ", ".join(Path(clip.path).name for clip, _zone, _ref in pending[:3])
        extra = len(pending) - 3
        if extra > 0:
            names += f" y {extra} más"
        self.lbl_status.setText(f"Primero mido la estabilización de {names}. Después exporto.")
        for clip, zone, ref in pending:
            if not self._analysis_queued(clip.path):
                self._enqueue(clip, zone, ref)
        if (self._analyze is None or not self._analyze.isRunning()) and not self._analyze_queue:
            self._maybe_export_after_analyze()

    def _confirm_export(self, linked: list[Clip]) -> bool:
        """Avisa de los que salen sin estabilización y de los que se miden antes de exportar."""
        off = []
        pending = []
        for clip in linked:
            state, _zone = self._perfo_plan(clip)
            name = Path(clip.path).name
            if state == "off":
                off.append(name)
            elif state == "pending":
                pending.append(name)
        if not off and not pending:
            return True
        lines = []
        if off:
            lines.append(
                "Estos no tienen la estabilización marcada y se exportan sin ella:\n\n"
                + self._name_block(off)
            )
        if pending:
            lines.append(
                "Estos la tienen marcada y todavía no está medida. "
                "Los analizo primero y después exporto:\n\n"
                + self._name_block(pending)
            )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Exportar")
        box.setText("\n\n".join(lines))
        box.setInformativeText("Puedes volver a marcar, o seguir con la exportación.")
        box.addButton("Volver", QMessageBox.ButtonRole.RejectRole)
        go = box.addButton("Exportar", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(go)
        box.exec()
        return box.clickedButton() is go

    def _name_block(self, names: list[str]) -> str:
        shown = "\n".join(names[:8])
        if len(names) > 8:
            shown += f"\n… y {len(names) - 8} más"
        return shown

    def _perfo_plan(self, clip: Clip):
        """off si no está marcada, pending si falta medir, ready si ya está hecha."""
        data = film_finish.load(Path(clip.path))
        perfo = data.get("perfo") or {}
        roi = perfo.get("roi")
        if not perfo.get("enabled") or not roi or len(roi) != 4:
            return "off", None
        zone = tuple(int(v) for v in roi)
        if film_finish.perfo_is_done(Path(clip.path), zone, clip.frames):
            return "ready", zone
        return "pending", zone

    def _stab_job(self, clip: Clip):
        """Zona marcada que todavía no tiene desplazamiento. None si no hay nada que medir."""
        state, zone = self._perfo_plan(clip)
        if state != "pending" or zone is None:
            return None
        return clip, zone, self._saved_ref(clip)

    def _saved_ref(self, clip: Clip) -> int:
        """El cuadro en el que se dibujó el rectángulo."""
        if clip.path == self._current and self._roi_ref is not None:
            raw = self._roi_ref
        else:
            raw = film_finish.perfo_ref(film_finish.load(Path(clip.path)).get("perfo") or {})
        try:
            ref = int(raw)
        except (TypeError, ValueError):
            ref = int(self.sld_frame.value()) if clip.path == self._current else 0
        last = max(0, int(clip.frames) - 1)
        return min(max(0, ref), last)

    def _analysis_queued(self, path: str) -> bool:
        if any(item[0] == path for item in self._analyze_queue):
            return True
        thread = self._analyze
        return bool(thread is not None and thread.isRunning() and any(item[0] == path for item in thread.jobs))

    def _fill_export_table(self, jobs, measuring: set[str]):
        self.table.setRowCount(0)
        self._export_rows.clear()
        self._bars.clear()
        for path, _dst, _codec, _fps in jobs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(Path(path).name))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
            self.table.setCellWidget(row, 1, bar)
            self.table.setItem(row, 2, QTableWidgetItem("Midiendo" if path in measuring else "—"))
            self._export_rows[path] = row
            self._bars[path] = bar
        self.table.setVisible(True)
        self.table.resizeColumnsToContents()

    def _maybe_export_after_analyze(self):
        jobs = self._export_after_analyze
        if not jobs:
            return
        if self._analyze is not None and self._analyze.isRunning():
            return
        if self._analyze_queue:
            return
        self._export_after_analyze = None
        kept = []
        failed = []
        for job in jobs:
            clip = self._clips.get(job[0])
            if clip is not None and self._stab_job(clip) is not None:
                failed.append(Path(job[0]).name)
                continue
            kept.append(job)
        if failed and not kept:
            self.btn_export.setEnabled(True)
            self.btn_cancel.setVisible(False)
            self.lbl_status.setText(
                "No pude medir la estabilización de " + ", ".join(failed) + ". No exporté."
            )
            return
        if failed:
            self.lbl_status.setText(
                "No pude medir " + ", ".join(failed) + ". Exporto el resto."
            )
        self._start_export(kept)

    def _start_export(self, jobs):
        if not jobs:
            self.btn_export.setEnabled(True)
            self.btn_cancel.setVisible(False)
            return
        self._export_started = time.perf_counter()
        self._export_file_started = self._export_started
        self._export_following = {}
        for path, _dst, _codec, _fps in jobs:
            clip = self._clips.get(path)
            self._export_following[path] = clip.frames if clip is not None else 0
            bar = self._bars.get(path)
            if bar is not None:
                bar.setValue(0)
            row = self._export_rows.get(path)
            if row is not None and self.table.item(row, 2) is not None:
                self.table.item(row, 2).setText("—")
        self.btn_export.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self._export = _ExportThread(jobs)
        self._export.frame.connect(self._on_export_frame)
        self._export.file_done.connect(self._on_export_file)
        self._export.finished.connect(self._on_export_finished)
        self._export.start()
        if not self.lbl_status.text().startswith("No pude medir"):
            self.lbl_status.setText("Exportando…")

    def _on_export_frame(self, path: str, done: int, total: int):
        bar = self._bars.get(path)
        if bar is not None and total:
            bar.setValue(int(round(100 * done / total)))
        row = self._export_rows.get(path)
        elapsed = max(0.001, time.perf_counter() - self._export_file_started)
        speed = done / elapsed
        following = 0
        seen = False
        for key, frames in self._export_following.items():
            if key == path:
                seen = True
                continue
            if seen:
                following += frames
        _file_eta, total_eta = estimate_remaining_seconds(speed, done, total, following)
        if row is not None:
            self.table.item(row, 2).setText(format_duration(_file_eta))
        pct = (100.0 * done / total) if total else 0
        self.lbl_status.setText(
            f"{Path(path).name}: cuadro {done}/{total} ({pct:.1f}%) · {speed:.2f} fps · "
            f"{self._export_engine} · faltan {format_duration(total_eta)}"
        )

    def _on_export_file(self, path: str, ok: bool, message: str):
        row = self._export_rows.get(path)
        bar = self._bars.get(path)
        if ok and bar is not None:
            bar.setValue(100)
        if row is not None and not ok:
            self.table.item(row, 2).setText(message or "Error")
        self._export_file_started = time.perf_counter()
        if not ok and message:
            self.lbl_status.setText(f"{Path(path).name}: {message}")

    def _on_export_finished(self):
        self.btn_export.setEnabled(True)
        if self._analyze is None or not self._analyze.isRunning():
            self.btn_cancel.setVisible(False)
        if self.lbl_status.text().startswith("Exportando") or "faltan" in self.lbl_status.text():
            self.lbl_status.setText("Exportación terminada.")

    def _cancel_work(self):
        self._export_after_analyze = None
        self._analyze_queue.clear()
        if self._analyze is not None and self._analyze.isRunning():
            self._analyze.cancel()
        if self._export is not None and self._export.isRunning():
            self._export.cancel()
        self.lbl_status.setText("Cancelando…")


def _clip_from_item(item: dict) -> Clip:
    path = str(item.get("path") or "")
    try:
        fps = float(item.get("fps") or 24)
    except (TypeError, ValueError):
        fps = 24.0
    adjust = item.get("adjust") if isinstance(item.get("adjust"), dict) else film_finish.empty()
    return Clip(
        path=path,
        fps=fps if fps > 0 else 24.0,
        fps_text=str(item.get("fps_text") or "24/1"),
        fps_edited=bool(item.get("fps_edited")),
        width=int(item.get("width") or 0),
        height=int(item.get("height") or 0),
        frames=int(item.get("frames") or 0),
        linked=Path(path).is_file(),
        adjust=adjust,
    )


def _clip_to_item(clip: Clip) -> dict:
    return {
        "path": clip.path,
        "fps": clip.fps,
        "fps_text": clip.fps_text,
        "fps_edited": clip.fps_edited,
        "width": clip.width,
        "height": clip.height,
        "frames": clip.frames,
        "adjust": clip.adjust,
    }


def _apply_probe(clip: Clip, info: VideoInfo) -> None:
    clip.width = info.width
    clip.height = info.height
    clip.frames = info.frames
    clip.linked = True
    if not clip.fps_edited:
        clip.fps = info.fps
        clip.fps_text = info.fps_text


def _clip_label(clip: Clip) -> str:
    name = Path(clip.path).name
    if not clip.linked:
        return f"Sin enlace · {name}"
    if abs(clip.fps - round(clip.fps)) < 0.001:
        fps = str(int(round(clip.fps)))
    else:
        fps = f"{clip.fps:.3f}".replace(".", ",")
    seconds = clip.frames / clip.fps if clip.fps else 0
    return f"{name}    {fps} fps    {format_duration(seconds)}"


def _hflip(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, ::-1])


def _to_u8(image) -> np.ndarray:
    return np.clip(image * np.float32(255.0) + np.float32(0.5), 0, 255).astype(np.uint8)


def _style() -> str:
    return """
        QMainWindow, QWidget { background: #161616; color: #e8e8e8; }
        QLabel { background: transparent; }
        QListWidget, QTableWidget, QComboBox, QDoubleSpinBox, QScrollArea {
            background: #222; color: #f2f2f2; border: 1px solid #3a3a3a;
        }
        QPushButton { background: #2c2c2c; color: #f2f2f2; border: 1px solid #4a4a4a; padding: 4px 8px; }
        QPushButton:checked { background: #0e4a7b; }
        QProgressBar { background: #2a2a2a; border: 1px solid #444; text-align: center; color: #eee; }
        QProgressBar::chunk { background: #0e6aa8; }
        QHeaderView::section { background: #222; color: #ccc; }
    """
