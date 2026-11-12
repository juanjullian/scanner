"""Ajustes del master: revelado, perforación, GamMac por tramos y recorte."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPropertyAnimation, QRect, Qt, QTimer, pyqtSignal
import time

from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFrame, QGraphicsOpacityEffect, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from timeline_bar import TimelineBar, _timeline_button

import film_finish
import gammac
import reveal_pipeline

FIXED_W = 2840
FIXED_H = 2200
_PREVIEW_W = 1100
_TASKS = ("Revelado", "Estabilización", "Color", "Recorte")
# Un tramo nuevo sale así, parecido al panel de GamMac que afloja la corrección.
# Un tramo viejo, sin estas claves, sigue a gamma completa.
_GENTLE = {
    "lo_th": 0.12,
    "hi_th": 0.25,
    "gam_max": 2.5,
    "rng_lim": 11,
    "strength": 0.5,
    "omin": 8,
    "omax": 240,
    "dark_on": True,
    "dark": 35,
    "high_on": True,
    "high": 220,
    "red_mul": 1.0,
    "grn_mul": 1.0,
    "blu_mul": 1.0,
}
_NEUTRAL = {
    "lo_th": 0.0,
    "hi_th": 0.0,
    "gam_max": 10.0,
    "rng_lim": 11,
    "strength": 1.0,
    "omin": 0,
    "omax": 255,
    "dark_on": False,
    "dark": 35,
    "high_on": False,
    "high": 220,
    "red_mul": 1.0,
    "grn_mul": 1.0,
    "blu_mul": 1.0,
}

_BADGE_STYLE = {
    "ok": "color:#b7d7b7; background:rgba(129,199,132,28); border:1px solid rgba(150,190,150,150);",
    "miss": "color:#b0b0b0; background:transparent; border:1px solid rgba(255,255,255,45);",
    "wait": "color:#e2c48a; background:rgba(224,181,106,22); border:1px solid rgba(210,170,110,130);",
    "run": "color:#b7cfe2; background:rgba(120,170,210,22); border:1px solid rgba(140,180,210,130);",
}
_CHIP_TAIL = "border-radius:8px; padding:0px 6px; font-size:8pt;"


def _paint_badges(layout, badges: list) -> None:
    labels = [layout.itemAt(i).widget() for i in range(layout.count())]
    labels = [item for item in labels if isinstance(item, QLabel)]
    if len(labels) != len(badges):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        labels = []
        for _item in badges:
            chip = QLabel()
            chip.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            chip.setFixedHeight(18)
            layout.addWidget(chip, 0, Qt.AlignmentFlag.AlignVCenter)
            labels.append(chip)
    for chip, item in zip(labels, badges):
        text, kind = item[0], item[1]
        chip.setText(text)
        chip.setToolTip(item[2] if len(item) > 2 else text)
        chip.setStyleSheet(_BADGE_STYLE.get(kind, _BADGE_STYLE["miss"]) + _CHIP_TAIL)


def _is_bayer(filename: str, meta: dict) -> bool:
    return meta.get(filename, {}).get("pixel_format", "bayer") not in ("rgb", "qoi_rgb")


def _is_bw(filename: str, meta: dict) -> bool:
    return "Blanco" in meta.get(filename, {}).get("type", "")


def _find_jobs(widget):
    host = widget
    while host is not None:
        jobs = getattr(host, "adjust_jobs", None)
        if jobs is not None:
            return jobs
        host = host.parent()
    return None


def _hflip(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, ::-1])


def _rgb_pixmap(rgb: np.ndarray) -> QPixmap:
    u8 = np.ascontiguousarray(rgb)
    height, width = u8.shape[:2]
    image = QImage(u8.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class _PlayButton(QWidget):
    """Círculo semitransparente, con triángulo o barras, encima del cuadro."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)
        self._playing = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_playing(self, playing: bool):
        self._playing = bool(playing)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 150), 1.5))
        painter.setBrush(QColor(0, 0, 0, 80))
        painter.drawEllipse(3, 3, 66, 66)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 210))
        if self._playing:
            painter.drawRoundedRect(26, 22, 7, 28, 1, 1)
            painter.drawRoundedRect(39, 22, 7, 28, 1, 1)
            return
        path = QPainterPath()
        path.moveTo(29, 22)
        path.lineTo(29, 50)
        path.lineTo(50, 36)
        path.closeSubpath()
        painter.drawPath(path)


class FrameView(QWidget):
    """Vista del cuadro. El rectángulo se dibuja encima y vuelve en píxeles de la imagen."""

    rect_ready = pyqtSignal(tuple)

    play_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self._pix = QPixmap()
        self._side = QPixmap()
        self._overlay = None
        self._color = QColor("#ffcc00")
        self._drawing = False
        self._allow = False
        self._origin = None
        self._draft = None
        self._aspect = None
        self._mode = None
        self._anchor = None
        self._grab = None
        self._grab_rect = None
        self._split = 0
        self._captions = ()
        self._empty = "Elige una película Bayer."
        self._play = _PlayButton(self)
        self._play.clicked.connect(self.play_clicked.emit)
        self._play.setToolTip(
            "Reproduce a los FPS del archivo. "
            "Color, estabilización y recorte se suman según la pestaña. "
            "Al pausar vuelve el cuadro fino."
        )
        self._play_fx = QGraphicsOpacityEffect(self._play)
        self._play_fx.setOpacity(0.0)
        self._play.setGraphicsEffect(self._play_fx)
        self._play_fade = QPropertyAnimation(self._play_fx, b"opacity", self)
        self._play_fade.setDuration(180)

    def set_view(self, rgb: np.ndarray, overlay=None, color="#ffcc00", side=None, allow_draw=False, captions=None, aspect=None):
        self._pix = _rgb_pixmap(rgb)
        self._side = _rgb_pixmap(side) if side is not None else QPixmap()
        self._overlay = overlay
        self._color = QColor(color)
        self._allow = allow_draw
        self._captions = tuple(captions) if captions else ()
        self._aspect = float(aspect) if aspect else None
        if self._mode is None:
            self._draft = None
        self._place_play()
        self.update()

    def set_empty(self, text: str):
        self._empty = text
        self.update()

    def clear(self):
        self._pix = QPixmap()
        self._side = QPixmap()
        self._overlay = None
        self._captions = ()
        self._allow = False
        self.update()

    def set_playing(self, playing: bool):
        self._play.set_playing(playing)
        if playing:
            self._fade_play(1.0)

    def enterEvent(self, event):
        self._fade_play(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._play._playing:
            self._fade_play(1.0)
        elif not self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            self._fade_play(0.0)
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_play"):
            self._place_play()

    def _fade_play(self, target: float):
        self._play.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, target <= 0.0,
        )
        self._play_fade.stop()
        self._play_fade.setStartValue(self._play_fx.opacity())
        self._play_fade.setEndValue(target)
        self._play_fade.start()

    def _place_play(self):
        left, _right = self._placed()
        if left.width() < 2:
            x = max(0, (self.width() - self._play.width()) // 2)
            y = max(0, (self.height() - self._play.height()) // 2)
        else:
            x = left.center().x() - self._play.width() // 2
            y = left.center().y() - self._play.height() // 2
        self._play.move(x, y)
        self._play.raise_()

    def _caption(self, painter: QPainter, placed: QRect, text: str):
        if not text or placed is None or placed.width() < 8:
            return
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 14
        height = metrics.height() + 8
        box = QRect(placed.left() + 8, placed.top() + 8, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRect(box)
        painter.setPen(QColor("#f4f4f4"))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if self._pix.isNull():
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty)
            return
        left, right = self._placed()
        painter.drawPixmap(left, self._pix)
        if not self._side.isNull() and right is not None:
            painter.drawPixmap(right, self._side)
        if self._captions:
            self._caption(painter, left, self._captions[0])
            if right is not None and len(self._captions) > 1:
                self._caption(painter, right, self._captions[1])
        rect = self._draft or self._overlay
        if rect and left.width() > 0:
            painter.setPen(QPen(self._color, 2))
            mapped = self._map_out(rect, left, self._pix)
            painter.drawRect(mapped)
            if self._aspect and self._allow:
                painter.setBrush(self._color)
                for corner in (mapped.topLeft(), mapped.topRight(), mapped.bottomLeft(), mapped.bottomRight()):
                    painter.drawRect(corner.x() - 3, corner.y() - 3, 6, 6)
                painter.setBrush(Qt.BrushStyle.NoBrush)

    def mousePressEvent(self, event):
        if not self._allow or self._pix.isNull() or event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position().toPoint()
        hit = self._crop_hit(point) if self._overlay else None
        if hit == "move":
            self._mode = "move"
            self._grab = point
            self._grab_rect = tuple(self._overlay)
            self._draft = tuple(self._overlay)
            return
        if hit and hit.startswith("scale"):
            self._mode = "scale"
            self._anchor = self._opposite(self._overlay, hit)
            self._draft = tuple(self._overlay)
            return
        self._mode = "draw"
        self._drawing = True
        self._origin = point
        self._draft = None

    def mouseMoveEvent(self, event):
        point = event.position().toPoint()
        if self._mode == "move" and self._grab is not None and self._grab_rect is not None:
            left, _right = self._placed()
            if left.width() < 2:
                return
            dx = self._axis(point.x(), left.left(), left.width(), self._pix.width())
            dy = self._axis(point.y(), left.top(), left.height(), self._pix.height())
            ox = self._axis(self._grab.x(), left.left(), left.width(), self._pix.width())
            oy = self._axis(self._grab.y(), left.top(), left.height(), self._pix.height())
            x, y, width, height = self._grab_rect
            self._draft = self._clamp_rect((x + dx - ox, y + dy - oy, width, height))
            self.update()
            return
        if self._mode == "scale" and self._anchor is not None:
            self._draft = self._scaled(point)
            self.update()
            return
        if self._mode == "draw" and self._origin is not None:
            draft = self._image_rect(self._origin, point)
            if self._aspect and draft:
                draft = film_finish.fit_aspect(
                    draft, self._aspect, self._pix.width(), self._pix.height(), minimum=4,
                )
            self._draft = draft
            self.update()
            return
        hit = self._crop_hit(point) if self._allow and self._overlay else None
        if hit == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif hit and hit.startswith("scale"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        mode = self._mode
        if mode is None:
            return
        rect = self._draft
        if mode == "draw":
            rect = self._image_rect(self._origin, event.position().toPoint())
            if self._aspect and rect:
                rect = film_finish.fit_aspect(
                    rect, self._aspect, self._pix.width(), self._pix.height(), minimum=4,
                )
        self._mode = None
        self._drawing = False
        self._origin = None
        self._anchor = None
        self._grab = None
        self._draft = None
        limit = 8 if mode == "draw" else 4
        if rect and rect[2] >= limit and rect[3] >= limit:
            self.rect_ready.emit(tuple(int(v) for v in rect))
        else:
            self.update()

    def _clamp_rect(self, rect):
        x, y, width, height = (int(v) for v in rect)
        limit_w = max(1, self._pix.width())
        limit_h = max(1, self._pix.height())
        width = min(max(1, width), limit_w)
        height = min(max(1, height), limit_h)
        x = min(max(0, x), limit_w - width)
        y = min(max(0, y), limit_h - height)
        return x, y, width, height

    def _scaled(self, point: QPoint):
        here = self._image_point(point)
        if here is None or self._anchor is None or not self._aspect:
            return self._draft
        ax, ay = self._anchor
        span_w = abs(here[0] - ax)
        span_h = abs(here[1] - ay)
        if span_w < 1 and span_h < 1:
            return self._draft
        if span_h < 1 or span_w / max(span_h, 1) > self._aspect:
            span_h = span_w / self._aspect
        else:
            span_w = span_h * self._aspect
        x = ax if here[0] >= ax else ax - span_w
        y = ay if here[1] >= ay else ay - span_h
        locked = film_finish.fit_aspect(
            (x, y, span_w, span_h), self._aspect, self._pix.width(), self._pix.height(), minimum=4,
        )
        return locked or self._draft

    def _image_point(self, point: QPoint):
        left, _right = self._placed()
        if left.width() < 2 or left.height() < 2:
            return None
        return (
            self._axis(point.x(), left.left(), left.width(), self._pix.width()),
            self._axis(point.y(), left.top(), left.height(), self._pix.height()),
        )

    def _crop_hit(self, point: QPoint):
        if not self._overlay or self._pix.isNull():
            return None
        left, _right = self._placed()
        if left.width() < 2:
            return None
        mapped = self._map_out(self._draft or self._overlay, left, self._pix)
        if self._aspect:
            corners = {
                "scale-tl": mapped.topLeft(),
                "scale-tr": mapped.topRight(),
                "scale-bl": mapped.bottomLeft(),
                "scale-br": mapped.bottomRight(),
            }
            for name, corner in corners.items():
                if abs(point.x() - corner.x()) <= 8 and abs(point.y() - corner.y()) <= 8:
                    return name
        if mapped.adjusted(-2, -2, 2, 2).contains(point):
            return "move"
        return None

    @staticmethod
    def _opposite(rect, hit: str):
        x, y, width, height = rect
        if hit == "scale-br":
            return x, y
        if hit == "scale-bl":
            return x + width, y
        if hit == "scale-tr":
            return x, y + height
        return x + width, y + height

    def _placed(self):
        gap = 12 if not self._side.isNull() else 0
        avail_w = max(1, self.width() - gap)
        if self._side.isNull():
            left = self._fit(self._pix, self.width(), self.height(), 0)
            return left, None
        half = avail_w // 2
        left = self._fit(self._pix, half, self.height(), 0)
        right = self._fit(self._side, half, self.height(), half + gap)
        return left, right

    def _fit(self, pix: QPixmap, width: int, height: int, x_off: int) -> QRect:
        scaled = pix.size()
        scaled.scale(max(1, width), max(1, height), Qt.AspectRatioMode.KeepAspectRatio)
        x = x_off + max(0, (width - scaled.width()) // 2)
        y = max(0, (height - scaled.height()) // 2)
        return QRect(x, y, scaled.width(), scaled.height())

    def _image_rect(self, a: QPoint, b: QPoint):
        left, _right = self._placed()
        if left.width() < 2 or left.height() < 2:
            return None
        x0 = self._axis(a.x(), left.left(), left.width(), self._pix.width())
        y0 = self._axis(a.y(), left.top(), left.height(), self._pix.height())
        x1 = self._axis(b.x(), left.left(), left.width(), self._pix.width())
        y1 = self._axis(b.y(), left.top(), left.height(), self._pix.height())
        x = min(x0, x1)
        y = min(y0, y1)
        return (x, y, max(1, abs(x1 - x0)), max(1, abs(y1 - y0)))

    @staticmethod
    def _axis(value, origin, span, limit):
        rel = (value - origin) / float(span)
        return int(np.clip(round(rel * limit), 0, limit - 1))

    @staticmethod
    def _map_out(rect, placed: QRect, pix: QPixmap) -> QRect:
        x, y, width, height = rect
        sx = placed.width() / float(pix.width())
        sy = placed.height() / float(pix.height())
        return QRect(
            placed.left() + int(round(x * sx)),
            placed.top() + int(round(y * sy)),
            max(1, int(round(width * sx))),
            max(1, int(round(height * sy))),
        )


class _AngleEdit(QLineEdit):
    escaped = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escaped.emit()
            return
        super().keyPressEvent(event)


class _AngleLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class TurnControl(QWidget):
    """Giro fino. La barra mueve 0,1° por paso, en un tramo de ±15°, y vuelve al centro."""

    changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._degrees = 0.0
        self._origin = 0.0
        self._lock = False
        self._editing = False
        tip = (
            "El recorte no se mueve. Gira la imagen detrás, como en Photoshop. "
            "A la derecha es positivo. Cada paso de la barra es 0,1°, "
            "en un tramo de 15° a cada lado, y al soltar vuelve al centro. "
            "Un clic en el número lo deja escribir, de -180 a 180."
        )
        title = QLabel("Giro")
        title.setToolTip(tip)
        self._label = _AngleLabel("0,0°")
        self._label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._label.setToolTip("Clic para escribir el ángulo.")
        self._label.clicked.connect(self._start_edit)
        self._edit = _AngleEdit()
        self._edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._edit.setFixedWidth(76)
        self._edit.setPlaceholderText("0,0")
        self._edit.hide()
        self._edit.returnPressed.connect(self._commit_edit)
        self._edit.editingFinished.connect(self._commit_edit)
        self._edit.escaped.connect(self._cancel_edit)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self._label)
        head.addWidget(self._edit)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(-150, 150)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.setValue(0)
        self._slider.setToolTip(tip)
        self._slider.sliderPressed.connect(self._grab)
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderReleased.connect(self._recenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(head)
        layout.addWidget(self._slider)

    def degrees(self) -> float:
        return self._degrees

    def set_degrees(self, degrees: float, emit: bool = False):
        self._degrees = max(-180.0, min(180.0, round(float(degrees), 1)))
        self._origin = self._degrees
        self._paint_label()
        self._recenter()
        if emit:
            self.changed.emit(self._degrees)

    def _grab(self):
        self._origin = self._degrees

    def _on_slider(self, value: int):
        if self._lock:
            return
        degrees = max(-180.0, min(180.0, round(self._origin + value / 10.0, 1)))
        if degrees != self._degrees:
            self._degrees = degrees
            self._paint_label()
            self.changed.emit(self._degrees)
        if not self._slider.isSliderDown():
            self._recenter()

    def _recenter(self):
        self._origin = self._degrees
        self._lock = True
        self._slider.setValue(0)
        self._lock = False

    def _paint_label(self):
        self._label.setText(f"{self._degrees:.1f}°".replace(".", ","))

    def _start_edit(self):
        self._editing = True
        self._edit.setText(f"{self._degrees:.1f}".replace(".", ","))
        self._label.hide()
        self._edit.show()
        self._edit.setFocus()
        self._edit.selectAll()

    def _commit_edit(self):
        if not self._editing:
            return
        self._editing = False
        text = self._edit.text().replace("°", "").replace(",", ".").strip()
        self._edit.hide()
        self._label.show()
        try:
            value = float(text)
        except ValueError:
            self._paint_label()
            return
        self.set_degrees(value, emit=True)

    def _cancel_edit(self):
        self._editing = False
        self._edit.hide()
        self._label.show()
        self._paint_label()


class RevealDialog(QDialog):
    def __init__(self, parent, folder: Path, filenames: list[str], collection_meta: dict, settings: dict, start_name: str | None):
        super().__init__(parent)
        self.setWindowTitle("Ajustes")
        self.resize(1360, 840)
        self._folder = Path(folder)
        self._filenames = list(filenames)
        self._meta = collection_meta or {}
        self._reveal_fallback = reveal_pipeline.coerce_reveal(settings)
        self._settings = reveal_pipeline.coerce_reveal(settings)
        self._lin_prev = None
        self._sigma_scale = 1.0
        self._preview_wh = (1, 1)
        self._count = 1
        self._current = None
        self._adjust = film_finish.empty()
        self._roi = None
        self._roi_ref = None
        self._crop = None
        self._shifts = None
        self._pad = 0
        self._task = 0
        self._segment = -1
        self._jobs = _find_jobs(self)
        self._playing = False
        self._play_recipe = None
        self._draft_id = None
        self._csliders = {}

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._on_reveal_timer)
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(120)
        self._frame_timer.timeout.connect(self._load_frame)
        self._measure_timer = QTimer(self)
        self._measure_timer.setSingleShot(True)
        self._measure_timer.setInterval(200)
        self._measure_timer.timeout.connect(self._commit_slider_measure)

        root = QVBoxLayout(self)
        self.view = FrameView()
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
        side.addWidget(QLabel("Película"))
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
        self.sld_frame.sliderPressed.connect(self._stop_play)
        self.sld_frame.sliderReleased.connect(self._load_frame)
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
        self.stack.addWidget(self._build_reveal())
        self.stack.addWidget(self._build_stab())
        self.stack.addWidget(self._build_color())
        self.stack.addWidget(self._build_crop())
        side.addWidget(self.stack, 1)

        buttons = QHBoxLayout()
        btn_reset = QPushButton("Restablecer revelado")
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
        wrap.setMinimumWidth(320)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(6)
        split.setChildrenCollapsible(False)
        split.addWidget(preview_col)
        split.addWidget(wrap)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([980, 340])
        root.addWidget(split, 1)

        self._load_sliders()
        self._set_task(0)
        if self._jobs is not None:
            self._jobs.progress.connect(self._on_job)
            self._jobs.settled.connect(self._on_job_done)
        if start_name in self._filenames:
            self.combo.setCurrentText(start_name)
        self.combo.currentTextChanged.connect(self._on_film)
        QTimer.singleShot(0, self._on_film)

    def settings(self) -> dict:
        return self._collect()

    def reference_name(self) -> str:
        return self.combo.currentText()

    def done(self, code: int):
        self._stop_play()
        self._save_current()
        super().done(code)

    def _build_reveal(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet("color: #888; font-size: 9pt;")
        layout.addWidget(self.lbl_note)
        layout.addWidget(QLabel("Dónde revelar"))
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("CPU", "cpu")
        self.combo_engine.addItem("GPU", "gpu")
        self.combo_engine.setToolTip(
            "CPU: demosaico del SDK y la curva en el procesador. "
            "GPU: demosaico NPP en la tarjeta, el promedio de vecinos del SDK. "
            "La curva, el color, la estabilización y el recorte van también en la tarjeta."
        )
        self.combo_engine.currentIndexChanged.connect(self._on_engine)
        layout.addWidget(self.combo_engine)
        self.lbl_engine = QLabel("")
        self.lbl_engine.setWordWrap(True)
        self.lbl_engine.setStyleSheet("color: #888; font-size: 9pt;")
        layout.addWidget(self.lbl_engine)
        self._rows = {}
        self._add_slider(layout, "gain_r", "Ganancia rojo", 50, 250, 100, "1,00 deja el rojo del visor ISP. El verde queda en 1.")
        self._add_slider(layout, "gain_b", "Ganancia azul", 50, 250, 100, "1,00 deja el azul del visor ISP. El verde queda en 1.")
        self._add_slider(layout, "ev", "Exposición (EV)", -200, 200, 100, "Pasos de diafragma, antes de la gamma.")
        self._add_slider(layout, "gamma", "Gamma", 50, 300, 100, "2,00 es la del visor de captura. Siempre se aplica. Un número más alto aclara.")
        self._add_slider(layout, "shadow_b", "Sombras", 0, 400, 10, "0 deja solo la gamma. Un número más alto abre las sombras antes de esa gamma.")
        self._add_slider(layout, "shadow_lift", "Levante de negro", 0, 200, 1000, "Sube el negro sin mover el blanco.")
        self._add_slider(layout, "midtone", "Detalle de medios tonos", 0, 150, 100, "Contraste de frecuencias medias, sin afilar el grano fino.")
        layout.addStretch(1)
        return page

    def _build_stab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Dibuja la zona alrededor de una perforación, sin el cuadro. "
            "El cuadro en el que dibujas queda como referencia de toda la estabilización. "
            "Reproducir muestra cómo quedó el desplazamiento. "
            "El análisis sigue si cierras esta ventana."
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
            "y el borde se corre cuadro a cuadro. La fuerza y la protección de sombras "
            "aflojan la corrección para que el negro no reviente."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.combo_grade = QComboBox()
        self.combo_grade.addItem("Color", "color")
        self.combo_grade.addItem("Blanco y negro", "bw")
        self.combo_grade.setToolTip(
            "Blanco y negro deja solo la saturación. Color mide GamMac y, además, puede bajar la saturación."
        )
        self.combo_grade.currentIndexChanged.connect(self._on_grade_mode)
        layout.addWidget(self.combo_grade)
        sat_head = QHBoxLayout()
        sat_title = QLabel("Saturación")
        sat_title.setToolTip("0 es gris. 100 deja el color como está. En blanco y negro parte en 0.")
        self.lbl_sat = QLabel("100")
        self.lbl_sat.setAlignment(Qt.AlignmentFlag.AlignRight)
        sat_head.addWidget(sat_title)
        sat_head.addWidget(self.lbl_sat)
        layout.addLayout(sat_head)
        self.sld_sat = QSlider(Qt.Orientation.Horizontal)
        self.sld_sat.setRange(0, 100)
        self.sld_sat.setValue(100)
        self.sld_sat.setToolTip(sat_title.toolTip())
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
        self._add_color_slider(
            form, "lo_th", "Umbral bajo", 0, 200, 100, False,
            "loTh, en porcentaje. 0,12 deja afuera el 0,12 % más oscuro del área.",
        )
        self._add_color_slider(
            form, "hi_th", "Umbral alto", 0, 200, 100, False,
            "hiTh, en porcentaje. Ignora las luces sueltas del área al medir.",
        )
        self._add_color_slider(
            form, "gam_max", "Tope de gamma", 10, 100, 10, False,
            "GamMax. Limita cuánto puede subir o bajar la gamma. 2,5 es suave. 10 es el tope del plugin.",
        )
        self._add_color_slider(
            form, "strength", "Fuerza", 0, 100, 100, True,
            "Cuánto de la corrección se aplica. 0,50 deja la mitad del cuadro original.",
        )
        self._add_color_slider(
            form, "rng_lim", "Rango mínimo", 1, 40, 1, False,
            "RngLim. Si el área casi no tiene contraste, no se estima gamma.",
        )
        self._add_color_slider(
            form, "omin", "Negro de salida", 0, 80, 1, True,
            "omin. El negro de la curva no baja de este código. 8 deja un piso.",
        )
        self._add_color_slider(
            form, "omax", "Blanco de salida", 175, 255, 1, True,
            "omax. El blanco de la curva no pasa de este código.",
        )
        self.chk_dark = QCheckBox("Proteger sombras")
        self.chk_dark.setToolTip("Por debajo de este nivel el píxel queda como está. Corta el ruido de color en el negro.")
        self.chk_dark.toggled.connect(self._on_apply_knob)
        form.addWidget(self.chk_dark)
        self._add_color_slider(
            form, "dark", "Nivel de sombras", 0, 120, 1, True,
            "DarkProtect, en código de 8 bits. 35 cubre el negro y el grano más oscuro.",
        )
        self.chk_high = QCheckBox("Proteger luces")
        self.chk_high.setToolTip("Por encima de este nivel el píxel queda como está.")
        self.chk_high.toggled.connect(self._on_apply_knob)
        form.addWidget(self.chk_high)
        self._add_color_slider(
            form, "high", "Nivel de luces", 140, 255, 1, True,
            "HighProtect, en código de 8 bits.",
        )
        self._add_color_slider(form, "red_mul", "Multiplicador rojo", 25, 400, 100, False, "RedMul. 1,00 no mueve el rojo.")
        self._add_color_slider(form, "grn_mul", "Multiplicador verde", 25, 400, 100, False, "GrnMul. 1,00 no mueve el verde.")
        self._add_color_slider(form, "blu_mul", "Multiplicador azul", 25, 400, 100, False, "BluMul. 1,00 no mueve el azul.")
        btn_preset = QPushButton("Dejar este ajuste como predeterminado")
        btn_preset.setToolTip(
            "Los clips que todavía no tienen color parten con estos controles y, si hay, con esta zona."
        )
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
            "Los siguientes se mueven y se agrandan o achican parejos, sin deformar. "
            "El borde negro de la estabilización tiene que quedar fuera."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.chk_crop = QCheckBox("Aplicar este recorte al exportar")
        self.chk_crop.toggled.connect(self._paint_preview)
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
        slot = self._on_apply_knob if apply_knob else self._touch_segment
        slider.valueChanged.connect(slot)
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
            if scale == 1:
                lbl.setText(f"{int(round(number))}")
            else:
                lbl.setText(f"{number:.2f}")

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

    def _host(self):
        host = self.parent()
        while host is not None:
            if hasattr(host, "_persist") and hasattr(host, "save_config"):
                return host
            host = host.parent()
        return None

    def _preset(self) -> dict:
        """Base de los clips nuevos. Si no hay una guardada, es el ajuste suave."""
        saved = {}
        host = self._host()
        if host is not None and isinstance(host._persist.get("gammac"), dict):
            saved = host._persist["gammac"]
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
        host = self._host()
        if host is None:
            self.lbl_color.setText("No pude guardar el predeterminado.")
            return
        values = self._color_values()
        values["lock"] = int(self.combo_lock.currentData())
        values["scale"] = int(self.combo_scale.currentData())
        segment = self._selected()
        values["rect"] = list(segment["rect"]) if segment and segment.get("rect") else None
        host._persist["gammac"] = values
        host.save_config()
        where = " y la zona" if values["rect"] else ""
        self.lbl_color.setText(f"Quedó como base para los clips que vengan{where}.")

    def _grade_setting(self) -> tuple[str, float]:
        grade = self._adjust.get("grade") if isinstance(self._adjust, dict) else None
        if isinstance(grade, dict) and grade.get("mode") in ("color", "bw"):
            mode = grade["mode"]
            try:
                sat = float(grade.get("sat", 0.0 if mode == "bw" else 1.0))
            except (TypeError, ValueError):
                sat = 0.0 if mode == "bw" else 1.0
            return mode, min(1.0, max(0.0, sat))
        mode = "bw" if _is_bw(self.combo.currentText(), self._meta) else "color"
        return mode, 0.0 if mode == "bw" else 1.0

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
        sat = self.sld_sat.value() / 100.0
        self._adjust["grade"] = {"mode": mode, "sat": sat}
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

    def _write_knobs(self, segment: dict):
        segment.update(self._color_values())
        segment["lock"] = int(self.combo_lock.currentData())
        segment["scale"] = int(self.combo_scale.currentData())

    def _stamp_apply(self, recipe: dict):
        knobs = self._measure_kwargs()
        recipe["strength"] = knobs["strength"]
        recipe["omin"] = knobs["omin"]
        recipe["omax"] = knobs["omax"]
        if knobs["dark"] is None:
            recipe.pop("dark", None)
        else:
            recipe["dark"] = knobs["dark"]
        if knobs["high"] is None:
            recipe.pop("high", None)
        else:
            recipe["high"] = knobs["high"]

    def _on_apply_knob(self, *_args):
        self._touch_segment()

    def _play_fps(self) -> int:
        name = self.combo.currentText()
        spins = getattr(self.parent(), "_fps", None)
        if isinstance(spins, dict) and name in spins:
            return max(1, int(spins[name].value()))
        try:
            return max(1, int((self._meta.get(name) or {}).get("fps", 18)))
        except (TypeError, ValueError):
            return 18

    def _toggle_play(self):
        if self._count <= 1:
            return
        self._playing = not self._playing
        self.view.set_playing(self._playing)
        if self._playing:
            self._advance()

    def _stop_play(self, *_args):
        if not self._playing:
            return
        self._playing = False
        self.view.set_playing(False)
        self._load_frame()

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
        self.lbl_frame.setText(f"Cuadro {nxt + 1}")
        self._refresh_bars()
        self._load_frame(fast=True)
        if not self._playing:
            return
        spent = (time.perf_counter() - started) * 1000.0
        wait = max(0, int(round(1000.0 / self._play_fps() - spent)))
        QTimer.singleShot(wait, self._advance)

    def _set_task(self, index: int):
        self._task = index
        self.stack.setCurrentIndex(index)
        for number, button in enumerate(self._task_btns):
            button.setChecked(number == index)
        self._timeline_host.setVisible(index in (2, 3))
        self._paint_preview()

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
        engine = self._settings.get("engine", "cpu")
        self.combo_engine.blockSignals(True)
        self.combo_engine.setCurrentIndex(1 if engine == "gpu" else 0)
        self.combo_engine.blockSignals(False)
        self._refresh_labels()
        self._refresh_engine_note()

    def _on_engine(self, *_args):
        self._refresh_engine_note()
        self._save_current()
        if self._lin_prev is not None:
            self._load_frame()

    def _refresh_engine_note(self, *_args):
        if self.combo_engine.currentData() == "gpu":
            self.lbl_engine.setText(
                "Esta vista y la exportación usan el mismo NPP: promedia vecinos, como el SDK. "
                "La curva, GamMac, traslación y recorte van en la tarjeta. "
                "Al pausar se ve el cuadro de la exportación. Gamma 2,00, ganancia 1,00 y sombras 0 igualan al visor."
            )
        else:
            self.lbl_engine.setText(
                "Demosaico del SDK en el CPU, como el visor ISP. "
                "Gamma 2,00, ganancia 1,00 y sombras 0 dejan ese tono. "
                "El HEVC lossless codifica con NVENC."
            )

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
        raw["engine"] = self.combo_engine.currentData() or "cpu"
        return reveal_pipeline.coerce_reveal(raw)

    def _schedule(self, *_args):
        self._refresh_labels()
        self._mark_recipes_stale()
        self._timer.start()

    def _on_reveal_timer(self):
        self._save_current()
        self._paint_preview()

    def _reset(self):
        engine = self._collect().get("engine", "cpu")
        self._settings = reveal_pipeline.default_reveal()
        self._settings["engine"] = engine
        self._load_sliders()
        self._mark_recipes_stale()
        self._save_current()
        self._paint_preview()

    def _on_film(self, *_args):
        self._measure_timer.stop()
        self._stop_play()
        name = self.combo.currentText()
        if self._current and self._current != name:
            self._save_current()
        self._current = name
        self._load_adjust()
        if not name:
            return
        path = self._folder / name
        if not path.is_file() or not _is_bayer(name, self._meta):
            self._lin_prev = None
            self.sld_frame.setRange(0, 0)
            self.lbl_note.setText("Esta película no es Bayer.")
            self.view.set_view(np.zeros((32, 32, 3), np.uint8))
            return
        frame_bytes = int(FIXED_W * FIXED_H * 1.5)
        count = path.stat().st_size // frame_bytes
        self._count = max(1, count)
        self._expand_single_frame_segment()
        self._refresh_bars()
        if count <= 0:
            self._lin_prev = None
            return
        self.sld_frame.blockSignals(True)
        self.sld_frame.setRange(0, count - 1)
        self.sld_frame.setValue(min(100, count - 1))
        self.sld_frame.blockSignals(False)
        note = "El revelado, el color, la perforación y el recorte son de este archivo."
        if _is_bw(name, self._meta):
            note = "Esta película es blanco y negro. " + note
        self.lbl_note.setText(note)
        self._load_frame()

    def _on_frame_label(self, value: int):
        self.lbl_frame.setText(f"Cuadro {value + 1}")
        self._follow_span()
        self._refresh_bars()
        if not self.sld_frame.isSliderDown():
            self._frame_timer.start()

    def _load_frame(self, fast: bool = False):
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
                return
            if fast:
                lin = reveal_pipeline.debayer_playback(packed, FIXED_W, FIXED_H)
                self._preview_wh = (int(lin.shape[1]), int(lin.shape[0]))
                self._sigma_scale = self._preview_wh[0] / float(FIXED_W)
                self._lin_prev = lin
            else:
                engine = self.combo_engine.currentData() or "cpu"
                lin = reveal_pipeline.debayer_linear(packed, FIXED_W, FIXED_H, engine)
                preview_w = min(_PREVIEW_W, FIXED_W)
                self._sigma_scale = preview_w / float(FIXED_W)
                small_h = max(1, int(round(FIXED_H * self._sigma_scale)))
                self._preview_wh = (preview_w, small_h)
                self._lin_prev = cv2.resize(lin, (preview_w, small_h), interpolation=cv2.INTER_AREA)
        except Exception as exc:
            self._lin_prev = None
            self.lbl_note.setText(f"No se pudo leer el cuadro. {exc}")
            return
        self._paint_preview()

    def _paint_preview(self, *_args):
        if self._lin_prev is None:
            return
        base = reveal_pipeline.apply_reveal(self._lin_prev, self._collect(), self._sigma_scale)
        index = int(self.sld_frame.value())
        mode, sat = self._grade_setting()
        recipe = self._recipe_for_preview(base, index) if mode == "color" and self._task in (2, 3) else None
        graded = gammac.apply(base, recipe) if recipe else base
        graded = gammac.saturate(graded, sat)
        base = gammac.saturate(base, sat)
        playing = self._playing
        if self._task == 1:
            shown_img = self._warp_preview(base)
            shown = self._to_u8(shown_img)
            overlay = None if playing else self._preview_rect(self._roi)
            color = "#ff9f1a"
            side = None
            allow = not playing
        elif self._task == 2:
            shown_img = self._warp_preview(graded) if playing else graded
            shown = self._to_u8(shown_img)
            overlay = None if playing else self._preview_rect(self._segment_rect())
            color = "#ffcc00"
            side = None
            allow = not playing
        elif self._task == 3:
            self._sync_crop_view()
            shown_img = self._warp_preview(graded)
            if playing and self.chk_crop.isChecked():
                rect = self._preview_rect(self._crop)
                if rect and rect[2] > 2 and rect[3] > 2:
                    x, y, width, height = rect
                    shown_img = shown_img[y:y + height, x:x + width]
                if self._flip_on():
                    shown_img = _hflip(shown_img)
                shown = self._to_u8(shown_img)
                overlay = None
                side = None
                allow = False
                aspect = None
            else:
                side = self._crop_side(shown_img)
                overlay = self._preview_rect(self._shown_crop())
                if self._flip_on():
                    shown_img = _hflip(shown_img)
                shown = self._to_u8(shown_img)
                allow = True
                aspect = self._crop_aspect()
            color = "#4cc9f0"
        else:
            shown = self._to_u8(base)
            overlay = None
            color = "#ffffff"
            side = None
            allow = False
            aspect = None
        if self._task != 3:
            aspect = None
        self.view.set_view(shown, overlay, color, side, allow, aspect=aspect)
        self._refresh_bars()
        self._refresh_crop_label()

    def _segment_at(self, index: int):
        for item in self._adjust["segments"]:
            if int(item["in"]) <= index <= int(item["out"]):
                return item
        return None

    def _recipe_for_preview(self, revealed, index: int):
        """La gamma medida se reusa en todo el tramo. No se vuelve a calcular al mover el cuadro."""
        owner = self._segment_at(index)
        if owner and owner.get("recipe") and not owner.get("dirty"):
            return owner["recipe"]
        selected = self._selected()
        if self._task == 2 and selected is owner and selected and selected.get("rect"):
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
        if self._task != 3:
            return 0.0
        try:
            return float(self._adjust.get("turn") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _spin_center(self, width: int, height: int) -> tuple[float, float]:
        if self._crop:
            x, y, rect_w, rect_h = self._crop
            sx = width / float(FIXED_W)
            sy = height / float(FIXED_H)
            return ((x + rect_w / 2.0) * sx, (y + rect_h / 2.0) * sy)
        return (width / 2.0, height / 2.0)

    def _warp_preview(self, image):
        tx, ty = 0.0, 0.0
        if self._shifts:
            index = int(self.sld_frame.value())
            if index < len(self._shifts[0]):
                tx = self._shifts[0][index] * self._sigma_scale
                ty = self._shifts[1][index] * self._sigma_scale
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
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _crop_side(self, image):
        rect = self._preview_rect(self._crop) if self.chk_crop.isChecked() else None
        if not rect:
            return None
        x, y, width, height = rect
        crop = image[y:y + height, x:x + width]
        if self._flip_on():
            crop = _hflip(crop)
        return self._to_u8(crop)

    def _to_u8(self, image) -> np.ndarray:
        return np.clip(image * np.float32(255.0) + np.float32(0.5), 0, 255).astype(np.uint8)

    def _preview_rect(self, rect):
        if not rect:
            return None
        sx = self._preview_wh[0] / float(FIXED_W)
        sy = self._preview_wh[1] / float(FIXED_H)
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
        sx = FIXED_W / float(self._preview_wh[0])
        sy = FIXED_H / float(self._preview_wh[1])
        return (
            int(round(rect[0] * sx)),
            int(round(rect[1] * sy)),
            max(8, int(round(rect[2] * sx))),
            max(8, int(round(rect[3] * sy))),
        )

    def _on_rect(self, rect):
        full = self._full_rect(rect)
        if self._task == 1:
            self._roi = full
            self._roi_ref = int(self.sld_frame.value())
            self.chk_perfo.setChecked(True)
            self.lbl_perfo.setText(
                f"Zona {full[2]}×{full[3]} en {full[0]}, {full[1]}."
                f" Referencia: cuadro {self._roi_ref + 1}."
            )
            self._save_current()
        elif self._task == 3:
            if self._flip_on():
                full = film_finish.mirror_x(full, FIXED_W)
            self._apply_crop_rect(full)
        elif self._task == 2:
            segment = self._ensure_segment()
            segment["rect"] = list(full)
            segment["dirty"] = True
            segment.pop("recipe", None)
            self._play_recipe = None
            self._draft_id = None
            self.lbl_color.setText("Vista previa del área. Medir la guarda para la exportación.")
        self._paint_preview()

    def _load_adjust(self):
        path = self._folder / (self._current or "")
        self._adjust = film_finish.load(path) if self._current else film_finish.empty()
        self._apply_file_reveal()
        perfo = self._adjust.get("perfo") or {}
        self._roi = tuple(perfo["roi"]) if perfo.get("roi") else None
        self._roi_ref = film_finish.perfo_ref(perfo)
        self.chk_perfo.blockSignals(True)
        self.chk_perfo.setChecked(bool(perfo.get("enabled")))
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
        self._load_shifts()
        self._sync_segment_controls()
        self._sync_grade_controls()
        self._sync_turn()
        self._refresh_bars()
        self._refresh_review()

    def _save_current(self):
        if not self._current:
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
        if getattr(self, "_rows", None):
            self._adjust["reveal"] = self._collect()
        film_finish.save(self._folder / self._current, self._adjust)
        self._refresh_review()

    def _on_horizontal(self, _on):
        self._save_current()
        self._paint_preview()

    def _apply_file_reveal(self):
        """Los sliders muestran el revelado guardado en este archivo."""
        saved = self._adjust.get("reveal") if isinstance(self._adjust, dict) else None
        source = saved if isinstance(saved, dict) else self._reveal_fallback
        self._settings = reveal_pipeline.coerce_reveal(source)
        if getattr(self, "_rows", None):
            self._load_sliders()

    def _load_shifts(self):
        self._shifts = None
        self._pad = 0
        if not self._current:
            return
        side_path = film_finish.perfo_path(self._folder / self._current)
        if not side_path.is_file():
            self._say_perfo()
            return
        try:
            import json
            data = json.loads(side_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if data.get("status") == "done" and data.get("tx"):
            self._shifts = (data["tx"], data["ty"])
            self._pad = int(data.get("pad") or 0)
        self._say_perfo(data.get("status"), data.get("message", ""))

    def _say_perfo(self, status=None, message=""):
        if self._roi is None:
            self.lbl_perfo.setText("Sin zona.")
            return
        ref = f" Referencia: cuadro {self._roi_ref + 1}." if self._roi_ref is not None else ""
        if status == "done":
            text = f"Lista. Margen negro de {self._pad} px.{ref}"
        elif status == "running":
            text = f"En curso. {message}{ref}"
        elif status == "error":
            text = (message or "Sin perforación.") + ref
        else:
            text = f"Zona guardada. Falta analizar.{ref}"
        self.lbl_perfo.setText(text)

    def _analyze_one(self):
        self._enqueue(self._current, force=True)

    def _analyze_all(self):
        self._save_current()
        queued = 0
        for name in self._filenames:
            if name == self._current and self._roi:
                zone = self._roi
            else:
                data = film_finish.load(self._folder / name)
                zone = (data.get("perfo") or {}).get("roi")
            if not zone:
                continue
            if film_finish.perfo_is_done(self._folder / name, zone):
                continue
            if self._enqueue(name, zone):
                queued += 1
        if queued == 0:
            self.lbl_perfo.setText("No queda nada por medir. Los que ya están listos se dejan.")

    def _saved_ref(self, name: str, data=None):
        """Cuadro en el que se dibujó el rectángulo. None si ese archivo no lo tiene guardado."""
        if name == self._current and self._roi_ref is not None:
            return int(self._roi_ref)
        if data is None and name:
            data = film_finish.load(self._folder / name)
        return film_finish.perfo_ref((data or {}).get("perfo") or {})

    def _job_ref(self, name: str, data=None) -> int:
        ref = self._saved_ref(name, data)
        if ref is None and name == self._current:
            return int(self.sld_frame.value())
        return 0 if ref is None else int(ref)

    def _enqueue(self, name, roi=None, force: bool = False):
        if not name:
            return
        zone = roi or self._roi
        if not zone:
            self.lbl_perfo.setText("Dibuja primero la zona de la perforación.")
            return
        if self._jobs is None:
            from adjust_jobs import AdjustJobs
            self._jobs = AdjustJobs(self)
            self._jobs.progress.connect(self._on_job)
            self._jobs.settled.connect(self._on_job_done)
        if name == self._current:
            if self._roi_ref is None:
                self._roi_ref = int(self.sld_frame.value())
            self.chk_perfo.setChecked(True)
            self._save_current()
        else:
            data = film_finish.load(self._folder / name)
            perfo = data.get("perfo") or {}
            perfo["enabled"] = True
            perfo["roi"] = list(zone)
            known = self._saved_ref(name, data)
            if known is not None:
                perfo["ref"] = int(known)
            data["perfo"] = perfo
            film_finish.save(self._folder / name, data)
        queued = self._jobs.enqueue(
            self._folder / name, zone, self._job_ref(name), force=force,
        )
        if name == self._current:
            if queued:
                self.lbl_perfo.setText("En la cola. Puedes cambiar de tarea o cerrar.")
            elif not force:
                self.lbl_perfo.setText("Este archivo ya está medido.")
        return bool(queued)

    def _on_job(self, name, done, total, _state):
        if name == self._current:
            self.lbl_perfo.setText(f"En curso, {done} de {total}.")
            percent = int(round(100 * done / total)) if total else 0
            self._refresh_review(percent)

    def _on_job_done(self, name, ok, message):
        if name == self._current:
            self._load_shifts()
            if not ok:
                self.lbl_perfo.setText(message or "Sin perforación.")
            self._paint_preview()
            self._refresh_review()

    def _refresh_review(self, percent=None):
        if not hasattr(self, "_badge_lay"):
            return
        if not self._current:
            _paint_badges(self._badge_lay, [])
            return
        state = film_finish.review_state(self._folder / self._current)
        if percent is not None and state.get("stab") != "listo":
            state["stab"] = "corriendo"
            state["percent"] = int(percent)
        _paint_badges(self._badge_lay, film_finish.review_badges(state))

    def _selected(self):
        if self._segment < 0 or self._segment >= len(self._adjust["segments"]):
            return None
        return self._adjust["segments"][self._segment]

    def _segment_rect(self):
        segment = self._selected()
        if not segment or not segment.get("rect"):
            return None
        return tuple(segment["rect"])

    def _ensure_segment(self):
        spans = self._covered_segments()
        frame = int(self.sld_frame.value())
        for index, item in enumerate(spans):
            if int(item["in"]) <= frame <= int(item["out"]):
                self._segment = index
                break
        else:
            self._segment = 0
        self._adjust["segments"] = spans
        self._sync_segment_controls()
        return spans[self._segment]

    def _expand_single_frame_segment(self):
        """Un tramo de un solo cuadro era el defecto anterior: al abrirlo, pasa a cubrir el rollo."""
        items = self._adjust["segments"]
        if len(items) != 1:
            return
        segment = items[0]
        if int(segment["in"]) != int(segment["out"]):
            return
        segment["in"] = 0
        segment["out"] = max(0, self._count - 1)

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
        if self._task == 3:
            self._crop_i = int(index)
            self._sync_crop_view()
        else:
            self._segment = int(index)
            self._sync_segment_controls()
        spans = self._covered_crops() if self._task == 3 else self._covered_segments()
        if 0 <= index < len(spans):
            start, end = int(spans[index]["in"]), int(spans[index]["out"])
            frame = int(self.sld_frame.value())
            if frame < start or frame > end:
                self.sld_frame.setValue(start)
                self._load_frame()
        self._refresh_bars()

    def _on_timeline_edit(self, spans, selected: int):
        if self._task == 3:
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
        if self._task == 3:
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
        if self._task == 3:
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
        if self._task == 3:
            spans = self._covered_crops()
            for index, span in enumerate(spans):
                if int(span["in"]) <= frame <= int(span["out"]) and index != self._crop_i:
                    self._crop_i = index
                    self._sync_crop_view()
                    return
        elif self._task == 2:
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
            rect = film_finish.snap_crop(full, FIXED_W, FIXED_H)
            if rect is None:
                self.lbl_crop.setText("Ese recorte no entra. Márcalo de nuevo.")
                return
            film_finish.refit_crops(spans, master, rect, FIXED_W, FIXED_H)
        else:
            aspect = float(master[2]) / float(master[3])
            rect = film_finish.fit_aspect(full, aspect, FIXED_W, FIXED_H)
            if rect is None:
                return
            spans[index]["crop"] = list(rect)
        self._adjust["crop_spans"] = spans
        self._crop_i = index
        self.chk_crop.setChecked(True)
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
        self.chk_flip.blockSignals(True)
        self.chk_flip.setChecked(bool(spans[index].get("flip_h")))
        self.chk_flip.blockSignals(False)

    def _select_segment(self, index: int):
        self._on_timeline_pick(index)

    def _sync_segment_controls(self):
        segment = self._selected()
        if segment is None:
            self._apply_preset_to_controls(self._preset())
            return
        values = dict(_NEUTRAL)
        values.update({key: segment[key] for key in _NEUTRAL if key in segment})
        lock = int(segment.get("lock", 1))
        scale = int(segment.get("scale", 2))
        self.combo_lock.blockSignals(True)
        self.combo_scale.blockSignals(True)
        self.combo_lock.setCurrentIndex(max(0, self.combo_lock.findData(lock)))
        self.combo_scale.setCurrentIndex(max(0, self.combo_scale.findData(scale)))
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

    def _commit_slider_measure(self):
        self._run_measure(allow_sample=True)

    def _delete_segment(self):
        if self._segment < 0:
            return
        del self._adjust["segments"][self._segment]
        self._segment = min(self._segment, len(self._adjust["segments"]) - 1)
        self._play_recipe = None
        self._draft_id = None
        self._sync_segment_controls()
        self._paint_preview()

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
        revealed = self._reveal_full(frame)
        if revealed is None:
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
        self._refresh_bars()
        self.lbl_color.setText(recipe.get("message") or "Medido.")
        self._save_current()
        self._paint_preview()

    def _reveal_full(self, index: int):
        path = self._folder / self.combo.currentText()
        frame_bytes = int(FIXED_W * FIXED_H * 1.5)
        try:
            with open(path, "rb") as handle:
                handle.seek(index * frame_bytes)
                packed = handle.read(frame_bytes)
            if len(packed) < frame_bytes:
                return None
            engine = self.combo_engine.currentData() or "cpu"
            lin = reveal_pipeline.debayer_linear(packed, FIXED_W, FIXED_H, engine)
            revealed = reveal_pipeline.apply_reveal(lin, self._collect(), 1.0)
            mode, _sat = self._grade_setting()
            if mode == "bw":
                revealed = reveal_pipeline.to_bw(revealed)
            return revealed
        except Exception as exc:
            self.lbl_color.setText(f"No se pudo medir. {exc}")
            return None

    def _mark_recipes_stale(self):
        for segment in self._adjust["segments"]:
            segment.pop("recipe", None)
            segment["dirty"] = True

    def _refresh_bars(self):
        if not hasattr(self, "bar"):
            return
        if self._task == 3:
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
            role = "Este tramo fija la proporción y el tamaño del video."
        else:
            role = "Misma proporción que el primero: se mueve y se escala parejo."
        text = (
            f"Recorte {self._crop[2]}×{self._crop[3]} desde {self._crop[0]}, {self._crop[1]}. "
            + role
        )
        if self._pad:
            text += f" La perforación deja {self._pad} px de negro."
        turn = self.turn.degrees()
        if abs(turn) >= 0.05:
            text += " Giro " + f"{turn:.1f}°".replace(".", ",") + "."
        if self.chk_flip.isChecked():
            text += " Volteado horizontal."
        self.lbl_crop.setText(text)

    def _flip_on(self) -> bool:
        return self.chk_flip.isChecked()

    def _shown_crop(self):
        if not self._crop or not self._flip_on():
            return self._crop
        return film_finish.mirror_x(self._crop, FIXED_W)

    def _on_flip(self, *_args):
        spans = self._covered_crops()
        if spans:
            index = min(max(self._crop_i, 0), len(spans) - 1)
            spans[index]["flip_h"] = self.chk_flip.isChecked()
            self._adjust["crop_spans"] = spans
        self._save_current()
        self._paint_preview()

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
