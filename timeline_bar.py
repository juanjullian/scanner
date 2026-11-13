"""Pista de tramos. Una sola línea, siempre cubierta, con cabeza de lectura.

El gesto es el de un editor: la regla mueve la cabeza, el borde se arrastra
cuadro a cuadro, y un rectángulo dibujado parte el tramo de abajo. No es un
editor de varias pistas.
"""
from __future__ import annotations

import copy

import film_finish
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygon
from PyQt6.QtWidgets import QPushButton, QScrollBar, QWidget


def _timeline_button(text: str, tip: str, slot) -> QPushButton:
    button = QPushButton(text)
    button.setFixedHeight(22)
    button.setStyleSheet("padding: 0 8px; font-size: 8pt;")
    button.setToolTip(tip)
    button.clicked.connect(slot)
    return button


class TimelineBar(QWidget):
    head_moved = pyqtSignal(int)
    picked = pyqtSignal(int)
    edited = pyqtSignal(object, int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(92)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._count = 1
        self._spans = []
        self._selected = -1
        self._head = 0
        self._ppf = 1.0
        self._origin = 0.0
        self._user_zoom = False
        self._drag = None
        self._press_frame = 0
        self._press_span = 0
        self._press_x = 0.0
        self._head_drag = 0
        self._armed_edge = 0
        self._keep = None
        self._scroll = QScrollBar(Qt.Orientation.Horizontal, self)
        self._scroll.setRange(0, 0)
        self._scroll.valueChanged.connect(self._on_scroll)

    def set_bar(self, count, spans, selected, head, keep=None):
        if self._drag:
            self._head = int(head)
            self._store_keep(keep, self._count)
            self.update()
            return
        count = max(1, int(count))
        if count != self._count:
            self._count = count
            self._user_zoom = False
        self._spans = [copy.deepcopy(item) for item in (spans or [])]
        self._selected = int(selected)
        self._head = max(0, min(int(head), self._count - 1))
        self._store_keep(keep, self._count)
        if not self._user_zoom:
            self._ppf = self._fit_ppf()
            self._origin = 0.0
        else:
            self._ppf = min(48.0, max(self._fit_ppf(), self._ppf))
            self._clamp_origin()
        self._sync_scroll()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scroll.setGeometry(0, self.height() - 16, self.width(), 16)
        if not self._user_zoom:
            self._ppf = self._fit_ppf()
            self._origin = 0.0
        else:
            self._clamp_origin()
        self._sync_scroll()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1c1c1c"))
        lane = self._lane()
        ruler = lane.adjusted(0, -18, 0, -lane.height())
        painter.fillRect(ruler, QColor("#2a2a2a"))
        painter.fillRect(lane, QColor("#141414"))
        if not self._spans:
            self._paint_keep(painter, lane)
            self._paint_head(painter, lane, ruler)
            return
        for index, span in enumerate(self._spans):
            x0 = self._x_of(int(span["in"]))
            x1 = self._x_of(int(span["out"]) + 1)
            color = QColor("#0078d7") if index == self._selected else QColor("#3d6f99")
            painter.fillRect(int(x0), lane.top() + 6, max(2, int(x1 - x0)), lane.height() - 12, color)
            if index > 0:
                painter.setPen(QPen(QColor("#f2f2f2"), 2))
                edge = int(round(x0))
                painter.drawLine(edge, lane.top() + 4, edge, lane.bottom() - 4)
        self._paint_ticks(painter, ruler)
        if self._drag == "carve":
            a = self._x_of(min(self._press_frame, self._head_drag))
            b = self._x_of(max(self._press_frame, self._head_drag) + 1)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawRect(int(min(a, b)), lane.top() + 4, max(1, int(abs(b - a))), lane.height() - 8)
        self._paint_keep(painter, lane)
        self._paint_head(painter, lane, ruler)

    def _store_keep(self, keep, count: int):
        self._keep = None
        if keep is None or int(count) <= 1:
            return
        lo = max(0, min(int(keep[0]), int(count) - 1))
        hi = max(lo, min(int(keep[1]), int(count) - 1))
        if lo > 0 or hi < int(count) - 1:
            self._keep = (lo, hi)

    def _paint_keep(self, painter: QPainter, lane):
        if not self._keep:
            return
        lo, hi = self._keep
        shade = QColor(0, 0, 0, 150)
        if lo > 0:
            x0 = self._x_of(0)
            x1 = self._x_of(lo)
            painter.fillRect(int(x0), lane.top(), max(0, int(x1 - x0)), lane.height(), shade)
        if hi < self._count - 1:
            x0 = self._x_of(hi + 1)
            x1 = self._x_of(self._count)
            painter.fillRect(int(x0), lane.top(), max(0, int(x1 - x0)), lane.height(), shade)

    def zoom_by(self, factor: float, anchor_x: float | None = None):
        """Acerca o aleja. Sin ancla, el centro de la pista se queda quieto."""
        if anchor_x is None:
            anchor_x = 8 + self._view_width() / 2.0
        frame = self._frame_at(anchor_x)
        fit = self._fit_ppf()
        self._ppf = min(48.0, max(fit, self._ppf * float(factor)))
        if self._ppf <= fit * 1.02:
            self.fit_film()
            return
        self._user_zoom = True
        self._origin = frame - (float(anchor_x) - 8) / self._ppf
        self._clamp_origin()
        self._sync_scroll()
        self.update()

    def fit_film(self):
        """El rollo entero, de borde a borde."""
        self._user_zoom = False
        self._ppf = self._fit_ppf()
        self._origin = 0.0
        self._sync_scroll()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        event.accept()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1.18 if delta > 0 else 1 / 1.18, event.position().x())
            return
        if not self._user_zoom:
            return
        step = max(1.0, self._view_width() / self._ppf * 0.12)
        self._origin += -step if delta > 0 else step
        self._clamp_origin()
        self._sync_scroll()
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = -1 if key == Qt.Key.Key_Left else 1
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._nudge_edge(step)
            else:
                self._set_head(self._head + step, emit=True)
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        x = event.position().x()
        frame = self._frame_at(x)
        self._press_x = x
        self._press_frame = frame
        if event.position().y() < self._lane().top():
            self._drag = "head"
            self._set_head(frame, emit=True)
            return
        edge = self._edge_at(x)
        if edge > 0:
            self._drag = "edge"
            self._armed_edge = edge
            self._move_edge_to(frame)
            return
        self._press_span = self._index_at(frame)
        self._drag = "down"
        self._head_drag = frame
        self._set_head(frame, emit=True)
        if self._press_span >= 0:
            self._selected = self._press_span
            self.picked.emit(self._press_span)
        self.update()

    def mouseMoveEvent(self, event):
        x = event.position().x()
        frame = self._frame_at(x)
        if self._drag == "head":
            self._set_head(frame, emit=True)
            return
        if self._drag == "edge":
            self._move_edge_to(frame)
            return
        if self._drag == "down":
            if abs(x - self._press_x) > 4:
                self._drag = "carve"
            self._head_drag = frame
            self.update()
            return
        if self._drag == "carve":
            self._head_drag = frame
            self.update()
            return
        edge = self._edge_at(x)
        if edge > 0:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        elif event.position().y() < self._lane().top():
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        drag = self._drag
        self._drag = None
        if drag == "carve":
            start = self._press_frame
            end = self._frame_at(event.position().x())
            spans, selected, changed = film_finish.carve(self._spans, start, end)
            if changed:
                self._spans = spans
                self._selected = selected
                self.edited.emit(copy.deepcopy(spans), selected)
        elif drag == "edge":
            self.edited.emit(copy.deepcopy(self._spans), self._selected)
        self.update()

    def _paint_head(self, painter: QPainter, lane, ruler):
        hx = int(round(self._x_of(self._head)))
        painter.setPen(QPen(QColor("#f2f2f2"), 2))
        painter.setBrush(QColor("#f2f2f2"))
        painter.drawLine(hx, ruler.top() + 8, hx, lane.bottom())
        painter.drawPolygon(QPolygon([
            QPoint(hx - 5, ruler.top()),
            QPoint(hx + 5, ruler.top()),
            QPoint(hx, ruler.top() + 8),
        ]))

    def _paint_ticks(self, painter: QPainter, ruler):
        painter.setPen(QColor("#9a9a9a"))
        step = 1
        while step * self._ppf < 56:
            step *= 2
        frame = int(self._origin // step) * step
        while frame < self._count:
            x = self._x_of(frame)
            if 0 <= x <= self.width():
                painter.drawLine(int(x), ruler.bottom() - 6, int(x), ruler.bottom())
                painter.drawText(int(x) + 3, ruler.top() + 12, str(frame + 1))
            frame += step

    def _nudge_edge(self, step: int):
        edge = self._armed_edge
        if edge <= 0 and 0 <= self._selected < len(self._spans) - 1:
            edge = self._selected + 1
        if edge <= 0 or edge >= len(self._spans):
            if self._selected > 0:
                edge = self._selected
            else:
                return
        frame = int(self._spans[edge]["in"]) + step
        self._armed_edge = edge
        self._move_edge_to(frame)
        self.edited.emit(copy.deepcopy(self._spans), self._selected)

    def _move_edge_to(self, frame: int):
        spans, _changed = film_finish.move_edge(self._spans, self._armed_edge, frame)
        self._spans = spans
        self.update()

    def _set_head(self, frame: int, emit: bool):
        frame = max(0, min(int(frame), self._count - 1))
        if frame == self._head and not emit:
            return
        self._head = frame
        self._reveal(frame)
        self.update()
        if emit:
            self.head_moved.emit(frame)

    def _reveal(self, frame: int):
        left = self._origin
        right = self._origin + max(1, self._view_width()) / self._ppf
        if frame < left:
            self._origin = frame
        elif frame > right - 1:
            self._origin = frame - (self._view_width() / self._ppf) + 1
        self._clamp_origin()
        self._sync_scroll()

    def _on_scroll(self, value: int):
        if self._drag:
            return
        self._origin = float(value)
        self.update()

    def _sync_scroll(self):
        span = max(0, int(round(self._count - self._view_width() / max(self._ppf, 0.01))))
        self._scroll.blockSignals(True)
        self._scroll.setRange(0, span)
        self._scroll.setValue(max(0, min(span, int(round(self._origin)))))
        self._scroll.setVisible(span > 0)
        self._scroll.blockSignals(False)

    def _fit_ppf(self) -> float:
        return max(1e-4, self._view_width() / float(max(1, self._count)))

    def _view_width(self) -> int:
        return max(1, self.width() - 16)

    def _lane(self):
        return QRect(8, 22, self._view_width(), max(24, self.height() - 16 - 26))

    def _x_of(self, frame: float) -> float:
        return 8 + (float(frame) - self._origin) * self._ppf

    def _frame_at(self, x: float) -> int:
        frame = int((float(x) - 8) / max(self._ppf, 0.01) + self._origin)
        return max(0, min(self._count - 1, frame))

    def _index_at(self, frame: int) -> int:
        for index, span in enumerate(self._spans):
            if int(span["in"]) <= frame <= int(span["out"]):
                return index
        return -1

    def _edge_at(self, x: float) -> int:
        for index, span in enumerate(self._spans):
            if index == 0:
                continue
            if abs(x - self._x_of(int(span["in"]))) <= 6:
                return index
        return 0

    def _clamp_origin(self):
        width_frames = self._view_width() / max(self._ppf, 0.01)
        max_origin = max(0.0, self._count - width_frames)
        self._origin = min(max(0.0, self._origin), max_origin)
