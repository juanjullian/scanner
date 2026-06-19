import sys
import os
import ctypes
import queue
import struct
import shutil
import numpy as np
import subprocess
import cv2
from datetime import datetime
from pathlib import Path
import json
import qoi_utils
import bayer_render
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QListWidget, QInputDialog, QMessageBox, 
                             QSplitter, QGroupBox, QProgressBar, QTabWidget, QSlider, QFileDialog,
                             QDoubleSpinBox, QSpinBox, QProgressDialog, QDialog, QCheckBox, 
                             QComboBox, QMenu, QListWidgetItem, QLineEdit, QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QRect, QEvent
from PyQt6.QtGui import QImage, QPixmap, QAction, QPainter, QColor, QFont, QIcon, QPen

# Importamos el núcleo del scanner
import scanner_core
from export_grade import ExportConfirmDialog, default_settings_for_file


def _windows_set_execution_state(display_required: bool, system_required: bool) -> None:
    """SetThreadExecutionState: pantalla y/o suspensión por inactividad del sistema."""
    if os.name != "nt":
        return
    ES_CONTINUOUS = 0x80000000
    ES_DISPLAY_REQUIRED = 0x00000002
    ES_SYSTEM_REQUIRED = 0x00000001
    try:
        flags = ES_CONTINUOUS
        if display_required:
            flags |= ES_DISPLAY_REQUIRED
        if system_required:
            flags |= ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


# --- PANTALLA DE CARGA (SPLASH SCREEN) ---
class IntroSplash(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(500, 300)
        
        # Layout principal con fondo oscuro y bordes redondeados
        layout = QVBoxLayout(self)
        self.container = QWidget()
        self.container.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border: 2px solid #333;
                border-radius: 15px;
            }
        """)
        inner_layout = QVBoxLayout(self.container)
        
        # Título
        lbl_title = QLabel("LUCID SCANNER SUITE")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #0078d7; font-size: 24px; font-weight: bold; border: none;")
        
        lbl_subtitle = QLabel("Archivo de La Unión")
        lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_subtitle.setStyleSheet("color: #888; font-size: 14px; border: none;")
        
        # Icono o Spinner (Texto simulado por ahora)
        self.lbl_status = QLabel("Iniciando sistema...")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #fff; font-size: 12px; margin-top: 20px; border: none;")
        
        # Barra de progreso
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #333;
                height: 4px;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
                border-radius: 2px;
            }
        """)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0) # Modo "infinito" (loading)

        self.lbl_detail = QLabel("")
        self.lbl_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet("color: #aaa; font-size: 10pt; margin: 8px 16px 0 16px; border: none;")
        self.lbl_detail.hide()

        self.btn_continue = QPushButton("Continuar en modo visor")
        self.btn_continue.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                font-weight: bold;
                padding: 10px 24px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #1084e3; }
        """)
        self.btn_continue.hide()
        self.btn_continue.clicked.connect(self.finish_loading)
        
        inner_layout.addStretch()
        inner_layout.addWidget(lbl_title)
        inner_layout.addWidget(lbl_subtitle)
        inner_layout.addStretch()
        inner_layout.addWidget(self.lbl_status)
        inner_layout.addWidget(self.lbl_detail)
        inner_layout.addWidget(self.progress)
        inner_layout.addWidget(self.btn_continue, 0, Qt.AlignmentFlag.AlignCenter)
        inner_layout.addSpacing(20)
        
        layout.addWidget(self.container)
        
        # Timer para simular pasos (y dar tiempo a la UI de cargar)
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_initialization)
        self.main_window = None
        self.steps = 0

    def start_loading(self):
        self.show()
        # Iniciamos la carga real de la app en el siguiente ciclo del event loop
        QTimer.singleShot(100, self.initialize_app)

    def initialize_app(self):
        self.lbl_status.setText("Cargando módulos...")
        try:
            self.main_window = MainWindow(defer_camera=True)
            self._cam_found = scanner_core.probe_camera_available(max_wait_sec=0)
            self.timer.start(500)
        except Exception as e:
            self.lbl_status.setText(f"Error fatal: {e}")
            self.lbl_status.setStyleSheet("color: red; border: none;")

    def check_initialization(self):
        self.steps += 1

        if self.steps == 1:
            self.lbl_status.setText("Buscando cámara Lucid...")
        elif self.steps <= 6:
            if not self._cam_found:
                self._cam_found = scanner_core.probe_camera_available(max_wait_sec=0)
            if self._cam_found:
                self.lbl_status.setText("Cámara detectada.")
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.lbl_status.setStyleSheet("color: #4caf50; font-weight: bold; border: none;")
                self.steps = 90
            else:
                dots = "." * ((self.steps % 3) + 1)
                self.lbl_status.setText(f"Buscando cámara{dots}")
        elif self.steps == 7:
            if not self._cam_found:
                self.lbl_status.setText("Cámara no detectada")
                self.lbl_status.setStyleSheet("color: #ff9800; font-weight: bold; border: none;")
                self.lbl_detail.setText(
                    "No se encontró ninguna cámara Lucid conectada.\n\n"
                    "La aplicación abrirá en modo visor (solo reproducción).\n"
                    "Se reintentará la conexión automáticamente cada pocos segundos."
                )
                self.lbl_detail.show()
                self.btn_continue.show()
                self.setFixedSize(520, 380)
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.timer.stop()
            else:
                self.lbl_status.setText("Listo.")
        elif self.steps >= 9:
            if self._cam_found:
                self.finish_loading()

    def finish_loading(self):
        self.timer.stop()
        self.btn_continue.setEnabled(False)
        self.close()
        if self.main_window:
            self.main_window.showMaximized()
            self.main_window.start_camera_thread()

# --- VISOR PERSONALIZADO (ZOOM + PANEO + PAINT EVENT) ---
# --- VISOR PERSONALIZADO (ZOOM + PANEO + PAINT EVENT) ---
class ViewMode:
    NORMAL = 0
    ZOOM_1_1 = 1
    CORNERS = 2

class ScanViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(False)
        self._pixmap = None
        self.mode = ViewMode.NORMAL
        
        # Offsets para Zoom 1:1 (Coordenada Top-Left del crop)
        self.off_x = 0
        self.off_y = 0
        
        # Variables de arrastre
        self.dragging = False
        self.last_pos = None

        self._exp_text = ""
        self._exp_color = "#888888"

    def setPixmap(self, pix):
        self._pixmap = pix
        self.update() # Repintar

    def set_exposure_overlay(self, text, color="#888888"):
        self._exp_text = text
        self._exp_color = color
        self.update()

    def _paint_exposure_overlay(self, painter):
        if not self._exp_text:
            return
        font = QFont("Consolas", 10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad_x, pad_y = 8, 6
        tw = metrics.horizontalAdvance(self._exp_text)
        th = metrics.height()
        box_w = tw + pad_x * 2
        box_h = th + pad_y * 2
        box_x, box_y = 8, 8
        painter.fillRect(box_x, box_y, box_w, box_h, QColor(0, 0, 0, 170))
        painter.setPen(QColor(self._exp_color))
        painter.drawText(box_x + pad_x, box_y + pad_y + metrics.ascent(), self._exp_text)

    def mousePressEvent(self, event):
        if self.mode == ViewMode.ZOOM_1_1 and event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_pos:
            delta = event.pos() - self.last_pos
            self.last_pos = event.pos()
            
            # Arrastrar la imagen: Mover el offset en dirección opuesta al mouse
            self.off_x -= delta.x()
            self.off_y -= delta.y()
            
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.last_pos = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if not self._pixmap or self._pixmap.isNull():
            self._paint_exposure_overlay(painter)
            return

        w_widget = self.width()
        h_widget = self.height()
        w_img = self._pixmap.width()
        h_img = self._pixmap.height()

        if self.mode == ViewMode.NORMAL:
            # Escalar manteniendo relación de aspecto (Fit)
            scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            x = (w_widget - scaled.width()) // 2
            y = (h_widget - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        elif self.mode == ViewMode.ZOOM_1_1:
            # Clampear offsets para no salirnos de la imagen
            # El "viewport" es del tamaño del widget (o menor si la imagen es chica)
            
            max_off_x = max(0, w_img - w_widget)
            max_off_y = max(0, h_img - h_widget)
            
            self.off_x = max(0, min(self.off_x, max_off_x))
            self.off_y = max(0, min(self.off_y, max_off_y))
            
            # Área a mostrar
            # Si el wiget es más grande que la imagen, mostramos toda la imagen centrada
            if w_widget >= w_img:
                draw_x = (w_widget - w_img) // 2
                src_rect = QRect(0, 0, w_img, min(h_img, h_widget)) # no crop en X
                painter.drawPixmap(draw_x, 0, self._pixmap) # Simplificado
                # (Mejor lógica genérica debajo)
            
            # Rectángulo fuente
            src_x = int(self.off_x)
            src_y = int(self.off_y)
            src_w = min(w_widget, w_img)
            src_h = min(h_widget, h_img)
            
            # Rectángulo destino (Centrado si sobra espacio)
            dst_x = max(0, (w_widget - w_img) // 2)
            dst_y = max(0, (h_widget - h_img) // 2)
            
            painter.drawPixmap(dst_x, dst_y, self._pixmap, src_x, src_y, src_w, src_h)
            
            # Indicador de posición (Mini-mapa opcional, o bordes)
            
        elif self.mode == ViewMode.CORNERS:
            # Dividir pantalla en 4 cuadrantes
            qw = w_widget // 2
            qh = h_widget // 2
            
            # Definir Crops (asumiendo 1:1 pixel a pixel)
            # Top-Left (TL)
            painter.drawPixmap(0, 0, self._pixmap, 0, 0, qw, qh)
            
            # Top-Right (TR) -> Esquina superior derecha de la IMAGEN
            painter.drawPixmap(qw, 0, self._pixmap, w_img - qw, 0, qw, qh)
            
            # Bottom-Left (BL)
            painter.drawPixmap(0, qh, self._pixmap, 0, h_img - qh, qw, qh)
            
            # Bottom-Right (BR)
            painter.drawPixmap(qw, qh, self._pixmap, w_img - qw, h_img - qh, qw, qh)
            
            # Dibujar líneas divisorias
            painter.setPen(QColor(0, 255, 255, 128)) # Cian semi-transparente
            painter.drawLine(qw, 0, qw, h_widget)
            painter.drawLine(0, qh, w_widget, qh)

        self._paint_exposure_overlay(painter)


# --- DIÁLOGO DE EXPORTACIÓN CON THUMBNAILS ---
class BatchExportDialog(QDialog):
    def __init__(self, parent, file_list, root_folder, collection, collection_meta=None):
        super().__init__(parent)
        self.setWindowTitle("Exportación por Lotes")
        self.resize(700, 600)
        self.root = root_folder
        self.coll = collection
        self.collection_meta = collection_meta or {}
        
        # Detectar si la colección tiene archivos QOI/RGB (usar el primer archivo como referencia)
        self.source_is_processed = False
        if file_list and self.collection_meta:
            first_info = self.collection_meta.get(file_list[0], {})
            pf = first_info.get("pixel_format", "bayer")
            self.source_is_processed = (pf in ("rgb", "qoi_rgb"))
        
        layout = QVBoxLayout(self)
        
        # Banner informativo si es QOI/RGB
        if self.source_is_processed:
            lbl_info = QLabel("Fuente: RGB procesado por ISP de cámara (ya tiene WB, Gamma, Sharpening)")
            lbl_info.setStyleSheet("background: #1a237e; color: #90caf9; padding: 6px; border-radius: 4px; font-weight: bold;")
            layout.addWidget(lbl_info)
        
        # 1. Lista Visual con Miniaturas
        layout.addWidget(QLabel("Archivos a procesar (Vista previa cuadro #100):"))
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(160, 120))
        
        for f_name in file_list:
            item = QListWidgetItem(f_name)
            item.setCheckState(Qt.CheckState.Checked)
            icon = self.generate_thumbnail(f_name)
            if icon: item.setIcon(icon)
            self.list_widget.addItem(item)
            
        layout.addWidget(self.list_widget)
        
        # Botones selección
        btn_box = QHBoxLayout()
        btn_all = QPushButton("Marcar Todos")
        btn_all.clicked.connect(lambda: self.set_all(True))
        btn_none = QPushButton("Desmarcar Todos")
        btn_none.clicked.connect(lambda: self.set_all(False))
        btn_box.addWidget(btn_all); btn_box.addWidget(btn_none)
        layout.addLayout(btn_box)
        
        # 2. Configuración (formato-aware)
        settings_group = QGroupBox("Configuración de Salida")
        sett_layout = QVBoxLayout()
        
        sett_layout.addWidget(QLabel("Formato de Salida:"))
        self.combo_fmt = QComboBox()
        
        if self.source_is_processed:
            # Fuente RGB/QOI: NO ofrecer DNG (no tiene sentido para datos ya debayerizados)
            self.combo_fmt.addItems([
                "TIFF 8-bit Sequence (Fiel al ISP - Sin procesamiento) [Recomendado]",
                "ProRes 4444 (Alta Calidad - 12bit)",
                "ProRes 422 HQ (Estándar - 10bit)",
                "HEVC 10-bit 4:4:4 (MP4 - Eficiente)",
                "AV1 NVENC 10-bit (MP4 - Eficiente)",
                "H.264 (MP4 - Proxy)"
            ])
            self.combo_fmt.setCurrentIndex(0)
        else:
            # Fuente Bayer: Opciones completas con DNG
            self.combo_fmt.addItems([
                "DNG Raw Sequence (DaVinci Resolve - 16bit) [Recomendado]",
                "ProRes 4444 (Premiere - 12bit - Alta Calidad)", 
                "GoPro CineForm (Premiere - 12bit - Intermedio)", 
                "ProRes 422 HQ (Premiere - 10bit - Estándar)",
                "HEVC 10-bit 4:4:4 (MP4 - Eficiente)",
                "AV1 NVENC 10-bit (MP4 - Eficiente)",
                "H.264 (MP4 - Proxy)"
            ])
            self.combo_fmt.setCurrentIndex(0)
        sett_layout.addWidget(self.combo_fmt)
        
        # Perfil de revelado (solo para Bayer, para RGB el ISP ya lo hizo)
        if self.source_is_processed:
            sett_layout.addWidget(QLabel("(Sin perfil de revelado - La cámara ya aplicó ISP)"))
            self.combo_sharp = QComboBox()
            self.combo_sharp.addItems(["N/A (ISP de cámara)"])
        else:
            sett_layout.addWidget(QLabel("Perfil de Revelado (Solo Video - No afecta DNG):"))
            self.combo_sharp = QComboBox()
            self.combo_sharp.addItems([
                "DCB Puro",
                "Suave (S:0.8 / A:1.5)",
                "Medio (S:1.3 / A:1.5)",
                "Grueso (S:2.0 / A:2.5) [Recomendado para Video]"
            ])
            self.combo_sharp.setCurrentIndex(3)
        sett_layout.addWidget(self.combo_sharp)
        
        settings_group.setLayout(sett_layout)
        layout.addWidget(settings_group)
        
        self.btn_export = QPushButton("Iniciar Cola")
        self.btn_export.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; padding: 10px;")
        self.btn_export.clicked.connect(self.accept)
        layout.addWidget(self.btn_export)

    def generate_thumbnail(self, filename):
        try:
            full_path = Path(self.root) / self.coll / filename
            fsize = full_path.stat().st_size
            w, h = 2840, 2200 
            
            # Detectar formato desde metadata si disponible
            file_meta = self.collection_meta.get(filename, {})
            pixel_fmt = file_meta.get("pixel_format", "")
            is_qoi = (pixel_fmt == "qoi_rgb")
            is_rgb = (pixel_fmt == "rgb")
            
            # Fallback por tamaño si no hay metadata
            if not pixel_fmt:
                if fsize % int(w * h * 3) == 0: is_rgb = True
                elif fsize % int(w * h * 1.5) == 0: is_rgb = False
            
            if is_qoi:
                # QOI: Leer frame usando el índice del contenedor
                frame_index = qoi_utils.build_frame_index(str(full_path))
                target = min(100, len(frame_index) - 1) if frame_index else 0
                if not frame_index: return None
                offset, fsz = frame_index[target]
                qoi_data = qoi_utils.read_frame_at(str(full_path), offset, fsz)
                rgb = qoi_utils.decode_qoi(qoi_data, w, h)
                small = cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_NEAREST)
            elif is_rgb:
                frame_bytes = int(w * h * 3)
                total_frames = fsize // frame_bytes
                target_frame = 100 if total_frames > 100 else max(0, total_frames - 1)
                with open(full_path, "rb") as f:
                    f.seek(target_frame * frame_bytes)
                    raw_data = f.read(frame_bytes)
                if len(raw_data) < frame_bytes: return None
                rgb = np.frombuffer(raw_data, dtype=np.uint8).reshape(h, w, 3)
                small = rgb[::8, ::8, :].copy()
            else:
                # BAYER RAW
                frame_bytes = int(w * h * 1.5)
                total_frames = fsize // frame_bytes
                target_frame = 100 if total_frames > 100 else max(0, total_frames - 1)
                with open(full_path, "rb") as f:
                    f.seek(target_frame * frame_bytes)
                    raw_data = f.read(frame_bytes)
                if len(raw_data) < frame_bytes: return None
                data = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
                b0, b1, b2 = data[:, 0], data[:, 1], data[:, 2]
                p0 = ((b1 & 0x0F) << 4) | (b0 >> 4)
                p1 = b2
                img_flat = np.empty(w*h, dtype=np.uint8)
                img_flat[0::2] = p0; img_flat[1::2] = p1
                img_bayer = img_flat.reshape(h, w)
                r_ch = img_bayer[0::2, 0::2]
                g_ch = img_bayer[0::2, 1::2]
                b_ch = img_bayer[1::2, 1::2]
                small = np.dstack((r_ch, g_ch, b_ch))
                small = cv2.resize(small, (160, 120), interpolation=cv2.INTER_NEAREST)
                avg = np.mean(small)
                if avg > 0: small = np.clip(small * (100/avg), 0, 255).astype(np.uint8)
            
            ih, iw = small.shape[:2]
            if small.ndim == 2: small = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
            if not small.flags['C_CONTIGUOUS']: small = np.ascontiguousarray(small)
            qimg = QImage(small.data, iw, ih, iw*3, QImage.Format.Format_RGB888)
            return QIcon(QPixmap.fromImage(qimg))
        except Exception as e:
            print(f"Thumbnail error ({filename}): {e}")
            return None
    def set_all(self, state):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

    def get_selection(self):
        files = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                files.append(item.text())
        
        if self.source_is_processed:
            fmt_map = {0: 'tiff_seq', 1: 'prores', 2: 'prores_hq', 3: 'hevc', 4: 'av1', 5: 'h264'}
            sharp = '0,0'
        else:
            fmt_map = {0: 'dng', 1: 'prores', 2: 'cineform', 3: 'prores_hq', 4: 'hevc', 5: 'av1', 6: 'h264'}
            sharp_map = {0: '0,0', 1: '0.8,1.5', 2: '1.3,1.5', 3: '2.0,2.5'}
            sharp = sharp_map.get(self.combo_sharp.currentIndex(), '0,0')
        
        return files, fmt_map.get(self.combo_fmt.currentIndex(), 'dng'), sharp


class CalibrationLiveViewer(QWidget):
    """Vista previa a pantalla completa con retícula (cruz + diagonales) en naranja 1px."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setStyleSheet("background-color: #111;")
        self.setMinimumSize(480, 360)

    def set_frame(self, frame: np.ndarray):
        is_color = frame.ndim == 3
        if is_color:
            disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            disp = frame
        h, w = disp.shape[:2]
        if is_color:
            qimg = QImage(disp.data, w, h, w * 3, QImage.Format.Format_RGB888)
        else:
            if disp.dtype == np.uint16:
                disp = (disp >> 4).astype(np.uint8)
            if not disp.flags["C_CONTIGUOUS"]:
                disp = np.ascontiguousarray(disp)
            qimg = QImage(disp.data, w, h, w, QImage.Format.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self._pixmap or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        x0 = (self.width() - scaled.width()) // 2
        y0 = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x0, y0, scaled)
        pen = QPen(QColor("#ff6600"))
        pen.setWidth(1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        iw, ih = scaled.width(), scaled.height()
        cx = x0 + iw // 2
        cy = y0 + ih // 2
        painter.drawLine(cx, y0, cx, y0 + ih)
        painter.drawLine(x0, cy, x0 + iw, cy)
        painter.drawLine(x0, y0, x0 + iw, y0 + ih)
        painter.drawLine(x0 + iw - 1, y0, x0, y0 + ih - 1)


class CameraCalibrationDialog(QDialog):
    def __init__(self, main_window: "MainWindow"):
        super().__init__(main_window)
        self.main_window = main_window
        self._restored = False
        self._exp_range = None
        self._gain_range = None
        self._calib_steps = 20
        self.setWindowTitle("Calibrar posición de cámara")
        self.resize(960, 750)
        layout = QVBoxLayout(self)
        self.viewer = CalibrationLiveViewer(self)
        layout.addWidget(self.viewer, 10)
        help_lbl = QLabel(
            "Vista en vivo continua, autoexposición, autoganancia y sin disparo externo. "
            "Usa la retícula para centrar la cámara."
        )
        help_lbl.setStyleSheet("color: #888; font-size: 9pt;")
        help_lbl.setWordWrap(True)
        layout.addWidget(help_lbl)

        # Controles manuales de exposición y gain (modo calibración)
        ctrl_box = QGroupBox("Exposición y ganancia (modo calibración)")
        ctrl_layout = QHBoxLayout(ctrl_box)

        # Exposición
        exp_col = QVBoxLayout()
        lbl_exp = QLabel("Exposición (µs)")
        lbl_exp.setStyleSheet("font-size: 9pt; color: #ccc;")
        self.sld_exp_calib = QSlider(Qt.Orientation.Horizontal)
        self.sld_exp_calib.setRange(0, self._calib_steps - 1)
        self.sld_exp_calib.setEnabled(False)
        self.lbl_exp_calib_val = QLabel("--")
        self.lbl_exp_calib_val.setStyleSheet("font-size: 9pt; color: #aaa;")
        exp_col.addWidget(lbl_exp)
        exp_col.addWidget(self.sld_exp_calib)
        exp_col.addWidget(self.lbl_exp_calib_val)
        ctrl_layout.addLayout(exp_col)

        # Gain
        gain_col = QVBoxLayout()
        lbl_gain = QLabel("Gain")
        lbl_gain.setStyleSheet("font-size: 9pt; color: #ccc;")
        self.sld_gain_calib = QSlider(Qt.Orientation.Horizontal)
        self.sld_gain_calib.setRange(0, self._calib_steps - 1)
        self.sld_gain_calib.setEnabled(False)
        self.lbl_gain_calib_val = QLabel("--")
        self.lbl_gain_calib_val.setStyleSheet("font-size: 9pt; color: #aaa;")
        gain_col.addWidget(lbl_gain)
        gain_col.addWidget(self.sld_gain_calib)
        gain_col.addWidget(self.lbl_gain_calib_val)
        ctrl_layout.addLayout(gain_col)

        layout.addWidget(ctrl_box)

        btn_row = QHBoxLayout()
        btn_back = QPushButton("Volver")
        btn_back.setMinimumHeight(32)
        btn_back.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(btn_back)
        layout.addLayout(btn_row)

        w = main_window.camera_worker
        if w:
            w.request_calibration_live_mode(True)
            w.image_received.connect(self._on_frame)

            # Intentar leer rangos de exposición y gain para los sliders
            try:
                ranges = w.get_exposure_gain_ranges()
            except Exception as e:
                print(f"WARN get_exposure_gain_ranges (dialog): {e}")
                ranges = None
            if ranges:
                self._exp_range = ranges.get("exp")
                self._gain_range = ranges.get("gain")

            if self._exp_range and self._exp_range[0] is not None and self._exp_range[1] is not None:
                self.sld_exp_calib.setEnabled(True)
                self.sld_exp_calib.valueChanged.connect(self._on_exp_calib_slider_changed)
                # Punto medio inicial
                self.sld_exp_calib.setValue(self._calib_steps // 2)

            if self._gain_range and self._gain_range[0] is not None and self._gain_range[1] is not None:
                self.sld_gain_calib.setEnabled(True)
                self.sld_gain_calib.valueChanged.connect(self._on_gain_calib_slider_changed)
                # Arrancar en mínimo
                self.sld_gain_calib.setValue(0)

    def _on_frame(self, frame: np.ndarray):
        self.viewer.set_frame(frame)

    def _map_slider_to_range(self, idx: int, lo: float, hi: float) -> float:
        if self._calib_steps <= 1 or hi <= lo:
            return lo
        t = max(0.0, min(1.0, idx / float(self._calib_steps - 1)))
        return lo + (hi - lo) * t

    def _on_exp_calib_slider_changed(self, idx: int):
        if not self._exp_range or self._exp_range[0] is None or self._exp_range[1] is None:
            return
        lo, hi = self._exp_range
        target = self._map_slider_to_range(idx, lo, hi)
        w = self.main_window.camera_worker
        if not w or not hasattr(w, "set_exposure_time_calibration"):
            return
        try:
            applied = w.set_exposure_time_calibration(target)
        except Exception as e:
            print(f"WARN set_exposure_time_calibration (dialog): {e}")
            return
        self.lbl_exp_calib_val.setText(f"{applied:.0f} µs")

    def _on_gain_calib_slider_changed(self, idx: int):
        if not self._gain_range or self._gain_range[0] is None or self._gain_range[1] is None:
            return
        lo, hi = self._gain_range
        target = self._map_slider_to_range(idx, lo, hi)
        w = self.main_window.camera_worker
        if not w or not hasattr(w, "set_gain_calibration"):
            return
        try:
            applied = w.set_gain_calibration(target)
        except Exception as e:
            print(f"WARN set_gain_calibration (dialog): {e}")
            return
        self.lbl_gain_calib_val.setText(f"{applied:.2f}")

    def _restore_camera(self):
        if self._restored:
            return
        self._restored = True
        w = self.main_window.camera_worker
        if w:
            try:
                w.image_received.disconnect(self._on_frame)
            except (TypeError, RuntimeError):
                pass
            mw = self.main_window
            idx = mw.sld_exposure.value()
            short = idx == mw.SHORT_EXP_IDX
            val = mw.EXP_STEPS[idx]
            w.set_restore_exposure_after_calibration(short, val)
            w.request_calibration_live_mode(False)

    def accept(self):
        self._restore_camera()
        super().accept()

    def reject(self):
        self._restore_camera()
        super().reject()

    def closeEvent(self, event):
        self._restore_camera()
        super().closeEvent(event)


class CameraTimerDialog(QDialog):
    def __init__(self, main_window: "MainWindow"):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Ajuste Timer0")
        self.setMinimumSize(470, 360)
        self.resize(520, 420)
        layout = QVBoxLayout(self)

        info = QLabel("Ajusta solo Timer0: TimerDuration y TimerDelay.")
        info.setStyleSheet("color: #999;")
        layout.addWidget(info)

        row_duration = QHBoxLayout()
        row_duration.addWidget(QLabel("TimerDuration"))
        self.spn_duration = QSpinBox()
        self.spn_duration.setRange(1, 10_000_000)
        self.spn_duration.setValue(90000)
        self.spn_duration.setSuffix(" us")
        row_duration.addWidget(self.spn_duration, 1)
        layout.addLayout(row_duration)

        row_delay = QHBoxLayout()
        row_delay.addWidget(QLabel("TimerDelay"))
        self.spn_delay = QSpinBox()
        self.spn_delay.setRange(0, 10_000_000)
        self.spn_delay.setValue(50)
        self.spn_delay.setSuffix(" us")
        row_delay.addWidget(self.spn_delay, 1)
        layout.addLayout(row_delay)

        row_actions = QHBoxLayout()
        self.btn_read = QPushButton("Leer actual")
        self.btn_apply = QPushButton("Aplicar y verificar")
        self.btn_close = QPushButton("Cerrar")
        self.btn_read.clicked.connect(self.read_current_values)
        self.btn_apply.clicked.connect(self.apply_values)
        self.btn_close.clicked.connect(self.accept)
        row_actions.addWidget(self.btn_read)
        row_actions.addWidget(self.btn_apply)
        row_actions.addStretch()
        row_actions.addWidget(self.btn_close)
        layout.addLayout(row_actions)

        self.lbl_result = QLabel("Estado: pendiente")
        self.lbl_result.setStyleSheet("font-weight: bold; color: #ff9800;")
        layout.addWidget(self.lbl_result)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setPlaceholderText("Logs de aplicación y verificación...")
        layout.addWidget(self.logs, 1)

        self.read_current_values()

    def _log(self, message: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {message}")

    def read_current_values(self):
        w = self.main_window.camera_worker
        if not w or not self.main_window._camera_connected:
            self.lbl_result.setText("Estado: sin cámara")
            self.lbl_result.setStyleSheet("font-weight: bold; color: #f44336;")
            self._log("No hay cámara conectada para leer Timer0.")
            return
        data = w.get_timer0_config()
        if not data:
            self.lbl_result.setText("Estado: lectura fallida")
            self.lbl_result.setStyleSheet("font-weight: bold; color: #f44336;")
            self._log("No se pudieron leer los valores actuales de Timer0.")
            return
        self.spn_duration.setValue(int(round(data["duration"])))
        self.spn_delay.setValue(int(round(data["delay"])))
        self.lbl_result.setText("Estado: valores cargados")
        self.lbl_result.setStyleSheet("font-weight: bold; color: #4caf50;")
        self._log(
            f"Leído Timer0 actual -> Duration={data['duration']:.0f} us, Delay={data['delay']:.0f} us."
        )

    def apply_values(self):
        w = self.main_window.camera_worker
        if not w or not self.main_window._camera_connected:
            self.lbl_result.setText("Estado: sin cámara")
            self.lbl_result.setStyleSheet("font-weight: bold; color: #f44336;")
            self._log("No hay cámara conectada para aplicar cambios.")
            return
        duration = float(self.spn_duration.value())
        delay = float(self.spn_delay.value())
        self._log(f"Aplicando Timer0 -> Duration={duration:.0f} us, Delay={delay:.0f} us.")
        result = w.set_timer0_config(duration, delay)
        applied_dur = result.get("applied_duration")
        applied_del = result.get("applied_delay")
        if result.get("ok"):
            self.lbl_result.setText("Estado: aplicado correctamente")
            self.lbl_result.setStyleSheet("font-weight: bold; color: #4caf50;")
            self._log(
                f"OK verificado -> Duration={applied_dur:.0f} us, Delay={applied_del:.0f} us."
            )
        else:
            self.lbl_result.setText("Estado: no se aplicó exactamente")
            self.lbl_result.setStyleSheet("font-weight: bold; color: #ff9800;")
            err = result.get("error", "Sin detalle.")
            if applied_dur is not None and applied_del is not None:
                self._log(
                    "DIFERENCIA: solicitado "
                    f"({duration:.0f}, {delay:.0f}) vs aplicado "
                    f"({applied_dur:.0f}, {applied_del:.0f})."
                )
            self._log(f"Detalle: {err}")


# --- VENTANA PRINCIPAL ---
class MainWindow(QMainWindow):
    def __init__(self, defer_camera=False):
        super().__init__()
        self.setWindowTitle("Escáner de películas - Archivo La Unión")
        self.resize(1400, 900)
        self._defer_camera = defer_camera
        self.load_config() # Carga o pide la carpeta raíz
        self.manager = scanner_core.CollectionManager(self.root_folder)
        self.active_collection = None
        self.is_recording = False
        # OPTIMIZACIÓN RAM: Aumentamos buffer a 600 frames (~5.4GB)
        # Esto permite absorber latencia de escritura en disco.
        self.frame_queue = queue.Queue(maxsize=600)
        self.camera_worker = None
        self.writer_worker = None
        self._stats_cache = {
            "fps": 0.0, "temp": 0.0, "qsize": 0, "cam_drops": 0, "disk_drops": 0,
            "total_cam": 0, "bw": 0.0, "bw_src": "C",
        }
        self._calibration_dialog = None
        self._timer_dialog = None
        self._execution_state_key = None  # None | (display, system) último estado aplicado

        # Variables visor
        self.raw_width = 2840
        self.raw_height = 2200
        
        self.init_ui()
        self.refresh_collections()
        self.update_disk_space()
        
        self.disk_timer = QTimer()
        self.disk_timer.timeout.connect(self.update_disk_space)
        self.disk_timer.start(10000)
        self.bayer_phase = 0

        self._toast = QLabel(self)
        self._toast.setStyleSheet(
            "QLabel { background-color: rgba(28,28,28,230); color: #eee; "
            "padding: 10px 18px; border-radius: 8px; border: 1px solid #555; font-size: 11pt; }"
        )
        self._toast.hide()
        self._toast.raise_()
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._toast.hide)
        self._pending_config_toast = False
        self._applying_persist = False
        self._persist_save_timer = QTimer(self)
        self._persist_save_timer.setSingleShot(True)
        self._persist_save_timer.setInterval(400)
        self._persist_save_timer.timeout.connect(self.save_config)

        self._applying_persist = True
        try:
            self.apply_persist_ui()
        finally:
            self._applying_persist = False

        if not defer_camera:
            self.start_camera_thread()

    def _persist_defaults(self):
        return {
            "root_folder": os.path.expanduser("~/Documents/Archivo_Scan_Data"),
            "ui": {
                "preview_mode": "ISP",
                "film_type": "Color (Pos/Neg)",
                "pixel_mode": "bayer",
                "exposure_index": 11,
                "real_fps": False,
                "peaking": False,
                "zoom_1to1": False,
                "corners": False,
                "playback_fps": 18,
                "bayer_phase": 0,
                "active_collection": None,
                "tab_index": 0,
                "window": None,
            },
            "isp": {
                "tone": [2.0, 0.0, 1.0],
            },
        }

    @staticmethod
    def _deep_update(base: dict, extra: dict):
        for key, val in extra.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                MainWindow._deep_update(base[key], val)
            else:
                base[key] = val

    def load_config(self):
        config_path = Path(__file__).parent / "persist.json"
        self._persist = self._persist_defaults()

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self._deep_update(self._persist, json.load(f))
            except Exception as e:
                print(f"Error cargando persist.json: {e}")

        legacy_path = Path(__file__).parent / "config.json"
        if legacy_path.exists():
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    leg = json.load(f)
                if "pixel_mode" in leg:
                    self._persist["ui"]["pixel_mode"] = leg["pixel_mode"]
            except Exception:
                pass

        self.root_folder = self._persist.get("root_folder") or self._persist_defaults()["root_folder"]
        if not self.root_folder or not os.path.isdir(self.root_folder):
            self.ask_root_folder_first_time()

    def ask_root_folder_first_time(self):
        # Usamos un QDialog temporal o QMessageBox porque self (MainWindow) aun no es visible
        msg = QMessageBox()
        msg.setWindowTitle("Configuración Inicial")
        msg.setText("Bienvenido al Scanner Suite.\n\nPor favor selecciona la carpeta donde se guardarán los escaneos (Colecciones).")
        msg.setIcon(QMessageBox.Icon.Information)
        # Importante: Hack para que aparezca encima del splash si es necesario
        msg.setWindowFlags(msg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        msg.exec()
        
        folder = QFileDialog.getExistingDirectory(None, "Seleccionar Carpeta de Datos", os.path.expanduser("~/Documents"))
        if folder:
            self.root_folder = folder
        else:
            # Si cancela, usamos default y avisamos
            if not os.path.exists(self.root_folder): 
                try: os.makedirs(self.root_folder, exist_ok=True)
                except: pass
            QMessageBox.warning(None, "Atención", f"No se seleccionó carpeta. Se usará la carpeta por defecto:\n{self.root_folder}")
        
        self.save_config()

    def _collect_ui_persist(self) -> dict:
        ui = dict(self._persist.get("ui", {}))
        ui["preview_mode"] = self.combo_preview.currentText()
        ui["film_type"] = self.combo_type.currentText()
        ui["pixel_mode"] = "qoi_rgb" if self.combo_pixel_mode.currentIndex() == 1 else "bayer"
        ui["exposure_index"] = int(self.sld_exposure.value())
        ui["real_fps"] = self.btn_real_fps.isChecked()
        ui["peaking"] = self.btn_peaking.isChecked()
        ui["zoom_1to1"] = self.btn_zoom_1to1.isChecked()
        ui["corners"] = self.btn_corners.isChecked()
        ui["playback_fps"] = int(self.sb_fps.value())
        ui["bayer_phase"] = int(getattr(self, "bayer_phase", 0))
        ui["active_collection"] = self.active_collection
        ui["tab_index"] = int(self.tabs.currentIndex())
        geo = self.geometry()
        ui["window"] = [geo.x(), geo.y(), geo.width(), geo.height()]
        return ui

    def _collect_isp_persist(self) -> dict:
        w = self.camera_worker
        isp = {"tone": [2.0, 0.0, 1.0]}
        if w:
            g, lift, contrast = w.get_isp_preview_tone()
            isp["tone"] = [g, lift, contrast]
        else:
            isp = dict(self._persist.get("isp", isp))
        return isp

    def schedule_save_config(self):
        if getattr(self, "_applying_persist", False):
            return
        self._persist_save_timer.start()

    def save_config(self):
        try:
            self._persist["root_folder"] = self.root_folder
            self._persist["ui"] = self._collect_ui_persist()
            self._persist["isp"] = self._collect_isp_persist()
            config_path = Path(__file__).parent / "persist.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._persist, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando persist.json: {e}")

    def apply_persist_ui(self):
        ui = self._persist.get("ui", self._persist_defaults()["ui"])

        win = ui.get("window")
        if isinstance(win, list) and len(win) == 4:
            try:
                self.setGeometry(int(win[0]), int(win[1]), int(win[2]), int(win[3]))
            except Exception:
                pass

        tab_idx = int(ui.get("tab_index", 0))
        if 0 <= tab_idx < self.tabs.count():
            self.tabs.setCurrentIndex(tab_idx)

        self.bayer_phase = int(ui.get("bayer_phase", 0))

        film = ui.get("film_type", "Color (Pos/Neg)")
        fi = self.combo_type.findText(film)
        if fi >= 0:
            self.combo_type.setCurrentIndex(fi)

        preview = ui.get("preview_mode", "ISP")
        pi = self.combo_preview.findText(preview)
        if pi >= 0:
            self.combo_preview.setCurrentIndex(pi)

        pixel_mode = ui.get("pixel_mode", "bayer")
        self.combo_pixel_mode.setCurrentIndex(1 if pixel_mode == "qoi_rgb" else 0)

        exp_idx = int(ui.get("exposure_index", self.sld_exposure.value()))
        exp_idx = max(0, min(len(self.EXP_STEPS) - 1, exp_idx))
        self.sld_exposure.setValue(exp_idx)
        self.on_exposure_slider_changed(exp_idx)

        self.btn_real_fps.setChecked(bool(ui.get("real_fps", False)))
        self.toggle_real_fps(self.btn_real_fps.isChecked())

        self.btn_peaking.setChecked(bool(ui.get("peaking", False)))
        self.toggle_peaking(self.btn_peaking.isChecked())

        zoom = bool(ui.get("zoom_1to1", False))
        corners = bool(ui.get("corners", False))
        self.btn_zoom_1to1.setChecked(zoom)
        self.btn_corners.setChecked(corners)
        if zoom:
            self.toggle_zoom_1to1(True)
        elif corners:
            self.toggle_corners(True)
        else:
            self.viewer_scan.mode = ViewMode.NORMAL
            self.viewer_scan.update()

        self.sb_fps.setValue(int(ui.get("playback_fps", 18)))

        coll = ui.get("active_collection")
        if coll and coll in self.manager.get_collections():
            self.active_collection = coll
            self.lbl_status.setText(f"Colección Activa: {coll}")
            self.lbl_status.setStyleSheet("font-size: 14pt; color: #4caf50;")
            items = self.col_list.findItems(coll, Qt.MatchFlag.MatchExactly)
            if items:
                self.col_list.setCurrentItem(items[0])
            self.refresh_file_list(coll)

    def _apply_persist_to_camera(self):
        if not self.camera_worker or not self._camera_connected:
            return
        ui = self._persist.get("ui", {})
        label = ui.get("preview_mode", self.combo_preview.currentText())
        self.on_preview_mode_changed(label)

        real_fps = bool(ui.get("real_fps", self.btn_real_fps.isChecked()))
        self.camera_worker.set_preview_skip_frames(not real_fps)

        self._apply_exposure_to_camera()

        isp = self._persist.get("isp", self._persist_defaults()["isp"])
        tone = isp.get("tone", [2.0, 0.0, 1.0])
        if not (isinstance(tone, (list, tuple)) and len(tone) == 3):
            tone = [2.0, 0.0, 1.0]
        if self.btn_peaking.isChecked():
            tone = [1.0, tone[1], tone[2]]
        self.camera_worker.set_isp_preview_tone(float(tone[0]), float(tone[1]), float(tone[2]))

    def _preview_gamma_normal(self) -> float:
        return 2.0

    def _apply_preview_gamma_for_peaking(self):
        if not self.camera_worker:
            return
        g = 1.0 if self.btn_peaking.isChecked() else self._preview_gamma_normal()
        self.camera_worker.set_isp_preview_tone(g, 0.0, 1.0)

    def change_root_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Nueva Carpeta Raíz", self.root_folder)
        if folder and folder != self.root_folder:
            self.root_folder = folder
            self.save_config()
            # Reiniciar Manager y listas
            self.manager = scanner_core.CollectionManager(self.root_folder)
            self.refresh_collections()
            self.file_list.clear() # Limpiar lista de archivos antigua
            self.update_disk_space()
            QMessageBox.information(self, "Cambio Exitoso", f"Carpeta de escaneo actualizada a:\n{folder}")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Panel Izquierdo
        left_panel = QGroupBox("Gestión de Colecciones")
        left_layout = QVBoxLayout()
        self.col_list = QListWidget()
        self.col_list.itemClicked.connect(self.on_collection_select)
        self.col_list.itemDoubleClicked.connect(self.activate_collection)
        self.col_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.col_list.customContextMenuRequested.connect(self.open_collection_context_menu)
        btn_new_col = QPushButton("Nueva Colección"); btn_new_col.clicked.connect(self.create_collection)
        btn_refresh = QPushButton("Refrescar"); btn_refresh.clicked.connect(self.refresh_collections)
        self.lbl_disk = QLabel("Espacio Libre: ...")
        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self.load_file_in_viewer)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.open_file_context_menu)
        
        left_layout.addWidget(btn_new_col); left_layout.addWidget(self.col_list)
        left_layout.addWidget(QLabel("Archivos:")); left_layout.addWidget(self.file_list)
        
        # Botonera de gestión
        btn_group = QGroupBox("Opciones")
        bg_layout = QVBoxLayout()
        btn_change_root = QPushButton("📂 Directorio Datos"); btn_change_root.clicked.connect(self.change_root_folder)
        
        bg_layout.addWidget(btn_refresh)
        bg_layout.addWidget(btn_change_root)
        btn_group.setLayout(bg_layout)
        
        left_layout.addWidget(btn_group)
        left_layout.addWidget(self.lbl_disk)
        left_panel.setLayout(left_layout); left_panel.setMaximumWidth(300)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.installEventFilter(self)
        
        # TAB 1: CAPTURA
        # TAB 1: CAPTURA
        self.tab_scan = QWidget()
        scan_layout = QVBoxLayout()
        scan_layout.setContentsMargins(5,5,5,5)
        
        # [HEADER] Título + estadísticas en una sola línea compacta + calibración debajo
        header_bar = QHBoxLayout()
        self.lbl_status = QLabel("Selecciona una colección")
        self.lbl_status.setStyleSheet("font-size: 14pt; color: #ff9800; font-weight: bold;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        self.lbl_cam_status = QLabel("●")
        self.lbl_cam_status.setToolTip("Estado de cámara: Buscando...")
        self.lbl_cam_status.setStyleSheet("color: #ff9800; font-size: 14pt; margin-right: 4px;")
        self._camera_connected = False
        self.lbl_stats_inline = QLabel("FPS: —  ·  Tmp: —  ·  Buf: —  ·  BW: —  ·  📷 —")
        self.lbl_stats_inline.setStyleSheet("font-size: 9pt; color: #888;")
        self.lbl_stats_inline.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        stats_row = QHBoxLayout()
        stats_row.setSpacing(6)
        stats_row.addWidget(self.lbl_cam_status)
        stats_row.addWidget(self.lbl_stats_inline, 1)
        header_right_col = QVBoxLayout()
        header_right_col.setSpacing(4)
        header_right_col.addLayout(stats_row)
        self.btn_isp_tone = QPushButton("Ajuste ISP")
        self.btn_isp_tone.setVisible(False)
        self.btn_calibrate_camera = QPushButton("Calibrar posición de cámara")
        self.btn_calibrate_camera.setEnabled(False)
        self.btn_calibrate_camera.setToolTip(
            "Vista en vivo sin disparo y con autoexposición, para alinear la cámara.")
        self.btn_calibrate_camera.clicked.connect(self.open_camera_calibration_dialog)
        self.btn_calibrate_camera.setStyleSheet("font-size: 9pt;")
        btn_row_calib_timer = QHBoxLayout()
        btn_row_calib_timer.setSpacing(6)
        btn_row_calib_timer.addWidget(self.btn_isp_tone)
        btn_row_calib_timer.addWidget(self.btn_calibrate_camera)
        self.btn_timer0 = QPushButton("Timer0")
        self.btn_timer0.setEnabled(False)
        self.btn_timer0.setToolTip("Ajustar TimerDuration y TimerDelay de Timer0")
        self.btn_timer0.setStyleSheet("font-size: 9pt;")
        self.btn_timer0.clicked.connect(self.open_timer_dialog)
        btn_row_calib_timer.addWidget(self.btn_timer0)
        header_right_col.addLayout(btn_row_calib_timer)
        header_bar.addWidget(self.lbl_status, 1)
        header_bar.addLayout(header_right_col, 1)
        scan_layout.addLayout(header_bar)
        
        # [VISOR] con overlay de exposición integrado en ScanViewer
        self.viewer_scan = ScanViewer()
        self.viewer_scan.setStyleSheet("background-color: #111; border: 1px solid #444;")
        self.viewer_scan.set_exposure_overlay("Hi —", "#666666")
        # Fix: Usar QSizePolicy enums correctamente
        from PyQt6.QtWidgets import QSizePolicy 
        self.viewer_scan.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scan_layout.addWidget(self.viewer_scan, 10)
        
        # [CONTROL BAR] Unificada
        control_bar = QGroupBox()
        control_bar.setMaximumHeight(80)
        cb_layout = QHBoxLayout(control_bar)
        cb_layout.setContentsMargins(8, 2, 8, 2)
        cb_layout.setSpacing(15)
        
        # A. Tipo
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Color (Pos/Neg)", "Blanco y Negro"])
        self.combo_type.setFixedWidth(130)
        self.combo_type.currentTextChanged.connect(lambda _: self.schedule_save_config())
        cb_layout.addWidget(self.combo_type)
        
        # B. Foco
        # B. Foco y Zoom
        self.btn_peaking = QPushButton("Peak")
        self.btn_peaking.setCheckable(True)
        self.btn_peaking.clicked.connect(self.toggle_peaking)
        self.btn_peaking.setFixedWidth(40)
        
        self.btn_zoom_1to1 = QPushButton("1:1")
        self.btn_zoom_1to1.setCheckable(True)
        self.btn_zoom_1to1.clicked.connect(self.toggle_zoom_1to1)
        self.btn_zoom_1to1.setFixedWidth(40)
        
        self.btn_corners = QPushButton("⛶") # Icono de esquinas/cuadrado
        self.btn_corners.setToolTip("Visor de Esquinas")
        self.btn_corners.setCheckable(True)
        self.btn_corners.clicked.connect(self.toggle_corners)
        self.btn_corners.setFixedWidth(30)
        
        # Grupo de Modo de Vista (Exclusivo)
        # Gestionaremos la exclusividad manualmente en los slots
        
        # Modo de previsualización (ISP / RAW / B-W)
        self.combo_preview = QComboBox()
        self.combo_preview.addItems(["ISP", "ISP Full", "HQ", "HQ½", "RAW", "B/W"])
        self.combo_preview.setCurrentIndex(0)
        self.combo_preview.setToolTip(
            "ISP: SDK mitad res · ISP Full: SDK 2840×2200 · HQ/HQ½: debayer software · RAW/B/W: rápido"
        )
        self.combo_preview.setFixedWidth(78)
        self.combo_preview.currentTextChanged.connect(self.on_preview_mode_changed)
        
        self.btn_real_fps = QPushButton("Real")
        self.btn_real_fps.setCheckable(True)
        self.btn_real_fps.setToolTip("Forzar FPS Reales (No saltar cuadros)")
        self.btn_real_fps.clicked.connect(self.toggle_real_fps)
        self.btn_real_fps.setFixedWidth(40)

        cb_layout.addWidget(self.btn_peaking)
        cb_layout.addWidget(self.btn_zoom_1to1)
        cb_layout.addWidget(self.btn_corners)
        cb_layout.addWidget(self.combo_preview)
        cb_layout.addWidget(self.btn_real_fps)
        
        # C. Velocidad de obturador
        # Índice 0: Short Exposure Mode (~2.3µs, ShortExposureEnable=True)
        # Índices 1–31: modo normal 25–100µs en pasos de 2.5µs (4 pasos/decena)
        # EXP_STEPS[0] = 2.5 es solo el valor indicativo; la cámara usa ~2.3µs en Short Mode
        self.EXP_STEPS = [2.5] + [round(25 + i * 2.5, 1) for i in range(31)]  # 32 valores
        self.SHORT_EXP_IDX = 0   # único índice que activa ShortExposureEnable

        exp_layout = QVBoxLayout()
        exp_layout.setSpacing(0)
        exp_layout.setContentsMargins(0, 0, 0, 0)

        lbl_exp_title = QLabel("Vel (µs)")
        lbl_exp_title.setStyleSheet("font-size: 8pt; color: #aaa;")
        lbl_exp_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        exp_row = QHBoxLayout()
        exp_row.setSpacing(2)

        self.btn_exp_down = QPushButton("◀")
        self.btn_exp_down.setFixedSize(20, 20)
        self.btn_exp_down.setToolTip("Bajar velocidad (un paso)")
        self.btn_exp_down.clicked.connect(lambda: self.step_exposure(-1))

        self.sld_exposure = QSlider(Qt.Orientation.Horizontal)
        self.sld_exposure.setRange(0, len(self.EXP_STEPS) - 1)
        default_idx = self.EXP_STEPS.index(50.0)
        self.sld_exposure.setValue(default_idx)
        self.sld_exposure.setFixedWidth(100)
        self.sld_exposure.setToolTip("Velocidad de obturador — paso 0: Short Mode (~2.3µs), pasos 1–31: 25–100µs")
        self.sld_exposure.valueChanged.connect(self.on_exposure_slider_changed)

        self.btn_exp_up = QPushButton("▶")
        self.btn_exp_up.setFixedSize(20, 20)
        self.btn_exp_up.setToolTip("Subir velocidad (un paso)")
        self.btn_exp_up.clicked.connect(lambda: self.step_exposure(1))

        self.lbl_exp_val = QLabel("50.0")
        self.lbl_exp_val.setFixedWidth(38)
        self.lbl_exp_val.setStyleSheet("font-weight: bold; color: #0078d7; font-size: 10pt;")
        self.lbl_exp_val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        exp_row.addWidget(self.btn_exp_down)
        exp_row.addWidget(self.sld_exposure)
        exp_row.addWidget(self.btn_exp_up)
        exp_row.addWidget(self.lbl_exp_val)

        exp_layout.addWidget(lbl_exp_title)
        exp_layout.addLayout(exp_row)

        self.exposure_value = 50.0
        self._pending_exp_short = False
        self._pending_exp_val = 50.0

        # Debounce: sólo envía el comando a la cámara 120ms después del último cambio
        # Evita bombardear la cámara con comandos GigE al arrastrar el slider
        self._exp_debounce = QTimer()
        self._exp_debounce.setSingleShot(True)
        self._exp_debounce.setInterval(120)
        self._exp_debounce.timeout.connect(self._apply_exposure_to_camera)

        cb_layout.addLayout(exp_layout)
        
        cb_layout.addSpacing(10)
        
        # D. Selector de formato de captura
        self.combo_pixel_mode = QComboBox()
        self.combo_pixel_mode.addItems(["RAW Bayer 12-bit", "QOI RGB8 (ISP)"])
        self.combo_pixel_mode.setToolTip("Formato de captura: Bayer crudo o RGB procesado por la cámara")
        self.combo_pixel_mode.setFixedWidth(145)
        current_mode = self._read_pixel_mode()
        if current_mode == "qoi_rgb":
            self.combo_pixel_mode.setCurrentIndex(1)
        self.combo_pixel_mode.currentIndexChanged.connect(self.on_pixel_mode_change)
        cb_layout.addWidget(self.combo_pixel_mode)
        
        # E. GRABAR
        self.btn_record = QPushButton("GRABAR")
        self.btn_record.setCheckable(True)
        self.btn_record.setEnabled(False)
        self.btn_record.setStyleSheet("""
            QPushButton { background-color: #d32f2f; color: white; border-radius: 4px; font-weight: bold; }
            QPushButton:checked { background-color: #b71c1c; border: 2px solid white; }
            QPushButton:disabled { background-color: #555; }
        """)
        self.btn_record.setMinimumWidth(100)
        self.btn_record.setMinimumHeight(35)
        self.btn_record.clicked.connect(self.toggle_recording)
        
        cb_layout.addWidget(self.btn_record)
        
        scan_layout.addWidget(control_bar)
        
        self.tab_scan.setLayout(scan_layout)

        # TAB 2: VISOR
        self.tab_view = QWidget()
        view_layout = QVBoxLayout()
        self.viewer_play = QLabel("Visor"); self.viewer_play.setAlignment(Qt.AlignmentFlag.AlignCenter); self.viewer_play.setStyleSheet("background:#222;color:#fff")
        self.viewer_play.setSizePolicy(
            self.viewer_play.sizePolicy().Policy.Ignored, 
            self.viewer_play.sizePolicy().Policy.Ignored
        )
        
        ctrl_lay = QHBoxLayout()
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.valueChanged.connect(self.seek_viewer_manual)
        self.lbl_frame_info = QLabel("0/0")
        ctrl_lay.addWidget(self.lbl_frame_info); ctrl_lay.addWidget(self.slider_frame)

        play_lay = QHBoxLayout()
        self.btn_play = QPushButton("▶"); self.btn_play.setCheckable(True); self.btn_play.clicked.connect(self.toggle_playback)
        self.sb_fps = QSpinBox(); self.sb_fps.setRange(1,60); self.sb_fps.setValue(18)
        self.sb_fps.valueChanged.connect(self.update_fps_metadata)
        self.sb_fps.valueChanged.connect(lambda _: self.schedule_save_config())
        play_lay.addWidget(self.btn_play); play_lay.addWidget(QLabel("FPS:")); play_lay.addWidget(self.sb_fps)

        exp_grp = QGroupBox("Exportación")
        exp_lay = QHBoxLayout()
        btn_tif = QPushButton("Exportar secuencia TIF"); btn_tif.clicked.connect(self.export_tif)
        btn_batch = QPushButton("📁 Exportar Video"); btn_batch.clicked.connect(self.open_batch_export_window)
        exp_lay.addWidget(btn_tif); exp_lay.addWidget(btn_batch)
        exp_grp.setLayout(exp_lay)

        view_layout.addWidget(self.viewer_play, 1)
        view_layout.addLayout(play_lay); view_layout.addLayout(ctrl_lay); view_layout.addWidget(exp_grp)
        self.tab_view.setLayout(view_layout)

        self.tabs.addTab(self.tab_scan, "Captura"); self.tabs.addTab(self.tab_view, "Visor")
        self.tabs.currentChanged.connect(self._sync_execution_state)
        self.tabs.currentChanged.connect(lambda _: self.schedule_save_config())
        self._sync_execution_state()
        splitter = QSplitter(); splitter.addWidget(left_panel); splitter.addWidget(self.tabs); splitter.setSizes([300,900])
        main_layout.addWidget(splitter)

        self.play_timer = QTimer(); self.play_timer.timeout.connect(self.next_frame_playback)
        self.export_queue = []; self.is_exporting_batch = False

    def show_toast(self, text, duration_ms=3000):
        self._toast.setText(text)
        self._toast.adjustSize()
        margin = 16
        x = max(margin, self.width() - self._toast.width() - margin)
        y = margin
        self._toast.move(x, y)
        self._toast.show()
        self._toast.raise_()
        self._toast_timer.start(duration_ms)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._toast.isVisible():
            margin = 16
            x = max(margin, self.width() - self._toast.width() - margin)
            self._toast.move(x, margin)

    def start_camera_thread(self):
        if self.camera_worker and self.camera_worker.isRunning():
            return
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker.wait()
        self.init_camera_thread()

    def _export_blocks_system_sleep(self) -> bool:
        """Cola por lotes o worker TIF activo."""
        if getattr(self, "is_exporting_batch", False):
            return True
        tw = getattr(self, "tif_worker", None)
        if tw is not None and tw.isRunning():
            return True
        return False

    def _sync_execution_state(self, index: int | None = None) -> None:
        """Pantalla: encendida fuera del Visor (índice 1). Suspensión del sistema: evitada fuera del Visor y mientras exporta."""
        if index is None:
            index = self.tabs.currentIndex()
        want_display = index != 1
        want_system = (index != 1) or self._export_blocks_system_sleep()
        key = (want_display, want_system)
        if self._execution_state_key == key:
            return
        self._execution_state_key = key
        _windows_set_execution_state(want_display, want_system)

    # --- LISTENER DE EVENTOS (Event Filter) ---
    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and source == self.tabs:
            if self.tabs.currentIndex() == 1:
                # Teclas de navegación
                if event.key() == Qt.Key.Key_Left:
                    self.slider_frame.setValue(self.slider_frame.value() - 1)
                    return True
                elif event.key() == Qt.Key.Key_Right:
                    self.slider_frame.setValue(self.slider_frame.value() + 1)
                    return True
                
                # --- NUEVO: TECLA 'C' PARA CAMBIAR PATRÓN BAYER ---
                elif event.key() == Qt.Key.Key_C:
                    self.bayer_phase = (self.bayer_phase + 1) % 4
                    print(f"Cambiando Patrón Bayer a índice: {self.bayer_phase}")
                    self.seek_viewer(self.slider_frame.value())
                    self.schedule_save_config()
                    return True
                # --------------------------------------------------
        
        return super().eventFilter(source, event)

    # --- MÉTODOS DE COLECCIONES ---
    def refresh_collections(self):
        self.col_list.clear()
        self.col_list.addItems(self.manager.get_collections())

    def create_collection(self):
        name, ok = QInputDialog.getText(self, "Nueva", "Nombre:")
        if not ok:
            return
        name = name.strip()
        if not name or "/" in name or "\\" in name or ":" in name:
            if name:
                QMessageBox.warning(self, "Nombre inválido", "Use un nombre simple, sin rutas.")
            return
        if not self.manager.create_collection(name):
            QMessageBox.warning(self, "Error", "Ya existe una colección con ese nombre.")
            return
        self.refresh_collections()

    def on_collection_select(self, item):
        coll_name = item.text()
        self.refresh_file_list(coll_name)

    def refresh_file_list(self, coll_name):
        self.file_list.clear()
        p = os.path.join(self.root_folder, coll_name)
        if os.path.exists(p):
            files = sorted([f for f in os.listdir(p) if f.endswith(".raw")])
            self.file_list.addItems(files)

    def activate_collection(self, item):
        self.active_collection = item.text()
        self.lbl_status.setText(f"Colección Activa: {self.active_collection}")
        self.lbl_status.setStyleSheet("font-size: 14pt; color: #4caf50;")
        can_record = self._camera_connected
        self.btn_record.setEnabled(can_record)
        if can_record:
            self.btn_record.setText("INICIAR CAPTURA")
            self.btn_record.setStyleSheet("background-color: #0078d7; font-size: 14pt; color: white;")
        else:
            self.btn_record.setText("SIN CÁMARA")
            self.btn_record.setStyleSheet("background-color: #555; font-size: 14pt; color: #aaa;")
        self.tabs.setCurrentIndex(0)
        self.schedule_save_config()

    def _collection_for_file_panel(self):
        it = self.col_list.currentItem()
        return it.text() if it else None

    def open_collection_context_menu(self, pos):
        it = self.col_list.itemAt(pos)
        if not it:
            return
        self.col_list.setCurrentItem(it)
        menu = QMenu(self)
        a_rn = QAction("Renombrar colección…", self)
        a_rn.triggered.connect(self.rename_selected_collection)
        menu.addAction(a_rn)
        a_pat = QAction("Patrón de archivos al grabar…", self)
        a_pat.triggered.connect(self.edit_collection_filename_pattern)
        menu.addAction(a_pat)
        menu.exec(self.col_list.mapToGlobal(pos))

    def rename_selected_collection(self):
        it = self.col_list.currentItem()
        if not it:
            return
        old = it.text()
        new_name, ok = QInputDialog.getText(
            self, "Renombrar colección", "Nuevo nombre de carpeta:", text=old)
        if not ok or not new_name.strip() or new_name.strip() == old:
            return
        new_name = new_name.strip()
        if "/" in new_name or "\\" in new_name or ":" in new_name:
            QMessageBox.warning(self, "Nombre inválido", "Use un nombre simple, sin rutas.")
            return
        if not self.manager.rename_collection(old, new_name):
            QMessageBox.warning(
                self, "Error",
                "No se pudo renombrar la colección (¿ya existe o nombre inválido?).")
            return
        if self.active_collection == old:
            self.active_collection = new_name
            self.lbl_status.setText(f"Colección Activa: {self.active_collection}")
            self.schedule_save_config()
        self.refresh_collections()
        for i in range(self.col_list.count()):
            if self.col_list.item(i).text() == new_name:
                self.col_list.setCurrentRow(i)
                break
        self.refresh_file_list(new_name)

    def edit_collection_filename_pattern(self):
        it = self.col_list.currentItem()
        if not it:
            return
        coll = it.text()
        cfg = self.manager.get_collection_config(coll)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Patrón de archivos — {coll}")
        dlg.setMinimumWidth(500)
        lay = QVBoxLayout(dlg)
        help_txt = QLabel(
            "Plantilla para el próximo archivo al grabar (formato str.format).\n"
            "· {coleccion} — nombre de la carpeta de esta colección\n"
            "· {n} — contador; use {n:04d} para 4 dígitos con ceros, {n:03d} para 3, etc.\n"
            "Debe terminar en .raw. Ejemplo: {coleccion}_{n:04d}.raw"
        )
        help_txt.setWordWrap(True)
        lay.addWidget(help_txt)
        edit = QLineEdit(cfg.get("filename_pattern", "{coleccion}_{n:03d}.raw"))
        lay.addWidget(edit)
        preview = QLabel("")
        preview.setWordWrap(True)
        preview.setStyleSheet("color: #888;")
        lay.addWidget(preview)

        def upd_preview():
            try:
                ex = self.manager.peek_next_filename(coll, edit.text().strip())
                preview.setText(f"Siguiente archivo (ejemplo): {ex}")
            except Exception as e:
                preview.setText(f"Vista previa no disponible: {e}")

        edit.textChanged.connect(lambda _t: upd_preview())
        upd_preview()
        row = QHBoxLayout()
        btn_ok = QPushButton("Guardar")
        btn_cancel = QPushButton("Cancelar")
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        lay.addLayout(row)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        pat = edit.text().strip()
        try:
            self.manager.validate_filename_pattern(coll, pat)
        except ValueError as e:
            QMessageBox.warning(self, "Patrón inválido", str(e))
            return
        self.manager.save_collection_config(coll, {"filename_pattern": pat})

    # --- VISOR Y PLAYBACK ---
    def load_file_in_viewer(self, item):
        if not self.col_list.currentItem(): return
        coll = self.col_list.currentItem().text()
        filename = item.text()
        filepath = os.path.join(self.root_folder, coll, filename)
        if not os.path.exists(filepath): return

        self.current_view_file = filepath
        
        # 1. LEER METADATA
        meta = self.manager.get_file_info(coll, filename)
        fps = meta.get("fps", 18)
        roi_key = meta.get("roi_key", "Full Sensor")
        pixel_fmt = meta.get("pixel_format", "bayer") 
        
        # 2. CONFIGURAR UI
        self.sb_fps.blockSignals(True); self.sb_fps.setValue(fps); self.sb_fps.blockSignals(False)
        self.tabs.setCurrentIndex(1)

        # 3. DIMENSIONES
        # 3. DIMENSIONES (Hardcoded)
        self.raw_width, self.raw_height = 2840, 2200

        self.view_is_rgb = (pixel_fmt == "rgb")
        self.view_is_qoi = (pixel_fmt == "qoi_rgb")
        self.qoi_frame_index = None
        
        fsize = os.path.getsize(filepath)

        if self.view_is_qoi:
            # QOI: Frames de tamaño variable, necesitamos indexar
            print(f"LOG QOI VIEWER: Indexando frames QOI en {filename}...")
            self.qoi_frame_index = qoi_utils.build_frame_index(filepath)
            self.total_frames = len(self.qoi_frame_index)
            self.bytes_per_frame = 0  # No aplica (variable)
            print(f"LOG QOI VIEWER: {self.total_frames} frames indexados.")
        else:
            # Bayer o RGB: Tamaño fijo por frame
            bytes_per_pixel = 3 if self.view_is_rgb else 1.5
            math_size = int(self.raw_width * self.raw_height * bytes_per_pixel)
            self.bytes_per_frame = math_size
            
            if math_size > 0:
                approx_frames = round(fsize / math_size)
                if approx_frames > 0:
                    real_stride = fsize // approx_frames
                    if real_stride != math_size:
                        self.bytes_per_frame = real_stride
                        print(f"INFO: Padding detectado. Math: {math_size} -> Real en disco: {self.bytes_per_frame}")

            self.total_frames = fsize // self.bytes_per_frame if self.bytes_per_frame > 0 else 0
        
        # Info UI
        mode_str = "QOI_RGB" if self.view_is_qoi else ("RGB" if self.view_is_rgb else "RAW")
        self.lbl_frame_info.setText(f"{filename} | {self.total_frames}f | {roi_key} | {mode_str}")
        self.slider_frame.setRange(0, max(0, self.total_frames-1))
        
        self.seek_viewer(0)

    def seek_viewer(self, idx, fast=False):
        if not hasattr(self, 'current_view_file'): return
        
        # 1. Preparar lectura
        w, h = self.raw_width, self.raw_height
        is_rgb_view = getattr(self, 'view_is_rgb', False)
        is_qoi_view = getattr(self, 'view_is_qoi', False)
        
        try:
            # --- LECTURA QOI (contenedor con headers de tamaño) ---
            if is_qoi_view and self.qoi_frame_index:
                if idx >= len(self.qoi_frame_index):
                    return
                offset, frame_size = self.qoi_frame_index[idx]
                qoi_data = qoi_utils.read_frame_at(self.current_view_file, offset, frame_size)
                arr = qoi_utils.decode_qoi(qoi_data, w, h)
                
                if fast:
                    arr = cv2.resize(arr, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_NEAREST)
                
                if not arr.flags['C_CONTIGUOUS']: arr = np.ascontiguousarray(arr)
                h_out, w_out, _ = arr.shape
                self._temp_img_ref = arr
                qimg = QImage(arr.data, w_out, h_out, w_out*3, QImage.Format.Format_RGB888)
                
                pix = QPixmap.fromImage(qimg)
                self.viewer_play.setPixmap(pix.scaled(self.viewer_play.size(), Qt.AspectRatioMode.KeepAspectRatio))
                tag = "[FAST]" if fast else "[QOI]"
                self.lbl_frame_info.setText(f"{idx}/{self.total_frames} {tag}")
                return

            # --- LECTURA ESTÁNDAR (Bayer / RGB) ---
            frame_size = int(w * h * 3) if is_rgb_view else int(w * h * 1.5)
            offset = idx * frame_size
        
            with open(self.current_view_file, "rb") as f:
                f.seek(offset)
                data = f.read(frame_size)
                
                if len(data) == 0: return # Fin de archivo
                
                # Relleno de seguridad si el frame está incompleto
                if len(data) < frame_size:
                    data += b'\x00' * (frame_size - len(data))

                # --- PROCESAMIENTO ---
                
                if is_rgb_view:
                    # CASO RGB
                    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                    
                    # Subsampling para reproducción fluida
                    if fast:
                        # Reducimos a un 33% (aprox 900px ancho) para volar en FPS
                        arr = cv2.resize(arr, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_NEAREST)
                    
                    if not arr.flags['C_CONTIGUOUS']: arr = np.ascontiguousarray(arr)
                    h_out, w_out, _ = arr.shape
                    # RGB ya viene en 8 bit, directo a QImage
                    qimg = QImage(arr.data, w_out, h_out, w_out*3, QImage.Format.Format_RGB888)

                else:
                    if fast:
                        # --- MODIFICACIÓN ULTRA-RÁPIDA (Strided Packed Access) ---
                        # Evitamos descomprimir toda la imagen. Accedemos directo a los bytes.
                        # data es un array plano de H * (W/2) * 3 bytes
                        
                        # 1. Vista estructurada del buffer crudo (H, W/2, 3)
                        # Nota: W/2 porque cada 3 bytes son 2 pixeles.
                        # Stride vertical y horizontal de 2 (tomamos 1 de cada 4 píxeles reales)
                        # Resultado: Imagen 1/4 (710x550)
                        
                        raw_view = np.frombuffer(data, dtype=np.uint8).reshape(h, w//2, 3)
                        
                        stride = 2
                        
                        # RG Rows (Filas Pares): Contienen R(p0) y G(p1)
                        # GB Rows (Filas Impares): Contienen G(p0) y B(p1)
                        
                        # Slice [Filas, Columnas, Bytes]
                        chunk_rg = raw_view[0::2*stride, 0::stride, :] 
                        chunk_gb = raw_view[1::2*stride, 0::stride, :] # Offset fila 1 para GB
                        
                        # --- Unpack RG Chunk ---
                        b0 = chunk_rg[:, :, 0].astype(np.uint16)
                        b1 = chunk_rg[:, :, 1].astype(np.uint16)
                        b2 = chunk_rg[:, :, 2].astype(np.uint16)
                        
                        # R está en p0, G está en p1
                        r_ch = b0 | ((b1 & 0x0F) << 8)
                        g_ch = (b1 >> 4) | (b2 << 4)
                        
                        # --- Unpack GB Chunk ---
                        # Solo nos interesa B (p1)
                        b1_g = chunk_gb[:, :, 1].astype(np.uint16)
                        b2_g = chunk_gb[:, :, 2].astype(np.uint16)
                        
                        b_ch = (b1_g >> 4) | (b2_g << 4)
                        
                        # Asegurar tamaños iguales (por si el slice impar queda corto)
                        min_h = min(r_ch.shape[0], b_ch.shape[0])
                        r_ch = r_ch[:min_h, :]
                        g_ch = g_ch[:min_h, :]
                        b_ch = b_ch[:min_h, :]
                        
                        # --- Fake Gamma & 8-bit conversion (Muy rápido) ---
                        # Usamos convertScaleAbs: (src * alpha + beta) -> uint8 saturado
                        # Alpha: Gain. 255/4095 = 0.062 (Linear). 
                        # Usamos 0.2 (~3.2x gain) para simular gamma/brillo sin math float.
                        gain = 0.2 
                        
                        r_8 = cv2.convertScaleAbs(r_ch, alpha=gain)
                        g_8 = cv2.convertScaleAbs(g_ch, alpha=gain)
                        b_8 = cv2.convertScaleAbs(b_ch, alpha=gain)
                        
                        rgb8 = np.dstack((r_8, g_8, b_8))
                        
                    else:
                        rgb8 = bayer_render.render_capture_view(data, w, h, downscale=1, to_bgr=False)

                    # QIMAGE (Común)
                    h_out, w_out, _ = rgb8.shape
                    if not rgb8.flags['C_CONTIGUOUS']: rgb8 = np.ascontiguousarray(rgb8)
                    self._temp_img_ref = rgb8 
                    qimg = QImage(rgb8.data, w_out, h_out, w_out*3, QImage.Format.Format_RGB888)

                # --- VISUALIZACIÓN ---
                pix = QPixmap.fromImage(qimg)
                # Escalamos al tamaño del visor (esto lo hace la GPU/Qt, es rápido)
                self.viewer_play.setPixmap(pix.scaled(self.viewer_play.size(), Qt.AspectRatioMode.KeepAspectRatio))
                
                if not fast:
                    self.lbl_frame_info.setText(f"{idx}/{self.total_frames} [H.Q.]")
                else:
                    self.lbl_frame_info.setText(f"{idx}/{self.total_frames} [FAST]")

        except Exception as e:
            print(f"Error Viewer: {e}")
    def seek_viewer_manual(self, val):
        self.seek_viewer(val, fast=self.play_timer.isActive())

    def toggle_playback(self, a):
        if a:
            self.btn_play.setText("⏸")
            self.play_timer.start(int(1000/self.sb_fps.value()))
        else:
            self.btn_play.setText("▶")
            self.play_timer.stop()

    def next_frame_playback(self):
        nxt = self.slider_frame.value() + 1
        if nxt >= self.total_frames:
            # LOOP: Volver al principio en lugar de detenerse
            self.slider_frame.setValue(0)
        else:
            self.slider_frame.setValue(nxt)

    def update_fps_metadata(self, val):
        # 1. Guardar en metadata (como antes)
        if hasattr(self, 'current_view_file') and self.active_collection:
            fname = Path(self.current_view_file).name
            self.manager.set_fps(self.active_collection, fname, val)
        
        # 2. ACTUALIZACIÓN EN VIVO:
        # Si está reproduciendo, reiniciamos el timer con la nueva velocidad YA.
        if self.play_timer.isActive():
            self.play_timer.setInterval(int(1000/val))

    # --- CÁMARA Y GRABACIÓN ---
    def init_camera_thread(self):
        mode = self._read_pixel_mode()
        self.camera_worker = scanner_core.CameraWorker("1.0.txt", pixel_mode=mode)
        self.camera_worker.image_received.connect(self.update_display)
        self.camera_worker.exposure_stats_updated.connect(self.update_exposure_overlay)
        self.camera_worker.stats_updated.connect(self.update_stats)
        self.camera_worker.error_occurred.connect(self.on_camera_error)
        self.camera_worker.camera_status_changed.connect(self.on_camera_status)
        self.camera_worker.config_applied.connect(self.on_config_applied)
        self.camera_worker.start()

    def update_display(self, frame):
        if self.tabs.currentIndex() != 0: return

        # 1. DETECCIÓN Y CORRECCIÓN DE COLOR
        is_color = (frame.ndim == 3)
        
        if is_color:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            disp = frame_rgb
        else:
            disp = frame

        h, w = disp.shape[:2]
        
        # PEAKING (Focus Assist)
        if self.btn_peaking.isChecked():
            if is_color: gray_peak = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
            else: gray_peak = disp
            
            lap = cv2.Laplacian(gray_peak, cv2.CV_16S, ksize=3)
            _, mask = cv2.threshold(cv2.convertScaleAbs(lap), 40, 255, cv2.THRESH_BINARY)
            
            m = (mask > 0)
            if is_color:
                if not disp.flags['WRITEABLE']: disp = disp.copy()
                disp[m] = [0, 255, 0] 
            else:
                if not disp.flags['WRITEABLE']: disp = disp.copy()
                disp[m] = 255 

        # CREAR QIMAGE
        if is_color:
            qimg = QImage(disp.data, w, h, w*3, QImage.Format.Format_RGB888)
        else:
            if disp.dtype == np.uint16: disp = (disp >> 4).astype(np.uint8)
            if not disp.flags['C_CONTIGUOUS']: disp = np.ascontiguousarray(disp)
            qimg = QImage(disp.data, w, h, w, QImage.Format.Format_Grayscale8)

        # ENVIAR AL VISOR
        self.viewer_scan.setPixmap(QPixmap.fromImage(qimg))

    def update_exposure_overlay(self, hi_pct, _lo_pct=None):
        if hi_pct >= 95.0:
            color = "#ff4444"
        elif hi_pct >= 85.0:
            color = "#ff8800"
        else:
            color = "#66cc66"
        self.viewer_scan.set_exposure_overlay(f"Hi {hi_pct:.0f}%", color)

    def update_stats(self, fps, temp, qsize, cam_drops=0, disk_drops=0, total_cam=0, bw=0.0, bw_src="C"):
        self._stats_cache = {
            "fps": fps,
            "temp": temp,
            "qsize": qsize,
            "cam_drops": cam_drops,
            "disk_drops": disk_drops,
            "total_cam": total_cam,
            "bw": bw,
            "bw_src": bw_src,
        }
        self._apply_stats_inline()

    def _apply_stats_inline(self):
        c = self._stats_cache
        fps = c.get("fps", 0.0)
        temp = c.get("temp", 0.0)
        qsize = c.get("qsize", 0)
        cam_drops = c.get("cam_drops", 0)
        disk_drops = c.get("disk_drops", 0)
        total_cam = c.get("total_cam", 0)
        bw = c.get("bw", 0.0)
        bw_src = c.get("bw_src", "C")
        txt_cam = f"📷 R:{total_cam} D:{cam_drops}"
        parts = [f"FPS: {fps:.1f}", f"Tmp: {temp:.1f}°", f"Buf: {qsize}"]
        if self.is_recording and self.writer_worker:
            curr_saved = self.writer_worker.frames_saved
            parts.append(f"💾 S:{curr_saved} D:{disk_drops}")
        parts.append(f"BW({bw_src}): {int(bw)} Mbps")
        parts.append(txt_cam)
        self.lbl_stats_inline.setText("  ·  ".join(parts))
        base = "font-size: 9pt; margin-left: 4px;"
        if self.is_recording and self.writer_worker:
            if cam_drops > 0 or disk_drops > 0:
                self.lbl_stats_inline.setStyleSheet(base + " color: red; font-weight: bold;")
            else:
                self.lbl_stats_inline.setStyleSheet(base + " color: #4caf50;")
        else:
            if cam_drops > 0:
                self.lbl_stats_inline.setStyleSheet(base + " color: red; font-weight: bold;")
            else:
                self.lbl_stats_inline.setStyleSheet(base + " color: #888;")

    def open_camera_calibration_dialog(self):
        if self.is_recording:
            QMessageBox.information(
                self, "Grabando",
                "Detén la grabación antes de abrir la calibración de cámara.")
            return
        if not self.camera_worker or not self._camera_connected:
            QMessageBox.warning(self, "Sin cámara", "No hay cámara conectada.")
            return
        if self._calibration_dialog is not None:
            self._calibration_dialog.raise_()
            self._calibration_dialog.activateWindow()
            return
        dlg = CameraCalibrationDialog(self)
        self._calibration_dialog = dlg
        dlg.finished.connect(self._on_calibration_dialog_closed)
        self._update_calibrate_camera_btn()
        dlg.show()

    def _on_calibration_dialog_closed(self, _result):
        self._calibration_dialog = None
        self._update_calibrate_camera_btn()

    def _update_calibrate_camera_btn(self):
        if not hasattr(self, "btn_calibrate_camera"):
            return
        self.btn_calibrate_camera.setEnabled(
            self._camera_connected and not self.is_recording and self._calibration_dialog is None)
        if hasattr(self, "btn_timer0"):
            self.btn_timer0.setEnabled(
                self._camera_connected and not self.is_recording and self._timer_dialog is None
            )

    def open_timer_dialog(self):
        if self.is_recording:
            QMessageBox.information(
                self, "Grabando",
                "Detén la grabación antes de ajustar Timer0.")
            return
        if not self.camera_worker or not self._camera_connected:
            QMessageBox.warning(self, "Sin cámara", "No hay cámara conectada.")
            return
        if self._timer_dialog is not None:
            self._timer_dialog.raise_()
            self._timer_dialog.activateWindow()
            return
        dlg = CameraTimerDialog(self)
        self._timer_dialog = dlg
        dlg.finished.connect(self._on_timer_dialog_closed)
        self._update_calibrate_camera_btn()
        dlg.show()

    def _on_timer_dialog_closed(self, _result):
        self._timer_dialog = None
        self._update_calibrate_camera_btn()

    def on_camera_error(self, e): QMessageBox.critical(self, "Cam Error", e)

    def on_camera_status(self, connected):
        was_connected = self._camera_connected
        self._camera_connected = connected
        if connected:
            self.lbl_cam_status.setText("●")
            self.lbl_cam_status.setStyleSheet("color: #4caf50; font-size: 14pt; margin-right: 4px;")
            self.lbl_cam_status.setToolTip("Cámara conectada")
            if not was_connected:
                self._pending_config_toast = True
                self.show_toast("Cámara conectada", 2500)
            if self.active_collection and not self.is_recording:
                self.btn_record.setEnabled(True)
                self.btn_record.setText("INICIAR CAPTURA")
                self.btn_record.setStyleSheet("background-color: #0078d7; font-size: 14pt; color: white;")
        else:
            self.lbl_cam_status.setText("●")
            self.lbl_cam_status.setStyleSheet("color: #f44336; font-size: 14pt; margin-right: 4px;")
            self.lbl_cam_status.setToolTip("Cámara no detectada — reintentando…")
            self._pending_config_toast = False
            if was_connected:
                self.show_toast("Cámara desconectada — reintentando…", 3500)
            if not self.is_recording:
                self.btn_record.setEnabled(False)
                self.btn_record.setText("SIN CÁMARA")
        self._update_calibrate_camera_btn()

    def on_config_applied(self):
        if self._pending_config_toast:
            self._pending_config_toast = False
            QTimer.singleShot(600, lambda: self.show_toast("Configuración de cámara aplicada", 3000))
        self._apply_persist_to_camera()

    def on_exposure_slider_changed(self, idx):
        val = self.EXP_STEPS[idx]
        self.exposure_value = val
        short_mode = (idx == self.SHORT_EXP_IDX)

        # --- Actualizar UI de inmediato (sin esperar a la cámara) ---
        if short_mode:
            self.lbl_exp_val.setText("SHORT")
            self.lbl_exp_val.setStyleSheet("font-weight: bold; color: #ff9800; font-size: 9pt;")
            self.sld_exposure.setToolTip("⚠ SHORT EXPOSURE (~2.3 µs) — ShortExposureEnable activo")
        else:
            self.lbl_exp_val.setText(f"{val}")
            self.lbl_exp_val.setStyleSheet("font-weight: bold; color: #0078d7; font-size: 10pt;")
            self.sld_exposure.setToolTip(f"{val} µs — paso 2.5 µs (modo normal)")

        # --- Encolar comando a la cámara con debounce ---
        # Solo se envía el comando 120ms después del último cambio,
        # evitando saturar la cámara al arrastrar el slider rápido
        self._pending_exp_short = short_mode
        self._pending_exp_val = val
        self._exp_debounce.start()
        self.schedule_save_config()

    def _apply_exposure_to_camera(self):
        """Llamado por el debounce timer; envía el último valor a la cámara."""
        if self.camera_worker:
            self.camera_worker.set_exposure_config(
                self._pending_exp_short, self._pending_exp_val)

    def step_exposure(self, delta):
        """Mueve el slider un paso (botones ◀/▶)."""
        cur = self.sld_exposure.value()
        self.sld_exposure.setValue(max(0, min(len(self.EXP_STEPS) - 1, cur + delta)))

    def _read_pixel_mode(self):
        ui = getattr(self, "_persist", {}).get("ui", {})
        return ui.get("pixel_mode", "bayer")

    def on_pixel_mode_change(self, index):
        mode = "qoi_rgb" if index == 1 else "bayer"
        self._persist.setdefault("ui", {})["pixel_mode"] = mode
        self.schedule_save_config()
        if getattr(self, "_applying_persist", False):
            return
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker.wait()
        self._camera_connected = False
        self.lbl_cam_status.setStyleSheet("color: #ff9800; font-size: 14pt; margin-right: 4px;")
        self.lbl_cam_status.setToolTip("Reconectando cámara...")
        self.start_camera_thread()

    # --- CONTROLADORES DE VISTA ---
    
    def toggle_zoom_1to1(self, checked):
        # Mutual exclusion
        if checked:
            self.btn_corners.setChecked(False)
            self.btn_corners.setStyleSheet("") # Clear style of other
            self.viewer_scan.mode = ViewMode.ZOOM_1_1
            self.viewer_scan.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.viewer_scan.mode = ViewMode.NORMAL
            self.viewer_scan.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Style self
        self.btn_zoom_1to1.setStyleSheet("background:red" if checked else "")
        self.viewer_scan.update()
        self.schedule_save_config()

    def toggle_corners(self, checked):
        # Mutual exclusion
        if checked:
            self.btn_zoom_1to1.setChecked(False)
            self.btn_zoom_1to1.setStyleSheet("") # Clear style of other
            self.viewer_scan.mode = ViewMode.CORNERS
            self.viewer_scan.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.viewer_scan.mode = ViewMode.NORMAL

        # Style self
        self.btn_corners.setStyleSheet("background:red" if checked else "")
        self.viewer_scan.update()
        self.schedule_save_config()

    def on_preview_mode_changed(self, label):
        mode_map = {
            "ISP": "isp", "ISP Full": "isp_full", "HQ": "hq",
            "HQ½": "hq_half", "RAW": "raw", "B/W": "bw",
        }
        mode = mode_map.get(label, "isp")
        if self.camera_worker:
            self.camera_worker.set_preview_mode(mode)
        self.schedule_save_config()

    def toggle_real_fps(self, checked):
        if self.camera_worker:
            #checked = TRUE -> Real FPS -> Skip Frames = FALSE
            self.camera_worker.set_preview_skip_frames(not checked)
        self.btn_real_fps.setStyleSheet("background:red" if checked else "")
        self.schedule_save_config()


    def toggle_peaking(self, checked):
        self.btn_peaking.setStyleSheet("background:red" if checked else "")
        self._apply_preview_gamma_for_peaking()
        self.schedule_save_config()
    
    # toggle_zoom y toggle_zoom_state eliminados/reemplazados por toggle_zoom_1to1


    def toggle_recording(self):
        if not self.is_recording:
            if not self.active_collection:
                QMessageBox.warning(self, "Error", "Selecciona una colección primero.")
                return
            if not self._camera_connected:
                QMessageBox.warning(self, "Sin cámara", "No hay cámara conectada. Espera a que se detecte.")
                return

            if self.camera_worker:
                 self.camera_worker.clear_queue()
                 self.camera_worker.reset_drop_count()
                 self.camera_worker.set_queue(self.frame_queue)
            
            # --- INICIAR GRABACIÓN ---
            # fmt = "Standard" # Eliminado selector
            ftype = self.combo_type.currentText() # Ej: "Color (Pos/Neg)"
            
            # 1. (Ya no se aplica ROI formato porque es fijo)
            
            # 2. Obtener nombres y workers
            try:
                fn, fp = self.manager.get_next_filename(self.active_collection)
            except ValueError as e:
                if self.camera_worker:
                    self.camera_worker.set_queue(None)
                QMessageBox.warning(self, "Nombre de archivo", str(e))
                return
            self.writer_worker = scanner_core.WriterWorker(self.frame_queue, fp)
            self.writer_worker.frames_saved_signal.connect(lambda _: self._apply_stats_inline())
            self.writer_worker.start()
            
            # 3. --- GUARDADO DE METADATA (LO IMPORTANTE) ---
            # Obtenemos el modo de pixel actual del worker (o 'bayer' por defecto)
            pixel_mode = getattr(self.camera_worker, "pixel_mode", "bayer")

            # Mapeamos a un descriptor de formato de archivo:
            # - "bayer"   -> guardamos RAW Bayer 12-bit empaquetado  -> pixel_format = "bayer"
            # - "qoi_rgb" -> guardamos RGB8 procesado por Arena SDK  -> pixel_format = "rgb"
            if pixel_mode == "qoi_rgb":
                pixel_format = "rgb"
            else:
                pixel_format = pixel_mode
            
            # Guardamos todo explícitamente usando argumentos con nombre (kwargs)
            self.manager.set_file_info(
                self.active_collection, 
                fn, 
                fps=self.sb_fps.value(),
                roi_key="Standard 2840x2200", # Valor fijo
                film_type=ftype,
                pixel_format=pixel_format
            )
            # -----------------------------------------------

            self.is_recording = True
            self.btn_record.setText(f"DETENER ({fn})")
            self.btn_record.setStyleSheet("background:red;color:white;font-weight:bold")
            self.combo_type.setEnabled(False)
            self.combo_pixel_mode.setEnabled(False)

        else:
            reply = QMessageBox.question(
                self,
                "Detener grabación",
                "¿Detener la grabación en curso?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            # --- DETENER GRABACIÓN ---
            self.is_recording = False
            if self.camera_worker:
                 self.camera_worker.set_queue(None)
                 self.camera_worker.reset_drop_count()

            if self.writer_worker:
                self.writer_worker.stop()
                self.writer_worker = None
            self.btn_record.setText("GRABAR")
            self.btn_record.setStyleSheet("")
            self.combo_type.setEnabled(True)
            self.combo_pixel_mode.setEnabled(True)
            self.refresh_file_list(self.active_collection)

        self._apply_stats_inline()
        self._update_calibrate_camera_btn()

    # --- EXPORTACIÓN ---
    def export_tif(self):
        if not hasattr(self, 'current_view_file'): 
            QMessageBox.warning(self, "Error", "Carga un archivo en el visor primero.")
            return

        target_input = Path(self.current_view_file)
        coll = target_input.parent.name
        coll_meta = self.manager.load_metadata(coll)
        confirm_dlg = ExportConfirmDialog(
            self, [target_input.name], self.root_folder, coll, coll_meta, "tif",
        )
        if not confirm_dlg.exec():
            return
        settings = confirm_dlg.get_all_settings().get(
            target_input.name, default_settings_for_file(target_input.name, coll_meta, "tif"),
        )

        self.pd_tif = QProgressDialog(f"Procesando {target_input.name}...", "Cancelar", 0, 100, self)
        self.pd_tif.setWindowTitle("Exportando Secuencia TIF")
        self.pd_tif.setWindowModality(Qt.WindowModality.ApplicationModal) 
        self.pd_tif.setAutoClose(False)
        self.pd_tif.setValue(0)
        self.pd_tif.show()

        # 2. Configurar Worker
        cmd = [
            sys.executable, "l2t.py", str(target_input),
            "--output-dir", settings["output_name"],
        ]
        
        self.tif_worker = UniversalExportWorker(cmd)
        
        # 3. Conectar Señales (Barra de progreso fluida)
        self.tif_worker.progress_signal.connect(self.pd_tif.setValue)
        self.tif_worker.info_signal.connect(self.pd_tif.setLabelText)
        self.tif_worker.finished_signal.connect(self.on_tif_finished)
        self.pd_tif.canceled.connect(self.on_tif_cancelled)
        
        self.current_tif_output = target_input.parent / settings["output_name"]
        
        # 4. Iniciar
        self.tif_worker.start()
        self._sync_execution_state()

    def on_tif_cancelled(self):
        if self.tif_worker: self.tif_worker.kill()
        
        resp = QMessageBox.question(self, "Cancelado", "¿Desea eliminar la carpeta incompleta generada?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp == QMessageBox.StandardButton.Yes:
            try:
                if hasattr(self, 'current_tif_output') and self.current_tif_output.exists():
                    shutil.rmtree(self.current_tif_output)
                    QMessageBox.information(self, "Info", "Carpeta eliminada.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo borrar: {e}")
        self.pd_tif.close()
        self._sync_execution_state()

    def on_tif_finished(self, success, msg):
        try: self.pd_tif.canceled.disconnect(self.on_tif_cancelled)
        except: pass

        self.pd_tif.close()
        if success:
            # Mensaje menos intrusivo o confirmación simple
            QMessageBox.information(self, "Listo", f"Exportación finalizada.\n{msg}")
        else:
            QMessageBox.critical(self, "Error", f"Fallo en l2t:\n{msg}")
        self._sync_execution_state()

    def open_batch_export_window(self):
        # 1. Determinar qué colección usar
        # Prioridad: Colección Activa (Grabación) -> Colección Seleccionada (Visual)
        target_collection = self.active_collection
        if not target_collection and self.col_list.currentItem():
            target_collection = self.col_list.currentItem().text()
            
        if not target_collection: 
            QMessageBox.warning(self, "Atención", "Selecciona una colección de la lista izquierda primero.")
            return
        
        # 2. Obtener archivos
        # Aseguramos que la lista visual corresponda a la colección objetivo
        # (Si el usuario seleccionó otra cosa en la lista pero tiene activa otra, podría haber desincronización,
        #  así que recargamos la lista visual si es necesario o confiamos en lo que ve el usuario).
        # Por simplicidad, usamos lo que está en la lista visual self.file_list
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "Atención", "La colección está vacía.")
            return

        files = [self.file_list.item(x).text() for x in range(self.file_list.count())]
        
        # 3. Abrir Diálogo (con metadata para detectar formato de captura)
        coll_meta = self.manager.load_metadata(target_collection)
        dlg = BatchExportDialog(self, files, self.root_folder, target_collection, coll_meta)
        
        if dlg.exec():
            sel, fmt, sharp = dlg.get_selection()
            if not sel:
                return

            export_settings = {}
            if fmt != "dng":
                confirm_dlg = ExportConfirmDialog(
                    self, sel, self.root_folder, target_collection, coll_meta, fmt,
                )
                if not confirm_dlg.exec():
                    return
                export_settings = confirm_dlg.get_all_settings()

            for f in sel:
                path = Path(self.root_folder) / target_collection / f
                settings = export_settings.get(
                    f, default_settings_for_file(f, coll_meta, fmt),
                ) if fmt != "dng" else None
                if settings:
                    from export_grade import persist_export_settings
                    persist_export_settings(self.manager, target_collection, f, settings)
                self.export_queue.append((path, fmt, sharp, settings))

            if not self.is_exporting_batch:
                self.process_export_queue()

    def process_export_queue(self):
        if not self.export_queue:
            self.is_exporting_batch = False
            self._sync_execution_state()
            QMessageBox.information(self, "Fin", "Cola terminada.")
            return

        self.is_exporting_batch = True
        nf, fmt, sharp, settings = self.export_queue.pop(0)
        
        # --- CORRECCIÓN CRÍTICA ---
        # No usamos self.active_collection porque puede ser None.
        # Extraemos el nombre de la colección directamente de la carpeta del archivo.
        # nf es: .../Documents/ScanData/NOMBRE_COLECCION/archivo.raw
        collection_name = nf.parent.name
        
        # Ahora pedimos la info usando ese nombre seguro
        info = self.manager.get_file_info(collection_name, nf.name)
        # --------------------------
        
        # Determinar modo BW/COLOR
        mode = "BW" if "Blanco" in info.get("type", "Color") else "COLOR"
        fps = int((settings or {}).get("fps", info.get("fps", 18)))
        output_name = (settings or {}).get("output_name")
        pixel_fmt = info.get("pixel_format", "bayer")

        # Si el usuario pidió DNG pero el archivo es RGB procesado (no Bayer),
        # cambiamos silenciosamente a TIFF sequence (fiel al ISP) y avisamos una vez.
        effective_fmt = fmt
        if fmt == "dng" and pixel_fmt in ("rgb", "qoi_rgb"):
            effective_fmt = "tiff_seq"
            QMessageBox.information(
                self,
                "Aviso de formato",
                f"\"{nf.name}\" fue capturado en RGB procesado por la cámara.\n"
                f"No es posible exportarlo como DNG RAW, se usará TIFF 8-bit Sequence en su lugar."
            )
        
        print(f"Procesando: {nf.name} | Colección: {collection_name} | Modo: {mode} | pixel_fmt={pixel_fmt} | codec={effective_fmt}")
        
        self.pd = QProgressDialog(f"Exportando {nf.name}...", "Cancelar", 0, 100, self)
        self.pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.pd.setAutoClose(False)
        self.pd.setValue(0)
        self.pd.show()
        
        cmd = [
            sys.executable, "raw2video.py", str(nf),
            "--codec", effective_fmt,
            "--fps", str(fps),
            "--sharp", sharp,
            "--mode", mode,
        ]
        if output_name:
            cmd.extend(["--output", output_name])

        grade = (settings or {}).get("grade")
        if grade and grade.get("enabled"):
            cmd.extend(["--grade", json.dumps(grade)])

        if settings and settings.get("test_export"):
            from export_grade import TEST_EXPORT_SKIP_FRAMES, TEST_EXPORT_FRAME_COUNT
            cmd.extend(["--skip-frames", str(TEST_EXPORT_SKIP_FRAMES)])
            cmd.extend(["--max-frames", str(TEST_EXPORT_FRAME_COUNT)])

        if output_name:
            self.current_video_output = nf.parent / output_name
        elif effective_fmt == 'dng':
            self.current_video_output = nf.parent / f"{nf.stem}_DNG_SEQ"
        elif effective_fmt == 'tiff_seq':
            self.current_video_output = nf.parent / f"{nf.stem}_TIFF_SEQ"
        else:
            ext_map = {
                'prores': '.mov', 'prores_hq': '.mov', 'ffv1': '.mkv',
                'h264': '.mp4', 'hevc': '.mp4', 'cineform': '.mov', 'av1': '.mp4',
            }
            ext = ext_map.get(effective_fmt, ".mp4")
            self.current_video_output = nf.parent / f"{nf.stem}_{effective_fmt}{ext}"
        
        self.worker = UniversalExportWorker(cmd)
        self.worker.progress_signal.connect(self.pd.setValue)
        self.worker.info_signal.connect(self.pd.setLabelText)
        self.worker.finished_signal.connect(self.on_batch_item_finished)
        self.pd.canceled.connect(self.on_batch_cancel)
        self.worker.start()
        self._sync_execution_state()

    def on_batch_cancel(self):
        self.export_queue = [] # Detener resto de la cola
        self.is_exporting_batch = False
        if self.worker: self.worker.kill()
        
        resp = QMessageBox.question(self, "Cancelado", "¿Desea eliminar los archivos incompletos generados?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp == QMessageBox.StandardButton.Yes:
            try:
                if hasattr(self, 'current_video_output') and self.current_video_output.exists():
                    if self.current_video_output.is_dir():
                        shutil.rmtree(self.current_video_output)
                        QMessageBox.information(self, "Info", f"Carpeta eliminada: {self.current_video_output.name}")
                    else:
                        os.remove(self.current_video_output)
                        QMessageBox.information(self, "Info", f"Archivo eliminado: {self.current_video_output.name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo borrar: {e}")
        self.pd.close()
        self._sync_execution_state()

    def on_batch_item_finished(self, s, m):
        try: self.pd.canceled.disconnect(self.on_batch_cancel)
        except: pass
        
        if not self.pd.wasCanceled(): # Evitar doble cierre/error
             self.pd.close()
             if not s: print(f"Error export: {m}")
             self.process_export_queue() # Siguiente

    def open_file_context_menu(self, pos):
        it = self.file_list.itemAt(pos)
        if not it:
            return
        self.file_list.setCurrentItem(it)
        menu = QMenu(self)
        act_rn = QAction("Cambiar nombre…", self)
        act_rn.triggered.connect(self.rename_selected_raw_file)
        menu.addAction(act_rn)
        act_mv = QAction("Cambiar de colección…", self)
        act_mv.triggered.connect(self.move_selected_raw_file)
        menu.addAction(act_mv)
        menu.addSeparator()
        act_del = QAction("Borrar", self)
        act_del.triggered.connect(self.delete_selected_file)
        menu.addAction(act_del)
        menu.exec(self.file_list.mapToGlobal(pos))

    def rename_selected_raw_file(self):
        coll = self._collection_for_file_panel()
        it = self.file_list.currentItem()
        if not coll or not it:
            QMessageBox.warning(self, "Selección", "Selecciona una colección y un archivo.")
            return
        old = it.text()
        new_name, ok = QInputDialog.getText(
            self, "Cambiar nombre", "Nuevo nombre (debe ser único y terminar en .raw):",
            text=old)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == old:
            return
        if not self.manager.rename_file(coll, old, new_name):
            QMessageBox.warning(
                self, "Error",
                "No se pudo renombrar: compruebe que el nombre no exista ya y que sea válido.")
            return
        self._refresh_viewer_path_after_file_rename(coll, old, new_name)
        self.refresh_file_list(coll)

    def move_selected_raw_file(self):
        coll = self._collection_for_file_panel()
        it = self.file_list.currentItem()
        if not coll or not it:
            QMessageBox.warning(self, "Selección", "Selecciona una colección y un archivo.")
            return
        fn = it.text()
        others = [c for c in self.manager.get_collections() if c != coll]
        if not others:
            QMessageBox.information(self, "Mover", "No hay otra colección de destino.")
            return
        dst, ok = QInputDialog.getItem(
            self, "Cambiar de colección",
            f"Mover «{fn}» a la colección:", others, 0, False)
        if not ok:
            return
        if not self.manager.move_file_to_collection(coll, dst, fn):
            QMessageBox.warning(
                self, "Error",
                "No se pudo mover (¿ya existe un archivo con el mismo nombre en destino?).")
            return
        self._refresh_viewer_path_after_file_move(coll, dst, fn)
        self.refresh_file_list(coll)

    def _refresh_viewer_path_after_file_rename(self, coll, old_name, new_name):
        if not getattr(self, "current_view_file", None):
            return
        p = Path(self.current_view_file)
        if p.parent.name == coll and p.name == old_name:
            self.current_view_file = str(p.parent / new_name)
            fi = self.lbl_frame_info.text()
            if old_name in fi:
                self.lbl_frame_info.setText(fi.replace(old_name, new_name, 1))

    def _refresh_viewer_path_after_file_move(self, src_coll, dst_coll, fname):
        if not getattr(self, "current_view_file", None):
            return
        p = Path(self.current_view_file)
        if p.parent.name == src_coll and p.name == fname:
            self.current_view_file = str(Path(self.root_folder) / dst_coll / fname)

    def delete_selected_file(self):
        it = self.file_list.currentItem()
        coll = self._collection_for_file_panel()
        if not it or not coll:
            return
        if QMessageBox.question(
                self, "Borrar", f"¿Borrar {it.text()}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        self.manager.delete_file(coll, it.text())
        if hasattr(self, "current_view_file") and self.current_view_file:
            cp = Path(self.current_view_file)
            if cp.parent.name == coll and cp.name == it.text():
                self.viewer_play.clear()
                self.play_timer.stop()
        self.refresh_file_list(coll)

    def update_disk_space(self):
        try: self.lbl_disk.setText(f"Libre: {shutil.disk_usage(self.root_folder).free // 2**30} GB")
        except: pass
    
    def closeEvent(self, e):
        if self.is_recording:
            e.ignore()
            self.show_toast("Captura en curso — detén la grabación antes de cerrar", 3500)
            return
        self.save_config()
        _windows_set_execution_state(False, False)
        self._execution_state_key = None
        if self.camera_worker: self.camera_worker.stop()
        if self.writer_worker: self.writer_worker.stop()
        e.accept()

# --- CLASE WORKER ---
# --- WORKER UNIVERSAL (VIDEO + TIF) ---
# --- WORKER UNIVERSAL MEJORADO ---
class UniversalExportWorker(QThread):
    progress_signal = pyqtSignal(int)
    info_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.process = None
        self.killed = False

    def kill(self):
        self.killed = True
        if self.process:
            try: self.process.kill()
            except: pass

    def run(self):
        try:
            # Flags para ocultar ventana cmd en Windows pero mantener pipes
            kwargs = {}
            if os.name == 'nt':
                kwargs['creationflags'] = 0x08000000
            
            # Unimos stderr y stdout para capturar errores de FFmpeg
            self.process = subprocess.Popen(
                self.cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                encoding='utf-8', 
                errors='replace',
                **kwargs
            )
            
            process = self.process # Alias local
            total_items = 0
            last_lines = [] # Guardaremos las últimas líneas para el reporte de error
            
            while True:
                if self.killed: break # Salir si fue matado
                
                line = process.stdout.readline()
                if not line and process.poll() is not None: break
                
                if line:
                    line = line.strip()
                    # Guardar últimas 10 líneas por si crashea
                    last_lines.append(line)
                    if len(last_lines) > 10: last_lines.pop(0)
                    
                    # Protocolo interno
                    if line.startswith("START|"):
                        try: total_items = int(line.split("|")[1])
                        except: pass
                    
                    elif line.startswith("PROG|"):
                        try:
                            parts = line.split("|")
                            current = int(parts[1])
                            
                            # Si raw2video manda PROG|current|total, aprovechamos para setear total
                            if len(parts) > 2:
                                try: 
                                    t = int(parts[2])
                                    if t > 0: total_items = t
                                except: pass

                            if total_items > 0:
                                percent = int((current / total_items) * 100)
                                self.progress_signal.emit(percent)
                        except: pass
                            
                    elif line.startswith("INFO|"):
                        msg = line.split("|")[1]
                        self.info_signal.emit(msg)
                        
                    elif line.startswith("ERROR|"):
                        print(f"Error Script: {line}")
                    
                    else:
                        # Si no es un comando nuestro, es output de FFmpeg (Logs/Errores)
                        # Lo imprimimos en la consola del IDE para debug
                        print(f"[FFMPEG/L2T]: {line}")

            if self.killed: return

            rc = process.poll()
            if rc == 0:
                self.finished_signal.emit(True, "Proceso completado correctamente.")
            else:
                # Si falló, mostramos las últimas líneas del log
                error_summary = "\n".join(last_lines)
                self.finished_signal.emit(False, f"El proceso terminó con código {rc}.\n\nÚltimos logs:\n{error_summary}")

        except Exception as e:
            if not self.killed:
                 self.finished_signal.emit(False, str(e))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.aboutToQuit.connect(lambda: _windows_set_execution_state(False, False))

    # Crear y mostrar Splash
    splash = IntroSplash()
    splash.start_loading()
    
    sys.exit(app.exec())