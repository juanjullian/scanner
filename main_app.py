import sys
import os
import queue
import shutil
import numpy as np
import subprocess
import cv2
from datetime import datetime
from pathlib import Path
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QListWidget, QInputDialog, QMessageBox, 
                             QSplitter, QGroupBox, QProgressBar, QTabWidget, QSlider, QFileDialog,
                             QDoubleSpinBox, QSpinBox, QProgressDialog, QDialog, QCheckBox, 
                             QComboBox, QMenu, QListWidgetItem)
from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QRect, QEvent
from PyQt6.QtGui import QImage, QPixmap, QAction, QPainter, QColor, QFont, QIcon

# Importamos el núcleo del scanner
import scanner_core

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
        
        inner_layout.addStretch()
        inner_layout.addWidget(lbl_title)
        inner_layout.addWidget(lbl_subtitle)
        inner_layout.addStretch()
        inner_layout.addWidget(self.lbl_status)
        inner_layout.addWidget(self.progress)
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
        # Aquí creamos la ventana principal pero NO la mostramos todavía
        # Esto dispara el init de MainWindow y la conexión a la cámara
        try:
            self.main_window = MainWindow()
            # Conectamos las señales del worker de la cámara para saber progreso real
            if self.main_window.camera_worker:
                 # Esperamos a que el worker diga que está "running" o haya pasado el setup
                 pass
            
            self.timer.start(500) # Chequear estado cada 500ms
            
        except Exception as e:
            self.lbl_status.setText(f"Error fatal: {e}")
            self.lbl_status.setStyleSheet("color: red; border: none;")

    def check_initialization(self):
        self.steps += 1
        
        if self.steps == 1:
            self.lbl_status.setText("Buscando cámara Lucid...")
        elif self.steps == 2:
            # Verificar si el worker de la cámara ya arrancó
            if self.main_window and self.main_window.camera_worker.isRunning():
                 self.lbl_status.setText("Cámara detectada. Aplicando configuración...")
            else:
                 self.lbl_status.setText("Esperando respuesta de la cámara...")
                 self.steps -= 1 # Repetir paso hasta que conecte
        elif self.steps == 3:
             self.lbl_status.setText("Cámara conectada correctamente.")
             self.progress.setRange(0, 100); self.progress.setValue(100)
             self.lbl_status.setStyleSheet("color: #4caf50; font-weight: bold; border: none;")
        elif self.steps == 7: # Esperamos unos ciclos (aprox 2 segs desde paso 3)
             self.finish_loading()

    def finish_loading(self):
        self.timer.stop()
        self.close()
        if self.main_window:
            self.main_window.showMaximized()

# --- VISOR PERSONALIZADO (ZOOM + PANEO + PAINT EVENT) ---
class ScanViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(False)
        self._pixmap = None
        self.max_w = 0; self.max_h = 0
        self.off_x = 0; self.off_y = 0

    def setPixmap(self, pix):
        self._pixmap = pix
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap and not self._pixmap.isNull():
            # Escalar manteniendo relación de aspecto dentro del widget
            scaled_pix = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            
            # Centrar
            x = (self.width() - scaled_pix.width()) // 2
            y = (self.height() - scaled_pix.height()) // 2
            
            painter.drawPixmap(x, y, scaled_pix)

# --- DIÁLOGO DE EXPORTACIÓN CON THUMBNAILS ---
class BatchExportDialog(QDialog):
    def __init__(self, parent, file_list, root_folder, collection):
        super().__init__(parent)
        self.setWindowTitle("Exportación por Lotes")
        self.resize(700, 600)
        self.root = root_folder
        self.coll = collection
        
        layout = QVBoxLayout(self)
        
        # 1. Lista Visual con Miniaturas
        layout.addWidget(QLabel("Archivos a procesar (Vista previa cuadro #100):"))
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(160, 120))
        
        for f_name in file_list:
            item = QListWidgetItem(f_name)
            item.setCheckState(Qt.CheckState.Checked)
            # Generar miniatura
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
        
        # 2. Configuración
        settings_group = QGroupBox("Configuración de Salida")
        sett_layout = QVBoxLayout()
        
        sett_layout.addWidget(QLabel("Formato de Salida:"))
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems([
            "ProRes 4444 (Premiere - 12bit - Alta Calidad)", 
            "GoPro CineForm (Premiere - 12bit - Intermedio)", 
            "ProRes 422 HQ (Premiere - 10bit - Estándar) [Default]",
            "HEVC 10-bit 4:4:4 (MP4 - Eficiente)", 
            "H.264 (MP4 - Proxy)"
        ])
        self.combo_fmt.setCurrentIndex(2) # Default ProRes 422 HQ
        sett_layout.addWidget(self.combo_fmt)
        
        sett_layout.addWidget(QLabel("Perfil de Revelado:"))
        self.combo_sharp = QComboBox()
        self.combo_sharp.addItems([
            "DCB Puro",
            "Suave (S:0.8 / A:1.5)",
            "Medio (S:1.3 / A:1.5)",
            "Grueso (S:2.0 / A:2.5) [Recomendado]"
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
            
            # --- DETECCIÓN USANDO METADATA (Ideal) O TAMAÑO ---
            # Intentamos leer metadata del manager si es posible, si no, adivinamos
            # (Aquí usamos la lógica rápida por tamaño para no cargar todo el json por cada item)
            
            w, h = 2840, 2200 
            is_rgb = False

            # Detección simple RGB vs Bayer basada en tamaño
            if fsize % int(w * h * 3) == 0: 
                is_rgb = True
            elif fsize % int(w * h * 1.5) == 0:
                is_rgb = False
            else:
                 # Default a Bayer si no cuadra perfecto, o podría ser RGB incompleto
                 pass
            
            # Leer Frame de muestra (Frame 100)
            frame_bytes = int(w * h * 3) if is_rgb else int(w * h * 1.5)
            total_frames = fsize // frame_bytes
            target_frame = 100 if total_frames > 100 else max(0, total_frames - 1)
            
            with open(full_path, "rb") as f:
                f.seek(target_frame * frame_bytes)
                raw_data = f.read(frame_bytes)
                
            if len(raw_data) < frame_bytes: return None

            if is_rgb:
                # RGB DIRECTO
                rgb = np.frombuffer(raw_data, dtype=np.uint8).reshape(h, w, 3)
                small = rgb[::8, ::8, :].copy() # Downscale para icono
            else:
                # BAYER RAW (Necesita proceso corrección color)
                # Extracción manual RG similar al modo Fast para consistencia
                data = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
                b0, b1, b2 = data[:, 0], data[:, 1], data[:, 2]
                
                # Desempaquetado 8-bit rápido
                p0 = ((b1 & 0x0F) << 4) | (b0 >> 4)
                p1 = b2
                img_flat = np.empty(w*h, dtype=np.uint8)
                img_flat[0::2] = p0; img_flat[1::2] = p1
                img_bayer = img_flat.reshape(h, w)
                
                # Subsampling directo RG (R en 0,0 | B en 1,1)
                r_ch = img_bayer[0::2, 0::2]
                g_ch = img_bayer[0::2, 1::2]
                b_ch = img_bayer[1::2, 1::2]
                
                # Crear RGB pequeño
                rows, cols = r_ch.shape
                small = np.dstack((r_ch, g_ch, b_ch))
                
                # Reducir más para icono (aprox 160x120)
                small = cv2.resize(small, (160, 120), interpolation=cv2.INTER_NEAREST)
                
                # Opcional: Auto Brightness simple
                avg = np.mean(small)
                if avg > 0: small = np.clip(small * (100/avg), 0, 255).astype(np.uint8)
            
            ih, iw, _ = small.shape
            if not small.flags['C_CONTIGUOUS']: small = np.ascontiguousarray(small)
            qimg = QImage(small.data, iw, ih, iw*3, QImage.Format.Format_RGB888)
            return QIcon(QPixmap.fromImage(qimg))
        except:
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
        
        fmt_map = {0: 'prores', 1: 'cineform', 2: 'prores_hq', 3: 'hevc', 4: 'h264'}
        sharp_map = {0: '0,0', 1: '0.8,1.5', 2: '1.3,1.5', 3: '2.0,2.5'}
        
        return files, fmt_map[self.combo_fmt.currentIndex()], sharp_map[self.combo_sharp.currentIndex()]

# --- VENTANA PRINCIPAL ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Escáner de películas - Archivo La Unión")
        self.resize(1400, 900)
        self.load_config() # Carga o pide la carpeta raíz
        self.manager = scanner_core.CollectionManager(self.root_folder)
        self.active_collection = None
        self.is_recording = False
        # OPTIMIZACIÓN RAM: Bajamos buffer de 500 (~4.5GB) a 180 (~1.6GB)
        # Esto es suficiente para absorber hipo del disco duro sin colapsar la RAM.
        self.frame_queue = queue.Queue(maxsize=180)
        self.camera_worker = None
        self.writer_worker = None
        
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

    def load_config(self):
        config_path = Path(__file__).parent / "persist.json"
        default_root = os.path.expanduser("~/Documents/Archivo_Scan_Data")
        self.root_folder = default_root
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    saved_root = data.get("root_folder")
                    if saved_root and os.path.isdir(saved_root):
                        self.root_folder = saved_root
                        return

            # Si no existe o no es válida, pedimos al usuario
            self.ask_root_folder_first_time()
            
        except Exception as e:
            print(f"Error cargando config: {e}")
            self.root_folder = default_root

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

    def save_config(self):
        try:
            config_path = Path(__file__).parent / "persist.json"
            with open(config_path, 'w') as f:
                json.dump({"root_folder": self.root_folder}, f, indent=4)
        except Exception as e:
            print(f"Error guardando config: {e}")

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
        
        # [HEADER] Título + Stats alineadas
        header_bar = QHBoxLayout()
        self.lbl_status = QLabel("Selecciona una colección")
        self.lbl_status.setStyleSheet("font-size: 14pt; color: #ff9800; font-weight: bold;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        stats_layout = QHBoxLayout()
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_temp = QLabel("Tmp: --")
        self.lbl_saved = QLabel("Sav: 0")
        self.lbl_buffer = QLabel("Buf: 0")
        self.lbl_bw = QLabel("BW: 0 Mbps")
        self.lbl_dropped = QLabel("Drp: 0")
        self.lbl_dropped.setStyleSheet("color: red; font-weight: bold;")
        
        for lb in [self.lbl_fps, self.lbl_temp, self.lbl_saved, self.lbl_buffer, self.lbl_bw, self.lbl_dropped]:
            lb.setStyleSheet(lb.styleSheet() + "; font-size: 10pt; margin-left: 10px; color: #888;")
            stats_layout.addWidget(lb)
            
        header_bar.addWidget(self.lbl_status, 1) # Stretch
        header_bar.addLayout(stats_layout)
        scan_layout.addLayout(header_bar)
        
        # [VISOR]
        self.viewer_scan = ScanViewer()
        self.viewer_scan.setStyleSheet("background-color: #111; border: 1px solid #444;")
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
        cb_layout.addWidget(self.combo_type)
        
        # B. Foco
        self.btn_peaking = QPushButton("Peak")
        self.btn_peaking.setCheckable(True)
        self.btn_peaking.clicked.connect(self.toggle_peaking)
        self.btn_peaking.setFixedWidth(50)
        
        self.btn_zoom_1to1 = QPushButton("1:1")
        self.btn_zoom_1to1.setCheckable(True)
        self.btn_zoom_1to1.clicked.connect(self.toggle_zoom) # FIX: usaba toggle_zoom_state antes, unificar
        self.btn_zoom_1to1.setFixedWidth(40)
        
        cb_layout.addWidget(self.btn_peaking)
        cb_layout.addWidget(self.btn_zoom_1to1)
        
        # C. Exposición
        exp_layout = QHBoxLayout()
        exp_layout.setSpacing(5)
        exp_layout.addWidget(QLabel("Exp:"))
        self.sl_exp = QSlider(Qt.Orientation.Horizontal)
        self.sl_exp.setRange(20, 100)
        self.sl_exp.setValue(50)
        self.sl_exp.setSingleStep(10) # 10 en 10
        self.sl_exp.setPageStep(10)
        self.sl_exp.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sl_exp.setTickInterval(10)
        self.sl_exp.valueChanged.connect(self.on_exposure_change)
        
        self.lbl_exp_val = QLabel("50")
        self.lbl_exp_val.setFixedWidth(30)
        self.lbl_exp_val.setStyleSheet("font-weight: bold; color: #0078d7;")
        self.lbl_exp_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        exp_layout.addWidget(self.sl_exp)
        exp_layout.addWidget(self.lbl_exp_val)
        
        cb_layout.addLayout(exp_layout, 1) # Stretch para que slider crezca
        
        # D. GRABAR
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
        scan_layout.addWidget(self.lbl_fps); scan_layout.addWidget(self.lbl_buffer)
        scan_layout.addWidget(self.lbl_temp); scan_layout.addWidget(self.lbl_saved)
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
        self.sb_fps = QSpinBox(); self.sb_fps.setRange(1,60); self.sb_fps.setValue(18); self.sb_fps.valueChanged.connect(self.update_fps_metadata)
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
        splitter = QSplitter(); splitter.addWidget(left_panel); splitter.addWidget(self.tabs); splitter.setSizes([300,900])
        main_layout.addWidget(splitter)

        self.play_timer = QTimer(); self.play_timer.timeout.connect(self.next_frame_playback)
        self.export_queue = []; self.is_exporting_batch = False
        
        self.init_camera_thread()

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
                    # Refrescar el frame actual
                    self.seek_viewer(self.slider_frame.value())
                    return True
                # --------------------------------------------------
        
        return super().eventFilter(source, event)

    # --- MÉTODOS DE COLECCIONES ---
    def refresh_collections(self):
        self.col_list.clear()
        self.col_list.addItems(self.manager.get_collections())

    def create_collection(self):
        name, ok = QInputDialog.getText(self, "Nueva", "Nombre:")
        if ok and name: self.manager.create_collection(name); self.refresh_collections()

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
        self.btn_record.setEnabled(True)
        self.btn_record.setText("INICIAR CAPTURA")
        self.btn_record.setStyleSheet("background-color: #0078d7; font-size: 14pt; color: white;")
        self.tabs.setCurrentIndex(0)

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
        
        # 4. CALCULAR TAMAÑO REAL DEL FRAME (Stride Detection)
        # Esto arregla el "efecto corrido"
        bytes_per_pixel = 3 if self.view_is_rgb else 1.5
        math_size = int(self.raw_width * self.raw_height * bytes_per_pixel)
        
        fsize = os.path.getsize(filepath)
        
        # Asumimos el tamaño matemático por defecto
        self.bytes_per_frame = math_size
        
        # Pero si el archivo no cuadra perfecto, calculamos el real
        if math_size > 0:
            approx_frames = round(fsize / math_size)
            if approx_frames > 0:
                real_stride = fsize // approx_frames
                
                # Si hay una diferencia (padding), usamos el real
                if real_stride != math_size:
                    self.bytes_per_frame = real_stride
                    print(f"INFO: Padding detectado. Math: {math_size} -> Real en disco: {self.bytes_per_frame}")

        self.total_frames = fsize // self.bytes_per_frame if self.bytes_per_frame > 0 else 0
        
        # Info UI
        mode_str = "RGB" if self.view_is_rgb else "RAW"
        self.lbl_frame_info.setText(f"{filename} | {self.total_frames}f | {roi_key} | {mode_str}")
        self.slider_frame.setRange(0, max(0, self.total_frames-1))
        
        self.seek_viewer(0)

    def seek_viewer(self, idx, fast=False):
        if not hasattr(self, 'current_view_file'): return
        
        # 1. Preparar lectura
        w, h = self.raw_width, self.raw_height
        is_rgb_view = getattr(self, 'view_is_rgb', False)
        
        # Cálculo de bytes seguro
        frame_size = int(w * h * 3) if is_rgb_view else int(w * h * 1.5)
        offset = idx * frame_size
        
        try:
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
                        # --- MODO CALIDAD (PAUSA) ---
                        # 1. Desempaquetado completo
                        arr_packed = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
                        b0 = arr_packed[:, 0].astype(np.uint16)
                        b1 = arr_packed[:, 1].astype(np.uint16)
                        b2 = arr_packed[:, 2].astype(np.uint16)
                        p0 = b0 | ((b1 & 0x0F) << 8)
                        p1 = (b1 >> 4) | (b2 << 4)
                        
                        img_flat = np.empty(w*h, dtype=np.uint16)
                        img_flat[0::2] = p0; img_flat[1::2] = p1
                        img_bayer = img_flat.reshape(h, w)

                        # 2. Debayering y Pipeline Color Completo
                        rgb16 = cv2.cvtColor(img_bayer, cv2.COLOR_BayerBG2RGB) # BG implica R en (0,0) con OpenCV
                        
                        # Gamma Precisa
                        rgb_f = rgb16.astype(np.float32) / 4095.0
                        rgb_gamma = np.power(rgb_f, 1.0/2.2) 
                        rgb8 = np.clip(rgb_gamma * 255, 0, 255).astype(np.uint8)

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
        self.camera_worker = scanner_core.CameraWorker("1.0.txt")
        self.camera_worker.set_queue(self.frame_queue)
        self.camera_worker.image_received.connect(self.update_display) # FIX: nombre correcto
        self.camera_worker.stats_updated.connect(self.update_stats)
        self.camera_worker.error_occurred.connect(self.on_camera_error)
        self.camera_worker.start()

    def update_display(self, frame):
        if self.tabs.currentIndex() != 0: return

        # 1. DETECCIÓN Y CORRECCIÓN DE COLOR
        is_color = (frame.ndim == 3)
        
        if is_color:
            h, w, c = frame.shape
            # El SDK de Lucid suele entregar BGR. Qt espera RGB.
            # Hacemos el swap aquí para que los colores sean correctos (Rojo es Rojo, no Azul)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- CORRECCIÓN VISUAL SIMPLE PARA BAYER ---
            # Las imágenes Bayer RAW suelen verse verdosas si la cámara no hace WB interno.
            # Si notas que se ve MUY verde, descomenta estas líneas para un WB automático simple:
            # ---------------------------------------------------------
            # avg_a = np.average(frame_rgb, axis=(0,1))
            # max_avg = np.max(avg_a)
            # if max_avg > 0:
            #     gains = max_avg / avg_a
            #     frame_rgb = np.clip(frame_rgb * gains, 0, 255).astype(np.uint8)
            # ---------------------------------------------------------
            
            disp = frame_rgb
            bytes_per_line = w * 3
        else:
            h, w = frame.shape
            disp = frame
            bytes_per_line = w

        self.viewer_scan.max_w = w
        self.viewer_scan.max_h = h

        # 2. LOGICA DE ZOOM (Sin cambios mayores, solo adaptada a color)
        if self.btn_zoom_1to1.isChecked():
            vw = self.viewer_scan.width(); vh = self.viewer_scan.height()
            cw = min(vw, w); ch = min(vh, h)
            self.viewer_scan.off_x = max(0, min(self.viewer_scan.off_x, w-cw))
            self.viewer_scan.off_y = max(0, min(self.viewer_scan.off_y, h-ch))
            sx, sy = int(self.viewer_scan.off_x), int(self.viewer_scan.off_y)
            
            if is_color:
                disp_crop = disp[sy:sy+ch, sx:sx+cw, :].copy()
            else:
                disp_crop = disp[sy:sy+ch, sx:sx+cw].copy()
            
            h, w = ch, cw
            if is_color: bytes_per_line = w * 3
            else: bytes_per_line = w
            
            final_disp = disp_crop
        else:
            # Downscale para rendimiento si es muy grande
            if w > 2000:
                if is_color: final_disp = disp[::2, ::2, :].copy()
                else: final_disp = disp[::2, ::2].copy()
            else:
                final_disp = disp.copy()
            
            h, w = final_disp.shape[:2]
            if is_color: bytes_per_line = w * 3
            else: bytes_per_line = w

        # 3. CREAR QIMAGE
        if is_color:
            qimg = QImage(final_disp.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            if final_disp.dtype == np.uint16: 
                 final_disp = (final_disp >> 4).astype(np.uint8)
            qimg = QImage(final_disp.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        # 4. PEAKING (Focus Assist)
        pix = QPixmap.fromImage(qimg)
        if self.btn_peaking.isChecked():
            if is_color: gray_peak = cv2.cvtColor(final_disp, cv2.COLOR_RGB2GRAY)
            else: gray_peak = final_disp
            
            lap = cv2.Laplacian(gray_peak, cv2.CV_16S, ksize=3)
            _, mask = cv2.threshold(cv2.convertScaleAbs(lap), 40, 255, cv2.THRESH_BINARY)
            
            ov = QImage(w, h, QImage.Format.Format_ARGB32)
            ov.fill(QColor(0,255,0,180))
            ov.setAlphaChannel(QImage(mask.tobytes(), w, h, w, QImage.Format.Format_Alpha8))
            
            p = QPainter(pix)
            p.drawImage(0,0,ov)
            p.end()

        # 5. ASIGNAR
        if self.btn_zoom_1to1.isChecked(): 
            self.viewer_scan.setPixmap(pix)
        else: 
            self.viewer_scan.setPixmap(pix.scaled(self.viewer_scan.size(), Qt.AspectRatioMode.KeepAspectRatio))
    def update_stats(self, fps, temp, qsize, drops=0, bw=0.0, bw_src="C"):
        # Filtro visual para FPS (que no salte tanto)
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_temp.setText(f"Tmp: {temp:.1f}°")
        self.lbl_buffer.setText(f"Buf: {qsize}")
        self.lbl_dropped.setText(f"Drp: {drops}")
        self.lbl_bw.setText(f"BW({bw_src}): {int(bw)} Mbps")
        
        if drops > 0:
             self.lbl_dropped.setStyleSheet("color: red; font-weight: bold; font-size: 10pt; margin-left: 10px;")
        else:
             self.lbl_dropped.setStyleSheet("color: #4caf50; font-size: 10pt; margin-left: 10px;")
        
        if self.writer_worker:
            self.lbl_saved.setText(f"Sav: {self.writer_worker.frames_saved}")

    def on_camera_error(self, e): QMessageBox.critical(self, "Cam Error", e)

    # on_format_changed eliminado

    def toggle_zoom(self, c):
         # Redirige a toggle_zoom_state para mantener compatibilidad
         self.toggle_zoom_state(c)

    def on_exposure_change(self, v):
        # Forzar pasos de 10
        step = 10
        val = int(round(v / step) * step)
        if val != v:
            self.sl_exp.setValue(val)
            return
            
        self.lbl_exp_val.setText(str(val))
        if self.camera_worker: 
            self.camera_worker.update_exposure(val)
    
    def toggle_peaking(self, c): self.btn_peaking.setStyleSheet("background:red" if c else "")
    
    def toggle_zoom_state(self, c):
        self.viewer_scan.zoom_active = c
        self.viewer_scan.setCursor(Qt.CursorShape.OpenHandCursor if c else Qt.CursorShape.ArrowCursor)

    def toggle_recording(self):
        if not self.is_recording:
            # --- INICIAR GRABACIÓN ---
            if not self.active_collection:
                QMessageBox.warning(self, "Error", "Selecciona una colección primero.")
                # Assuming btn_record is a QPushButton, not a QCheckBox, so no setChecked(False)
                return

            if self.camera_worker:
                 # FIX: Limpiar frames viejos del buffer antes de empezar
                 self.camera_worker.clear_queue()
                 self.camera_worker.reset_drop_count() 
                 
                 # Conectar la queue real solo ahora
                 self.camera_worker.set_queue(self.frame_queue)
            
            # --- INICIAR GRABACIÓN ---
            # fmt = "Standard" # Eliminado selector
            ftype = self.combo_type.currentText() # Ej: "Color (Pos/Neg)"
            
            # 1. (Ya no se aplica ROI formato porque es fijo)
            
            # 2. Obtener nombres y workers
            fn, fp = self.manager.get_next_filename(self.active_collection)
            self.writer_worker = scanner_core.WriterWorker(self.frame_queue, fp)
            self.writer_worker.frames_saved_signal.connect(lambda x: self.lbl_saved.setText(f"G: {x}"))
            self.writer_worker.start()
            
            # 3. --- GUARDADO DE METADATA (LO IMPORTANTE) ---
            # Obtenemos el modo de pixel actual del worker (o 'bayer' por defecto)
            pixel_mode = getattr(self.camera_worker, "pixel_mode", "bayer")
            
            # Guardamos todo explícitamente usando argumentos con nombre (kwargs)
            self.manager.set_file_info(
                self.active_collection, 
                fn, 
                fps=self.sb_fps.value(),
                roi_key="Standard 2840x2200", # Valor fijo
                film_type=ftype,
                pixel_format=pixel_mode
            )
            # -----------------------------------------------

            self.is_recording = True
            self.btn_record.setText(f"DETENER ({fn})")
            self.btn_record.setStyleSheet("background:red;color:white;font-weight:bold")
            # self.combo_format.setEnabled(False) # Eliminado
            self.combo_type.setEnabled(False)
            
        else:
            # --- DETENER GRABACIÓN ---
            self.is_recording = False
            if self.camera_worker:  
                 # Desconectar queue para que los nuevos frames se descarten (no se acumulen)
                 self.camera_worker.set_queue(None)
                 self.camera_worker.reset_drop_count()

            if self.writer_worker:
                self.writer_worker.stop()
                self.writer_worker = None
            self.btn_record.setText("GRABAR")
            self.btn_record.setStyleSheet("")
            # self.combo_format.setEnabled(True) # Eliminado
            self.combo_type.setEnabled(True)
            self.refresh_file_list(self.active_collection)

    # --- EXPORTACIÓN ---
    def export_tif(self):
        if not hasattr(self, 'current_view_file'): 
            QMessageBox.warning(self, "Error", "Carga un archivo en el visor primero.")
            return

        # --- CORRECCIÓN ---
        # Enviamos la ruta del ARCHIVO exacta, no la carpeta padre.
        # Así l2t.py solo procesará este archivo.
        target_input = Path(self.current_view_file)
        
        # 1. Configurar Diálogo de Progreso (MODAL)
        self.pd_tif = QProgressDialog(f"Procesando {target_input.name}...", "Cancelar", 0, 100, self)
        self.pd_tif.setWindowTitle("Exportando Secuencia TIF")
        self.pd_tif.setWindowModality(Qt.WindowModality.ApplicationModal) 
        self.pd_tif.setAutoClose(False)
        self.pd_tif.setValue(0)
        self.pd_tif.show()

        # 2. Configurar Worker
        cmd = [sys.executable, "l2t.py", str(target_input)]
        
        self.tif_worker = UniversalExportWorker(cmd)
        
        # 3. Conectar Señales (Barra de progreso fluida)
        self.tif_worker.progress_signal.connect(self.pd_tif.setValue)
        self.tif_worker.info_signal.connect(self.pd_tif.setLabelText)
        self.tif_worker.finished_signal.connect(self.on_tif_finished)
        self.pd_tif.canceled.connect(self.on_tif_cancelled)
        
        # Guardar ruta de salida para posible borrado (Carpeta con el mismo nombre del archivo sin ext)
        self.current_tif_output = target_input.parent / target_input.stem
        
        # 4. Iniciar
        self.tif_worker.start()

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

    def on_tif_finished(self, success, msg):
        try: self.pd_tif.canceled.disconnect(self.on_tif_cancelled)
        except: pass

        self.pd_tif.close()
        if success:
            # Mensaje menos intrusivo o confirmación simple
            QMessageBox.information(self, "Listo", f"Exportación finalizada.\n{msg}")
        else:
            QMessageBox.critical(self, "Error", f"Fallo en l2t:\n{msg}")
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
        
        # 3. Abrir Diálogo
        # Pasamos target_collection en vez de self.active_collection
        dlg = BatchExportDialog(self, files, self.root_folder, target_collection)
        
        if dlg.exec():
            sel, fmt, sharp = dlg.get_selection()
            if not sel: return
            
            for f in sel:
                # Usamos target_collection aquí también para construir la ruta correcta
                path = Path(self.root_folder) / target_collection / f
                self.export_queue.append((path, fmt, sharp))
            
            QMessageBox.information(self, "Cola", f"{len(sel)} archivos añadidos a la cola.")
            
            if not self.is_exporting_batch: 
                self.process_export_queue()

    def process_export_queue(self):
        if not self.export_queue:
            self.is_exporting_batch = False
            QMessageBox.information(self, "Fin", "Cola terminada.")
            return

        self.is_exporting_batch = True
        nf, fmt, sharp = self.export_queue.pop(0)
        
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
        fps = info.get("fps", 18)
        
        print(f"Procesando: {nf.name} | Colección: {collection_name} | Modo: {mode}")
        
        self.pd = QProgressDialog(f"Exportando {nf.name}...", "Cancelar", 0, 100, self)
        self.pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.pd.setAutoClose(False)
        self.pd.setValue(0)
        self.pd.show()
        
        cmd = [
            sys.executable, "raw2video.py", str(nf), 
            "--codec", fmt, 
            "--fps", str(fps), 
            "--sharp", sharp, 
            "--mode", mode
        ]
        
        # Pre-calcular ruta de salida para posible borrado
        ext_map = {'prores': '.mov', 'ffv1': '.mkv', 'h264': '.mp4', 'hevc': '.mp4', 'jxl': ''} # jxl usa carpeta
        ext = ext_map.get(fmt, ".mp4")
        self.current_video_output = nf.parent / f"{nf.stem}_{fmt}{ext}"
        
        self.worker = UniversalExportWorker(cmd)
        self.worker.progress_signal.connect(self.pd.setValue)
        self.worker.info_signal.connect(self.pd.setLabelText)
        self.worker.finished_signal.connect(self.on_batch_item_finished)
        self.pd.canceled.connect(self.on_batch_cancel)
        self.worker.start()

    def on_batch_cancel(self):
        self.export_queue = [] # Detener resto de la cola
        self.is_exporting_batch = False
        if self.worker: self.worker.kill()
        
        resp = QMessageBox.question(self, "Cancelado", "¿Desea eliminar el archivo incompleto generado?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp == QMessageBox.StandardButton.Yes:
            try:
                if hasattr(self, 'current_video_output') and self.current_video_output.exists():
                    os.remove(self.current_video_output)
                    QMessageBox.information(self, "Info", f"Archivo eliminado: {self.current_video_output.name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo borrar: {e}")
        self.pd.close()

    def on_batch_item_finished(self, s, m):
        try: self.pd.canceled.disconnect(self.on_batch_cancel)
        except: pass
        
        if not self.pd.wasCanceled(): # Evitar doble cierre/error
             self.pd.close()
             if not s: print(f"Error export: {m}")
             self.process_export_queue() # Siguiente

    def open_file_context_menu(self, pos):
        if not self.file_list.itemAt(pos): return
        menu = QMenu()
        act = QAction("Borrar", self); act.triggered.connect(self.delete_selected_file)
        menu.addAction(act); menu.exec(self.file_list.mapToGlobal(pos))

    def delete_selected_file(self):
        it = self.file_list.currentItem()
        if it and self.active_collection:
            if QMessageBox.question(self, "Borrar", f"¿Borrar {it.text()}?", QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                self.manager.delete_file(self.active_collection, it.text())
                if hasattr(self, 'current_view_file') and Path(self.current_view_file).name == it.text():
                    self.viewer_play.clear(); self.play_timer.stop()
                self.refresh_file_list(self.active_collection)

    def update_disk_space(self):
        try: self.lbl_disk.setText(f"Libre: {shutil.disk_usage(self.root_folder).free // 2**30} GB")
        except: pass
    
    def closeEvent(self, e):
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
    
    # Crear y mostrar Splash
    splash = IntroSplash()
    splash.start_loading()
    
    sys.exit(app.exec())