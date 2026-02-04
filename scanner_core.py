import os
import time
import queue
import numpy as np
import json
import shutil
from PyQt6.QtCore import pyqtSignal, QThread
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.enums import PixelFormat
from pathlib import Path

# --- CONFIGURACIÓN DE FORMATOS (ROI) ---
FORMAT_ROIS = {
    "Super 8":       {"w": 2840, "h": 2080}, 
    "Regular 8mm":   {"w": 2840, "h": 2136}, 
    "16mm (Mudo)":   {"w": 2840, "h": 2072}, 
    "16mm (Sonido)": {"w": 2840, "h": 1700}, 
    "Full Sensor":   {"w": 2840, "h": 2840} 
}

class CameraWorker(QThread):
    image_received = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(float, float, int) 
    error_occurred = pyqtSignal(str)

    def __init__(self, settings_file="strobe2.txt"):
        super().__init__()
        # --- NUEVO: Cargar modo desde config.json ---
        self.pixel_mode = "bayer" # Default seguro
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    self.pixel_mode = cfg.get("pixel_mode", "bayer")
            except: pass
        
        # Seleccionar archivo de configuración según el modo
        if self.pixel_mode == "rgb":
            self.settings_file = "config_rgb.txt"
        else:
            self.settings_file = "config_bayer.txt"
        # ---------------------------------------------

        self.running = False
        self.device = None
        self.write_queue = None
        self.frame_count = 0
        self.start_time = 0
        self.current_roi_key = "Full Sensor"

    def set_queue(self, q):
        self.write_queue = q

    def setup_camera(self):
        print("Buscando dispositivos...")
        tries = 0
        while tries < 3:
            devices = system.create_device()
            if not devices:
                time.sleep(1)
                tries += 1
            else:
                self.device = devices[0]
                break
        
        if not self.device:
            raise RuntimeError("No se encontró cámara Lucid.")

        print(f"Cargando configuración: {self.settings_file} (Modo: {self.pixel_mode})")
        self.apply_settings_from_file(self.device, self.settings_file)
        
        nodemap = self.device.nodemap
        tl_stream = self.device.tl_stream_nodemap

        try:
            # --- SECCIÓN IF/ELSE PARA PIXEL FORMAT ---
            if self.pixel_mode == "bayer":
                # Solo forzamos Bayer si estamos en modo Bayer
                try: nodemap.get_node('PixelFormat').value = 'BayerRG12p'
                except: pass
            else:
                # En modo RGB, confiamos en el archivo .txt o forzamos RGB8 si es necesario
                try: nodemap.get_node('PixelFormat').value = 'RGB8'
                except: pass

            try:
                # Esto es CRÍTICO. Si está en True, agrega bytes extra que corren la imagen.
                nodemap.get_node('ChunkModeActive').value = False
                nodemap.get_node('GevSCPSPacketSize').value = 9000
            except Exception as e:
                print(f"No se pudo desactivar ChunkMode: {e}")

            tl_stream['StreamBufferHandlingMode'].value = "OldestFirst"
            tl_stream['StreamAutoNegotiatePacketSize'].value = True
            tl_stream['StreamPacketResendEnable'].value = True
            
            # Trigger Overlap: Dejar que el archivo txt lo controle o poner ReadOut
            # nodemap.get_node('TriggerOverlap').value = 'ReadOut' 

            if nodemap.get_node('GammaEnable'): 
                nodemap.get_node('GammaEnable').value = True

        except Exception as e:
            print(f"Warning setup: {e}")

        return nodemap.get_node('DeviceTemperature')

    def apply_settings_from_file(self, device, filepath):
        if not os.path.exists(filepath): 
            print(f"ERROR: No se encontró el archivo {filepath}")
            return
            
        print(f"INFO: Aplicando configuración desde {filepath}...")
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        nodemap = device.nodemap 
        errors = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                key = parts[0].strip()
                val_str = parts[-1].strip()

                # --- FILTRO NUEVO: IGNORAR NODOS DE SOLO LECTURA ---
                # CounterValue es un indicador, no se puede configurar.
                # Chunk data también suele ser readonly o automático.
                if "CounterValue" in key or "Chunk" in key:
                    continue
                # ---------------------------------------------------
                
                
                try:
                    node = nodemap.get_node(key)
                    if node is None or not node.is_writable:
                        continue

                    # --- DETECCIÓN INTELIGENTE DE TIPO ---
                    # Obtenemos el tipo de nodo como string para saber cómo tratarlo
                    # Ej: "<class 'arena_api.node.NodeBoolean'>"
                    node_type = str(type(node)).lower()

                    try:
                        if 'boolean' in node_type:
                            # Arena espera True/False, no "0" o "1" strings
                            if val_str.lower() in ['1', 'true', 'on']:
                                node.value = True
                            else:
                                node.value = False
                        
                        elif 'float' in node_type:
                            # Soporta "2e+03" y "48.5"
                            node.value = float(val_str)
                            
                        elif 'integer' in node_type:
                            # Soporta "100"
                            node.value = int(float(val_str)) # float() intermedio por si viene como "100.0"
                        
                        else:
                            # Enums y Strings directos
                            # Aquí sí intentamos asignar el string directo
                            # Y si falla, usamos from_string solo si es Enum
                            try:
                                node.value = val_str
                            except:
                                if hasattr(node, 'from_string'):
                                    node.from_string(val_str)
                                else:
                                    raise # Relanzar error si no tiene arreglo

                    except Exception as e_conv:
                        errors.append(f"{key} ({node_type}) -> {val_str}: {e_conv}")

                except Exception as e:
                    errors.append(f"{key} -> {val_str}: {str(e)}")

        if errors:
            print(f"WARN: Hubo {len(errors)} problemas aplicando la configuración:")
            for err in errors[:10]: # Muestro 10 para ver más detalle
                print(f"  - {err}")
        else:
            print("EXITO: Configuración aplicada perfectamente.")
    def set_format_roi(self, format_name):
        if format_name not in FORMAT_ROIS: return
        if self.current_roi_key == format_name: return

        print(f"Cambiando ROI a: {format_name}...")
        self.running = False
        
        # Detener stream para desbloquear
        if self.device:
            try: self.device.stop_stream()
            except: pass 
        
        # Esperar a que el hilo termine (evita conflicto de hilos)
        self.wait(2000) 
        
        if self.device:
            try:
                nodemap = self.device.nodemap
                cfg = FORMAT_ROIS[format_name]
                
                nodemap.get_node('OffsetX').value = 0
                nodemap.get_node('OffsetY').value = 0
                nodemap.get_node('Width').value = cfg['w']
                nodemap.get_node('Height').value = cfg['h']
                
                self.device.tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
                self.current_roi_key = format_name
                print(f"ROI Aplicado exitosamente: {cfg['w']}x{cfg['h']}")
                
            except Exception as e:
                print(f"Error aplicando ROI: {e}")
                self.error_occurred.emit(f"Error ROI: {e}")
            
            # Reiniciar
            self.start()

    def update_exposure(self, value_us):
        if self.device:
            try: self.device.nodemap.get_node('ExposureTime').value = float(value_us)
            except: pass

    def update_gamma(self, value):
        if self.device:
            try: self.device.nodemap.get_node('Gamma').value = float(value)
            except: pass

    def run(self):
        try:
            if not self.device: self.setup_camera()
            temp_node = self.device.nodemap.get_node('DeviceTemperature')
            
            self.running = True
            self.start_time = time.time()
            last_temp_check = 0
            current_temp = 0.0

            try:
                self.device.start_stream(100)
            except Exception as e:
                print(f"Error start_stream: {e}")
                self.running = False
                return

            while self.running:
                # --- A. LECTURA DE TEMPERATURA (Independiente del video) ---
                # Hacemos esto AL PRINCIPIO del bucle para que siempre se ejecute
                now = time.time()
                if now - last_temp_check > 1.0: # Actualizar cada 1 segundo
                    try: 
                        current_temp = temp_node.value 
                        # Emitimos stats aquí mismo
                        elapsed = now - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        q_size = self.write_queue.qsize() if self.write_queue else 0
                        self.stats_updated.emit(fps, current_temp, q_size)
                    except: pass
                    last_temp_check = now

                # --- B. CAPTURA DE IMAGEN ---
                try:
                    # Timeout REDUCIDO a 200ms.
                    # Si no hay imagen en 200ms, salta la excepción, el bucle da la vuelta 
                    # y vuelve a chequear la temperatura.
                    buffer = self.device.get_buffer(timeout=200)
                except Exception: 
                    # Si fue timeout, continuamos el bucle (polling)
                    if not self.running: break
                    continue 

                if buffer.is_incomplete:
                    self.device.requeue_buffer(buffer)
                    continue

                # 1. Copia Maestra
                image_raw = BufferFactory.copy(buffer)
                self.device.requeue_buffer(buffer)
                
                # --- CORRECCIÓN DE DESPLAZAMIENTO (PADDING) ---
                # Calculamos el tamaño matemático exacto que DEBERÍA tener la imagen
                h, w = image_raw.height, image_raw.width
                
                # Detectamos modo (asumiendo que pixel_mode está seteado, si no, inferimos)
                is_rgb = (image_raw.pixel_format == PixelFormat.RGB8 or image_raw.pixel_format == PixelFormat.BGR8)
                
                if is_rgb:
                    expected_size = int(w * h * 3)
                else:
                    # BayerRG12p son 12 bits = 1.5 bytes por pixel
                    expected_size = int(w * h * 1.5)

                # Extraemos el array completo
                data_full = np.ctypeslib.as_array(image_raw.pdata, shape=(image_raw.size_filled,))
                
                # ¡AQUÍ ESTÁ EL FIX! Recortamos cualquier basura extra al final
                if data_full.size > expected_size:
                    raw_bytes = data_full[:expected_size].copy().tobytes()
                else:
                    raw_bytes = data_full.copy().tobytes()
                # -----------------------------------------------

                # A. Guardar
                if self.write_queue:
                    try: self.write_queue.put_nowait(raw_bytes)
                    except queue.Full: pass

                # B. Previsualización
                # B. Previsualización
                image_for_gui = None
                try:
                    # SIEMPRE intentamos convertir a BGR8 para la vista previa.
                    # El SDK de Arena es muy rápido haciendo esto.
                    # Si la imagen viene en Bayer, el SDK la debayeriza automáticamente.
                    # Si viene en RGB, simplemente la copia o reordena.
                    
                    # Usamos BGR8 porque es el estándar de Windows/OpenCV y suele ser más compatible
                    image_converted = BufferFactory.convert(image_raw, PixelFormat.BGR8)
                    
                    h, w = image_converted.height, image_converted.width
                    
                    # Extraemos el array numpy (H, W, 3)
                    image_for_gui = np.ctypeslib.as_array(image_converted.pdata, shape=(h, w, 3)).copy()
                    
                    BufferFactory.destroy(image_converted)

                except Exception as e:
                    # Fallback de seguridad: Si falla la conversión de color, intentamos Mono8
                    try:
                        print(f"Warn Preview Color: {e} - Intentando Mono8")
                        image_converted = BufferFactory.convert(image_raw, PixelFormat.Mono8)
                        h, w = image_converted.height, image_converted.width
                        image_for_gui = np.ctypeslib.as_array(image_converted.pdata, shape=(h, w)).copy()
                        BufferFactory.destroy(image_converted)
                    except: pass
                finally:
                    BufferFactory.destroy(image_raw)

               
               
                if image_for_gui is not None:
                    self.image_received.emit(image_for_gui)
                    self.frame_count += 1
                    
                    # (Ya no emitimos stats aquí abajo para no duplicar, 
                    #  se emiten arriba de forma constante)

        except Exception as e:
            if "aborted" not in str(e).lower() and self.running:
                self.error_occurred.emit(str(e))
        finally:
            if self.device:
                try: self.device.stop_stream()
                except: pass

    def stop(self):
        self.running = False
        self.wait()
        if self.device:
            try: 
                system.destroy_device(self.device)
                self.device = None
            except: pass

# --- WORKER DE ESCRITURA (Sin cambios) ---
class WriterWorker(QThread):
    frames_saved_signal = pyqtSignal(int)
    def __init__(self, frame_queue, save_path):
        super().__init__()
        self.frame_queue = frame_queue
        self.save_path = save_path
        self.running = True
        self.frames_saved = 0

    def run(self):
        with open(self.save_path, "wb") as f:
            while self.running or not self.frame_queue.empty():
                try:
                    data = self.frame_queue.get(timeout=0.1)
                    f.write(data)
                    self.frames_saved += 1
                    self.frames_saved_signal.emit(self.frames_saved)
                    self.frame_queue.task_done()
                except queue.Empty: continue
    def stop(self):
        self.running = False
        self.wait()

# --- GESTOR DE COLECCIONES (Sin cambios) ---
class CollectionManager:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def get_collections(self):
        return sorted([d.name for d in self.root_path.iterdir() if d.is_dir()])

    def create_collection(self, name):
        path = self.root_path / name
        if not path.exists():
            path.mkdir()
            return True
        return False

    def get_next_filename(self, collection_name):
        coll_path = self.root_path / collection_name
        files = list(coll_path.glob(f"{collection_name}_*.raw"))
        max_idx = 0
        for f in files:
            try:
                idx = int(f.stem.split('_')[-1])
                if idx > max_idx: max_idx = idx
            except: pass
        return f"{collection_name}_{max_idx + 1:03d}.raw", str(coll_path / f"{collection_name}_{max_idx + 1:03d}.raw")

    def get_metadata_path(self, collection_name):
        return self.root_path / collection_name / "metadata.json"

    def load_metadata(self, collection_name):
        json_path = self.get_metadata_path(collection_name)
        if json_path.exists():
            try:
                with open(json_path, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def save_metadata(self, collection_name, data):
        with open(self.get_metadata_path(collection_name), 'w') as f:
            json.dump(data, f, indent=4)

    def get_fps(self, collection_name, filename):
        data = self.load_metadata(collection_name)
        return data.get(filename, {}).get("fps", 18)

    def set_fps(self, collection_name, filename, fps):
        data = self.load_metadata(collection_name)
        if filename not in data: data[filename] = {}
        data[filename]["fps"] = fps
        self.save_metadata(collection_name, data)

    def set_file_info(self, collection_name, filename, **kwargs):
        """
        Guarda metadatos de forma flexible.
        Uso: manager.set_file_info(col, file, fps=18, pixel_format="rgb", ...)
        """
        data = self.load_metadata(collection_name)
        if filename not in data: data[filename] = {}
        
        # Guardamos/Actualizamos todos los argumentos que lleguen
        data[filename].update(kwargs)
        
        self.save_metadata(collection_name, data)

    def get_file_info(self, collection_name, filename):
        data = self.load_metadata(collection_name)
        return data.get(filename, {}) 

    def delete_file(self, collection_name, filename):
        file_path = self.root_path / collection_name / filename
        if file_path.exists():
            try: os.remove(file_path)
            except: return False
        data = self.load_metadata(collection_name)
        if filename in data:
            del data[filename]
            self.save_metadata(collection_name, data)
        return True