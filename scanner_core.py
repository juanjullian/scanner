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

# Intentamos importar psutil para medir ancho de banda real del SO
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARN: 'psutil' no instalado. La medición de ancho de banda será estimada.")

# Se ha unificado a una única resolución.
FIXED_WIDTH = 2840
FIXED_HEIGHT = 2200

class PreviewWorker(QThread):
    """
    Hilo dedicado a convertir el buffer RAW de la cámara a imagen visualizable (RGB/Mono).
    Esto libera al hilo principal de la cámara (CameraWorker) para que nunca deje de vaciar el buffer del driver.
    """
    image_ready = pyqtSignal(np.ndarray)

    def __init__(self, in_queue):
        super().__init__()
        self.in_queue = in_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                # Esperamos un buffer (objeto Arena)
                # Timeout corto para revisar self.running seguido
                buffer = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # --- CONVERSIÓN (Arena SDK) ---
                # Esto es lo que consumía CPU en el hilo crítico
                
                # Intentamos convertir a BGR8
                # Nota: Podríamos implementar lógica de "Quality vs Speed" aquí también si la cola crece.
                try:
                    image_converted = BufferFactory.convert(buffer, PixelFormat.BGR8)
                    h, w = image_converted.height, image_converted.width
                    
                    # Subsampling 2x siempre para rendimiento GUI (1420x1100)
                    # Es un buen compromiso fijo.
                    full_arr = np.ctypeslib.as_array(image_converted.pdata, shape=(h, w, 3))
                    image_final = full_arr[::2, ::2, :].copy()
                    
                    BufferFactory.destroy(image_converted)
                    
                    self.image_ready.emit(image_final)

                except Exception as e:
                    # Fallback Mono8
                    try:
                        image_converted = BufferFactory.convert(buffer, PixelFormat.Mono8)
                        h, w = image_converted.height, image_converted.width
                        full_arr = np.ctypeslib.as_array(image_converted.pdata, shape=(h, w))
                        image_final = full_arr[::2, ::2].copy()
                        BufferFactory.destroy(image_converted)
                        self.image_ready.emit(image_final)
                    except: pass
            
            except Exception as e:
                print(f"Error PreviewWorker: {e}")
            finally:
                # IMPORTANTE: Destruir la copia del buffer que nos pasaron
                if buffer:
                    BufferFactory.destroy(buffer)
                self.in_queue.task_done()

    def stop(self):
        self.running = False
        self.wait()


class CameraWorker(QThread):
    image_received = pyqtSignal(np.ndarray)
    # fps, temp, qsize, dropped_frames, bandwidth_mbps, bw_source
    stats_updated = pyqtSignal(float, float, int, int, float, str) 
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
        self.instant_frame_count = 0
        self.start_time = 0
        self.last_stats_time = 0
        self.last_frame_id = -1
        self.dropped_frames = 0
        self.last_frame_id = -1
        self.dropped_frames = 0
        self.last_stream_bytes = 0
        self.last_os_bytes = 0 # Para medición psutil
        
        # Cola y Worker para Preview (Desacoplado)
        self.preview_queue = queue.Queue(maxsize=2) # Max 2 frames de lag visual, si se llena descartamos
        self.preview_worker = PreviewWorker(self.preview_queue)
        # Reenviamos la señal del worker interno hacia fuera para que main_app no se entere del cambio
        self.preview_worker.image_ready.connect(self.image_received.emit)
        self.preview_worker.start()

    def set_queue(self, q):
        self.write_queue = q

    def clear_queue(self):
        if self.write_queue:
            with self.write_queue.mutex:
                self.write_queue.queue.clear()
            print("INFO: Cola de fotogramas limpiada antes de grabar.")

    def reset_drop_count(self):
        self.dropped_frames = 0
        self.last_frame_id = -1 # Reiniciamos tracking de ID para evitar falsos positivos al reconectar
        print("Contadores de Drop reseteados.")

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

            try:
                # Esto es CRÍTICO. Aumentamos el buffer del driver para evitar drops si el PC hipa.
                # Valor por defecto suele ser bajo (10). Lo subimos a 200.
                tl_stream['StreamDefaultBufferCount'].value = 200
                # A veces se llama StreamInputBufferCount o similar dependiendo version
            except: pass

            # Optimización Latencia: NewestOnly descarta frames viejos si la UI se traba
            # ¡OJO! Para evitar drops en grabación, necesitamos buffer grande.
            tl_stream['StreamBufferHandlingMode'].value = "NewestOnly"
            tl_stream['StreamAutoNegotiatePacketSize'].value = True
            tl_stream['StreamPacketResendEnable'].value = True
            
            # Trigger Overlap: Dejar que el archivo txt lo controle o poner ReadOut
            # nodemap.get_node('TriggerOverlap').value = 'ReadOut' 

            if nodemap.get_node('GammaEnable'): 
                nodemap.get_node('GammaEnable').value = True

            # --- FORZAR RESOLUCIÓN FIJA ---
            try:
                nodemap.get_node('OffsetX').value = 0
                nodemap.get_node('OffsetY').value = 0
                nodemap.get_node('Width').value = FIXED_WIDTH
                nodemap.get_node('Height').value = FIXED_HEIGHT
                print(f"Resolución fijada a: {FIXED_WIDTH}x{FIXED_HEIGHT}")
            except Exception as e:
                print(f"Error fijando resolución: {e}")

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
# Método set_format_roi eliminado por unificación de resolución.

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
            self.last_stats_time = time.time()
            last_temp_check = 0
            current_temp = 0.0
            
            # Reset counters
            self.frame_count = 0
            self.instant_frame_count = 0
            self.dropped_frames = 0
            self.last_frame_id = -1
            self.preview_skip_counter = 0 # Para "Best Effort Preview"

            try:
                self.device.start_stream(100)
            except Exception as e:
                print(f"Error start_stream: {e}")
                self.running = False
                return
            
            # Nodo de Bytes Totales (Medición Real desde Driver/OS)
            stream_node = None
            try:
                stream_node = self.device.tl_stream_nodemap.get_node('StreamTotalBytes')
            except:
                try:
                    stream_node = self.device.tl_stream_nodemap.get_node('StreamOctets')
                except:
                    print("WARN: No se encontraron nodos de estadísticas de flujo (StreamTotalBytes/StreamOctets). Usando cálculo manual.")
                    stream_node = None

            while self.running:
                # --- A. LECTURA DE TEMPERATURA Y ESTADÍSTICAS ---
                now = time.time()
                dt = now - self.last_stats_time
                if dt > 1.0: # Actualizar cada 1 segundo aprox
                    try: 
                        current_temp = temp_node.value 
                        
                        # Cálculo FPS Instantáneo
                        fps = self.instant_frame_count / dt
                        
                        # Cálculo Bandwidth
                        mbps = 0.0
                        bw_src = "C" # Default: Calculated via FPS * Size
                        
                        # ESTRATEGIA 1: OS Level (psutil) - La más precisa a nivel sistema
                        if PSUTIL_AVAILABLE:
                            try:
                                net = psutil.net_io_counters()
                                curr_os_bytes = net.bytes_recv # Tráfico total de bajada
                                if self.last_os_bytes > 0:
                                    delta = curr_os_bytes - self.last_os_bytes
                                    mbps = (delta * 8) / (dt * 1_000_000.0)
                                self.last_os_bytes = curr_os_bytes
                                bw_src = "OS"
                            except: pass

                        # ESTRATEGIA 2: Driver Level (StreamTotalBytes) - Si OS falla
                        if bw_src == "C" and stream_node:
                            try:
                                curr_bytes = stream_node.value
                                if self.last_stream_bytes > 0:
                                    delta = curr_bytes - self.last_stream_bytes
                                    mbps = (delta * 8) / (dt * 1_000_000.0)
                                self.last_stream_bytes = curr_bytes
                                bw_src = "D"
                            except: pass

                        # ESTRATEGIA 3: Calculated (FPS * Payload) - Fallback final
                        if bw_src == "C":
                             frame_size_mb = (2840 * 2200 * 1.5) / (1024*1024)
                             mbps = fps * frame_size_mb * 8

                        self.instant_frame_count = 0 # Reset para el próximo segundo
                        self.last_stats_time = now
                        
                        q_size = self.write_queue.qsize() if self.write_queue else 0
                        self.stats_updated.emit(fps, current_temp, q_size, self.dropped_frames, mbps, bw_src)
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

                # 1. Copia Maestra Y Detección de Drops
                # Verificamos si saltó el FrameID
                curr_id = buffer.frame_id
                if self.last_frame_id != -1:
                    diff = curr_id - self.last_frame_id
                    if diff > 1:
                        # Se perdieron (diff - 1) frames
                        drop_count = diff - 1
                        self.dropped_frames += drop_count
                        print(f"WARN: Drop detectado! Saltó de {self.last_frame_id} a {curr_id} (Perdidos: {drop_count})")
                
                self.last_frame_id = curr_id
                
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

                # A. Guardar (PRIORIDAD 1)
                is_recording = False
                if self.write_queue:
                    is_recording = True
                    try: self.write_queue.put_nowait(raw_bytes)
                    except queue.Full: pass
                
                # B. Previsualización (PRIORIDAD 2 - Best Effort)
                # Si estamos grabando, saltamos 1 de cada 2 cuadros de la vista previa
                # para liberar CPU/RAM y asegurar que la grabación no pierda frames.
                if is_recording:
                    self.preview_skip_counter += 1
                    if self.preview_skip_counter % 2 != 0:
                        BufferFactory.destroy(image_raw)
                        continue # Saltamos preview de este frame
                
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
                    # OPTIMIZACIÓN VITAL: Subsampling inmediato [::2]
                    # Reducimos de 2840x2200 a ~1420x1100 (50% resolución).
                    # Equilibrio entre fluidez y detalle para Foco (Peaking).
                    full_arr = np.ctypeslib.as_array(image_converted.pdata, shape=(h, w, 3))
                    image_for_gui = full_arr[::1, ::1, :].copy()
                    
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
                    self.instant_frame_count += 1
                    
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
        if self.preview_worker:
            self.preview_worker.stop()
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
        # Aumentar buffer de archivo a 64MB (64*1024*1024) para minimizar I/O ops
        with open(self.save_path, "wb", buffering=67108864) as f:
            while self.running or not self.frame_queue.empty():
                try:
                    data = self.frame_queue.get(timeout=0.1)
                    
                    t0 = time.perf_counter()
                    f.write(data)
                    t1 = time.perf_counter()
                    
                    dt_ms = (t1 - t0) * 1000.0
                    if dt_ms > 40.0:
                        print(f"ALERTA DISCO: Escritura lenta detectada ({dt_ms:.1f} ms). Posible cuello de botella.")
                    
                    self.frames_saved += 1
                    # Optimización: No emitir señal CADA frame, sino cada 5 frames para liberar CPU del GUI
                    if self.frames_saved % 5 == 0:
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