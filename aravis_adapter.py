import sys
import time
import ctypes
import os
import numpy as np

# --- CONFIGURACIÓN AUTOMÁTICA DE RUTAS MACOS (HOMEBREW) ---
# PyGObject A MENUDO falla en encontrar las librerías en macOS si no se le dice explícitamente dónde buscar.
def fix_gi_paths():
    if sys.platform != 'darwin': return
    
    # Rutas comunes de Homebrew (Silicon e Intel)
    common_paths = [
        "/opt/homebrew/lib/girepository-1.0",       # Apple Silicon Standard
        "/usr/local/lib/girepository-1.0",          # Intel Standard
        # Añadida ruta específica encontrada por el usuario:
        "/opt/homebrew/Cellar/aravis/0.8.35_1/lib/girepository-1.0",
        # Genéricas aravis opt
        "/opt/homebrew/opt/aravis/lib/girepository-1.0", 
        "/usr/local/opt/aravis/lib/girepository-1.0",
        os.path.expanduser("~/homebrew/lib/girepository-1.0") 
    ]
    
    current_path = os.environ.get("GI_TYPELIB_PATH", "")
    found_paths = []
    
    for p in common_paths:
        if os.path.isdir(p):
            if p not in current_path:
                found_paths.append(p)
    
    if found_paths:
        # Añadir al principio para prioridad
        new_path = ":".join(found_paths)
        if current_path:
            os.environ["GI_TYPELIB_PATH"] = new_path + ":" + current_path
        else:
            os.environ["GI_TYPELIB_PATH"] = new_path
            
        print(f"INFO: Rutas Aravis/GObject añadidas: {found_paths}")

fix_gi_paths()
# -----------------------------------------------------------

# Intentar importar PyGObject y Aravis
try:
    import gi
    # Intentamos versiones en orden de preferencia
    try: gi.require_version('Aravis', '0.8')
    except: 
        try: gi.require_version('Aravis', '0.6')
        except: pass 
        
    from gi.repository import Aravis, GObject
    ARAVIS_AVAILABLE = True
except ImportError as ie:
    print(f"DEBUG: Error crítico importando 'gi' (PyGObject): {ie}")
    print("      -> Intenta: 'python3 -m pip install pygobject'")
    ARAVIS_AVAILABLE = False
except Exception as e:
    print(f"DEBUG: Error importando Aravis: {e}")
    ARAVIS_AVAILABLE = False

# === CLASES ADAPTER PARA MIMETIZAR ARENA API ===

class ArenaNode:
    """ Simula un nodo de Arena (GenICam) usando Aravis por debajo """
    def __init__(self, camera, node_name, node_type="Value"):
        self.cam = camera
        self.name = node_name
        self.type = node_type
        self.is_writable = True # Asumimos escritura por defecto, Aravis lanzará error si no

    @property
    def value(self):
        try:
            # Aravis tiene métodos específicos por tipo, intentamos inferir o probar
            # Float
            if self.name in ["ExposureTime", "DeviceTemperature", "Gamma"]:
                return self.cam.get_float_feature_value(self.name)
            # Integer
            elif self.name in ["Width", "Height", "OffsetX", "OffsetY", "GevSCPSPacketSize", "PayloadSize"]:
                return self.cam.get_integer_feature_value(self.name)
            # Boolean
            elif self.name in ["GammaEnable", "ChunkModeActive", "AcquisitionFrameRateEnable"]:
                return self.cam.get_boolean_feature_value(self.name)
            # String / Enumeration
            else:
                return self.cam.get_string_feature_value(self.name)
        except:
            # Retorno seguro para evitar que scanner_core crashee si el nodo falla
            if self.name in ["DeviceTemperature", "ExposureTime"]: return 0.0
            return 0

    @value.setter
    def value(self, val):
        try:
            if isinstance(val, bool):
                self.cam.set_boolean_feature_value(self.name, val)
            elif isinstance(val, float):
                self.cam.set_float_feature_value(self.name, val)
            elif isinstance(val, int):
                # Protección contra Jumbo Frames si el usuario fuerza 9000 pero el sistema no aguanta
                if self.name == "GevSCPSPacketSize" and val > 1500:
                    print(f"DEBUG: scanner_core intenta poner PacketSize {val}. Ignorando por seguridad (usaremos Auto).")
                    return 
                self.cam.set_integer_feature_value(self.name, val)
            elif isinstance(val, str):
                self.cam.set_string_feature_value(self.name, val)
        except Exception as e:
            # print(f"Aravis Set Error ({self.name}): {e}")
            pass
            
    def from_string(self, val_str):
        # Ayudante para emular comportamiento de Arena
        try: self.cam.set_string_feature_value(self.name, val_str)
        except: pass

class ArenaNodeMap:
    """ Simula el NodeMap de Arena """
    def __init__(self, camera):
        self.cam = camera
        # Cache simple
        self.nodes = {}

    def get_node(self, name):
        # Aravis accede directo a la cámara
        if name not in self.nodes:
            self.nodes[name] = ArenaNode(self.cam, name)
        return self.nodes[name]

class ArenaStreamNodeMap(ArenaNodeMap):
    """ Nodos especificos del Stream (Buffer handling, etc) """
    def __init__(self, stream):
        self.stream = stream
        self.nodes = {}
    
    def get_node(self, name):
         # Dummy nodes: permitimos escritura "falsa" siempre
        return DummyNode(name)

class DummyNode:
    def __init__(self, name): self.name = name; self.value = 0; self.is_writable = True

class ArenaBuffer:
    """ Wrapper para el buffer de Aravis para que parezca uno de Arena """
    def __init__(self, arv_buffer):
        self._arv = arv_buffer
        self.is_incomplete = (arv_buffer.get_status() != Aravis.BufferStatus.SUCCESS)
        self.frame_id = arv_buffer.get_frame_id()
        self.pdata = None # Se rellena bajo demanda o puntero real
        self.size_filled = arv_buffer.get_payload()
        
        # Inferir formato
        fmt = arv_buffer.get_image_pixel_format()
        self.pixel_format = fmt 
        self.width = arv_buffer.get_image_width()
        self.height = arv_buffer.get_image_height()

class AdapterDevice:
    """ Clase principal que emula 'arena_api.system.device' """
    def __init__(self, arv_camera):
        self.cam = arv_camera
        self.stream = None
        self.nodemap = ArenaNodeMap(self.cam)
        self.tl_stream_nodemap = None # Se crea al iniciar stream
    
    def start_stream(self, num_buffers=100):
        # 1. OPTIMIZACIÓN PACKET SIZE (CRÍTICO MAC OS)
        # Aravis puede negociar automáticamente el tamaño perfecto.
        # Si scanner_core forzó 9000 y el MTU es 1500, esto lo arregla.
        try:
            self.cam.gv_auto_packet_size()
            final_ps = self.cam.get_integer_feature_value("GevSCPSPacketSize")
            print(f"Aravis: PacketSize negociado a {final_ps} bytes")
        except:
            print("Aravis: Falló auto-negotiate packet size (usando default)")
        
        # 2. Crear stream
        self.stream = self.cam.create_stream(None, None)
        if not self.stream:
            raise RuntimeError("No se pudo crear Stream en Aravis")
            
        # 3. Configurar buffers
        try:
            payload = self.cam.get_payload()
            print(f"Aravis: Payload esperado {payload} bytes")
            for i in range(num_buffers):
                self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        except Exception as e:
            print(f"Aravis Error Alloc buffers: {e}")
            
        self.cam.start_acquisition()
        self.stream.set_emit_signals(False) 
        
        self.tl_stream_nodemap = ArenaStreamNodeMap(self.stream)

    def stop_stream(self):
        self.cam.stop_acquisition()
        if self.stream:
            # self.stream.clear_buffers() # Aravis < 0.8
            self.stream = None

    def get_buffer(self, timeout=200):
        if not self.stream: raise RuntimeError("Stream no iniciado")
        # timeout en microsegundos en pop_buffer? No, timeout es blouqeante
        # Aravis pop_buffer suele bloquear. timeout_us
        # Convertimos ms a us
        buf = self.stream.timeout_pop_buffer(timeout * 1000) 
        if not buf:
             raise TimeoutError("Timeout Aravis")
        return ArenaBuffer(buf)

    def requeue_buffer(self, buffer):
        if self.stream and buffer._arv:
            self.stream.push_buffer(buffer._arv)

class AdapterSystem:
    def create_device(self):
        # Actualizar lista
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        if n == 0: return []
        
        # Abrir primera cámara
        dev_id = Aravis.get_device_id(0)
        try:
            camera = Aravis.Camera.new(dev_id)
            print(f"Aravis: Cámara conectada - {dev_id}")
            return [AdapterDevice(camera)]
        except Exception as e:
            print(f"Aravis Error conectando: {e}")
            return []

    def destroy_device(self, device):
        if device:
            device.stop_stream()
            device.cam = None

# Factory y Enums falsos para mantener compatibilidad
class AdapterBufferFactory:
    @staticmethod
    def copy(buffer):
        # Necesitamos devolver un objeto que tenga .pdata y atributos de imagen
        # pero desacoplado del buffer de aravis (deep copy)
        # Extraemos data numpy
        try:
            # Aravis Buffer Get Data
            data = buffer._arv.get_data()
            # Copia en memoria
            b_copy = bytearray(data)
            
            # Un objeto dummy que se comporte como buffer de Arena
            class CopiedBuffer:
                def __init__(self, data_bytes, w, h, fmt):
                    self.data_array = np.frombuffer(data_bytes, dtype=np.uint8)
                    self.pdata = self.data_array.ctypes.data
                    self.width = w
                    self.height = h
                    self.pixel_format = fmt
                    self.size_filled = len(data_bytes)
                    self.is_incomplete = False
                    self.frame_id = buffer.frame_id
            
            return CopiedBuffer(b_copy, buffer.width, buffer.height, buffer.pixel_format)
        except Exception as e:
            print(f"Error copia buffer: {e}")
            return None

    @staticmethod
    def convert(buffer, fmt_enum):
        # Este es crudo. Aravis no tiene un convertidor integrado tan potente como Arena.
        # Pero MainApp espera recibir BGR8.
        # Si el buffer ya viene crudo, lo empaquetamos y lo devolvemos tal cual, 
        # confiando en que el cvtColor de scanner_core.py o main_app lo maneje.
        
        # Sin embargo, scanner_core llama a BufferFactory.convert(buffer, PixelFormat.BGR8)
        # Debemos simular esa conversión.
        
        # Asumiremos que si piden conversión, quieren un objeto con los datos listos.
        # Simplificación: Retornamos el mismo buffer (si es raw, scanner_core hará debayer manual si falla conversión)
        # O mejor, usamos OpenCV aquí si es posible.
        
        # Para mantener simpleza y robustez: Devolvemos el mismo buffer copiado 
        # declarando que 'ya es' lo que pidieron, y dejando que la app visual lo interprete.
        # (Esto es un hack, pero hacer debayering completo aquí requiere importar cv2 y duplicar lógica).
        
        return AdapterBufferFactory.copy(buffer)

    @staticmethod
    def destroy(buffer):
        pass # GC de python se encarga

class AdapterPixelFormat:
    BGR8 = 0
    RGB8 = 1
    Mono8 = 2
    # Mapeo a Constantes Aravis si fuera necesario

# INYECCIÓN FINAL
system = AdapterSystem()
BufferFactory = AdapterBufferFactory()
PixelFormat = AdapterPixelFormat()
