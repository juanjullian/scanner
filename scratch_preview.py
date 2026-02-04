import cv2
import numpy as np
import time
import queue
from PyQt6.QtCore import QThread, pyqtSignal

# Clase dedicada a procesar la imagen para la GUI en paralelo
class PreviewWorker(QThread):
    image_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, in_queue):
        super().__init__()
        self.in_queue = in_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                # Timeout bloqueante para no quemar CPU si no hay datos
                # Packet: (raw_bytes, w, h, is_rgb)
                packet = self.in_queue.get(timeout=0.1)
                
                raw_bytes, w, h, is_rgb = packet
                
                # --- PROCESAMIENTO PREVIEW (Desacoplado del Driver) ---
                if is_rgb:
                    # RGB8 (Ya viene en formato visible, solo reshape)
                    img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, w, 3)
                    # OpenCV usa BGR, RGB8 suele ser RGB o BGR dependiendo de la cámara.
                    # Asumimos que viene lista o hacemos swap si colores raros.
                    # Para preview rápido, a veces el swap BGR/RGB se puede hacer en el visor o ignorar.
                    # Haremos un subsampling inmediato para velocidad
                    preview_img = img[::2, ::2, :].copy() 
                else:
                    # BAYER RG12Packed o similar.
                    # Como ya tenemos los bytes crudos, usar OpenCV para debayering es complejo con 12p.
                    # TRUCO: Mismo algoritmo "Fast" que usamos en raw2video playback
                    # Interpretamos como uint8 crudo y hacemos subsampling directo RG.
                    
                    # Desempaquetado rápido visual (tomando MSBs) o simplemente ver canal verde
                    # Para máxima velocidad en vivo, usamos el método "Super Fast Preview" 
                    # Simular imagen BN o Color falso rápido.
                    
                    # Si queremos COLOR REAL: Necesitamos debayer.
                    # cv2.demosaicing requiere 8 o 16 bits. Tenemos 12 packed.
                    # Desempaquetar 12p en Python es lento.
                    
                    # SOLUCIÓN HÍBRIDA:
                    # El CameraWorker ya extraía una copia para grabar.
                    # Pero para la GUI, antes usábamos el SDK (BufferFactory.convert).
                    # El SDK es muy rápido (C++). Replicarlo en Python puro será más lento.
                    
                    # ESTRATEGIA REVISADA:
                    # El PreviewWorker DEBE recibir el Buffer de Arena (copiado) para usar el SDK, 
                    # O debemos aceptar que la conversión sea más lenta/distinta.
                    
                    # Volvemos a la opción: CameraWorker hace BufferFactory.copy() -> PreviewWorker.
                    # PreviewWorker llama BufferFactory.convert() -> destroy().
                    pass

                self.in_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error PreviewWorker: {e}")

    def stop(self):
        self.running = False
        self.wait()
