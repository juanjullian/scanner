import subprocess
import numpy as np
import cv2
import argparse
import sys
import os
import json
from pathlib import Path

# --- DICCIONARIO DE RESOLUCIONES (Igual que en scanner_core) ---
FORMAT_ROIS_LOOKUP = {
    "Super 8":       (2840, 2080), 
    "Regular 8mm":   (2840, 2136), 
    "16mm (Mudo)":   (2840, 2072), 
    "16mm (Sonido)": (2840, 1700), 
    "Full Sensor":   (2840, 2840) 
}

# Fallback por si no hay metadata (Tu lista original)
KNOWN_RESOLUTIONS = [
    (2840, 2080), (2840, 2136), (2840, 2072), 
    (2840, 1700), (2840, 2200), (2840, 2840)
]

def detect_format_fallback(fsize):
    """
    Intenta adivinar el formato por tamaño si no hay metadata.
    Retorna: w, h, is_rgb
    """
    for w, h in KNOWN_RESOLUTIONS:
        # Opción RGB (3 bytes)
        if fsize % int(w * h * 3) == 0:
            return w, h, True
        # Opción Bayer (1.5 bytes)
        if fsize % int(w * h * 1.5) == 0:
            return w, h, False
            
    # Default absoluto
    return 2840, 2200, False

def unpack_12bit_packed_manual(raw_bytes, width, height):
    """ Desempaqueta Bayer RG12p a uint16 """
    expected_size = int(width * height * 1.5)
    if len(raw_bytes) > expected_size: raw_bytes = raw_bytes[:expected_size]
    elif len(raw_bytes) < expected_size: raw_bytes += b'\x00' * (expected_size - len(raw_bytes))
    
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    b0 = data[:, 0].astype(np.uint16)
    b1 = data[:, 1].astype(np.uint16)
    b2 = data[:, 2].astype(np.uint16)
    
    p0 = b0 | ((b1 & 0x0F) << 8)
    p1 = (b1 >> 4) | (b2 << 4)
    
    img_flat = np.empty(width * height, dtype=np.uint16)
    img_flat[0::2] = p0
    img_flat[1::2] = p1
    return img_flat.reshape(height, width)

def main():
    # 1. Configuración de Argumentos
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--codec", default="prores") 
    parser.add_argument("--fps", default="18")
    parser.add_argument("--sharp", default="0,0")
    parser.add_argument("--mode", default="COLOR")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR|Archivo no encontrado: {input_path}")
        return

    # --- 2. DETERMINAR FORMATO (Estrategia Metadata First) ---
    w, h = 2840, 2200 # Valores iniciales
    is_rgb = False
    
    # Buscar metadata.json en la carpeta del archivo
    meta_path = input_path.parent / "metadata.json"
    
    metadata_found = False
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
                file_info = data.get(input_path.name, {})
                
                # A. Obtener Dimensiones (ROI)
                roi_key = file_info.get("roi_key")
                if roi_key in FORMAT_ROIS_LOOKUP:
                    w, h = FORMAT_ROIS_LOOKUP[roi_key]
                    metadata_found = True
                
                # B. Obtener Modo de Color
                pixel_fmt = file_info.get("pixel_format", "bayer")
                if pixel_fmt == "rgb":
                    is_rgb = True
                    metadata_found = True
        except Exception as e:
            print(f"WARN|Error leyendo metadata: {e}")

    # Si no encontramos datos en el JSON, usamos el fallback (detectar por tamaño)
    if not metadata_found:
        fsize = input_path.stat().st_size
        w, h, is_rgb = detect_format_fallback(fsize)
        print("WARN|Usando detección automática por tamaño (Metadata no disponible)")

    # Calcular bytes por cuadro
    if is_rgb:
        frame_bytes = int(w * h * 3)
    else:
        frame_bytes = int(w * h * 1.5)

    total_size = input_path.stat().st_size
    num_frames = total_size // frame_bytes
    
    print(f"INFO|Procesando: {input_path.name}")
    print(f"INFO|Resolución: {w}x{h} | Modo: {'RGB' if is_rgb else 'BAYER'} | Frames: {num_frames}")

    # --- 3. CONFIGURAR FFMPEG ---
    # Nota: Siempre le enviamos 'rgb48le' (16-bit) a FFmpeg para mantener la calidad máxima
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'rgb48le', '-r', args.fps,
        '-i', '-' 
    ]
    
    # Configuración de Codec de salida
    if args.codec == 'prores':
        ffmpeg_cmd += ['-c:v', 'prores_ks', '-profile:v', '4', '-vendor', 'apl0', '-bits_per_mb', '8000', '-pix_fmt', 'yuva444p10le']
    elif args.codec == 'hevc':
        ffmpeg_cmd += ['-c:v', 'libx265', '-x265-params', 'lossless=1', '-pix_fmt', 'yuv444p10le']
    elif args.codec == 'ffv1':
        ffmpeg_cmd += ['-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '24', '-slicecrc', '1']
    else:
        # H.264 Proxy
        ffmpeg_cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p']

    output_file = input_path.with_suffix(f".{args.codec}.mov" if args.codec == 'prores' else ".mkv" if args.codec == 'ffv1' else ".mp4")
    ffmpeg_cmd.append(str(output_file))

    # Parsear sharpness
    try: sigma, amount = map(float, args.sharp.split(','))
    except: sigma, amount = 0, 0

    # Iniciar proceso FFmpeg
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # --- 4. BUCLE DE PROCESAMIENTO ---
    try:
        with open(input_path, "rb") as f:
            for i in range(num_frames):
                raw_data = f.read(frame_bytes)
                if len(raw_data) < frame_bytes: break

                # === AQUI ESTA LA MAGIA DEL IF/ELSE ===
                if is_rgb:
                    # CASO A: RGB (Ya viene en color 8-bit)
                    img = np.frombuffer(raw_data, dtype=np.uint8).reshape(h, w, 3)
                    # Normalizamos a float (0.0 - 1.0) para aplicar efectos
                    img_f = img.astype(np.float32) / 255.0
                
                else:
                    # CASO B: BAYER (Viene crudo 12-bit packed)
                    img_unpacked = unpack_12bit_packed_manual(raw_data, w, h)
                    # Debayering simple para video (rápido)
                    # FORZADO A BG (Índice 1)
                    img_rgb = cv2.cvtColor(img_unpacked, cv2.COLOR_BayerBG2RGB)
                    # Normalizamos a float (0.0 - 1.0) desde 12-bits (4095)
                    img_f = img_rgb.astype(np.float32) / 4095.0
                # =======================================

                # --- PROCESADO VISUAL (Sharpening / BW / Color) ---
                if args.mode == "BW":
                    # Convertir a grises
                    if len(img_f.shape) == 3:
                        gray = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_f

                    if sigma > 0:
                        blur = cv2.GaussianBlur(gray, (0,0), sigma)
                        gray = gray + (gray - blur) * amount
                    
                    gray = np.clip(gray, 0, 1)
                    img_final = cv2.merge((gray, gray, gray))
                
                else:
                    # Modo Color
                    # Truco: Pasar a LAB para enfocar solo la luminancia (L)
                    lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
                    l, a, b = cv2.split(lab)
                    
                    # Suavizar canales de color (reduce ruido cromático)
                    a = cv2.GaussianBlur(a, (0,0), 3.0) 
                    b = cv2.GaussianBlur(b, (0,0), 3.0)
                    
                    if sigma > 0:
                        blur = cv2.GaussianBlur(l, (0,0), sigma)
                        l = l + (l - blur) * amount
                    
                    img_final = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_Lab2RGB)

                # --- PREPARAR SALIDA PARA FFMPEG ---
                # Convertimos a 16-bit entero (escala 0-65535)
                # Esto nos asegura la máxima calidad al entrar al encoder
                data_out = (np.clip(img_final, 0, 1) * 65535).astype(np.uint16).tobytes()
                
                try:
                    process.stdin.write(data_out)
                except (BrokenPipeError, OSError):
                    break
                
                # Reporte de progreso para la UI
                if i % 10 == 0:
                    print(f"PROG|{i}|{num_frames}")

    except Exception as e:
        print(f"ERROR|{e}")
    finally:
        if process.stdin: process.stdin.close()
        process.wait()
        print("INFO|Proceso finalizado.")

if __name__ == "__main__":
    main()