import subprocess
import numpy as np
import cv2
import argparse
import sys
import os
import json
import time
from pathlib import Path

# Intento de importar tifffile para exportación DNG
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# ==========================================
# === ÁREA DE CONFIGURACIÓN MANUAL ===
# ==========================================

# 1. CONTROL DE EXPOSICIÓN Y GAMMA (Solo afecta video, no DNG)
CFG_GAMMA = 1.4  
CFG_BLACK_LVL = 0.0
CFG_WHITE_LVL = 4095.0

# 2. CONTROL DE NITIDEZ (Solo afecta video)
CFG_SHARP_SIGMA = 2.0   
CFG_SHARP_AMOUNT = 2.5  

# 3. REDUCCIÓN DE RUIDO (Solo afecta video)
CFG_CHROMA_BLUR_SIZE = 3.0 

# ==========================================
# ==========================================

FIXED_W = 2840
FIXED_H = 2200

def detect_format_fallback(fsize):
    """ Intenta adivinar modo (RGB vs Bayer) """
    w, h = FIXED_W, FIXED_H
    if fsize % int(w * h * 3) == 0:
        return w, h, True
    if fsize % int(w * h * 1.5) == 0:
        return w, h, False
    return w, h, False

def unpack_12bit_packed_manual(raw_bytes, width, height):
    """ 
    Desempaqueta Bayer RG12p a uint16 siguiendo la receta del proyecto .NET 'RawBayer2DNG'.
    
    La lógica analizada de 'convert12pInputto16bit' en DataFormatConverter.cs es:
    - Entrada: 3 bytes (B0, B1, B2) contienen 2 píxeles de 12 bits.
    - Píxel 0 (P0) se forma con B0 y los 4 bits bajos de B1.
    - Píxel 1 (P1) se forma con los 4 bits altos de B1 y B2.
    - El proyecto .NET escala estos 12 bits a 16 bits desplazando 4 bits a la izquierda.
    """
    expected_size = int(width * height * 1.5)
    if len(raw_bytes) > expected_size: 
        raw_bytes = raw_bytes[:expected_size]
    elif len(raw_bytes) < expected_size: 
        raw_bytes += b'\x00' * (expected_size - len(raw_bytes))
    
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    
    b0 = data[:, 0].astype(np.uint16)
    b1 = data[:, 1].astype(np.uint16)
    b2 = data[:, 2].astype(np.uint16)
    
    # Receta exacta RawBayer2DNG:
    # INPUT:  [B0] [B1] [B2] (3 bytes)
    # OUTPUT: [P0_low] [P0_high] [P1_low] [P1_high] (4 bytes = 2x uint16)
    
    # Lógica C#: 
    # output[o] (low byte P0) = (input[i] & 0x0F) << 4
    # output[o+1] (high byte P0) = ((input[i] & 0xF0) >> 4) | ((input[i+1] & 0x0F) << 4)
    # -> P0 = OutputHigh << 8 | OutputLow 
    #       = ((B0 & 0xF0) << 4 | (B1 & 0x0F) << 12) | ((B0 & 0x0F) << 4)
    #       = ((B0 & 0xF0) | (B0 & 0x0F)) << 4 | (B1 & 0x0F) << 12
    #       = (B0 << 4) | ((B1 & 0x0F) << 12)
    #       = (B0 | ((B1 & 0x0F) << 8)) << 4
    
    # Implementación NumPy optimizada
    # P0: (Byte0 | (Byte1_low << 8)) << 4
    p0 = (b0 | ((b1 & 0x0F) << 8)) << 4
    
    # Lógica C# para P1:
    # output[o+2] (low byte P1) = input[i+1] & 0xF0
    # output[o+3] (high byte P1) = input[i+2]
    # -> P1 = OutputHigh << 8 | OutputLow
    #       = (B2 << 8) | (B1 & 0xF0)
    #       = ((B2 << 4) | (B1 >> 4)) << 4
    
    # P1: ((Byte1_high >> 4) | (Byte2 << 4)) << 4
    p1 = ((b1 >> 4) | (b2 << 4)) << 4
    
    img_flat = np.empty(width * height, dtype=np.uint16)
    img_flat[0::2] = p0
    img_flat[1::2] = p1
    return img_flat.reshape(height, width)

def save_dng_frame(img_16bit, path, width, height):
    """
    Guarda DNG compatible con DaVinci Resolve usando las etiquetas y matrices de 'RawBayer2DNG'.
    """
    if not HAS_TIFFFILE:
        raise ImportError("Instala tifffile: pip install tifffile")

    # RawBayer2DNG usa una matriz 'sRGB hack' para ColorMatrix1.
    # Transforma XYZ a Camera (asumiendo Camera = sRGB).
    # Matriz sRGB a XYZ (D65) inversa aproximada usada en C#:
    # R: 3.2404542, -1.5371385, -0.4985314
    # G: -0.9692660, 1.8760108, 0.0415560
    # B: 0.0556434, -0.2040259, 1.0572252
    
    # Convertimos a SRATIONAL (Num/Denom) multiplicando por 10000
    denom = 10000
    color_matrix_flat = [
        int(3.2404542 * denom), denom, int(-1.5371385 * denom), denom, int(-0.4985314 * denom), denom,
        int(-0.9692660 * denom), denom, int(1.8760108 * denom), denom, int(0.0415560 * denom), denom,
        int(0.0556434 * denom), denom, int(-0.2040259 * denom), denom, int(1.0572252 * denom), denom
    ]

    # Patrón Bayer (BGGR suele ser habitual, RawBayer2DNG permite configurar).
    # Asumimos BGGR [2,1,1,0] o RGGB [0,1,1,2]. 
    # Gige Vision suele ser RGGB.
    # RawBayer2DNG code: 0=Red, 1=Green, 2=Blue.
    # RGGB = [0, 1, 1, 2]
    cfa_pattern = [0, 1, 1, 2] 

    tiff_options = {
        'photometric': 'cfa', # 32803
        'planarconfig': 'contig',
        'rowsperstrip': height,
        'bitspersample': 16,
        'compression': None,
        'extratags': [
            # --- Etiquetas Esenciales DNG 1.4 ---
            (50706, 'B', 4, [1, 4, 0, 0], True), # DNGVersion
            (50707, 'B', 4, [1, 4, 0, 0], True), # DNGBackwardVersion
            
            # --- Etiquetas de Sensor (RawBayer2DNG usa estas ID específicas) ---
            # EXIF_CFAPattern (41730 / 0xA302)
            (41730, 'B', 4, cfa_pattern, True),
            # CFAPattern (33422 / 0x828E) - Estándar TIFF/EP
            (33422, 'B', 4, cfa_pattern, True),
            # CFARepeatPatternDim (33421 / 0x828D)
            (33421, 'H', 2, [2, 2], True),
            
            # --- Espacio de Color y Modelo ---
            (50708, 's', 0, "CustomCamera_Py", True), # UniqueCameraModel
            (50721, 10, 9, color_matrix_flat, True),   # ColorMatrix1 (SRATIONAL)
            (50778, 'H', 1, 21, True),                 # CalibrationIlluminant1 (D65)
            (50710, 10, 3, [10000, 10000, 10000, 10000, 10000, 10000], True), # AsShotNeutral (1.0, 1.0, 1.0)
            
            # --- Otros ---
            (254, 'I', 1, 0, True),     # NewSubfileType (0 = Full Res)
            (274, 'H', 1, 1, True),     # Orientation (1 = Normal)
            
            # BaselineExposure: RawBayer2DNG lo usa a veces (valor 4) si el raw es oscuro. 
            # Como ya escalamos << 4 a bright, no es estrictamente necesario, pero explicito 0.
            # (50714, 's', 1, 0, True) 
        ]
    }

    try:
        tifffile.imwrite(
            path,
            img_16bit,
            shape=(height, width),
            dtype=np.uint16,
            metadata={
                'ImageWidth': width,
                'ImageLength': height,
                'Make': 'PythonDNG',
                'Model': 'RawBayer2DNG_Clone',
                'Software': 'RawBayer2DNG_Python_Port'
            },
            **tiff_options
        )
    except Exception as e:
        print(f"ERROR|Al guardar DNG: {e}")

def apply_processing_chain(img_f, gamma, blk, wht, sigma, amount, chroma_blur):
    # ... (Tu código de procesado de video se mantiene igual) ...
    norm_blk = blk / 4095.0
    norm_wht = wht / 4095.0
    img_f = (img_f - norm_blk) / (norm_wht - norm_blk)
    img_f = np.clip(img_f, 0.0, 1.0)
    if gamma != 1.0:
        img_f = np.power(img_f, 1.0 / gamma)
    img_yuv = cv2.cvtColor(img_f, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_yuv)
    if chroma_blur > 0:
        cr = cv2.GaussianBlur(cr, (0,0), chroma_blur)
        cb = cv2.GaussianBlur(cb, (0,0), chroma_blur)
    if sigma > 0 and amount > 0:
        blur_y = cv2.GaussianBlur(y, (0,0), sigma)
        y = cv2.addWeighted(y, 1.0 + amount, blur_y, -amount, 0)
    img_processed = cv2.merge((y, cr, cb))
    img_final = cv2.cvtColor(img_processed, cv2.COLOR_YCrCb2RGB)
    return np.clip(img_final, 0.0, 1.0)

def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--codec", default="prores") # 'dng' activa el modo raw
    parser.add_argument("--fps", default="18")
    parser.add_argument("--sharp", default="0,0") 
    parser.add_argument("--mode", default="COLOR")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR|Archivo no encontrado: {input_path}")
        return

    # --- DETECCIÓN DE FORMATO ---
    w, h = FIXED_W, FIXED_H
    is_rgb = False
    
    # ... (Lógica de detección de tamaño igual a la tuya) ...
    meta_path = input_path.parent / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
                file_info = data.get(input_path.name, {})
                pixel_fmt = file_info.get("pixel_format", "bayer")
                if pixel_fmt == "rgb": is_rgb = True
        except: pass

    if not meta_path.exists():
        fsize = input_path.stat().st_size
        w, h, is_rgb = detect_format_fallback(fsize)

    frame_bytes = int(w * h * 3) if is_rgb else int(w * h * 1.5)
    total_size = input_path.stat().st_size
    num_frames = total_size // frame_bytes
    
    print(f"INFO|Procesando: {input_path.name}")
    
    # === MODO DNG ===
    is_dng_mode = (args.codec.lower() == 'dng')
    
    if is_dng_mode:
        if is_rgb:
            print("ERROR|No se puede exportar DNG desde una fuente RGB. Se requiere RAW Bayer.")
            return
        
        # Crear Carpeta contenedora: NombreArchivo_DNG
        dng_folder = input_path.parent / f"{input_path.stem}_DNG_SEQ"
        dng_folder.mkdir(exist_ok=True)
        print(f"INFO|Modo DNG RAW activo. Salida en: {dng_folder}")
        process = None # No usamos ffmpeg
    else:
        # === MODO VIDEO (FFMPEG) ===
        print(f"INFO|Res: {w}x{h} | Modo Video: {args.codec}")
        # ... (Tu configuración de FFmpeg original aquí) ...
        ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'rgb48le', '-r', args.fps,
        '-i', '-' 
        ]
        # (Aquí pongo una versión simplificada de tu selección de codec para brevedad)
        if args.codec == 'prores':
            ffmpeg_cmd += ['-c:v', 'prores_ks', '-profile:v', '2', '-vendor', 'apl0', '-qscale:v', '9', '-pix_fmt', 'yuv444p10le']
            ext = ".mov"
        else: # Default fallback
            ffmpeg_cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p']
            ext = ".mp4"
            
        output_file = input_path.parent / f"{input_path.stem}_{args.codec}{ext}"
        ffmpeg_cmd.append(str(output_file))
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=sys.stderr)

    # --- BUCLE PRINCIPAL ---
    print(f"START|{num_frames}")
    sys.stdout.flush()
    start_time = time.time()
    
    try:
        with open(input_path, "rb") as f:
            for i in range(num_frames):
                raw_data = f.read(frame_bytes)
                if len(raw_data) < frame_bytes: break

                # 1. UNPACKING (Común para ambos)
                if is_rgb:
                    img = np.frombuffer(raw_data, dtype=np.uint8).reshape(h, w, 3)
                    img_f = img.astype(np.float32) / 255.0
                else:
                    # AQUÍ está la clave: Obtenemos el Bayer 16-bit ya escalado (0-65520)
                    img_unpacked_16bit = unpack_12bit_packed_manual(raw_data, w, h)

                # 2. RAMIFICACIÓN DE PROCESO
                if is_dng_mode and not is_rgb:
                    # === RUTA DNG (RAW PURO) ===
                    # Nombre formato: NombreOriginal_000000.dng
                    frame_name = f"{input_path.stem}_{i:06d}.dng"
                    # Ya tenemos 16-bit "Bright" (<<4), listo para guardar
                    save_dng_frame(img_unpacked_16bit, dng_folder / frame_name, w, h)
                    
                else:
                    # === RUTA VIDEO (DEBAYER + COLOR + FFMPEG) ===
                    if not is_rgb:
                        # Debayer manual para video requires 8/16 bit normalizado?
                        # img_unpacked_16bit es 0-65535.
                        img_rgb = cv2.cvtColor(img_unpacked_16bit, cv2.COLOR_BayerBG2RGB)
                        img_f = img_rgb.astype(np.float32) / 65535.0 # Normalizamos a 0-1
                    
                    # Tu cadena de procesado
                    img_proc = apply_processing_chain(
                        img_f, CFG_GAMMA, CFG_BLACK_LVL, CFG_WHITE_LVL, 
                        CFG_SHARP_SIGMA, CFG_SHARP_AMOUNT, CFG_CHROMA_BLUR_SIZE
                    )
                    
                    # Manejo BW
                    if args.mode == "BW":
                        if len(img_proc.shape) == 3:
                            gray = cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY)
                        else:
                            gray = img_proc
                        img_final = cv2.merge((gray, gray, gray))
                    else:
                        img_final = img_proc

                    # Envío a FFmpeg
                    data_out = (np.clip(img_final, 0, 1) * 65535).astype(np.uint16).tobytes()
                    if process:
                        try:
                            process.stdin.write(data_out)
                        except (BrokenPipeError, OSError):
                            break
                
                # Reporte de progreso
                if i % 10 == 0: # Reportar cada 10 frames para no saturar consola en DNG
                    elapsed = time.time() - start_time
                    fps_proc = (i + 1) / elapsed if elapsed > 0 else 0
                    percent = ((i+1) / num_frames) * 100
                    print(f"PROG|{i+1}|{num_frames}")
                    print(f"INFO|Frame: {i+1}/{num_frames} ({percent:.1f}%) | Vel: {fps_proc:.2f} fps")
                    sys.stdout.flush()

    except Exception as e:
        print(f"ERROR|{e}")
    finally:
        if process and process.stdin: 
            process.stdin.close()
            process.wait()
        print("INFO|Proceso finalizado.")

if __name__ == "__main__":
    main()
