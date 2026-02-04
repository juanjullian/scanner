import subprocess
import numpy as np
import cv2
import argparse
import sys
import os
import json
import time
from pathlib import Path

# ==========================================
# === ÁREA DE CONFIGURACIÓN MANUAL ===
# ==========================================

# 1. CONTROL DE EXPOSICIÓN Y GAMMA
# Gamma: 1.0 = Lineal. 1.4 es un buen punto de partida para "ver" en las sombras.
CFG_GAMMA = 1.4  

# Niveles de Entrada (Rango 0 - 4095 para 12-bit)
CFG_BLACK_LVL = 0.0
CFG_WHITE_LVL = 4095.0

# 2. CONTROL DE NITIDEZ (SHARPENING)
# Restaurados a tus valores preferidos (Fuerte)
CFG_SHARP_SIGMA = 2.0   # Radio del enfoque (DCB anterior usaba 2.0)
CFG_SHARP_AMOUNT = 2.5  # Fuerza del enfoque (DCB anterior usaba 2.5)

# 3. REDUCCIÓN DE RUIDO DE COLOR (CHROMA DENOISE)
# Esto desenfoca solo el color para borrar los pixeles rojos/verdes aleatorios
# sin perder nitidez en la imagen (borde).
CFG_CHROMA_BLUR_SIZE = 3.0 # Valor típico para borrar ruido de sensor Bayer

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
    """ Desempaqueta Bayer RG12p a uint16 de forma optimizada """
    expected_size = int(width * height * 1.5)
    if len(raw_bytes) > expected_size: 
        raw_bytes = raw_bytes[:expected_size]
    elif len(raw_bytes) < expected_size: 
        raw_bytes += b'\x00' * (expected_size - len(raw_bytes))
    
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

def apply_processing_chain(img_f, gamma, blk, wht, sigma, amount, chroma_blur):
    """
    Cadena de procesado: Niveles -> Gamma -> YCrCb -> Denoise Color -> Sharpen Luma
    """
    # 1. NIVELES (Normalización segura)
    norm_blk = blk / 4095.0
    norm_wht = wht / 4095.0
    
    # Estirar histograma
    img_f = (img_f - norm_blk) / (norm_wht - norm_blk)
    img_f = np.clip(img_f, 0.0, 1.0)
    
    # 2. GAMMA (Levantar sombras)
    if gamma != 1.0:
        img_f = np.power(img_f, 1.0 / gamma)

    # 3. SEPARACIÓN LUMA / CHROMA (YCrCb)
    # Usamos YCrCb en lugar de LAB porque es más estable para video y evita clipping raro.
    # Y = Luma (Brillo/Detalle) | Cr, Cb = Chroma (Color)
    img_yuv = cv2.cvtColor(img_f, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_yuv)

    # 4. CHROMA DENOISE (Aquí borramos el ruido de color)
    # Aplicamos Blur solo a los canales de color (Cr, Cb)
    if chroma_blur > 0:
        # El kernel debe ser impar, calculamos dinámicamente o usamos 0 para que sigma decida
        cr = cv2.GaussianBlur(cr, (0,0), chroma_blur)
        cb = cv2.GaussianBlur(cb, (0,0), chroma_blur)

    # 5. SHARPENING (Solo en canal Y)
    # Esto da nitidez sin aumentar el ruido de color
    if sigma > 0 and amount > 0:
        blur_y = cv2.GaussianBlur(y, (0,0), sigma)
        # Unsharp Mask: Original + (Original - Blur) * Amount
        y = cv2.addWeighted(y, 1.0 + amount, blur_y, -amount, 0)
    
    # Reconstruir imagen RGB
    img_processed = cv2.merge((y, cr, cb))
    img_final = cv2.cvtColor(img_processed, cv2.COLOR_YCrCb2RGB)
    
    # Clip final de seguridad
    return np.clip(img_final, 0.0, 1.0)

def main():
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

    # --- DETECCIÓN DE FORMATO ---
    w, h = FIXED_W, FIXED_H
    is_rgb = False
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
    print(f"INFO|Res: {w}x{h} | Modo: {'RGB' if is_rgb else 'BAYER'}")
    print(f"INFO|Config: Gamma={CFG_GAMMA}, Sharp={CFG_SHARP_AMOUNT}/{CFG_SHARP_SIGMA}, ChromaBlur={CFG_CHROMA_BLUR_SIZE}")

    # --- CONFIGURAR FFMPEG ---
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'rgb48le', '-r', args.fps,
        '-i', '-' 
    ]
    
    ext = ".mp4"
    if args.codec == 'prores':
        ffmpeg_cmd += ['-c:v', 'prores_ks', '-profile:v', '2', '-vendor', 'apl0', '-qscale:v', '9', '-pix_fmt', 'yuv444p10le']
        ext = ".mov"
    elif args.codec == 'prores_hq':
        ffmpeg_cmd += ['-c:v', 'prores_ks', '-profile:v', '2', '-vendor', 'apl0', '-qscale:v', '9', '-pix_fmt', 'yuv422p10le']
        ext = ".mov"
    elif args.codec == 'cineform':
        # Calculamos padding al próximo múltiplo de 16
        pad_w = ((w + 15) // 16) * 16
        pad_h = ((h + 15) // 16) * 16
        
        # Filtro de padding. 
        # Importante: Cineform en ffmpeg (cfhd) requiere gbrp12le o yuv422p10le.
        ffmpeg_cmd += [
            '-vf', f'pad={pad_w}:{pad_h}:0:0:black', # Agrega borde negro a la derecha/abajo
            '-c:v', 'cfhd',
            '-quality:v', '4',  # 0 (Low) a 12 (High). 4 es "Film Scan 1".
            '-pix_fmt', 'gbrp12le' 
        ]
        ext = ".mov"
    elif args.codec == 'hevc':
        ffmpeg_cmd += ['-c:v', 'libx265', '-x265-params', 'crf=12:keyint=24', '-preset', 'slow', '-pix_fmt', 'yuv444p12le']
        ext = ".mov"
    elif args.codec == 'ffv1':
        ffmpeg_cmd += ['-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '24', '-slicecrc', '1']
        ext = ".mkv"
    else:
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

                # 1. LECTURA RAW -> FLOAT 0.0-1.0
                if is_rgb:
                    img = np.frombuffer(raw_data, dtype=np.uint8).reshape(h, w, 3)
                    img_f = img.astype(np.float32) / 255.0
                else:
                    img_unpacked = unpack_12bit_packed_manual(raw_data, w, h)
                    img_rgb = cv2.cvtColor(img_unpacked, cv2.COLOR_BayerBG2RGB)
                    img_f = img_rgb.astype(np.float32) / 4095.0

                # 2. PROCESAMIENTO UNIFICADO
                img_proc = apply_processing_chain(
                    img_f, 
                    CFG_GAMMA, 
                    CFG_BLACK_LVL, 
                    CFG_WHITE_LVL, 
                    CFG_SHARP_SIGMA, 
                    CFG_SHARP_AMOUNT,
                    CFG_CHROMA_BLUR_SIZE
                )

                # 3. MODO BW (Si se requiere)
                if args.mode == "BW":
                    if len(img_proc.shape) == 3:
                        gray = cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_proc
                    img_final = cv2.merge((gray, gray, gray))
                else:
                    img_final = img_proc

                # 4. SALIDA (16-bit int)
                data_out = (np.clip(img_final, 0, 1) * 65535).astype(np.uint16).tobytes()
                
                try:
                    process.stdin.write(data_out)
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
                
                if i % 1 == 0:
                    elapsed = time.time() - start_time
                    fps_proc = (i + 1) / elapsed if elapsed > 0 else 0
                    percent = ((i+1) / num_frames) * 100
                    print(f"PROG|{i+1}|{num_frames}")
                    print(f"INFO|Frame: {i+1}/{num_frames} ({percent:.1f}%) | Vel: {fps_proc:.2f} fps")
                    sys.stdout.flush()

    except Exception as e:
        print(f"ERROR|{e}")
    finally:
        if process.stdin: process.stdin.close()
        process.wait()
        print("INFO|Proceso finalizado.")

if __name__ == "__main__":
    main()