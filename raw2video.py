import subprocess
import struct
import numpy as np
import cv2
import argparse
import sys
import os
import json
import time
from pathlib import Path
import qoi_utils
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque

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


def es_glitch_magenta(img_uint16, factor=1.4, umbral=40):
    """
    Detecta dominancia magenta analizando las 4 fases del mosaico Bayer.
    Los valores están en escala 16-bit (uint16 desplazado <<4 desde 12-bit).
    umbral=40 equivale al umbral_minimo del script original: se aplica sobre los
    valores 16-bit tal como los almacena el DNG (rawpy devuelve estos mismos valores),
    por lo que 40 es prácticamente "cualquier señal por encima del ruido puro".
    El guardián real contra falsos positivos es la ratio (factor), no el umbral.
    Devuelve True si el frame tiene dominancia R+B sobre G (glitch magenta).
    """
    f00 = np.mean(img_uint16[0::2, 0::2])
    f01 = np.mean(img_uint16[0::2, 1::2])
    f10 = np.mean(img_uint16[1::2, 0::2])
    f11 = np.mean(img_uint16[1::2, 1::2])
    fases = sorted([f00, f01, f10, f11])
    g_alto  = fases[1]
    rb_bajo = fases[2]
    return rb_bajo > (g_alto * factor) and rb_bajo > umbral


def _save_dng_task(img_16bit, out_path, width, height):
    """
    Guarda un frame ya desempaquetado como DNG.
    La detección de glitch y la elección de destino ocurren en el hilo principal;
    este task solo realiza la escritura I/O en el thread pool.
    """
    save_dng_frame(img_16bit, out_path, width, height)
    return out_path.name


def process_video_frame(raw_data, is_rgb, width, height, mode):
    """
    Procesa un frame (RGB o Bayer) y devuelve los bytes listos
    para enviar a FFmpeg (rgb24 o rgb48le según el caso).
    Pensado para ejecutarse en hilos paralelos.
    """
    if is_rgb:
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
        img_f = img.astype(np.float32) / 255.0
    else:
        img_unpacked_16bit = unpack_12bit_packed_manual(raw_data, width, height)
        img_rgb = cv2.cvtColor(img_unpacked_16bit, cv2.COLOR_BayerBG2RGB)
        img_f = img_rgb.astype(np.float32) / 65535.0

    img_proc = apply_processing_chain(
        img_f, CFG_GAMMA, CFG_BLACK_LVL, CFG_WHITE_LVL,
        CFG_SHARP_SIGMA, CFG_SHARP_AMOUNT, CFG_CHROMA_BLUR_SIZE,
    )

    if mode == "BW":
        if len(img_proc.shape) == 3:
            gray = cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_proc
        img_final = cv2.merge((gray, gray, gray))
    else:
        img_final = img_proc

    rgb8 = (np.clip(img_final, 0, 1) * 255).astype(np.uint8)

    if is_rgb:
        return rgb8.tobytes()

    img_final_16 = (rgb8.astype(np.uint32) * 65535 // 255).astype(np.uint16)
    return img_final_16.tobytes()

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
    parser.add_argument("--output", default=None, help="Nombre de archivo o carpeta de salida (relativo al directorio del .raw)")
    args = parser.parse_args()

    global CFG_SHARP_SIGMA, CFG_SHARP_AMOUNT
    try:
        sigma, amount = map(float, str(args.sharp).split(","))
        CFG_SHARP_SIGMA = sigma
        CFG_SHARP_AMOUNT = amount
    except (TypeError, ValueError):
        pass
    if CFG_SHARP_SIGMA > 0 or CFG_SHARP_AMOUNT > 0:
        print(f"INFO|Sharpen export: sigma={CFG_SHARP_SIGMA:.1f} amount={CFG_SHARP_AMOUNT:.1f}")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR|Archivo no encontrado: {input_path}")
        return

    # --- DETECCIÓN DE FORMATO ---
    w, h = FIXED_W, FIXED_H
    is_rgb = False
    is_qoi = False
    pixel_fmt = "bayer"
    
    meta_path = input_path.parent / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
                file_info = data.get(input_path.name, {})
                pixel_fmt = file_info.get("pixel_format", "bayer")
                if pixel_fmt == "rgb": is_rgb = True
                elif pixel_fmt == "qoi_rgb": is_qoi = True
        except: pass

    if not meta_path.exists():
        fsize = input_path.stat().st_size
        w, h, is_rgb = detect_format_fallback(fsize)

    # Calcular frames según formato
    if is_qoi:
        print(f"INFO|Formato detectado: QOI_RGB8 (contenedor con headers). Indexando frames...")
        qoi_index = qoi_utils.build_frame_index(str(input_path))
        num_frames = len(qoi_index)
        frame_bytes = 0  # Variable
        print(f"INFO|{num_frames} frames QOI encontrados.")
    else:
        frame_bytes = int(w * h * 3) if is_rgb else int(w * h * 1.5)
        total_size = input_path.stat().st_size
        num_frames = total_size // frame_bytes
        qoi_index = None
    
    print(f"INFO|Procesando: {input_path.name}")
    
    # === MODO DNG ===
    is_dng_mode = (args.codec.lower() == 'dng')
    is_tiff_seq_mode = (args.codec.lower() == 'tiff_seq')
    
    if is_dng_mode:
        if is_rgb or is_qoi:
            print("ERROR|No se puede exportar DNG desde una fuente RGB/QOI. Se requiere RAW Bayer.")
            return
        
        dng_folder      = input_path.parent / (args.output or f"{input_path.stem}_DNG_SEQ")
        descarte_folder = input_path.parent / f"{dng_folder.name}_descarte"
        dng_folder.mkdir(exist_ok=True)
        descarte_folder.mkdir(exist_ok=True)
        print(f"INFO|Modo DNG RAW activo. Salida en: {dng_folder}")
        print(f"INFO|Carpeta de descarte (glitch magenta): {descarte_folder}")
        process = None

    elif is_tiff_seq_mode:
        tiff_folder = input_path.parent / (args.output or f"{input_path.stem}_TIFF_SEQ")
        tiff_folder.mkdir(exist_ok=True)
        print(f"INFO|Modo TIFF Sequence activo (fiel al ISP). Salida en: {tiff_folder}")
        process = None
    else:
        # === MODO VIDEO (FFMPEG) ===
        # Para RGB/QOI: Entrada en rgb24 (8bit). Para Bayer: rgb48le (16bit).
        is_processed_src = (is_rgb or is_qoi)
        input_pix_fmt = 'rgb24' if is_processed_src else 'rgb48le'
        print(f"INFO|Res: {w}x{h} | Modo Video: {args.codec} | Fuente: {'RGB8 ISP' if is_processed_src else 'Bayer'} | pix_fmt: {input_pix_fmt}")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', input_pix_fmt, '-r', args.fps,
            '-i', '-'
        ]
        
        if args.codec == 'prores':
            ffmpeg_cmd += ['-c:v', 'prores_ks', '-profile:v', '4', '-vendor', 'apl0', '-qscale:v', '5', '-pix_fmt', 'yuv444p10le']
            ext = ".mov"
        elif args.codec == 'prores_hq':
            ffmpeg_cmd += ['-c:v', 'prores_ks', '-profile:v', '3', '-vendor', 'apl0', '-qscale:v', '9', '-pix_fmt', 'yuv422p10le']
            ext = ".mov"
        elif args.codec == 'cineform':
            ffmpeg_cmd += ['-c:v', 'cfhd', '-quality', '5']
            ext = ".mov"
        elif args.codec == 'hevc':
            ffmpeg_cmd += ['-c:v', 'libx265', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv444p10le', '-tag:v', 'hvc1']
            ext = ".mp4"
        elif args.codec == 'h264':
            ffmpeg_cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p']
            ext = ".mp4"
        else:
            ffmpeg_cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p']
            ext = ".mp4"
            
        output_file = input_path.parent / (args.output or f"{input_path.stem}_{args.codec}{ext}")
        ffmpeg_cmd.append(str(output_file))
        print(f"INFO|FFmpeg cmd: {' '.join(ffmpeg_cmd)}")
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=sys.stderr)

    # --- BUCLE PRINCIPAL ---
    print(f"START|{num_frames}")
    sys.stdout.flush()
    start_time = time.time()
    
    is_processed_src = (is_rgb or is_qoi)

    try:
        if is_qoi:
            # --- LECTURA QOI (contenedor con headers) ---
            for i in range(num_frames):
                offset, fsz = qoi_index[i]
                qoi_data = qoi_utils.read_frame_at(str(input_path), offset, fsz)
                img = qoi_utils.decode_qoi(qoi_data, w, h)  # -> (H, W, 3) uint8 RGB

                if is_tiff_seq_mode:
                    if not HAS_TIFFFILE:
                        print("ERROR|Se requiere tifffile para exportar TIFF. Instálalo: pip install tifffile")
                        return
                    frame_name = f"{input_path.stem}_{i:06d}.tif"
                    tifffile.imwrite(str(tiff_folder / frame_name), img)
                elif is_dng_mode:
                    pass
                else:
                    # Video FFmpeg: RGB8 directo
                    if args.mode == "BW":
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img = cv2.merge((gray, gray, gray))
                    data_out = img.tobytes()
                    if process:
                        try:
                            process.stdin.write(data_out)
                        except (BrokenPipeError, OSError):
                            break
                
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_proc = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"PROG|{i+1}|{num_frames}")
                    print(f"INFO|Frame: {i+1}/{num_frames} ({(i+1)/num_frames*100:.1f}%) | Vel: {fps_proc:.2f} fps")
                    sys.stdout.flush()
        else:
            # --- LECTURA ESTÁNDAR (Bayer / RGB plano) ---
            if is_dng_mode:
                # MODO DNG: pipeline de dos pools para mantener CPU y NVMe saturados.
                #
                # unpack_pool: desempaqueta 12-bit → uint16 y detecta glitch (CPU-bound, paralelo).
                # save_pool:   escribe el DNG en disco (I/O-bound, paralelo).
                # main thread: recoge resultados del unpack EN ORDEN y aplica la ventana temporal;
                #              solo decide el destino de cada frame, sin hacer trabajo pesado.
                #
                # Ventana temporal: rachas ≤ MAX_GLITCH_RUN → glitch (_descarte, nro original).
                #                   rachas > MAX_GLITCH_RUN → cast natural (dng_folder, nro continuo).
                MAX_GLITCH_RUN = 5

                total_cores = multiprocessing.cpu_count()
                dng_workers = max(1, total_cores - 2)
                PIPELINE    = max(8, dng_workers * 3)   # frames en vuelo simultáneo
                print(f"INFO|DNG pipeline: {dng_workers} workers unpack + {dng_workers} workers escritura "
                      f"({total_cores} cores totales, profundidad {PIPELINE})")

                unpack_pool = ThreadPoolExecutor(max_workers=dng_workers)
                save_pool   = ThreadPoolExecutor(max_workers=dng_workers)

                unpack_queue = deque()   # (orig_idx, Future<(img, is_magenta)>)
                save_futures  = []
                max_save_pend = dng_workers * 8

                pending_magenta      = []    # [(img, orig_idx)] – frames aún sin destino decidido
                in_long_run          = False
                saved_count          = 0
                glitches_descartados = 0
                stem = input_path.stem

                def _unpack_detect(raw_data):
                    img = unpack_12bit_packed_manual(raw_data, w, h)
                    return img, es_glitch_magenta(img)

                def _enqueue_save(img, out_path):
                    save_futures.append(save_pool.submit(_save_dng_task, img, out_path, w, h))
                    if len(save_futures) >= max_save_pend:
                        done, not_done = wait(save_futures, return_when=FIRST_COMPLETED)
                        save_futures[:] = list(not_done)
                        for fut in done:
                            fut.result()

                def _commit_dng(img):
                    nonlocal saved_count
                    _enqueue_save(img, dng_folder / f"{stem}_{saved_count:06d}.dng")
                    saved_count += 1

                def _flush_to_dng():
                    for buf_img, _ in pending_magenta:
                        _commit_dng(buf_img)
                    pending_magenta.clear()

                def _flush_to_descarte():
                    nonlocal glitches_descartados
                    for buf_img, orig_idx in pending_magenta:
                        _enqueue_save(buf_img, descarte_folder / f"{stem}_{orig_idx:06d}.dng")
                    glitches_descartados += len(pending_magenta)
                    pending_magenta.clear()

                def _process_result(orig_idx, img, is_magenta):
                    nonlocal in_long_run
                    if is_magenta:
                        if in_long_run:
                            _commit_dng(img)
                        elif len(pending_magenta) < MAX_GLITCH_RUN:
                            pending_magenta.append((img, orig_idx))
                        else:
                            in_long_run = True
                            _flush_to_dng()
                            _commit_dng(img)
                    else:
                        in_long_run = False
                        if pending_magenta:
                            _flush_to_descarte()
                        _commit_dng(img)

                with open(input_path, "rb") as f:
                    read_idx      = 0
                    file_exhausted = False

                    def _fill_pipeline():
                        nonlocal read_idx, file_exhausted
                        while not file_exhausted and len(unpack_queue) < PIPELINE:
                            raw_data = f.read(frame_bytes)
                            if len(raw_data) < frame_bytes:
                                file_exhausted = True
                                break
                            unpack_queue.append(
                                (read_idx, unpack_pool.submit(_unpack_detect, raw_data))
                            )
                            read_idx += 1

                    _fill_pipeline()

                    while unpack_queue:
                        orig_idx, fut = unpack_queue.popleft()
                        img, is_magenta = fut.result()   # espera solo el frame más antiguo
                        _fill_pipeline()                 # rellena el pipeline inmediatamente
                        _process_result(orig_idx, img, is_magenta)

                        if orig_idx % 10 == 0:
                            elapsed = time.time() - start_time
                            fps_proc = (orig_idx + 1) / elapsed if elapsed > 0 else 0
                            print(f"PROG|{orig_idx+1}|{num_frames}")
                            print(f"INFO|Frame: {orig_idx+1}/{num_frames} "
                                  f"({(orig_idx+1)/num_frames*100:.1f}%) | Vel: {fps_proc:.2f} fps")
                            sys.stdout.flush()

                # Vaciar buffer trailing (≤ MAX_GLITCH_RUN al final → glitches)
                if pending_magenta:
                    _flush_to_descarte()

                if save_futures:
                    done, _ = wait(save_futures)
                    for fut in done:
                        fut.result()
                unpack_pool.shutdown(wait=True)
                save_pool.shutdown(wait=True)

                if glitches_descartados:
                    print(f"INFO|Glitches magenta descartados: {glitches_descartados} frames → {descarte_folder.name}")

            elif is_tiff_seq_mode:
                # MODO TIFF SEQUENCE (RGB procesado)
                if not HAS_TIFFFILE:
                    print("ERROR|Se requiere tifffile para exportar TIFF. Instálalo: pip install tifffile")
                    return

                with open(input_path, "rb") as f:
                    for i in range(num_frames):
                        raw_data = f.read(frame_bytes)
                        if len(raw_data) < frame_bytes:
                            break

                        if not is_rgb:
                            # Por diseño, tiff_seq se ofrece solo para fuentes RGB.
                            continue

                        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(h, w, 3)
                        frame_name = f"{input_path.stem}_{i:06d}.tif"
                        tifffile.imwrite(str(tiff_folder / frame_name), img)

                        if i % 10 == 0:
                            elapsed = time.time() - start_time
                            fps_proc = (i + 1) / elapsed if elapsed > 0 else 0
                            print(f"PROG|{i+1}|{num_frames}")
                            print(f"INFO|Frame: {i+1}/{num_frames} ({(i+1)/num_frames*100:.1f}%) | Vel: {fps_proc:.2f} fps")
                            sys.stdout.flush()

            else:
                # MODO VIDEO (NO QOI): paralelizamos el procesado previo a FFmpeg
                total_cores = multiprocessing.cpu_count()
                video_workers = max(1, total_cores - 2)
                print(f"INFO|Video multithreading: usando {video_workers} de {total_cores} cores (dejando 2 libres)")

                executor = ThreadPoolExecutor(max_workers=video_workers)
                future_map = {}
                next_to_write = 0
                max_in_flight = video_workers * 4

                def flush_ready_frames():
                    nonlocal next_to_write
                    while next_to_write in future_map and future_map[next_to_write].done():
                        data_out = future_map[next_to_write].result()
                        if process:
                            try:
                                process.stdin.write(data_out)
                            except (BrokenPipeError, OSError):
                                return False
                        del future_map[next_to_write]
                        next_to_write += 1
                    return True

                with open(input_path, "rb") as f:
                    for i in range(num_frames):
                        raw_data = f.read(frame_bytes)
                        if len(raw_data) < frame_bytes:
                            break

                        fut = executor.submit(
                            process_video_frame,
                            raw_data,
                            is_rgb,
                            w,
                            h,
                            args.mode
                        )
                        future_map[i] = fut

                        if len(future_map) >= max_in_flight:
                            wait(list(future_map.values()), return_when=FIRST_COMPLETED)
                            if not flush_ready_frames():
                                break

                        if i % 10 == 0:
                            elapsed = time.time() - start_time
                            fps_proc = (i + 1) / elapsed if elapsed > 0 else 0
                            print(f"PROG|{i+1}|{num_frames}")
                            print(f"INFO|Frame: {i+1}/{num_frames} ({(i+1)/num_frames*100:.1f}%) | Vel: {fps_proc:.2f} fps")
                            sys.stdout.flush()

                # Vaciar cualquier frame pendiente
                if future_map:
                    wait(list(future_map.values()))
                    flush_ready_frames()
                executor.shutdown(wait=True)

    except Exception as e:
        print(f"ERROR|{e}")
    finally:
        if process and process.stdin: 
            process.stdin.close()
            process.wait()
        print("INFO|Proceso finalizado.")

if __name__ == "__main__":
    main()
