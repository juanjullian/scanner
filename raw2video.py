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
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque

FFMPEG_STDIN_BUFSIZE = 4 * 1024 * 1024  # buffer tubería stdin → FFmpeg
RESERVE_CORES_FOR_OS = 2  # cores libres para el SO (p. ej. E-cores en i7 híbrido)
DETECT_PREFETCH_WORKERS = 2  # detección glitch: pocos hilos, el resto va al procesado
PROCESS_DEPTH_EXTRA = 12  # cuadros extra en vuelo para evitar ráfagas CPU


def _parallel_worker_count() -> tuple[int, int]:
    """Hilos de procesado: todos los cores lógicos menos la reserva para el SO."""
    total = multiprocessing.cpu_count()
    workers = max(1, total - RESERVE_CORES_FOR_OS)
    return workers, total


def _process_depth(video_workers: int) -> int:
    return max(4, video_workers + PROCESS_DEPTH_EXTRA)

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


MAX_GLITCH_RUN = 5


class MagentaGlitchFilter:
    """
    Ventana temporal anti-glitch magenta (Bayer).
    Rachas cortas (≤ MAX_GLITCH_RUN) se descartan; rachas largas se tratan como cast natural.
    """

    def __init__(self, max_glitch_run: int = MAX_GLITCH_RUN):
        self.max_glitch_run = max_glitch_run
        self.pending: list = []
        self.in_long_run = False
        self.discarded_count = 0

    def feed(self, item, is_magenta: bool) -> list:
        """Devuelve los items que deben incluirse en la salida (0, 1 o varios)."""
        out = []
        if is_magenta:
            if self.in_long_run:
                out.append(item)
            elif len(self.pending) < self.max_glitch_run:
                self.pending.append(item)
            else:
                self.in_long_run = True
                out.extend(self.pending)
                self.pending.clear()
                out.append(item)
        else:
            self.in_long_run = False
            if self.pending:
                self.discarded_count += len(self.pending)
                self.pending.clear()
            out.append(item)
        return out

    def finalize(self):
        """Descarta frames magenta pendientes al final del archivo."""
        if self.pending:
            self.discarded_count += len(self.pending)
            self.pending.clear()


def _save_dng_task(img_16bit, out_path, width, height):
    """
    Guarda un frame ya desempaquetado como DNG.
    La detección de glitch y la elección de destino ocurren en el hilo principal;
    este task solo realiza la escritura I/O en el thread pool.
    """
    save_dng_frame(img_16bit, out_path, width, height)
    return out_path.name


def _apply_bw_mode(img_rgb8, mode):
    if mode != "BW":
        return img_rgb8
    gray = cv2.cvtColor(img_rgb8, cv2.COLOR_RGB2GRAY)
    return cv2.merge((gray, gray, gray))


def _rgb8_to_export_bytes(img_rgb8, is_rgb):
    if is_rgb:
        return img_rgb8.tobytes()
    img_final_16 = (img_rgb8.astype(np.uint32) * 65535 // 255).astype(np.uint16)
    return img_final_16.tobytes()


def _finalize_export_frame(img_rgb8, grade, width, height, is_rgb, is_bayer):
    if grade and grade.get("enabled"):
        from export_grade import apply_frame_geometry_from_settings, geometry_has_changes
        if geometry_has_changes(grade, width):
            img_rgb8 = apply_frame_geometry_from_settings(img_rgb8, grade, preview_downscale=1)
    return _rgb8_to_export_bytes(img_rgb8, is_rgb or (not is_bayer))


def process_video_frame_passthrough(raw_data, is_rgb, is_bayer, width, height, mode, grade=None):
    """
    Exportación sin gradación: ISP/RGB directo o debayer base (gamma 2.2) para Bayer.
    """
    if is_rgb:
        img_rgb8 = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3).copy()
    elif is_bayer:
        import bayer_render
        img_rgb8 = bayer_render.render_capture_view(
            raw_data, width, height, downscale=1, to_bgr=False,
        )
    else:
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
        img_rgb8 = img.copy()

    img_rgb8 = _apply_bw_mode(img_rgb8, mode)
    return _finalize_export_frame(img_rgb8, grade, width, height, is_rgb, is_bayer)


def process_video_frame_graded(raw_data, is_rgb, is_bayer, width, height, mode, grade):
    """Exportación con gradación (--grade JSON)."""
    from export_grade import apply_export_grade

    if is_rgb or not is_bayer:
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
        img_f = img.astype(np.float32) / 255.0
    else:
        import bayer_render
        bayer = bayer_render.unpack_12bit_le(raw_data, width, height)
        rgb16 = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
        img_f = rgb16.astype(np.float32) / 4095.0

    img_proc = apply_export_grade(img_f, grade, mode)
    img_rgb8 = (np.clip(img_proc, 0, 1) * 255).astype(np.uint8)
    return _finalize_export_frame(img_rgb8, grade, width, height, is_rgb, is_bayer)


def process_video_frame(raw_data, is_rgb, width, height, mode, grade=None, is_bayer=False):
    """
    Procesa un frame y devuelve bytes listos para FFmpeg (rgb24 o rgb48le).
    Sin grade → passthrough ISP / debayer base. Con grade → cadena de gradación.
    """
    if grade and grade.get("enabled"):
        return process_video_frame_graded(
            raw_data, is_rgb, is_bayer, width, height, mode, grade,
        )
    return process_video_frame_passthrough(
        raw_data, is_rgb, is_bayer, width, height, mode, grade,
    )


class TempFileFrameWriter:
    """Escribe frames procesados en orden a un archivo temporal (fase 1 del pipeline)."""

    def __init__(self, path: Path, queue_size: int = 6):
        self._path = path
        self._f = open(path, "wb")
        self._q: queue.Queue = queue.Queue(maxsize=max(2, queue_size))
        self._err = None
        self._thread = threading.Thread(target=self._run, name="temp-raw-writer", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                data = self._q.get()
                if data is None:
                    break
                self._f.write(data)
        except OSError as exc:
            self._err = exc

    def submit(self, data: bytes):
        if self._err:
            raise self._err
        self._q.put(data)

    def close(self):
        self._q.put(None)
        self._thread.join()
        self._f.close()
        if self._err:
            raise self._err


def _append_ffmpeg_codec_args(cmd: list, codec: str) -> str:
    """Añade códec/salida a cmd; devuelve extensión de archivo."""
    if codec == 'prores':
        cmd += ['-c:v', 'prores_ks', '-profile:v', '4', '-vendor', 'apl0', '-qscale:v', '5', '-pix_fmt', 'yuv444p10le']
        return ".mov"
    if codec == 'prores_hq':
        cmd += ['-c:v', 'prores_ks', '-profile:v', '3', '-vendor', 'apl0', '-qscale:v', '9', '-pix_fmt', 'yuv422p10le']
        return ".mov"
    if codec == 'cineform':
        cmd += ['-c:v', 'cfhd', '-quality', '5']
        return ".mov"
    if codec == 'hevc':
        cmd += ['-c:v', 'libx265', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv444p10le', '-tag:v', 'hvc1']
        return ".mp4"
    if codec == 'h264':
        cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p']
        return ".mp4"
    if codec == 'av1':
        cmd += [
            '-c:v', 'av1_nvenc', '-preset', 'p7', '-rc', 'vbr', '-b:v', '9M',
            '-maxrate', '13M', '-bufsize', '24M', '-multipass', 'fullres',
            '-spatial-aq', '1', '-temporal-aq', '1', '-rc-lookahead', '32', '-pix_fmt', 'p010le',
        ]
        return ".mp4"
    cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p']
    return ".mp4"


def _build_ffmpeg_cmd(
    input_spec: str,
    export_w: int,
    export_h: int,
    input_pix_fmt: str,
    fps: str,
    codec: str,
    output_file: Path,
) -> list:
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{export_w}x{export_h}', '-pix_fmt', input_pix_fmt, '-r', str(fps),
        '-i', input_spec,
    ]
    _append_ffmpeg_codec_args(cmd, codec)
    cmd.append(str(output_file))
    return cmd


def _run_ffmpeg_encode(cmd: list) -> int:
    print(f"INFO|FFmpeg cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, stderr=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg fallo con codigo {result.returncode}")
    return result.returncode


def _skip_raw_frames(
    raw_file,
    skip_count: int,
    frame_bytes: int,
    w: int,
    h: int,
    glitch_filter: MagentaGlitchFilter | None,
):
    if skip_count <= 0:
        return
    print(f"INFO|Omitiendo primeros {skip_count} cuadros...")
    if glitch_filter:
        for _ in range(skip_count):
            raw_data = raw_file.read(frame_bytes)
            if len(raw_data) < frame_bytes:
                return
            raw_data, is_magenta = _unpack_detect_magenta(raw_data, w, h)
            for _chunk in glitch_filter.feed(raw_data, is_magenta):
                pass
    else:
        raw_file.seek(skip_count * frame_bytes, os.SEEK_CUR)


class FfmpegStdinWriter:
    """Hilo dedicado que alimenta FFmpeg; cola acotada aplica backpressure y limita RAM."""

    def __init__(self, proc, queue_size: int = 6):
        self._proc = proc
        self._q: queue.Queue = queue.Queue(maxsize=max(2, queue_size))
        self._err = None
        self._thread = threading.Thread(target=self._run, name="ffmpeg-stdin-writer", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                data = self._q.get()
                if data is None:
                    break
                self._proc.stdin.write(data)
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._err = exc

    def submit(self, data: bytes):
        if self._err:
            raise self._err
        self._q.put(data)

    def close(self):
        self._q.put(None)
        self._thread.join()
        if self._err:
            raise self._err


def _log_export_progress(current: int, total: int, start_time: float):
    elapsed = time.time() - start_time
    fps_proc = current / elapsed if elapsed > 0 else 0
    print(f"PROG|{current}|{total}")
    pct = (current / total * 100) if total else 0
    print(f"INFO|Frame: {current}/{total} ({pct:.1f}%) | Vel: {fps_proc:.2f} fps")
    sys.stdout.flush()


def _unpack_detect_magenta(raw_data, w, h):
    img = unpack_12bit_packed_manual(raw_data, w, h)
    return raw_data, es_glitch_magenta(img)


def _export_video_ordered(
    raw_file,
    num_frames: int,
    frame_bytes: int,
    frame_writer,
    w, h,
    is_rgb: bool,
    args,
    grade_settings,
    glitch_filter: MagentaGlitchFilter | None,
    start_time: float,
    skip_frames: int = 0,
):
    """
    Pipeline video: detección glitch (paralela, Bayer) → procesado (pool) →
    hilo escritor (cola acotada) → FFmpeg. Evita acumular decenas de frames en RAM.
    """
    video_workers, total_cores = _parallel_worker_count()
    process_depth = _process_depth(video_workers)
    detect_depth = max(8, video_workers + 4) if glitch_filter else 0
    write_queue_size = max(8, video_workers + 4)

    print(
        f"INFO|Fase 1 multihilo: {video_workers} workers procesado "
        f"({total_cores} cores, {RESERVE_CORES_FOR_OS} reservados SO), "
        f"profundidad={process_depth}, cola_escritura={write_queue_size}"
        + (f", prefetch_glitch={DETECT_PREFETCH_WORKERS}" if glitch_filter else "")
    )

    writer = frame_writer
    executor = ThreadPoolExecutor(max_workers=video_workers)
    detect_pool = (
        ThreadPoolExecutor(max_workers=DETECT_PREFETCH_WORKERS) if glitch_filter else None
    )

    future_map: dict[int, object] = {}
    next_to_write = 0
    submit_idx = 0
    read_count = 0
    detect_queue: deque = deque()

    def drain_completed():
        nonlocal next_to_write
        while next_to_write in future_map and future_map[next_to_write].done():
            writer.submit(future_map[next_to_write].result())
            del future_map[next_to_write]
            next_to_write += 1

    def submit_raw(raw_data: bytes):
        nonlocal submit_idx
        while len(future_map) >= process_depth:
            wait(list(future_map.values()), return_when=FIRST_COMPLETED)
            drain_completed()
        future_map[submit_idx] = executor.submit(
            process_video_frame,
            raw_data, is_rgb, w, h, args.mode, grade_settings, not is_rgb,
        )
        submit_idx += 1
        drain_completed()

    def fill_detect():
        nonlocal read_count
        while len(detect_queue) < detect_depth and read_count < num_frames:
            raw_data = raw_file.read(frame_bytes)
            if len(raw_data) < frame_bytes:
                break
            detect_queue.append(
                detect_pool.submit(_unpack_detect_magenta, raw_data, w, h)
            )
            read_count += 1

    try:
        _skip_raw_frames(raw_file, skip_frames, frame_bytes, w, h, glitch_filter)
        if glitch_filter:
            fill_detect()
            processed = 0
            while detect_queue:
                raw_data, is_magenta = detect_queue.popleft().result()
                fill_detect()
                processed += 1
                for chunk in glitch_filter.feed(raw_data, is_magenta):
                    submit_raw(chunk)
                drain_completed()
                if processed % 10 == 0:
                    _log_export_progress(processed, num_frames, start_time)
            glitch_filter.finalize()
        else:
            next_raw = raw_file.read(frame_bytes)
            for i in range(num_frames):
                raw_data = next_raw
                if len(raw_data) < frame_bytes:
                    break
                if i + 1 < num_frames:
                    next_raw = raw_file.read(frame_bytes)
                read_count = i + 1
                submit_raw(raw_data)
                if (i + 1) % 10 == 0:
                    _log_export_progress(i + 1, num_frames, start_time)

        while future_map:
            wait(list(future_map.values()), return_when=FIRST_COMPLETED)
            drain_completed()

        if glitch_filter and glitch_filter.discarded_count:
            print(
                f"INFO|Glitches magenta descartados del video: {glitch_filter.discarded_count} frames "
                f"(de {read_count} leídos, {next_to_write} en salida)"
            )
    finally:
        writer.close()
        if detect_pool:
            detect_pool.shutdown(wait=True)
        executor.shutdown(wait=True)


def _export_video_qoi(
    input_path,
    qoi_index,
    num_frames: int,
    frame_writer,
    w, h,
    args,
    grade_settings,
    start_time: float,
    skip_frames: int = 0,
):
    """Exportación QOI/RGB8 con procesado paralelo y hilo escritor a FFmpeg."""
    video_workers, total_cores = _parallel_worker_count()
    process_depth = _process_depth(video_workers)
    write_queue_size = max(8, video_workers + 4)
    path_str = str(input_path)

    print(
        f"INFO|Fase 1 multihilo (QOI): {video_workers} workers "
        f"({total_cores} cores, {RESERVE_CORES_FOR_OS} reservados SO), "
        f"profundidad={process_depth}, cola={write_queue_size}"
    )

    def _decode_and_process(entry):
        offset, fsz = entry
        qoi_data = qoi_utils.read_frame_at(path_str, offset, fsz)
        img = qoi_utils.decode_qoi(qoi_data, w, h)
        return process_video_frame(
            img.tobytes(), True, w, h, args.mode, grade_settings, False,
        )

    writer = frame_writer
    executor = ThreadPoolExecutor(max_workers=video_workers)
    future_map: dict[int, object] = {}
    next_to_write = 0

    def drain_completed():
        nonlocal next_to_write
        while next_to_write in future_map and future_map[next_to_write].done():
            writer.submit(future_map[next_to_write].result())
            del future_map[next_to_write]
            next_to_write += 1

    try:
        start_idx = skip_frames
        end_idx = min(len(qoi_index), start_idx + num_frames)
        export_indices = list(range(start_idx, end_idx))
        if skip_frames > 0:
            print(f"INFO|Omitiendo primeros {skip_frames} cuadros QOI...")
        for seq, idx in enumerate(export_indices):
            while len(future_map) >= process_depth:
                wait(list(future_map.values()), return_when=FIRST_COMPLETED)
                drain_completed()
            future_map[seq] = executor.submit(_decode_and_process, qoi_index[idx])
            drain_completed()
            if (seq + 1) % 10 == 0:
                _log_export_progress(seq + 1, len(export_indices), start_time)

        while future_map:
            wait(list(future_map.values()), return_when=FIRST_COMPLETED)
            drain_completed()
    finally:
        writer.close()
        executor.shutdown(wait=True)


def _create_video_frame_writer(video_ctx: dict, args):
    video_workers, _total = _parallel_worker_count()
    write_queue_size = max(8, video_workers + 4)
    if video_ctx['stage_temp']:
        return TempFileFrameWriter(video_ctx['temp_path'], queue_size=write_queue_size)
    if video_ctx['process'] is None:
        cmd = _build_ffmpeg_cmd(
            '-', video_ctx['export_w'], video_ctx['export_h'],
            video_ctx['input_pix_fmt'], args.fps, args.codec, video_ctx['output_file'],
        )
        print(f"INFO|FFmpeg cmd (stdin): {' '.join(cmd)}")
        video_ctx['process'] = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=sys.stderr, bufsize=FFMPEG_STDIN_BUFSIZE,
        )
    return FfmpegStdinWriter(video_ctx['process'], queue_size=write_queue_size)


def _finish_video_export(video_ctx: dict | None, args):
    if not video_ctx:
        return
    if video_ctx['stage_temp']:
        tp = video_ctx['temp_path']
        if tp.exists():
            size_mb = tp.stat().st_size / (1024 * 1024)
            print(f"INFO|Fase 2: codificando desde temporal ({size_mb:.1f} MiB)...")
            cmd = _build_ffmpeg_cmd(
                str(tp), video_ctx['export_w'], video_ctx['export_h'],
                video_ctx['input_pix_fmt'], args.fps, args.codec, video_ctx['output_file'],
            )
            _run_ffmpeg_encode(cmd)
            tp.unlink(missing_ok=True)
    elif video_ctx.get('process'):
        proc = video_ctx['process']
        if proc.stdin:
            proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg fallo con codigo {ret}")


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
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--codec", default="prores") # 'dng' activa el modo raw
    parser.add_argument("--fps", default="18")
    parser.add_argument("--sharp", default="0,0") 
    parser.add_argument("--mode", default="COLOR")
    parser.add_argument("--output", default=None, help="Nombre de archivo o carpeta de salida (relativo al directorio del .raw)")
    parser.add_argument("--grade", default=None, help="JSON con ajustes de gradación (export_grade.DEFAULT_GRADE)")
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cantidad maxima de cuadros a exportar (despues de --skip-frames)",
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Cuadros iniciales a omitir antes de exportar",
    )
    parser.add_argument(
        "--no-stage-temp", action="store_true",
        help="Enviar frames a FFmpeg por stdin (sin archivo temporal intermedio)",
    )
    args = parser.parse_args()

    grade_settings = None
    if args.grade:
        try:
            grade_settings = json.loads(args.grade)
            if grade_settings.get("enabled"):
                print(
                    f"INFO|Gradación activa: gamma={grade_settings.get('gamma')} "
                    f"black={grade_settings.get('black_level')} "
                    f"white={grade_settings.get('white_level')}"
                )
                from export_grade import geometry_ffmpeg_filter, geometry_has_changes, export_frame_size
                if geometry_has_changes(grade_settings, FIXED_W):
                    ew, eh = export_frame_size(grade_settings, FIXED_W, FIXED_H)
                    vf = geometry_ffmpeg_filter(grade_settings, FIXED_W, FIXED_H)
                    print(
                        f"INFO|Encuadre: rot={grade_settings.get('rotate_deg')}deg "
                        f"zoom={grade_settings.get('zoom_px')}px "
                        f"pan=({grade_settings.get('pan_x')},{grade_settings.get('pan_y')})px "
                        f"salida={ew}x{eh}px"
                    )
                    if vf:
                        print(f"INFO|Encuadre equivalente FFmpeg -vf: {vf}")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"WARN|No se pudo parsear --grade: {e}")
            grade_settings = None
    
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

    file_frame_count = num_frames
    skip = max(0, int(args.skip_frames or 0))
    if skip >= file_frame_count:
        print(f"ERROR|--skip-frames ({skip}) >= cuadros en archivo ({file_frame_count})")
        return

    remaining = file_frame_count - skip
    export_count = remaining
    if args.max_frames is not None and args.max_frames > 0:
        export_count = min(export_count, args.max_frames)

    if skip > 0 or export_count < file_frame_count:
        first = skip + 1
        last = skip + export_count
        print(
            f"INFO|Rango exportacion: cuadros {first}-{last} "
            f"({export_count} total, archivo tiene {file_frame_count})"
        )
    num_frames = export_count
    skip_frames = skip
    
    print(f"INFO|Procesando: {input_path.name}")
    
    # === MODO DNG ===
    is_dng_mode = (args.codec.lower() == 'dng')
    is_tiff_seq_mode = (args.codec.lower() == 'tiff_seq')
    video_ctx = None
    process = None
    temp_path = None
    
    if is_dng_mode:
        if is_rgb or is_qoi:
            print("ERROR|No se puede exportar DNG desde una fuente RGB/QOI. Se requiere RAW Bayer.")
            return
        
        dng_folder = input_path.parent / (args.output or f"{input_path.stem}_DNG_SEQ")
        dng_folder.mkdir(exist_ok=True)
        print(f"INFO|Modo DNG RAW activo. Salida en: {dng_folder}")
        process = None

    elif is_tiff_seq_mode:
        tiff_folder = input_path.parent / (args.output or f"{input_path.stem}_TIFF_SEQ")
        tiff_folder.mkdir(exist_ok=True)
        print(f"INFO|Modo TIFF Sequence activo (fiel al ISP). Salida en: {tiff_folder}")
        process = None
    else:
        # === MODO VIDEO (FFMPEG) ===
        is_processed_src = (is_rgb or is_qoi)
        input_pix_fmt = 'rgb24' if is_processed_src else 'rgb48le'
        export_w, export_h = w, h
        if grade_settings and grade_settings.get("enabled"):
            from export_grade import export_frame_size
            export_w, export_h = export_frame_size(grade_settings, w, h)
        print(
            f"INFO|Res captura: {w}x{h} | Salida video: {export_w}x{export_h} | "
            f"Modo Video: {args.codec} | Fuente: {'RGB8 ISP' if is_processed_src else 'Bayer'} | "
            f"pix_fmt: {input_pix_fmt}"
        )

        ext = _append_ffmpeg_codec_args([], args.codec)
        output_file = input_path.parent / (args.output or f"{input_path.stem}_{args.codec}{ext}")
        stage_temp = not args.no_stage_temp
        temp_path = input_path.parent / f".{input_path.stem}_stage_{os.getpid()}.raw"
        if stage_temp:
            bpf = export_w * export_h * (3 if input_pix_fmt == 'rgb24' else 6)
            est_mb = (bpf * num_frames) / (1024 * 1024)
            vw, tc = _parallel_worker_count()
            print(
                f"INFO|Pipeline 2 fases: fase 1 multihilo ({vw}/{tc} cores) -> "
                f"temporal (~{est_mb:.0f} MiB) -> FFmpeg"
            )
        else:
            print("INFO|Pipeline directo: procesado -> FFmpeg stdin")

        video_ctx = {
            'stage_temp': stage_temp,
            'temp_path': temp_path,
            'export_w': export_w,
            'export_h': export_h,
            'input_pix_fmt': input_pix_fmt,
            'output_file': output_file,
            'process': None,
        }

    # --- BUCLE PRINCIPAL ---
    print(f"START|{num_frames}")
    sys.stdout.flush()
    start_time = time.time()
    
    is_processed_src = (is_rgb or is_qoi)

    try:
        if is_qoi:
            if is_tiff_seq_mode:
                if not HAS_TIFFFILE:
                    print("ERROR|Se requiere tifffile para exportar TIFF. Instálalo: pip install tifffile")
                    return
                for i in range(num_frames):
                    offset, fsz = qoi_index[i]
                    qoi_data = qoi_utils.read_frame_at(str(input_path), offset, fsz)
                    img = qoi_utils.decode_qoi(qoi_data, w, h)
                    frame_name = f"{input_path.stem}_{i:06d}.tif"
                    tifffile.imwrite(str(tiff_folder / frame_name), img)
                    if i % 10 == 0:
                        _log_export_progress(i + 1, num_frames, start_time)
            elif not is_dng_mode:
                writer = _create_video_frame_writer(video_ctx, args)
                try:
                    _export_video_qoi(
                        input_path, qoi_index, num_frames, writer,
                        w, h, args, grade_settings, start_time, skip_frames,
                    )
                finally:
                    writer.close()
                _finish_video_export(video_ctx, args)
        else:
            # --- LECTURA ESTÁNDAR (Bayer / RGB plano) ---
            if is_dng_mode:
                # MODO DNG: pipeline de dos pools para mantener CPU y NVMe saturados.
                dng_workers, total_cores = _parallel_worker_count()
                PIPELINE = max(8, dng_workers * 3)
                print(f"INFO|DNG pipeline: {dng_workers} workers unpack + {dng_workers} workers escritura "
                      f"({total_cores} cores, {RESERVE_CORES_FOR_OS} reservados SO, profundidad {PIPELINE})")

                unpack_pool = ThreadPoolExecutor(max_workers=dng_workers)
                save_pool = ThreadPoolExecutor(max_workers=dng_workers)
                glitch_filter = MagentaGlitchFilter()

                unpack_queue = deque()
                save_futures = []
                max_save_pend = dng_workers * 8
                saved_count = 0
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

                def _process_result(_orig_idx, img, is_magenta):
                    for to_save in glitch_filter.feed(img, is_magenta):
                        _commit_dng(to_save)

                with open(input_path, "rb") as f:
                    read_idx = 0
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
                        img, is_magenta = fut.result()
                        _fill_pipeline()
                        _process_result(orig_idx, img, is_magenta)

                        if orig_idx % 10 == 0:
                            elapsed = time.time() - start_time
                            fps_proc = (orig_idx + 1) / elapsed if elapsed > 0 else 0
                            print(f"PROG|{orig_idx+1}|{num_frames}")
                            print(f"INFO|Frame: {orig_idx+1}/{num_frames} "
                                  f"({(orig_idx+1)/num_frames*100:.1f}%) | Vel: {fps_proc:.2f} fps")
                            sys.stdout.flush()

                glitch_filter.finalize()

                if save_futures:
                    done, _ = wait(save_futures)
                    for fut in done:
                        fut.result()
                unpack_pool.shutdown(wait=True)
                save_pool.shutdown(wait=True)

                if glitch_filter.discarded_count:
                    print(f"INFO|Glitches magenta descartados: {glitch_filter.discarded_count} frames")

            elif is_tiff_seq_mode:
                if not HAS_TIFFFILE:
                    print("ERROR|Se requiere tifffile para exportar TIFF. Instálalo: pip install tifffile")
                    return

                with open(input_path, "rb") as f:
                    for i in range(num_frames):
                        raw_data = f.read(frame_bytes)
                        if len(raw_data) < frame_bytes:
                            break

                        if not is_rgb:
                            continue

                        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(h, w, 3)
                        frame_name = f"{input_path.stem}_{i:06d}.tif"
                        tifffile.imwrite(str(tiff_folder / frame_name), img)

                        if i % 10 == 0:
                            _log_export_progress(i + 1, num_frames, start_time)

            else:
                use_glitch_filter = not is_rgb
                glitch_filter = MagentaGlitchFilter() if use_glitch_filter else None
                if use_glitch_filter:
                    print(f"INFO|Filtro glitch magenta activo (misma regla que DNG, racha <={MAX_GLITCH_RUN})")

                writer = _create_video_frame_writer(video_ctx, args)
                try:
                    with open(input_path, "rb") as f:
                        _export_video_ordered(
                            f, num_frames, frame_bytes, writer,
                            w, h, is_rgb, args, grade_settings,
                            glitch_filter, start_time, skip_frames,
                        )
                finally:
                    writer.close()
                _finish_video_export(video_ctx, args)

    except Exception as e:
        print(f"ERROR|{e}")
        if video_ctx and video_ctx.get('stage_temp'):
            tp = video_ctx['temp_path']
            if tp.exists():
                try:
                    tp.unlink()
                except OSError:
                    pass
    finally:
        print("INFO|Proceso finalizado.")

if __name__ == "__main__":
    main()
