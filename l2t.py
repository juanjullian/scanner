import numpy as np
import imageio
import sys
import os
import json
import cv2
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count, freeze_support

# --- CONFIGURACIÓN DE RESOLUCIONES ---
FORMAT_ROIS_LOOKUP = {
    "Super 8":       (2840, 2080), 
    "Regular 8mm":   (2840, 2136), 
    "16mm (Mudo)":   (2840, 2072), 
    "16mm (Sonido)": (2840, 1700), 
    "Full Sensor":   (2840, 2840) 
}

def get_format_params(file_path):
    """
    Calcula:
    1. Dimensions (w, h)
    2. MATH_SIZE: Lo que pesan los píxeles reales (sin basura).
    3. REAL_STRIDE: Cuánto espacio ocupa cada frame en el disco (con basura).
    """
    fsize = file_path.stat().st_size
    name = file_path.name
    
    # Defaults
    w, h = 2840, 2200
    is_rgb = False
    
    # 1. Intentar Metadata (Siempre es lo más seguro)
    meta_path = file_path.parent / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
                info = data.get(name, {})
                roi = info.get("roi_key")
                if roi in FORMAT_ROIS_LOOKUP: w, h = FORMAT_ROIS_LOOKUP[roi]
                if info.get("pixel_format") == "rgb": is_rgb = True
        except: pass

    # 2. Calcular Tamaños Matemáticos
    if is_rgb:
        math_size = int(w * h * 3)
    else:
        math_size = int(w * h * 1.5)
        
    # 3. Calcular Stride Real (Bytes por frame en disco)
    # Dividimos el tamaño del archivo por el tamaño matemático para estimar frames
    approx_frames = max(1, round(fsize / math_size))
    
    # El stride real es el tamaño total dividido por la cantidad de frames
    # Esto absorbe cualquier padding oculto al final de cada frame
    real_stride = fsize // approx_frames
    
    return w, h, math_size, real_stride, approx_frames, is_rgb

def unpack_12bit_packed_manual(raw_bytes, width, height):
    """ Desempaqueta Bayer 12-bit Packed a 16-bit """
    # Aseguramos que tenemos EXACTAMENTE los bytes necesarios, ni uno más ni uno menos.
    expected_size = int(width * height * 1.5)
    
    # Recorte o Relleno de seguridad
    if len(raw_bytes) > expected_size:
        raw_bytes = raw_bytes[:expected_size]
    elif len(raw_bytes) < expected_size:
        raw_bytes += b'\x00' * (expected_size - len(raw_bytes))

    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    triplets = data.reshape(-1, 3)
    
    b0 = triplets[:, 0].astype(np.uint16)
    b1 = triplets[:, 1].astype(np.uint16)
    b2 = triplets[:, 2].astype(np.uint16)
    
    p0 = b0 | ((b1 & 0x0F) << 8)
    p1 = (b1 >> 4) | (b2 << 4)
    
    img_flat = np.empty(width * height, dtype=np.uint16)
    img_flat[0::2] = p0
    img_flat[1::2] = p1
    
    return img_flat.reshape(height, width)

def process_frame_task(args):
    # args: f_path, offset, math_size, w, h, out_path, mode, sharp, is_rgb
    f_path, offset, math_size, w, h, out_path, mode, sharp_profile, is_rgb = args
    
    try:
        with open(f_path, 'rb') as f:
            f.seek(offset)
            # LEEMOS SOLO LA DATA ÚTIL (MATH_SIZE), IGNORANDO EL PADDING DEL DISCO
            raw_data = f.read(math_size)
            
        if len(raw_data) == 0: return "Vacío"

        # --- PIPELINE DE COLOR ---
        if is_rgb:
            rgb = np.frombuffer(raw_data, dtype=np.uint8).reshape(h, w, 3)
            img_f = rgb.astype(np.float32) / 255.0
        else:
            # 1. Desempaquetar
            img_raw = unpack_12bit_packed_manual(raw_data, w, h)
            
            # 2. Debayering (FIJO A BG - ÍNDICE 1)
            # Usamos EdgeAware si está disponible, sino el normal
            rgb16 = cv2.cvtColor(img_raw, cv2.COLOR_BayerBG2RGB)
            
            # 3. Normalizar (0-1) y Gamma Correct
            img_f = rgb16.astype(np.float32) / 4095.0
            img_f = np.power(img_f, 1.0/2.2) # Gamma Monitor Standard

        # --- PROCESADO ---
        try: sigma, amount = map(float, sharp_profile.split(','))
        except: sigma, amount = 0, 0

        if mode == "BW":
            if len(img_f.shape) == 3: gray = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
            else: gray = img_f 
            if sigma > 0:
                blur = cv2.GaussianBlur(gray, (0,0), sigma)
                gray = gray + (gray - blur) * amount
            gray = np.clip(gray, 0, 1)
            final_img = cv2.merge((gray, gray, gray))
        else:
            lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
            l, a, b = cv2.split(lab)
            if sigma > 0:
                blur = cv2.GaussianBlur(l, (0,0), sigma)
                l = l + (l - blur) * amount
            final_img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_Lab2RGB)

        # 4. Guardar
        imageio.imwrite(out_path, (np.clip(final_img, 0, 1) * 65535).astype(np.uint16))
        return None
        
    except Exception as e:
        return f"Err: {e}"

def main():
    freeze_support()
    sys.stdout.reconfigure(line_buffering=True)
    print("INFO|Iniciando motor L2T...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--sharp", default="2.0,2.5")
    args = parser.parse_args()
    
    target_path = Path(args.input).resolve()
    
    if target_path.is_file():
        files = [target_path]
        base_dir = target_path.parent
    else:
        files = sorted(list(target_path.glob("*.raw")))
        base_dir = target_path
    
    if not files: print("ERROR|No files"); return

    # Config Global
    mode_arg = args.mode if args.mode else "COLOR"
    meta_data = {}
    try:
        with open(base_dir / "metadata.json", 'r') as f: meta_data = json.load(f)
    except: pass
    
    tasks = []
    print(f"INFO|Analizando {len(files)} archivos...")
    
    for f in files:
        # Detectar parámetros críticos
        w, h, math_size, real_stride, count, is_rgb = get_format_params(f)
        
        # Detectar modo BW si aplica
        mode_use = mode_arg
        if not args.mode and f.name in meta_data:
            if "Blanco" in meta_data[f.name].get("type", ""): mode_use = "BW"

        raw_folder = base_dir / f.stem
        raw_folder.mkdir(exist_ok=True)
        
        for i in range(count):
            # LA CLAVE: 
            # Saltamos usando 'real_stride' (incluye basura)
            # Pero le decimos al worker que lea solo 'math_size'
            offset = i * real_stride 
            
            out_name = f"{f.stem}_{i:06d}.tif" if count > 1 else f"{f.stem}.tif"
            out_path = raw_folder / out_name
            tasks.append((f, offset, math_size, w, h, out_path, mode_use, args.sharp, is_rgb))

    print(f"START|{len(tasks)}")
    sys.stdout.flush()
    
    with Pool(max(1, cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(process_frame_task, tasks)):
            if res: print(f"ERROR|{res}")
            print(f"PROG|{i+1}")
            sys.stdout.flush()
    print("INFO|Listo.")

if __name__ == "__main__":
    main()