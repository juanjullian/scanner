import numpy as np
import rawpy
import imageio
import tifffile
import io
import sys
import cv2
from pathlib import Path

# --- CONFIGURACIÓN DE TU LOOK (H_Grueso_Medio) ---
SHARP_SIGMA = 2.0
SHARP_AMOUNT = 2.5
# --------------------------------------------------

# --- SELECTOR DE MODO ---
# "COLOR": Para Ektachrome, Vision3 (sin invertir), etc.
# "BW":    Para Tri-X, Plus-X. (Elimina 100% del ruido de color del sensor).
MODE = "BW" 
# ------------------------

WIDTH = 2840
HEIGHT = 2200
FRAME_SIZE_PACKED = int(WIDTH * HEIGHT * 1.5)

def unpack_12bit_little_endian(raw_bytes, width, height):
    """ Desempaquetado Little Endian (Fix H) """
    expected_size = int(width * height * 1.5)
    if len(raw_bytes) > expected_size: raw_bytes = raw_bytes[:expected_size]
    elif len(raw_bytes) < expected_size: raw_bytes += b'\x00' * (expected_size - len(raw_bytes))
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    b0, b1, b2 = data[:, 0].astype(np.uint16), data[:, 1].astype(np.uint16), data[:, 2].astype(np.uint16)
    p0 = ((b1 & 0x0F) << 8) | b0
    p1 = (b2 << 4) | ((b1 & 0xF0) >> 4)
    img_flat = np.empty(width * height, dtype=np.uint16)
    img_flat[0::2] = p0
    img_flat[1::2] = p1
    return img_flat.reshape(height, width)

def create_in_memory_dng(image_array):
    buffer = io.BytesIO()
    extratags = [
        (50706, 'B', 4, [1, 4, 0, 0], True),
        (50717, 'I', 1, 4095, True),
        (50718, 'I', 1, 0, True),
        (33422, 'B', 4, [0, 1, 1, 2], True),
        (50710, 'B', 3, [0, 1, 2], True),
        (50708, 's', 0, "Lucid Phoenix", True),
    ]
    tifffile.imwrite(buffer, image_array, photometric='cfa', planarconfig=1, extratags=extratags)
    buffer.seek(0)
    return buffer

def process_color(img_rgb):
    """
    MODO COLOR:
    Separa Luma/Chroma. Desenfoca Chroma (para quitar ruido). Enfoca Luma.
    """
    # Pasar a Float para precisión
    img_float = img_rgb.astype(np.float32) / 65535.0
    
    # 1. Convertir a LAB
    lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    
    # 2. Chroma Denoise (Blur en canales de color)
    # Esto elimina el 'confetti' RGB del sensor
    a = cv2.GaussianBlur(a, (0, 0), 3.0) 
    b = cv2.GaussianBlur(b, (0, 0), 3.0)
    
    # 3. Luma Sharpen (Enfoque en brillo)
    gaussian = cv2.GaussianBlur(l, (0, 0), SHARP_SIGMA)
    detail = l - gaussian
    l_sharp = l + (detail * SHARP_AMOUNT)
    
    # 4. Recomponer
    lab_clean = cv2.merge((l_sharp, a, b))
    rgb_clean = cv2.cvtColor(lab_clean, cv2.COLOR_Lab2RGB)
    
    return np.clip(rgb_clean * 65535, 0, 65535).astype(np.uint16)

def process_bw(img_rgb):
    """
    MODO BLANCO Y NEGRO:
    Convierte a gris puro (eliminando 100% ruido de color).
    Aplica Sharpening sobre el canal gris.
    Devuelve un RGB gris para compatibilidad.
    """
    # 1. Convertir a Gris (Luminancia pura)
    # Usamos los coeficientes de peso visual correctos (Rec.709)
    img_float = img_rgb.astype(np.float32) / 65535.0
    gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
    
    # 2. Sharpening directo
    gaussian = cv2.GaussianBlur(gray, (0, 0), SHARP_SIGMA)
    detail = gray - gaussian
    gray_sharp = gray + (detail * SHARP_AMOUNT)
    
    # 3. Volver a "RGB" (Gris repetido 3 veces)
    # Esto es para que el archivo TIFF sea compatible con todos los editores
    # y para mantener consistencia con el modo Color.
    gray_sharp = np.clip(gray_sharp, 0, 1.0)
    rgb_bw = cv2.merge((gray_sharp, gray_sharp, gray_sharp))
    
    return (rgb_bw * 65535).astype(np.uint16)

def main():
    print(f"--- REVELADOR UNIVERSAL (Modo: {MODE}) ---")
    
    # 1. Buscar RAW
    files = list(Path(".").glob("*.raw"))
    if not files: 
        print("No hay archivos .raw")
        return
    target = files[0]
    print(f"Procesando: {target.name}")
    
    out_dir = target.parent / "_universal_test"
    out_dir.mkdir(exist_ok=True)
    
    # 2. Desempaquetar
    with open(target, 'rb') as f:
        raw_data = f.read(FRAME_SIZE_PACKED)
    img_raw = unpack_12bit_little_endian(raw_data, WIDTH, HEIGHT)
    
    # 3. Revelado Base (DCB)
    print("1. Revelando RAW con DCB...")
    dng_mem = create_in_memory_dng(img_raw)
    with rawpy.imread(dng_mem) as raw:
        base_rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DCB,
            use_camera_wb=True,
            no_auto_bright=True,
            bright=1.1, # Un toque de brillo suele ayudar a los positivos
            user_sat=None,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16
        )

    # 4. Procesamiento según Modo
    if MODE == "COLOR":
        print("2. Aplicando limpieza de Color y Sharpening...")
        final_img = process_color(base_rgb)
        suffix = "Color"
    elif MODE == "BW":
        print("2. Aplicando conversión B&W pura y Sharpening...")
        final_img = process_bw(base_rgb)
        suffix = "BW"
    else:
        print("Modo desconocido.")
        return

    # 5. Guardar
    out_name = f"{target.stem}_{suffix}.tif"
    imageio.imwrite(out_dir / out_name, final_img)
    
    print(f"\n¡Listo! Guardado en: {out_dir / out_name}")

if __name__ == "__main__":
    main()