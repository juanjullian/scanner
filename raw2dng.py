import numpy as np
import sys
import os
import time
from pathlib import Path
import tifffile

# --- CONFIGURACIÓN ---
WIDTH = 2840
HEIGHT = 2200
FRAME_SIZE_PACKED = int(WIDTH * HEIGHT * 1.5)

# --- MATRIZ DE COLOR GENÉRICA (sRGB D65) ---
# Esto ayuda a que el DNG no se vea verde/morado al abrirlo.
# No es calibración perfecta, pero es un punto de partida decente.
COLOR_MATRIX_1 = [
    3.2404542, -1.5371385, -0.4985314,
    -0.9692660, 1.8760108, 0.0415560,
    0.0556434, -0.2040259, 1.0572252
]

def unpack_12bit_little_endian(raw_bytes, width, height):
    """
    Tu desempaquetado Little Endian (Fix H) probado.
    """
    expected_size = int(width * height * 1.5)
    if len(raw_bytes) > expected_size: raw_bytes = raw_bytes[:expected_size]
    elif len(raw_bytes) < expected_size: raw_bytes += b'\x00' * (expected_size - len(raw_bytes))

    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    b0, b1, b2 = data[:, 0].astype(np.uint16), data[:, 1].astype(np.uint16), data[:, 2].astype(np.uint16)
    
    # Lógica H
    p0 = ((b1 & 0x0F) << 8) | b0
    p1 = (b2 << 4) | ((b1 & 0xF0) >> 4)
    
    img_flat = np.empty(width * height, dtype=np.uint16)
    img_flat[0::2] = p0
    img_flat[1::2] = p1
    return img_flat.reshape(height, width)

def save_as_dng(image_array, output_path):
    """
    Guarda el array numpy (uint16) como un archivo DNG válido.
    """
    # Etiquetas DNG (Tags TIFF)
    # Definimos que es un RAW (CFA) y el patrón de Bayer.
    # CFAPattern: 0=Red, 1=Green, 2=Blue. 
    # BayerRG = [Red, Green] / [Green, Blue] -> [0, 1, 1, 2]
    
    extratags = [
        # Tag 50706: DNGVersion
        (50706, 'B', 4, [1, 4, 0, 0], True),
        # Tag 50717: WhiteLevel (12 bits = 4095)
        (50717, 'I', 1, 4095, True),
        # Tag 50718: BlackLevel (Asumimos 0 por ahora)
        (50718, 'I', 1, 0, True),
        # Tag 33422: CFAPattern (RGGB)
        (33422, 'B', 4, [0, 1, 1, 2], True), 
        # Tag 50721: ColorMatrix1
        (50721, 'd', 9, COLOR_MATRIX_1, True),
        # Tag 50710: CFAPlaneColor (R, G, B)
        (50710, 'B', 3, [0, 1, 2], True),
        # Tag 50708: UniqueCameraModel
        (50708, 's', 0, "Lucid Phoenix IMX566", True),
    ]

    tifffile.imwrite(
        output_path,
        image_array,
        photometric='cfa', # Indispensable para que lo reconozca como RAW
        planarconfig=1,
        extrasamples=None,
        tile=None, # Guardar en tiras, no tiles, para mayor compatibilidad
        extratags=extratags
    )

def main():
    print("--- CONVERSOR RAW -> DNG (Para uso con AMaZE) ---")
    
    # 1. Obtener archivo
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    else:
        files = list(Path(".").glob("*.raw"))
        if not files:
            print("No hay archivos .raw para convertir.")
            return
        target_path = files[0]

    print(f"Procesando: {target_path.name}")
    
    # 2. Desempaquetar
    with open(target_path, 'rb') as f:
        raw_data = f.read(FRAME_SIZE_PACKED)
    
    print("Desempaquetando 12-bit...")
    img_16 = unpack_12bit_little_endian(raw_data, WIDTH, HEIGHT)
    
    # 3. Guardar DNG
    dng_path = target_path.with_suffix(".dng")
    print(f"Guardando DNG: {dng_path.name}...")
    save_as_dng(img_16, dng_path)
    
    print("\n--- ¡ÉXITO! ---")
    print("Ahora puedes abrir este archivo en RawTherapee.")
    print("1. Abre RawTherapee.")
    print("2. Navega a esta carpeta.")
    print("3. Doble clic en el DNG.")
    print("4. Ve a la pestaña 'Raw' (icono mosaico) -> Demosaicing -> Method: AMaZE.")
    print("5. Haz zoom al 100% y disfruta.")

if __name__ == "__main__":
    main()