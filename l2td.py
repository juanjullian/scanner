import numpy as np
import cv2
import sys
from pathlib import Path

# --- CONFIGURACIÓN ---
WIDTH = 2840
HEIGHT = 2200
FRAME_SIZE_PACKED = int(WIDTH * HEIGHT * 1.5)

def unpack_little_endian_v1(raw_bytes, width, height):
    """
    HIPÓTESIS H: Little Endian Packed
    Byte 0: P0 Bits bajos [7:0]
    Byte 1: P0 Bits altos [11:8] | P1 Bits bajos [3:0]
    Byte 2: P1 Bits altos [11:4]
    """
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    b0, b1, b2 = data[:, 0].astype(np.uint16), data[:, 1].astype(np.uint16), data[:, 2].astype(np.uint16)

    # Pixel 0: B0 es la parte baja, (B1 & 0x0F) es la parte alta
    p0 = b0 | ((b1 & 0x0F) << 8)
    
    # Pixel 1: (B1 alto) es la parte baja, B2 es la parte alta
    p1 = ((b1 & 0xF0) >> 4) | (b2 << 4)
    
    img = np.empty(width * height, dtype=np.uint16)
    img[0::2] = p0
    img[1::2] = p1
    return img.reshape(height, width)

def unpack_little_endian_v2(raw_bytes, width, height):
    """
    HIPÓTESIS I: Little Endian Packed (Variante inversa de B1)
    A veces los bits de en medio están invertidos.
    """
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    b0, b1, b2 = data[:, 0].astype(np.uint16), data[:, 1].astype(np.uint16), data[:, 2].astype(np.uint16)

    # Pixel 0: B0 + (B1 altos)
    p0 = b0 | ((b1 & 0xF0) << 4)
    
    # Pixel 1: (B1 bajos) + B2
    p1 = (b1 & 0x0F) | (b2 << 4)
    
    img = np.empty(width * height, dtype=np.uint16)
    img[0::2] = p0
    img[1::2] = p1
    return img.reshape(height, width)

def process_and_save(raw_data, name, unpack_func, bayer_code):
    try:
        img_bayer = unpack_func(raw_data, WIDTH, HEIGHT)
        img_color = cv2.cvtColor(img_bayer, bayer_code)
        img_color = img_color * 16 # Escalar 12 a 16 bits
        cv2.imwrite(name, img_color)
        print(f"Generado: {name}")
    except Exception as e:
        print(f"Error generando {name}: {e}")

def main():
    print("--- DIAGNÓSTICO FINAL (LITTLE ENDIAN) ---")
    
    files = list(Path(".").glob("*.raw"))
    if not files:
        print("No hay archivos .raw")
        return
    target_file = files[0]
    
    with open(target_file, 'rb') as f:
        raw_data = f.read(FRAME_SIZE_PACKED)
        if len(raw_data) > FRAME_SIZE_PACKED: raw_data = raw_data[:FRAME_SIZE_PACKED]

    # PRUEBA H: Little Endian + BayerRG (Muy probable si antes era ruido puro)
    process_and_save(raw_data, "Test_H_Little_RG.tif", unpack_little_endian_v1, cv2.COLOR_BayerRG2BGR)
    
    # PRUEBA I: Little Endian Alt + BayerRG
    process_and_save(raw_data, "Test_I_LittleAlt_RG.tif", unpack_little_endian_v2, cv2.COLOR_BayerRG2BGR)
    
    # PRUEBA J: ¿Quizás el Bayer es BG? (Prueba de color sobre H)
    process_and_save(raw_data, "Test_J_Little_BG.tif", unpack_little_endian_v1, cv2.COLOR_BayerBG2BGR)

    print("\nRevisa Test_H, I y J.")
    print("Si Test_H se ve bien pero oscura, es normal (12 bits).")
    print("Si se ve con ruido estático (nieve), entonces el formato no es packed.")

if __name__ == "__main__":
    main()