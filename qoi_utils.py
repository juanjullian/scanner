"""
Utilidades para leer/decodificar frames QOI del contenedor .raw generado por la app de escaneo.

Formato del contenedor:
  [4 bytes LE: tamaño_frame_N][tamaño_frame_N bytes: datos QOI]
  [4 bytes LE: tamaño_frame_N+1][tamaño_frame_N+1 bytes: datos QOI]
  ...

El dato QOI viene tal cual del buffer de la cámara (Arena SDK PixelFormat.QOI_RGB8).
"""

import struct
import numpy as np

# Intentar importar la librería C rápida de QOI
try:
    import qoi as _qoi_lib
    HAS_QOI_LIB = True
except ImportError:
    HAS_QOI_LIB = False


def build_frame_index(filepath):
    """
    Escanea un archivo .raw con contenedor QOI y devuelve lista de (offset, frame_size).
    Cada entrada indica dónde empieza el header de 4 bytes y cuántos bytes tiene el frame.
    """
    index = []
    with open(filepath, "rb") as f:
        while True:
            pos = f.tell()
            header = f.read(4)
            if len(header) < 4:
                break
            frame_size = struct.unpack('<I', header)[0]
            if frame_size == 0 or frame_size > 100_000_000:
                print(f"QOI_UTILS: Frame size sospechoso ({frame_size}) en offset {pos}. Deteniendo indexación.")
                break
            index.append((pos, frame_size))
            f.seek(frame_size, 1)
    return index


def read_frame_at(filepath, offset, frame_size):
    """Lee los bytes QOI de un frame dado su offset (al header de 4 bytes) y tamaño."""
    with open(filepath, "rb") as f:
        f.seek(offset + 4)  # Saltar el header de 4 bytes
        return f.read(frame_size)


def decode_qoi(data, width=None, height=None):
    """
    Decodifica bytes QOI a un numpy array RGB uint8 de shape (H, W, 3).
    Usa la librería C si está disponible, si no, decodificador Python puro.
    """
    if HAS_QOI_LIB:
        try:
            arr = _qoi_lib.decode(data)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return arr
        except Exception as e:
            print(f"QOI_UTILS: Lib qoi falló ({e}), intentando decoder Python...")

    return _decode_qoi_python(data, width, height)


def _decode_qoi_python(data, expected_w=None, expected_h=None):
    """
    Decodificador QOI puro en Python (spec: https://qoiformat.org/qoi-specification.pdf).
    Lento pero sin dependencias externas.
    """
    if len(data) < 14:
        raise ValueError(f"QOI data demasiado corta: {len(data)} bytes")

    magic = data[0:4]
    has_header = (magic == b'qoif')

    if has_header:
        width = struct.unpack('>I', data[4:8])[0]
        height = struct.unpack('>I', data[8:12])[0]
        channels = data[12]
        p = 14
    else:
        if expected_w is None or expected_h is None:
            raise ValueError("QOI data sin header 'qoif' y no se proporcionaron width/height")
        width, height = expected_w, expected_h
        channels = 3
        p = 0

    px_len = width * height
    pixels = np.zeros((px_len, 3), dtype=np.uint8)
    index = np.zeros((64, 4), dtype=np.uint8)

    r, g, b, a = np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(255)
    run = 0
    px_pos = 0
    data_len = len(data) - (8 if has_header else 0)

    while px_pos < px_len and p < data_len:
        if run > 0:
            run -= 1
        else:
            b1 = data[p]; p += 1

            if b1 == 0xfe:  # QOI_OP_RGB
                r = data[p]; g = data[p+1]; b = data[p+2]
                p += 3
            elif b1 == 0xff:  # QOI_OP_RGBA
                r = data[p]; g = data[p+1]; b = data[p+2]; a = data[p+3]
                p += 4
            elif (b1 & 0xc0) == 0x00:  # QOI_OP_INDEX
                idx = b1 & 0x3f
                r, g, b, a = index[idx]
            elif (b1 & 0xc0) == 0x40:  # QOI_OP_DIFF
                r = (r + ((b1 >> 4) & 0x03) - 2) & 0xff
                g = (g + ((b1 >> 2) & 0x03) - 2) & 0xff
                b = (b + (b1 & 0x03) - 2) & 0xff
            elif (b1 & 0xc0) == 0x80:  # QOI_OP_LUMA
                b2 = data[p]; p += 1
                vg = (b1 & 0x3f) - 32
                r = (r + vg - 8 + ((b2 >> 4) & 0x0f)) & 0xff
                g = (g + vg) & 0xff
                b = (b + vg - 8 + (b2 & 0x0f)) & 0xff
            elif (b1 & 0xc0) == 0xc0:  # QOI_OP_RUN
                run = (b1 & 0x3f)

            hash_idx = (int(r) * 3 + int(g) * 5 + int(b) * 7 + int(a) * 11) % 64
            index[hash_idx] = [r, g, b, a]

        pixels[px_pos] = [r, g, b]
        px_pos += 1

    return pixels.reshape(height, width, 3)
