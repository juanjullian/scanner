"""Demosaico del fabricante sobre un BayerRG12p ya grabado.

Es la misma llamada del visor de captura en modo ISP: ImageFactory::Convert
con el entero 0, que la DLL trata como interpolación direccional. El destino
es RGB16. El sensor sigue en 12 bits: el código 4095 llega como 65520.
"""
from __future__ import annotations

import ctypes
import threading

import numpy as np

RAW12_MAX = 4095
# El entero que manda el visor con el combo en su valor de fábrica.
# En ArenaC, 0 es la interpolación direccional.
VIEWER_ALGORITHM = 0

arena_lock = threading.Lock()
_ready = False
_warned = False


def demosaic_codes(packed: bytes, width: int, height: int, algorithm: int = VIEWER_ALGORITHM) -> np.ndarray:
    """RGB uint16 con códigos 0–4095. El 4095 es el clip del sensor."""
    with arena_lock:
        shifted = _convert_rgb16(packed, width, height, int(algorithm))
    codes = (shifted.astype(np.uint32) + 8) >> 4
    np.minimum(codes, RAW12_MAX, out=codes)
    return np.ascontiguousarray(codes.astype(np.uint16))


def _convert_rgb16(packed: bytes, width: int, height: int, algorithm: int) -> np.ndarray:
    global _ready, _warned
    expected = width * height * 3 // 2
    raw = bytes(packed)
    if len(raw) < expected:
        raise ValueError("El cuadro Bayer quedó corto.")
    try:
        from arena_api.buffer import BufferFactory
        from arena_api.enums import PixelFormat
        from arena_api.system import system  # abre Arena la primera vez

        _ = system
        _ready = True
        holder = (ctypes.c_ubyte * expected).from_buffer_copy(raw[:expected])
        pointer = ctypes.cast(holder, ctypes.POINTER(ctypes.c_ubyte))
        source = BufferFactory.create(pointer, expected, width, height, int(PixelFormat.BayerRG12p))
        try:
            converted = BufferFactory.convert(source, int(PixelFormat.RGB16), algorithm)
            try:
                data = bytes(converted.data)
            finally:
                BufferFactory.destroy(converted)
        finally:
            BufferFactory.destroy(source)
    except Exception as exc:
        if not _warned:
            print(f"WARN|El demosaico del SDK no respondió ({exc}).", flush=True)
            _warned = True
        raise
    count = width * height * 3
    if len(data) < count * 2:
        raise RuntimeError("El SDK devolvió un RGB16 incompleto.")
    return np.frombuffer(data[: count * 2], dtype=np.uint16).reshape(height, width, 3).copy()
