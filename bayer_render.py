"""Pipeline compartido: BayerRG12p empaquetado → RGB8 (como reproducción HQ)."""
import numpy as np
import cv2

RAW12_MAX = 4095

# LUT gamma 2.2 sobre 12-bit (evita np.power por píxel en cada frame).
_GAMMA_LUT = (np.power(np.arange(RAW12_MAX + 1, dtype=np.float32) / RAW12_MAX, 1.0 / 2.2) * 255.0)
_GAMMA_LUT = np.clip(_GAMMA_LUT, 0, 255).astype(np.uint8)


def unpack_12bit_le(packed_bytes, width, height):
    """Desempaqueta BayerRG12p little-endian a matriz uint16 (H, W)."""
    arr = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(-1, 3)
    b0 = arr[:, 0].astype(np.uint16)
    b1 = arr[:, 1].astype(np.uint16)
    b2 = arr[:, 2].astype(np.uint16)
    p0 = b0 | ((b1 & 0x0F) << 8)
    p1 = (b1 >> 4) | (b2 << 4)
    flat = np.empty(width * height, dtype=np.uint16)
    flat[0::2] = p0
    flat[1::2] = p1
    return flat.reshape(height, width)


def render_capture_view(packed_bytes, width, height, downscale=1, to_gray=False, to_bgr=True):
    """
    Render fiel a la reproducción HQ del visor (pausa):
    unpack completo → debayer BayerBG → gamma 2.2 (LUT).

    downscale=1 → resolución completa (~16 fps en i7 con LUT).
    downscale=2 → mitad de resolución (~29 fps), misma curva tonal.
    """
    bayer = unpack_12bit_le(packed_bytes, width, height)
    if downscale > 1:
        bayer = bayer[::downscale, ::downscale]

    rgb16 = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
    rgb8 = _GAMMA_LUT[rgb16]

    if to_gray:
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    if to_bgr:
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    return rgb8
