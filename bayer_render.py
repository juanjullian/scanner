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


def render_linear_from_shifted(bayer_u16):
    """RGB lineal float32 desde el mosaico uint16 ya desplazado << 4.

    1.0 equivale al código 4095 del sensor. El debayer es el bilineal BayerBG
    del visor HQ, así la prueba se compara con el mismo revelado de base.
    """
    rgb = cv2.cvtColor(np.ascontiguousarray(bayer_u16), cv2.COLOR_BayerBG2RGB)
    full_scale = np.float32(RAW12_MAX << 4)  # 4095 << 4 = 65520
    return rgb.astype(np.float32) * (np.float32(1.0) / full_scale)


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
    # El debayer Bayer pide ancho y alto pares. Un salto que deja un lado impar
    # (8 sobre 2840×2200 → 355×275) hace que OpenCV pierda el color o rechace el cuadro.
    even_h = bayer.shape[0] - (bayer.shape[0] % 2)
    even_w = bayer.shape[1] - (bayer.shape[1] % 2)
    if even_h < 2 or even_w < 2:
        raise ValueError("El mosaico quedó demasiado chico para el debayer")
    bayer = np.ascontiguousarray(bayer[:even_h, :even_w])

    rgb16 = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
    rgb8 = _GAMMA_LUT[rgb16]

    if to_gray:
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    if to_bgr:
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    return rgb8


def render_thumb_rgb(packed_bytes, width, height, out_w=160, out_h=120):
    """Miniatura a color: debayer del visor HQ sobre un recorte par, y después se achica.

    El origen del recorte es par para no correr la fase del mosaico Bayer.
    """
    bayer = unpack_12bit_le(packed_bytes, width, height)
    crop_w = min(width, 720)
    crop_h = min(height, 560)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    x0 = max(0, ((width - crop_w) // 2) & ~1)
    y0 = max(0, ((height - crop_h) // 2) & ~1)
    tile = np.ascontiguousarray(bayer[y0:y0 + crop_h, x0:x0 + crop_w])
    rgb16 = cv2.cvtColor(tile, cv2.COLOR_BayerBG2RGB)
    rgb8 = _GAMMA_LUT[rgb16]
    return cv2.resize(rgb8, (out_w, out_h), interpolation=cv2.INTER_AREA)
