"""Revelado de visionado: Bayer 12-bit lineal → Rec.709.

El demosaico corre en el CPU (edge-aware de OpenCV). En este i7 un cuadro
de 2840×2200 tarda unos 3 ms. El mismo paso en la GPU es más rápido como
kernel, pero subir el mosaico y bajar el RGB lo deja igual o más lento.
"""
from __future__ import annotations

import sys
import threading

import cv2
import numpy as np

import bayer_render

RAW12_MAX = 4095
FULL_W = 2840
# En este OpenCV, BayerBG deja el píxel (0,0) en el rojo: es el RGGB del sensor.
_BAYER = cv2.COLOR_BayerBG2RGB_EA
_SIGMA_FINE = np.float32(1.15)
_SIGMA_COARSE = np.float32(4.0)
_LUMA_R = np.float32(0.2126)
_LUMA_G = np.float32(0.7152)
_LUMA_B = np.float32(0.0722)


def default_reveal() -> dict:
    return {
        "gain_r": 1.0,
        "gain_b": 1.0,
        "ev": 0.0,
        "shadow_b": 12.0,
        "shadow_lift": 0.0,
        "midtone": 0.0,
    }


def coerce_reveal(data) -> dict:
    out = default_reveal()
    if isinstance(data, dict):
        for key in out:
            if key in data:
                try:
                    out[key] = float(data[key])
                except (TypeError, ValueError):
                    pass
    out["gain_r"] = float(np.clip(out["gain_r"], 0.25, 4.0))
    out["gain_b"] = float(np.clip(out["gain_b"], 0.25, 4.0))
    out["ev"] = float(np.clip(out["ev"], -4.0, 4.0))
    out["shadow_b"] = float(np.clip(out["shadow_b"], 0.5, 64.0))
    out["shadow_lift"] = float(np.clip(out["shadow_lift"], 0.0, 0.35))
    out["midtone"] = float(np.clip(out["midtone"], 0.0, 2.0))
    return out


def summary_line(settings: dict) -> str:
    s = coerce_reveal(settings)
    return (
        f"Revelado XQ guardado: rojo {s['gain_r']:.2f}, azul {s['gain_b']:.2f}, "
        f"EV {s['ev']:+.2f}, sombras {s['shadow_b']:.0f}, detalle {s['midtone']:.2f}"
    )


def debayer_codes(packed: bytes, width: int, height: int) -> np.ndarray:
    """RGB uint16 del edge-aware. El código 4095 es el clip del sensor."""
    bayer = bayer_render.unpack_12bit_le(packed, width, height)
    return cv2.cvtColor(np.ascontiguousarray(bayer), _BAYER)


def debayer_dng_mosaic(mosaic: np.ndarray) -> np.ndarray:
    """Mosaico DNG de esta app → RGB uint16 del mismo edge-aware.

    El DNG guarda el código de 12 bits corrido 4 bits a la izquierda.
    """
    img = np.ascontiguousarray(mosaic)
    if img.dtype != np.uint16:
        img = img.astype(np.uint16)
    if int(img.max()) > RAW12_MAX:
        img = img >> 4
    return cv2.cvtColor(img, _BAYER)


def debayer_linear(packed: bytes, width: int, height: int) -> np.ndarray:
    """RGB lineal float32. 1.0 = código 4095."""
    rgb16 = debayer_codes(packed, width, height)
    return rgb16.astype(np.float32) * (np.float32(1.0) / np.float32(RAW12_MAX))


_LUT_LEN = 65536
_lut_cache: dict[tuple, tuple] = {}


def _luma(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * _LUMA_R + img[..., 1] * _LUMA_G + img[..., 2] * _LUMA_B


def _channel_curve(gain: float, ev_mul: float, curve_b: np.float32, inv_log: np.float32, lift: float) -> np.ndarray:
    """Curva de un canal para cada código uint16. 1.0 = código 4095."""
    code = np.arange(_LUT_LEN, dtype=np.float32)
    x = np.minimum(code, np.float32(RAW12_MAX)) * (np.float32(1.0) / np.float32(RAW12_MAX))
    x = x * np.float32(gain) * np.float32(ev_mul)
    y = np.log1p(curve_b * np.maximum(x, np.float32(0))) * inv_log
    if lift > 0.0:
        under = np.clip(np.float32(1.0) - y, 0.0, 1.0)
        y = y + np.float32(lift) * under * under
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _quantize_curve(curve: np.ndarray) -> np.ndarray:
    return np.clip(curve * np.float32(65535.0) + np.float32(0.5), 0, 65535).astype(np.uint16)


def reveal_tables(settings: dict) -> tuple[tuple, tuple]:
    """Curvas float (preview y detalle) y uint16 (exportación directa a rgb48)."""
    s = coerce_reveal(settings)
    key = (s["gain_r"], s["gain_b"], s["ev"], s["shadow_b"], s["shadow_lift"])
    cached = _lut_cache.get(key)
    if cached is not None:
        return cached
    curve_b = np.float32(max(float(s["shadow_b"]), 1e-3))
    inv_log = np.float32(1.0) / np.float32(np.log1p(curve_b))
    ev_mul = float(2.0 ** s["ev"]) if s["ev"] != 0.0 else 1.0
    lift = float(s["shadow_lift"])
    floats = (
        _channel_curve(s["gain_r"], ev_mul, curve_b, inv_log, lift),
        _channel_curve(1.0, ev_mul, curve_b, inv_log, lift),
        _channel_curve(s["gain_b"], ev_mul, curve_b, inv_log, lift),
    )
    packed = tuple(_quantize_curve(curve) for curve in floats)
    _lut_cache[key] = (floats, packed)
    return floats, packed


def reveal_luts(settings: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tres curvas float32, una por canal, indexadas por el código del demosaico."""
    return reveal_tables(settings)[0]


def _midtone(img: np.ndarray, amount: float, sigma_scale: float) -> np.ndarray:
    scale = max(float(sigma_scale), 0.15)
    y = _luma(img)
    fine = cv2.GaussianBlur(y, (0, 0), float(_SIGMA_FINE) * scale)
    coarse = cv2.GaussianBlur(y, (0, 0), float(_SIGMA_COARSE) * scale)
    y_new = y + np.float32(amount) * (fine - coarse)
    eps = np.float32(1e-4)
    ratio = np.where(y < eps, np.float32(1.0), y_new / np.maximum(y, eps))
    ratio = np.clip(ratio, 0.0, 3.0).astype(np.float32)
    return img * ratio[..., None]


def _apply_luts(codes: np.ndarray, luts: tuple[np.ndarray, np.ndarray, np.ndarray], out: np.ndarray) -> None:
    out[..., 0] = luts[0][codes[..., 0]]
    out[..., 1] = luts[1][codes[..., 1]]
    out[..., 2] = luts[2][codes[..., 2]]


def apply_reveal(lin: np.ndarray, settings: dict, sigma_scale: float = 1.0) -> np.ndarray:
    """Balance, exposición, curva y detalle. La curva es la misma tabla del export."""
    luts = reveal_luts(settings)
    work = np.asarray(lin, dtype=np.float32)
    codes = np.clip(np.rint(work * np.float32(RAW12_MAX)), 0, _LUT_LEN - 1).astype(np.uint16)
    img = np.empty(codes.shape, dtype=np.float32)
    _apply_luts(codes, luts, img)
    mid = float(coerce_reveal(settings)["midtone"])
    if mid > 1e-4:
        img = np.clip(_midtone(img, mid, sigma_scale), 0.0, 1.0)
    return img


# Rec.709, rango limitado de 10 bits. Calza con el rgb48→YUV de este FFmpeg
# dentro de un código. 879 y 899.5 salen de esa medición, no del 876/896 de libro.
_INV_RGB16 = np.float32(1.0 / 65535.0)
_CB_DIV = np.float32(1.0 / (2.0 * (1.0 - 0.0722)))
_CR_DIV = np.float32(1.0 / (2.0 * (1.0 - 0.2126)))
_Y_SCALE = np.float32(879.0)
_C_SCALE = np.float32(899.5)
_Y_BIAS = np.float32(64.0)
_C_BIAS = np.float32(512.0)
# Una matriz sobre el RGB uint16. OpenCV la aplica en un paso y suelta el GIL.
_SCALE_Y = _Y_SCALE * _INV_RGB16
_SCALE_C = _C_SCALE * _INV_RGB16
_HALF_C = np.float32(0.5) * _SCALE_C
_YUV_M = np.array(
    [
        [_LUMA_R * _SCALE_Y, _LUMA_G * _SCALE_Y, _LUMA_B * _SCALE_Y],
        [-_LUMA_R * _CB_DIV * _SCALE_C, -_LUMA_G * _CB_DIV * _SCALE_C, _HALF_C],
        [_HALF_C, -_LUMA_G * _CR_DIV * _SCALE_C, -_LUMA_B * _CR_DIV * _SCALE_C],
    ],
    dtype=np.float32,
)
_YUV_BIAS = np.array([_Y_BIAS, _C_BIAS, _C_BIAS], dtype=np.float32)


def _shown_u16(codes: np.ndarray, settings: dict, mode: str, sigma_scale: float) -> np.ndarray:
    """RGB uint16 0–65535 ya revelado. Es el mismo número que iría en el rgb48."""
    floats, packed = reveal_tables(settings)
    mid = float(coerce_reveal(settings)["midtone"])
    if mid > 1e-4:
        img = np.empty(codes.shape, dtype=np.float32)
        _apply_luts(codes, floats, img)
        img = np.clip(_midtone(img, mid, sigma_scale), 0.0, 1.0)
        if mode == "BW":
            img = to_bw(img)
        return np.clip(img * np.float32(65535.0) + np.float32(0.5), 0, 65535).astype(np.uint16)
    shown = np.empty(codes.shape, dtype=np.uint16)
    _apply_luts(codes, packed, shown)
    if mode == "BW":
        y = shown[..., 0].astype(np.float32) * _LUMA_R
        y += shown[..., 1].astype(np.float32) * _LUMA_G
        y += shown[..., 2].astype(np.float32) * _LUMA_B
        y16 = np.clip(y + np.float32(0.5), 0, 65535).astype(np.uint16)
        shown[..., 0] = y16
        shown[..., 1] = y16
        shown[..., 2] = y16
    return shown


def reveal_codes_rgb48(codes: np.ndarray, settings: dict, mode: str = "COLOR", sigma_scale: float = 1.0) -> bytes:
    """Códigos del demosaico → rgb48le. Sin detalle, una lectura de tabla por canal."""
    return np.ascontiguousarray(_shown_u16(codes, settings, mode, sigma_scale)).tobytes()


def _rgb_u16_to_yuv10_planes(shown: np.ndarray) -> np.ndarray:
    """RGB uint16 → planos Y, U, V de 10 bits, alineados al bit alto (yuv444p10msble)."""
    yuv = cv2.transform(np.ascontiguousarray(shown, dtype=np.float32), _YUV_M)
    yuv += _YUV_BIAS
    np.rint(yuv, out=yuv)
    np.clip(yuv, 0, 1023, out=yuv)
    stored = yuv.astype(np.uint16)
    stored <<= 6
    h, w, _ = stored.shape
    n = h * w
    planes = np.empty(n * 3, dtype=np.uint16)
    planes[:n] = stored[..., 0].reshape(n)
    planes[n:n * 2] = stored[..., 1].reshape(n)
    planes[n * 2:] = stored[..., 2].reshape(n)
    return planes


# La matriz del CPU son varias pasadas de memoria (~140 ms). En la 5060 Ti
# el mismo cálculo, con subida y bajada, queda en unos 6 ms. Un candado
# alcanza: el revelado sigue en paralelo y este paso es corto.
_YUV_GPU_LOCK = threading.Lock()
_YUV_GPU = None
_YUV_KERNEL = r"""
extern "C" __global__
void rgb_to_yuv10(const unsigned short* rgb, unsigned short* y, unsigned short* u, unsigned short* v, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float R = rgb[i * 3] * (1.0f / 65535.0f);
    float G = rgb[i * 3 + 1] * (1.0f / 65535.0f);
    float B = rgb[i * 3 + 2] * (1.0f / 65535.0f);
    float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    float Cb = (B - Y) * (1.0f / (2.0f * (1.0f - 0.0722f)));
    float Cr = (R - Y) * (1.0f / (2.0f * (1.0f - 0.2126f)));
    int yi = (int)rintf(Y * 879.0f + 64.0f);
    int ui = (int)rintf(Cb * 899.5f + 512.0f);
    int vi = (int)rintf(Cr * 899.5f + 512.0f);
    if (yi < 0) yi = 0; else if (yi > 1023) yi = 1023;
    if (ui < 0) ui = 0; else if (ui > 1023) ui = 1023;
    if (vi < 0) vi = 0; else if (vi > 1023) vi = 1023;
    y[i] = (unsigned short)(yi << 6);
    u[i] = (unsigned short)(ui << 6);
    v[i] = (unsigned short)(vi << 6);
}
"""


def _yuv_gpu_state():
    """Compila el kernel una vez. False si CuPy o la tarjeta no responden."""
    global _YUV_GPU
    if _YUV_GPU is not None:
        return _YUV_GPU
    try:
        import cupy as cp

        kernel = cp.RawKernel(_YUV_KERNEL, "rgb_to_yuv10")
        _YUV_GPU = {"cp": cp, "kernel": kernel, "n": 0}
    except Exception:
        _YUV_GPU = False
    return _YUV_GPU


def _rgb_u16_to_yuv10_gpu(shown: np.ndarray):
    """Planos yuv444p10msble, o None si hay que usar el CPU."""
    state = _yuv_gpu_state()
    if not state:
        return None
    rgb = np.ascontiguousarray(shown, dtype=np.uint16)
    n = rgb.shape[0] * rgb.shape[1]
    cp = state["cp"]
    try:
        with _YUV_GPU_LOCK:
            if state["n"] != n:
                state["rgb"] = cp.empty(n * 3, dtype=cp.uint16)
                state["y"] = cp.empty(n, dtype=cp.uint16)
                state["u"] = cp.empty(n, dtype=cp.uint16)
                state["v"] = cp.empty(n, dtype=cp.uint16)
                state["host"] = np.empty(n * 3, dtype=np.uint16)
                state["n"] = n
            state["rgb"].set(rgb.ravel())
            threads = 256
            blocks = (n + threads - 1) // threads
            state["kernel"](
                (blocks,),
                (threads,),
                (state["rgb"], state["y"], state["u"], state["v"], np.int32(n)),
            )
            host = state["host"]
            state["y"].get(out=host[:n])
            state["u"].get(out=host[n:n * 2])
            state["v"].get(out=host[n * 2:])
            cp.cuda.Stream.null.synchronize()
            return host.tobytes()
    except Exception:
        global _YUV_GPU
        _YUV_GPU = False
        return None


def warmup_yuv10() -> bool:
    """Compila y corre un cuadro chico. True si el export puede usar la GPU."""
    if not _yuv_gpu_state():
        return False
    dummy = np.zeros((8, 8, 3), dtype=np.uint16)
    return _rgb_u16_to_yuv10_gpu(dummy) is not None


_YUV_CPU_WARNED = False


def yuv10_from_rgb48(data: bytes, width: int, height: int) -> bytes:
    """rgb48le → yuv444p10msble. Lo llama un solo hilo, el que escribe a FFmpeg."""
    global _YUV_CPU_WARNED
    shown = np.frombuffer(data, dtype=np.uint16).reshape(height, width, 3)
    packed = _rgb_u16_to_yuv10_gpu(shown)
    if packed is None:
        if not _YUV_CPU_WARNED:
            print(
                "WARN|La tarjeta no armó el YUV. Sigue en el CPU, más lento.",
                file=sys.stderr, flush=True,
            )
            _YUV_CPU_WARNED = True
        return _rgb_u16_to_yuv10_planes(shown).tobytes()
    return packed


def to_bw(img: np.ndarray) -> np.ndarray:
    y = _luma(img)
    return np.stack((y, y, y), axis=-1)


def to_rgb48_bytes(img: np.ndarray) -> bytes:
    u16 = np.clip(img, 0.0, 1.0) * np.float32(65535.0) + np.float32(0.5)
    return np.ascontiguousarray(u16.astype(np.uint16)).tobytes()


def to_preview_u8(img: np.ndarray, max_width: int = 1100) -> np.ndarray:
    u8 = np.clip(img * np.float32(255.0) + np.float32(0.5), 0, 255).astype(np.uint8)
    height, width = u8.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        u8 = cv2.resize(u8, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(u8)
