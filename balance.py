"""Balance de blancos y temperatura, antes del GamMac.

El blanco se mide en un rectángulo: la luz de la perforación. Las ganancias
dejan esa zona en gris, con la misma luminancia. 6500 K no tiñe. Un kelvin
más bajo entibia y uno más alto enfría, después de ese blanco.
"""
from __future__ import annotations

import numpy as np

_LUMA = np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)


def coerce(data) -> dict:
    """Lo guardado, o un balance apagado. Un archivo viejo no trae la clave."""
    out = {"on": False, "rect": None, "gains": [1.0, 1.0, 1.0]}
    if not isinstance(data, dict):
        return out
    rect = data.get("rect")
    if isinstance(rect, (list, tuple)) and len(rect) == 4:
        try:
            out["rect"] = [int(v) for v in rect]
        except (TypeError, ValueError):
            out["rect"] = None
    gains = data.get("gains")
    if isinstance(gains, (list, tuple)) and len(gains) == 3:
        parsed = []
        try:
            for value in gains:
                parsed.append(float(np.clip(float(value), 0.05, 8.0)))
        except (TypeError, ValueError):
            parsed = []
        if len(parsed) == 3:
            out["gains"] = parsed
    if "on" in data:
        out["on"] = bool(data["on"])
    return out


def gains_of(data) -> tuple | None:
    """Ganancias que de verdad mueven el cuadro. Apagado, o en 1, no hace nada."""
    wb = coerce(data)
    if not wb["on"]:
        return None
    gains = tuple(float(v) for v in wb["gains"])
    if all(abs(v - 1.0) <= 0.003 for v in gains):
        return None
    return gains


def measure(image, rect) -> dict:
    """Ganancias para que el promedio del rectángulo quede gris, con su misma luz."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        return {"ok": False, "message": "El cuadro no tiene color."}
    try:
        x, y, width, height = (int(v) for v in rect)
    except (TypeError, ValueError):
        return {"ok": False, "message": "El rectángulo no sirve."}
    limit_h, limit_w = img.shape[:2]
    x = max(0, min(x, limit_w - 1))
    y = max(0, min(y, limit_h - 1))
    width = max(0, min(width, limit_w - x))
    height = max(0, min(height, limit_h - y))
    if width < 4 or height < 4:
        return {"ok": False, "message": "El rectángulo es muy chico."}
    crop = img[y:y + height, x:x + width, :3]
    means = crop.reshape(-1, 3).mean(axis=0).astype(np.float64)
    luma = float(means @ _LUMA)
    if luma < 0.02:
        return {
            "ok": False,
            "message": "Esa zona está muy oscura. Márcala sobre la luz de la perforación.",
        }
    gains = []
    for value in means:
        if float(value) < 1e-4:
            return {
                "ok": False,
                "message": "A esa luz le falta un canal. Márcala de nuevo.",
            }
        gains.append(float(np.clip(luma / float(value), 0.05, 8.0)))
    return {
        "ok": True,
        "on": True,
        "rect": [x, y, width, height],
        "gains": gains,
        "message": (
            f"Blanco {width}×{height} en {x}, {y}. "
            f"Ganancias {gains[0]:.2f}, {gains[1]:.2f}, {gains[2]:.2f}."
        ),
    }


def _kelvin_rgb(kelvin: float) -> np.ndarray:
    """Color de un cuerpo negro, de Tanner Helland. Sirve para teñir, no para medir."""
    temp = max(float(kelvin), 1000.0) / 100.0
    if temp <= 66.0:
        red = 255.0
        green = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
    if temp >= 66.0:
        blue = 255.0
    elif temp <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * np.log(temp - 10.0) - 305.0447927307
    return np.clip(np.array((red, green, blue), dtype=np.float64), 0.0, 255.0) / 255.0


def kelvin_gains(kelvin) -> tuple | None:
    """Tinte respecto de 6500 K, sin cambiar la luminancia del blanco. 6500 no hace nada."""
    try:
        value = float(kelvin)
    except (TypeError, ValueError):
        return None
    if abs(value - 6500.0) < 15.0:
        return None
    value = float(np.clip(value, 2000.0, 12000.0))
    gains = _kelvin_rgb(value) / np.maximum(_kelvin_rgb(6500.0), 1e-6)
    luma = float(gains @ _LUMA)
    if luma < 1e-6:
        return None
    gains = np.clip(gains / luma, 0.05, 8.0)
    return tuple(float(v) for v in gains)


def apply(image: np.ndarray, gains) -> np.ndarray:
    """Multiplica cada canal. Sin ganancias, devuelve la misma imagen."""
    if not gains:
        return image
    scale = np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    if float(np.max(np.abs(scale - 1.0))) <= 0.003:
        return image
    img = np.asarray(image, dtype=np.float32)
    return np.clip(img * scale, 0.0, 1.0)


def before_gammac(image: np.ndarray, wb_gains=None, kelvin_gains=None) -> np.ndarray:
    """Primero el blanco medido, después la temperatura. Recién ahí va el GamMac."""
    return apply(apply(image, wb_gains), kelvin_gains)
