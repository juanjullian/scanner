"""Cálculo de GamMac 1.10 sobre la imagen ya revelada.

El área es lo único que se mide. La gamma de cada canal lleva el promedio
de esa zona al canal de anclaje. El cuadro completo recibe después esa gamma.
"""
from __future__ import annotations

import numpy as np

_LOCKS = {
    0: "rojo",
    1: "verde",
    2: "azul",
    -1: "valor",
    -2: "promedio",
    -3: "mediana",
}


def measure(
    image,
    rect,
    *,
    lock: int = 1,
    scale: int = 2,
    lo_th: float = 0.0,
    hi_th: float = 0.0,
    muls=(1.0, 1.0, 1.0),
    lock_val: float = 128.0,
    rng_lim: int = 11,
    gam_max: float = 10.0,
    strength: float = 1.0,
    omin: float = 0.0,
    omax: float = 1.0,
    dark: float | None = None,
    high: float | None = None,
) -> dict:
    """`image` es float 0–1 o uint8. `rect` es (x, y, ancho, alto) de esa imagen."""
    crop = _crop_u8(image, rect)
    if crop is None:
        return _empty("El área no entra en el cuadro.")
    height, width = crop.shape[:2]
    if width < 8 or height < 8:
        return _empty("El área es muy chica.")

    mins = []
    maxs = []
    aves = []
    for channel in range(3):
        plane = crop[..., channel]
        lo, hi = _ends(plane, lo_th, hi_th)
        mins.append(lo)
        maxs.append(hi)
        aves.append(float(plane.mean()))
    ranges = [maxs[c] - mins[c] for c in range(3)]
    if all(span < rng_lim for span in ranges):
        recipe = _identity(lock, scale)
        recipe["message"] = "El área es casi un solo color. No se estimó gamma."
        recipe["limited"] = True
        return recipe

    used = int(scale)
    if used == 2 and any(span < rng_lim for span in ranges):
        used = 1
    shared_lo = min(mins)
    shared_hi = max(maxs)
    scaled = []
    for channel in range(3):
        if used == 0:
            scaled.append(aves[channel])
        elif used == 1:
            scaled.append(_stretch(aves[channel], shared_lo, shared_hi))
        else:
            scaled.append(_stretch(aves[channel], mins[channel], maxs[channel]))
    anchor = _anchor(lock, scaled, lock_val)
    gammas = []
    for channel in range(3):
        target = float(np.clip(scaled[channel] * float(muls[channel]), 1.0, 254.0))
        gammas.append(_guess(target, anchor, gam_max))
    recipe = {
        "ok": True,
        "limited": False,
        "scale": used,
        "lock": int(lock),
        "mins": [value / 255.0 for value in mins],
        "maxs": [value / 255.0 for value in maxs],
        "shared_min": shared_lo / 255.0,
        "shared_max": shared_hi / 255.0,
        "gammas": gammas,
        "message": _describe(used, lock, gammas),
    }
    recipe.update(_finish_fields(strength, omin, omax, dark, high))
    return recipe


def saturate(image: np.ndarray, amount: float) -> np.ndarray:
    """1 deja el color. 0 lo deja en gris, con la misma luminancia Rec.709."""
    sat = float(np.clip(amount, 0.0, 1.0))
    if sat >= 0.999:
        return image
    img = np.asarray(image, dtype=np.float32)
    luma = (
        img[..., 0] * np.float32(0.2126)
        + img[..., 1] * np.float32(0.7152)
        + img[..., 2] * np.float32(0.0722)
    )
    return luma[..., None] + (img - luma[..., None]) * np.float32(sat)


def apply(image: np.ndarray, recipe: dict | None) -> np.ndarray:
    """Aplica la gamma sobre una imagen float 0–1. Sin receta, la devuelve igual."""
    if not recipe or not recipe.get("ok") or recipe.get("limited"):
        return image
    img = np.asarray(image, dtype=np.float32)
    gammas = recipe["gammas"]
    used = int(recipe.get("scale", 2))
    curved = np.empty_like(img, dtype=np.float32)
    if used == 1:
        lo = float(recipe["shared_min"])
        hi = float(recipe["shared_max"])
        span = max(hi - lo, 1e-6)
        for channel in range(3):
            sample = np.clip((img[..., channel] - lo) / span, 0.0, 1.0)
            curved[..., channel] = np.power(sample, np.float32(gammas[channel]))
    elif used == 2:
        mins = recipe["mins"]
        maxs = recipe["maxs"]
        for channel in range(3):
            span = max(float(maxs[channel]) - float(mins[channel]), 1e-6)
            sample = np.clip((img[..., channel] - float(mins[channel])) / span, 0.0, 1.0)
            curved[..., channel] = np.power(sample, np.float32(gammas[channel]))
    else:
        for channel in range(3):
            sample = np.clip(img[..., channel], 0.0, 1.0)
            curved[..., channel] = np.power(sample, np.float32(gammas[channel]))
    return _mix(img, curved, recipe)


def recipe_at(segments: list, index: int) -> dict | None:
    for segment in segments:
        if int(segment["in"]) <= index <= int(segment["out"]):
            recipe = segment.get("recipe")
            if recipe and recipe.get("ok"):
                return recipe
    return None


def _crop_u8(image, rect):
    x, y, width, height = (int(v) for v in rect)
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] < 3:
        return None
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img.shape[1], x + width)
    y1 = min(img.shape[0], y + height)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = img[y0:y1, x0:x1, :3]
    if crop.dtype == np.uint8:
        return crop
    return np.clip(np.asarray(crop, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _ends(plane: np.ndarray, lo_th: float, hi_th: float) -> tuple[int, int]:
    hist = np.bincount(plane.ravel(), minlength=256)
    total = int(plane.size)
    if lo_th < 0:
        lo = 0
    else:
        need = total * float(lo_th) / 100.0
        acc = 0
        lo = 0
        for value, count in enumerate(hist):
            acc += int(count)
            if acc > need:
                lo = value
                break
    if hi_th < 0:
        hi = 255
    else:
        need = total * float(hi_th) / 100.0
        acc = 0
        hi = 255
        for value in range(255, -1, -1):
            acc += int(hist[value])
            if acc > need:
                hi = value
                break
    if hi <= lo:
        hi = min(255, lo + 1)
    return lo, hi


def _stretch(average: float, lo: float, hi: float) -> float:
    span = max(float(hi) - float(lo), 1.0)
    return float(np.clip((average - lo) * 255.0 / span, 0.0, 255.0))


def _anchor(lock: int, scaled: list[float], lock_val: float) -> float:
    if lock == 0:
        value = scaled[0]
    elif lock == 2:
        value = scaled[2]
    elif lock == -2:
        value = (scaled[0] + scaled[1] + scaled[2]) / 3.0
    elif lock == -3:
        ordered = sorted(scaled)
        value = ordered[1]
    elif lock == -1:
        value = float(lock_val)
    else:
        value = scaled[1]
    return float(np.clip(value, 1.0, 254.0))


def _guess(average: float, lock: float, gam_max: float) -> float:
    """Bipartición. Se guarda la gamma más cercana, no la última del corte."""
    lo = 1.0 / float(gam_max)
    hi = float(gam_max)
    best = 1.0
    best_err = 1e9
    for guess in (lo, hi, 1.0):
        err = abs(_render(average, guess) - lock)
        if err < best_err:
            best, best_err = guess, err
    for _ in range(24):
        mid = (lo + hi) * 0.5
        rendered = _render(average, mid)
        err = abs(rendered - lock)
        if err < best_err:
            best, best_err = mid, err
        if rendered > lock:
            lo = mid
        else:
            hi = mid
    return float(best)


# Ancho de la rampa, en códigos de 8 bits, entre "no tocar" y "gamma completa".
_FEATHER = np.float32(16.0 / 255.0)


def _finish_fields(strength, omin, omax, dark, high) -> dict:
    omin_v = float(np.clip(omin, 0.0, 1.0))
    omax_v = float(np.clip(omax, 0.0, 1.0))
    if omax_v < omin_v:
        omin_v, omax_v = omax_v, omin_v
    fields = {
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "omin": omin_v,
        "omax": omax_v,
    }
    if dark is not None:
        fields["dark"] = float(np.clip(dark, 0.0, 1.0))
    if high is not None:
        fields["high"] = float(np.clip(high, 0.0, 1.0))
    return fields


def _mix(source: np.ndarray, curved: np.ndarray, recipe: dict) -> np.ndarray:
    """Fuerza, piso/techo y protección de sombras y luces, después de la gamma."""
    omin = np.float32(recipe.get("omin", 0.0))
    omax = np.float32(recipe.get("omax", 1.0))
    graded = omin + curved * (omax - omin)
    strength = float(recipe.get("strength", 1.0))
    dark = recipe.get("dark")
    high = recipe.get("high")
    plain = (
        strength >= 0.999
        and dark is None
        and high is None
        and float(omin) <= 1e-6
        and float(omax) >= 1.0 - 1e-6
    )
    if plain:
        return graded
    weight = np.full(source.shape, np.float32(strength), dtype=np.float32)
    if dark is not None:
        weight *= np.clip((source - np.float32(dark)) / _FEATHER, 0.0, 1.0)
    if high is not None:
        weight *= np.clip((np.float32(high) - source) / _FEATHER, 0.0, 1.0)
    return source * (np.float32(1.0) - weight) + graded * weight


def _render(average: float, gamma: float) -> float:
    sample = float(np.clip(average, 0.0, 255.0)) / 255.0
    return 255.0 * (sample ** gamma)


def _identity(lock: int, scale: int) -> dict:
    return {
        "ok": True,
        "limited": True,
        "scale": int(scale),
        "lock": int(lock),
        "mins": [0.0, 0.0, 0.0],
        "maxs": [1.0, 1.0, 1.0],
        "shared_min": 0.0,
        "shared_max": 1.0,
        "gammas": [1.0, 1.0, 1.0],
        "message": "",
    }


def _empty(message: str) -> dict:
    recipe = _identity(1, 2)
    recipe["ok"] = False
    recipe["message"] = message
    return recipe


def _describe(scale: int, lock: int, gammas: list[float]) -> str:
    name = _LOCKS.get(int(lock), "verde")
    return (
        f"Escala {scale}, anclaje {name}. "
        f"Gamma {gammas[0]:.3f}, {gammas[1]:.3f}, {gammas[2]:.3f}."
    )
