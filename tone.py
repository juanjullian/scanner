"""Curva de luces por tramo. Los puntos los pone quien ajusta.

Un slider tira de un gris fijo y deja al vecino quieto. Para que la curva
no se devuelva, el tramo de en medio se aplana y el gris siguiente pega un
salto. Aquí cada punto se queda entre el anterior y el siguiente, y la
tabla sale de esos vecinos: siempre sube, sin solarizar.
"""
from __future__ import annotations

import numpy as np

_PIVOTS = np.array((0.0, 0.18, 0.40, 0.72, 1.0), dtype=np.float64)
_REACH = np.array((0.25, 0.32, 0.28, 0.28, 0.18), dtype=np.float64)
_ENDS = ((0.0, 0.0), (1.0, 1.0))
_KEYS = ("negros", "sombras", "medias", "claros", "altos")
_LUT_N = 1024
_CACHE: dict[tuple, np.ndarray] = {}


def control_points(curve) -> list[tuple[float, float]]:
    """Puntos de la curva. Sin ajuste, solo los dos extremos de la recta."""
    if isinstance(curve, dict):
        sliders = _slider_key(curve)
        if sliders is None:
            return list(_ENDS)
        ys = _slider_knots(sliders)
        return [(float(x), float(y)) for x, y in zip(_PIVOTS, ys)]
    if not isinstance(curve, list) or len(curve) < 2:
        return list(_ENDS)
    return _clean(curve)


def normalize(curve) -> list[tuple[float, float]] | None:
    """Puntos que de verdad cambian el cuadro, o nada si es la recta."""
    points = control_points(curve)
    if all(abs(x - y) <= 0.004 for x, y in points):
        return None
    return points


def key_of(curve) -> tuple | None:
    points = normalize(curve)
    if points is None:
        return None
    return tuple((round(x, 4), round(y, 4)) for x, y in points)


def active(curve) -> bool:
    return key_of(curve) is not None


def move_point(points, index: int, x: float, y: float) -> list[tuple[float, float]]:
    """Mueve un punto sin cruzar al vecino. Los extremos solo suben o bajan."""
    pts = [(float(a), float(b)) for a, b in control_points(points)]
    count = len(pts)
    index = int(index)
    if count < 2 or not (0 <= index < count):
        return pts
    if index == 0:
        high = pts[1][1]
        pts[0] = (0.0, min(max(0.0, float(y)), high))
        return pts
    if index == count - 1:
        low = pts[-2][1]
        pts[-1] = (1.0, min(1.0, max(low, float(y))))
        return pts
    left, right = pts[index - 1], pts[index + 1]
    span = right[0] - left[0]
    if span <= 0.045:
        x = (left[0] + right[0]) / 2.0
    else:
        x = min(right[0] - 0.02, max(left[0] + 0.02, float(x)))
    y = min(right[1], max(left[1], float(y)))
    pts[index] = (x, y)
    return pts


def insert_point(points, x: float, y: float, limit: int = 8) -> list[tuple[float, float]]:
    """Agrega un punto entre los vecinos. Si no cabe, devuelve los mismos."""
    pts = [(float(a), float(b)) for a, b in control_points(points)]
    if len(pts) >= limit + 2:
        return pts
    x = min(1.0, max(0.0, float(x)))
    y = min(1.0, max(0.0, float(y)))
    for index in range(1, len(pts)):
        if x >= pts[index][0]:
            continue
        left, right = pts[index - 1], pts[index]
        if right[0] - left[0] < 0.05:
            return pts
        x = min(right[0] - 0.02, max(left[0] + 0.02, x))
        y = min(right[1], max(left[1], y))
        return pts[:index] + [(x, y)] + pts[index:]
    return pts


def remove_point(points, index: int) -> list[tuple[float, float]]:
    pts = [(float(a), float(b)) for a, b in control_points(points)]
    index = int(index)
    if index <= 0 or index >= len(pts) - 1:
        return pts
    return pts[:index] + pts[index + 1:]


def lut(key: tuple) -> np.ndarray:
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    table = _build(key)
    if len(_CACHE) > 32:
        _CACHE.clear()
    _CACHE[key] = table
    return table


def samples(curve, count: int = 128) -> np.ndarray:
    """Salida de la curva en `count` pasos, para dibujarla igual que se aplica."""
    key = key_of(curve)
    grid = np.linspace(0.0, 1.0, int(count))
    if key is None:
        return grid.astype(np.float32)
    table = lut(key)
    index = np.clip(np.round(grid * (table.size - 1)).astype(int), 0, table.size - 1)
    return table[index]


def apply(image: np.ndarray, curve) -> np.ndarray:
    """Tabla sobre cada canal, en float 0–1. Sin curva, devuelve la misma imagen."""
    key = key_of(curve)
    if key is None:
        return image
    table = lut(key)
    img = np.asarray(image, dtype=np.float32)
    flat = np.clip(img, 0.0, 1.0).reshape(-1)
    mapped = np.interp(flat, np.linspace(0.0, 1.0, table.size), table).astype(np.float32)
    return mapped.reshape(img.shape)


def _slider_key(curve: dict) -> tuple | None:
    values = []
    for name in _KEYS:
        try:
            number = int(round(float(curve.get(name, 0) or 0)))
        except (TypeError, ValueError):
            number = 0
        values.append(max(-100, min(100, number)))
    if all(number == 0 for number in values):
        return None
    return tuple(values)


def _slider_knots(key: tuple) -> np.ndarray:
    move = np.asarray(key, dtype=np.float64) / 100.0
    out = _PIVOTS + move * _REACH
    out[0] = min(max(out[0], 0.0), 0.9)
    for index in range(1, len(out)):
        out[index] = min(max(out[index], out[index - 1] + 1e-4), 1.0)
    return out


def _clean(curve: list) -> list[tuple[float, float]]:
    parsed = []
    for item in curve:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue
        parsed.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    if len(parsed) < 2:
        return list(_ENDS)
    parsed.sort(key=lambda point: point[0])
    parsed[0] = (0.0, parsed[0][1])
    parsed[-1] = (1.0, parsed[-1][1])
    kept = [parsed[0]]
    for x, y in parsed[1:-1]:
        if x - kept[-1][0] < 0.015 or 1.0 - x < 0.015:
            continue
        kept.append((x, y))
    kept.append(parsed[-1])
    fixed = [kept[0]]
    for x, y in kept[1:]:
        fixed.append((x, min(1.0, max(y, fixed[-1][1]))))
    return fixed


def _end_slope(near: float, far: float, delta_near: float, delta_far: float) -> float:
    slope = ((2 * near + far) * delta_near - near * delta_far) / (near + far)
    if slope * delta_near <= 0:
        return 0.0
    if delta_near * delta_far < 0 and abs(slope) > abs(3 * delta_near):
        return 3 * delta_near
    return float(slope)


def _slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    widths = np.diff(xs)
    delta = np.diff(ys) / np.maximum(widths, 1e-6)
    slopes = np.zeros(len(xs), dtype=np.float64)
    for index in range(1, len(xs) - 1):
        left, right = delta[index - 1], delta[index]
        if left * right <= 0:
            slopes[index] = 0.0
            continue
        w1 = 2 * widths[index] + widths[index - 1]
        w2 = widths[index] + 2 * widths[index - 1]
        slopes[index] = (w1 + w2) / (w1 / left + w2 / right)
    slopes[0] = _end_slope(widths[0], widths[1], delta[0], delta[1])
    slopes[-1] = _end_slope(widths[-1], widths[-2], delta[-1], delta[-2])
    return slopes


def _build(key: tuple) -> np.ndarray:
    xs = np.array([point[0] for point in key], dtype=np.float64)
    ys = np.array([point[1] for point in key], dtype=np.float64)
    grid = np.linspace(0.0, 1.0, _LUT_N)
    if len(xs) == 2:
        span = max(float(xs[1] - xs[0]), 1e-6)
        out = ys[0] + (ys[1] - ys[0]) * np.clip((grid - xs[0]) / span, 0.0, 1.0)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    slopes = _slopes(xs, ys)
    out = np.empty(_LUT_N, dtype=np.float64)
    span = 0
    last = len(xs) - 2
    for index, sample in enumerate(grid):
        while span < last and sample > xs[span + 1]:
            span += 1
        width = max(float(xs[span + 1] - xs[span]), 1e-6)
        t = (sample - xs[span]) / width
        t2 = t * t
        t3 = t2 * t
        out[index] = (
            (2 * t3 - 3 * t2 + 1) * ys[span]
            + (t3 - 2 * t2 + t) * width * slopes[span]
            + (-2 * t3 + 3 * t2) * ys[span + 1]
            + (t3 - t2) * width * slopes[span + 1]
        )
    out = np.clip(out, 0.0, 1.0)
    out[0] = ys[0]
    out[-1] = ys[-1]
    out = np.maximum.accumulate(out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
