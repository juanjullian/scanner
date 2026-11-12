"""Ajustes de un archivo: revelado, tramos de GamMac, perforación y recorte.

El revelado es de cada raw. Si ese archivo no tiene uno, la exportación usa
el revelado general. Esto se guarda al lado del raw.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np

import gammac

FULL_W = 2840
FULL_H = 2200


def adjust_path(raw: Path) -> Path:
    return Path(raw).with_name(Path(raw).name + ".adjust.json")


def perfo_path(raw: Path) -> Path:
    return Path(raw).with_name(Path(raw).name + ".perfo.json")


def empty() -> dict:
    return {
        "segments": [],
        "crop": None,
        "perfo": {"enabled": False, "roi": None, "ref": None, "horizontal": True},
        "grade": {"mode": "color", "sat": 1.0},
        "turn": 0.0,
        "flip_h": False,
        "crop_spans": [],
        "reveal": None,
    }


def _turn(data) -> float:
    try:
        value = float(data.get("turn", 0.0)) if isinstance(data, dict) else 0.0
    except (TypeError, ValueError):
        value = 0.0
    return max(-180.0, min(180.0, round(value, 1)))


def mirror_x(rect, width: int):
    """El mismo recuadro, visto en la imagen volteada en horizontal."""
    if not rect or len(rect) != 4:
        return None
    x, y, rect_w, rect_h = (int(v) for v in rect)
    return int(width) - x - rect_w, y, rect_w, rect_h


def sampling_matrix(tx: float, ty: float, degrees: float, center: tuple[float, float]) -> np.ndarray:
    """De un píxel de salida al píxel de origen. Grados positivos giran a la derecha."""
    if abs(degrees) < 0.05:
        return np.array([[1, 0, tx], [0, 1, ty]], np.float32)
    forward = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), -float(degrees), 1.0)
    inverse = cv2.invertAffineTransform(forward)
    inverse[0, 2] += float(tx)
    inverse[1, 2] += float(ty)
    return inverse.astype(np.float32)


def _grade(data) -> dict:
    grade = data.get("grade") if isinstance(data, dict) else None
    mode = "color"
    sat = 1.0
    if isinstance(grade, dict):
        if grade.get("mode") == "bw":
            mode = "bw"
        try:
            sat = float(grade.get("sat", 0.0 if mode == "bw" else 1.0))
        except (TypeError, ValueError):
            sat = 0.0 if mode == "bw" else 1.0
    return {"mode": mode, "sat": min(1.0, max(0.0, sat))}


def load(raw: Path) -> dict:
    path = adjust_path(raw)
    if not path.is_file():
        return empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty()
    base = empty()
    if isinstance(data, dict):
        base["segments"] = list(data.get("segments") or [])
        base["crop"] = data.get("crop")
        perfo = data.get("perfo") or {}
        base["perfo"] = {
            "enabled": bool(perfo.get("enabled")),
            "roi": perfo.get("roi"),
            "ref": perfo_ref(perfo),
            "horizontal": perfo_horizontal(perfo),
        }
        base["grade"] = _grade(data)
        base["turn"] = _turn(data)
        base["flip_h"] = bool(data.get("flip_h"))
        spans = data.get("crop_spans") or []
        base["crop_spans"] = [item for item in spans if isinstance(item, dict)]
        reveal = data.get("reveal")
        base["reveal"] = dict(reveal) if isinstance(reveal, dict) else None
    return base


def save(raw: Path, data: dict) -> None:
    path = adjust_path(raw)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def has_adjust(data: dict) -> bool:
    """Hay revelado, color, recorte o zona de perforación para copiar."""
    if data.get("crop"):
        return True
    if (data.get("perfo") or {}).get("roi"):
        return True
    if abs(_turn(data)) >= 0.05:
        return True
    if data.get("flip_h"):
        return True
    for span in data.get("crop_spans") or []:
        if isinstance(span, dict) and (span.get("crop") or span.get("flip_h")):
            return True
    if isinstance(data.get("reveal"), dict):
        return True
    return bool(data.get("segments"))


def file_reveal(raw: Path, fallback=None) -> dict:
    """Revelado de este archivo. Si no tiene, el general que se pasa como respaldo."""
    from reveal_pipeline import coerce_reveal
    saved = load(raw).get("reveal")
    return coerce_reveal(saved if isinstance(saved, dict) else fallback)


def copy_adjust(src: Path, dst: Path) -> bool:
    """Copia revelado, color, recorte y zona. No copia el desplazamiento medido.

    Si el origen no tiene revelado propio, el del destino se queda.
    """
    data = load(src)
    if not has_adjust(data):
        return False
    if not isinstance(data.get("reveal"), dict):
        data["reveal"] = load(dst).get("reveal")
    save(dst, json.loads(json.dumps(data)))
    return True


def review_state(raw: Path, frames: int | None = None) -> dict:
    """Qué tiene el archivo para darse por revisado, y en qué va la estabilización."""
    data = load(raw)
    side = read_perfo(raw) or {}
    grade = _grade(data)
    usable = [
        item for item in data["segments"]
        if (item.get("recipe") or {}).get("ok") and not (item.get("recipe") or {}).get("limited")
    ]
    if grade["mode"] == "bw":
        color, color_kind = True, "bw"
    elif usable:
        color, color_kind = True, "gammac"
    else:
        color, color_kind = False, ""
    roi = (data.get("perfo") or {}).get("roi")
    status = side.get("status")
    if roi and perfo_is_done(raw, roi, frames):
        stab = "listo"
    elif status == "running":
        stab = "corriendo"
    elif roi:
        stab = "por_correr"
    else:
        stab = "sin_zona"
    percent = None
    message = str(side.get("message") or "")
    if "/" in message:
        left, right = message.split("/", 1)
        try:
            done = int(left.strip())
            total = int(right.strip())
            if total > 0:
                percent = max(0, min(100, int(round(100 * done / total))))
        except ValueError:
            percent = None
    return {
        "crop": bool(data.get("crop")),
        "color": color,
        "color_kind": color_kind,
        "roi": bool(roi),
        "stab": stab,
        "percent": percent,
        "reviewed": bool(data.get("crop") and color and roi),
    }


def review_badges(state: dict) -> list[tuple[str, str, str]]:
    """Texto corto, tono (ok, miss, wait, run) y la explicación del chip."""
    if state.get("reviewed"):
        badges = [("Revisado", "ok", "Tiene recorte, color y recuadro.")]
    else:
        badges = [("Pendiente", "miss", "Falta recorte, color o recuadro.")]
    if state.get("crop"):
        badges.append(("Recorte", "ok", "Recorte marcado."))
    else:
        badges.append(("Recorte", "miss", "Sin recorte."))
    if state.get("color_kind") == "bw":
        badges.append(("B/N", "ok", "Blanco y negro."))
    elif state.get("color"):
        badges.append(("GamMac", "ok", "Color medido."))
    else:
        badges.append(("Color", "miss", "Sin medición de color."))
    stab = state.get("stab")
    if stab == "listo":
        badges.append(("Lista", "ok", "Estabilización lista."))
    elif stab == "corriendo":
        percent = state.get("percent")
        text = f"{int(percent)}%" if percent is not None else "…"
        badges.append((text, "run", "Estabilización en curso."))
    elif stab == "por_correr":
        badges.append(("Por correr", "wait", "Hay recuadro y falta el análisis."))
    else:
        badges.append(("Recuadro", "miss", "Sin recuadro de estabilización."))
    return badges


def summary(raw: Path) -> str:
    data = load(raw)
    parts = []
    usable = [item for item in data["segments"] if (item.get("recipe") or {}).get("ok")]
    if usable:
        parts.append(f"{len(usable)} tramo" + ("s" if len(usable) != 1 else ""))
    side = _read_perfo(perfo_path(raw))
    if data["perfo"].get("enabled"):
        status = (side or {}).get("status")
        if status == "done":
            parts.append("perforación lista")
        elif status == "running":
            parts.append("perforación en curso")
        elif status == "error":
            parts.append("perforación rechazada")
        else:
            parts.append("perforación sin medir")
    if data.get("crop"):
        parts.append("recorte")
    return " · ".join(parts) if parts else "Sin color ni recorte"


def read_perfo(raw: Path) -> dict | None:
    return _read_perfo(perfo_path(raw))


def perfo_horizontal(perfo) -> bool:
    """El eje de lado. Un archivo de antes, sin la clave, sigue corrigiendo los dos ejes."""
    if not isinstance(perfo, dict) or "horizontal" not in perfo:
        return True
    value = perfo.get("horizontal")
    if isinstance(value, str):
        return value.strip().lower() != "false"
    return value is not False


def perfo_ref(perfo) -> int | None:
    """Cuadro en el que se dibujó la zona. None si todavía no quedó guardado."""
    if not isinstance(perfo, dict):
        return None
    value = perfo.get("ref")
    if value is None or value is False:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def perfo_is_done(raw: Path, roi, frames: int | None = None) -> bool:
    """Ya hay un desplazamiento completo para esta zona. No hace falta medirlo de nuevo.

    `frames` es la cantidad de cuadros cuando el archivo no es un raw Bayer.
    """
    data = read_perfo(raw)
    if not data or data.get("status") != "done":
        return False
    tx = data.get("tx") or []
    if not tx:
        return False
    if frames is None:
        try:
            count = Path(raw).stat().st_size // (FULL_W * FULL_H * 3 // 2)
        except OSError:
            return False
    else:
        count = max(0, int(frames))
    if count and len(tx) != count:
        return False
    saved = data.get("roi")
    if roi and saved and tuple(int(v) for v in saved) != tuple(int(v) for v in roi):
        return False
    wanted = perfo_ref(load(raw).get("perfo") or {})
    if wanted is not None:
        try:
            if int(data.get("ref")) != wanted:
                return False
        except (TypeError, ValueError):
            return False
    return True


def write_perfo_status(
    raw: Path,
    status: str,
    message: str = "",
    result: dict | None = None,
    *,
    partial=None,
    roi=None,
    ref=None,
) -> None:
    path = perfo_path(raw)
    existing = _read_perfo(path) or {}
    payload = {"status": status, "message": message, "heartbeat": time.time()}
    if status == "done" and result:
        payload.update(result)
    elif status == "running":
        kept_roi = list(roi) if roi is not None else existing.get("roi")
        kept_ref = int(ref) if ref is not None else existing.get("ref")
        kept_partial = partial if partial is not None else existing.get("partial")
        if kept_roi is not None:
            payload["roi"] = list(kept_roi)
        if kept_ref is not None:
            payload["ref"] = kept_ref
        if kept_partial is not None:
            payload["partial"] = kept_partial
    elif status == "error":
        for key in ("roi", "ref", "partial"):
            if existing.get(key) is not None:
                payload[key] = existing[key]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _replace_file(tmp, path)


def _replace_file(tmp: Path, path: Path) -> None:
    """En Windows el reemplazo falla si la ventana está leyendo el mismo archivo."""
    delay = 0.02
    last = None
    for _attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(delay)
            delay = min(0.25, delay * 1.5)
    if last is not None:
        raise last


def _read_perfo(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def wait_perfo(raw: Path, timeout_idle: float = 45.0) -> dict | None:
    """Espera si la medición sigue viva. Si el pulso se corta, sigue sin ella."""
    path = perfo_path(raw)
    while True:
        data = _read_perfo(path)
        if not data or data.get("status") != "running":
            return data
        age = time.time() - float(data.get("heartbeat") or 0)
        if age > timeout_idle:
            print("WARN|La perforación quedó a medias. El video sale sin estabilizar.")
            return None
        time.sleep(0.4)


def cover(spans: list, count: int, blank) -> list:
    """Parte el rollo en tramos que lo cubren entero, sin huecos ni superposición."""
    last = max(0, int(count) - 1)
    ordered = []
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        start = max(0, min(int(span.get("in", 0)), last))
        end = max(start, min(int(span.get("out", start)), last))
        item = copy.deepcopy(span)
        item["in"] = start
        item["out"] = end
        ordered.append(item)
    ordered.sort(key=lambda item: int(item["in"]))
    if not ordered:
        return [blank(0, last)]
    out = []
    cursor = 0
    for span in ordered:
        start, end = int(span["in"]), int(span["out"])
        if start > cursor:
            out.append(blank(cursor, start - 1))
        if start < cursor:
            start = cursor
            span["in"] = start
        if start <= end:
            out.append(span)
            cursor = end + 1
        if cursor > last:
            break
    if cursor <= last:
        out.append(blank(cursor, last))
    return out


def retarget(spans: list, ranges: list) -> list:
    """Arma tramos nuevos copiando el que contiene el cuadro del medio de cada rango."""
    fresh = []
    for start, end in ranges:
        start, end = int(start), int(end)
        mid = (start + end) // 2
        source = None
        for span in spans:
            if int(span["in"]) <= mid <= int(span["out"]):
                source = span
                break
        if source is None and spans:
            source = spans[0]
        item = copy.deepcopy(source) if source is not None else {}
        item["in"] = start
        item["out"] = end
        fresh.append(item)
    return fresh


def split_at(spans: list, head: int, count: int):
    """Corta el tramo en la cabeza. Las dos mitades salen con la misma copia."""
    head = int(head)
    if not spans:
        return spans, 0, False
    ranges = []
    selected = 0
    changed = False
    for span in spans:
        start, end = int(span["in"]), int(span["out"])
        if start < head <= end:
            ranges.append((start, head - 1))
            ranges.append((head, end))
            selected = len(ranges) - 1
            changed = True
        else:
            ranges.append((start, end))
            if start <= head <= end:
                selected = len(ranges) - 1
    if not changed:
        return spans, selected, False
    return retarget(spans, ranges), selected, True


def carve(spans: list, start: int, end: int):
    """Recorta un tramo nuevo dentro del que contiene el cuadro inicial. El resto se queda."""
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    owner = None
    owner_at = 0
    for index, span in enumerate(spans):
        if int(span["in"]) <= start <= int(span["out"]):
            owner = span
            owner_at = index
            break
    if owner is None:
        return spans, 0, False
    left, right = int(owner["in"]), int(owner["out"])
    start = max(start, left)
    end = min(end, right)
    if start <= left and end >= right:
        return spans, owner_at, False
    ranges = []
    selected = 0
    for index, span in enumerate(spans):
        if index != owner_at:
            ranges.append((int(span["in"]), int(span["out"])))
            continue
        if start > left:
            ranges.append((left, start - 1))
        ranges.append((start, end))
        selected = len(ranges) - 1
        if end < right:
            ranges.append((end + 1, right))
    return retarget(spans, ranges), selected, True


def move_edge(spans: list, edge: int, frame: int):
    """Corre el borde compartido. Cada lado conserva al menos un cuadro."""
    edge = int(edge)
    if edge <= 0 or edge >= len(spans):
        return spans, False
    left, right = spans[edge - 1], spans[edge]
    low = int(left["in"]) + 1
    high = int(right["out"])
    frame = max(low, min(int(frame), high))
    if frame == int(right["in"]):
        return spans, False
    left["out"] = frame - 1
    right["in"] = frame
    return spans, True


def merge_span(spans: list, index: int):
    """Funde el tramo con el anterior. El anterior conserva su ajuste."""
    if len(spans) <= 1:
        return spans, 0, False
    index = max(0, min(int(index), len(spans) - 1))
    spans = [copy.deepcopy(item) for item in spans]
    if index == 0:
        spans[1]["in"] = int(spans[0]["in"])
        del spans[0]
        return spans, 0, True
    spans[index - 1]["out"] = int(spans[index]["out"])
    del spans[index]
    return spans, index - 1, True


def fit_aspect(rect, aspect: float, width: int, height: int, minimum: int = 8):
    """El recuadro más grande con esa proporción que cabe en el gesto y en el cuadro."""
    if not rect or len(rect) != 4 or aspect <= 0 or width < 2 or height < 2:
        return None
    x, y, rect_w, rect_h = (float(v) for v in rect)
    if rect_w < 1 or rect_h < 1:
        return None
    center_x = x + rect_w / 2.0
    center_y = y + rect_h / 2.0
    if rect_w / rect_h > aspect:
        rect_w = rect_h * aspect
    else:
        rect_h = rect_w / aspect
    min_w = min(float(width), max(float(minimum), aspect * max(1.0, minimum / 2.0)))
    if rect_w < min_w:
        rect_w = min_w
        rect_h = rect_w / aspect
    if rect_w > width:
        rect_w = float(width)
        rect_h = rect_w / aspect
    if rect_h > height:
        rect_h = float(height)
        rect_w = rect_h * aspect
    rect_w = max(1.0, min(rect_w, float(width)))
    rect_h = max(1.0, min(rect_h, float(height)))
    rect_h = rect_w / aspect
    if rect_h > height:
        rect_h = float(height)
        rect_w = rect_h * aspect
    x = min(max(0.0, center_x - rect_w / 2.0), width - rect_w)
    y = min(max(0.0, center_y - rect_h / 2.0), height - rect_h)
    snapped_w = max(1, int(round(rect_w)))
    snapped_h = max(1, int(round(snapped_w / aspect)))
    if snapped_h > height:
        snapped_h = int(height)
        snapped_w = max(1, int(round(snapped_h * aspect)))
    if snapped_w > width:
        snapped_w = int(width)
        snapped_h = max(1, int(round(snapped_w / aspect)))
    origin_x = min(max(0, int(round(x))), width - snapped_w)
    origin_y = min(max(0, int(round(y))), height - snapped_h)
    return origin_x, origin_y, snapped_w, snapped_h


def place_box(center_x: float, center_y: float, rect_w: float, rect_h: float, width: int, height: int):
    rect_w = max(1.0, min(float(rect_w), float(width)))
    rect_h = max(1.0, min(float(rect_h), float(height)))
    x = min(max(0.0, center_x - rect_w / 2.0), width - rect_w)
    y = min(max(0.0, center_y - rect_h / 2.0), height - rect_h)
    return int(round(x)), int(round(y)), max(1, int(round(rect_w))), max(1, int(round(rect_h)))


def refit_crops(spans: list, old, new, width: int, height: int) -> list:
    """El primer recorte cambia de tamaño. Los demás conservan el centro y la escala."""
    if not spans or not new:
        return spans
    old_w = float(old[2]) if old else float(new[2])
    spans[0]["crop"] = [int(v) for v in new]
    for span in spans[1:]:
        crop = span.get("crop") or old or new
        scale = (float(crop[2]) / old_w) if old_w else 1.0
        center_x = float(crop[0]) + float(crop[2]) / 2.0
        center_y = float(crop[1]) + float(crop[3]) / 2.0
        span["crop"] = list(place_box(
            center_x, center_y, float(new[2]) * scale, float(new[3]) * scale, width, height,
        ))
    return spans


def clip_box(crop, width: int, height: int):
    """Mete el recuadro en el cuadro sin cambiar la proporción más de lo que pide el borde."""
    if not crop or len(crop) != 4 or width < 2 or height < 2:
        return None
    x, y, rect_w, rect_h = (int(round(v)) for v in crop)
    if rect_w < 1 or rect_h < 1:
        return None
    aspect = rect_w / float(rect_h)
    if rect_w > width:
        rect_w = width
        rect_h = max(1, int(round(rect_w / aspect)))
    if rect_h > height:
        rect_h = height
        rect_w = max(1, int(round(rect_h * aspect)))
    x = min(max(0, x), width - rect_w)
    y = min(max(0, y), height - rect_h)
    if x == 0 and y == 0 and rect_w == width and rect_h == height:
        return None
    return int(x), int(y), int(rect_w), int(rect_h)


class Finish:
    """Lo que el codificador aplica a cada cuadro ya revelado."""

    def __init__(self, raw: Path, *, gpu: bool, codec: str):
        self.raw = Path(raw)
        self.data = load(self.raw)
        self.gpu = gpu
        self.codec = codec
        side = None
        if self.data["perfo"].get("enabled"):
            side = wait_perfo(self.raw)
        self.tx = list((side or {}).get("tx") or []) if side and side.get("status") == "done" else []
        self.ty = list((side or {}).get("ty") or []) if side and side.get("status") == "done" else []
        self.crop = _clip_crop(self.data.get("crop"))
        self.flip_h = bool(self.data.get("flip_h"))
        self.crop_spans = _finish_spans(self.data, self.crop, self.flip_h)
        if self.crop is None and self.crop_spans:
            self.crop = _clip_crop(self.crop_spans[0].get("crop"))
        self.segments = [
            item for item in self.data["segments"]
            if (item.get("recipe") or {}).get("ok") and not (item.get("recipe") or {}).get("limited")
        ]
        grade = _grade(self.data)
        self.grade_mode = grade["mode"]
        self.grade_sat = grade["sat"]
        self.turn = _turn(self.data)
        self.horizontal = perfo_horizontal(self.data.get("perfo"))

    @property
    def active(self) -> bool:
        graded = self.grade_mode == "bw" or self.grade_sat < 0.999
        turned = abs(self.turn) >= 0.05
        flipped = self.flip_h or any(bool(span.get("flip_h")) for span in self.crop_spans)
        return bool(self.segments or self.tx or self.crop or graded or turned or flipped)

    def sampling(self, index: int, width: int, height: int):
        tx, ty = self.shift_at(index)
        if abs(tx) < 1e-4 and abs(ty) < 1e-4 and abs(self.turn) < 0.05:
            return None
        box = self.crop_box(width, height, index)
        if box:
            center = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        else:
            center = (width / 2.0, height / 2.0)
        return sampling_matrix(tx, ty, self.turn, center)

    def output_size(self, width: int = FULL_W, height: int = FULL_H) -> tuple[int, int]:
        box = self.crop_box(width, height)
        cw, ch = (box[2], box[3]) if box else (width, height)
        if self.gpu and self.codec == "prores_4444":
            from reveal_gpu import prores_nv_size
            return prores_nv_size(cw, ch)
        even_w = max(2, cw - (cw % 2))
        even_h = max(2, ch - (ch % 2))
        return even_w, even_h

    def _span_at(self, index: int):
        if not self.crop_spans:
            return None
        index = int(index)
        for span in self.crop_spans:
            if int(span["in"]) <= index <= int(span["out"]):
                return span
        if index > int(self.crop_spans[-1]["out"]):
            return self.crop_spans[-1]
        return self.crop_spans[0]

    def flip_at(self, index: int) -> bool:
        span = self._span_at(index)
        if span is None:
            return self.flip_h
        return bool(span.get("flip_h"))

    def crop_box(self, width: int, height: int, index: int | None = None):
        if index is None or not self.crop_spans:
            return _clip_crop(self.crop, width, height)
        span = self._span_at(index)
        if span is None:
            return _clip_crop(self.crop, width, height)
        if span is self.crop_spans[0]:
            return _clip_crop(span.get("crop"), width, height)
        return clip_box(span.get("crop"), width, height)

    def recipe_at(self, index: int) -> dict | None:
        return gammac.recipe_at(self.segments, int(index))

    def shift_at(self, index: int) -> tuple[float, float]:
        if not self.tx or index < 0 or index >= len(self.tx):
            return 0.0, 0.0
        tx = float(self.tx[index])
        if not self.horizontal:
            tx = 0.0
        return tx, float(self.ty[index])

    def apply_u16(self, image: np.ndarray, index: int, out_wh: tuple[int, int]) -> np.ndarray:
        work = image.astype(np.float32) * np.float32(1.0 / 65535.0)
        work = self.apply_float(work, index, out_wh)
        return np.clip(work * np.float32(65535.0) + np.float32(0.5), 0, 65535).astype(np.uint16)

    def apply_float(self, image: np.ndarray, index: int, out_wh: tuple[int, int] | None = None) -> np.ndarray:
        work = image if self.grade_mode == "bw" else gammac.apply(image, self.recipe_at(index))
        work = gammac.saturate(work, self.grade_sat)
        height, width = work.shape[:2]
        matrix = self.sampling(index, width, height)
        if matrix is not None:
            work = cv2.warpAffine(
                work, matrix, (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        if out_wh is None:
            cropped = work
        else:
            master = self.crop_box(width, height)
            box = self.crop_box(width, height, index)
            used = box if box is not None else master
            other_size = (
                used is not None and master is not None
                and (used[2], used[3]) != (master[2], master[3])
            )
            if other_size:
                cropped = _crop_scale(work, used, out_wh)
            else:
                cropped = _crop_pad(work, used, out_wh)
        if self.flip_at(index):
            cropped = np.ascontiguousarray(cropped[:, ::-1])
        return cropped


def _finish_spans(data: dict, crop, flip_h: bool) -> list:
    """Sin el recorte de arriba no se exporta ninguno. Un archivo viejo es un solo tramo."""
    if not crop and not data.get("crop"):
        return []
    spans = [item for item in (data.get("crop_spans") or []) if isinstance(item, dict) and item.get("crop")]
    if spans:
        return spans
    if not data.get("crop"):
        return []
    return [{"in": 0, "out": 10**12, "crop": data.get("crop"), "flip_h": flip_h}]


def _crop_scale(image: np.ndarray, box, out_wh: tuple[int, int]) -> np.ndarray:
    """Escala el recorte al tamaño único del archivo. La proporción ya es la del primero."""
    out_w, out_h = out_wh
    x, y, rect_w, rect_h = box
    cropped = image[y:y + rect_h, x:x + rect_w]
    if cropped.shape[1] == out_w and cropped.shape[0] == out_h:
        return np.ascontiguousarray(cropped)
    shrink = cropped.shape[1] > out_w or cropped.shape[0] > out_h
    interp = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    return np.ascontiguousarray(cv2.resize(cropped, (out_w, out_h), interpolation=interp))


def snap_crop(crop, width: int = FULL_W, height: int = FULL_H):
    """Ancho múltiplo de 16 y alto múltiplo de 8, que es lo que pide prores_nv.

    Se achica desde la esquina de arriba a la izquierda, para no volver a
    incluir borde que el recorte había dejado afuera. Si no cabe el mínimo,
    la esquina se corre hacia adentro del cuadro.
    """
    if not crop or len(crop) != 4:
        return None
    x, y, rect_w, rect_h = (int(round(v)) for v in crop)
    if width < 16 or height < 8:
        return None
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    rect_w = max(1, min(rect_w, width - x))
    rect_h = max(1, min(rect_h, height - y))
    snapped_w = max(16, (rect_w // 16) * 16)
    snapped_h = max(8, (rect_h // 8) * 8)
    snapped_w = min(snapped_w, (width // 16) * 16)
    snapped_h = min(snapped_h, (height // 8) * 8)
    if x + snapped_w > width:
        x = width - snapped_w
    if y + snapped_h > height:
        y = height - snapped_h
    return int(x), int(y), int(snapped_w), int(snapped_h)


def _clip_crop(crop, width: int = FULL_W, height: int = FULL_H):
    snapped = snap_crop(crop, width, height)
    if snapped is None:
        return None
    x, y, rect_w, rect_h = snapped
    if x == 0 and y == 0 and rect_w == width and rect_h == height:
        return None
    return snapped


def _crop_pad(image: np.ndarray, box, out_wh: tuple[int, int]) -> np.ndarray:
    out_w, out_h = out_wh
    if box is None:
        cropped = image
    else:
        x, y, rect_w, rect_h = box
        cropped = image[y:y + rect_h, x:x + rect_w]
    height, width = cropped.shape[:2]
    if width == out_w and height == out_h:
        return np.ascontiguousarray(cropped)
    canvas = np.zeros((out_h, out_w, cropped.shape[2]), dtype=cropped.dtype)
    copy_w = min(width, out_w)
    copy_h = min(height, out_h)
    canvas[:copy_h, :copy_w] = cropped[:copy_h, :copy_w]
    if copy_w < out_w and copy_w > 0:
        canvas[:, copy_w:] = canvas[:, copy_w - 1:copy_w]
    if copy_h < out_h and copy_h > 0:
        canvas[copy_h:, :] = canvas[copy_h - 1:copy_h, :]
    return np.ascontiguousarray(canvas)
