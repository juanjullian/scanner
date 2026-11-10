"""Ajustes de un archivo: tramos de GamMac, perforación y recorte.

El revelado sigue siendo uno para toda la exportación. Esto se guarda al lado
del raw y se aplica al codificar, después del revelado.
"""
from __future__ import annotations

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
        "perfo": {"enabled": False, "roi": None},
        "grade": {"mode": "color", "sat": 1.0},
        "turn": 0.0,
    }


def _turn(data) -> float:
    try:
        value = float(data.get("turn", 0.0)) if isinstance(data, dict) else 0.0
    except (TypeError, ValueError):
        value = 0.0
    return max(-180.0, min(180.0, round(value, 1)))


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
        }
        base["grade"] = _grade(data)
        base["turn"] = _turn(data)
    return base


def save(raw: Path, data: dict) -> None:
    path = adjust_path(raw)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def has_adjust(data: dict) -> bool:
    """Hay color, recorte o zona de perforación para copiar."""
    if data.get("crop"):
        return True
    if (data.get("perfo") or {}).get("roi"):
        return True
    if abs(_turn(data)) >= 0.05:
        return True
    return bool(data.get("segments"))


def copy_adjust(src: Path, dst: Path) -> bool:
    """Copia color, recorte y zona. No copia el desplazamiento medido de la perforación."""
    data = load(src)
    if not has_adjust(data):
        return False
    save(dst, json.loads(json.dumps(data)))
    return True


def review_state(raw: Path) -> dict:
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
    if roi and perfo_is_done(raw, roi):
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


def perfo_is_done(raw: Path, roi) -> bool:
    """Ya hay un desplazamiento completo para esta zona. No hace falta medirlo de nuevo."""
    data = read_perfo(raw)
    if not data or data.get("status") != "done":
        return False
    tx = data.get("tx") or []
    if not tx:
        return False
    try:
        count = Path(raw).stat().st_size // (FULL_W * FULL_H * 3 // 2)
    except OSError:
        return False
    if count and len(tx) != count:
        return False
    saved = data.get("roi")
    if roi and saved and tuple(int(v) for v in saved) != tuple(int(v) for v in roi):
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
        self.segments = [
            item for item in self.data["segments"]
            if (item.get("recipe") or {}).get("ok") and not (item.get("recipe") or {}).get("limited")
        ]
        grade = _grade(self.data)
        self.grade_mode = grade["mode"]
        self.grade_sat = grade["sat"]
        self.turn = _turn(self.data)

    @property
    def active(self) -> bool:
        graded = self.grade_mode == "bw" or self.grade_sat < 0.999
        turned = abs(self.turn) >= 0.05
        return bool(self.segments or self.tx or self.crop or graded or turned)

    def sampling(self, index: int, width: int, height: int):
        tx, ty = self.shift_at(index)
        if abs(tx) < 1e-4 and abs(ty) < 1e-4 and abs(self.turn) < 0.05:
            return None
        box = self.crop_box(width, height)
        if box:
            center = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        else:
            center = (width / 2.0, height / 2.0)
        return sampling_matrix(tx, ty, self.turn, center)

    def output_size(self, width: int = FULL_W, height: int = FULL_H) -> tuple[int, int]:
        box = self.crop_box(width, height)
        cw, ch = (box[2], box[3]) if box else (width, height)
        if self.gpu and self.codec == "prores_xq":
            from reveal_gpu import prores_nv_size
            return prores_nv_size(cw, ch)
        even_w = max(2, cw - (cw % 2))
        even_h = max(2, ch - (ch % 2))
        return even_w, even_h

    def crop_box(self, width: int, height: int):
        return _clip_crop(self.crop, width, height)

    def recipe_at(self, index: int) -> dict | None:
        return gammac.recipe_at(self.segments, int(index))

    def shift_at(self, index: int) -> tuple[float, float]:
        if not self.tx or index < 0 or index >= len(self.tx):
            return 0.0, 0.0
        return float(self.tx[index]), float(self.ty[index])

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
            return work
        return _crop_pad(work, self.crop_box(work.shape[1], work.shape[0]), out_wh)


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
