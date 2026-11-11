"""Estabilización por perforación, método Perfo de docs/perfo-lock.md.

La medición corre en varios procesos, un hilo de OpenCV cada uno, sobre el
recorte en gris del raw. La trayectoria se limpia recién cuando están todos
los cuadros.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

MAX_DIM = 1200
ECC_ITERS = 50
ECC_EPS = 0.0001
MIN_CC = 0.5
MIN_STD_VISTO = 5.0
MIN_STD_CANDIDATO = 10.0
MAX_FRAC_MALOS = 0.30
ROI_MIN = 8

_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)


class PerfoError(RuntimeError):
    pass


def analyze_grays(frames: list[np.ndarray], rect, ref_index: int = 0) -> dict:
    """Misma medición, con los grises ya en memoria. Sirve para la prueba."""
    if not frames:
        raise PerfoError("No hay cuadros.")
    height, width = frames[0].shape[:2]
    ident = _clip_rect(rect, width, height)
    search = _search_rect(ident, width, height)
    ref = _pick_ref(frames, ident, ref_index)
    spec = _spec_from_frame(frames[ref], ident, search)
    rows = [_register(frame[search[1]:search[1] + search[3], search[0]:search[0] + search[2]], spec) for frame in frames]
    return _pack(rows, ident, search, ref)


def analyze_file(
    path,
    rect,
    *,
    width: int = 2840,
    height: int = 2200,
    ref_index: int = 0,
    workers: int | None = None,
    on_progress=None,
) -> dict:
    """Lee el raw de corrido, en tramos, y reparte el ECC entre procesos."""
    path = Path(path)
    frame_bytes = width * height * 3 // 2
    count = path.stat().st_size // frame_bytes
    if count <= 0:
        raise PerfoError("El archivo no tiene cuadros.")
    ident = _clip_rect(rect, width, height)
    search = _search_rect(ident, width, height)
    ref = int(np.clip(ref_index, 0, count - 1))
    rows, saved_ref = _load_resume(path, ident, count)
    if saved_ref != ref:
        rows = [None] * count
    ref_gray = _read_search(path, frame_bytes, width, ref, search)
    if ref_gray is None:
        raise PerfoError("No se pudo leer el cuadro de referencia.")
    spec = _spec_from_crop(ref_gray, ident, search)
    if workers is None:
        jobs = min(8, max(2, (os.cpu_count() or 4) - 2))
    else:
        jobs = max(1, int(workers))
    jobs = min(jobs, max(1, count // 8 or 1))
    chunk = 48 if count >= 64 else count
    payload = (str(path), frame_bytes, width, search, spec)
    spans = _pending_spans(rows, chunk)
    if not spans:
        return _pack(rows, ident, search, ref)
    done = sum(row is not None for row in rows)
    if on_progress and done:
        on_progress(done, count)

    def _store(start: int, part: list):
        rows[start:start + len(part)] = part
        filled = sum(row is not None for row in rows)
        if on_progress:
            on_progress(filled, count, rows, ref, ident)

    if jobs == 1 or count < 32:
        previous = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            for start, end in spans:
                _store(start, _span(payload, start, end))
        finally:
            cv2.setNumThreads(previous)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_span, payload, start, end): start for start, end in spans}
            for future in as_completed(futures):
                _store(futures[future], future.result())
    return _pack(rows, ident, search, ref)


def _load_resume(path: Path, ident, count: int) -> tuple[list, int | None]:
    """Cuadros ya medidos y el cuadro de referencia, si la zona no cambió."""
    from film_finish import read_perfo
    data = read_perfo(path) or {}
    saved = data.get("roi")
    partial = data.get("partial")
    blank = [None] * count
    if data.get("ref") is None or not saved or not isinstance(partial, list):
        return blank, None
    if tuple(int(v) for v in saved) != tuple(int(v) for v in ident):
        return blank, None
    rows = []
    for item in partial[:count]:
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            rows.append((float(item[0]), float(item[1]), float(item[2]), int(item[3])))
        else:
            rows.append(None)
    if len(rows) < count:
        rows.extend([None] * (count - len(rows)))
    if not any(row is not None for row in rows):
        return blank, None
    return rows, int(data["ref"])


def _pending_spans(rows: list, chunk: int) -> list[tuple[int, int]]:
    spans = []
    start = None
    limit = max(1, int(chunk))
    for index, row in enumerate(rows):
        if row is None:
            if start is None:
                start = index
            if index - start + 1 >= limit:
                spans.append((start, index + 1))
                start = None
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(rows)))
    return spans


def _span(payload, start: int, end: int):
    cv2.setNumThreads(1)
    path, frame_bytes, width, search, spec = payload
    out = []
    with open(path, "rb") as handle:
        for index in range(start, end):
            gray = _read_search_handle(handle, frame_bytes, width, index, search)
            out.append(_register(gray, spec) if gray is not None else (0.0, 0.0, 0.0, 0))
    return out


def _read_search(path, frame_bytes, width, index, search):
    with open(path, "rb") as handle:
        return _read_search_handle(handle, frame_bytes, width, index, search)


def _read_search_handle(handle, frame_bytes, width, index, search):
    x, y, rect_w, rect_h = search
    row_bytes = width * 3 // 2
    handle.seek(index * frame_bytes + y * row_bytes)
    raw = handle.read(rect_h * row_bytes)
    if len(raw) < rect_h * row_bytes:
        return None
    mosaic = _unpack_rows(raw, width, rect_h)
    crop = mosaic[:, x:x + rect_w]
    return np.clip(crop.astype(np.float32) * (255.0 / 4095.0) + 0.5, 0, 255).astype(np.uint8)


def _unpack_rows(raw: bytes, width: int, height: int) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    b0 = arr[:, 0].astype(np.uint16)
    b1 = arr[:, 1].astype(np.uint16)
    b2 = arr[:, 2].astype(np.uint16)
    flat = np.empty(width * height, dtype=np.uint16)
    flat[0::2] = b0 | ((b1 & 0x0F) << 8)
    flat[1::2] = (b1 >> 4) | (b2 << 4)
    return flat.reshape(height, width)


def _clip_rect(rect, width, height):
    x, y, rect_w, rect_h = (int(round(v)) for v in rect)
    x = max(0, min(x, width - ROI_MIN))
    y = max(0, min(y, height - ROI_MIN))
    rect_w = max(ROI_MIN, min(rect_w, width - x))
    rect_h = max(ROI_MIN, min(rect_h, height - y))
    return x, y, rect_w, rect_h


def _even_span(x, y, rect_w, rect_h, width, height):
    """Origen y tamaño pares.

    En 4:2:2 ffmpeg achica el lado impar. Si después se lee el tamaño pedido,
    cada cuadro queda una fila corrido y la perforación no calza con nada.
    """
    x = max(0, int(x) - (int(x) % 2))
    y = max(0, int(y) - (int(y) % 2))
    limit_x = int(width) - (int(width) % 2)
    limit_y = int(height) - (int(height) % 2)
    x1 = min(limit_x, x + int(rect_w))
    y1 = min(limit_y, y + int(rect_h))
    if (x1 - x) % 2:
        x1 -= 1
    if (y1 - y) % 2:
        y1 -= 1
    if x1 - x < 2 and x + 2 <= limit_x:
        x1 = x + 2
    if y1 - y < 2 and y + 2 <= limit_y:
        y1 = y + 2
    return x, y, max(0, x1 - x), max(0, y1 - y)


def _search_rect(ident, width, height):
    margin = int(np.clip(round(0.05 * max(width, height)), 40, 400))
    x = max(0, ident[0] - margin)
    y = max(0, ident[1] - margin)
    x1 = min(width, ident[0] + ident[2] + margin)
    y1 = min(height, ident[1] + ident[3] + margin)
    return _even_span(x, y, x1 - x, y1 - y, width, height)


def _pick_ref(frames, ident, ref_index):
    x, y, rect_w, rect_h = ident
    asked = frames[ref_index][y:y + rect_h, x:x + rect_w]
    if float(asked.std()) >= MIN_STD_VISTO:
        return ref_index
    last = len(frames) - 1
    for frac in (0, 0.05, 0.15, 0.35, 0.60, 0.85):
        index = int(np.clip(round(frac * last), 0, last))
        crop = frames[index][y:y + rect_h, x:x + rect_w]
        if float(crop.std()) >= MIN_STD_CANDIDATO:
            return index
    raise PerfoError("Sin perforación")


def analyze_ordered(
    sample_crop,
    crops,
    count: int,
    rect,
    width: int,
    height: int,
    ref_index: int = 0,
    on_progress=None,
) -> dict:
    """Mide la perforación sobre grises del recuadro de búsqueda, en orden.

    `sample_crop(index)` lee un cuadro suelto. `crops` recorre el video
    desde el primero, ya recortado a la zona de búsqueda.
    """
    if count <= 0:
        raise PerfoError("El archivo no tiene cuadros.")
    ident = _clip_rect(rect, width, height)
    search = _search_rect(ident, width, height)
    ref = int(np.clip(ref_index, 0, count - 1))
    ref_gray = sample_crop(ref)
    if ref_gray is None:
        raise PerfoError("No se pudo leer el cuadro de referencia.")
    spec = _spec_from_crop(ref_gray, ident, search)
    rows = []
    for index, gray in enumerate(crops):
        if index >= count:
            break
        rows.append(_register(gray, spec) if gray is not None else (0.0, 0.0, 0.0, 0))
        if on_progress and index % 8 == 7:
            on_progress(index + 1, count)
    if not rows:
        raise PerfoError("No salió ningún cuadro.")
    if on_progress:
        on_progress(len(rows), max(count, len(rows)))
    return _pack(rows, ident, search, ref)


def _pick_ref_reader(sample_crop, ident, search, ref_index, count):
    asked = sample_crop(int(np.clip(ref_index, 0, count - 1)))
    if asked is not None and _crop_std(asked, ident, search) >= MIN_STD_VISTO:
        return int(np.clip(ref_index, 0, count - 1))
    last = count - 1
    for frac in (0, 0.05, 0.15, 0.35, 0.60, 0.85):
        index = int(np.clip(round(frac * last), 0, last))
        gray = sample_crop(index)
        if gray is not None and _crop_std(gray, ident, search) >= MIN_STD_CANDIDATO:
            return index
    raise PerfoError("Sin perforación")


def _crop_std(crop, ident, search) -> float:
    x = ident[0] - search[0]
    y = ident[1] - search[1]
    window = crop[y:y + ident[3], x:x + ident[2]]
    if window.size == 0:
        return 0.0
    return float(window.std())


def _pick_ref_file(path, frame_bytes, width, ident, ref_index, count):
    x, y, rect_w, rect_h = ident
    asked = _read_search(path, frame_bytes, width, ref_index, (x, y, rect_w, rect_h))
    if asked is not None and float(asked.std()) >= MIN_STD_VISTO:
        return ref_index
    last = count - 1
    for frac in (0, 0.05, 0.15, 0.35, 0.60, 0.85):
        index = int(np.clip(round(frac * last), 0, last))
        gray = _read_search(path, frame_bytes, width, index, (x, y, rect_w, rect_h))
        if gray is not None and float(gray.std()) >= MIN_STD_CANDIDATO:
            return index
    raise PerfoError("Sin perforación")


def _spec_from_frame(frame, ident, search):
    crop = frame[search[1]:search[1] + search[3], search[0]:search[0] + search[2]]
    return _spec_from_crop(crop, ident, search)


def _spec_from_crop(search_gray, ident, search):
    local = (ident[0] - search[0], ident[1] - search[1], ident[2], ident[3])
    return _prepare(search_gray, local, (0, 0, search[2], search[3]))


def _prepare(search_gray, ident_in_search, search_box):
    """`ident_in_search` está en coordenadas del recorte de búsqueda."""
    tight = _tighten(search_gray, ident_in_search)
    long_side = float(max(tight[2], tight[3]))
    scale = min(1.0, MAX_DIM / long_side)
    # Un agujero chico cierra en pocas vueltas a tamaño real. Uno grande, como
    # el de 405×666, gasta las 50. A la mitad el vector sigue en píxeles del
    # cuadro porque el residuo se divide por esta escala.
    if long_side > 240:
        scale *= 0.5
    window = search_gray[tight[1]:tight[1] + tight[3], tight[0]:tight[0] + tight[2]]
    template = _edges(_reduce(window, scale))
    use_match = search_box[2] > tight[2] or search_box[3] > tight[3]
    scale_b = min(1.0, MAX_DIM / float(max(search_box[2], search_box[3])))
    return {
        "template": np.ascontiguousarray(template, dtype=np.float32),
        "scale": float(scale),
        "origin": (int(tight[0]), int(tight[1])),
        "tight": (int(tight[2]), int(tight[3])),
        "use_match": bool(use_match),
        "scale_b": float(scale_b),
    }


def _tighten(gray, ident):
    x, y, rect_w, rect_h = ident
    crop = gray[y:y + rect_h, x:x + rect_w]
    if crop.size == 0:
        return ident
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    shown = clahe.apply(crop)
    shown = cv2.GaussianBlur(shown, (7, 7), 0)
    best = None
    best_area = 0
    for flag in (cv2.THRESH_BINARY | cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU):
        _level, mask = cv2.threshold(shown, 0, 255, flag)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        areas = [cv2.contourArea(item) for item in contours]
        peak = max(areas)
        if peak <= 0:
            continue
        kept = [contours[i] for i, area in enumerate(areas) if area >= 0.15 * peak]
        points = np.vstack(kept)
        box_x, box_y, box_w, box_h = cv2.boundingRect(points)
        if box_w * box_h > 0.88 * rect_w * rect_h:
            continue
        total = float(sum(cv2.contourArea(item) for item in kept))
        if total > best_area:
            best_area = total
            best = (box_x, box_y, box_w, box_h)
    if best is None:
        return ident
    box_x, box_y, box_w, box_h = best
    if box_w * box_h >= 0.5 * rect_w * rect_h:
        return ident
    mx = max(6, int(box_w * 0.08))
    my = max(6, int(box_h * 0.08))
    ax = max(0, box_x - mx)
    ay = max(0, box_y - my)
    ax1 = min(rect_w, box_x + box_w + mx)
    ay1 = min(rect_h, box_y + box_h + my)
    if ax1 - ax < ROI_MIN or ay1 - ay < ROI_MIN:
        return ident
    return x + ax, y + ay, ax1 - ax, ay1 - ay


def _reduce(image, scale):
    if scale >= 0.999:
        return image
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _edges(gray):
    soft = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _register(search_gray, spec) -> tuple[float, float, float, int]:
    if search_gray is None:
        return 0.0, 0.0, 0.0, 0
    ox, oy = spec["origin"]
    tw, th = spec["tight"]
    window = search_gray[oy:oy + th, ox:ox + tw]
    current = _fit(_edges(_reduce(window, spec["scale"])), spec["template"].shape)
    tx, ty, cc, flags = _ecc(spec["template"], current, spec["scale"], (0.0, 0.0))
    if cc < MIN_CC:
        phase = _phase(spec["template"], current, spec["scale"])
        if phase is not None:
            tx2, ty2, cc2, _flags = _ecc(spec["template"], current, spec["scale"], phase)
            if cc2 > cc:
                tx, ty, cc = tx2, ty2, cc2
                if cc2 >= MIN_CC:
                    flags |= 1
    if spec["use_match"] and cc < 0.85:
        jumped = _match(search_gray, spec, (tx, ty, cc))
        if jumped is not None:
            tx, ty, cc = jumped
            flags |= 2
    return float(tx), float(ty), float(cc), int(flags)


def _fit(image, shape):
    if image.shape == shape:
        return np.ascontiguousarray(image, dtype=np.float32)
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _ecc(template, current, scale, start):
    acc_x, acc_y = float(start[0]), float(start[1])
    cc = 0.0
    current = np.ascontiguousarray(current, dtype=np.float32)
    template = np.ascontiguousarray(template, dtype=np.float32)
    for pass_index in range(2):
        image = current
        if acc_x or acc_y:
            matrix = np.array([[1, 0, acc_x * scale], [0, 1, acc_y * scale]], np.float32)
            image = cv2.warpAffine(
                current, matrix, (current.shape[1], current.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        warp = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        try:
            cc_pass, warp = cv2.findTransformECC(
                template, image, warp, cv2.MOTION_TRANSLATION, _CRITERIA, None, 5,
            )
        except cv2.error:
            if pass_index == 0:
                return 0.0, 0.0, 0.0, 0
            break
        acc_x += float(warp[0, 2]) / scale
        acc_y += float(warp[1, 2]) / scale
        if pass_index == 0:
            cc = float(cc_pass)
    return acc_x, acc_y, cc, 0


def _phase(template, current, scale):
    try:
        window = cv2.createHanningWindow((template.shape[1], template.shape[0]), cv2.CV_32F)
        shift, _response = cv2.phaseCorrelate(template, current, window)
    except cv2.error:
        return None
    return float(shift[0]) / scale, float(shift[1]) / scale


def _match(search_gray, spec, fine):
    tx0, ty0, cc0 = fine
    scale_b = spec["scale_b"]
    ox, oy = spec["origin"]
    tw, th = spec["tight"]
    window = search_gray[oy:oy + th, ox:ox + tw]
    templ = _edges(_reduce(window, scale_b))
    reduced = _edges(_reduce(search_gray, scale_b))
    if templ.shape[0] >= reduced.shape[0] or templ.shape[1] >= reduced.shape[1]:
        return None
    try:
        score = cv2.matchTemplate(reduced, templ, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None
    _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(score)
    if not np.isfinite(max_v) or max_v < 0.12:
        return None
    px = int(np.clip(round(max_l[0] / scale_b), 0, search_gray.shape[1] - spec["tight"][0]))
    py = int(np.clip(round(max_l[1] / scale_b), 0, search_gray.shape[0] - spec["tight"][1]))
    sx = px - spec["origin"][0]
    sy = py - spec["origin"][1]
    if cc0 >= MIN_CC and abs(sx - tx0) <= 8 and abs(sy - ty0) <= 8:
        return None
    window = search_gray[py:py + spec["tight"][1], px:px + spec["tight"][0]]
    current = _fit(_edges(_reduce(window, spec["scale"])), spec["template"].shape)
    tx, ty, cc, _flags = _ecc(spec["template"], current, spec["scale"], (0.0, 0.0))
    if cc < MIN_CC:
        return None
    return tx + sx, ty + sy, cc


def _pack(rows, ident, search, ref) -> dict:
    bad = sum(1 for row in rows if row[2] < MIN_CC)
    if bad > MAX_FRAC_MALOS * len(rows):
        raise PerfoError("Sin perforación")
    tx = np.array([row[0] for row in rows], np.float64)
    ty = np.array([row[1] for row in rows], np.float64)
    cc = np.array([row[2] for row in rows], np.float64)
    flags = np.array([row[3] for row in rows], np.int32)
    protected = ((flags & 1) != 0) & (cc >= MIN_CC) | ((flags & 2) != 0)
    _fill_low(tx, ty, cc, protected)
    _fill_spikes(tx, protected)
    _fill_spikes(ty, protected)
    _fill_outliers(tx, protected)
    _fill_outliers(ty, protected)
    reach = float(np.max(np.maximum(np.abs(tx), np.abs(ty)))) if len(tx) else 0.0
    return {
        "tx": [round(float(v), 3) for v in tx],
        "ty": [round(float(v), 3) for v in ty],
        "cc": [round(float(v), 3) for v in cc],
        "ref": int(ref),
        "roi": list(ident),
        "search": list(search),
        "pad": int(np.ceil(reach) + 8),
        "mean_cc": round(float(cc.mean()), 3),
    }


def _fill_low(tx, ty, cc, protected):
    good = np.flatnonzero((cc >= MIN_CC) | protected)
    if good.size == 0:
        return
    for index in range(len(tx)):
        if protected[index] or cc[index] >= MIN_CC:
            continue
        prev_i = good[good < index]
        next_i = good[good > index]
        if prev_i.size and next_i.size:
            a = int(prev_i[-1])
            b = int(next_i[0])
            t = (index - a) / float(b - a)
            tx[index] = tx[a] + (tx[b] - tx[a]) * t
            ty[index] = ty[a] + (ty[b] - ty[a]) * t
        elif next_i.size:
            tx[index] = tx[int(next_i[0])]
            ty[index] = ty[int(next_i[0])]
        else:
            tx[index] = tx[int(prev_i[-1])]
            ty[index] = ty[int(prev_i[-1])]


def _fill_spikes(values, protected):
    count = len(values)
    if count < 3:
        return
    delta = np.diff(values)
    med = np.median(delta)
    mad = np.median(np.abs(delta - med))
    limit = max(4.0, 4 * 1.4826 * mad)
    marked = np.zeros(count, dtype=bool)
    for index in range(1, count - 1):
        if protected[index]:
            continue
        left = values[index] - values[index - 1]
        right = values[index] - values[index + 1]
        if abs(left) > limit and abs(right) > limit and left * right > 0:
            marked[index] = True
    if marked.sum() > 0.20 * count:
        return
    _interp_marked(values, marked)


def _fill_outliers(values, protected):
    count = len(values)
    if count < 5:
        return
    trend = np.empty(count, np.float64)
    for index in range(count):
        a = max(0, index - 3)
        b = min(count, index + 4)
        trend[index] = np.median(values[a:b])
    resid = values - trend
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    limit = max(8.0, 4 * 1.4826 * mad)
    marked = (np.abs(resid) > limit) & ~protected
    if marked.sum() > 0.25 * count:
        return
    _interp_marked(values, marked)


def _interp_marked(values, marked):
    original = values.copy()
    good = np.flatnonzero(~marked)
    if good.size == 0:
        return
    for index in np.flatnonzero(marked):
        prev_i = good[good < index]
        next_i = good[good > index]
        if prev_i.size and next_i.size:
            a = int(prev_i[-1])
            b = int(next_i[0])
            t = (index - a) / float(b - a)
            values[index] = original[a] + (original[b] - original[a]) * t
        elif next_i.size:
            values[index] = original[int(next_i[0])]
        elif prev_i.size:
            values[index] = original[int(prev_i[-1])]
