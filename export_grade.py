"""Confirmación de exportación, gradación opcional y preview A/B."""

from __future__ import annotations

import copy
from pathlib import Path

import bayer_render
import cv2
import numpy as np
import qoi_utils
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QSpinBox,
    QAbstractItemView, QWidget, QSlider, QGroupBox, QCheckBox,
    QGridLayout, QDoubleSpinBox,
)

import scanner_core

FIXED_W, FIXED_H = 2840, 2200
DEFAULT_FPS = 18
TEST_EXPORT_SKIP_FRAMES = 100
TEST_EXPORT_FRAME_COUNT = 200
RAW12_MAX = 4095

_EXT_MAP = {
    "prores": ".mov",
    "prores_hq": ".mov",
    "cineform": ".mov",
    "hevc": ".mp4",
    "h264": ".mp4",
    "av1": ".mp4",
    "ffv1": ".mkv",
}

DEFAULT_GRADE: dict = {
    "enabled": False,
    "gamma": 1.4,
    "exposure": 1.0,
    "black_level": 0.0,
    "white_level": 4095.0,
    "shadow_lift": 0.0,
    "contrast": 1.0,
    "chroma_blur": 3.0,
    "sharp_sigma": 2.0,
    "sharp_amount": 2.5,
    "wb_red": 1.0,
    "wb_green": 1.0,
    "wb_blue": 1.0,
    "rotate_deg": 0.0,
    "zoom_px": 0,
    "pan_x": 0,
    "pan_y": 0,
    "output_width": FIXED_W,
}

# Límites de encuadre a resolución de captura (2840×2200)
GEO_ROTATE_MIN, GEO_ROTATE_MAX = -15.0, 15.0
GEO_ROTATE_STEP = 0.1
GEO_ZOOM_MAX = 1000
GEO_PAN_X_MAX = int(round(FIXED_W * 0.35))
GEO_PAN_Y_MAX = int(round(FIXED_H * 0.35))
GEO_WIDTH_MIN = 1200
GEO_WIDTH_MAX = 3600
CANVAS_BORDER_PX = 2
CANVAS_BORDER_COLOR = (0, 210, 255)


def default_grade_settings() -> dict:
    return copy.deepcopy(DEFAULT_GRADE)


def default_output_name(stem: str, fmt: str) -> str:
    if fmt == "tiff_seq":
        return f"{stem}_TIFF_SEQ"
    if fmt == "dng":
        return f"{stem}_DNG_SEQ"
    if fmt == "tif":
        return stem
    ext = _EXT_MAP.get(fmt, ".mp4")
    return f"{stem}_{fmt}{ext}"


def default_settings_for_file(filename: str, collection_meta: dict, fmt: str = "prores") -> dict:
    info = collection_meta.get(filename, {})
    try:
        fps = int(info.get("fps", DEFAULT_FPS))
    except (TypeError, ValueError):
        fps = DEFAULT_FPS
    stem = Path(filename).stem
    saved_grade = info.get("export_grade")
    if isinstance(saved_grade, dict) and saved_grade:
        grade = copy.deepcopy(saved_grade)
    else:
        grade = default_grade_settings()
    output_name = info.get("export_output_name") or default_output_name(stem, fmt)
    return {
        "fps": fps,
        "output_name": output_name,
        "grade": grade,
    }


def persist_export_settings(collection_manager, collection: str, filename: str, settings: dict):
    """Guarda gradación y parámetros de exportación en metadata.json del archivo."""
    grade = settings.get("grade")
    if isinstance(grade, dict):
        grade = copy.deepcopy(grade)
    collection_manager.set_file_info(
        collection,
        filename,
        fps=int(settings.get("fps", DEFAULT_FPS)),
        export_grade=grade,
        export_output_name=str(settings.get("output_name", "")).strip(),
    )


def file_pixel_format(filename: str, collection_meta: dict) -> str:
    info = collection_meta.get(filename, {})
    return info.get("pixel_format", "bayer")


def count_frames(path: Path, pixel_fmt: str, w=FIXED_W, h=FIXED_H) -> int:
    fsize = path.stat().st_size
    if pixel_fmt == "qoi_rgb":
        return len(qoi_utils.build_frame_index(str(path)))
    if pixel_fmt in ("rgb", "qoi_rgb"):
        fb = w * h * 3
    else:
        fb = int(w * h * 1.5)
    return max(1, fsize // fb) if fb else 1


def render_bayer_frame_rgb(packed: bytes, w=FIXED_W, h=FIXED_H, downscale=1) -> np.ndarray:
    """Pipeline base (HQ debayer + gamma 2.2 LUT) — igual que exportación sin gradación."""
    bgr = bayer_render.render_capture_view(packed, w, h, downscale=downscale, to_bgr=True)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_frame_rgb(
    path: Path, frame_idx: int, pixel_fmt: str,
    w=FIXED_W, h=FIXED_H, downscale=2,
) -> np.ndarray | None:
    """Frame tal como se exporta sin gradación (original / panel izquierdo)."""
    try:
        if pixel_fmt == "qoi_rgb":
            index = qoi_utils.build_frame_index(str(path))
            if not index:
                return None
            idx = max(0, min(frame_idx, len(index) - 1))
            offset, fsz = index[idx]
            qoi_data = qoi_utils.read_frame_at(str(path), offset, fsz)
            rgb = qoi_utils.decode_qoi(qoi_data, w, h).copy()
        elif pixel_fmt == "rgb":
            fb = w * h * 3
            with open(path, "rb") as f:
                f.seek(frame_idx * fb)
                raw = f.read(fb)
            if len(raw) < fb:
                return None
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        else:
            fb = int(w * h * 1.5)
            with open(path, "rb") as f:
                f.seek(frame_idx * fb)
                packed = f.read(fb)
            if len(packed) < fb:
                return None
            rgb = render_bayer_frame_rgb(packed, w, h, downscale=downscale)
        if downscale > 1 and pixel_fmt in ("qoi_rgb", "rgb"):
            rgb = rgb[::downscale, ::downscale, :].copy()
        return rgb
    except Exception as e:
        print(f"WARN read_frame_rgb {path.name} #{frame_idx}: {e}")
        return None


def read_frame_linear_float(
    path: Path, frame_idx: int, pixel_fmt: str,
    w=FIXED_W, h=FIXED_H, downscale=2,
) -> np.ndarray | None:
    """Frame lineal 0–1 para la cadena de gradación (panel derecho / export con grade)."""
    try:
        if pixel_fmt == "qoi_rgb":
            index = qoi_utils.build_frame_index(str(path))
            if not index:
                return None
            idx = max(0, min(frame_idx, len(index) - 1))
            offset, fsz = index[idx]
            qoi_data = qoi_utils.read_frame_at(str(path), offset, fsz)
            rgb = qoi_utils.decode_qoi(qoi_data, w, h).copy()
            img_f = rgb.astype(np.float32) / 255.0
        elif pixel_fmt == "rgb":
            fb = w * h * 3
            with open(path, "rb") as f:
                f.seek(frame_idx * fb)
                raw = f.read(fb)
            if len(raw) < fb:
                return None
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            img_f = rgb.astype(np.float32) / 255.0
        else:
            fb = int(w * h * 1.5)
            with open(path, "rb") as f:
                f.seek(frame_idx * fb)
                packed = f.read(fb)
            if len(packed) < fb:
                return None
            bayer = bayer_render.unpack_12bit_le(packed, w, h)
            if downscale > 1:
                bayer = bayer[::downscale, ::downscale]
            rgb16 = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
            img_f = rgb16.astype(np.float32) / RAW12_MAX
        if downscale > 1 and pixel_fmt in ("qoi_rgb", "rgb"):
            img_f = img_f[::downscale, ::downscale, :].copy()
        return np.clip(img_f, 0.0, 1.0)
    except Exception as e:
        print(f"WARN read_frame_linear_float {path.name} #{frame_idx}: {e}")
        return None


def apply_export_grade(img_f: np.ndarray, settings: dict, mode: str = "COLOR") -> np.ndarray:
    """Procesa un frame float RGB exactamente como raw2video con --grade."""
    from raw2video import apply_processing_chain

    s = settings
    img = img_f.astype(np.float32).copy()
    wb = np.array([s["wb_red"], s["wb_green"], s["wb_blue"]], dtype=np.float32)
    img *= wb

    exp = float(s.get("exposure", 1.0))
    if abs(exp - 1.0) > 1e-6:
        img *= exp

    img = apply_processing_chain(
        img,
        float(s["gamma"]),
        float(s["black_level"]),
        float(s["white_level"]),
        float(s["sharp_sigma"]),
        float(s["sharp_amount"]),
        float(s["chroma_blur"]),
    )

    lift = float(s.get("shadow_lift", 0.0))
    contrast = float(s.get("contrast", 1.0))
    if lift > 0.0 or contrast != 1.0:
        rgb8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        bgr = scanner_core.apply_isp_preview_tone(bgr, 1.0, lift, contrast)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if mode == "BW":
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        img = cv2.merge((gray, gray, gray))

    return np.clip(img, 0.0, 1.0)


def geometry_is_default(settings: dict, src_w: int = FIXED_W) -> bool:
    return (
        abs(float(settings.get("rotate_deg", 0.0))) < 1e-6
        and int(settings.get("zoom_px", 0)) == 0
        and int(settings.get("pan_x", 0)) == 0
        and int(settings.get("pan_y", 0)) == 0
        and int(settings.get("output_width", src_w)) == src_w
    )


def export_frame_size(
    settings: dict | None,
    src_w: int = FIXED_W,
    src_h: int = FIXED_H,
    preview_downscale: int = 1,
) -> tuple[int, int]:
    """Dimensiones del frame tras encuadre (ancho ajustable, alto = captura)."""
    ds = max(1, int(preview_downscale))
    if not settings:
        return src_w // ds, src_h // ds
    out_w = int(settings.get("output_width", src_w))
    out_w = max(GEO_WIDTH_MIN, min(out_w, GEO_WIDTH_MAX))
    return max(2, int(round(out_w / ds))), max(2, int(round(src_h / ds)))


def geometry_has_changes(settings: dict, src_w: int = FIXED_W) -> bool:
    return not geometry_is_default(settings, src_w)


def _apply_zoom_pan(
    img: np.ndarray,
    zoom_px: int,
    pan_x: float,
    pan_y: float,
) -> np.ndarray:
    """Recorte con zoom; pan desplaza el centro de la ventana (negro fuera de imagen)."""
    h, w = img.shape[:2]
    z = max(0, int(zoom_px))
    crop_w = max(2, w - 2 * z)
    crop_h = max(2, h - 2 * z)
    x0 = int(round(w / 2.0 + float(pan_x) - crop_w / 2.0))
    y0 = int(round(h / 2.0 + float(pan_y) - crop_h / 2.0))

    if x0 >= 0 and y0 >= 0 and x0 + crop_w <= w and y0 + crop_h <= h:
        cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
    else:
        cropped = np.zeros((crop_h, crop_w, 3), dtype=img.dtype)
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(w, x0 + crop_w)
        src_y1 = min(h, y0 + crop_h)
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        copy_w = src_x1 - src_x0
        copy_h = src_y1 - src_y0
        if copy_w > 0 and copy_h > 0:
            cropped[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
                img[src_y0:src_y1, src_x0:src_x1]

    if cropped.shape[0] != h or cropped.shape[1] != w:
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return cropped


def draw_canvas_border(
    rgb: np.ndarray,
    thickness: int = CANVAS_BORDER_PX,
    color: tuple[int, int, int] = CANVAS_BORDER_COLOR,
) -> np.ndarray:
    """Trazo del borde del lienzo (preview)."""
    out = rgb.copy()
    h, w = out.shape[:2]
    t = max(1, int(thickness))
    c = np.array(color, dtype=out.dtype)
    out[:t, :, :] = c
    out[h - t:h, :, :] = c
    out[:, :t, :] = c
    out[:, w - t:w, :] = c
    return out


def _compose_output_canvas(img: np.ndarray, out_w: int) -> np.ndarray:
    """
    Ancho de salida = lienzo del video. La imagen no se escala:
    recorte centrado si el lienzo es más estrecho; bandas negras si es más ancho.
    """
    h, w = img.shape[:2]
    if out_w == w:
        return img
    if out_w < w:
        x0 = max(0, (w - out_w) // 2)
        return img[:, x0:x0 + out_w]
    canvas = np.zeros((h, out_w, 3), dtype=img.dtype)
    x_off = (out_w - w) // 2
    canvas[:, x_off:x_off + w] = img
    return canvas


def apply_frame_geometry(
    rgb: np.ndarray,
    rotate_deg: float = 0.0,
    zoom_px: int = 0,
    pan_x: float = 0,
    pan_y: float = 0,
    output_width: int | None = None,
    canvas_height: int | None = None,
) -> np.ndarray:
    """
    Gira la imagen; zoom+pan mueven la ventana de encuadre;
    ancho de salida define el lienzo del video (sin escalar la imagen).
    """
    if rgb is None or rgb.size == 0:
        return rgb
    h, w = rgb.shape[:2]
    out_w = max(2, int(output_width if output_width is not None else w))
    eff_px = float(pan_x)
    eff_py = float(pan_y)

    spatial_default = (
        abs(rotate_deg) < 1e-6 and zoom_px == 0
        and abs(eff_px) < 1e-6 and abs(eff_py) < 1e-6
    )
    if spatial_default and out_w == w:
        return rgb

    img = rgb
    if abs(rotate_deg) >= 1e-6:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotate_deg, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    if zoom_px > 0 or abs(eff_px) >= 1e-6 or abs(eff_py) >= 1e-6:
        img = _apply_zoom_pan(img, zoom_px, eff_px, eff_py)

    return _compose_output_canvas(img, out_w)


def apply_frame_geometry_from_settings(
    rgb: np.ndarray, settings: dict, preview_downscale: int = 1,
) -> np.ndarray:
    ds = max(1, int(preview_downscale))
    out_w_full = int(settings.get("output_width", FIXED_W))
    out_w = max(2, int(round(out_w_full / ds)))
    pan_scale = 1.0 / ds
    return apply_frame_geometry(
        rgb,
        rotate_deg=float(settings.get("rotate_deg", 0.0)),
        zoom_px=int(round(int(settings.get("zoom_px", 0)) / ds)),
        pan_x=float(int(settings.get("pan_x", 0)) * pan_scale),
        pan_y=float(int(settings.get("pan_y", 0)) * pan_scale),
        output_width=out_w,
    )


def geometry_ffmpeg_filter(settings: dict, w: int, h: int) -> str | None:
    """Cadena -vf equivalente (referencia / log; el frame ya va transformado por stdin)."""
    out_w = max(GEO_WIDTH_MIN, min(int(settings.get("output_width", w)), GEO_WIDTH_MAX))
    if geometry_is_default(settings, w) and out_w == w:
        return None
    import math
    rot = float(settings.get("rotate_deg", 0.0))
    z = max(0, int(settings.get("zoom_px", 0)))
    px = int(settings.get("pan_x", 0))
    py = int(settings.get("pan_y", 0))
    parts = []
    if abs(rot) >= 1e-6:
        parts.append(f"rotate={rot * math.pi / 180.0}:fillcolor=black:ow=iw:oh=ih")
    if z > 0 or px != 0 or py != 0:
        crop_w = max(2, w - 2 * z)
        crop_h = max(2, h - 2 * z)
        cx = w / 2.0 + px
        cy = h / 2.0 + py
        x0 = max(0, min(int(round(cx - crop_w / 2.0)), w - crop_w))
        y0 = max(0, min(int(round(cy - crop_h / 2.0)), h - crop_h))
        parts.append(f"crop={crop_w}:{crop_h}:{x0}:{y0}")
        parts.append(f"scale={w}:{h}")
    if out_w < w:
        parts.append(f"crop={out_w}:{h}:{(w - out_w) // 2}:0")
    elif out_w > w:
        parts.append(f"pad={out_w}:{h}:{(out_w - w) // 2}:0:black")
    return ",".join(parts) if parts else None


def linear_to_display_rgb(img_f: np.ndarray) -> np.ndarray:
    return (np.clip(img_f, 0, 1) * 255).astype(np.uint8)


def rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def make_thumbnail(path: Path, pixel_fmt: str) -> QPixmap | None:
    n = count_frames(path, pixel_fmt)
    idx = min(100, max(0, n - 1))
    rgb = read_frame_rgb(path, idx, pixel_fmt, downscale=8)
    if rgb is None:
        return None
    small = cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA)
    return rgb_to_qpixmap(small)


# Tooltips basados en Phoenix 8.9 MP TRM / PHX081S Camera Features (docs/)
_GRADE_TIPS = {
    "gamma": (
        "Nodo Gamma (GenICam): Y = X^Gamma sobre intensidad normalizada [0–1]. "
        "Gamma = 1 sin cambio; 0,2–1 aclara; 1–2 oscurece (rango cámara 0,2–2). "
        "Aquí se aplica en software sobre el frame antes de codificar video."
    ),
    "black_level": (
        "Nodo BlackLevel (GenICam): controla el nivel de negro como valor absoluto "
        "sobre la señal raw 12-bit (0–4095). Sustrae offset antes de escalar; "
        "equivalente al BlackLevel de la cámara Phoenix PHX081S."
    ),
    "white_level": (
        "Tope de blancos en escala 12-bit (2048–4095) para la ventana lineal previa a gamma. "
        "Bajarlo estira la señal hacia blanco (como un techo más bajo); no sustituye Exposición."
    ),
    "exposure": (
        "Ganancia lineal de exposición (100 = neutro, 200 = 2×) aplicada por igual a R, G y B "
        "tras el balance de blancos. Equivalente a Gain digital en post (GenICam); "
        "sube la iluminación global sin alterar el color como el WB."
    ),
    "shadow_lift": (
        "Ajuste solo de software (post-ISP): eleva las sombras sin mover los blancos, "
        "útil cuando el material capturado queda muy cerrado en negro."
    ),
    "contrast": (
        "Ajuste solo de software: escala la señal alrededor del punto medio (100 = neutro). "
        "No modifica la captura Bayer/ISP, solo la exportación gradada."
    ),
    "chroma_blur": (
        "Reducción de ruido en canales Cr/Cb (espacio YCrCb) antes de codificar. "
        "Suaviza manchas de color sin tocar la nitidez de luminancia."
    ),
    "sharp_sigma": (
        "Radio (sigma) de la máscara de enfoque tipo Unsharp sobre el canal de luminancia. "
        "0 = desactivado. Valores altos afectan detalle más grueso."
    ),
    "sharp_amount": (
        "Intensidad del enfoque Unsharp en luminancia. "
        "Análogo al efecto de SharpeningAmount del ISP, aplicado en exportación."
    ),
    "wb_red": (
        "Nodo BalanceRatio Red (GenICam): factor de amplificación absoluto aplicado "
        "al canal rojo. 1,00 = neutro; >1 enriquece rojos."
    ),
    "wb_green": (
        "Nodo BalanceRatio Green (GenICam): factor de amplificación del canal verde. "
        "1,00 = neutro; referencia habitual del balance de blancos."
    ),
    "wb_blue": (
        "Nodo BalanceRatio Blue (GenICam): factor de amplificación del canal azul. "
        "1,00 = neutro; >1 enriquece azules."
    ),
    "rotate_deg": (
        f"Giro en grados (paso {GEO_ROTATE_STEP}°) aplicado solo a la imagen, "
        "no al lienzo del video. Negativo = sentido horario."
    ),
    "zoom_px": (
        "Zoom sobre la imagen: píxeles recortados en cada borde (ancho/alto). "
        "Mayor valor = más acercamiento; la imagen se reescala al tamaño de captura."
    ),
    "pan_x": (
        f"Desplaza la ventana de encuadre horizontalmente (±{GEO_PAN_X_MAX} px, "
        f"≈35% del ancho). Tras zoom, revela lo oculto fuera del cuadro."
    ),
    "pan_y": (
        f"Desplaza la ventana de encuadre verticalmente (±{GEO_PAN_Y_MAX} px, "
        f"≈35% del alto). Tras zoom, revela lo oculto fuera del cuadro."
    ),
    "output_width": (
        f"Ancho del lienzo de video ({GEO_WIDTH_MIN}-{GEO_WIDTH_MAX} px). "
        f"La imagen no se estira: más estrecho recorta los lados; más ancho muestra "
        f"más contenido (o bandas negras si ya se veía todo el ancho de captura {FIXED_W} px)."
    ),
}


def _info_icon(tooltip: str) -> QLabel:
    icon = QLabel("ⓘ")
    icon.setToolTip(tooltip)
    icon.setFixedSize(18, 18)
    icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icon.setStyleSheet(
        "color: #6eb5ff; font-size: 11px; font-weight: bold; border: none; padding: 0;"
    )
    icon.setCursor(Qt.CursorShape.WhatsThisCursor)
    return icon


def _make_grade_card(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    card = QGroupBox(title)
    card.setStyleSheet("""
        QGroupBox {
            font-size: 9pt;
            font-weight: bold;
            color: #aaa;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
    """)
    lay = QVBoxLayout(card)
    lay.setSpacing(4)
    lay.setContentsMargins(8, 4, 8, 8)
    return card, lay


def _add_compact_slider(
    parent_layout, title: str, tip_key: str, lo: int, hi: int, default: int, fmt_fn,
):
    block = QVBoxLayout()
    block.setSpacing(0)
    block.setContentsMargins(0, 0, 0, 0)

    hdr = QHBoxLayout()
    hdr.setSpacing(4)
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("color: #ccc; font-size: 9pt; border: none;")
    hdr.addWidget(title_lbl)
    hdr.addWidget(_info_icon(_GRADE_TIPS[tip_key]))
    hdr.addStretch()
    val_lbl = QLabel(fmt_fn(default))
    val_lbl.setMinimumWidth(40)
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    val_lbl.setStyleSheet("color: #888; font-size: 9pt; border: none;")
    hdr.addWidget(val_lbl)
    block.addLayout(hdr)

    sld = QSlider(Qt.Orientation.Horizontal)
    sld.setRange(lo, hi)
    sld.setValue(default)
    sld.setFixedHeight(16)
    block.addWidget(sld)
    parent_layout.addLayout(block)
    return sld, val_lbl, fmt_fn


def _add_spin_slider_row(
    grid: QGridLayout, row: int, col: int,
    title: str, tip_key: str,
    *,
    is_float: bool = False,
    lo: float = 0, hi: float = 0, step: float = 1.0, default: float = 0,
    keyboard_tracking: bool = True,
) -> tuple[QWidget, QSlider, QDoubleSpinBox | QSpinBox]:
    """Fila compacta: título + ⓘ + spinbox + slider (sincronizados)."""
    cell = QWidget()
    lay = QVBoxLayout(cell)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)

    hdr = QHBoxLayout()
    hdr.setSpacing(4)
    lbl = QLabel(title)
    lbl.setStyleSheet("color: #ccc; font-size: 9pt; border: none;")
    hdr.addWidget(lbl)
    hdr.addWidget(_info_icon(_GRADE_TIPS[tip_key]))
    hdr.addStretch()
    lay.addLayout(hdr)

    row_ctrl = QHBoxLayout()
    row_ctrl.setSpacing(6)
    if is_float:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(1)
        spin.setValue(float(default))
        spin.setFixedWidth(72)
        spin.setKeyboardTracking(keyboard_tracking)
        sld_lo, sld_hi = int(lo / step), int(hi / step)
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(sld_lo, sld_hi)
        sld.setValue(int(round(default / step)))

        def spin_to_sld(v):
            sld.blockSignals(True)
            sld.setValue(int(round(v / step)))
            sld.blockSignals(False)

        def sld_to_spin(v):
            spin.blockSignals(True)
            spin.setValue(round(v * step, 1))
            spin.blockSignals(False)

        spin.valueChanged.connect(spin_to_sld)
        sld.valueChanged.connect(sld_to_spin)
    else:
        spin = QSpinBox()
        spin.setRange(int(lo), int(hi))
        spin.setSingleStep(int(step))
        spin.setValue(int(default))
        spin.setFixedWidth(72)
        spin.setKeyboardTracking(keyboard_tracking)
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(int(lo), int(hi))
        sld.setValue(int(default))

        def spin_to_sld_i(v):
            sld.blockSignals(True)
            sld.setValue(v)
            sld.blockSignals(False)

        def sld_to_spin_i(v):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)

        spin.valueChanged.connect(spin_to_sld_i)
        sld.valueChanged.connect(sld_to_spin_i)

    row_ctrl.addWidget(spin)
    row_ctrl.addWidget(sld, 1)
    lay.addLayout(row_ctrl)
    grid.addWidget(cell, row, col)
    return cell, sld, spin


class PreviewZoomLabel(QLabel):
    """Preview con zoom visual (rueda del mouse); no modifica settings de exportación."""

    ZOOM_MIN = 1.0
    ZOOM_MAX = 12.0
    ZOOM_STEP = 1.12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setStyleSheet("background: #111; border: 1px solid #333;")
        self.setToolTip("Rueda del mouse: acercar / alejar (solo vista previa)")
        self._base_pixmap: QPixmap | None = None
        self._view_zoom = 1.0

    def set_base_pixmap(self, pixmap: QPixmap):
        self._base_pixmap = pixmap
        self._update_display()

    def reset_view_zoom(self):
        self._view_zoom = 1.0
        self._update_display()

    def _update_display(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            self.clear()
            return
        lw, lh = max(1, self.width()), max(1, self.height())
        if self._view_zoom <= 1.0:
            fit = self._base_pixmap.scaled(
                lw, lh,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(fit)
            return
        fit = self._base_pixmap.scaled(
            lw, lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        zw = max(1, int(fit.width() * self._view_zoom))
        zh = max(1, int(fit.height() * self._view_zoom))
        zoomed = self._base_pixmap.scaled(
            zw, zh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(zoomed)

    def wheelEvent(self, event: QWheelEvent):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
        self._view_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._view_zoom * factor))
        self._update_display()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class ExportGradeDialog(QDialog):
    """Preview A/B en vivo: original (izq) vs exportación gradada (der)."""

    PREVIEW_DOWNSCALE = 2

    def __init__(
        self,
        parent,
        path: Path,
        pixel_fmt: str,
        mode: str,
        grade_settings: dict | None = None,
    ):
        super().__init__(parent)
        self.path = path
        self.pixel_fmt = pixel_fmt
        self.mode = mode
        self.settings = copy.deepcopy(grade_settings or default_grade_settings())
        self._frame_count = count_frames(path, pixel_fmt)
        self._frame_idx = min(100, max(0, self._frame_count - 1))

        self.setWindowTitle(f"Gradación — {path.name}")
        self.setMinimumSize(960, 720)
        self.resize(1020, 780)

        outer = QVBoxLayout(self)

        info = QLabel(
            "Izquierda: captura original (referencia). "
            "Derecha: lienzo de exportación con gradación; borde cian = tamaño del cuadro de video. "
            "Rueda del mouse sobre cualquier preview: acercar / alejar (solo inspección)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #999; font-size: 9pt;")
        outer.addWidget(info)

        prev_row = QHBoxLayout()
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Original"))
        self.lbl_original = PreviewZoomLabel()
        left_col.addWidget(self.lbl_original, 1)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Procesado (exportación)"))
        self.lbl_processed = PreviewZoomLabel()
        right_col.addWidget(self.lbl_processed, 1)

        prev_row.addLayout(left_col, 1)
        prev_row.addLayout(right_col, 1)
        outer.addLayout(prev_row, 1)

        scrub_row = QHBoxLayout()
        self.lbl_frame = QLabel()
        self.sld_frame = QSlider(Qt.Orientation.Horizontal)
        self.sld_frame.setRange(0, max(0, self._frame_count - 1))
        self.sld_frame.setValue(self._frame_idx)
        self.sld_frame.valueChanged.connect(self._on_frame_changed)
        scrub_row.addWidget(QLabel("Cuadro:"))
        scrub_row.addWidget(self.sld_frame, 1)
        scrub_row.addWidget(self.lbl_frame)
        outer.addLayout(scrub_row)

        geom_card, geom_lay = _make_grade_card(
            f"Encuadre — lienzo hasta {GEO_WIDTH_MAX} px (captura {FIXED_W}×{FIXED_H})",
        )
        geom_grid = QGridLayout()
        geom_grid.setHorizontalSpacing(8)
        geom_grid.setVerticalSpacing(4)

        _, self.sld_rotate, self.spin_rotate = _add_spin_slider_row(
            geom_grid, 0, 0, "Giro °", "rotate_deg",
            is_float=True, lo=GEO_ROTATE_MIN, hi=GEO_ROTATE_MAX,
            step=GEO_ROTATE_STEP, default=0.0, keyboard_tracking=False,
        )
        _, self.sld_zoom, self.spin_zoom = _add_spin_slider_row(
            geom_grid, 0, 1, "Zoom px", "zoom_px",
            lo=0, hi=GEO_ZOOM_MAX, step=1, default=0, keyboard_tracking=False,
        )
        _, self.sld_pan_x, self.spin_pan_x = _add_spin_slider_row(
            geom_grid, 0, 2, "Pan X px", "pan_x",
            lo=-GEO_PAN_X_MAX, hi=GEO_PAN_X_MAX, step=1, default=0,
            keyboard_tracking=False,
        )
        _, self.sld_pan_y, self.spin_pan_y = _add_spin_slider_row(
            geom_grid, 0, 3, "Pan Y px", "pan_y",
            lo=-GEO_PAN_Y_MAX, hi=GEO_PAN_Y_MAX, step=1, default=0,
            keyboard_tracking=False,
        )
        _, self.sld_out_w, self.spin_out_w = _add_spin_slider_row(
            geom_grid, 1, 0, "Ancho salida px", "output_width",
            lo=GEO_WIDTH_MIN, hi=GEO_WIDTH_MAX, step=1, default=FIXED_W,
            keyboard_tracking=False,
        )
        aspect_cell = QWidget()
        aspect_lay = QVBoxLayout(aspect_cell)
        aspect_lay.setContentsMargins(0, 8, 0, 0)
        hdr_aspect = QHBoxLayout()
        lbl_aspect_title = QLabel("Proporción")
        lbl_aspect_title.setStyleSheet("color: #ccc; font-size: 9pt; border: none;")
        hdr_aspect.addWidget(lbl_aspect_title)
        hdr_aspect.addWidget(_info_icon(
            "Relación ancho:alto del video exportado según el ancho de salida y "
            f"el alto fijo de captura ({FIXED_H} px)."
        ))
        hdr_aspect.addStretch()
        aspect_lay.addLayout(hdr_aspect)
        self.lbl_aspect = QLabel()
        self.lbl_aspect.setStyleSheet("color: #6eb5ff; font-size: 10pt; font-weight: bold; border: none;")
        aspect_lay.addWidget(self.lbl_aspect)
        aspect_lay.addStretch()
        geom_grid.addWidget(aspect_cell, 1, 1, 1, 3)
        geom_grid.setColumnStretch(0, 1)
        geom_grid.setColumnStretch(1, 1)
        geom_grid.setColumnStretch(2, 1)
        geom_grid.setColumnStretch(3, 1)
        geom_lay.addLayout(geom_grid)
        outer.addWidget(geom_card)

        cards_grid = QGridLayout()
        cards_grid.setHorizontalSpacing(10)
        cards_grid.setVerticalSpacing(8)

        card_gamma, lay_gamma = _make_grade_card("Gamma y niveles")
        self.sld_gamma, self.lbl_gamma, self._fmt_gamma = _add_compact_slider(
            lay_gamma, "Gamma", "gamma", 50, 300, 140, lambda v: f"{v / 100:.2f}",
        )
        self.sld_black, self.lbl_black, self._fmt_black = _add_compact_slider(
            lay_gamma, "Black level", "black_level", 0, 512, 0, lambda v: str(v),
        )
        self.sld_white, self.lbl_white, self._fmt_white = _add_compact_slider(
            lay_gamma, "White level", "white_level", 2048, 4095, 4095, lambda v: str(v),
        )
        cards_grid.addWidget(card_gamma, 0, 0)

        card_tone, lay_tone = _make_grade_card("Exposición y tono")
        self.sld_exposure, self.lbl_exposure, self._fmt_exposure = _add_compact_slider(
            lay_tone, "Exposición", "exposure", 25, 400, 100, lambda v: f"{v / 100:.2f}×",
        )
        self.sld_lift, self.lbl_lift, self._fmt_lift = _add_compact_slider(
            lay_tone, "Lift sombras", "shadow_lift", 0, 40, 0, lambda v: f"{v}%",
        )
        self.sld_contrast, self.lbl_contrast, self._fmt_contrast = _add_compact_slider(
            lay_tone, "Contraste", "contrast", 80, 130, 100, lambda v: f"{v}%",
        )
        lay_tone.addStretch()
        cards_grid.addWidget(card_tone, 0, 1)

        card_wb, lay_wb = _make_grade_card("Balance de blancos")
        self.sld_wb_r, self.lbl_wb_r, self._fmt_wb_r = _add_compact_slider(
            lay_wb, "Canal R", "wb_red", 50, 400, 100, lambda v: f"{v / 100:.2f}",
        )
        self.sld_wb_g, self.lbl_wb_g, self._fmt_wb_g = _add_compact_slider(
            lay_wb, "Canal G", "wb_green", 50, 400, 100, lambda v: f"{v / 100:.2f}",
        )
        self.sld_wb_b, self.lbl_wb_b, self._fmt_wb_b = _add_compact_slider(
            lay_wb, "Canal B", "wb_blue", 50, 400, 100, lambda v: f"{v / 100:.2f}",
        )
        cards_grid.addWidget(card_wb, 0, 2)

        card_sharp, lay_sharp = _make_grade_card("Nitidez")
        self.sld_sharp_s, self.lbl_sharp_s, self._fmt_sharp_s = _add_compact_slider(
            lay_sharp, "Sigma", "sharp_sigma", 0, 40, 20, lambda v: f"{v / 10:.1f}",
        )
        self.sld_sharp_a, self.lbl_sharp_a, self._fmt_sharp_a = _add_compact_slider(
            lay_sharp, "Amount", "sharp_amount", 0, 50, 25, lambda v: f"{v / 10:.1f}",
        )
        lay_sharp.addStretch()
        cards_grid.addWidget(card_sharp, 1, 0)

        card_nr, lay_nr = _make_grade_card("Ruido de color")
        self.sld_chroma, self.lbl_chroma, self._fmt_chroma = _add_compact_slider(
            lay_nr, "Blur croma", "chroma_blur", 0, 60, 30, lambda v: f"{v / 10:.1f}",
        )
        lay_nr.addStretch()
        cards_grid.addWidget(card_nr, 1, 1)

        cards_grid.setColumnStretch(0, 1)
        cards_grid.setColumnStretch(1, 1)
        cards_grid.setColumnStretch(2, 1)
        outer.addLayout(cards_grid)

        self._sliders = [
            self.sld_gamma, self.sld_black, self.sld_white,
            self.sld_exposure, self.sld_lift, self.sld_contrast,
            self.sld_chroma, self.sld_sharp_s, self.sld_sharp_a,
            self.sld_wb_r, self.sld_wb_g, self.sld_wb_b,
        ]
        for sld in self._sliders:
            sld.valueChanged.connect(self._on_slider_change)

        self._geom_spins = (
            self.spin_rotate, self.spin_zoom, self.spin_pan_x, self.spin_pan_y, self.spin_out_w,
        )
        self._geom_sliders = (
            self.sld_rotate, self.sld_zoom, self.sld_pan_x, self.sld_pan_y, self.sld_out_w,
        )
        for spin in self._geom_spins:
            spin.editingFinished.connect(self._on_geometry_change)
        for sld in self._geom_sliders:
            sld.valueChanged.connect(self._on_geometry_change)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Restablecer")
        btn_reset.setAutoDefault(False)
        btn_reset.setDefault(False)
        btn_reset.clicked.connect(self._reset_defaults)
        btn_disable = QPushButton("Desactivar gradación")
        btn_disable.setAutoDefault(False)
        btn_disable.setDefault(False)
        btn_disable.clicked.connect(self._disable_grade)
        btn_ok = QPushButton("Aplicar gradación")
        btn_ok.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold;")
        btn_ok.setAutoDefault(True)
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self._apply_and_close)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setAutoDefault(False)
        btn_cancel.setDefault(False)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_disable)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        outer.addLayout(btn_row)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._load_sliders_from_settings()
        self._update_aspect_label()
        self._refresh_preview()

    def get_settings(self) -> dict:
        return copy.deepcopy(self.settings)

    def _load_sliders_from_settings(self):
        s = self.settings
        mapping = (
            (self.sld_gamma, int(round(s["gamma"] * 100))),
            (self.sld_black, int(round(s["black_level"]))),
            (self.sld_white, int(round(s["white_level"]))),
            (self.sld_exposure, int(round(float(s.get("exposure", 1.0)) * 100))),
            (self.sld_lift, int(round(s.get("shadow_lift", 0) * 200))),
            (self.sld_contrast, int(round(s.get("contrast", 1.0) * 100))),
            (self.sld_chroma, int(round(s["chroma_blur"] * 10))),
            (self.sld_sharp_s, int(round(s["sharp_sigma"] * 10))),
            (self.sld_sharp_a, int(round(s["sharp_amount"] * 10))),
            (self.sld_wb_r, int(round(s["wb_red"] * 100))),
            (self.sld_wb_g, int(round(s["wb_green"] * 100))),
            (self.sld_wb_b, int(round(s["wb_blue"] * 100))),
        )
        for sld, val in mapping:
            sld.blockSignals(True)
            sld.setValue(val)
            sld.blockSignals(False)
        self._refresh_slider_labels()
        self._load_geometry_from_settings()

    def _load_geometry_from_settings(self):
        s = self.settings
        specs = (
            (self.spin_rotate, self.sld_rotate, float(s.get("rotate_deg", 0.0)), GEO_ROTATE_STEP),
            (self.spin_zoom, self.sld_zoom, int(s.get("zoom_px", 0)), 1.0),
            (self.spin_pan_x, self.sld_pan_x, int(s.get("pan_x", 0)), 1.0),
            (self.spin_pan_y, self.sld_pan_y, int(s.get("pan_y", 0)), 1.0),
            (self.spin_out_w, self.sld_out_w, int(s.get("output_width", FIXED_W)), 1.0),
        )
        for spin, sld, val, step in specs:
            spin.blockSignals(True)
            sld.blockSignals(True)
            spin.setValue(val)
            if isinstance(spin, QDoubleSpinBox):
                sld.setValue(int(round(val / step)))
            else:
                sld.setValue(int(val))
            spin.blockSignals(False)
            sld.blockSignals(False)

    def _refresh_slider_labels(self):
        self.lbl_gamma.setText(self._fmt_gamma(self.sld_gamma.value()))
        self.lbl_black.setText(self._fmt_black(self.sld_black.value()))
        self.lbl_white.setText(self._fmt_white(self.sld_white.value()))
        self.lbl_exposure.setText(self._fmt_exposure(self.sld_exposure.value()))
        self.lbl_lift.setText(self._fmt_lift(self.sld_lift.value()))
        self.lbl_contrast.setText(self._fmt_contrast(self.sld_contrast.value()))
        self.lbl_chroma.setText(self._fmt_chroma(self.sld_chroma.value()))
        self.lbl_sharp_s.setText(self._fmt_sharp_s(self.sld_sharp_s.value()))
        self.lbl_sharp_a.setText(self._fmt_sharp_a(self.sld_sharp_a.value()))
        self.lbl_wb_r.setText(self._fmt_wb_r(self.sld_wb_r.value()))
        self.lbl_wb_g.setText(self._fmt_wb_g(self.sld_wb_g.value()))
        self.lbl_wb_b.setText(self._fmt_wb_b(self.sld_wb_b.value()))

    def _collect_settings(self) -> dict:
        return {
            "enabled": self.settings.get("enabled", False),
            "gamma": self.sld_gamma.value() / 100.0,
            "exposure": self.sld_exposure.value() / 100.0,
            "black_level": float(self.sld_black.value()),
            "white_level": float(self.sld_white.value()),
            "shadow_lift": self.sld_lift.value() / 200.0,
            "contrast": self.sld_contrast.value() / 100.0,
            "chroma_blur": self.sld_chroma.value() / 10.0,
            "sharp_sigma": self.sld_sharp_s.value() / 10.0,
            "sharp_amount": self.sld_sharp_a.value() / 10.0,
            "wb_red": self.sld_wb_r.value() / 100.0,
            "wb_green": self.sld_wb_g.value() / 100.0,
            "wb_blue": self.sld_wb_b.value() / 100.0,
            "rotate_deg": float(self.spin_rotate.value()),
            "zoom_px": int(self.spin_zoom.value()),
            "pan_x": int(self.spin_pan_x.value()),
            "pan_y": int(self.spin_pan_y.value()),
            "output_width": int(self.spin_out_w.value()),
        }

    def _update_aspect_label(self):
        ow = int(self.spin_out_w.value())
        ratio = ow / FIXED_H
        self.lbl_aspect.setText(
            f"{ow} x {FIXED_H} px  |  {ratio:.4f}:1  ({ow / FIXED_W * 100:.1f}% vs captura)"
        )

    def _on_geometry_change(self, *_args):
        self._update_aspect_label()
        self.settings = self._collect_settings()
        self._preview_timer.start()

    def _on_slider_change(self, *_args):
        self._refresh_slider_labels()
        self.settings = self._collect_settings()
        self._preview_timer.start()

    def _on_frame_changed(self, value: int):
        self._frame_idx = value
        self.lbl_frame.setText(f"{value + 1} / {self._frame_count}")
        self.lbl_original.reset_view_zoom()
        self.lbl_processed.reset_view_zoom()
        self._preview_timer.start()

    def _set_preview_pixmap(self, label: PreviewZoomLabel, rgb: np.ndarray):
        label.set_base_pixmap(rgb_to_qpixmap(rgb))

    def _refresh_preview(self):
        self.lbl_frame.setText(f"{self._frame_idx + 1} / {self._frame_count}")
        ds = self.PREVIEW_DOWNSCALE
        original = read_frame_rgb(
            self.path, self._frame_idx, self.pixel_fmt, downscale=ds,
        )
        if original is None:
            return
        settings = self._collect_settings()
        self._set_preview_pixmap(self.lbl_original, original)

        linear = read_frame_linear_float(
            self.path, self._frame_idx, self.pixel_fmt, downscale=ds,
        )
        if linear is None:
            return
        processed_f = apply_export_grade(linear, settings, self.mode)
        processed = linear_to_display_rgb(processed_f)
        processed = apply_frame_geometry_from_settings(processed, settings, ds)
        border_t = max(1, int(round(CANVAS_BORDER_PX / ds)))
        processed = draw_canvas_border(processed, thickness=border_t)
        self._set_preview_pixmap(self.lbl_processed, processed)

    def _reset_defaults(self):
        defaults = default_grade_settings()
        defaults["enabled"] = self.settings.get("enabled", False)
        self.settings = defaults
        self._load_sliders_from_settings()
        self._preview_timer.start()

    def _disable_grade(self):
        self.settings = default_grade_settings()
        self.settings["enabled"] = False
        self.accept()

    def _apply_and_close(self):
        self.settings = self._collect_settings()
        self.settings["enabled"] = True
        self.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._preview_timer.start()


class ExportConfirmDialog(QDialog):
    """Confirmación previa a exportación: miniatura, nombre de salida, FPS y gradación opcional."""

    def __init__(
        self,
        parent,
        filenames: list[str],
        root: str,
        collection: str,
        collection_meta: dict,
        fmt: str,
    ):
        super().__init__(parent)
        self.root = Path(root)
        self.collection = collection
        self.collection_meta = collection_meta or {}
        self.filenames = list(filenames)
        self.fmt = fmt
        self._output_edits: dict[str, QLineEdit] = {}
        self._fps_spins: dict[str, QSpinBox] = {}
        self._grade_settings: dict[str, dict] = {}
        self._grade_labels: dict[str, QLabel] = {}
        self._test_checks: dict[str, QCheckBox] = {}
        self._show_grade = fmt not in ("tiff_seq", "dng", "tif")

        self.setWindowTitle("Confirmar exportación")
        self.setMinimumSize(920, 480)
        self.resize(980, 580)

        layout = QVBoxLayout(self)
        if self._show_grade:
            info_text = (
                "Revisa el nombre de salida y los FPS de cada archivo. "
                "Por defecto se exporta con el ISP de la cámara (Arena), sin gradación extra. "
                "Usa <b>Ajustar gradación…</b> solo si quieres gamma, black levels u otros ajustes de tono."
            )
        else:
            info_text = (
                "Revisa el nombre de salida y los FPS de cada archivo antes de exportar. "
                "La imagen se procesa con el ISP de la cámara (Arena), sin ajustes extra de tono."
            )
        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 9pt;")
        layout.addWidget(info)

        col_count = 6 if self._show_grade else 4
        headers = ["", "Archivo origen", "Nombre de salida", "FPS"]
        if self._show_grade:
            headers.extend(["Gradación", "Prueba"])
        self.table = QTableWidget(len(self.filenames), col_count)
        self.table.setHorizontalHeaderLabels(headers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        if self._show_grade:
            hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
            hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(3, 72)
        if self._show_grade:
            self.table.setColumnWidth(4, 130)
            self.table.setColumnWidth(5, 72)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setIconSize(QSize(160, 120))

        for row, fn in enumerate(self.filenames):
            self.table.setRowHeight(row, 128)
            path = self.root / self.collection / fn
            pf = file_pixel_format(fn, self.collection_meta)
            thumb = make_thumbnail(path, pf)
            thumb_item = QTableWidgetItem()
            thumb_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if thumb is not None:
                thumb_item.setIcon(QIcon(thumb))
            self.table.setItem(row, 0, thumb_item)

            src_item = QTableWidgetItem(fn)
            src_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, src_item)

            defaults = default_settings_for_file(fn, self.collection_meta, self.fmt)
            out_edit = QLineEdit(defaults["output_name"])
            out_edit.setPlaceholderText("Nombre de salida")
            self._output_edits[fn] = out_edit
            self.table.setCellWidget(row, 2, out_edit)

            fps_spin = QSpinBox()
            fps_spin.setRange(1, 120)
            fps_spin.setValue(int(defaults["fps"]))
            self._fps_spins[fn] = fps_spin
            fps_wrap = QWidget()
            fps_lay = QHBoxLayout(fps_wrap)
            fps_lay.setContentsMargins(4, 0, 4, 0)
            fps_lay.addWidget(fps_spin)
            self.table.setCellWidget(row, 3, fps_wrap)

            self._grade_settings[fn] = defaults["grade"]
            if self._show_grade:
                grade_wrap = QWidget()
                grade_lay = QVBoxLayout(grade_wrap)
                grade_lay.setContentsMargins(2, 4, 2, 4)
                lbl_grade = QLabel(self._grade_status_text(defaults["grade"]))
                lbl_grade.setStyleSheet("color: #888; font-size: 8pt;")
                self._grade_labels[fn] = lbl_grade
                btn_grade = QPushButton("Ajustar…")
                btn_grade.clicked.connect(lambda _checked=False, f=fn: self._open_grade_dialog(f))
                grade_lay.addWidget(lbl_grade)
                grade_lay.addWidget(btn_grade)
                self.table.setCellWidget(row, 4, grade_wrap)

                test_wrap = QWidget()
                test_lay = QHBoxLayout(test_wrap)
                test_lay.setContentsMargins(4, 0, 4, 0)
                test_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
                chk_test = QCheckBox()
                chk_test.setToolTip(
                    f"Exportar {TEST_EXPORT_FRAME_COUNT} cuadros "
                    f"(desde el #{TEST_EXPORT_SKIP_FRAMES + 1}, omitiendo los primeros {TEST_EXPORT_SKIP_FRAMES})"
                )
                self._test_checks[fn] = chk_test
                test_lay.addWidget(chk_test)
                self.table.setCellWidget(row, 5, test_wrap)

        layout.addWidget(self.table, 1)

        if self._show_grade:
            copy_row = QHBoxLayout()
            self.chk_copy_grade = QCheckBox("Copiar gradación del primer archivo a todos al ajustar")
            copy_row.addWidget(self.chk_copy_grade)
            self.chk_copy_test = QCheckBox(
                f"Marcar prueba ({TEST_EXPORT_FRAME_COUNT} cuadros desde #{TEST_EXPORT_SKIP_FRAMES + 1}) en todos"
            )
            copy_row.addWidget(self.chk_copy_test)
            copy_row.addStretch()
            layout.addLayout(copy_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_export = QPushButton("Exportar")
        btn_export.setStyleSheet(
            "background-color: #0078d7; color: white; font-weight: bold; padding: 8px 20px;"
        )
        btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_export)
        layout.addLayout(btn_row)

    def _grade_status_text(self, grade: dict) -> str:
        if grade.get("enabled"):
            exp = float(grade.get("exposure", 1.0))
            extra = f", exp={exp:.2f}×" if abs(exp - 1.0) > 1e-3 else ""
            return f"Activa (γ={grade['gamma']:.2f}{extra})"
        if geometry_has_changes(grade):
            return "Encuadre guardado"
        return "Sin ajustes"

    def _file_mode(self, filename: str) -> str:
        info = self.collection_meta.get(filename, {})
        return "BW" if "Blanco" in info.get("type", "Color") else "COLOR"

    def _open_grade_dialog(self, filename: str):
        path = self.root / self.collection / filename
        pf = file_pixel_format(filename, self.collection_meta)
        dlg = ExportGradeDialog(
            self, path, pf, self._file_mode(filename),
            self._grade_settings.get(filename),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_grade = dlg.get_settings()
        self._grade_settings[filename] = new_grade
        self._grade_labels[filename].setText(self._grade_status_text(new_grade))
        if getattr(self, "chk_copy_grade", None) and self.chk_copy_grade.isChecked():
            for fn in self.filenames:
                if fn == filename:
                    continue
                self._grade_settings[fn] = copy.deepcopy(new_grade)
                self._grade_labels[fn].setText(self._grade_status_text(new_grade))

    def _on_export(self):
        from PyQt6.QtWidgets import QMessageBox
        if getattr(self, "chk_copy_test", None) and self.chk_copy_test.isChecked():
            first_fn = self.filenames[0]
            state = self._test_checks[first_fn].isChecked()
            for fn in self.filenames:
                self._test_checks[fn].setChecked(state)
        for fn in self.filenames:
            name = self._output_edits[fn].text().strip()
            if not name:
                QMessageBox.warning(self, "Nombre vacío", f"Indica un nombre de salida para «{fn}».")
                return
        self.accept()

    def get_all_settings(self) -> dict[str, dict]:
        out = {}
        for fn in self.filenames:
            entry = {
                "output_name": self._output_edits[fn].text().strip(),
                "fps": int(self._fps_spins[fn].value()),
                "grade": copy.deepcopy(self._grade_settings.get(fn, default_grade_settings())),
            }
            if fn in self._test_checks:
                entry["test_export"] = self._test_checks[fn].isChecked()
            out[fn] = entry
        return out
