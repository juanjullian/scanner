"""Pruebas del diálogo de exportación: color de la miniatura, borradores y tiempo."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import bayer_render
from export_session import (
    draft_for_file,
    estimate_remaining_seconds,
    format_duration,
    format_options,
    parse_export_speed,
)


def _pack(img):
    flat = img.reshape(-1)
    p0 = flat[0::2].astype(np.uint16)
    p1 = flat[1::2].astype(np.uint16)
    packed = np.empty(p0.size * 3, dtype=np.uint8)
    packed[0::3] = (p0 & 0xFF).astype(np.uint8)
    packed[1::3] = (((p0 >> 8) & 0x0F) | ((p1 & 0x0F) << 4)).astype(np.uint8)
    packed[2::3] = ((p1 >> 4) & 0xFF).astype(np.uint8)
    return packed.tobytes()


def test_thumb_is_color():
    h, w = 32, 32
    mosaic = np.zeros((h, w), dtype=np.uint16)
    mosaic[0::2, 0::2] = 200
    mosaic[0::2, 1::2] = 2000
    mosaic[1::2, 0::2] = 2000
    mosaic[1::2, 1::2] = 3800
    rgb = bayer_render.render_thumb_rgb(_pack(mosaic), w, h)
    assert rgb.shape == (120, 160, 3)
    means = rgb.reshape(-1, 3).mean(axis=0)
    assert max(means) - min(means) > 15

    odd = bayer_render.render_capture_view(_pack(mosaic), w, h, downscale=5, to_bgr=False)
    assert odd.ndim == 3 and odd.shape[2] == 3
    assert odd.shape[0] % 2 == 0 and odd.shape[1] % 2 == 0


def test_speed_keeps_the_velocity_after_the_second_bar():
    body = "INFO|Frame: 40/2000 (2.0%) | Vel: 12,50 fps".split("|", 1)[1]
    assert parse_export_speed(body) == (40, 2000, 12.5)


def test_speed_and_groups():
    parsed = parse_export_speed("Frame: 40/2000 (2.0%) | Vel: 12.50 fps")
    assert parsed == (40, 2000, 12.5)
    file_eta, total_eta = estimate_remaining_seconds(12.5, 40, 200, 1000)
    assert abs(file_eta - 160 / 12.5) < 1e-6
    assert abs(total_eta - 1160 / 12.5) < 1e-6
    assert estimate_remaining_seconds(0, 1, 10, 0) == (None, None)
    assert format_duration(None) == "—"
    assert format_duration(45) == "45 s"
    assert "min" in format_duration(90)

    dng = format_options("dng", False)
    assert not any(dng.values())
    exr = format_options("exr", False)
    assert exr == {"reveal": False, "sharp": False, "grade": False, "test": True, "fps": False}
    reveal = format_options("prores_4444", False)
    assert reveal["reveal"] and reveal["fps"] and reveal["test"]
    assert not reveal["grade"] and not reveal["sharp"]
    video = format_options("h264", False)
    assert video["sharp"] and video["grade"] and video["test"] and video["fps"]
    isp = format_options("prores", True)
    assert isp["grade"] and not isp["sharp"] and not isp["reveal"]


def test_draft_roundtrip_across_formats():
    from PyQt6.QtWidgets import QApplication, QComboBox
    import export_session

    app = QApplication.instance() or QApplication([])

    class Memory:
        def __init__(self):
            self.files = {}

        def get_file_info(self, _collection, filename):
            return dict(self.files.get(filename, {}))

        def set_file_info(self, _collection, filename, **kwargs):
            self.files.setdefault(filename, {}).update(kwargs)

    meta = {"toma.raw": {"fps": 16, "pixel_format": "bayer", "export_output_name": "viejo_prores.mov"}}
    memory = Memory()
    dialog = export_session.ExportSessionDialog(
        None, ["toma.raw"], ".", "coleccion", meta, memory,
    )
    assert dialog.combo_fmt.currentData() == "dng"
    assert dialog._edits["toma.raw"].text() == "toma_DNG_SEQ"
    assert dialog.table.isColumnHidden(3)
    assert dialog.table.isColumnHidden(4)

    dialog._edits["toma.raw"].setText("mi_dng")

    def choose(fmt):
        for index in range(dialog.combo_fmt.count()):
            if dialog.combo_fmt.itemData(index) == fmt:
                dialog.combo_fmt.setCurrentIndex(index)
                return
        raise AssertionError(fmt)

    choose("prores_4444")
    assert dialog._fmt == "prores_4444"
    assert dialog._edits["toma.raw"].text().endswith(".mov")
    assert not dialog.table.isColumnHidden(3)
    assert not dialog.reveal_row.isHidden()
    assert dialog.sharp_row.isHidden()
    saved = memory.files["toma.raw"]["export_by_format"]["dng"]["output_name"]
    assert saved == "mi_dng"

    dialog._edits["toma.raw"].setText("mi_xq.mov")
    choose("dng")
    assert dialog._edits["toma.raw"].text() == "mi_dng"
    choose("prores_4444")
    assert dialog._edits["toma.raw"].text() == "mi_xq.mov"

    again = draft_for_file("toma.raw", {"toma.raw": memory.files["toma.raw"]}, "dng")
    assert again["output_name"] == "mi_dng"
    dialog.close()
    assert isinstance(dialog.combo_fmt, QComboBox)


def test_thumb_marks_each_percent():
    from raw2video import _jpeg_rgb8, _thumb_due
    state = {"pct": -1}
    assert _thumb_due(state, 1, 200)
    assert not _thumb_due(state, 1, 200)
    assert _thumb_due(state, 3, 200)
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = 180
    jpg = _jpeg_rgb8(rgb)
    assert jpg is not None and jpg[:2] == b"\xff\xd8"


if __name__ == "__main__":
    test_thumb_is_color()
    test_speed_keeps_the_velocity_after_the_second_bar()
    test_speed_and_groups()
    test_thumb_marks_each_percent()
    test_draft_roundtrip_across_formats()
    print("ok")
