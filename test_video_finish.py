"""Sesión, fps y medición de perforación sobre grises, sin la ventana."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import film_finish
import perfo_lock
from video_export import output_path
from video_session import (
    empty_session, load_session, match_folder, meaningful_adjust, missing_files, save_session,
)
from video_source import info_from_stream, parse_rate


def test_parse_rate():
    assert parse_rate("24000/1001") == 24000 / 1001
    assert parse_rate("24/1") == 24
    assert parse_rate("0/0") == 0
    assert parse_rate("N/A") == 0
    assert abs(parse_rate("18.0") - 18) < 1e-6


def test_info_from_stream_uses_duration_when_frames_are_missing():
    info = info_from_stream(
        Path("rollo.mov"),
        {"width": 1920, "height": 1080, "r_frame_rate": "18/1", "nb_frames": "N/A", "duration": "2"},
    )
    assert info.frames == 36
    assert info.fps == 18
    assert info.fps_text == "18/1"


def test_session_roundtrip_and_missing():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        path = root / "video_finish_session.json"
        data = empty_session()
        data["files"] = [{
            "path": str(root / "no_esta.mov"),
            "fps": 16,
            "fps_text": "16/1",
            "fps_edited": True,
            "adjust": {"grade": {"mode": "bw", "sat": 0}},
        }]
        save_session(path, data)
        loaded = load_session(path)
        assert loaded["files"][0]["fps_edited"] is True
        assert len(missing_files(loaded)) == 1
        assert meaningful_adjust(loaded["files"][0]["adjust"]) is True
        assert meaningful_adjust(film_finish.empty()) is False


def test_match_folder_by_name():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        new = root / "nueva"
        new.mkdir()
        (new / "toma.mov").write_bytes(b"x")
        missing = [{"path": str(root / "vieja" / "toma.mov")}, {"path": str(root / "vieja" / "otra.mp4")}]
        found = match_folder(missing, new)
        assert found[str(root / "vieja" / "toma.mov")] == new / "toma.mov"
        assert len(found) == 1


def test_output_names_do_not_collide():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        taken: set[str] = set()
        first = output_path(root, Path("rollo.mov"), "prores_4444", taken)
        second = output_path(root, Path("rollo.mov"), "prores_4444", taken)
        hevc = output_path(root, Path("rollo.mov"), "hevc_nvenc", taken)
        assert first.name == "rollo_acabado.mov"
        assert second.name == "rollo_acabado_2.mov"
        assert hevc.suffix == ".mp4"
        first.write_bytes(b"x")
        third = output_path(root, Path("otro.mov"), "prores_4444", set())
        assert third.name == "otro_acabado.mov"
        again = output_path(root, Path("rollo.mov"), "prores_4444", set())
        assert again.name == "rollo_acabado_2.mov"


def test_analyze_ordered_on_a_still_hole():
    width, height = 400, 300
    rect = (30, 20, 80, 60)
    ident = perfo_lock._clip_rect(rect, width, height)
    search = perfo_lock._search_rect(ident, width, height)
    crop = np.zeros((search[3], search[2]), np.uint8)
    x = ident[0] - search[0] + 8
    y = ident[1] - search[1] + 8
    crop[y:y + 36, x:x + 28] = 240
    crop[y + 10:y + 26, x + 8:x + 20] = 10

    def sample(_index):
        return crop.copy()

    result = perfo_lock.analyze_ordered(
        sample, (crop.copy() for _ in range(4)), 4, rect, width, height, 0,
    )
    assert result["ref"] == 0
    assert len(result["tx"]) == 4
    short = perfo_lock.analyze_ordered(
        sample, (crop.copy() for _ in range(3)), 5, rect, width, height, 0,
    )
    assert len(short["tx"]) == 3
    assert max(abs(v) for v in result["tx"]) < 1.5
    assert max(abs(v) for v in result["ty"]) < 1.5
    odd = perfo_lock._search_rect((0, 0, 145, 341), 2840, 2160)
    assert odd == (0, 0, 286, 482)


if __name__ == "__main__":
    test_parse_rate()
    test_info_from_stream_uses_duration_when_frames_are_missing()
    test_session_roundtrip_and_missing()
    test_match_folder_by_name()
    test_output_names_do_not_collide()
    test_analyze_ordered_on_a_still_hole()
    print("ok")
