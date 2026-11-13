"""GamMac y perforación sobre imágenes chicas, sin la cámara ni la GPU."""
import numpy as np

import gammac
import perfo_lock


def test_gammac_locks_green():
    ramp = np.linspace(0.15, 0.85, 60, dtype=np.float32)
    image = np.zeros((80, 100, 3), np.float32)
    image[20:60, 20:80, 0] = 0.55
    image[20:60, 20:80, 1] = 0.35
    image[20:60, 20:80, 2] = 0.28
    image[20:60, 20:80, :] *= ramp[None, :, None]
    recipe = gammac.measure(image, (24, 24, 40, 28), lock=1, scale=2)
    assert recipe["ok"] and not recipe["limited"]
    corrected = gammac.apply(image, recipe)
    area = corrected[24:52, 24:64]
    means = [float(area[..., c].mean()) for c in range(3)]
    assert abs(means[0] - means[1]) < 0.04
    assert abs(means[2] - means[1]) < 0.04


def test_gammac_ignores_border():
    ramp = np.linspace(0.2, 0.9, 100, dtype=np.float32)
    image = np.zeros((80, 100, 3), np.float32)
    image[:, :, 0] = 0.45 * ramp
    image[:, :, 1] = 0.45 * ramp
    image[:, :, 2] = 0.25 * ramp
    image[:8, :, :] = 0
    image[:, :8, :] = 1
    inside = gammac.measure(image, (16, 16, 60, 50), lock=1, scale=2)
    whole = gammac.measure(image, (0, 0, 100, 80), lock=1, scale=2)
    assert abs(inside["gammas"][2] - whole["gammas"][2]) > 0.05


def test_strength_and_black_floor():
    image = np.zeros((8, 8, 3), np.float32)
    image[..., 0] = 0.55
    image[..., 1] = 0.40
    image[..., 2] = 0.25
    recipe = {
        "ok": True,
        "limited": False,
        "scale": 0,
        "gammas": [0.45, 0.45, 0.45],
        "strength": 0.0,
        "omin": 0.0,
        "omax": 1.0,
    }
    assert np.allclose(gammac.apply(image, recipe), image)

    recipe["strength"] = 1.0
    recipe["gammas"] = [1.0, 1.0, 1.0]
    recipe["omin"] = 8 / 255
    recipe["omax"] = 240 / 255
    black = np.zeros_like(image)
    white = np.ones_like(image)
    assert abs(float(gammac.apply(black, recipe)[0, 0, 0]) - 8 / 255) < 1e-5
    assert abs(float(gammac.apply(white, recipe)[0, 0, 0]) - 240 / 255) < 1e-5

    shadow = np.full_like(image, 0.02)
    recipe["dark"] = 0.20
    recipe["gammas"] = [0.4, 0.4, 0.4]
    assert np.allclose(gammac.apply(shadow, recipe), shadow, atol=1e-5)


def test_turn_spins_clockwise_inside_the_crop():
    import cv2
    import film_finish
    image = np.zeros((40, 60, 3), np.float32)
    image[20, 36] = 1.0
    matrix = film_finish.sampling_matrix(0, 0, 90, (20, 20))
    spun = cv2.warpAffine(
        image, matrix, (60, 40),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    spot = np.argwhere(spun[..., 0] > 0.5)
    assert len(spot) == 1
    y, x = (int(v) for v in spot[0])
    assert abs(x - 20) <= 1 and abs(y - 36) <= 1
    same = film_finish.sampling_matrix(4, 0, 0, (20, 20))
    assert np.allclose(same, [[1, 0, 4], [0, 1, 0]])


def test_flip_mirrors_after_the_crop():
    import tempfile
    from pathlib import Path
    import film_finish
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "a.raw"
        raw.write_bytes(b"")
        film_finish.save(raw, {
            "crop": [0, 0, 32, 40],
            "flip_h": True,
            "turn": 0,
            "segments": [],
            "perfo": {"enabled": False, "roi": None},
            "grade": {"mode": "color", "sat": 1},
        })
        finish = film_finish.Finish(raw, gpu=False, codec="prores_4444")
        image = np.zeros((40, 80, 3), np.float32)
        image[:, 0, 0] = 1
        out = finish.apply_float(image, 0, finish.output_size(80, 40))
        assert out.shape[1] == 32
        assert out[0, -1, 0] > 0.5
        assert out[0, 0, 0] < 0.1
        assert film_finish.load(raw)["crop"][0] == 0
        assert film_finish.mirror_x((0, 0, 32, 40), 80) == (48, 0, 32, 40)


def test_crop_snaps_for_prores_nv():
    import film_finish
    snapped = film_finish.snap_crop((10, 20, 100, 50), 2840, 2200)
    assert snapped == (10, 20, 96, 48)
    full = film_finish.snap_crop((0, 0, 2840, 2200), 2840, 2200)
    assert full == (0, 0, 2832, 2200)
    assert full[2] % 16 == 0 and full[3] % 8 == 0
    edge = film_finish.snap_crop((2830, 2190, 20, 20), 2840, 2200)
    assert edge[2] % 16 == 0 and edge[3] % 8 == 0
    assert edge[0] + edge[2] <= 2840 and edge[1] + edge[3] <= 2200


def test_reveal_belongs_to_one_file():
    import tempfile
    from pathlib import Path
    import film_finish
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        one = root / "one.raw"
        two = root / "two.raw"
        one.write_bytes(b"")
        two.write_bytes(b"")
        own = film_finish.empty()
        own["reveal"] = {"gamma": 1.5, "ev": 0.5, "engine": "cpu"}
        film_finish.save(one, own)
        film_finish.save(two, film_finish.empty())
        assert film_finish.file_reveal(one, {"gamma": 2.4})["gamma"] == 1.5
        assert film_finish.file_reveal(one, {"gamma": 2.4})["ev"] == 0.5
        assert film_finish.file_reveal(two, {"gamma": 2.4})["gamma"] == 2.4
        assert film_finish.load(two)["reveal"] is None
        assert film_finish.copy_adjust(one, two)
        assert film_finish.file_reveal(two, {"gamma": 2.4})["gamma"] == 1.5
        bare = film_finish.empty()
        bare["crop"] = [0, 0, 16, 8]
        film_finish.save(one, bare)
        kept = film_finish.file_reveal(two, {"gamma": 2.4})["gamma"]
        assert film_finish.copy_adjust(one, two)
        assert film_finish.file_reveal(two, {"gamma": 2.4})["gamma"] == kept


def test_copy_adjust_skips_motion():
    import json
    from pathlib import Path
    import tempfile
    import film_finish
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        src = root / "a.raw"
        dst = root / "b.raw"
        src.write_bytes(b"")
        dst.write_bytes(b"")
        film_finish.save(src, {
            "segments": [{"in": 0, "out": 3, "recipe": {"ok": True, "gammas": [1, 1, 1]}}],
            "crop": [0, 0, 2832, 2200],
            "perfo": {"enabled": True, "roi": [10, 20, 80, 120], "ref": 40},
        })
        film_finish.perfo_path(src).write_text('{"tx":[1]}', encoding="utf-8")
        assert film_finish.copy_adjust(src, dst)
        copied = json.loads(film_finish.adjust_path(dst).read_text(encoding="utf-8"))
        assert copied["crop"][2] == 2832
        assert copied["perfo"]["roi"][0] == 10
        assert copied["perfo"]["ref"] == 40
        assert not film_finish.perfo_path(dst).is_file()
        empty = root / "c.raw"
        empty.write_bytes(b"")
        film_finish.save(empty, film_finish.empty())
        assert film_finish.copy_adjust(empty, dst) is False


def test_perfo_skips_what_is_done_and_can_resume():
    import tempfile
    from pathlib import Path
    import film_finish
    import perfo_lock
    rows = [(0.0, 0.0, 1.0, 0)] * 10
    rows[3] = None
    rows[4] = None
    rows[8] = None
    assert perfo_lock._pending_spans(rows, 48) == [(3, 5), (8, 9)]
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "rollo.raw"
        raw.write_bytes(b"\x00" * (2840 * 2200 * 3 // 2))
        film_finish.write_perfo_status(
            raw, "done", "Lista",
            {"tx": [0.0], "ty": [0.0], "roi": [1, 2, 3, 4]},
        )
        assert film_finish.perfo_is_done(raw, [1, 2, 3, 4]) is True
        assert film_finish.perfo_is_done(raw, [9, 9, 9, 9]) is False
        adjust = film_finish.load(raw)
        adjust["perfo"] = {"enabled": True, "roi": [1, 2, 3, 4], "ref": 9}
        film_finish.save(raw, adjust)
        assert film_finish.perfo_is_done(raw, [1, 2, 3, 4]) is False
        film_finish.write_perfo_status(
            raw, "running", "2/4",
            partial=[[0.1, 0.2, 0.9, 0], None, [0.0, 0.0, 0.8, 0], None],
            roi=[4, 5, 30, 40], ref=7,
        )
        loaded, ref = perfo_lock._load_resume(raw, (4, 5, 30, 40), 4)
        assert ref == 7
        assert loaded[0][0] == 0.1 and loaded[1] is None and loaded[2][2] == 0.8
        assert perfo_lock._load_resume(raw, (0, 0, 8, 8), 4)[1] is None
        film_finish.write_perfo_status(raw, "error", "tropiezo")
        kept = film_finish.read_perfo(raw)
        assert kept["status"] == "error"
        assert kept["partial"][0][0] == 0.1 and kept["ref"] == 7


def test_quarter_preview_keeps_bayer_phase():
    import reveal_pipeline
    width, height = 2840, 2200
    mosaic = np.empty((height, width), np.uint16)
    mosaic[0::2, 0::2] = 1000
    mosaic[0::2, 1::2] = 2000
    mosaic[1::2, 0::2] = 1800
    mosaic[1::2, 1::2] = 3000
    flat = mosaic.reshape(-1)
    p0 = flat[0::2]
    p1 = flat[1::2]
    packed = np.empty((p0.size, 3), np.uint8)
    packed[:, 0] = p0 & 0xFF
    packed[:, 1] = ((p0 >> 8) & 0x0F) | ((p1 & 0x0F) << 4)
    packed[:, 2] = (p1 >> 4) & 0xFF
    rgb = reveal_pipeline.debayer_quarter(packed.tobytes(), width, height)
    assert rgb.shape == (height // 4, width // 4, 3)
    pixel = rgb[20, 20]
    assert abs(float(pixel[0]) - 1000 / 4095) < 0.02
    assert abs(float(pixel[1]) - 1900 / 4095) < 0.05
    assert abs(float(pixel[2]) - 3000 / 4095) < 0.02
    play = reveal_pipeline.debayer_playback(packed.tobytes(), width, height)
    assert play.shape == ((height // 6) * 2, (width // 6) * 2, 3)
    pixel = play[20, 20]
    assert abs(float(pixel[0]) - 1000 / 4095) < 0.02
    assert abs(float(pixel[1]) - 1900 / 4095) < 0.05
    assert abs(float(pixel[2]) - 3000 / 4095) < 0.02


def test_saturation_goes_gray():
    image = np.zeros((4, 4, 3), np.float32)
    image[..., 0] = 0.8
    image[..., 1] = 0.2
    image[..., 2] = 0.1
    same = gammac.saturate(image, 1)
    assert same is image
    gray = gammac.saturate(image, 0)
    assert np.allclose(gray[..., 0], gray[..., 1])
    assert np.allclose(gray[..., 1], gray[..., 2])
    assert float(gray[0, 0, 0]) > 0.2


def test_segment_lookup():
    segments = [
        {"in": 10, "out": 20, "recipe": {"ok": True, "gammas": [1, 1, 1], "scale": 0, "mins": [0, 0, 0], "maxs": [1, 1, 1], "shared_min": 0, "shared_max": 1, "limited": False}},
        {"in": 30, "out": 40, "recipe": {"ok": True, "limited": True}},
    ]
    assert gammac.recipe_at(segments, 15)["scale"] == 0
    assert gammac.recipe_at(segments, 25) is None
    assert gammac.recipe_at(segments, 35)["limited"]


def test_review_badges():
    import tempfile
    from pathlib import Path
    import film_finish
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "a.raw"
        raw.write_bytes(b"\x00" * (2840 * 2200 * 3 // 2))
        film_finish.save(raw, film_finish.empty())
        state = film_finish.review_state(raw)
        assert state["reviewed"] is False
        assert state["stab"] == "sin_zona"
        assert "Color" in [item[0] for item in film_finish.review_badges(state)]
        data = film_finish.load(raw)
        data["crop"] = [0, 0, 2832, 2192]
        data["grade"] = {"mode": "bw", "sat": 0.0}
        data["perfo"] = {"enabled": True, "roi": [1, 2, 30, 40]}
        film_finish.save(raw, data)
        state = film_finish.review_state(raw)
        assert state["reviewed"] is True and state["color_kind"] == "bw"
        assert state["stab"] == "por_correr"
        labels = [item[0] for item in film_finish.review_badges(state)]
        assert "Revisado" in labels and "Por correr" in labels
        data["grade"] = {"mode": "color", "sat": 1.0}
        data["segments"] = [{
            "in": 0, "out": 1,
            "recipe": {"ok": True, "limited": True, "gammas": [1, 1, 1]},
        }]
        film_finish.save(raw, data)
        assert film_finish.review_state(raw)["color"] is False
        data["segments"][0]["recipe"]["limited"] = False
        film_finish.save(raw, data)
        assert film_finish.review_state(raw)["color_kind"] == "gammac"
        film_finish.write_perfo_status(raw, "running", "25/100", roi=[1, 2, 30, 40], partial=[], ref=0)
        state = film_finish.review_state(raw)
        assert state["stab"] == "corriendo" and state["percent"] == 25
        film_finish.write_perfo_status(
            raw, "done", "Lista",
            {"tx": [0.0], "ty": [0.0], "roi": [1, 2, 30, 40]},
        )
        assert film_finish.review_state(raw)["stab"] == "listo"


def test_horizontal_off_keeps_only_the_vertical():
    import tempfile
    from pathlib import Path
    import film_finish
    assert film_finish.perfo_horizontal({}) is True
    assert film_finish.perfo_horizontal({"horizontal": "false"}) is False
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "rollo.raw"
        raw.write_bytes(b"")
        film_finish.save(raw, {"perfo": {"enabled": True, "roi": [1, 2, 30, 40]}, "segments": []})
        assert film_finish.load(raw)["perfo"]["horizontal"] is True
        film_finish.save(raw, {
            "perfo": {"enabled": True, "roi": [1, 2, 30, 40], "horizontal": False},
            "segments": [],
        })
        film_finish.write_perfo_status(
            raw, "done", "Lista",
            {"tx": [4.0], "ty": [-2.0], "roi": [1, 2, 30, 40]},
        )
        finish = film_finish.Finish(raw, gpu=False, codec="prores_4444")
        assert finish.shift_at(0) == (0.0, -2.0)
        assert np.allclose(finish.sampling(0, 60, 40), [[1, 0, 0], [0, 1, -2]])
    from PyQt6.QtWidgets import QApplication
    from reveal_dialog import RevealDialog
    app = QApplication.instance() or QApplication([])
    dialog = RevealDialog(None, Path("."), [], {}, {}, None)
    dialog._set_task(1)
    assert dialog.chk_horizontal.isChecked()
    dialog._shifts = ([4.0], [-2.0])
    image = np.zeros((20, 40, 3), np.float32)
    image[8, 20] = 1.0
    dialog.chk_horizontal.setChecked(False)
    warped = dialog._warp_preview(image)
    spot = np.argwhere(warped[..., 0] > 0.5)
    assert len(spot) == 1
    y, x = (int(v) for v in spot[0])
    assert abs(x - 20) <= 1 and abs(y - 10) <= 1
    dialog.close()
    assert app is not None


def test_perfo_follows_a_hole():
    frames = []
    shifts = [(0, 0), (4, -2), (-3, 5), (1, 1), (6, 0), (-4, -3)]
    for dx, dy in shifts:
        gray = np.full((180, 240), 30, np.uint8)
        x = 40 + dx
        y = 50 + dy
        gray[y:y + 36, x:x + 28] = 230
        frames.append(gray)
    result = perfo_lock.analyze_grays(frames, (30, 40, 56, 64), ref_index=0)
    assert abs(result["tx"][0]) < 0.8 and abs(result["ty"][0]) < 0.8
    assert abs(result["tx"][1] - 4) < 0.8
    assert abs(result["ty"][1] + 2) < 0.8
    assert abs(result["tx"][2] + 3) < 0.8
    assert abs(result["ty"][2] - 5) < 0.8


def test_split_copies_the_crop_and_the_later_one_keeps_proportion():
    import film_finish
    spans = [{"in": 0, "out": 9, "crop": [0, 0, 32, 16], "flip_h": False}]
    parted, selected, changed = film_finish.split_at(spans, 4, 10)
    assert changed and selected == 1
    assert parted[0]["crop"] == [0, 0, 32, 16]
    assert parted[1]["crop"] == [0, 0, 32, 16]
    assert parted[0]["out"] == 3 and parted[1]["in"] == 4
    fitted = film_finish.fit_aspect((10, 4, 40, 40), 32 / 16, 80, 40)
    assert fitted[2] / fitted[3] == 2 or abs(fitted[2] / fitted[3] - 2) < 0.05
    moved, edge_changed = film_finish.move_edge(parted, 1, 6)
    assert edge_changed
    assert moved[0]["out"] == 5 and moved[1]["in"] == 6
    assert moved[0]["crop"] == [0, 0, 32, 16]


def test_later_crop_scales_to_the_first_and_flips_on_its_own():
    import tempfile
    from pathlib import Path
    import film_finish
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "a.raw"
        raw.write_bytes(b"")
        film_finish.save(raw, {
            "crop": [0, 0, 32, 16],
            "flip_h": False,
            "crop_spans": [
                {"in": 0, "out": 4, "crop": [0, 0, 32, 16], "flip_h": False},
                {"in": 5, "out": 9, "crop": [8, 4, 16, 8], "flip_h": True},
            ],
            "segments": [],
            "perfo": {"enabled": False, "roi": None},
            "grade": {"mode": "color", "sat": 1},
        })
        finish = film_finish.Finish(raw, gpu=False, codec="prores_4444")
        assert finish.output_size(80, 40) == (32, 16)
        image = np.zeros((40, 80, 3), np.float32)
        image[0, 0, 0] = 1
        image[0, 31, 2] = 1
        image[4:12, 8:16, 0] = 1
        image[4:12, 16:24, 2] = 1
        first = finish.apply_float(image, 0, finish.output_size(80, 40))
        assert first.shape[:2] == (16, 32)
        assert first[0, 0, 0] > 0.5
        assert first[0, 31, 2] > 0.5
        second = finish.apply_float(image, 5, finish.output_size(80, 40))
        assert second.shape[:2] == (16, 32)
        assert second[8, 4, 2] > second[8, 4, 0]
        assert second[8, 28, 0] > second[8, 28, 2]


def _gamma_recipe(gamma: float) -> dict:
    return {
        "ok": True,
        "limited": False,
        "scale": 0,
        "gammas": [gamma, gamma, gamma],
        "strength": 1.0,
        "omin": 0.0,
        "omax": 1.0,
    }


def test_two_segments_keep_their_gamma_and_reverse_moves_the_range():
    import tempfile
    from pathlib import Path
    import film_finish
    count = 1000
    assert film_finish.source_to_view(100, count, True) == 899
    assert film_finish.source_to_view(200, count, True) == 799
    spans = [{"in": 100, "out": 200, "recipe": _gamma_recipe(2.0)}]
    view = film_finish.spans_to_view(spans, count, True)
    assert (view[0]["in"], view[0]["out"]) == (799, 899)
    back = film_finish.spans_to_source(view, count, True)
    assert (back[0]["in"], back[0]["out"]) == (100, 200)
    assert film_finish.output_sources(10, False, None, None) == list(range(10))
    assert film_finish.output_sources(10, True, 2, 8) == [8, 7, 6, 5, 4, 3, 2]
    assert film_finish.trim_from_view(10, False, 0, 9) == (None, None)
    assert film_finish.natural_key("F8_0000100_01.mp4") < film_finish.natural_key("F8_0000148_01.mp4")
    assert film_finish.natural_key("F8_2.mp4") < film_finish.natural_key("F8_10.mp4")
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "rollo.raw"
        raw.write_bytes(b"")
        film_finish.save(raw, {
            "segments": [
                {"in": 0, "out": 0, "recipe": _gamma_recipe(2.0)},
                {"in": 1, "out": 1, "recipe": _gamma_recipe(1.0)},
            ],
            "grade": {"mode": "color", "sat": 1.0},
            "reverse": True,
            "trim_in": 0,
            "trim_out": 1,
        })
        finish = film_finish.Finish(raw, gpu=False, codec="prores_4444")
        assert finish.sources(4) == [1, 0]
        image = np.full((8, 8, 3), 0.5, np.float32)
        dark = finish.apply_float(image, 0, None)
        flat = finish.apply_float(image, 1, None)
        assert float(dark[0, 0, 0]) < 0.3
        assert abs(float(flat[0, 0, 0]) - 0.5) < 1e-5
        assert film_finish.load(raw)["reverse"] is True


def test_perfo_analysis_leaves_the_video_folder():
    import json
    import tempfile
    from pathlib import Path
    import film_finish
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        left = root / "uno"
        right = root / "dos"
        left.mkdir()
        right.mkdir()
        raw = left / "rollo.mp4"
        other = right / "rollo.mp4"
        raw.write_bytes(b"")
        other.write_bytes(b"")
        legacy = left / "rollo.mp4.perfo.json"
        legacy.write_text('{"status": "running", "message": "1/4", "ref": 3}', encoding="utf-8")
        moved = film_finish.perfo_path(raw)
        assert moved.parent == film_finish.analysis_dir()
        assert not legacy.is_file()
        assert json.loads(moved.read_text(encoding="utf-8"))["source"]
        film_finish.write_perfo_status(other, "running", "2/4", roi=[1, 2, 3, 4], partial=[], ref=1)
        assert film_finish.perfo_path(raw) != film_finish.perfo_path(other)
        found = list(film_finish.running_analysis(right))
        assert len(found) == 1 and found[0][0] == other
        assert list(film_finish.running_analysis(left))[0][0] == raw
        film_finish.write_perfo_status(raw, "done", "Lista", {"tx": [0.0], "ty": [0.0]})
        assert film_finish.read_perfo(raw)["status"] == "done"
        assert not any(left.glob("*.perfo.json"))
        moved.unlink(missing_ok=True)
        film_finish.perfo_path(other).unlink(missing_ok=True)


if __name__ == "__main__":
    test_gammac_locks_green()
    test_gammac_ignores_border()
    test_strength_and_black_floor()
    test_turn_spins_clockwise_inside_the_crop()
    test_flip_mirrors_after_the_crop()
    test_crop_snaps_for_prores_nv()
    test_reveal_belongs_to_one_file()
    test_copy_adjust_skips_motion()
    test_perfo_skips_what_is_done_and_can_resume()
    test_review_badges()
    test_saturation_goes_gray()
    test_quarter_preview_keeps_bayer_phase()
    test_segment_lookup()
    test_two_segments_keep_their_gamma_and_reverse_moves_the_range()
    test_horizontal_off_keeps_only_the_vertical()
    test_perfo_follows_a_hole()
    test_perfo_analysis_leaves_the_video_folder()
    test_split_copies_the_crop_and_the_later_one_keeps_proportion()
    test_later_crop_scales_to_the_first_and_flips_on_its_own()
    print("ok")
