"""Lectura de videos ya revelados: datos del contenedor y cuadros RGB."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mov", ".mp4", ".mkv"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class VideoError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    fps_text: str
    frames: int

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frames / self.fps


def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise VideoError("No está ffmpeg en el PATH.")
    return found


def ffprobe_bin() -> str:
    found = shutil.which("ffprobe")
    if not found:
        raise VideoError("No está ffprobe en el PATH.")
    return found


def parse_rate(text: str) -> float:
    raw = str(text or "").strip()
    if not raw or raw in {"0/0", "N/A"}:
        return 0.0
    if "/" in raw:
        num, den = raw.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return 0.0
            return float(num) / den_f
        except ValueError:
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _rate_text(stream: dict) -> str:
    for key in ("r_frame_rate", "avg_frame_rate"):
        text = str(stream.get(key) or "").strip()
        if parse_rate(text) > 0:
            return text
    return "24/1"


def info_from_stream(path: Path, stream: dict, fmt: dict | None = None) -> VideoInfo:
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width < 2 or height < 2:
        raise VideoError("El video no tiene tamaño.")
    fps_text = _rate_text(stream)
    fps = parse_rate(fps_text)
    if fps <= 0:
        fps = 24.0
        fps_text = "24/1"
    frames = 0
    raw_frames = str(stream.get("nb_frames") or "").strip()
    if raw_frames.isdigit():
        frames = int(raw_frames)
    if frames <= 0:
        duration = stream.get("duration")
        if duration in (None, "", "N/A") and isinstance(fmt, dict):
            duration = fmt.get("duration")
        try:
            seconds = float(duration)
        except (TypeError, ValueError):
            seconds = 0.0
        if seconds > 0:
            frames = max(1, int(round(seconds * fps)))
    if frames <= 0:
        raise VideoError("No pude saber cuántos cuadros tiene.")
    return VideoInfo(Path(path), width, height, fps, fps_text, frames)


def probe(path: Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise VideoError("El archivo no está.")
    cmd = [
        ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
        data = json.loads(raw.decode("utf-8", "replace"))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise VideoError(f"No pude leer el video. {exc}") from exc
    streams = data.get("streams") or []
    if not streams:
        raise VideoError("El archivo no tiene video.")
    return info_from_stream(path, streams[0], data.get("format"))


def _run_bytes(cmd: list) -> bytes:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VideoError(f"ffmpeg no pudo leer el cuadro. {exc}") from exc


def _preview_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    if not max_width or width <= max_width:
        return width, height
    out_w = int(max_width)
    out_h = max(2, int(round(height * out_w / float(width))))
    if out_h % 2:
        out_h -= 1
    return max(2, out_w), max(2, out_h)


def read_rgb(path: Path, index: int, fps: float, width: int, height: int, *, full: bool = False, max_width: int | None = None) -> np.ndarray:
    """Un cuadro en float 0–1, RGB. `full` pide 16 bits."""
    out_w, out_h = (width, height) if full else _preview_size(width, height, max_width)
    pix = "rgb48le" if full else "rgb24"
    seconds = 0.0 if fps <= 0 else max(0.0, float(index) / float(fps))
    cmd = [
        ffmpeg_bin(), "-v", "error", "-ss", f"{seconds:.6f}", "-i", str(path),
        "-an", "-frames:v", "1",
    ]
    if (out_w, out_h) != (width, height):
        cmd += ["-vf", f"scale={out_w}:{out_h}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix, "pipe:1"]
    raw = _run_bytes(cmd)
    channels = 6 if full else 3
    need = out_w * out_h * channels
    if len(raw) < need:
        raise VideoError("El cuadro llegó incompleto.")
    if full:
        image = np.frombuffer(raw[:need], dtype=np.uint16).reshape(out_h, out_w, 3).copy()
        return image.astype(np.float32) * np.float32(1.0 / 65535.0)
    image = np.frombuffer(raw[:need], dtype=np.uint8).reshape(out_h, out_w, 3).copy()
    return image.astype(np.float32) * np.float32(1.0 / 255.0)


def _even_crop(rect) -> tuple[int, int, int, int]:
    """Mismo ajuste que perfo_lock._even_span, sin conocer el cuadro entero."""
    x, y, rect_w, rect_h = (int(v) for v in rect)
    x -= x % 2
    y -= y % 2
    rect_w -= rect_w % 2
    rect_h -= rect_h % 2
    return x, y, max(2, rect_w), max(2, rect_h)


def read_gray_crop(path: Path, search: tuple[int, int, int, int], index: int, fps: float) -> np.ndarray | None:
    x, y, rect_w, rect_h = _even_crop(search)
    seconds = 0.0 if fps <= 0 else max(0.0, float(index) / float(fps))
    cmd = [
        ffmpeg_bin(), "-v", "error", "-ss", f"{seconds:.6f}", "-i", str(path),
        "-an", "-vf", f"crop={rect_w}:{rect_h}:{x}:{y},format=gray",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except (OSError, subprocess.CalledProcessError):
        return None
    need = rect_w * rect_h
    if len(raw) < need:
        return None
    return np.frombuffer(raw[:need], dtype=np.uint8).reshape(rect_h, rect_w).copy()


class GrayCropStream:
    """Cuadros en gris, recortados, desde el primero. Al terminar queda el código de ffmpeg."""

    def __init__(self, path: Path, search: tuple[int, int, int, int], cancel=None):
        x, y, rect_w, rect_h = _even_crop(search)
        self._need = rect_w * rect_h
        self._shape = (rect_h, rect_w)
        self._cancel = cancel
        self._errors: list[str] = []
        self.code = None
        self.error = ""
        self._clean = False
        self._closed = False
        cmd = [
            ffmpeg_bin(), "-v", "error", "-i", str(path), "-an",
            "-vf", f"crop={rect_w}:{rect_h}:{x}:{y},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW,
        )
        self._drain = threading.Thread(target=_collect_text, args=(self._proc.stderr, self._errors), daemon=True)
        self._drain.start()

    def __iter__(self):
        try:
            while True:
                if self._cancel is not None and self._cancel():
                    raise Cancelled()
                raw = _read_exact(self._proc.stdout, self._need)
                if raw is None:
                    self._clean = True
                    break
                yield np.frombuffer(raw, dtype=np.uint8).reshape(self._shape).copy()
        finally:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc.poll() is None:
            if self._clean and proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            else:
                proc.kill()
        self.code = proc.wait()
        self._drain.join(timeout=2)
        self.error = "".join(self._errors)[-400:].strip()


def _collect_text(pipe, sink: list):
    if pipe is None:
        return
    for line in iter(pipe.readline, b""):
        sink.append(line.decode("utf-8", "replace"))


def iter_rgb48(path: Path, width: int, height: int, cancel=None):
    cmd = [
        ffmpeg_bin(), "-v", "error", "-i", str(path), "-an",
        "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
    )
    need = width * height * 6
    try:
        while True:
            if cancel is not None and cancel():
                raise Cancelled()
            raw = _read_exact(proc.stdout, need)
            if raw is None:
                break
            yield np.frombuffer(raw, dtype=np.uint16).reshape(height, width, 3).copy()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def iter_yuv422(path: Path, width: int, height: int, cancel=None):
    """Cuadros YUV 4:2:2 de 10 bits, sin pasar por RGB. El ancho tiene que ser par."""
    y_n = int(width) * int(height)
    c_n = (int(width) // 2) * int(height)
    need = (y_n + c_n * 2) * 2
    cmd = [
        ffmpeg_bin(), "-v", "error", "-i", str(path), "-an",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le", "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
    )
    try:
        while True:
            if cancel is not None and cancel():
                raise Cancelled()
            raw = _read_exact(proc.stdout, need)
            if raw is None:
                break
            yield raw
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _read_exact(handle, size: int) -> bytes | None:
    if handle is None:
        return None
    chunks = []
    left = size
    while left:
        part = handle.read(left)
        if not part:
            return None if not chunks else None
        chunks.append(part)
        left -= len(part)
    return b"".join(chunks)


class PreviewReader:
    """Vista rápida. Si OpenCV no abre el archivo, el cuadro sale por ffmpeg."""

    def __init__(self):
        self._cap = None
        self._path = None
        self._index = -1
        self._bad = False

    def close(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._index = -1

    def frame(self, info: VideoInfo, index: int, max_width: int = 1100) -> np.ndarray:
        index = int(np.clip(index, 0, max(0, info.frames - 1)))
        path = str(info.path)
        if self._path != path:
            self.close()
            self._path = path
            self._bad = False
        if self._bad:
            return read_rgb(
                info.path, index, info.fps, info.width, info.height, max_width=max_width,
            )
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(path)
            self._index = -1
            if not self._cap.isOpened():
                self._bad = True
                return read_rgb(
                    info.path, index, info.fps, info.width, info.height, max_width=max_width,
                )
        if self._cap is not None and self._cap.isOpened():
            if index != self._index + 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self._cap.read()
            if ok and bgr is not None and bgr.shape[1] == info.width and bgr.shape[0] == info.height:
                self._index = index
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if max_width and rgb.shape[1] > max_width:
                    out_h = max(1, int(round(rgb.shape[0] * max_width / float(rgb.shape[1]))))
                    rgb = cv2.resize(rgb, (int(max_width), out_h), interpolation=cv2.INTER_AREA)
                return rgb.astype(np.float32) * np.float32(1.0 / 255.0)
            if ok and bgr is not None:
                self._bad = True
                self.close()
        self._index = -1
        return read_rgb(
            info.path, index, info.fps, info.width, info.height, max_width=max_width,
        )
