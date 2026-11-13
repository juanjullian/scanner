"""Exporta un video ya revelado con el acabado guardado al lado."""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import queue
from pathlib import Path

import numpy as np

import film_finish
from raw2video import _append_ffmpeg_codec_args
from video_source import Cancelled, VideoError, ffmpeg_bin, iter_rgb48, iter_yuv422, probe

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def output_path(folder: Path, source: Path, codec: str, taken: set[str]) -> Path:
    ext = ".mp4" if codec == "hevc_nvenc" else ".mov"
    stem = Path(source).stem + "_acabado"
    candidate = Path(folder) / f"{stem}{ext}"
    number = 2
    while candidate.name.lower() in taken or candidate.is_file():
        candidate = Path(folder) / f"{stem}_{number}{ext}"
        number += 1
    taken.add(candidate.name.lower())
    return candidate


def _indexes(finish, frames: int, limit: int | None) -> list[int]:
    indexes = finish.sources(int(frames))
    if limit is not None:
        return indexes[:int(limit)]
    return indexes


def _forward_span(indexes: list[int]) -> tuple[int, int] | None:
    """Si la salida es un tramo seguido del archivo, el primero y el último. Si no, None."""
    if not indexes:
        return None
    if indexes == list(range(indexes[0], indexes[-1] + 1)):
        return indexes[0], indexes[-1]
    return None


def _pack(payload):
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload), ("bytes", None, None)
    arr = np.ascontiguousarray(payload)
    return arr.tobytes(), ("array", arr.shape, arr.dtype)


def _unpack(blob: bytes, kind, shape, dtype):
    if kind == "bytes":
        return blob
    return np.frombuffer(blob, dtype=dtype).reshape(shape).copy()


def _select_ordered(items, indexes: list[int]):
    """Entrega los cuadros en el orden pedido. Si van al revés, los deja en un archivo temporal."""
    closer = getattr(items, "close", None)
    try:
        if not indexes:
            return
        span = _forward_span(indexes)
        if span is not None:
            start, end = span
            for index, payload in enumerate(items):
                if index < start:
                    continue
                if index > end:
                    break
                yield index, payload
            return
        needed = set(indexes)
        last = max(needed)
        spool = tempfile.NamedTemporaryFile(prefix="scanner-orden-", delete=False)
        offsets: dict[int, int] = {}
        size = None
        kind = shape = dtype = None
        try:
            for index, payload in enumerate(items):
                if index not in needed:
                    if size is not None and index > last:
                        break
                    continue
                raw, meta = _pack(payload)
                if size is None:
                    size = len(raw)
                    kind, shape, dtype = meta
                offsets[index] = spool.tell()
                spool.write(raw)
                if len(offsets) == len(needed):
                    break
            spool.flush()
            for index in indexes:
                if index not in offsets or size is None:
                    continue
                spool.seek(offsets[index])
                yield index, _unpack(spool.read(size), kind, shape, dtype)
        finally:
            name = spool.name
            spool.close()
            try:
                os.unlink(name)
            except OSError:
                pass
    finally:
        if closer is not None:
            closer()


def export_clip(path: Path, dst: Path, codec: str, fps_text: str, cancel, on_frame, limit: int | None = None) -> None:
    """Decodifica, aplica color, estabilización y recorte, y vuelve a codificar."""
    info = probe(path)
    on_gpu = codec in ("prores_4444", "hevc_nvenc") and info.width % 2 == 0 and info.height % 2 == 0
    finish = film_finish.Finish(path, gpu=(codec == "prores_4444" and on_gpu), codec=codec)
    out_w, out_h = finish.output_size(info.width, info.height)
    if on_gpu:
        _export_gpu(path, dst, codec, fps_text, info, finish, out_w, out_h, cancel, on_frame, limit)
        return
    _export_cpu(path, dst, codec, fps_text, info, finish, out_w, out_h, cancel, on_frame, limit)


def _export_gpu(path, dst, codec, fps_text, info, finish, out_w, out_h, cancel, on_frame, limit=None) -> None:
    """El mismo acabado que el raw: warp y YUV en la tarjeta, ProRes NV o NVENC."""
    from reveal_gpu import GpuReveal, open_prores_nv
    from reveal_pipeline import coerce_reveal

    shift = 0 if codec == "prores_4444" else 6
    try:
        gpu = GpuReveal(
            coerce_reveal({}), info.width, info.height, mode="COLOR",
            shift=shift, out_w=out_w, out_h=out_h, finish=finish,
            chroma422=codec == "prores_4444",
        )
    except Exception as exc:
        raise VideoError(f"No pude usar la tarjeta. {exc}") from exc
    errors: list[str] = []
    if codec == "prores_4444":
        proc = open_prores_nv(
            dst, out_w, out_h, 0, stderr=subprocess.PIPE, rate=fps_text, creationflags=_NO_WINDOW,
        )
    else:
        proc = _open_hevc(dst, out_w, out_h, fps_text)
    drain = threading.Thread(target=_collect_err, args=(proc.stderr, errors), daemon=True)
    drain.start()
    indexes = _indexes(finish, info.frames, limit)
    _write_frames(
        path, info, proc, errors, drain, cancel, on_frame, dst,
        None, None, _yuv_packets(path, info, gpu, cancel, indexes), len(indexes),
    )


def _yuv_packets(path, info, gpu, cancel, indexes):
    """La decodificación va un cuadro adelante. El acabado corre mientras llega el siguiente.

    Si la salida no es un tramo seguido, los cuadros se reordenan desde un archivo temporal.
    """
    span = _forward_span(indexes)
    if span is None:
        stream = iter_yuv422(path, info.width, info.height, cancel)
        for index, raw in _select_ordered(stream, indexes):
            if cancel():
                break
            yield gpu.frame_yuv422(raw, index)
        return
    start, end = span
    stop = threading.Event()
    found: list = []
    pending: queue.Queue = queue.Queue(maxsize=2)

    def produce():
        try:
            for index, raw in enumerate(iter_yuv422(path, info.width, info.height, cancel)):
                if stop.is_set() or cancel():
                    break
                if index > end:
                    break
                while not stop.is_set():
                    try:
                        pending.put((index, raw), timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            found.append(exc)
        finally:
            while not stop.is_set():
                try:
                    pending.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = pending.get()
            if item is None:
                break
            index, raw = item
            if index < start:
                continue
            if index > end:
                break
            yield gpu.frame_yuv422(raw, index)
    finally:
        stop.set()
        while True:
            try:
                pending.get_nowait()
            except queue.Empty:
                break
        thread.join(timeout=3)
    if found:
        raise found[0]


def _open_hevc(dst, out_w, out_h, fps_text):
    cmd = [
        ffmpeg_bin(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "yuv444p10msble",
        "-r", str(fps_text), "-i", "pipe:0",
    ]
    _append_ffmpeg_codec_args(cmd, "hevc_nvenc")
    cmd.append(str(dst))
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
    )


def _export_cpu(path, dst, codec, fps_text, info, finish, out_w, out_h, cancel, on_frame, limit=None) -> None:
    hevc = codec == "hevc_nvenc"
    pix = "yuv444p10msble" if hevc else "rgb48le"
    cmd = [
        ffmpeg_bin(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", pix,
        "-r", str(fps_text), "-i", "pipe:0",
    ]
    _append_ffmpeg_codec_args(cmd, codec)
    cmd.append(str(dst))
    errors: list[str] = []
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
    )
    drain = threading.Thread(target=_collect_err, args=(proc.stderr, errors), daemon=True)
    drain.start()

    def payload_of(image, index):
        out = finish.apply_u16(image, index, (out_w, out_h))
        if out.shape[1] != out_w or out.shape[0] != out_h:
            raise VideoError("El cuadro de salida no coincide con el tamaño pedido.")
        raw = np.ascontiguousarray(out).tobytes()
        if hevc:
            from reveal_pipeline import yuv10_from_rgb48
            return yuv10_from_rgb48(raw, out_w, out_h)
        return raw

    indexes = _indexes(finish, info.frames, limit)

    def packets():
        stream = iter_rgb48(path, info.width, info.height, cancel)
        for index, image in _select_ordered(stream, indexes):
            if cancel():
                raise Cancelled()
            yield payload_of(image, index)

    _write_frames(
        path, info, proc, errors, drain, cancel, on_frame, dst, None, None, packets(), len(indexes),
    )


def _write_frames(path, info, proc, errors, drain, cancel, on_frame, dst, payload_of, limit=None, packets=None, total=None) -> None:
    written = 0
    failed: Exception | None = None
    total = info.frames if total is None else int(total)
    try:
        if packets is None:
            packets = (
                payload_of(frame, index)
                for index, frame in enumerate(iter_rgb48(path, info.width, info.height, cancel))
            )
        for payload in packets:
            if cancel():
                raise Cancelled()
            try:
                proc.stdin.write(payload)
            except (BrokenPipeError, OSError) as exc:
                raise VideoError("ffmpeg cerró la salida antes de tiempo.") from exc
            written += 1
            if on_frame:
                on_frame(written, total)
            if written >= total or (limit is not None and written >= limit):
                break
        if written <= 0 and failed is None:
            failed = VideoError("No salió ningún cuadro.")
    except Exception as exc:
        failed = exc
        if proc.poll() is None:
            proc.kill()
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass
        code = proc.wait()
        drain.join(timeout=2)
    if isinstance(failed, Cancelled) or cancel():
        _drop(dst)
        raise Cancelled()
    if failed is not None or code != 0:
        _drop(dst)
        tail = "".join(errors)[-500:].strip()
        if isinstance(failed, VideoError) and not tail:
            raise failed
        raise VideoError(tail or (str(failed) if failed else f"ffmpeg terminó con código {code}."))


def _drop(path: Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


def _collect_err(pipe, sink: list):
    if pipe is None:
        return
    for line in iter(pipe.readline, b""):
        sink.append(line.decode("utf-8", "replace"))
