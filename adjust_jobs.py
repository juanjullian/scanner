"""Cola de perforación. Vive en la ventana principal, no en el diálogo."""
from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal

import film_finish
import perfo_lock


class _Run(QThread):
    progressed = pyqtSignal(int, int)
    finished_run = pyqtSignal(bool, str)

    def __init__(self, raw: Path, roi, ref_frame: int, workers: int):
        super().__init__()
        self.raw = Path(raw)
        self.roi = tuple(roi)
        self.ref_frame = int(ref_frame)
        self.workers = int(workers)

    def run(self):
        existing = film_finish.read_perfo(self.raw) or {}
        saved = existing.get("roi")
        same_zone = bool(saved) and tuple(int(v) for v in saved) == tuple(self.roi)
        try:
            same_ref = same_zone and int(existing.get("ref")) == int(self.ref_frame)
        except (TypeError, ValueError):
            same_ref = False
        film_finish.write_perfo_status(
            self.raw, "running", "Midiendo perforación",
            roi=self.roi,
            ref=self.ref_frame,
            partial=existing.get("partial") if same_ref else [],
        )
        try:
            result = perfo_lock.analyze_file(
                self.raw,
                self.roi,
                ref_index=self.ref_frame,
                workers=self.workers,
                on_progress=self._pulse,
            )
        except perfo_lock.PerfoError as exc:
            film_finish.write_perfo_status(self.raw, "error", str(exc))
            self.finished_run.emit(False, str(exc))
            return
        except Exception as exc:
            film_finish.write_perfo_status(self.raw, "error", str(exc))
            self.finished_run.emit(False, str(exc))
            return
        film_finish.write_perfo_status(self.raw, "done", "Lista", result)
        self.finished_run.emit(True, "")

    def _pulse(self, done: int, total: int, rows=None, ref=None, ident=None):
        partial = None
        if rows is not None:
            partial = [
                None if row is None else [round(float(row[0]), 3), round(float(row[1]), 3), round(float(row[2]), 3), int(row[3])]
                for row in rows
            ]
        try:
            film_finish.write_perfo_status(
                self.raw, "running", f"{done}/{total}",
                partial=partial, roi=ident or self.roi, ref=self.ref_frame if ref is None else ref,
            )
        except OSError:
            pass
        self.progressed.emit(done, total)


class AdjustJobs(QObject):
    progress = pyqtSignal(str, int, int, str)
    settled = pyqtSignal(str, bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: list[tuple] = []
        self._thread: _Run | None = None
        self._name = ""

    def enqueue(self, raw: Path, roi, ref_frame: int = 0, workers: int | None = None, force: bool = False):
        raw = Path(raw)
        zone = tuple(int(v) for v in roi)
        if not force and film_finish.perfo_is_done(raw, zone):
            return False
        if self._thread is not None and self._thread.raw == raw:
            return False
        if any(item[0] == raw for item in self._queue):
            return False
        count = workers if workers is not None else min(8, max(2, (os.cpu_count() or 4) - 2))
        host = self.parent()
        if host is not None and getattr(host, "is_exporting_batch", False):
            count = min(count, 4)
        self._queue.append((raw, zone, int(ref_frame), count))
        self._kick()
        return True

    def resume_folder(self, folder: Path):
        """Vuelve a encolar lo que quedó a medias cuando se cerró el programa."""
        folder = Path(folder)
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.perfo.json")):
            data = film_finish.read_perfo(folder / path.name[: -len(".perfo.json")])
            if not data or data.get("status") != "running":
                continue
            raw = folder / path.name[: -len(".perfo.json")]
            if not raw.is_file():
                continue
            adjust = film_finish.load(raw)
            roi = (adjust.get("perfo") or {}).get("roi") or data.get("roi")
            if not roi:
                continue
            ref = film_finish.perfo_ref(adjust.get("perfo") or {})
            if ref is None:
                ref = int(data.get("ref") or 0)
            self.enqueue(raw, roi, ref_frame=ref, force=True)

    def running_name(self) -> str:
        return self._name

    def _kick(self):
        if self._thread is not None or not self._queue:
            return
        raw, roi, ref_frame, workers = self._queue.pop(0)
        self._name = raw.name
        self._thread = _Run(raw, roi, ref_frame, workers)
        self._thread.progressed.connect(self._on_progress)
        self._thread.finished_run.connect(self._on_done)
        self.progress.emit(self._name, 0, 1, "en curso")
        self._thread.start()

    def _on_progress(self, done: int, total: int):
        self.progress.emit(self._name, done, total, "en curso")

    def _on_done(self, ok: bool, message: str):
        name = self._name
        self.settled.emit(name, ok, message)
        thread = self._thread
        self._thread = None
        self._name = ""
        if thread is not None:
            thread.wait(1000)
        self._kick()
