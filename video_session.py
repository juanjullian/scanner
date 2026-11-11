"""Lista de videos a tratar. Los seteos de cada archivo siguen al lado del video."""
from __future__ import annotations

import json
from pathlib import Path

import film_finish

VIDEO_EXTS = {".mov", ".mp4", ".mkv"}


def session_path(root: Path | None = None) -> Path:
    if root is None:
        root = _data_root()
    folder = Path(root)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "video_finish_session.json"


def _data_root() -> Path:
    persist = Path(__file__).resolve().parent / "persist.json"
    if persist.is_file():
        try:
            data = json.loads(persist.read_text(encoding="utf-8"))
            folder = data.get("root_folder")
            if folder:
                return Path(folder)
        except (OSError, json.JSONDecodeError):
            pass
    return Path.home() / "Documents" / "Archivo_Scan_Data"


def empty_session() -> dict:
    return {"version": 1, "gammac": {}, "files": []}


def load_session(path: Path) -> dict:
    if not path.is_file():
        return empty_session()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_session()
    if not isinstance(data, dict):
        return empty_session()
    files = []
    for item in data.get("files") or []:
        if isinstance(item, dict) and item.get("path"):
            files.append(item)
    out = empty_session()
    out["files"] = files
    if isinstance(data.get("gammac"), dict):
        out["gammac"] = data["gammac"]
    return out


def save_session(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def is_video(path: Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def missing_files(data: dict) -> list[dict]:
    return [item for item in data.get("files") or [] if not Path(item.get("path", "")).is_file()]


def match_folder(missing: list[dict], folder: Path) -> dict[str, Path]:
    """Empareja por nombre de archivo dentro de la carpeta nueva."""
    folder = Path(folder)
    found = {}
    for item in missing:
        old = Path(item.get("path", ""))
        candidate = folder / old.name
        if candidate.is_file() and is_video(candidate):
            found[str(old)] = candidate
    return found


def meaningful_adjust(data: dict | None) -> bool:
    if not isinstance(data, dict):
        return False
    if film_finish.has_adjust(data):
        return True
    grade = data.get("grade") or {}
    if grade.get("mode") == "bw":
        return True
    try:
        sat = float(grade.get("sat", 1.0))
    except (TypeError, ValueError):
        sat = 1.0
    return abs(sat - 1.0) > 0.001


def same_parent(items: list[dict]) -> Path | None:
    parents = {str(Path(item.get("path", "")).parent) for item in items if item.get("path")}
    if len(parents) == 1:
        return Path(next(iter(parents)))
    return None
