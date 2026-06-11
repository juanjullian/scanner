# Bitácora — Lucid Scanner

## 2026-06-11 — Exportación y bloqueo de cierre

### Exportación de video (no-DNG)
- Nuevo diálogo `ExportConfirmDialog` (`export_grade.py`): tabla con miniatura, nombre de salida editable y FPS por archivo.
- Eliminado ajuste de gamma/black level en la exportación; QOI/RGB usan el ISP de Arena sin tono extra (evita salidas demasiado oscuras).
- La exportación arranca al confirmar, sin mensaje intermedio de cola.
- `raw2video.py`: soporte `--output`; sin `--grade`.
- `l2t.py`: soporte `--output-dir`; sin `--grade`.
- Módulos auxiliares: `bayer_render.py`, `qoi_utils.py`.

### Cierre de la aplicación durante captura
- La X de cerrar no cierra el programa mientras `is_recording`; se muestra toast: *Captura en curso — detén la grabación antes de cerrar*.

### Calibración de cámara
- Se revirtieron experimentos de reinicio completo del worker y modos preview/record.
- Flujo restaurado: `request_calibration_live_mode(True/False)` con `enter_calib` / `exit_calib` y reaplicación de `config_bayer.txt` + exposición del slider al salir.
