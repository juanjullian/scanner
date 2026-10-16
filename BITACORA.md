# Bitácora — Lucid Scanner

## 2026-10-16 — Restaurar la cámara al salir de calibración

### Por qué
Al abrir **Calibrar posición de cámara** la PHX pasa a corrida libre: `TriggerMode Off`, `ExposureAuto Continuous` y `GainAuto Continuous`. Al cerrar el visor se reaplicaba `config_bayer.txt` en caliente. Ese archivo escribe `TriggerMode` y `Gain` antes de apagar la autoexposición y la autoganancia, y esos nodos no son escribibles mientras el automático sigue activo. El disparo no volvía, y la ganancia alta de la calibración se quedaba pegada (imagen con mucho ruido). Solo un cierre completo del programa recargaba el user set de fábrica.

### Cómo
- Al cerrar el visor se destruye el `CameraWorker` y se crea uno nuevo, el mismo camino que al abrir el programa (`setup_camera` + exposición, preview e ISP actuales).
- Después de cargar `config_bayer.txt`, `_force_manual_capture_levels` apaga `ExposureAuto` y `GainAuto`, fija `Gain` y black level en 0, y deja `TriggerMode On` con fuente `Line0`.
- Mientras rearma, el indicador queda naranja y el toast dice que se está restaurando la configuración.

### Hecho
- Salir de calibración devuelve el disparo externo y la ganancia manual del arranque, sin tener que cerrar el programa.

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
