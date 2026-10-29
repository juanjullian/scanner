# Bitácora — Lucid Scanner

## 2026-10-29 — Revelado Rec.709, HEVC NVENC y lote desde DNG

### Por qué
El master de visionado tenía que salir del Bayer sin pasar por Resolve. Un solo ajuste de revelado, aplicado a todos los rollos elegidos. ProRes 4444 XQ queda en el CPU. HEVC 4:4:4 de 10 bits, lossless, usa la RTX 5060 Ti cuando se quiere el mismo look con el encoder de la tarjeta. Las secuencias DNG que ya estaban exportadas tenían que poder revelarse igual, aunque el raw ya no estuviera.

### Cómo
- Pantalla opcional de revelado, un preset guardado para todos los rollos: balance de rojo y azul, EV, curva logarítmica, levante de negro y detalle de medios tonos. La curva es una tabla. El demosaico es el edge-aware BayerBG del CPU. En 0 de detalle no se calculan los desenfoques.
- ProRes 4444 XQ con `prores_ks`, perfil 5, qscale 4, 10 bits 4:4:4 y 4 workers de revelado. El resto del CPU codifica.
- HEVC 4:4:4 lossless NVENC, perfil Rext, 10 bits, con 12 workers en este i7. Los workers solo revelan a RGB. Otro proceso arma el YUV 4:4:4 de 10 bits en la tarjeta y se lo entrega a FFmpeg, para que esa conversión no le quite el procesador al revelado.
- El filtro de cuadros magenta es el mismo de la secuencia DNG.
- En Opciones, **Revelar secuencias DNG…** toma carpetas de DNG exportadas por esta app, el revelado guardado, el codec (ProRes XQ o HEVC) y una carpeta de salida. No vuelve a filtrar magenta.
- En **Confirmar exportación** se puede elegir la carpeta de salida. Al abrir queda la de la colección, junto al archivo. **Junto al archivo** la restaura. El nombre de cada salida no cambia.
- Al borrar un raw también se borra la carpeta `_EXR_SEQ`, igual que la `_DNG_SEQ`.

### Hecho
- Exportación de raw Bayer a ProRes 4444 XQ y a HEVC 4:4:4 lossless con el mismo revelado.
- Lote de secuencias DNG hacia esos dos codecs.
- Carpeta de salida elegible en la confirmación de exportación, con el destino de siempre como valor inicial.

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
