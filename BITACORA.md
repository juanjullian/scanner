# Bitácora — Lucid Scanner

## 2026-11-10 — Balance de la cámara, contenedor y revelado en la GPU

### Por qué
El visor de captura se veía más neutro que el revelado: la cámara ya aplica su balance (rojo ×1,92, azul ×2,32) y la curva del programa partía de 1. Un nombre de salida guardado de otro codec podía dejar un ProRes en `.mp4`. Y el edge-aware del CPU, repartido en workers, no usa la 5060 Ti. Malvar en la tarjeta se ve igual, pero escribe más: en 20 cuadros del mismo tramo, con el mismo ProRes, fueron 134 MB contra 123 MB del edge-aware.

### Cómo
- 1,00 en los sliders de rojo y azul es el balance de la PHX. La curva lo multiplica. El verde sigue en 1.
- Al exportar, la extensión la manda el codec. ProRes y CineForm quedan en `.mov`.
- En Ajustes de revelado se elige CPU o GPU. El preset sigue partiendo en CPU.
- CPU: edge-aware, 4 workers, `prores_ks`. El HEVC lossless revela en el CPU y codifica con NVENC.
- GPU: el Bayer empaquetado de 12 bits sube tal cual. En la tarjeta se desempaqueta, corre NPP (`nppiCFAToRGB` de 16 bits, códigos 0–4095, sin correrlos a la izquierda), la curva, el detalle y el YUV. El ProRes 4444 XQ lo escribe `prores_nv` y el ancho pasa al múltiplo de 16 (2840 → 2848). El HEVC lossless arma el YUV en la tarjeta y lo entrega a NVENC. Un solo proceso. El filtro magenta sigue en el CPU. La vista de los ajustes sigue con el edge-aware.
- En 20 cuadros de `f8_99` (450 a 469), mismo revelado y `prores_ks`: edge-aware 123,3 MB, NPP 122,7 MB, bilineal 117,3 MB, Malvar 134,1 MB.
- Quedaron afuera del repo el `persist.json` de esta máquina, los `__pycache__`, copias locales de `config_bayer`, los manuales de la PHX y los pesos de Gharbi (el script los vuelve a bajar).

### Hecho
- El revelado arranca en el blanco de la cámara.
- El archivo de video sale con la extensión de su codec.
- Se puede revelar y codificar el ProRes en la GPU con NPP, o seguir en el CPU.

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
