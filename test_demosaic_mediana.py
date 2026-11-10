"""Tercera prueba: edge-aware lineal y mediana 3x3 en R-G y B-G.

El demosaico del programa no aplica este filtro. Corre en CPU, después del
mismo edge-aware de la prueba anterior. Una ida a la GPU solo para la
mediana 3x3 costaría más que el filtro.

G no se toca. R y B se reconstruyen como G más la mediana de la diferencia.
Salida: TIFF 16-bit lineal, sin revelar.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import test_demosaic_gharbi as base

START = 5055
COUNT = 20
OUT = Path(r"E:\prueba_demosaic\carrasco_002_5055-5074\mediana_rg_bg")
NOTE = (
    "Lineal, sin revelar. Edge-aware BayerBG y mediana 3x3 de R-G y B-G. "
    "G queda igual."
)


def mediana_diferencias(lineal: np.ndarray) -> tuple[np.ndarray, float]:
    """RGB lineal float32 (H, W, 3) → el mismo tamaño, mediana 3x3 en R-G y B-G."""
    t0 = time.perf_counter()
    verde = lineal[:, :, 1]
    dr = cv2.medianBlur(lineal[:, :, 0] - verde, 3)
    db = cv2.medianBlur(lineal[:, :, 2] - verde, 3)
    limpio = np.empty_like(lineal)
    limpio[:, :, 0] = dr + verde
    limpio[:, :, 1] = verde
    limpio[:, :, 2] = db + verde
    np.clip(limpio, 0.0, 1.0, out=limpio)
    return limpio, time.perf_counter() - t0


def main():
    end = START + COUNT - 1
    nframes = base.DEFAULT_RAW.stat().st_size // base.FRAME_BYTES
    if end >= nframes:
        print(f"El rollo tiene {nframes} cuadros. Pediste hasta {end}.")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    times_edge = []
    times_med = []
    write_s = 0.0
    read_s = 0.0
    t_all = time.perf_counter()

    with base.DEFAULT_RAW.open("rb") as handle:
        for n, index in enumerate(range(START, end + 1), start=1):
            t0 = time.perf_counter()
            bayer = base.read_bayer(handle, index)
            read_s += time.perf_counter() - t0

            lineal, dt_edge = base.demosaic_edge(bayer)
            limpio, dt_med = mediana_diferencias(lineal)
            times_edge.append(dt_edge)
            times_med.append(dt_med)

            t0 = time.perf_counter()
            base.save_linear_tiff(
                limpio,
                OUT / f"carrasco_002_{index:06d}.tif",
                NOTE,
            )
            write_s += time.perf_counter() - t0

            if n == 1 or n == COUNT or n % 10 == 0:
                print(
                    f"{index} ({n}/{COUNT})  "
                    f"edge {dt_edge * 1000:.1f} ms  "
                    f"mediana {dt_med * 1000:.1f} ms",
                    flush=True,
                )

    total_s = time.perf_counter() - t_all
    edge = base._stats(times_edge)
    med = base._stats(times_med)
    report = {
        "rollo": str(base.DEFAULT_RAW),
        "cuadro_inicio": START,
        "cuadro_fin": end,
        "cantidad": COUNT,
        "algoritmo": "OpenCV COLOR_BayerBG2RGB_EA y mediana 3x3 de R-G y B-G en CPU",
        "verde": "sin filtrar",
        "carpeta": str(OUT),
        "lectura_s": read_s,
        "escritura_s": write_s,
        "total_s": total_s,
        "edge_aware": edge,
        "mediana": med,
        "tiff": "RGB lineal de 16 bits, sin compresión, blanco 65535, sin balance ni curva",
    }
    parent = OUT.parent
    (parent / "tiempos_mediana.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text = "\n".join([
        "Tercera prueba. Demosaico lineal y mediana 3x3 en R-G y B-G.",
        f"Cuadros {START} a {end} ({COUNT} cuadros, índice del visor).",
        "El verde no se filtra. Rojo y azul salen de G más la mediana de la diferencia.",
        "CPU. El cuadro no va a la GPU.",
        "",
        f"Edge-aware: {edge['suma_s']:.2f} s, media {edge['media_ms']:.2f} ms, "
        f"mediana {edge['mediana_ms']:.2f} ms",
        f"Filtro mediana: {med['suma_s']:.2f} s, media {med['media_ms']:.2f} ms, "
        f"mediana {med['mediana_ms']:.2f} ms",
        f"Escritura TIFF: {write_s:.2f} s",
        f"Lectura del raw: {read_s:.2f} s",
        f"Total: {total_s:.2f} s",
        "",
        f"Carpeta: {OUT}",
    ])
    (parent / "tiempos_mediana.txt").write_text(text + "\n", encoding="utf-8")
    print(
        f"Listo. Mediana {med['suma_s']:.2f} s "
        f"({med['media_ms']:.2f} ms por cuadro). {parent / 'tiempos_mediana.txt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
