"""Prueba aparte del programa: demosaico lineal, sin revelado, a TIFF 16-bit.

Compara dos métodos sobre Carrasco_002, cuadros del visor (índice desde 0,
el mismo número que muestra el slider):

- edge-aware: el demosaico del revelado, OpenCV COLOR_BayerBG2RGB_EA.
- gharbi: Deep Joint Demosaicing, red publicada de Gharbi et al. (SIGGRAPH
  Asia 2016), pesos bayer.pth de mgharbi/demosaicnet. Corre en la GPU.

El TIFF es RGB lineal de 16 bits, sin compresión. El blanco es el código 4095
del sensor. No hay balance, EV ni curva de revelado.

La red se entrenó con imágenes sRGB. Cada muestra lineal se codifica con la
OETF sRGB, se demosaica, y se devuelve a lineal con la curva inversa. El
patrón de los pesos es GRBG; el sensor es RGGB, así que el mosaico se
rellena por reflexión (33 px a la izquierda, 32 arriba) para alinear la fase
y alejar el borde del recorte válido de la red.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

import bayer_render

WIDTH = 2840
HEIGHT = 2200
FRAME_BYTES = int(WIDTH * HEIGHT * 1.5)
RAW12_MAX = 4095
WHITE16 = 65535

DEFAULT_RAW = Path(r"E:\filmscanner_export\Carrasco\Carrasco_002.raw")
DEFAULT_OUT = Path(r"E:\prueba_demosaic\carrasco_002_5055-5500")
WEIGHTS_URL = (
    "https://raw.githubusercontent.com/mgharbi/demosaicnet/master/"
    "demosaicnet/data/bayer.pth"
)
WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "gharbi_bayer.pth"

# Margen para que el cuadro original quede fuera del borde de la convolución
# válida (31 px) y para pasar de RGGB a la fase GRBG de los pesos.
PAD_TOP = 32
PAD_BOTTOM = 32
PAD_LEFT = 33
PAD_RIGHT = 33

def save_linear_tiff(linear_rgb: np.ndarray, path: Path, note: str):
    """RGB lineal float 0–1 (H, W, 3) → TIFF 16-bit sin compresión y sin revelar.

    El blanco del sensor (código 4095) queda en 65535. No es un DNG: el DNG de
    este programa guarda el mosaico Bayer, y Resolve no abre un RGB ya demosaicado.
    """
    image = np.clip(np.ascontiguousarray(linear_rgb) * WHITE16 + 0.5, 0, WHITE16)
    image = image.astype(np.uint16)
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        planarconfig="contig",
        compression=None,
        metadata=None,
        extratags=[(270, "s", 0, note, True)],
    )


def ensure_weights(path: Path) -> Path:
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Bajando pesos Gharbi a {path}")
    urllib.request.urlretrieve(WEIGHTS_URL, path)
    return path


class BayerDemosaick(nn.Module):
    """Red publicada en demosaicnet (Gharbi 2016), profundidad 15, ancho 64.

    Es la versión liberada: al final separa filtros y máscaras y los multiplica.
    Con pad=True la salida conserva el tamaño de entrada.
    """

    def __init__(self):
        super().__init__()
        depth = 15
        width = 64
        layers = OrderedDict([
            ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),
        ])
        for i in range(depth):
            n_in = 4 if i == 0 else width
            n_out = 2 * width if i == depth - 1 else width
            layers[f"conv{i + 1}"] = nn.Conv2d(n_in, n_out, 3, padding=1)
            layers[f"relu{i + 1}"] = nn.ReLU(inplace=True)
        self.width = width
        self.main_processor = nn.Sequential(layers)
        self.residual_predictor = nn.Conv2d(width, 12, 1)
        self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)
        self.fullres_processor = nn.Sequential(OrderedDict([
            ("post_conv", nn.Conv2d(6, width, 3, padding=1)),
            ("post_relu", nn.ReLU(inplace=True)),
            ("output", nn.Conv2d(width, 3, 1)),
        ]))

    def forward(self, mosaic):
        features = self.main_processor(mosaic)
        filters = features[:, 0:self.width]
        masks = features[:, self.width:2 * self.width]
        residual = self.residual_predictor(filters * masks)
        upsampled = self.upsampler(residual)
        packed = torch.cat([mosaic, upsampled], 1)
        return self.fullres_processor(packed)


def load_gharbi(path: Path, device: torch.device) -> BayerDemosaick:
    model = BayerDemosaick()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(cleaned)
    model.eval()
    return model.to(device)


def srgb_encode(linear):
    linear = linear.clamp(0, 1)
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * torch.pow(linear, 1.0 / 2.4) - 0.055,
    )


def srgb_decode(encoded):
    encoded = encoded.clamp(0, 1)
    return torch.where(
        encoded <= 0.04045,
        encoded / 12.92,
        torch.pow((encoded + 0.055) / 1.055, 2.4),
    )


def pack_grbg(encoded_hw: torch.Tensor) -> torch.Tensor:
    """Mosaico de un canal, fase GRBG, a las tres capas con ceros que espera la red."""
    mosaic = torch.zeros(
        1, 3, encoded_hw.shape[0], encoded_hw.shape[1],
        device=encoded_hw.device, dtype=encoded_hw.dtype,
    )
    mosaic[0, 1, 0::2, 0::2] = encoded_hw[0::2, 0::2]
    mosaic[0, 0, 0::2, 1::2] = encoded_hw[0::2, 1::2]
    mosaic[0, 2, 1::2, 0::2] = encoded_hw[1::2, 0::2]
    mosaic[0, 1, 1::2, 1::2] = encoded_hw[1::2, 1::2]
    return mosaic


def demosaic_edge(bayer_u16: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    rgb16 = cv2.cvtColor(np.ascontiguousarray(bayer_u16), cv2.COLOR_BayerBG2RGB_EA)
    linear = np.clip(rgb16.astype(np.float32) * (1.0 / RAW12_MAX), 0.0, 1.0)
    return linear, time.perf_counter() - t0


def demosaic_gharbi(bayer_u16: np.ndarray, model: BayerDemosaick, device) -> tuple[np.ndarray, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    linear = torch.from_numpy(np.ascontiguousarray(bayer_u16)).to(device).float()
    linear.mul_(1.0 / RAW12_MAX)
    encoded = srgb_encode(linear)
    encoded = F.pad(encoded[None, None], (PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM), mode="reflect")
    mosaic = pack_grbg(encoded[0, 0])
    with torch.inference_mode():
        rgb = model(mosaic)
    if rgb.shape[-2] != HEIGHT + PAD_TOP + PAD_BOTTOM or rgb.shape[-1] != WIDTH + PAD_LEFT + PAD_RIGHT:
        raise RuntimeError(f"La red devolvió {tuple(rgb.shape)}, no el tamaño rellenado.")
    rgb = rgb[:, :, PAD_TOP:PAD_TOP + HEIGHT, PAD_LEFT:PAD_LEFT + WIDTH]
    out = srgb_decode(rgb.clamp(0, 1))
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).contiguous()
    cpu = out.cpu().numpy()
    torch.cuda.synchronize()
    return cpu, time.perf_counter() - t0


def read_bayer(handle, index: int) -> np.ndarray:
    handle.seek(index * FRAME_BYTES)
    packed = handle.read(FRAME_BYTES)
    if len(packed) != FRAME_BYTES:
        raise RuntimeError(f"Cuadro {index} incompleto ({len(packed)} bytes).")
    return bayer_render.unpack_12bit_le(packed, WIDTH, HEIGHT)


def _stats(samples: list[float]) -> dict:
    arr = np.asarray(samples, dtype=np.float64)
    steady = arr[1:] if len(arr) > 1 else arr
    return {
        "cuadros": int(arr.size),
        "suma_s": float(arr.sum()),
        "media_ms": float(arr.mean() * 1000.0),
        "mediana_ms": float(np.median(arr) * 1000.0),
        "media_sin_el_primero_ms": float(steady.mean() * 1000.0),
    }


def write_report(path: Path, report: dict):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    edge = report["edge_aware"]["demosaico"]
    gharbi = report["gharbi"]["demosaico"]
    lines = [
        "Prueba de demosaico sin revelar. Carrasco_002.",
        f"Cuadros {report['cuadro_inicio']} a {report['cuadro_fin']} "
        f"({report['cantidad']} cuadros, índice del visor, desde 0).",
        "",
        "Edge-aware (OpenCV COLOR_BayerBG2RGB_EA, el del revelado):",
        f"  demosaico {edge['suma_s']:.2f} s, media {edge['media_ms']:.2f} ms, "
        f"mediana {edge['mediana_ms']:.2f} ms",
        f"  escritura TIFF {report['edge_aware']['escritura_s']:.2f} s",
        "",
        "Gharbi 2016 (GPU, pesos bayer.pth, salida devuelta a lineal):",
        f"  demosaico {gharbi['suma_s']:.2f} s, media {gharbi['media_ms']:.2f} ms, "
        f"mediana {gharbi['mediana_ms']:.2f} ms",
        f"  escritura TIFF {report['gharbi']['escritura_s']:.2f} s",
        "",
        f"Lectura del raw {report['lectura_s']:.2f} s.",
        f"Total de la prueba {report['total_s']:.2f} s.",
        "",
        "Los TIFF son RGB lineal de 16 bits, sin compresión. Blanco del sensor = 65535. Sin balance ni curva.",
        report["nota_gharbi"],
    ]
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prueba Gharbi vs edge-aware, TIFF lineal.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", type=int, default=5055)
    parser.add_argument("--end", type=int, default=5500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No hay CUDA. Esta prueba del modelo de Gharbi corre en la GPU.")
        return 1
    device = torch.device("cuda")
    if args.end < args.start:
        print("El cuadro final es anterior al inicial.")
        return 1

    nframes = args.raw.stat().st_size // FRAME_BYTES
    if args.end >= nframes:
        print(f"El rollo tiene {nframes} cuadros (0 a {nframes - 1}). Pediste hasta {args.end}.")
        return 1

    weights = ensure_weights(WEIGHTS_PATH)
    model = load_gharbi(weights, device)
    print(
        f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
        f"pesos {weights.name}"
    )

    folder_edge = args.out / "edge_aware"
    folder_gharbi = args.out / "gharbi"
    folder_edge.mkdir(parents=True, exist_ok=True)
    folder_gharbi.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, HEIGHT + PAD_TOP + PAD_BOTTOM, WIDTH + PAD_LEFT + PAD_RIGHT, device=device)
    with torch.inference_mode():
        warm = model(dummy)
    torch.cuda.synchronize()
    if warm.shape[-2:] != dummy.shape[-2:]:
        print(f"El calentamiento devolvió {tuple(warm.shape)}.")
        return 1
    del dummy, warm

    indices = range(args.start, args.end + 1)
    note_edge = "Lineal, sin revelar. Demosaico edge-aware BayerBG (RGGB del sensor)."
    note_gharbi = (
        "Lineal, sin revelar. Demosaico Gharbi 2016 en GPU. "
        "OETF sRGB de ida y vuelta. Fase GRBG por reflejo de 33 px."
    )
    times_edge = []
    times_gharbi = []
    write_edge = 0.0
    write_gharbi = 0.0
    read_s = 0.0
    t_all = time.perf_counter()

    with args.raw.open("rb") as handle:
        for n, index in enumerate(indices, start=1):
            t0 = time.perf_counter()
            bayer = read_bayer(handle, index)
            read_s += time.perf_counter() - t0

            linear_edge, dt_edge = demosaic_edge(bayer)
            linear_gharbi, dt_gharbi = demosaic_gharbi(bayer, model, device)
            times_edge.append(dt_edge)
            times_gharbi.append(dt_gharbi)

            t0 = time.perf_counter()
            save_linear_tiff(
                linear_edge,
                folder_edge / f"carrasco_002_{index:06d}.tif",
                note_edge,
            )
            write_edge += time.perf_counter() - t0

            t0 = time.perf_counter()
            save_linear_tiff(
                linear_gharbi,
                folder_gharbi / f"carrasco_002_{index:06d}.tif",
                note_gharbi,
            )
            write_gharbi += time.perf_counter() - t0

            if n == 1 or n % 25 == 0 or index == args.end:
                print(
                    f"{index} ({n}/{args.end - args.start + 1})  "
                    f"edge {dt_edge * 1000:.1f} ms  "
                    f"gharbi {dt_gharbi * 1000:.1f} ms  "
                    f"media gharbi {np.mean(times_gharbi) * 1000:.1f} ms",
                    flush=True,
                )

    total_s = time.perf_counter() - t_all
    report = {
        "rollo": str(args.raw),
        "cuadro_inicio": args.start,
        "cuadro_fin": args.end,
        "cantidad": args.end - args.start + 1,
        "indice": "El mismo número del slider del visor, desde 0, inclusive.",
        "salida": str(args.out),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "pesos": str(weights),
        "lectura_s": read_s,
        "total_s": total_s,
        "edge_aware": {
            "algoritmo": "OpenCV COLOR_BayerBG2RGB_EA",
            "demosaico": _stats(times_edge),
            "escritura_s": write_edge,
            "carpeta": str(folder_edge),
        },
        "gharbi": {
            "algoritmo": "BayerDemosaick Gharbi 2016, depth 15, width 64, pad mismo tamaño",
            "fase": "GRBG de los pesos; RGGB del sensor alineado con reflejo",
            "relleno_px": {"arriba": PAD_TOP, "abajo": PAD_BOTTOM, "izquierda": PAD_LEFT, "derecha": PAD_RIGHT},
            "dominio": "OETF sRGB antes de la red y OETF inversa después. El TIFF queda lineal.",
            "demosaico": _stats(times_gharbi),
            "escritura_s": write_gharbi,
            "carpeta": str(folder_gharbi),
        },
        "nota_gharbi": note_gharbi,
        "tiff": {
            "bits": 16,
            "compresion": "ninguna",
            "white_level": WHITE16,
            "revelado": "ninguno",
        },
    }
    report_path = args.out / "tiempos.json"
    write_report(report_path, report)
    print(
        f"Listo. Edge-aware {report['edge_aware']['demosaico']['suma_s']:.2f} s, "
        f"Gharbi {report['gharbi']['demosaico']['suma_s']:.2f} s. "
        f"Informe: {report_path.with_suffix('.txt')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
