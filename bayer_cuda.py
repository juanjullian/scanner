"""Demosaico Malvar en la GPU (CuPy) para Bayer RGGB.

La exportación en GPU ya no usa este módulo: el demosaico de ese camino es NPP.


En este OpenCV, COLOR_BayerBG2RGB deja el píxel (0,0) en el rojo. El sensor
es BayerRG12p, así que (0,0) es rojo: el kernel usa ese mismo patrón RGGB.
"""
from __future__ import annotations

import numpy as np

_KERNEL = r"""
extern "C" __global__
void malvar_rggb(const unsigned short* __restrict__ src,
                  unsigned short* __restrict__ dst,
                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    auto at = [&](int yy, int xx) -> int {
        yy = yy < 0 ? 0 : (yy >= height ? height - 1 : yy);
        xx = xx < 0 ? 0 : (xx >= width ? width - 1 : xx);
        return (int)src[yy * width + xx];
    };

    int p = y & 1;
    int q = x & 1;
    int c = at(y, x);
    int r, g, b;

    // Verde en un sitio rojo o azul. Coeficientes / 8.
    auto green_at_rb = [&]() -> int {
        int acc = 4 * c;
        acc += 2 * (at(y - 1, x) + at(y + 1, x) + at(y, x - 1) + at(y, x + 1));
        acc -= at(y - 2, x) + at(y + 2, x) + at(y, x - 2) + at(y, x + 2);
        return acc >> 3;
    };
    // Color horizontal (rojo en fila de rojos, azul en fila de azules). / 16.
    auto horiz = [&]() -> int {
        int acc = 10 * c;
        acc += 8 * (at(y, x - 1) + at(y, x + 1));
        acc += at(y - 2, x) + at(y + 2, x);
        acc -= 2 * (at(y - 1, x - 1) + at(y - 1, x + 1) + at(y + 1, x - 1) + at(y + 1, x + 1));
        acc -= 2 * (at(y, x - 2) + at(y, x + 2));
        return acc >> 4;
    };
    // El mismo filtro, transpuesto. / 16.
    auto vert = [&]() -> int {
        int acc = 10 * c;
        acc += 8 * (at(y - 1, x) + at(y + 1, x));
        acc += at(y, x - 2) + at(y, x + 2);
        acc -= 2 * (at(y - 1, x - 1) + at(y + 1, x - 1) + at(y - 1, x + 1) + at(y + 1, x + 1));
        acc -= 2 * (at(y - 2, x) + at(y + 2, x));
        return acc >> 4;
    };
    // Rojo en azul, o azul en rojo. / 16.
    auto diag = [&]() -> int {
        int acc = 12 * c;
        acc += 4 * (at(y - 1, x - 1) + at(y - 1, x + 1) + at(y + 1, x - 1) + at(y + 1, x + 1));
        acc -= 3 * (at(y - 2, x) + at(y + 2, x) + at(y, x - 2) + at(y, x + 2));
        return acc >> 4;
    };

    if (p == 0 && q == 0) {
        r = c; g = green_at_rb(); b = diag();
    } else if (p == 0 && q == 1) {
        r = horiz(); g = c; b = vert();
    } else if (p == 1 && q == 0) {
        r = vert(); g = c; b = horiz();
    } else {
        r = diag(); g = green_at_rb(); b = c;
    }

    auto sat = [](int v) -> unsigned short {
        if (v < 0) return 0;
        if (v > 4095) return 4095;
        return (unsigned short)v;
    };
    int i = (y * width + x) * 3;
    dst[i] = sat(r);
    dst[i + 1] = sat(g);
    dst[i + 2] = sat(b);
}
"""

_malvar = None
_gpu_ok = None


def gpu_available() -> bool:
    global _gpu_ok
    if _gpu_ok is not None:
        return _gpu_ok
    try:
        import cupy as cp
        _gpu_ok = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _gpu_ok = False
    return _gpu_ok


def _kernel():
    global _malvar
    if _malvar is None:
        import cupy as cp
        _malvar = cp.RawKernel(_KERNEL, "malvar_rggb")
    return _malvar


def debayer_malvar_u16(bayer_u16: np.ndarray) -> np.ndarray:
    """Mosaico uint16 0–4095 (H, W) → RGB uint16 (H, W, 3), Malvar en GPU."""
    import cupy as cp

    src = np.ascontiguousarray(bayer_u16, dtype=np.uint16)
    height, width = src.shape
    g_src = cp.asarray(src)
    g_dst = cp.empty((height, width, 3), dtype=cp.uint16)
    threads = (16, 16)
    blocks = ((width + 15) // 16, (height + 15) // 16)
    _kernel()((blocks[0], blocks[1]), threads, (g_src, g_dst, np.int32(width), np.int32(height)))
    return cp.asnumpy(g_dst)
