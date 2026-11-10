"""Revelado en la GPU para ProRes 4444 XQ y HEVC lossless.

El cuadro empaquetado sube tal cual. En la tarjeta se desempaqueta, corre
el demosaico NPP, la curva, el detalle y el YUV. El ProRes lo escribe
prores_nv. El HEVC lossless sigue en NVENC, con el YUV ya armado.
Un solo proceso: varios hilos de Python sobre la misma tarjeta se pisan.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from reveal_pipeline import _SIGMA_COARSE, _SIGMA_FINE, reveal_luts

CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5")
CUDA_BIN = CUDA_ROOT / "bin"
CUDA_BIN_X64 = CUDA_BIN / "x64"
NPP_BIN = Path(
    r"C:\Users\juan\src\proresnv-test\npp\libnpp-windows-x86_64-13.2.0.93-archive\bin\x64"
)
FFMPEG_NV = Path(r"C:\Users\juan\src\proresnv-test\dist\ffmpeg.exe")
_NPPI_BAYER_RGGB = 1
_NPPI_INTER_UNDEFINED = 0
_CUPY_SITE = Path(
    r"C:\Users\juan\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python313\site-packages"
)

_YUV = r"""
extern "C" __global__
void rgb01_to_yuv10(const float* __restrict__ rgb, unsigned short* __restrict__ y,
                    unsigned short* __restrict__ u, unsigned short* __restrict__ v,
                    int width, int height, int out_w, int out_h, int shift) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || yy >= out_h) return;
    int sx = x < width ? x : width - 1;
    int sy = yy < height ? yy : height - 1;
    int i = (sy * width + sx) * 3;
    float R = rgb[i];
    float G = rgb[i + 1];
    float B = rgb[i + 2];
    float Y01 = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    float Cb = (B - Y01) / (2.f * (1.f - 0.0722f));
    float Cr = (R - Y01) / (2.f * (1.f - 0.2126f));
    int Y10 = (int)rintf(Y01 * 879.f + 64.f);
    int U10 = (int)rintf(Cb * 899.5f + 512.f);
    int V10 = (int)rintf(Cr * 899.5f + 512.f);
    if (Y10 < 0) Y10 = 0; else if (Y10 > 1023) Y10 = 1023;
    if (U10 < 0) U10 = 0; else if (U10 > 1023) U10 = 1023;
    if (V10 < 0) V10 = 0; else if (V10 > 1023) V10 = 1023;
    int o = yy * out_w + x;
    y[o] = (unsigned short)(Y10 << shift);
    u[o] = (unsigned short)(U10 << shift);
    v[o] = (unsigned short)(V10 << shift);
}
"""

_UNPACK = r"""
extern "C" __global__
void unpack12(const unsigned char* __restrict__ packed, unsigned short* __restrict__ dst, int pairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    int b = i * 3;
    unsigned int b0 = packed[b];
    unsigned int b1 = packed[b + 1];
    unsigned int b2 = packed[b + 2];
    dst[i * 2] = (unsigned short)(b0 | ((b1 & 0x0Fu) << 8));
    dst[i * 2 + 1] = (unsigned short)((b1 >> 4) | (b2 << 4));
}
"""

_LUT = r"""
extern "C" __global__
void apply_lut(const unsigned short* __restrict__ codes, float* __restrict__ dst,
               const float* __restrict__ lr, const float* __restrict__ lg,
               const float* __restrict__ lb, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = i * 3;
    dst[b] = lr[codes[b]];
    dst[b + 1] = lg[codes[b + 1]];
    dst[b + 2] = lb[codes[b + 2]];
}
"""

_MID = r"""
extern "C" __global__
void apply_ratio(float* __restrict__ img, const float* __restrict__ y,
                 const float* __restrict__ fine, const float* __restrict__ coarse,
                 float amount, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float yy = y[i];
    float yn = yy + amount * (fine[i] - coarse[i]);
    float ratio = 1.f;
    if (yy >= 1e-4f) {
        ratio = yn / yy;
        if (ratio < 0.f) ratio = 0.f;
        else if (ratio > 3.f) ratio = 3.f;
    }
    int b = i * 3;
    float r = img[b] * ratio;
    float g = img[b + 1] * ratio;
    float bl = img[b + 2] * ratio;
    img[b] = r < 0.f ? 0.f : (r > 1.f ? 1.f : r);
    img[b + 1] = g < 0.f ? 0.f : (g > 1.f ? 1.f : g);
    img[b + 2] = bl < 0.f ? 0.f : (bl > 1.f ? 1.f : bl);
}
"""


def prores_nv_size(width: int, height: int) -> tuple[int, int]:
    """prores_nv pide ancho múltiplo de 16 y alto múltiplo de 8."""
    out_w = (int(width) + 15) // 16 * 16
    out_h = (int(height) + 7) // 8 * 8
    return out_w, out_h


def prepare():
    """Deja CUDA_PATH listo y devuelve CuPy. Lo llama solo el proceso que exporta."""
    os.environ["CUDA_PATH"] = str(CUDA_ROOT)
    try:
        import cupy as cp
    except ModuleNotFoundError:
        site = str(_CUPY_SITE)
        if site not in sys.path:
            sys.path.append(site)
        try:
            import cupy as cp
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CuPy no está en este Python. El revelado en GPU no puede partir."
            ) from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("No hay una GPU NVIDIA visible para el revelado.")
    return cp


class _NppiSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int)]


class _NppiRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int), ("y", ctypes.c_int),
        ("width", ctypes.c_int), ("height", ctypes.c_int),
    ]


class _NppStreamContext(ctypes.Structure):
    _fields_ = [
        ("hStream", ctypes.c_void_p),
        ("nCudaDeviceId", ctypes.c_int),
        ("nMultiProcessorCount", ctypes.c_int),
        ("nMaxThreadsPerMultiProcessor", ctypes.c_int),
        ("nMaxThreadsPerBlock", ctypes.c_int),
        ("nSharedMemPerBlock", ctypes.c_size_t),
        ("nCudaDevAttrComputeCapabilityMajor", ctypes.c_int),
        ("nCudaDevAttrComputeCapabilityMinor", ctypes.c_int),
        ("nStreamFlags", ctypes.c_uint),
        ("nReserved0", ctypes.c_uint),
    ]


def _load_npp():
    if not (NPP_BIN / "nppicc64_13.dll").is_file():
        raise RuntimeError(f"No está la librería NPP: {NPP_BIN}")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(CUDA_BIN_X64))
        os.add_dll_directory(str(CUDA_BIN))
        os.add_dll_directory(str(NPP_BIN))
    ctypes.WinDLL(str(NPP_BIN / "nppc64_13.dll"))
    icc = ctypes.WinDLL(str(NPP_BIN / "nppicc64_13.dll"))
    fn = icc.nppiCFAToRGB_16u_C1C3R_Ctx
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_int, _NppiSize, _NppiRect,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, _NppStreamContext,
    ]
    fn.restype = ctypes.c_int
    return fn


def _npp_context(cp) -> _NppStreamContext:
    props = cp.cuda.runtime.getDeviceProperties(0)
    ctx = _NppStreamContext()
    ctx.hStream = None
    ctx.nCudaDeviceId = int(cp.cuda.Device().id)
    ctx.nMultiProcessorCount = int(props["multiProcessorCount"])
    ctx.nMaxThreadsPerMultiProcessor = int(props["maxThreadsPerMultiProcessor"])
    ctx.nMaxThreadsPerBlock = int(props["maxThreadsPerBlock"])
    ctx.nSharedMemPerBlock = int(props["sharedMemPerBlock"])
    ctx.nCudaDevAttrComputeCapabilityMajor = int(props["major"])
    ctx.nCudaDevAttrComputeCapabilityMinor = int(props["minor"])
    ctx.nStreamFlags = 0
    ctx.nReserved0 = 0
    return ctx


def open_prores_nv(dst, out_w: int, out_h: int, fps: int) -> subprocess.Popen:
    """FFmpeg aparte, con prores_nv. No reemplaza al ffmpeg del sistema."""
    if not FFMPEG_NV.is_file():
        raise RuntimeError(f"No está el FFmpeg con prores_nv: {FFMPEG_NV}")
    cmd = [
        str(FFMPEG_NV), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "yuv444p10le",
        "-r", str(int(fps)), "-i", "pipe:0",
        "-c:v", "prores_nv", "-profile:v", "4444xq", "-q:v", "4",
        "-pix_fmt", "yuv444p10le",
        str(dst),
    ]
    env = os.environ.copy()
    env["CUDA_PATH"] = str(CUDA_ROOT)
    env["PATH"] = str(CUDA_BIN) + os.pathsep + str(FFMPEG_NV.parent) + os.pathsep + env.get("PATH", "")
    print(f"INFO|FFmpeg cmd (stdin): {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=sys.stderr, bufsize=4 << 20, env=env,
    )


class GpuReveal:
    """Cuadro Bayer empaquetado o mosaico 0–4095 → YUV de 10 bits. El RGB no baja."""

    def __init__(self, settings: dict, width: int, height: int, mode: str = "COLOR",
                 shift: int = 0, out_w: int | None = None, out_h: int | None = None):
        cp = prepare()
        from cupyx.scipy.ndimage import gaussian_filter

        self.cp = cp
        self.gaussian_filter = gaussian_filter
        self.width = int(width)
        self.height = int(height)
        if self.width % 2 or self.height % 2:
            raise RuntimeError(
                f"NPP pide ancho y alto pares. Este cuadro es {self.width}x{self.height}."
            )
        self.out_w = int(out_w or self.width)
        self.out_h = int(out_h or self.height)
        self.shift = int(shift)
        self.bw = mode == "BW"
        self.amount = float(settings["midtone"])
        luts = reveal_luts(settings)
        self.lr = cp.asarray(luts[0])
        self.lg = cp.asarray(luts[1])
        self.lb = cp.asarray(luts[2])
        self.npp = _load_npp()
        self.npp_ctx = _npp_context(cp)
        self.npp_size = _NppiSize(self.width, self.height)
        self.npp_roi = _NppiRect(0, 0, self.width, self.height)
        self.unpack = cp.RawKernel(_UNPACK, "unpack12")
        self.lut = cp.RawKernel(_LUT, "apply_lut")
        self.ratio = cp.RawKernel(_MID, "apply_ratio")
        self.yuv = cp.RawKernel(_YUV, "rgb01_to_yuv10")
        self.packed = cp.empty(self.width * self.height * 3 // 2, dtype=cp.uint8)
        self.src = cp.empty((self.height, self.width), dtype=cp.uint16)
        self.codes = cp.empty((self.height, self.width, 3), dtype=cp.uint16)
        self.img = cp.empty((self.height, self.width, 3), dtype=cp.float32)
        self.y = cp.empty((self.height, self.width), dtype=cp.float32)
        self.fine = cp.empty((self.height, self.width), dtype=cp.float32)
        self.coarse = cp.empty((self.height, self.width), dtype=cp.float32)
        n = self.out_w * self.out_h
        self.yp = cp.empty(n, dtype=cp.uint16)
        self.up = cp.empty(n, dtype=cp.uint16)
        self.vp = cp.empty(n, dtype=cp.uint16)
        self.host = np.empty(n * 3, dtype=np.uint16)
        self.pairs = self.width * self.height // 2
        self.n = self.width * self.height
        self.even = True

    def frame_raw(self, packed: bytes) -> bytes:
        """BayerRG12p empaquetado. El desempaque ocurre en la tarjeta."""
        self.packed.set(np.frombuffer(packed, dtype=np.uint8))
        self.unpack(
            ((self.pairs + 255) // 256,), (256,),
            (self.packed, self.src, np.int32(self.pairs)),
        )
        return self._after_mosaic()

    def frame(self, mosaic: np.ndarray) -> bytes:
        """Mosaico uint16 ya desempaquetado, códigos 0–4095."""
        self.src.set(np.ascontiguousarray(mosaic, dtype=np.uint16))
        return self._after_mosaic()

    def _after_mosaic(self) -> bytes:
        cp = self.cp
        status = self.npp(
            self.src.data.ptr, self.width * 2, self.npp_size, self.npp_roi,
            self.codes.data.ptr, self.width * 6,
            _NPPI_BAYER_RGGB, _NPPI_INTER_UNDEFINED, self.npp_ctx,
        )
        if status != 0:
            raise RuntimeError(f"NPP devolvió {status}")
        cp.cuda.Stream.null.synchronize()
        self.lut(
            ((self.n + 255) // 256,), (256,),
            (self.codes, self.img, self.lr, self.lg, self.lb, np.int32(self.n)),
        )
        if self.amount > 1e-4:
            r = self.img[:, :, 0]
            g = self.img[:, :, 1]
            b = self.img[:, :, 2]
            self.y[...] = r * np.float32(0.2126) + g * np.float32(0.7152) + b * np.float32(0.0722)
            self.fine[...] = self.gaussian_filter(self.y, float(_SIGMA_FINE))
            if self.even:
                small = (
                    self.y[0::2, 0::2] + self.y[1::2, 0::2]
                    + self.y[0::2, 1::2] + self.y[1::2, 1::2]
                ) * np.float32(0.25)
                coarse_s = self.gaussian_filter(small, float(_SIGMA_COARSE) * 0.5)
                self.coarse[0::2, 0::2] = coarse_s
                self.coarse[0::2, 1::2] = coarse_s
                self.coarse[1::2, 0::2] = coarse_s
                self.coarse[1::2, 1::2] = coarse_s
            else:
                self.coarse[...] = self.gaussian_filter(self.y, float(_SIGMA_COARSE))
            self.ratio(
                ((self.n + 255) // 256,), (256,),
                (self.img, self.y, self.fine, self.coarse, np.float32(self.amount), np.int32(self.n)),
            )
        if self.bw:
            yb = (
                self.img[:, :, 0] * np.float32(0.2126)
                + self.img[:, :, 1] * np.float32(0.7152)
                + self.img[:, :, 2] * np.float32(0.0722)
            )
            self.img[:, :, 0] = yb
            self.img[:, :, 1] = yb
            self.img[:, :, 2] = yb
        self.yuv(
            ((self.out_w + 15) // 16, (self.out_h + 15) // 16), (16, 16),
            (
                self.img, self.yp, self.up, self.vp,
                np.int32(self.width), np.int32(self.height),
                np.int32(self.out_w), np.int32(self.out_h), np.int32(self.shift),
            ),
        )
        n = self.out_w * self.out_h
        self.host[:n] = cp.asnumpy(self.yp)
        self.host[n:n * 2] = cp.asnumpy(self.up)
        self.host[n * 2:] = cp.asnumpy(self.vp)
        return self.host.tobytes()
