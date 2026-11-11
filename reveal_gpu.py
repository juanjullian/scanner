"""Revelado en la GPU para ProRes 4444 y HEVC lossless.

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
# NPP exige 0. Es bilineal con correlación de verde: promedia vecinos,
# la misma familia que la interpolación direccional del SDK.
_NPPI_INTER_UNDEFINED = 0
_CUPY_SITE = Path(
    r"C:\Users\juan\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python313\site-packages"
)

_YUV = r"""
extern "C" __global__
void rgb01_to_yuv10(const float* __restrict__ rgb, unsigned short* __restrict__ y,
                    unsigned short* __restrict__ u, unsigned short* __restrict__ v,
                    int width, int height, int out_w, int out_h, int shift,
                    int crop_x, int crop_y, int crop_w, int crop_h, int flip) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || yy >= out_h) return;
    int local = x;
    if (flip) {
        local = x < crop_w ? crop_w - 1 - x : 0;
    } else if (x >= crop_w) {
        local = crop_w - 1;
    }
    int sx = crop_x + local;
    int sy = crop_y + yy;
    if (yy >= crop_h) sy = crop_y + crop_h - 1;
    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    if (sx >= width) sx = width - 1;
    if (sy >= height) sy = height - 1;
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

_YUV422 = r"""
extern "C" __global__
void yuv422_to_rgb(const unsigned short* __restrict__ y, const unsigned short* __restrict__ u,
                   const unsigned short* __restrict__ v, float* __restrict__ rgb, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || yy >= h) return;
    const float kr = 0.2126f;
    const float kb = 0.0722f;
    const float kg = 1.f - kr - kb;
    float Y = ((float)y[yy * w + x] - 64.f) / 879.f;
    int cx = x >> 1;
    int cw = w >> 1;
    float Cb = ((float)u[yy * cw + cx] - 512.f) / 899.5f;
    float Cr = ((float)v[yy * cw + cx] - 512.f) / 899.5f;
    float R = Y + Cr * (2.f * (1.f - kr));
    float B = Y + Cb * (2.f * (1.f - kb));
    float G = (Y - kr * R - kb * B) / kg;
    int o = (yy * w + x) * 3;
    rgb[o] = R < 0.f ? 0.f : (R > 1.f ? 1.f : R);
    rgb[o + 1] = G < 0.f ? 0.f : (G > 1.f ? 1.f : G);
    rgb[o + 2] = B < 0.f ? 0.f : (B > 1.f ? 1.f : B);
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

_GAMMA = r"""
extern "C" __global__
void apply_gamma(float* img, int n, int scale,
                 float rmin, float rmax, float gmin, float gmax, float bmin, float bmax,
                 float smin, float smax, float gr, float gg, float gb,
                 float strength, float omin, float omax, float dark, float high) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = i * 3;
    float los[3] = {rmin, gmin, bmin};
    float his[3] = {rmax, gmax, bmax};
    float gs[3] = {gr, gg, gb};
    const float feather = 16.f / 255.f;
    for (int c = 0; c < 3; ++c) {
        float value = img[b + c];
        float sample = value;
        if (scale == 1) {
            float span = smax - smin;
            if (span < 1e-6f) span = 1e-6f;
            sample = (value - smin) / span;
        } else if (scale == 2) {
            float span = his[c] - los[c];
            if (span < 1e-6f) span = 1e-6f;
            sample = (value - los[c]) / span;
        }
        if (sample < 0.f) sample = 0.f;
        else if (sample > 1.f) sample = 1.f;
        float graded = omin + powf(sample, gs[c]) * (omax - omin);
        float w = strength;
        if (dark >= 0.f) {
            float t = (value - dark) / feather;
            if (t < 0.f) t = 0.f;
            else if (t > 1.f) t = 1.f;
            w *= t;
        }
        if (high >= 0.f) {
            float t = (high - value) / feather;
            if (t < 0.f) t = 0.f;
            else if (t > 1.f) t = 1.f;
            w *= t;
        }
        img[b + c] = value * (1.f - w) + graded * w;
    }
}
"""

_SAT = r"""
extern "C" __global__
void apply_sat(float* img, int n, float sat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = i * 3;
    float r = img[b];
    float g = img[b + 1];
    float bl = img[b + 2];
    float y = 0.2126f * r + 0.7152f * g + 0.0722f * bl;
    img[b] = y + (r - y) * sat;
    img[b + 1] = y + (g - y) * sat;
    img[b + 2] = y + (bl - y) * sat;
}
"""

_WARP = r"""
extern "C" __global__
void sample_rgb(const float* __restrict__ src, float* __restrict__ dst,
                 int w, int h, float a, float b, float c, float d, float e, float f) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float fx = a * x + b * y + c;
    float fy = d * x + e * y + f;
    int o = (y * w + x) * 3;
    if (fx < 0.f || fy < 0.f || fx > w - 1.0001f || fy > h - 1.0001f) {
        dst[o] = 0.f; dst[o + 1] = 0.f; dst[o + 2] = 0.f;
        return;
    }
    int x0 = (int)fx;
    int y0 = (int)fy;
    float ax = fx - x0;
    float ay = fy - y0;
    int x1 = x0 + 1 < w ? x0 + 1 : x0;
    int y1 = y0 + 1 < h ? y0 + 1 : y0;
    int i00 = (y0 * w + x0) * 3;
    int i10 = (y0 * w + x1) * 3;
    int i01 = (y1 * w + x0) * 3;
    int i11 = (y1 * w + x1) * 3;
    float w00 = (1.f - ax) * (1.f - ay);
    float w10 = ax * (1.f - ay);
    float w01 = (1.f - ax) * ay;
    float w11 = ax * ay;
    for (int c = 0; c < 3; ++c) {
        dst[o + c] = src[i00 + c] * w00 + src[i10 + c] * w10
                   + src[i01 + c] * w01 + src[i11 + c] * w11;
    }
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


class _NppDemosaic:
    """El mismo desempaque y el mismo NPP de la exportación, sin curva ni YUV."""

    def __init__(self, width: int, height: int):
        if width % 2 or height % 2:
            raise RuntimeError(f"NPP pide ancho y alto pares. Este cuadro es {width}x{height}.")
        self.cp = prepare()
        self.width = int(width)
        self.height = int(height)
        self.npp = _load_npp()
        self.npp_ctx = _npp_context(self.cp)
        self.npp_size = _NppiSize(self.width, self.height)
        self.npp_roi = _NppiRect(0, 0, self.width, self.height)
        self.unpack = self.cp.RawKernel(_UNPACK, "unpack12")
        self.pairs = self.width * self.height // 2
        self.packed = self.cp.empty(self.pairs * 3, dtype=self.cp.uint8)
        self.src = self.cp.empty((self.height, self.width), dtype=self.cp.uint16)
        self.codes = self.cp.empty((self.height, self.width, 3), dtype=self.cp.uint16)

    def __call__(self, packed: bytes) -> np.ndarray:
        expected = self.pairs * 3
        raw = np.frombuffer(packed, dtype=np.uint8, count=expected)
        if raw.size < expected:
            raise ValueError("El cuadro Bayer quedó corto.")
        self.packed.set(raw)
        self.unpack(
            ((self.pairs + 255) // 256,), (256,),
            (self.packed, self.src, np.int32(self.pairs)),
        )
        status = self.npp(
            self.src.data.ptr, self.width * 2, self.npp_size, self.npp_roi,
            self.codes.data.ptr, self.width * 6,
            _NPPI_BAYER_RGGB, _NPPI_INTER_UNDEFINED, self.npp_ctx,
        )
        if status != 0:
            raise RuntimeError(f"NPP devolvió {status}")
        self.cp.cuda.Stream.null.synchronize()
        return self.cp.asnumpy(self.codes).copy()


_npp_demosaic: dict[tuple[int, int], _NppDemosaic] = {}


def npp_demosaic_codes(packed: bytes, width: int, height: int) -> np.ndarray:
    """RGB uint16 0–4095. Es el demosaico de la exportación en GPU."""
    key = (int(width), int(height))
    worker = _npp_demosaic.get(key)
    if worker is None:
        worker = _NppDemosaic(*key)
        _npp_demosaic[key] = worker
    return worker(packed)


def open_prores_nv(dst, out_w: int, out_h: int, fps: int, *, stderr=None, rate=None, creationflags=0) -> subprocess.Popen:
    """FFmpeg aparte, con prores_nv. No reemplaza al ffmpeg del sistema."""
    if not FFMPEG_NV.is_file():
        raise RuntimeError(f"No está el FFmpeg con prores_nv: {FFMPEG_NV}")
    cmd = [
        str(FFMPEG_NV), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "yuv444p10le",
        "-r", str(int(fps) if rate is None else rate), "-i", "pipe:0",
        "-c:v", "prores_nv", "-profile:v", "4444", "-q:v", "5",
        "-pix_fmt", "yuv444p10le",
        str(dst),
    ]
    env = os.environ.copy()
    env["CUDA_PATH"] = str(CUDA_ROOT)
    env["PATH"] = str(CUDA_BIN) + os.pathsep + str(FFMPEG_NV.parent) + os.pathsep + env.get("PATH", "")
    print(f"INFO|FFmpeg cmd (stdin): {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=sys.stderr if stderr is None else stderr,
        bufsize=4 << 20, env=env, creationflags=creationflags,
    )


class GpuReveal:
    """Cuadro Bayer empaquetado o mosaico 0–4095 → YUV de 10 bits. El RGB no baja."""

    def __init__(self, settings: dict, width: int, height: int, mode: str = "COLOR",
                 shift: int = 0, out_w: int | None = None, out_h: int | None = None,
                 finish=None):
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
        self.finish = finish
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
        self.gamma_k = cp.RawKernel(_GAMMA, "apply_gamma")
        self.sat_k = cp.RawKernel(_SAT, "apply_sat")
        self.warp_k = cp.RawKernel(_WARP, "sample_rgb")
        self._from422 = None
        self._y16 = None
        self.warped = None
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

    def frame_raw(self, packed: bytes, index: int = 0) -> bytes:
        """BayerRG12p empaquetado. El desempaque ocurre en la tarjeta."""
        self.packed.set(np.frombuffer(packed, dtype=np.uint8))
        self.unpack(
            ((self.pairs + 255) // 256,), (256,),
            (self.packed, self.src, np.int32(self.pairs)),
        )
        return self._after_mosaic(index)

    def frame(self, mosaic: np.ndarray, index: int = 0) -> bytes:
        """Mosaico uint16 ya desempaquetado, códigos 0–4095."""
        self.src.set(np.ascontiguousarray(mosaic, dtype=np.uint16))
        return self._after_mosaic(index)

    def frame_codes(self, codes: np.ndarray, index: int = 0) -> bytes:
        """RGB uint16 ya demosaiado, códigos 0–4095. La curva sigue en la tarjeta."""
        self.codes.set(np.ascontiguousarray(codes, dtype=np.uint16))
        return self._after_codes(index)

    def _after_mosaic(self, index: int = 0) -> bytes:
        cp = self.cp
        status = self.npp(
            self.src.data.ptr, self.width * 2, self.npp_size, self.npp_roi,
            self.codes.data.ptr, self.width * 6,
            _NPPI_BAYER_RGGB, _NPPI_INTER_UNDEFINED, self.npp_ctx,
        )
        if status != 0:
            raise RuntimeError(f"NPP devolvió {status}")
        cp.cuda.Stream.null.synchronize()
        return self._after_codes(index)

    def _after_codes(self, index: int = 0) -> bytes:
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
        return self._encode_yuv(index)

    def frame_rgb(self, image: np.ndarray, index: int = 0) -> bytes:
        """RGB ya revelado, uint16 o float 0–1. Color, warp, recorte y YUV quedan en la tarjeta."""
        arr = np.ascontiguousarray(image)
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) * np.float32(1.0 / 65535.0)
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.shape != (self.height, self.width, 3):
            raise RuntimeError(
                f"El cuadro es {arr.shape[1]}×{arr.shape[0]} y este video es {self.width}×{self.height}."
            )
        self.img.set(arr)
        return self._encode_yuv(index)

    def frame_yuv422(self, raw: bytes, index: int = 0) -> bytes:
        """YUV 4:2:2 de 10 bits, el que entrega el ProRes. El paso a RGB ocurre en la tarjeta."""
        cp = self.cp
        if self._from422 is None:
            self._from422 = cp.RawKernel(_YUV422, "yuv422_to_rgb")
            half = self.width // 2
            self._y16 = cp.empty((self.height, self.width), dtype=cp.uint16)
            self._u16 = cp.empty((self.height, half), dtype=cp.uint16)
            self._v16 = cp.empty((self.height, half), dtype=cp.uint16)
        y_n = self.width * self.height
        c_n = (self.width // 2) * self.height
        plane = np.frombuffer(raw, dtype=np.uint16)
        self._y16.set(plane[:y_n].reshape(self.height, self.width))
        self._u16.set(plane[y_n:y_n + c_n].reshape(self.height, self.width // 2))
        self._v16.set(plane[y_n + c_n:].reshape(self.height, self.width // 2))
        self._from422(
            ((self.width + 15) // 16, (self.height + 15) // 16), (16, 16),
            (self._y16, self._u16, self._v16, self.img, np.int32(self.width), np.int32(self.height)),
        )
        return self._encode_yuv(index)

    def _encode_yuv(self, index: int) -> bytes:
        cp = self.cp
        src = self._finish_on_device(index)
        self._view = src
        crop_x, crop_y, crop_w, crop_h = self._crop_args()
        self.yuv(
            ((self.out_w + 15) // 16, (self.out_h + 15) // 16), (16, 16),
            (
                src, self.yp, self.up, self.vp,
                np.int32(self.width), np.int32(self.height),
                np.int32(self.out_w), np.int32(self.out_h), np.int32(self.shift),
                np.int32(crop_x), np.int32(crop_y), np.int32(crop_w), np.int32(crop_h),
                np.int32(1 if self.finish is not None and self.finish.flip_h else 0),
            ),
        )
        n = self.out_w * self.out_h
        self.host[:n] = cp.asnumpy(self.yp)
        self.host[n:n * 2] = cp.asnumpy(self.up)
        self.host[n * 2:] = cp.asnumpy(self.vp)
        return self.host.tobytes()

    def preview_rgb(self) -> np.ndarray | None:
        """El cuadro ya revelado, chico, para la miniatura de la exportación."""
        view = getattr(self, "_view", None)
        if view is None:
            return None
        cp = self.cp
        x, y, width, height = self._crop_args()
        limit_h, limit_w = int(view.shape[0]), int(view.shape[1])
        x = max(0, min(int(x), limit_w - 1))
        y = max(0, min(int(y), limit_h - 1))
        width = max(1, min(int(width), limit_w - x))
        height = max(1, min(int(height), limit_h - y))
        step = max(1, int(max(width, height) / 160))
        small = view[y:y + height:step, x:x + width:step]
        rgb = cp.asnumpy(cp.clip(small * cp.float32(255.0) + cp.float32(0.5), 0, 255).astype(cp.uint8))
        if self.finish is not None and self.finish.flip_h:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        return rgb

    def _crop_args(self) -> tuple[int, int, int, int]:
        if self.finish is not None:
            box = self.finish.crop_box(self.width, self.height)
            if box is not None:
                return box
        return 0, 0, self.width, self.height

    def _finish_on_device(self, index: int):
        src = self.img
        if self.finish is None:
            return src
        recipe = None if self.finish.grade_mode == "bw" else self.finish.recipe_at(index)
        if recipe:
            mins = recipe["mins"]
            maxs = recipe["maxs"]
            gammas = recipe["gammas"]
            self.gamma_k(
                ((self.n + 255) // 256,), (256,),
                (
                    self.img, np.int32(self.n), np.int32(int(recipe.get("scale", 2))),
                    np.float32(mins[0]), np.float32(maxs[0]),
                    np.float32(mins[1]), np.float32(maxs[1]),
                    np.float32(mins[2]), np.float32(maxs[2]),
                    np.float32(recipe.get("shared_min", 0.0)),
                    np.float32(recipe.get("shared_max", 1.0)),
                    np.float32(gammas[0]), np.float32(gammas[1]), np.float32(gammas[2]),
                    np.float32(recipe.get("strength", 1.0)),
                    np.float32(recipe.get("omin", 0.0)),
                    np.float32(recipe.get("omax", 1.0)),
                    np.float32(-1.0 if recipe.get("dark") is None else recipe["dark"]),
                    np.float32(-1.0 if recipe.get("high") is None else recipe["high"]),
                ),
            )
        sat = float(getattr(self.finish, "grade_sat", 1.0))
        if sat < 0.999:
            self.sat_k(
                ((self.n + 255) // 256,), (256,),
                (self.img, np.int32(self.n), np.float32(sat)),
            )
        matrix = None if self.finish is None else self.finish.sampling(index, self.width, self.height)
        if matrix is not None:
            if self.warped is None:
                self.warped = self.cp.empty_like(self.img)
            self.warp_k(
                ((self.width + 15) // 16, (self.height + 15) // 16), (16, 16),
                (
                    self.img, self.warped, np.int32(self.width), np.int32(self.height),
                    np.float32(matrix[0, 0]), np.float32(matrix[0, 1]), np.float32(matrix[0, 2]),
                    np.float32(matrix[1, 0]), np.float32(matrix[1, 1]), np.float32(matrix[1, 2]),
                ),
            )
            src = self.warped
        return src
