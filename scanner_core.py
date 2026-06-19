import os
import re
import time
import queue
import struct
import threading
import numpy as np
import json
import shutil
import qoi_utils
import bayer_render
from PyQt6.QtCore import pyqtSignal, QThread
try:
    from arena_api.system import system
    from arena_api.buffer import BufferFactory
    from arena_api.enums import PixelFormat, IncMode, BayerAlgorithm
    ARENA_AVAILABLE = True
except Exception as e:
    # Fallback a Aravis (macOS / Open Source)
    print(f"Arena SDK no encontrado ({e}). Intentando Aravis...")
    try:
        import aravis_adapter
        if aravis_adapter.ARAVIS_AVAILABLE:
            system = aravis_adapter.system
            BufferFactory = aravis_adapter.BufferFactory
            PixelFormat = aravis_adapter.PixelFormat
            ARENA_AVAILABLE = True # Engañamos al sistema diciendo que "Arena" (el wrapper) está listo
            print("ÉXITO: Backend Aravis cargado correctamente.")
            try:
                from arena_api.enums import IncMode, BayerAlgorithm
            except Exception:
                IncMode = None  # type: ignore
                BayerAlgorithm = None  # type: ignore
        else:
            raise ImportError("Librería Aravis no detectada en el sistema.")
    except Exception as e2:
        print(f"WARN: Backend Aravis no disponible: {e2}")
        ARENA_AVAILABLE = False
        IncMode = None  # type: ignore
        BayerAlgorithm = None  # type: ignore

        # Mock classes para permitir carga de la App en modo "Solo Visor"
        class MockSystem:
            def create_device(self): return []
            def destroy_device(self, d): pass
        class MockFactory:
            def convert(self, b, f): raise Exception("No Arena")
            def destroy(self, b): pass
            def copy(self, b): return b
        class MockEnums:
            BGR8 = 1; Mono8 = 2; RGB8 = 3; QOI_RGB8 = 4
            
        system = MockSystem()
        BufferFactory = MockFactory()
        PixelFormat = MockEnums()
from pathlib import Path

# Intentamos importar psutil para medir ancho de banda real del SO
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARN: 'psutil' no instalado. La medición de ancho de banda será estimada.")

# Se ha unificado a una única resolución (ROI centrado en sensor 2840x2840).
FIXED_WIDTH = 2840
FIXED_HEIGHT = 2200
FIXED_OFFSET_X = 0
FIXED_OFFSET_Y = 320  # (2840 - 2200) / 2

ISP_CT_GAIN_KEYS = tuple(f"gain{i}{j}" for i in range(3) for j in range(3))
ISP_CT_GAIN_SELECTORS = tuple(f"Gain{i}{j}" for i in range(3) for j in range(3))
ISP_CT_GAIN_DEFAULTS = {
    "gain00": 1.6729, "gain01": -0.49341, "gain02": -0.17944,
    "gain10": -0.46411, "gain11": 1.9543, "gain12": -0.49023,
    "gain20": 0.125, "gain21": -0.65698, "gain22": 1.532,
}

CAMERA_RECONNECT_SEC = 5.0


def probe_camera_available(max_wait_sec=0):
    """Detecta cámara sin aplicar configuración (solo para splash / sondeo)."""
    if not ARENA_AVAILABLE:
        return False
    attempts = max(1, int(max_wait_sec) + 1) if max_wait_sec else 1
    for i in range(attempts):
        devices = None
        try:
            devices = system.create_device()
            if devices:
                try:
                    system.destroy_device(devices[0])
                except Exception:
                    pass
                return True
        except Exception:
            pass
        if i < attempts - 1:
            time.sleep(1.0)
    return False

RAW12_MAX = 4095


def compute_raw12_exposure_stats(packed_bytes, width, height, sample_step=8):
    """
    Porcentajes respecto al rango 12-bit (100% = pico del sensor).
    Hi: máximo del frame.
    Lo: p10 de píxeles por encima del piso digital (ignora 0 absolutos del sensor).
    """
    view = _packed_bayer_view(packed_bytes, width, height)
    chunk = view[::sample_step, ::sample_step, :]
    p0, p1 = _unpack12_pair(chunk[:, :, 0], chunk[:, :, 1], chunk[:, :, 2])
    pixels = np.concatenate([p0.ravel(), p1.ravel()])
    if pixels.size == 0:
        return 0.0, 0.0
    hi_pct = 100.0 * float(pixels.max()) / RAW12_MAX
    # p1 suele ser 0 en Bayer (píxeles muertos / piso ADC); medimos sombras reales.
    floor_dn = 32
    lifted = pixels[pixels >= floor_dn]
    pool = lifted if lifted.size >= 64 else pixels
    lo_pct = 100.0 * float(np.percentile(pool, 10)) / RAW12_MAX
    return hi_pct, lo_pct


def _packed_bayer_view(packed_bytes, width, height):
    return np.frombuffer(packed_bytes, dtype=np.uint8).reshape(height, width // 2, 3)


def _unpack12_pair(b0, b1, b2):
    b0 = b0.astype(np.uint16)
    b1 = b1.astype(np.uint16)
    b2 = b2.astype(np.uint16)
    p0 = b0 | ((b1 & 0x0F) << 8)
    p1 = (b1 >> 4) | (b2 << 4)
    return p0, p1


def debayer_raw12_preview(packed_bytes, width, height, downscale=4, to_gray=False):
    """Debayer rápido sobre raw 12-bit (sin ISP de Arena)."""
    import cv2 as _cv2
    view = _packed_bayer_view(packed_bytes, width, height)
    chunk = view[::downscale, ::downscale, :]
    p0, p1 = _unpack12_pair(chunk[:, :, 0], chunk[:, :, 1], chunk[:, :, 2])
    h_out, w_half = chunk.shape[0], chunk.shape[1]
    w_out = w_half * 2
    img_flat = np.empty(h_out * w_out, dtype=np.uint16)
    img_flat[0::2] = p0.ravel()
    img_flat[1::2] = p1.ravel()
    bayer = img_flat.reshape(h_out, w_out)
    rgb16 = _cv2.cvtColor(bayer, _cv2.COLOR_BayerRG2RGB)
    rgb8 = np.clip(rgb16.astype(np.float32) * (255.0 / RAW12_MAX), 0, 255).astype(np.uint8)
    if to_gray:
        return _cv2.cvtColor(rgb8, _cv2.COLOR_RGB2GRAY)
    return _cv2.cvtColor(rgb8, _cv2.COLOR_RGB2BGR)


def apply_isp_preview_tone(bgr, gamma=1.0, shadow_lift=0.0, contrast=1.0):
    """Post-proceso BGR8 solo para preview ISP. No modifica captura Bayer."""
    if gamma == 1.0 and shadow_lift == 0.0 and contrast == 1.0:
        return bgr
    x = bgr.astype(np.float32) / 255.0
    if shadow_lift > 0.0:
        x = x + shadow_lift * (1.0 - x)
    if contrast != 1.0:
        x = (x - 0.5) * contrast + 0.5
    if gamma != 1.0:
        x = np.clip(x, 0.0, 1.0)
        x = np.power(x, 1.0 / max(gamma, 0.01))
    out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return out if out.flags["C_CONTIGUOUS"] else np.ascontiguousarray(out)


class PreviewWorker(QThread):
    """
    Convierte buffers Bayer crudos a imagen visualizable.
    Modos: isp, hq, hq_half, raw, bw.
    Las estadísticas de exposición se calculan siempre sobre el raw 12-bit.
    """
    image_ready = pyqtSignal(np.ndarray)
    exposure_stats_updated = pyqtSignal(float, float)

    def __init__(self, in_queue):
        super().__init__()
        self.in_queue = in_queue
        self.running = True
        self.preview_mode = "isp"
        self.skip_frames = True
        self.preview_skip_counter = 0
        self._tone_lock = threading.Lock()
        self._isp_gamma = 2.0
        self._isp_lift = 0.0
        self._isp_contrast = 1.0
        self._last_isp_base = None
        self._isp_reconvert_buffer = None
        self._bayer_algorithm = 0
        self._isp_buffer_lock = threading.Lock()
        self._cmd_queue = queue.Queue()

    def request_refresh_isp(self):
        """Encola re-debayer SDK en el hilo de preview (Arena no es thread-safe)."""
        try:
            self._cmd_queue.put_nowait("refresh_isp")
        except queue.Full:
            pass

    def refresh_isp_preview(self):
        self.request_refresh_isp()

    def _drain_commands(self):
        refresh = False
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            if cmd == "refresh_isp":
                refresh = True
        if refresh:
            self._refresh_isp_preview_inner()

    def _refresh_isp_preview_inner(self):
        with self._isp_buffer_lock:
            buf = self._isp_reconvert_buffer
            if buf is None or self.preview_mode not in ("isp", "isp_full"):
                return
            try:
                image_final = self._render_isp_from_buffer(buf)
            except Exception as e:
                print(f"WARN refresh_isp_preview: {e}")
                return
        self.image_ready.emit(image_final)

    def set_bayer_algorithm(self, algo):
        self._bayer_algorithm = int(algo)

    def get_bayer_algorithm(self):
        return self._bayer_algorithm

    def _convert_to_bgr8(self, arena_buffer):
        try:
            if BayerAlgorithm is not None:
                return BufferFactory.convert(
                    arena_buffer, PixelFormat.BGR8, BayerAlgorithm(int(self._bayer_algorithm))
                )
        except Exception:
            pass
        return BufferFactory.convert(arena_buffer, PixelFormat.BGR8)

    def set_preview_mode(self, mode):
        if mode in ("isp", "isp_full", "raw", "bw", "hq", "hq_half"):
            self.preview_mode = mode

    def set_skip_frames(self, enabled):
        self.skip_frames = enabled

    def set_isp_tone(self, gamma, shadow_lift, contrast):
        with self._tone_lock:
            self._isp_gamma = float(gamma)
            self._isp_lift = float(shadow_lift)
            self._isp_contrast = float(contrast)
            base = self._last_isp_base.copy() if self._last_isp_base is not None else None
        if base is not None:
            toned = apply_isp_preview_tone(base, gamma, shadow_lift, contrast)
            self.image_ready.emit(toned)

    def get_isp_tone(self):
        with self._tone_lock:
            return self._isp_gamma, self._isp_lift, self._isp_contrast

    def _apply_isp_tone(self, base_bgr):
        with self._tone_lock:
            self._last_isp_base = base_bgr.copy()
            g, lift, c = self._isp_gamma, self._isp_lift, self._isp_contrast
        return apply_isp_preview_tone(base_bgr, g, lift, c)

    def _isp_downscale(self):
        return 2 if self.preview_mode == "isp" else 1

    def _render_isp_from_buffer(self, arena_buffer):
        downscale = self._isp_downscale()
        image_converted = self._convert_to_bgr8(arena_buffer)
        ch, cw = image_converted.height, image_converted.width
        full_arr = np.ctypeslib.as_array(image_converted.pdata, shape=(ch, cw, 3))
        step = max(1, downscale)
        base = full_arr[::step, ::step, :].copy()
        BufferFactory.destroy(image_converted)
        return self._apply_isp_tone(base)

    def _cache_isp_buffer(self, arena_buffer):
        with self._isp_buffer_lock:
            if self._isp_reconvert_buffer:
                try:
                    BufferFactory.destroy(self._isp_reconvert_buffer)
                except Exception:
                    pass
                self._isp_reconvert_buffer = None
            try:
                self._isp_reconvert_buffer = BufferFactory.copy(arena_buffer)
            except Exception:
                self._isp_reconvert_buffer = None

    def _extract_packed_bayer(self, arena_buffer):
        w, h = arena_buffer.width, arena_buffer.height
        expected_size = int(w * h * 1.5)
        data_full = np.ctypeslib.as_array(arena_buffer.pdata, shape=(arena_buffer.size_filled,))
        if data_full.size >= expected_size:
            return data_full[:expected_size].tobytes(), w, h
        return data_full.copy().tobytes(), w, h

    def run(self):
        while self.running:
            self._drain_commands()
            try:
                item = self.in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            arena_buffer = item
            try:
                packed, w, h = self._extract_packed_bayer(arena_buffer)
                hi_pct, lo_pct = compute_raw12_exposure_stats(packed, w, h)
                self.exposure_stats_updated.emit(hi_pct, lo_pct)

                skip_display = False
                if self.skip_frames:
                    self.preview_skip_counter += 1
                    if self.preview_skip_counter % 2 != 0:
                        skip_display = True

                if skip_display:
                    continue

                mode = self.preview_mode
                if mode in ("isp", "isp_full"):
                    self._cache_isp_buffer(arena_buffer)
                    image_final = self._render_isp_from_buffer(arena_buffer)
                elif mode == "hq":
                    image_final = bayer_render.render_capture_view(packed, w, h, downscale=1)
                elif mode == "hq_half":
                    image_final = bayer_render.render_capture_view(packed, w, h, downscale=2)
                else:
                    image_final = debayer_raw12_preview(
                        packed, w, h, downscale=4, to_gray=(mode == "bw")
                    )
                self.image_ready.emit(image_final)
                self._drain_commands()

            except Exception as e:
                print(f"Error PreviewWorker: {e}")
            finally:
                if arena_buffer:
                    BufferFactory.destroy(arena_buffer)
                self.in_queue.task_done()

    def stop(self):
        self.running = False
        self.wait()
        with self._isp_buffer_lock:
            if self._isp_reconvert_buffer:
                try:
                    BufferFactory.destroy(self._isp_reconvert_buffer)
                except Exception:
                    pass
                self._isp_reconvert_buffer = None


class CameraWorker(QThread):
    image_received = pyqtSignal(np.ndarray)
    exposure_stats_updated = pyqtSignal(float, float)
    config_applied = pyqtSignal()
    # fps, temp, qsize, cam_dropped, disk_dropped, TOTAL_FRAMES, bandwidth_mbps, bw_source
    stats_updated = pyqtSignal(float, float, int, int, int, int, float, str) 
    error_occurred = pyqtSignal(str)
    camera_status_changed = pyqtSignal(bool)  # True=conectada, False=desconectada

    def __init__(self, settings_file="strobe2.txt", pixel_mode=None):
        super().__init__()
        if pixel_mode:
            self.pixel_mode = pixel_mode
        else:
            self.pixel_mode = "bayer"
            persist_path = Path(__file__).resolve().parent / "persist.json"
            if persist_path.exists():
                try:
                    with open(persist_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        self.pixel_mode = cfg.get("ui", {}).get("pixel_mode", "bayer")
                except Exception:
                    pass
            else:
                config_path = "config.json"
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            cfg = json.load(f)
                            self.pixel_mode = cfg.get("pixel_mode", "bayer")
                    except Exception:
                        pass
        
        # Todos los modos usan el mismo config base (triggers, timers, ISP).
        # setup_camera() sobrescribe PixelFormat según pixel_mode después.
        self.settings_file = "config_bayer.txt"

        self.running = False
        self.device = None
        self.write_queue = None
        self.frame_count = 0
        self.instant_frame_count = 0
        self.start_time = 0
        self.last_stats_time = 0
        self.last_frame_id = -1
        self.dropped_frames = 0 # Driver drops
        self.disk_dropped_frames = 0 # App/Disk drops
        self.last_stream_bytes = 0
        self.last_os_bytes = 0 # Para medición psutil

        # Lock para acceso thread-safe al nodemap de la cámara
        self._cmd_lock = threading.Lock()
        self._short_mode_active = False  # Estado conocido del ShortExposureEnable
        # Calibración en vivo: alternar trigger / auto-exposición sin reiniciar el hilo
        self._stream_cmd_lock = threading.Lock()
        self._pending_stream_cmd = None  # None | "enter_calib" | "exit_calib"
        self._restore_exp_short = False
        self._restore_exp_us = 50.0
        self._restore_exp_lock = threading.Lock()
        self._refresh_temp_node = False
        self._isp_baseline = None
        
        # Cola y Worker para Preview (Desacoplado)
        self.preview_queue = queue.Queue(maxsize=2) # Max 2 frames de lag visual, si se llena descartamos
        self.preview_worker = PreviewWorker(self.preview_queue)
        # Reenviamos la señal del worker interno hacia fuera para que main_app no se entere del cambio
        self.preview_worker.image_ready.connect(self.image_received.emit)
        self.preview_worker.exposure_stats_updated.connect(self.exposure_stats_updated.emit)
        self.preview_worker.start()

        self._buffer_fail_streak = 0

    def set_preview_mode(self, mode):
        self.preview_worker.set_preview_mode(mode)

    def set_preview_skip_frames(self, enabled):
        self.preview_worker.set_skip_frames(enabled)

    def set_isp_preview_tone(self, gamma, shadow_lift, contrast):
        self.preview_worker.set_isp_tone(gamma, shadow_lift, contrast)

    def set_isp_preview_adjustments(self, tone=None, camera=None, sdk=None):
        """Aplica tono (numpy), nodos cámara (GenICam) y debayer SDK."""
        if tone is not None:
            self.preview_worker.set_isp_tone(*tone)
        if camera:
            with self._cmd_lock:
                self._apply_isp_camera_settings(camera)
        if sdk and "bayer_algorithm" in sdk:
            self.preview_worker.set_bayer_algorithm(sdk["bayer_algorithm"])
            self.preview_worker.request_refresh_isp()

    def get_isp_preview_tone(self):
        return self.preview_worker.get_isp_tone()

    @staticmethod
    def _node_float(node, default=0.0):
        try:
            if node is None:
                return default
            return float(node.value)
        except Exception:
            return default

    @staticmethod
    def _node_bool(node, default=False):
        try:
            if node is None:
                return default
            return bool(node.value)
        except Exception:
            return default

    @staticmethod
    def _node_str(node, default=""):
        try:
            if node is None:
                return default
            return str(node.value)
        except Exception:
            return default

    @staticmethod
    def _clamp_to_node(node, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return value
        try:
            v = max(float(node.min), min(float(node.max), v))
        except Exception:
            pass
        try:
            inc = float(node.inc)
            vmin = float(node.min)
            if inc > 0:
                steps = round((v - vmin) / inc)
                v = vmin + steps * inc
                v = max(vmin, min(float(node.max), v))
        except Exception:
            pass
        return v

    def _safe_set_float(self, nm, node_name, value):
        n = nm.get_node(node_name)
        if n is None or not n.is_writable:
            return
        n.value = self._clamp_to_node(n, value)

    def _safe_set_bool(self, nm, node_name, value):
        n = nm.get_node(node_name)
        if n is None or not n.is_writable:
            return
        n.value = bool(value)

    def _safe_set_str(self, nm, node_name, value):
        n = nm.get_node(node_name)
        if n is None or not n.is_writable:
            return
        n.value = str(value)

    def _isp_settings_defaults(self):
        out = {
            "defect_correction_enable": False,
            "gain": 0.0,
            "gain_auto": "Off",
            "black_level": 0.0,
            "balance_white_enable": True,
            "balance_white_auto": "Off",
            "balance_ratio_red": 1.9248,
            "balance_ratio_green": 1.0,
            "balance_ratio_blue": 2.321,
            "awb_damping": 100.0,
            "lut_enable": False,
            "gamma_enable": True,
            "gamma": 0.6995,
            "ct_enable": True,
            "offset0": 0.0,
            "offset1": 0.0,
            "offset2": 0.0,
            "sharpening_enable": False,
            "sharpening_amount": 0.0,
            "sharpening_threshold": 0.0,
            "isp_clock_speed": "Normal",
        }
        out.update(ISP_CT_GAIN_DEFAULTS)
        return out

    def _read_ct_matrix(self, nm, out):
        try:
            sel = nm.get_node("ColorTransformationSelector")
            if sel and sel.is_writable:
                sel.value = "RGBtoRGB"
            vsel = nm.get_node("ColorTransformationValueSelector")
            vval = nm.get_node("ColorTransformationValue")
            for key, name in zip(ISP_CT_GAIN_KEYS, ISP_CT_GAIN_SELECTORS):
                if vsel and vsel.is_writable:
                    vsel.value = name
                if vval:
                    out[key] = self._node_float(vval, out.get(key, 0.0))
            for key, name in (("offset0", "Offset0"), ("offset1", "Offset1"), ("offset2", "Offset2")):
                if vsel and vsel.is_writable:
                    vsel.value = name
                if vval:
                    out[key] = self._node_float(vval, out.get(key, 0.0))
        except Exception:
            pass

    def _read_isp_camera_settings(self, nodemap=None):
        nm = nodemap or (self.device.nodemap if self.device else None)
        out = self._isp_settings_defaults()
        if nm is None:
            return out
        try:
            out["defect_correction_enable"] = self._node_bool(nm.get_node("DefectCorrectionEnable"))
        except Exception:
            pass
        try:
            gs = nm.get_node("GainSelector")
            if gs and gs.is_writable:
                gs.value = "All"
            out["gain"] = self._node_float(nm.get_node("Gain"))
        except Exception:
            pass
        try:
            out["gain_auto"] = self._node_str(nm.get_node("GainAuto"), "Off")
        except Exception:
            pass
        try:
            bls = nm.get_node("BlackLevelSelector")
            if bls and bls.is_writable:
                bls.value = "All"
            out["black_level"] = self._node_float(nm.get_node("BlackLevel"))
        except Exception:
            pass
        try:
            out["balance_white_enable"] = self._node_bool(nm.get_node("BalanceWhiteEnable"), True)
        except Exception:
            pass
        try:
            out["balance_white_auto"] = self._node_str(nm.get_node("BalanceWhiteAuto"), "Off")
        except Exception:
            pass
        try:
            sel = nm.get_node("BalanceRatioSelector")
            val = nm.get_node("BalanceRatio")
            for key, name in (
                ("balance_ratio_red", "Red"),
                ("balance_ratio_green", "Green"),
                ("balance_ratio_blue", "Blue"),
            ):
                if sel and sel.is_writable:
                    sel.value = name
                if val:
                    out[key] = self._node_float(val, out[key])
        except Exception:
            pass
        try:
            out["awb_damping"] = self._node_float(nm.get_node("AwbDamping"), 100.0)
        except Exception:
            pass
        try:
            out["lut_enable"] = self._node_bool(nm.get_node("LUTEnable"), False)
        except Exception:
            pass
        try:
            out["gamma_enable"] = self._node_bool(nm.get_node("GammaEnable"), True)
        except Exception:
            pass
        try:
            out["gamma"] = self._node_float(nm.get_node("Gamma"), 0.6995)
        except Exception:
            pass
        try:
            out["ct_enable"] = self._node_bool(nm.get_node("ColorTransformationEnable"), True)
        except Exception:
            pass
        self._read_ct_matrix(nm, out)
        for node_name, out_key, is_bool in (
            ("SharpeningEnable", "sharpening_enable", True),
            ("SharpeningAmount", "sharpening_amount", False),
            ("SharpeningThreshold", "sharpening_threshold", False),
        ):
            try:
                node = nm.get_node(node_name)
                if node is None:
                    continue
                if is_bool:
                    out[out_key] = self._node_bool(node, out[out_key])
                else:
                    out[out_key] = self._node_float(node, out[out_key])
            except Exception:
                pass
        try:
            out["isp_clock_speed"] = self._node_str(nm.get_node("ISPClockSpeed"), "Normal")
        except Exception:
            pass
        return out

    def _apply_ct_matrix(self, nm, settings):
        sel = nm.get_node("ColorTransformationSelector")
        if sel and sel.is_writable:
            sel.value = "RGBtoRGB"
        vsel = nm.get_node("ColorTransformationValueSelector")
        vval = nm.get_node("ColorTransformationValue")
        for key, name in zip(ISP_CT_GAIN_KEYS, ISP_CT_GAIN_SELECTORS):
            if key not in settings:
                continue
            try:
                if vsel and vsel.is_writable:
                    vsel.value = name
                if vval and vval.is_writable:
                    vval.value = self._clamp_to_node(vval, settings[key])
            except Exception as e:
                print(f"WARN CCM {name}: {e}")
        for key, name in (("offset0", "Offset0"), ("offset1", "Offset1"), ("offset2", "Offset2")):
            if key not in settings:
                continue
            try:
                if vsel and vsel.is_writable:
                    vsel.value = name
                if vval and vval.is_writable:
                    vval.value = self._clamp_to_node(vval, settings[key])
            except Exception as e:
                print(f"WARN CCM {name}: {e}")

    def _apply_isp_camera_settings(self, settings: dict):
        if not self.device:
            return False
        nm = self.device.nodemap
        try:
            if "defect_correction_enable" in settings:
                self._safe_set_bool(nm, "DefectCorrectionEnable", settings["defect_correction_enable"])
            if "gain_auto" in settings:
                self._safe_set_str(nm, "GainAuto", settings["gain_auto"])
            if "gain" in settings and str(settings.get("gain_auto", "Off")) == "Off":
                gs = nm.get_node("GainSelector")
                if gs and gs.is_writable:
                    gs.value = "All"
                n = nm.get_node("Gain")
                if n and n.is_writable:
                    n.value = self._quantize_gain(n, float(settings["gain"]))
            if "black_level" in settings:
                bls = nm.get_node("BlackLevelSelector")
                if bls and bls.is_writable:
                    bls.value = "All"
                self._safe_set_float(nm, "BlackLevel", settings["black_level"])
            if "balance_white_enable" in settings:
                self._safe_set_bool(nm, "BalanceWhiteEnable", settings["balance_white_enable"])
            if "balance_white_auto" in settings:
                self._safe_set_str(nm, "BalanceWhiteAuto", settings["balance_white_auto"])
            sel = nm.get_node("BalanceRatioSelector")
            val = nm.get_node("BalanceRatio")
            for key, name in (
                ("balance_ratio_red", "Red"),
                ("balance_ratio_green", "Green"),
                ("balance_ratio_blue", "Blue"),
            ):
                if key not in settings:
                    continue
                try:
                    if sel and sel.is_writable:
                        sel.value = name
                    if val and val.is_writable:
                        val.value = self._clamp_to_node(val, settings[key])
                except Exception as e:
                    print(f"WARN BalanceRatio {name}: {e}")
            if "awb_damping" in settings:
                self._safe_set_float(nm, "AwbDamping", settings["awb_damping"])
            if "lut_enable" in settings:
                self._safe_set_bool(nm, "LUTEnable", settings["lut_enable"])
            if "gamma_enable" in settings:
                self._safe_set_bool(nm, "GammaEnable", settings["gamma_enable"])
            if "gamma" in settings:
                self._safe_set_float(nm, "Gamma", settings["gamma"])
            if "ct_enable" in settings:
                self._safe_set_bool(nm, "ColorTransformationEnable", settings["ct_enable"])
            ct_keys = set(ISP_CT_GAIN_KEYS) | {"offset0", "offset1", "offset2"}
            if ct_keys.intersection(settings.keys()):
                self._apply_ct_matrix(nm, settings)
            for src, node_name in (
                ("sharpening_enable", "SharpeningEnable"),
                ("sharpening_amount", "SharpeningAmount"),
                ("sharpening_threshold", "SharpeningThreshold"),
            ):
                if src not in settings:
                    continue
                try:
                    if src.endswith("_enable"):
                        self._safe_set_bool(nm, node_name, settings[src])
                    else:
                        self._safe_set_float(nm, node_name, settings[src])
                except Exception as e:
                    print(f"WARN {node_name}: {e}")
            if "isp_clock_speed" in settings:
                try:
                    self._safe_set_str(nm, "ISPClockSpeed", settings["isp_clock_speed"])
                except Exception as e:
                    print(f"WARN ISPClockSpeed: {e}")
            return True
        except Exception as e:
            print(f"WARN _apply_isp_camera_settings: {e}")
            return False

    def get_isp_camera_settings(self):
        with self._cmd_lock:
            return self._read_isp_camera_settings()

    def get_isp_sdk_settings(self):
        return {"bayer_algorithm": self.preview_worker.get_bayer_algorithm()}

    def set_isp_sdk_settings(self, settings: dict):
        if "bayer_algorithm" in settings:
            self.preview_worker.set_bayer_algorithm(settings["bayer_algorithm"])
            self.preview_worker.request_refresh_isp()

    def set_isp_camera_settings(self, settings: dict):
        if not self.device:
            return
        with self._cmd_lock:
            self._apply_isp_camera_settings(settings)

    def reset_isp_camera_settings(self):
        if not self.device or not self._isp_baseline:
            return
        self.set_isp_camera_settings(dict(self._isp_baseline))

    def reset_isp_preview_all(self):
        """Restablece post-proceso pantalla, SDK debayer y nodos ISP al baseline de config."""
        self.set_isp_preview_tone(1.0, 0.0, 1.0)
        self.set_isp_sdk_settings({"bayer_algorithm": 0})
        self.reset_isp_camera_settings()

    def get_isp_camera_baseline(self):
        if self._isp_baseline:
            return dict(self._isp_baseline)
        return self.get_isp_camera_settings()

    def set_restore_exposure_after_calibration(self, short_mode: bool, value_us: float):
        """Antes de salir del modo calibración, fija exposición manual (slider)."""
        with self._restore_exp_lock:
            self._restore_exp_short = bool(short_mode)
            self._restore_exp_us = float(value_us)

    def request_calibration_live_mode(self, enabled: bool):
        """Encola cambio de disparo (hilo de captura: stop_stream → nodos → start_stream)."""
        with self._stream_cmd_lock:
            self._pending_stream_cmd = "enter_calib" if enabled else "exit_calib"

    def _pop_pending_stream_cmd(self):
        with self._stream_cmd_lock:
            cmd = self._pending_stream_cmd
            self._pending_stream_cmd = None
            return cmd

    def _process_stream_reconfigure(self):
        cmd = self._pop_pending_stream_cmd()
        if not cmd or not self.device:
            return
        try:
            self.device.stop_stream()
        except Exception as e:
            print(f"WARN stop_stream (reconfigure): {e}")
        try:
            cfg_path = Path(__file__).resolve().parent / "config_bayer.txt"
            with self._cmd_lock:
                if cmd == "enter_calib":
                    self._apply_calibration_preview_nodes()
                else:
                    self.apply_settings_from_file(self.device, str(cfg_path))
                    with self._restore_exp_lock:
                        rs, ru = self._restore_exp_short, self._restore_exp_us
                    self._apply_exposure_inner(rs, ru)
        except Exception as e:
            print(f"WARN stream reconfigure: {e}")
        try:
            self.device.start_stream(100)
        except Exception as e:
            print(f"WARN start_stream (reconfigure): {e}")

    def _apply_calibration_preview_nodes(self):
        """Vista en vivo continua: sin trigger, autoexposición y autoganancia (sin límite de FPS)."""
        nm = self.device.nodemap
        try:
            ts = nm.get_node("TriggerSelector")
            if ts and ts.is_writable:
                ts.value = "FrameStart"
        except Exception as e:
            print(f"WARN calib TriggerSelector: {e}")
        try:
            tm = nm.get_node("TriggerMode")
            if tm and tm.is_writable:
                try:
                    tm.value = "Off"
                except Exception:
                    tm.value = False
        except Exception as e:
            print(f"WARN calib TriggerMode: {e}")
        try:
            node_short = nm.get_node("ShortExposureEnable")
            if node_short and node_short.is_writable:
                node_short.value = False
                self._short_mode_active = False
        except Exception as e:
            print(f"WARN calib ShortExposure: {e}")
        try:
            ea = nm.get_node("ExposureAuto")
            if ea and ea.is_writable:
                try:
                    ea.value = "Continuous"
                except Exception:
                    if hasattr(ea, "from_string"):
                        ea.from_string("Continuous")
        except Exception as e:
            print(f"WARN calib ExposureAuto: {e}")
        try:
            ga = nm.get_node("GainAuto")
            if ga and ga.is_writable:
                try:
                    ga.value = "Continuous"
                except Exception:
                    if hasattr(ga, "from_string"):
                        ga.from_string("Continuous")
        except Exception as e:
            print(f"WARN calib GainAuto: {e}")

    def _set_exposure_auto_off(self, nodemap):
        """La exposición manual requiere ExposureAuto=Off (p. ej. tras vista de calibración)."""
        try:
            ea = nodemap.get_node("ExposureAuto")
            if ea and ea.is_writable:
                try:
                    ea.value = "Off"
                except Exception:
                    if hasattr(ea, "from_string"):
                        ea.from_string("Off")
        except Exception as e:
            print(f"WARN ExposureAuto Off: {e}")

    def get_exposure_gain_ranges(self):
        """
        Devuelve rangos (min, max) de ExposureTime (µs) y Gain actuales.
        Estructura: {"exp": (min_us, max_us), "gain": (min_gain, max_gain)}
        """
        if not self.device:
            return None
        with self._cmd_lock:
            nm = self.device.nodemap
            exp_min = exp_max = gain_min = gain_max = None
            try:
                node_exp = nm.get_node("ExposureTime")
                if node_exp:
                    exp_min = float(node_exp.min)
                    exp_max = float(node_exp.max)
            except Exception as e:
                print(f"WARN get_exposure_gain_ranges ExposureTime: {e}")
            try:
                node_gain = nm.get_node("Gain")
                if node_gain:
                    gain_min = float(node_gain.min)
                    gain_max = float(node_gain.max)
            except Exception as e:
                print(f"WARN get_exposure_gain_ranges Gain: {e}")
        return {"exp": (exp_min, exp_max), "gain": (gain_min, gain_max)}

    def _quantize_exposure_time_us(self, node_exp, value_us: float) -> float:
        """
        Ajusta el tiempo al rastro permitido por GenICam (min/max/inc e inc_mode).
        Valores como 27,5 µs suelen fallar con SC_ERR -1001 si el paso del sensor no es 2,5 µs.
        """
        v = float(value_us)
        try:
            vmin = float(node_exp.min)
            vmax = float(node_exp.max)
        except Exception:
            return v
        v = max(vmin, min(vmax, v))
        node_type = str(type(node_exp)).lower()
        if "integer" in node_type:
            return float(int(round(v)))
        inc = None
        try:
            inc = float(node_exp.inc)
        except Exception:
            pass
        try:
            imode = node_exp.inc_mode
        except Exception:
            imode = None
        use_fixed_inc = False
        if IncMode is not None:
            try:
                use_fixed_inc = imode == IncMode.FIXED
            except Exception:
                pass
        if not use_fixed_inc and imode is not None:
            try:
                use_fixed_inc = int(imode) == 1
            except Exception:
                pass
        if use_fixed_inc and inc is not None and inc > 0:
            steps = round((v - vmin) / inc)
            q = vmin + steps * inc
            q = max(vmin, min(vmax, q))
            return float(q)
        return float(v)

    def _quantize_gain(self, node_gain, value: float) -> float:
        """Cuantiza Gain al rango/incremento permitido por GenICam."""
        try:
            vmin = float(node_gain.min)
            vmax = float(node_gain.max)
        except Exception:
            return float(value)
        v = max(vmin, min(vmax, float(value)))
        inc = None
        try:
            inc = float(node_gain.inc)
        except Exception:
            pass
        if inc and inc > 0:
            steps = round((v - vmin) / inc)
            q = vmin + steps * inc
            q = max(vmin, min(vmax, q))
            return float(q)
        return float(v)

    def set_exposure_time_calibration(self, value_us: float) -> float:
        """
        Modo calibración: fija ExposureTime en modo manual (sin Short Mode), devolviendo
        el valor realmente aplicado tras cuantización GenICam.
        """
        if not self.device:
            return float(value_us)
        with self._cmd_lock:
            nm = self.device.nodemap
            # Asegurar modo manual (ExposureAuto Off)
            self._set_exposure_auto_off(nm)
            # Forzar ShortExposureEnable desactivado
            try:
                node_short = nm.get_node("ShortExposureEnable")
                if node_short and node_short.is_writable:
                    node_short.value = False
                self._short_mode_active = False
            except Exception as e:
                print(f"WARN calib set exposure short off: {e}")
            node_exp = nm.get_node("ExposureTime")
            if not node_exp or not node_exp.is_writable:
                return float(value_us)
            q = self._quantize_exposure_time_us(node_exp, value_us)
            if abs(q - float(value_us)) > 1e-4:
                print(f"INFO: Calib ExposureTime solicitado {value_us} µs → {q} µs")
            try:
                node_exp.value = float(q)
            except Exception as e:
                print(f"WARN set_exposure_time_calibration: {e}")
            return float(q)

    def set_gain_calibration(self, value: float) -> float:
        """
        Modo calibración: fija Gain en modo manual, devolviendo el valor realmente aplicado.
        """
        if not self.device:
            return float(value)
        with self._cmd_lock:
            nm = self.device.nodemap
            # Asegurar GainAuto=Off
            try:
                ga = nm.get_node("GainAuto")
                if ga and ga.is_writable:
                    try:
                        ga.value = "Off"
                    except Exception:
                        if hasattr(ga, "from_string"):
                            ga.from_string("Off")
            except Exception as e:
                print(f"WARN GainAuto Off (calib): {e}")
            node_gain = nm.get_node("Gain")
            if not node_gain or not node_gain.is_writable:
                return float(value)
            q = self._quantize_gain(node_gain, value)
            try:
                node_gain.value = float(q)
            except Exception as e:
                print(f"WARN set_gain_calibration: {e}")
            return float(q)

    def set_timer0_config(self, duration_us: float, delay_us: float):
        """
        Aplica TimerDuration y TimerDelay sobre Timer0 y verifica lectura posterior.
        Retorna dict con requested/applied y bandera ok.
        """
        result = {
            "ok": False,
            "requested_duration": float(duration_us),
            "requested_delay": float(delay_us),
            "applied_duration": None,
            "applied_delay": None,
            "error": "",
        }
        if not self.device:
            result["error"] = "No hay cámara conectada."
            return result
        with self._cmd_lock:
            try:
                nm = self.device.nodemap
                node_sel = nm.get_node("TimerSelector")
                node_dur = nm.get_node("TimerDuration")
                node_del = nm.get_node("TimerDelay")

                if not node_sel:
                    result["error"] = "Nodo TimerSelector no disponible."
                    return result
                if not node_dur or not node_del:
                    result["error"] = "Nodos TimerDuration/TimerDelay no disponibles."
                    return result
                if not node_sel.is_writable:
                    result["error"] = "TimerSelector no es escribible."
                    return result
                if not node_dur.is_writable or not node_del.is_writable:
                    result["error"] = "TimerDuration/TimerDelay no son escribibles."
                    return result

                # Seleccionar explícitamente Timer0 antes de escribir.
                node_sel.value = "Timer0"
                node_dur.value = float(duration_us)
                node_del.value = float(delay_us)

                # Releer sobre Timer0 para validar aplicación real.
                node_sel.value = "Timer0"
                applied_dur = float(node_dur.value)
                applied_del = float(node_del.value)
                result["applied_duration"] = applied_dur
                result["applied_delay"] = applied_del
                result["ok"] = (
                    abs(applied_dur - float(duration_us)) < 1e-6
                    and abs(applied_del - float(delay_us)) < 1e-6
                )
                if not result["ok"]:
                    result["error"] = (
                        "La cámara devolvió valores distintos a los solicitados."
                    )
            except Exception as e:
                result["error"] = str(e)
        return result

    def get_timer0_config(self):
        """Lee TimerDuration y TimerDelay de Timer0."""
        if not self.device:
            return None
        with self._cmd_lock:
            try:
                nm = self.device.nodemap
                node_sel = nm.get_node("TimerSelector")
                node_dur = nm.get_node("TimerDuration")
                node_del = nm.get_node("TimerDelay")
                if not node_sel or not node_dur or not node_del:
                    return None
                node_sel.value = "Timer0"
                return {
                    "duration": float(node_dur.value),
                    "delay": float(node_del.value),
                }
            except Exception as e:
                print(f"WARN get_timer0_config: {e}")
                return None

    def _apply_exposure_inner(self, short_mode: bool, value_us: float):
        """Aplica exposición asumiendo device presente (llamar bajo _cmd_lock desde fuera)."""
        if not self.device:
            return
        try:
            nodemap = self.device.nodemap
            self._set_exposure_auto_off(nodemap)
            node_short = nodemap.get_node("ShortExposureEnable")

            if short_mode:
                if node_short and node_short.is_writable:
                    node_short.value = True
                    self._short_mode_active = True
                    print("INFO: ShortExposureEnable = True")
            else:
                if self._short_mode_active and node_short and node_short.is_writable:
                    node_short.value = False
                    self._short_mode_active = False
                    print("INFO: ShortExposureEnable = False")
                node_exp = nodemap.get_node("ExposureTime")
                if node_exp and node_exp.is_writable:
                    q = self._quantize_exposure_time_us(node_exp, value_us)
                    if abs(q - float(value_us)) > 1e-4:
                        print(
                            f"INFO: ExposureTime solicitado {value_us} µs → "
                            f"valor permitido por la cámara {q} µs (incremento GenICam)"
                        )
                    node_exp.value = float(q)
                    print(f"INFO: ExposureTime = {q} µs")
        except Exception as e:
            print(f"WARN _apply_exposure_inner: {e!r}")

    def set_queue(self, q):
        self.write_queue = q

    def clear_queue(self):
        if self.write_queue:
            with self.write_queue.mutex:
                self.write_queue.queue.clear()
            print("INFO: Cola de fotogramas limpiada antes de grabar.")

    def reset_drop_count(self):
        self.dropped_frames = 0
        self.disk_dropped_frames = 0
        self.last_frame_id = -1 # Reiniciamos tracking de ID para evitar falsos positivos al reconectar
        print("Contadores de Drop reseteados.")

    def setup_camera(self):
        if not ARENA_AVAILABLE:
            raise RuntimeError("Arena SDK (Cámara) no disponible en este sistema.")
        print("Buscando dispositivos...")
        tries = 0
        while tries < 3:
            devices = system.create_device()
            if not devices:
                time.sleep(1)
                tries += 1
            else:
                self.device = devices[0]
                break
        
        if not self.device:
            raise RuntimeError("No se encontró cámara Lucid.")

        print(f"Cargando configuración de cámara ({Path(self.settings_file).name}, modo {self.pixel_mode})")
        self.apply_settings_from_file(self.device, self.settings_file)
        nodemap = self.device.nodemap
        self._apply_capture_hardware_settings(nodemap)

        try:
            self._isp_baseline = self._read_isp_camera_settings(nodemap)
        except Exception:
            self._isp_baseline = None

        return nodemap.get_node('DeviceTemperature')

    def _apply_capture_hardware_settings(self, nodemap=None):
        """Pixel format, buffers de stream y resolución fija (post-config de archivo)."""
        nodemap = nodemap or self.device.nodemap
        tl_stream = self.device.tl_stream_nodemap
        try:
            if self.pixel_mode in ("bayer", "qoi_rgb"):
                try:
                    nodemap.get_node('PixelFormat').value = 'BayerRG12p'
                except Exception:
                    pass
            else:
                try:
                    nodemap.get_node('PixelFormat').value = 'RGB8'
                except Exception:
                    pass

            try:
                nodemap.get_node('ChunkModeActive').value = False
                nodemap.get_node('GevSCPSPacketSize').value = 9000
            except Exception as e:
                print(f"No se pudo desactivar ChunkMode: {e}")

            try:
                if tl_stream:
                    tl_stream['StreamDefaultBufferCount'].value = 200
            except Exception:
                pass

            if tl_stream:
                tl_stream['StreamBufferHandlingMode'].value = "NewestOnly"
                tl_stream['StreamAutoNegotiatePacketSize'].value = True
                tl_stream['StreamPacketResendEnable'].value = True

            if nodemap.get_node('GammaEnable'):
                nodemap.get_node('GammaEnable').value = True

            try:
                nodemap.get_node('OffsetX').value = FIXED_OFFSET_X
                nodemap.get_node('OffsetY').value = FIXED_OFFSET_Y
                nodemap.get_node('Width').value = FIXED_WIDTH
                nodemap.get_node('Height').value = FIXED_HEIGHT
                print(
                    f"Resolución fijada a: {FIXED_WIDTH}x{FIXED_HEIGHT} "
                    f"@ offset ({FIXED_OFFSET_X}, {FIXED_OFFSET_Y})"
                )
            except Exception as e:
                print(f"Error fijando resolución: {e}")
        except Exception as e:
            print(f"Warning _apply_capture_hardware_settings: {e}")

    def apply_settings_from_file(self, device, filepath):
        if not os.path.exists(filepath): 
            print(f"ERROR: No se encontró el archivo {filepath}")
            return
            
        print(f"INFO: Aplicando parámetros de cámara...")
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        nodemap = device.nodemap 
        errors = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                key = parts[0].strip()
                val_str = parts[-1].strip()

                # --- FILTRO NUEVO: IGNORAR NODOS DE SOLO LECTURA ---
                # CounterValue es un indicador, no se puede configurar.
                # Chunk data también suele ser readonly o automático.
                if "CounterValue" in key or "Chunk" in key:
                    continue
                # ---------------------------------------------------
                
                
                try:
                    node = nodemap.get_node(key)
                    if node is None or not node.is_writable:
                        continue

                    # --- DETECCIÓN INTELIGENTE DE TIPO ---
                    # Obtenemos el tipo de nodo como string para saber cómo tratarlo
                    # Ej: "<class 'arena_api.node.NodeBoolean'>"
                    node_type = str(type(node)).lower()

                    try:
                        if 'boolean' in node_type:
                            # Arena espera True/False, no "0" o "1" strings
                            if val_str.lower() in ['1', 'true', 'on']:
                                node.value = True
                            else:
                                node.value = False
                        
                        elif 'float' in node_type:
                            # Soporta "2e+03" y "48.5"
                            node.value = float(val_str)
                            
                        elif 'integer' in node_type:
                            # Soporta "100"
                            node.value = int(float(val_str)) # float() intermedio por si viene como "100.0"
                        
                        else:
                            # Enums y Strings directos
                            # Aquí sí intentamos asignar el string directo
                            # Y si falla, usamos from_string solo si es Enum
                            try:
                                node.value = val_str
                            except:
                                if hasattr(node, 'from_string'):
                                    node.from_string(val_str)
                                else:
                                    raise # Relanzar error si no tiene arreglo

                    except Exception as e_conv:
                        errors.append(f"{key} ({node_type}) -> {val_str}: {e_conv}")

                except Exception as e:
                    errors.append(f"{key} -> {val_str}: {str(e)}")

        if errors:
            print(f"WARN: Hubo {len(errors)} problemas aplicando la configuración:")
            for err in errors[:10]: # Muestro 10 para ver más detalle
                print(f"  - {err}")
        else:
            print("EXITO: Configuración aplicada perfectamente.")
# Método set_format_roi eliminado por unificación de resolución.

    def set_exposure_config(self, short_mode: bool, value_us: float):
        """
        Aplica configuración de exposición de forma thread-safe.
        Secuencia correcta según spec PHX081S:
          Entrar Short Mode : ShortExposureEnable=True  (ExposureTime se ignora)
          Salir  Short Mode : ShortExposureEnable=False, LUEGO ExposureTime=value_us
        """
        if not self.device:
            return
        with self._cmd_lock:
            self._apply_exposure_inner(short_mode, value_us)

    def update_exposure(self, value_us):
        """Compatibilidad: delega a set_exposure_config en modo normal."""
        self.set_exposure_config(False, value_us)

    def update_short_exposure_mode(self, enabled: bool):
        """Compatibilidad: delega a set_exposure_config."""
        self.set_exposure_config(enabled, self.exposure_value if hasattr(self, 'exposure_value') else 50.0)

    def update_gamma(self, value):
        if self.device:
            try: self.device.nodemap.get_node('Gamma').value = float(value)
            except: pass

    def _release_device(self, notify_disconnect=True):
        if self.device:
            try:
                self.device.stop_stream()
            except Exception:
                pass
            try:
                system.destroy_device(self.device)
            except Exception:
                pass
            self.device = None
        if notify_disconnect:
            self.camera_status_changed.emit(False)

    def run(self):
        self.running = True
        try:
            while self.running:
                # --- Fase de conexión (reintenta cada CAMERA_RECONNECT_SEC) ---
                while self.running and not self.device:
                    try:
                        self.setup_camera()
                        self.camera_status_changed.emit(True)
                        self.config_applied.emit()
                    except Exception as setup_err:
                        print(f"WARN: No se pudo conectar a la cámara: {setup_err}")
                        self._release_device(notify_disconnect=True)
                        waited = 0.0
                        while self.running and waited < CAMERA_RECONNECT_SEC:
                            time.sleep(0.1)
                            waited += 0.1

                if not self.running or not self.device:
                    break

                temp_node = self.device.nodemap.get_node('DeviceTemperature')
                self.start_time = time.time()
                self.last_stats_time = time.time()
                current_temp = 0.0
                self.frame_count = 0
                self.instant_frame_count = 0
                self.dropped_frames = 0
                self.last_frame_id = -1
                self.preview_skip_counter = 0
                self._buffer_fail_streak = 0

                try:
                    self.device.start_stream(100)
                except Exception as e:
                    print(f"Error start_stream: {e}")
                    self._release_device(notify_disconnect=True)
                    continue

                stream_node = None
                try:
                    stream_node = self.device.tl_stream_nodemap.get_node('StreamTotalBytes')
                except Exception:
                    try:
                        stream_node = self.device.tl_stream_nodemap.get_node('StreamOctets')
                    except Exception:
                        stream_node = None

                stream_lost = False
                while self.running and self.device:
                    if self._refresh_temp_node:
                        self._refresh_temp_node = False
                        try:
                            temp_node = self.device.nodemap.get_node('DeviceTemperature')
                        except Exception:
                            pass
                    self._process_stream_reconfigure()
                    now = time.time()
                    dt = now - self.last_stats_time
                    if dt > 1.0:
                        try:
                            current_temp = temp_node.value
                            self._buffer_fail_streak = 0
                            fps = self.instant_frame_count / dt
                            mbps = 0.0
                            bw_src = "C"

                            if PSUTIL_AVAILABLE:
                                try:
                                    net = psutil.net_io_counters()
                                    curr_os_bytes = net.bytes_recv
                                    if self.last_os_bytes > 0:
                                        delta = curr_os_bytes - self.last_os_bytes
                                        mbps = (delta * 8) / (dt * 1_000_000.0)
                                    self.last_os_bytes = curr_os_bytes
                                    bw_src = "OS"
                                except Exception:
                                    pass

                            if bw_src == "C" and stream_node:
                                try:
                                    curr_bytes = stream_node.value
                                    if self.last_stream_bytes > 0:
                                        delta = curr_bytes - self.last_stream_bytes
                                        mbps = (delta * 8) / (dt * 1_000_000.0)
                                    self.last_stream_bytes = curr_bytes
                                    bw_src = "D"
                                except Exception:
                                    pass

                            if bw_src == "C":
                                if self.pixel_mode == "qoi_rgb":
                                    frame_size_mb = (2840 * 2200 * 3 * 0.5) / (1024 * 1024)
                                elif self.pixel_mode == "rgb":
                                    frame_size_mb = (2840 * 2200 * 3) / (1024 * 1024)
                                else:
                                    frame_size_mb = (2840 * 2200 * 1.5) / (1024 * 1024)
                                mbps = fps * frame_size_mb * 8

                            self.instant_frame_count = 0
                            self.last_stats_time = now
                            q_size = self.write_queue.qsize() if self.write_queue else 0
                            self.stats_updated.emit(
                                fps, current_temp, q_size, self.dropped_frames,
                                self.disk_dropped_frames, self.frame_count, mbps, bw_src)
                        except Exception as dev_err:
                            err_s = str(dev_err).lower()
                            if "timeout" not in err_s and "timed out" not in err_s:
                                self._buffer_fail_streak += 1
                                if self._buffer_fail_streak >= 3:
                                    print(f"WARN: Cámara inaccesible, reintentando: {dev_err}")
                                    stream_lost = True
                                    break
                            pass

                    try:
                        buffer = self.device.get_buffer(timeout=200)
                        self._buffer_fail_streak = 0
                    except Exception as buf_err:
                        if not self.running:
                            break
                        err_s = str(buf_err).lower()
                        # Sin disparo (modo trigger) get_buffer hace timeout: es normal, no desconectar.
                        if "timeout" in err_s or "timed out" in err_s:
                            continue
                        self._buffer_fail_streak += 1
                        if self._buffer_fail_streak >= 5:
                            print(f"WARN: Error de cámara, reintentando conexión: {buf_err}")
                            stream_lost = True
                            break
                        continue

                    capture_rgb_via_sdk = (self.pixel_mode == "qoi_rgb")

                    if buffer.is_incomplete:
                        self.device.requeue_buffer(buffer)
                        continue

                    curr_id = buffer.frame_id
                    if self.last_frame_id != -1:
                        diff = curr_id - self.last_frame_id
                        if diff > 1:
                            self.dropped_frames += diff - 1
                    self.last_frame_id = curr_id

                    self.frame_count += 1
                    self.instant_frame_count += 1

                    if capture_rgb_via_sdk:
                        try:
                            image_raw = BufferFactory.copy(buffer)
                            self.device.requeue_buffer(buffer)
                        except Exception:
                            self.device.requeue_buffer(buffer)
                            continue

                        try:
                            converted = BufferFactory.convert(image_raw, PixelFormat.BGR8)
                            h, w = converted.height, converted.width
                            full_arr = np.ctypeslib.as_array(converted.pdata, shape=(h, w, 3))
                            rgb_arr = full_arr.copy()
                            BufferFactory.destroy(converted)
                        except Exception as e_conv:
                            print(f"ERROR RGB capture convert: {e_conv}")
                            BufferFactory.destroy(image_raw)
                            continue

                        if self.write_queue:
                            try:
                                self.write_queue.put_nowait(rgb_arr.tobytes())
                            except queue.Full:
                                self.disk_dropped_frames += 1

                        try:
                            self.preview_queue.put_nowait(image_raw)
                            image_raw = None
                        except queue.Full:
                            pass

                        if image_raw:
                            BufferFactory.destroy(image_raw)
                        continue

                    image_raw = BufferFactory.copy(buffer)
                    self.device.requeue_buffer(buffer)

                    h, w = image_raw.height, image_raw.width
                    is_rgb = (image_raw.pixel_format == PixelFormat.RGB8 or image_raw.pixel_format == PixelFormat.BGR8)

                    if is_rgb:
                        expected_size = int(w * h * 3)
                        data_full = np.ctypeslib.as_array(image_raw.pdata, shape=(image_raw.size_filled,))
                        if data_full.size > expected_size:
                            raw_bytes = data_full[:expected_size].copy().tobytes()
                        else:
                            raw_bytes = data_full.copy().tobytes()
                    else:
                        expected_size = int(w * h * 1.5)
                        data_full = np.ctypeslib.as_array(image_raw.pdata, shape=(image_raw.size_filled,))
                        if data_full.size > expected_size:
                            raw_bytes = data_full[:expected_size].copy().tobytes()
                        else:
                            raw_bytes = data_full.copy().tobytes()

                    if self.write_queue:
                        try:
                            self.write_queue.put_nowait(raw_bytes)
                        except queue.Full:
                            self.disk_dropped_frames += 1

                    try:
                        self.preview_queue.put_nowait(image_raw)
                    except queue.Full:
                        BufferFactory.destroy(image_raw)

                if stream_lost:
                    self._release_device(notify_disconnect=True)

        except Exception as e:
            if "aborted" not in str(e).lower() and self.running:
                self.error_occurred.emit(str(e))
        finally:
            self._release_device(notify_disconnect=False)

    def stop(self):
        self.running = False
        self.wait()
        if self.preview_worker:
            self.preview_worker.stop()
        self._release_device(notify_disconnect=False)

# --- WORKER DE ESCRITURA (Sin cambios) ---
class WriterWorker(QThread):
    frames_saved_signal = pyqtSignal(int)
    def __init__(self, frame_queue, save_path):
        super().__init__()
        self.frame_queue = frame_queue
        self.save_path = save_path
        self.running = True
        self.frames_saved = 0

    def run(self):
        # Aumentar buffer de archivo a 64MB (64*1024*1024) para minimizar I/O ops
        with open(self.save_path, "wb", buffering=67108864) as f:
            while self.running or not self.frame_queue.empty():
                try:
                    data = self.frame_queue.get(timeout=0.1)
                    
                    t0 = time.perf_counter()
                    f.write(data)
                    t1 = time.perf_counter()
                    
                    dt_ms = (t1 - t0) * 1000.0
                    if dt_ms > 40.0:
                        print(f"ALERTA DISCO: Escritura lenta detectada ({dt_ms:.1f} ms). Posible cuello de botella.")
                    
                    self.frames_saved += 1
                    # Optimización: No emitir señal CADA frame, sino cada 5 frames para liberar CPU del GUI
                    if self.frames_saved % 5 == 0:
                        self.frames_saved_signal.emit(self.frames_saved)
                        
                    self.frame_queue.task_done()
                except queue.Empty: continue
    
    def stop(self):
        self.running = False
        self.wait()

# --- GESTOR DE COLECCIONES (Sin cambios) ---
class CollectionManager:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def get_collections(self):
        return sorted([d.name for d in self.root_path.iterdir() if d.is_dir()])

    def create_collection(self, name):
        path = self.root_path / name
        if not path.exists():
            path.mkdir()
            self.save_collection_config(
                name, {"filename_pattern": "{coleccion}_{n:03d}.raw"})
            return True
        return False

    def get_collection_config_path(self, collection_name):
        return self.root_path / collection_name / "collection_config.json"

    def get_collection_config(self, collection_name):
        p = self.get_collection_config_path(collection_name)
        default = {"filename_pattern": "{coleccion}_{n:03d}.raw"}
        if not p.exists():
            return dict(default)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = dict(default)
            if isinstance(data, dict):
                out.update(data)
            return out
        except Exception:
            return dict(default)

    def save_collection_config(self, collection_name, updates: dict):
        cur = self.get_collection_config(collection_name)
        cur.update(updates)
        p = self.get_collection_config_path(collection_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2, ensure_ascii=False)

    def validate_filename_pattern(self, collection_name: str, pattern: str) -> None:
        if not pattern or not isinstance(pattern, str):
            raise ValueError("La plantilla no puede estar vacía.")
        if not re.search(r"\{n[}:]", pattern):
            raise ValueError(
                "La plantilla debe incluir {n} o {n:04d} (numeración obligatoria).")
        if ".raw" not in pattern.lower():
            raise ValueError("La plantilla debe generar un nombre que termine en .raw")
        try:
            test = pattern.format(coleccion=collection_name, n=1)
        except KeyError as e:
            raise ValueError(
                f"Solo se permiten los placeholders {{coleccion}} y {{n}}…: {e}") from e
        except ValueError as e:
            raise ValueError(f"Formato inválido en {{n}}: {e}") from e
        if "/" in test or "\\" in test or Path(test).name != test:
            raise ValueError("No use barras ni rutas; solo el nombre del archivo.")
        if not test.lower().endswith(".raw"):
            raise ValueError("El nombre debe terminar en .raw")

    def _next_numeric_suffix_in_collection(self, coll_path: Path) -> int:
        max_n = 0
        if not coll_path.is_dir():
            return max_n
        for f in coll_path.glob("*.raw"):
            m = re.search(r"(\d+)$", f.stem)
            if m:
                try:
                    max_n = max(max_n, int(m.group(1)))
                except ValueError:
                    pass
        return max_n

    def peek_next_filename(self, collection_name, pattern=None):
        coll_path = self.root_path / collection_name
        cfg = self.get_collection_config(collection_name)
        pat = pattern if pattern is not None else cfg.get(
            "filename_pattern", "{coleccion}_{n:03d}.raw")
        self.validate_filename_pattern(collection_name, pat)
        next_n = self._next_numeric_suffix_in_collection(coll_path) + 1
        return pat.format(coleccion=collection_name, n=next_n)

    def get_next_filename(self, collection_name):
        coll_path = self.root_path / collection_name
        cfg = self.get_collection_config(collection_name)
        pattern = cfg.get("filename_pattern", "{coleccion}_{n:03d}.raw")
        self.validate_filename_pattern(collection_name, pattern)
        next_n = self._next_numeric_suffix_in_collection(coll_path) + 1
        for bump in range(200000):
            name = pattern.format(coleccion=collection_name, n=next_n + bump)
            if "/" in name or "\\" in name or Path(name).name != name:
                raise ValueError("Patrón de nombre inválido (segmentos de ruta).")
            fp = coll_path / name
            if not fp.exists():
                return name, str(fp)
        raise ValueError("No se encontró nombre .raw libre (demasiados intentos).")

    def rename_collection(self, old_name: str, new_name: str) -> bool:
        old_name, new_name = old_name.strip(), new_name.strip()
        if not old_name or not new_name or old_name == new_name:
            return False
        old_p = self.root_path / old_name
        new_p = self.root_path / new_name
        if not old_p.is_dir() or new_p.exists():
            return False
        try:
            old_p.rename(new_p)
        except OSError:
            return False
        return True

    def rename_file(self, collection_name: str, old_name: str, new_name: str) -> bool:
        old_name, new_name = old_name.strip(), new_name.strip()
        if not old_name or not new_name or old_name == new_name:
            return False
        if not new_name.lower().endswith(".raw"):
            new_name = new_name + ".raw"
        old_p = self.root_path / collection_name / old_name
        new_p = self.root_path / collection_name / new_name
        if not old_p.is_file() or new_p.exists():
            return False
        if "/" in new_name or "\\" in new_name:
            return False
        try:
            old_p.rename(new_p)
        except OSError:
            return False
        data = self.load_metadata(collection_name)
        if old_name in data:
            data[new_name] = data.pop(old_name)
            self.save_metadata(collection_name, data)
        return True

    def move_file_to_collection(
            self, src_collection: str, dst_collection: str, filename: str) -> bool:
        src_collection = src_collection.strip()
        dst_collection = dst_collection.strip()
        filename = filename.strip()
        if not src_collection or not dst_collection or src_collection == dst_collection:
            return False
        src_p = self.root_path / src_collection / filename
        dst_dir = self.root_path / dst_collection
        dst_p = dst_dir / filename
        if not src_p.is_file() or not dst_dir.is_dir() or dst_p.exists():
            return False
        try:
            shutil.move(str(src_p), str(dst_p))
        except OSError:
            return False
        src_meta = self.load_metadata(src_collection)
        info = src_meta.pop(filename, {})
        self.save_metadata(src_collection, src_meta)
        dst_meta = self.load_metadata(dst_collection)
        dst_meta[filename] = info
        self.save_metadata(dst_collection, dst_meta)
        return True

    def get_metadata_path(self, collection_name):
        return self.root_path / collection_name / "metadata.json"

    def load_metadata(self, collection_name):
        json_path = self.get_metadata_path(collection_name)
        if json_path.exists():
            try:
                with open(json_path, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def save_metadata(self, collection_name, data):
        with open(self.get_metadata_path(collection_name), 'w') as f:
            json.dump(data, f, indent=4)

    def get_fps(self, collection_name, filename):
        data = self.load_metadata(collection_name)
        return data.get(filename, {}).get("fps", 18)

    def set_fps(self, collection_name, filename, fps):
        data = self.load_metadata(collection_name)
        if filename not in data: data[filename] = {}
        data[filename]["fps"] = fps
        self.save_metadata(collection_name, data)

    def set_file_info(self, collection_name, filename, **kwargs):
        """
        Guarda metadatos de forma flexible.
        Uso: manager.set_file_info(col, file, fps=18, pixel_format="rgb", ...)
        """
        data = self.load_metadata(collection_name)
        if filename not in data: data[filename] = {}
        
        # Guardamos/Actualizamos todos los argumentos que lleguen
        data[filename].update(kwargs)
        
        self.save_metadata(collection_name, data)

    def get_file_info(self, collection_name, filename):
        data = self.load_metadata(collection_name)
        return data.get(filename, {}) 

    def delete_file(self, collection_name, filename):
        file_path = self.root_path / collection_name / filename
        coll_dir = self.root_path / collection_name
        stem = Path(filename).stem

        if file_path.exists():
            try:
                os.remove(file_path)
            except OSError:
                return False

        dng_folder = coll_dir / f"{stem}_DNG_SEQ"
        if dng_folder.is_dir():
            try:
                shutil.rmtree(dng_folder)
            except OSError as e:
                print(f"WARN no se pudo borrar carpeta DNG {dng_folder.name}: {e}")

        data = self.load_metadata(collection_name)
        if filename in data:
            del data[filename]
            self.save_metadata(collection_name, data)
        return True