"""Diálogo de ajustes ISP (post-proceso pantalla, nodos GenICam y debayer SDK)."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QSlider,
    QPushButton, QCheckBox, QComboBox, QTabWidget, QWidget, QScrollArea,
    QGridLayout, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QTimer

import scanner_core


def _slider_row(parent_layout, title, tip, lo, hi, default, fmt_fn):
    box = QGroupBox(title)
    box.setToolTip(tip)
    v = QVBoxLayout(box)
    row = QHBoxLayout()
    sld = QSlider(Qt.Orientation.Horizontal)
    sld.setRange(lo, hi)
    sld.setValue(default)
    lbl = QLabel(fmt_fn(default))
    lbl.setMinimumWidth(56)
    lbl.setStyleSheet("color: #aaa; font-size: 9pt;")
    row.addWidget(sld, 1)
    row.addWidget(lbl)
    v.addLayout(row)
    parent_layout.addWidget(box)
    return sld, lbl, fmt_fn


class IspPreviewToneDialog(QDialog):
    """Ajustes en vivo del preview ISP/ISP Full (pantalla + nodos cámara + SDK)."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Ajuste visor ISP")
        self.setMinimumSize(500, 560)
        self.resize(520, 640)

        outer = QVBoxLayout(self)
        info = QLabel(
            "Modos <b>ISP</b> / <b>ISP Full</b>. Pipeline cámara: "
            "Defect → Gain/BlackLevel → Debayer* → WB → LUT → Gamma → CCM → salida.<br>"
            "Con <b>BayerRG12p</b> la grabación es cruda; el visor ISP usa "
            "<code>BufferFactory.convert</code> en CPU (debayer SDK, no GPU).<br>"
            "Nodos post-debayer de cámara (WB/LUT/Gamma/CCM) pueden no verse en el preview SDK; "
            "Gain/BlackLevel sí afectan el Bayer."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #999; font-size: 9pt;")
        outer.addWidget(info)

        self._sliders = []
        self._tabs = QTabWidget()
        self._tabs.addTab(self._tab_screen(), "Pantalla")
        self._tabs.addTab(self._tab_sdk(), "SDK debayer")
        self._tabs.addTab(self._tab_signal(), "Señal")
        self._tabs.addTab(self._tab_wb(), "Balance blancos")
        self._tabs.addTab(self._tab_gamma_lut(), "Gamma / LUT")
        self._tabs.addTab(self._tab_ccm(), "Matriz color")
        self._tabs.addTab(self._tab_extra(), "Nitidez / ISP")
        outer.addWidget(self._tabs, 1)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Restablecer todo")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        outer.addLayout(btn_row)

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(180)
        self._apply_timer.timeout.connect(self._apply_to_worker)

        self._load_from_worker()

    def _scroll_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        lay = QVBoxLayout(inner)
        return scroll, lay

    def _tab_screen(self):
        scroll, lay = self._scroll_tab()
        self.sld_gamma, self.lbl_gamma, self._fmt_gamma = _slider_row(
            lay, "Gamma display",
            "Curva extra tras el SDK. Mayor ≈ más claro (tipo Bayer crudo). "
            "Se aplica también en exportación procesada (video/TIF), no en DNG.",
            50, 300, 100,
            lambda v: f"{v / 100:.2f}",
        )
        self.sld_lift, self.lbl_lift, self._fmt_lift = _slider_row(
            lay, "Lift sombras", "Eleva negros (solo software).", 0, 40, 0,
            lambda v: f"{v}%",
        )
        self.sld_contrast, self.lbl_contrast, self._fmt_contrast = _slider_row(
            lay, "Contraste", "100 = neutro.", 80, 130, 100, lambda v: f"{v}%",
        )
        lay.addStretch()
        self._wire_sliders(self.sld_gamma, self.sld_lift, self.sld_contrast)
        return scroll

    def _tab_sdk(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        sdk_info = QLabel(
            "Algoritmo de demosaico del host (Arena <code>BufferFactory.convert</code>). "
            "No replica WB/Gamma/CCM de la cámara."
        )
        sdk_info.setWordWrap(True)
        sdk_info.setStyleSheet("color: #888; font-size: 9pt;")
        lay.addWidget(sdk_info)
        row = QHBoxLayout()
        row.addWidget(QLabel("Bayer algorithm:"))
        self.combo_bayer = QComboBox()
        self.combo_bayer.addItem("Default (UNKNOWN)", 0)
        self.combo_bayer.addItem("Directional interpolation", 1)
        self.combo_bayer.addItem("Adaptive homogeneity directed", 2)
        self.combo_bayer.currentIndexChanged.connect(self._on_change)
        row.addWidget(self.combo_bayer, 1)
        lay.addLayout(row)
        lay.addStretch()
        return w

    def _tab_signal(self):
        scroll, lay = self._scroll_tab()
        self.chk_defect = QCheckBox("DefectCorrectionEnable")
        self.chk_defect.toggled.connect(self._on_change)
        lay.addWidget(self.chk_defect)

        gain_lo, gain_hi = 0, 480
        w = self.main_window.camera_worker
        if w:
            ranges = w.get_exposure_gain_ranges()
            if ranges and ranges.get("gain") and ranges["gain"][1]:
                gain_lo = int(ranges["gain"][0] or 0)
                gain_hi = int(ranges["gain"][1] or 480)

        ga_row = QHBoxLayout()
        ga_row.addWidget(QLabel("GainAuto:"))
        self.combo_gain_auto = QComboBox()
        self.combo_gain_auto.addItems(["Off", "Once", "Continuous"])
        self.combo_gain_auto.currentTextChanged.connect(self._on_change)
        ga_row.addWidget(self.combo_gain_auto)
        lay.addLayout(ga_row)

        self.sld_gain, self.lbl_gain, self._fmt_gain = _slider_row(
            lay, "Gain", "Amplificación analógica/digital (afecta Bayer).",
            gain_lo, gain_hi, gain_lo, lambda v: f"{v:.1f}",
        )
        self.sld_black, self.lbl_black, self._fmt_black = _slider_row(
            lay, "Black level", "Nodo BlackLevel (All).", 0, 512, 0, lambda v: str(v),
        )
        lay.addStretch()
        self._wire_sliders(self.sld_gain, self.sld_black)
        return scroll

    def _tab_wb(self):
        scroll, lay = self._scroll_tab()
        self.chk_wb_en = QCheckBox("BalanceWhiteEnable")
        self.chk_wb_en.toggled.connect(self._on_change)
        lay.addWidget(self.chk_wb_en)
        awb_row = QHBoxLayout()
        awb_row.addWidget(QLabel("BalanceWhiteAuto:"))
        self.combo_wb_auto = QComboBox()
        self.combo_wb_auto.addItems(["Off", "Once", "Continuous"])
        self.combo_wb_auto.currentTextChanged.connect(self._on_change)
        awb_row.addWidget(self.combo_wb_auto)
        lay.addLayout(awb_row)
        self.sld_ratio_r, self.lbl_ratio_r, self._fmt_ratio_r = _slider_row(
            lay, "BalanceRatio Red", "Ganancia canal R.", 50, 400, 192,
            lambda v: f"{v / 100:.2f}",
        )
        self.sld_ratio_g, self.lbl_ratio_g, self._fmt_ratio_g = _slider_row(
            lay, "BalanceRatio Green", "Ganancia canal G.", 50, 400, 100,
            lambda v: f"{v / 100:.2f}",
        )
        self.sld_ratio_b, self.lbl_ratio_b, self._fmt_ratio_b = _slider_row(
            lay, "BalanceRatio Blue", "Ganancia canal B.", 50, 400, 232,
            lambda v: f"{v / 100:.2f}",
        )
        self.sld_awb_damp, self.lbl_awb_damp, self._fmt_awb_damp = _slider_row(
            lay, "AwbDamping", "Suavizado convergencia AWB.", 0, 100, 100,
            lambda v: f"{v}%",
        )
        lay.addStretch()
        self._wire_sliders(
            self.sld_ratio_r, self.sld_ratio_g, self.sld_ratio_b, self.sld_awb_damp,
        )
        return scroll

    def _tab_gamma_lut(self):
        scroll, lay = self._scroll_tab()
        self.chk_gamma_en = QCheckBox("GammaEnable")
        self.chk_gamma_en.toggled.connect(self._on_change)
        lay.addWidget(self.chk_gamma_en)
        self.sld_cam_gamma, self.lbl_cam_gamma, self._fmt_cam_gamma = _slider_row(
            lay, "Gamma cámara", "Y = X^Gamma (0.2–2.0).", 20, 200, 70,
            lambda v: f"{v / 100:.2f}",
        )
        self.chk_lut_en = QCheckBox("LUTEnable (tabla Luminance en cámara)")
        self.chk_lut_en.setToolTip(
            "Activa LUT guardada en cámara. Edición punto a punto: nodos LUTIndex/LUTValue o config."
        )
        self.chk_lut_en.toggled.connect(self._on_change)
        lay.addWidget(self.chk_lut_en)
        lay.addStretch()
        self._wire_sliders(self.sld_cam_gamma)
        return scroll

    def _tab_ccm(self):
        scroll, lay = self._scroll_tab()
        self.chk_ct_en = QCheckBox("ColorTransformationEnable (RGB→RGB)")
        self.chk_ct_en.toggled.connect(self._on_change)
        lay.addWidget(self.chk_ct_en)

        grid = QGridLayout()
        grid.addWidget(QLabel(""), 0, 0)
        for col, label in enumerate(("R", "G", "B")):
            grid.addWidget(QLabel(label), 0, col + 1)
        self._ct_spins = {}
        for row, row_label in enumerate(("R", "G", "B")):
            grid.addWidget(QLabel(row_label), row + 1, 0)
            for col in range(3):
                key = f"gain{row}{col}"
                spin = QDoubleSpinBox()
                spin.setRange(-3.0, 3.0)
                spin.setDecimals(4)
                spin.setSingleStep(0.01)
                spin.valueChanged.connect(self._on_change)
                grid.addWidget(spin, row + 1, col + 1)
                self._ct_spins[key] = spin
        lay.addLayout(grid)

        off_grp = QGroupBox("Offsets CCM")
        off_lay = QVBoxLayout(off_grp)
        self.sld_off_r, self.lbl_off_r, self._fmt_off_r = _slider_row(
            off_lay, "Offset0 (R)", "", -100, 100, 0, lambda v: f"{v / 100:.2f}",
        )
        self.sld_off_g, self.lbl_off_g, self._fmt_off_g = _slider_row(
            off_lay, "Offset1 (G)", "", -100, 100, 0, lambda v: f"{v / 100:.2f}",
        )
        self.sld_off_b, self.lbl_off_b, self._fmt_off_b = _slider_row(
            off_lay, "Offset2 (B)", "", -100, 100, 0, lambda v: f"{v / 100:.2f}",
        )
        lay.addWidget(off_grp)
        lay.addStretch()
        self._wire_sliders(self.sld_off_r, self.sld_off_g, self.sld_off_b)
        return scroll

    def _tab_extra(self):
        scroll, lay = self._scroll_tab()
        self.chk_sharp_en = QCheckBox("SharpeningEnable")
        self.chk_sharp_en.toggled.connect(self._on_change)
        lay.addWidget(self.chk_sharp_en)
        self.sld_sharp_amt, self.lbl_sharp_amt, self._fmt_sharp_amt = _slider_row(
            lay, "SharpeningAmount", "Intensidad (si el nodo existe).", 0, 100, 0,
            lambda v: str(v),
        )
        self.sld_sharp_thr, self.lbl_sharp_thr, self._fmt_sharp_thr = _slider_row(
            lay, "SharpeningThreshold", "Umbral.", 0, 255, 0, lambda v: str(v),
        )
        clk_row = QHBoxLayout()
        clk_row.addWidget(QLabel("ISPClockSpeed:"))
        self.combo_isp_clock = QComboBox()
        self.combo_isp_clock.addItems(["Normal", "Fast"])
        self.combo_isp_clock.currentTextChanged.connect(self._on_change)
        clk_row.addWidget(self.combo_isp_clock)
        lay.addLayout(clk_row)
        lay.addStretch()
        self._wire_sliders(self.sld_sharp_amt, self.sld_sharp_thr)
        return scroll

    def _wire_sliders(self, *sliders):
        for sld in sliders:
            self._sliders.append(sld)
            sld.valueChanged.connect(self._on_change)

    def _load_from_worker(self):
        w = self.main_window.camera_worker
        for sld in self._sliders:
            sld.blockSignals(True)
        self.combo_bayer.blockSignals(True)
        self.combo_gain_auto.blockSignals(True)
        self.combo_wb_auto.blockSignals(True)
        self.combo_isp_clock.blockSignals(True)
        for spin in self._ct_spins.values():
            spin.blockSignals(True)

        if w:
            g, lift, contrast = w.get_isp_preview_tone()
            self.sld_gamma.setValue(int(round(g * 100)))
            self.sld_lift.setValue(int(round(lift * 200)))
            self.sld_contrast.setValue(int(round(contrast * 100)))
            sdk = w.get_isp_sdk_settings()
            idx = self.combo_bayer.findData(int(sdk.get("bayer_algorithm", 0)))
            if idx >= 0:
                self.combo_bayer.setCurrentIndex(idx)
            cam = w.get_isp_camera_settings()
            self.chk_defect.setChecked(cam.get("defect_correction_enable", False))
            ga = cam.get("gain_auto", "Off")
            gi = self.combo_gain_auto.findText(ga)
            if gi >= 0:
                self.combo_gain_auto.setCurrentIndex(gi)
            self.sld_gain.setValue(int(round(cam.get("gain", 0))))
            self.sld_black.setValue(int(round(cam.get("black_level", 0))))
            self.chk_wb_en.setChecked(cam.get("balance_white_enable", True))
            wa = cam.get("balance_white_auto", "Off")
            wi = self.combo_wb_auto.findText(wa)
            if wi >= 0:
                self.combo_wb_auto.setCurrentIndex(wi)
            self.sld_ratio_r.setValue(int(round(cam.get("balance_ratio_red", 1.0) * 100)))
            self.sld_ratio_g.setValue(int(round(cam.get("balance_ratio_green", 1.0) * 100)))
            self.sld_ratio_b.setValue(int(round(cam.get("balance_ratio_blue", 1.0) * 100)))
            self.sld_awb_damp.setValue(int(round(cam.get("awb_damping", 100))))
            self.chk_gamma_en.setChecked(cam.get("gamma_enable", True))
            self.sld_cam_gamma.setValue(int(round(cam.get("gamma", 0.6995) * 100)))
            self.chk_lut_en.setChecked(cam.get("lut_enable", False))
            self.chk_ct_en.setChecked(cam.get("ct_enable", True))
            for key, spin in self._ct_spins.items():
                spin.setValue(float(cam.get(key, scanner_core.ISP_CT_GAIN_DEFAULTS.get(key, 0.0))))
            self.sld_off_r.setValue(int(round(cam.get("offset0", 0) * 100)))
            self.sld_off_g.setValue(int(round(cam.get("offset1", 0) * 100)))
            self.sld_off_b.setValue(int(round(cam.get("offset2", 0) * 100)))
            self.chk_sharp_en.setChecked(cam.get("sharpening_enable", False))
            self.sld_sharp_amt.setValue(int(round(cam.get("sharpening_amount", 0))))
            self.sld_sharp_thr.setValue(int(round(cam.get("sharpening_threshold", 0))))
            ci = self.combo_isp_clock.findText(cam.get("isp_clock_speed", "Normal"))
            if ci >= 0:
                self.combo_isp_clock.setCurrentIndex(ci)

        for sld in self._sliders:
            sld.blockSignals(False)
        self.combo_bayer.blockSignals(False)
        self.combo_gain_auto.blockSignals(False)
        self.combo_wb_auto.blockSignals(False)
        self.combo_isp_clock.blockSignals(False)
        for spin in self._ct_spins.values():
            spin.blockSignals(False)
        self._refresh_labels()

    def _refresh_labels(self):
        self.lbl_gamma.setText(self._fmt_gamma(self.sld_gamma.value()))
        self.lbl_lift.setText(self._fmt_lift(self.sld_lift.value()))
        self.lbl_contrast.setText(self._fmt_contrast(self.sld_contrast.value()))
        self.lbl_gain.setText(self._fmt_gain(float(self.sld_gain.value())))
        self.lbl_black.setText(self._fmt_black(self.sld_black.value()))
        self.lbl_ratio_r.setText(self._fmt_ratio_r(self.sld_ratio_r.value()))
        self.lbl_ratio_g.setText(self._fmt_ratio_g(self.sld_ratio_g.value()))
        self.lbl_ratio_b.setText(self._fmt_ratio_b(self.sld_ratio_b.value()))
        self.lbl_awb_damp.setText(self._fmt_awb_damp(self.sld_awb_damp.value()))
        self.lbl_cam_gamma.setText(self._fmt_cam_gamma(self.sld_cam_gamma.value()))
        self.lbl_off_r.setText(self._fmt_off_r(self.sld_off_r.value()))
        self.lbl_off_g.setText(self._fmt_off_g(self.sld_off_g.value()))
        self.lbl_off_b.setText(self._fmt_off_b(self.sld_off_b.value()))
        self.lbl_sharp_amt.setText(self._fmt_sharp_amt(self.sld_sharp_amt.value()))
        self.lbl_sharp_thr.setText(self._fmt_sharp_thr(self.sld_sharp_thr.value()))

    def _current_tone(self):
        return (
            self.sld_gamma.value() / 100.0,
            self.sld_lift.value() / 200.0,
            self.sld_contrast.value() / 100.0,
        )

    def _current_sdk(self):
        return {"bayer_algorithm": int(self.combo_bayer.currentData())}

    def _current_camera(self):
        out = {
            "defect_correction_enable": self.chk_defect.isChecked(),
            "gain_auto": self.combo_gain_auto.currentText(),
            "gain": float(self.sld_gain.value()),
            "black_level": float(self.sld_black.value()),
            "balance_white_enable": self.chk_wb_en.isChecked(),
            "balance_white_auto": self.combo_wb_auto.currentText(),
            "balance_ratio_red": self.sld_ratio_r.value() / 100.0,
            "balance_ratio_green": self.sld_ratio_g.value() / 100.0,
            "balance_ratio_blue": self.sld_ratio_b.value() / 100.0,
            "awb_damping": float(self.sld_awb_damp.value()),
            "gamma_enable": self.chk_gamma_en.isChecked(),
            "gamma": self.sld_cam_gamma.value() / 100.0,
            "lut_enable": self.chk_lut_en.isChecked(),
            "ct_enable": self.chk_ct_en.isChecked(),
            "offset0": self.sld_off_r.value() / 100.0,
            "offset1": self.sld_off_g.value() / 100.0,
            "offset2": self.sld_off_b.value() / 100.0,
            "sharpening_enable": self.chk_sharp_en.isChecked(),
            "sharpening_amount": float(self.sld_sharp_amt.value()),
            "sharpening_threshold": float(self.sld_sharp_thr.value()),
            "isp_clock_speed": self.combo_isp_clock.currentText(),
        }
        for key, spin in self._ct_spins.items():
            out[key] = spin.value()
        return out

    def _on_change(self, *_args):
        self._refresh_labels()
        self._apply_timer.start()

    def _apply_to_worker(self):
        w = self.main_window.camera_worker
        if not w:
            return
        w.set_isp_preview_adjustments(
            tone=self._current_tone(),
            camera=self._current_camera(),
            sdk=self._current_sdk(),
        )
        if hasattr(self.main_window, "schedule_save_config"):
            self.main_window.schedule_save_config()

    def _reset_defaults(self):
        w = self.main_window.camera_worker
        if w:
            w.reset_isp_preview_all()
            self._load_from_worker()
            self.main_window.schedule_save_config()
        else:
            for sld, val in (
                (self.sld_gamma, 100), (self.sld_lift, 0), (self.sld_contrast, 100),
            ):
                sld.blockSignals(True)
                sld.setValue(val)
                sld.blockSignals(False)
            self._apply_to_worker()

    def closeEvent(self, event):
        if self._apply_timer.isActive():
            self._apply_timer.stop()
            self._apply_to_worker()
        super().closeEvent(event)
