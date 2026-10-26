from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.acquisition.file_source import FileIQSource
from app.acquisition.live_worker import LiveSpectrumWorker
from app.acquisition.sweep_source import HackRFSweepController
from app.core.bands import BAND_PRESETS, band_center, validate_band
from app.dsp.spectrum import calculate_spectrum
from app.gui.waterfall_widget import WaterfallWidget


STYLE = """
QMainWindow, QWidget { background: #0b1220; color: #e6edf7; font-size: 13px; }
QLabel#AppTitle { font-size: 23px; font-weight: 700; color: #f8fafc; }
QLabel#Subtitle { color: #94a3b8; font-size: 12px; }
QLabel#SectionTitle { font-size: 15px; font-weight: 600; }
QLabel#Hint { color: #94a3b8; font-size: 12px; }
QLabel#Connection { color: #67e8f9; font-weight: 600; }
QGroupBox {
    border: 1px solid #273449; border-radius: 9px;
    margin-top: 14px; padding: 15px 10px 10px;
    font-weight: 600; background: #111c2d;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; top: 0px;
    padding: 0 5px; color: #e2e8f0;
}
QComboBox, QDoubleSpinBox {
    background: #17263b; color: #f1f5f9;
    border: 1px solid #34445a; border-radius: 6px;
    min-height: 30px; padding: 3px 8px;
}
QComboBox:hover, QDoubleSpinBox:hover { border-color: #579bdb; }
QComboBox:disabled, QDoubleSpinBox:disabled { color: #74839a; }
QComboBox QAbstractItemView {
    background: #17263b; color: #f1f5f9;
    selection-background-color: #2763a3;
}
QPushButton {
    background: #22354d; color: #f8fafc;
    border: 1px solid #36516d; border-radius: 7px;
    min-height: 34px; padding: 4px 12px; font-weight: 600;
}
QPushButton:hover { background: #2f4865; }
QPushButton:disabled { background: #172234; color: #68788e; border-color: #263348; }
QPushButton#PrimaryButton { background: #1973bd; border-color: #278bd8; }
QPushButton#PrimaryButton:hover { background: #2489da; }
QScrollArea { border: none; background: #101a2a; }
QScrollArea > QWidget > QWidget { background: #101a2a; }
QFrame#Header { background: #101a2a; border-bottom: 1px solid #263449; }
QSplitter::handle { background: #25354c; }
QStatusBar { color: #a7b8cf; background: #101a2a; }
QCheckBox { spacing: 7px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.live_worker = None
        self.sweep_controller = HackRFSweepController(self)
        self.sweep_controller.started.connect(self.on_sweep_started)
        self.sweep_controller.spectrum_ready.connect(self.update_sweep_spectrum)
        self.sweep_controller.error.connect(self.on_sweep_error)
        self.sweep_controller.finished.connect(self.on_sweep_finished)
        self.setWindowTitle("SpectraIQ — Signal Analyzer")
        self.resize(1350, 850)
        self.setStyleSheet(STYLE)
        self.setup_ui()

    def setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)

        title_column = QVBoxLayout()
        title = QLabel("SpectraIQ")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Wireless Signal Analysis Platform")
        subtitle.setObjectName("Subtitle")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header_layout.addLayout(title_column)
        header_layout.addStretch()

        self.connection_label = QLabel("● Not receiving")
        self.connection_label.setObjectName("Connection")
        header_layout.addWidget(self.connection_label)
        root_layout.addWidget(header)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(14, 18, 14, 18)
        sidebar_layout.setSpacing(12)

        mode_title = QLabel("Operating mode")
        mode_title.setObjectName("SectionTitle")
        sidebar_layout.addWidget(mode_title)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "Monitor Frequency",
            "Scan Frequency Band",
            "Analyze Recording",
        ])
        sidebar_layout.addWidget(self.mode_selector)

        self.mode_explanation = QLabel()
        self.mode_explanation.setObjectName("Hint")
        self.mode_explanation.setWordWrap(True)
        sidebar_layout.addWidget(self.mode_explanation)

        self.pages = QStackedWidget()
        self.pages.addWidget(self.make_monitor_page())
        self.pages.addWidget(self.make_band_page())
        self.pages.addWidget(self.make_offline_page())
        sidebar_layout.addWidget(self.pages)
        sidebar_layout.addStretch()

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setMinimumWidth(295)

        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(12, 15, 15, 15)
        plot_layout.setSpacing(10)

        self.plot_title = QLabel("Frequency Spectrum")
        self.plot_title.setObjectName("SectionTitle")
        plot_layout.addWidget(self.plot_title)

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("#0e1728")
        self.spectrum_plot.setLabel("left", "PSD", units="dB/Hz")
        self.spectrum_plot.setLabel("bottom", "Frequency", units="MHz")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen("#45b7f5", width=1.7)
        )

        self.waterfall_plot = WaterfallWidget(history_rows=180)
        self.waterfall_plot.setMinimumHeight(180)
        self.spectrum_plot.setMinimumHeight(180)

        self.graph_splitter = QSplitter(Qt.Vertical)
        self.graph_splitter.setChildrenCollapsible(False)
        self.graph_splitter.addWidget(self.spectrum_plot)
        self.graph_splitter.addWidget(self.waterfall_plot)
        self.graph_splitter.setSizes([340, 340])
        plot_layout.addWidget(self.graph_splitter, 1)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(sidebar_scroll)
        self.main_splitter.addWidget(plot_area)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([345, 1000])
        root_layout.addWidget(self.main_splitter, 1)

        self.statusBar().showMessage("Ready — choose a mode to begin")
        self.mode_selector.currentIndexChanged.connect(self.change_mode)
        self.change_mode(0)

    def make_monitor_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 5, 0, 0)
        layout.setSpacing(12)

        settings = QGroupBox("Receiver Settings")
        form = QFormLayout(settings)

        self.monitor_frequency = QDoubleSpinBox()
        self.monitor_frequency.setRange(1, 6000)
        self.monitor_frequency.setDecimals(4)
        self.monitor_frequency.setValue(98.0)
        self.monitor_frequency.setSuffix(" MHz")

        self.monitor_rate = QDoubleSpinBox()
        self.monitor_rate.setRange(2, 20)
        self.monitor_rate.setDecimals(2)
        self.monitor_rate.setValue(2.0)
        self.monitor_rate.setSuffix(" MS/s")

        form.addRow("Center (Fc):", self.monitor_frequency)
        form.addRow("Sample rate (Fs):", self.monitor_rate)
        layout.addWidget(settings)

        self.monitor_preview = QLabel()
        self.monitor_preview.setObjectName("Hint")
        self.monitor_preview.setWordWrap(True)
        layout.addWidget(self.monitor_preview)
        self.monitor_frequency.valueChanged.connect(self.update_monitor_preview)
        self.monitor_rate.valueChanged.connect(self.update_monitor_preview)
        self.update_monitor_preview()

        actions = QHBoxLayout()
        self.start_button = QPushButton("Start Monitoring")
        self.start_button.setObjectName("PrimaryButton")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_live)
        self.stop_button.clicked.connect(self.stop_live)
        actions.addWidget(self.start_button, 1)
        actions.addWidget(self.stop_button)
        layout.addLayout(actions)

        layout.addStretch()
        return page

    def make_band_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 5, 0, 0)
        layout.setSpacing(12)

        band_settings = QGroupBox("Band Selection")
        form = QFormLayout(band_settings)

        self.band_selector = QComboBox()
        self.band_selector.addItems([*BAND_PRESETS.keys(), "Custom"])
        self.band_start = QDoubleSpinBox()
        self.band_end = QDoubleSpinBox()

        for field in (self.band_start, self.band_end):
            field.setRange(1, 6000)
            field.setDecimals(3)
            field.setSuffix(" MHz")

        form.addRow("Preset:", self.band_selector)
        form.addRow("Start:", self.band_start)
        form.addRow("End:", self.band_end)
        layout.addWidget(band_settings)

        self.antenna_label = QLabel()
        self.antenna_label.setObjectName("Hint")
        self.antenna_label.setWordWrap(True)
        layout.addWidget(self.antenna_label)

        self.band_preview = QLabel()
        self.band_preview.setObjectName("Hint")
        self.band_preview.setWordWrap(True)
        layout.addWidget(self.band_preview)

        self.midpoint_button = QPushButton("Monitor Band Midpoint Instead")
        self.midpoint_button.clicked.connect(self.monitor_band_midpoint)
        layout.addWidget(self.midpoint_button)

        self.scan_button = QPushButton("Start Band Scan")
        self.scan_button.setObjectName("PrimaryButton")
        self.scan_button.clicked.connect(self.start_band_scan)
        layout.addWidget(self.scan_button)

        self.stop_scan_button = QPushButton("Stop Scan")
        self.stop_scan_button.setEnabled(False)
        self.stop_scan_button.clicked.connect(self.stop_band_scan)
        layout.addWidget(self.stop_scan_button)

        note = QLabel(
            "Band Scan surveys the selected range with hackrf_sweep. "
            "These are relative spectrum-power readings, not I/Q recordings. "
            "Stop scanning before switching to Monitor Frequency."
        )
        note.setObjectName("Hint")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.band_selector.currentTextChanged.connect(self.on_band_changed)
        self.band_start.valueChanged.connect(self.update_band_preview)
        self.band_end.valueChanged.connect(self.update_band_preview)
        self.on_band_changed(self.band_selector.currentText())
        layout.addStretch()
        return page

    def make_offline_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 5, 0, 0)
        layout.setSpacing(12)

        settings = QGroupBox("Recording Metadata")
        form = QFormLayout(settings)

        self.offline_rate = QDoubleSpinBox()
        self.offline_rate.setRange(0.001, 1000)
        self.offline_rate.setDecimals(3)
        self.offline_rate.setValue(2.0)
        self.offline_rate.setSuffix(" MS/s")

        self.offline_center_known = QCheckBox("Center frequency known")
        self.offline_center_known.setChecked(True)
        self.offline_frequency = QDoubleSpinBox()
        self.offline_frequency.setRange(0, 6000)
        self.offline_frequency.setDecimals(4)
        self.offline_frequency.setValue(100.0)
        self.offline_frequency.setSuffix(" MHz")
        self.offline_center_known.toggled.connect(
            self.offline_frequency.setEnabled
        )

        self.sample_format = QComboBox()
        self.sample_format.addItem("Complex Float32", "cf32_le")
        self.sample_format.addItem("HackRF Int8", "ci8")

        form.addRow("Sample rate:", self.offline_rate)
        form.addRow("", self.offline_center_known)
        form.addRow("Center (Fc):", self.offline_frequency)
        form.addRow("File format:", self.sample_format)
        layout.addWidget(settings)

        hint = QLabel(
            "Enter the values used when the file was recorded. "
            "Changing them here does not change the original recording."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.open_button = QPushButton("Open I/Q Recording")
        self.open_button.setObjectName("PrimaryButton")
        self.open_button.clicked.connect(self.open_recording)
        layout.addWidget(self.open_button)

        self.file_label = QLabel("No recording loaded")
        self.file_label.setObjectName("Hint")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        layout.addStretch()
        return page

    def change_mode(self, index):
        self.pages.setCurrentIndex(index)
        self.spectrum_curve.setData([], [])
        self.waterfall_plot.reset()
        self.waterfall_plot.setVisible(index == 0)
        self.spectrum_plot.setLabel("bottom", "Frequency", units="MHz")
        self.spectrum_plot.setLabel("left", "PSD", units="dB/Hz")

        descriptions = [
            "Continuously observe one RF window using HackRF.",
            "Survey your selected band using HackRF frequency sweeps.",
            "Open previously captured I/Q samples for offline analysis.",
        ]
        self.mode_explanation.setText(descriptions[index])
        self.plot_title.setText(
            "Live Spectrum" if index == 0 else "Frequency Spectrum"
        )
        self.statusBar().showMessage(descriptions[index])

    def update_monitor_preview(self, *_):
        center = self.monitor_frequency.value()
        half = self.monitor_rate.value() / 2
        self.monitor_preview.setText(
            f"Nominal window: {center - half:.3f}–{center + half:.3f} MHz. "
            "Actual usable bandwidth may be smaller."
        )

    def on_band_changed(self, name):
        custom = name == "Custom"
        self.band_start.setEnabled(custom)
        self.band_end.setEnabled(custom)

        if not custom:
            start, end, antenna = BAND_PRESETS[name]
            self.band_start.setValue(start)
            self.band_end.setValue(end)
            self.antenna_label.setText(f"Suggested antenna: {antenna}")
        else:
            self.antenna_label.setText(
                "Suggested antenna: one covering your custom range."
            )
        self.update_band_preview()

    def update_band_preview(self, *_):
        start = self.band_start.value()
        end = self.band_end.value()
        try:
            validate_band(start, end)
            center = band_center(start, end)
            self.band_preview.setText(
                f"Selected range: {start:g}–{end:g} MHz\n"
                f"Midpoint: {center:.3f} MHz\n"
                "A scan would survey the range in steps; it cannot "
                "all be captured simultaneously as one I/Q stream."
            )
            self.midpoint_button.setEnabled(True)
            self.scan_button.setEnabled(True)
        except ValueError as exc:
            self.band_preview.setText(str(exc))
            self.midpoint_button.setEnabled(False)
            self.scan_button.setEnabled(False)

    def monitor_band_midpoint(self):
        try:
            center = band_center(
                self.band_start.value(), self.band_end.value()
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Band", str(exc))
            return

        self.monitor_frequency.setValue(center)
        self.mode_selector.setCurrentIndex(0)
        self.statusBar().showMessage(
            f"Monitoring midpoint {center:.3f} MHz — not scanning the band"
        )

    def start_band_scan(self):
        if self.live_worker is not None or self.sweep_controller.running:
            return
        try:
            start = self.band_start.value()
            end = self.band_end.value()
            validate_band(start, end)
            self.sweep_controller.start(start, end, bin_width_hz=100_000)
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Band Scan Error", str(exc))
            return

        self.mode_selector.setEnabled(False)
        self.band_selector.setEnabled(False)
        self.band_start.setEnabled(False)
        self.band_end.setEnabled(False)
        self.midpoint_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.stop_scan_button.setEnabled(True)
        self.connection_label.setText("● Starting scan...")
        self.plot_title.setText(f"Band Scan — {start:g} to {end:g} MHz")
        self.spectrum_plot.setLabel("left", "Sweep power", units="dB")
        self.spectrum_plot.setLabel("bottom", "Frequency", units="MHz")
        self.spectrum_curve.setData([], [])
        self.spectrum_plot.setXRange(start, end, padding=0)
        self.waterfall_plot.setVisible(False)

    def stop_band_scan(self):
        self.stop_scan_button.setEnabled(False)
        self.connection_label.setText("● Stopping scan...")
        self.sweep_controller.stop()

    def on_sweep_started(self):
        self.connection_label.setText("● Scanning band")
        self.statusBar().showMessage("Scanning selected band with HackRF...")

    def update_sweep_spectrum(self, data):
        if self.sweep_controller.stopping:
            return
        self.spectrum_curve.setData(
            data["frequencies"], data["power"], connect="finite"
        )
        self.statusBar().showMessage(
            f"Band scan | {data['sweeps']} completed sweeps | "
            f"{data['coverage']} frequency bins observed"
        )

    def on_sweep_error(self, message):
        self.statusBar().showMessage(f"Band scan error: {message}")
        QMessageBox.warning(self, "Band Scan Error", message)

    def on_sweep_finished(self):
        self.mode_selector.setEnabled(True)
        self.band_selector.setEnabled(True)
        self.on_band_changed(self.band_selector.currentText())
        self.stop_scan_button.setEnabled(False)
        self.connection_label.setText("● Not receiving")
        self.statusBar().showMessage("Band scanning stopped")

    def open_recording(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open I/Q Recording",
            "",
            "I/Q Files (*.raw *.iq *.cf32 *.sigmf-data);;All Files (*)",
        )
        if not path:
            return
        try:
            self.analyze_recording(path)
        except Exception as exc:
            QMessageBox.critical(self, "Recording Error", str(exc))

    def analyze_recording(self, path):
        center = (
            self.offline_frequency.value() * 1e6
            if self.offline_center_known.isChecked()
            else None
        )
        source = FileIQSource(
            file_path=path,
            sample_rate=self.offline_rate.value() * 1e6,
            center_frequency=center,
            sample_format=self.sample_format.currentData(),
        )

        source.open()
        try:
            block = source.read(65536)
        finally:
            source.close()

        if block is None or block.num_samples < 2048:
            raise ValueError("Recording must contain at least 2048 I/Q samples")

        spectrum = calculate_spectrum(block, fft_size=2048)
        if spectrum.is_absolute_frequency:
            frequencies = spectrum.frequencies_hz / 1e6
            unit = "MHz"
        else:
            frequencies = spectrum.frequencies_hz / 1e3
            unit = "kHz"

        self.spectrum_plot.setLabel("bottom", "Frequency", units=unit)
        self.spectrum_curve.setData(frequencies, spectrum.psd_db_per_hz)
        self.spectrum_plot.enableAutoRange()
        self.file_label.setText(Path(path).name)
        self.statusBar().showMessage(
            f"Offline | {block.num_samples:,} samples | "
            f"Duration: {block.duration:.4f} s"
        )

    def start_live(self):
        if self.live_worker is not None:
            return

        self.live_worker = LiveSpectrumWorker(
            center_frequency=self.monitor_frequency.value() * 1e6,
            sample_rate=self.monitor_rate.value() * 1e6,
        )
        self.live_worker.connected.connect(self.on_live_connected)
        self.live_worker.spectrum_ready.connect(self.update_live_spectrum)
        self.live_worker.error.connect(self.on_live_error)
        self.live_worker.finished.connect(self.on_live_finished)

        self.mode_selector.setEnabled(False)
        self.monitor_frequency.setEnabled(False)
        self.monitor_rate.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.connection_label.setText("● Connecting...")
        self.spectrum_curve.setData([], [])
        self.waterfall_plot.reset()
        self.spectrum_plot.setLabel("bottom", "Frequency", units="MHz")
        self.live_worker.start()

    def on_live_connected(self):
        if self.live_worker is not None and not self.live_worker.isInterruptionRequested():
            self.connection_label.setText("● Receiving live")
            self.statusBar().showMessage("HackRF connected — receiving I/Q samples")

    def update_live_spectrum(self, data):
        if self.live_worker is None or self.live_worker.isInterruptionRequested():
            return
        self.spectrum_curve.setData(data["frequencies"], data["power"])
        self.waterfall_plot.add_spectrum(data["frequencies"], data["power"])
        self.statusBar().showMessage(
            f"Live | Strongest spectral bin: {data['peak']:.4f} MHz"
        )

    def stop_live(self):
        if self.live_worker is None:
            return
        self.connection_label.setText("● Stopping...")
        self.stop_button.setEnabled(False)
        self.live_worker.requestInterruption()

    def on_live_error(self, message):
        self.statusBar().showMessage(f"HackRF error: {message}")
        QMessageBox.warning(self, "HackRF Error", message)

    def on_live_finished(self):
        self.live_worker = None
        self.mode_selector.setEnabled(True)
        self.monitor_frequency.setEnabled(True)
        self.monitor_rate.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.connection_label.setText("● Not receiving")

    def closeEvent(self, event):
        if self.sweep_controller.process.state() != QProcess.NotRunning:
            self.sweep_controller.stop()
            event.ignore()
            self.connection_label.setText("● Waiting for scan to stop...")
            self.sweep_controller.finished.connect(self.close)
            return
        if self.live_worker is not None:
            self.live_worker.requestInterruption()
            if not self.live_worker.wait(3000):
                event.ignore()
                self.connection_label.setText("● Waiting for HackRF to stop...")
                return
        super().closeEvent(event)
