
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QGroupBox,
)

from app.acquisition.file_source import FileIQSource
from app.acquisition.live_worker import LiveSpectrumWorker
from app.dsp.spectrum import calculate_spectrum


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SpectraIQ - Signal Analyzer")
        self.resize(1150, 750)

        self.live_worker = None

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # Receiver / recording configuration
        config = QGroupBox("Signal Configuration")
        form = QFormLayout(config)

        self.sample_rate = QDoubleSpinBox()
        self.sample_rate.setRange(0.001, 20.0)
        self.sample_rate.setDecimals(3)
        self.sample_rate.setValue(2.0)
        self.sample_rate.setSuffix(" MS/s")

        self.center_frequency = QDoubleSpinBox()
        self.center_frequency.setRange(0, 6000)
        self.center_frequency.setDecimals(4)
        self.center_frequency.setValue(100)
        self.center_frequency.setSuffix(" MHz")

        self.known_center = QCheckBox(
            "Center frequency is known (offline)"
        )
        self.known_center.setChecked(True)
        self.known_center.toggled.connect(
            self.center_frequency.setEnabled
        )

        self.sample_format = QComboBox()
        self.sample_format.addItem(
            "Complex Float32", "cf32_le"
        )
        self.sample_format.addItem(
            "HackRF Int8", "ci8"
        )

        form.addRow("Sample Rate:", self.sample_rate)
        form.addRow("Center Frequency:", self.center_frequency)
        form.addRow("", self.known_center)
        form.addRow("File Format:", self.sample_format)

        layout.addWidget(config)

        # Offline controls
        offline = QGroupBox("Offline Analysis")
        offline_layout = QHBoxLayout(offline)

        self.open_button = QPushButton("Open I/Q Recording")
        self.open_button.clicked.connect(self.open_recording)

        self.file_label = QLabel("No recording loaded")

        offline_layout.addWidget(self.open_button)
        offline_layout.addWidget(self.file_label, 1)

        layout.addWidget(offline)

        # Live HackRF controls
        live = QGroupBox("Live HackRF Reception")
        live_layout = QHBoxLayout(live)

        self.start_button = QPushButton("Start Live")
        self.stop_button = QPushButton("Stop Live")

        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_live)
        self.stop_button.clicked.connect(self.stop_live)

        self.live_status = QLabel("Disconnected")

        live_layout.addWidget(self.start_button)
        live_layout.addWidget(self.stop_button)
        live_layout.addWidget(self.live_status, 1)

        layout.addWidget(live)

        # Interactive spectrum plot
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("#101827")
        self.spectrum_plot.setTitle("Frequency Spectrum")

        self.spectrum_plot.setLabel(
            "left",
            "Power Spectral Density",
            units="dB/Hz",
        )

        self.spectrum_plot.setLabel(
            "bottom", "Frequency", units="MHz"
        )

        self.spectrum_plot.showGrid(
            x=True, y=True, alpha=0.3
        )

        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen("#38BDF8", width=1.5)
        )

        layout.addWidget(self.spectrum_plot, 1)

        self.statusBar().showMessage("SpectraIQ Ready")

    def set_controls_enabled(self, enabled):
        self.sample_rate.setEnabled(enabled)
        self.known_center.setEnabled(enabled)
        self.center_frequency.setEnabled(
            enabled and self.known_center.isChecked()
        )
        self.sample_format.setEnabled(enabled)
        self.open_button.setEnabled(enabled)
        self.start_button.setEnabled(enabled)

    # -------------------------
    # Offline analysis
    # -------------------------

    def open_recording(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open I/Q Recording",
            "",
            "I/Q Files (*.raw *.iq *.cf32 *.sigmf-data);;"
            "All Files (*)",
        )

        if not file_path:
            return

        try:
            self.analyze_recording(file_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Recording Error", str(exc)
            )

    def analyze_recording(self, file_path):
        center_hz = (
            self.center_frequency.value() * 1e6
            if self.known_center.isChecked()
            else None
        )

        source = FileIQSource(
            file_path=file_path,
            sample_rate=self.sample_rate.value() * 1e6,
            center_frequency=center_hz,
            sample_format=self.sample_format.currentData(),
        )

        source.open()

        try:
            block = source.read(65536)
        finally:
            source.close()

        if block is None or block.num_samples < 2048:
            raise ValueError(
                "Recording needs at least 2048 samples"
            )

        result = calculate_spectrum(block)

        if result.is_absolute_frequency:
            frequencies = result.frequencies_hz / 1e6
            unit = "MHz"
        else:
            frequencies = result.frequencies_hz / 1000
            unit = "kHz"

        self.spectrum_plot.setLabel(
            "bottom", "Frequency", units=unit
        )

        self.spectrum_curve.setData(
            frequencies, result.psd_db_per_hz
        )

        self.spectrum_plot.enableAutoRange()

        self.file_label.setText(Path(file_path).name)

        self.statusBar().showMessage(
            f"Offline | {block.num_samples:,} samples | "
            f"Duration: {block.duration:.4f} s"
        )

    # -------------------------
    # Live HackRF reception
    # -------------------------

    def start_live(self):
        if self.live_worker is not None:
            return

        if not self.known_center.isChecked():
            QMessageBox.warning(
                self,
                "Frequency Required",
                "Live reception requires a center frequency.",
            )
            return

        self.live_worker = LiveSpectrumWorker(
            center_frequency=(
                self.center_frequency.value() * 1e6
            ),
            sample_rate=self.sample_rate.value() * 1e6,
        )

        self.live_worker.connected.connect(
            self.on_live_connected
        )
        self.live_worker.spectrum_ready.connect(
            self.update_live_spectrum
        )
        self.live_worker.error.connect(
            self.on_live_error
        )
        self.live_worker.finished.connect(
            self.on_live_finished
        )

        self.set_controls_enabled(False)
        self.stop_button.setEnabled(True)
        self.live_status.setText("Connecting...")

        self.spectrum_plot.setLabel(
            "bottom", "Frequency", units="MHz"
        )

        self.spectrum_curve.setData([], [])

        self.live_worker.start()

    def on_live_connected(self):
        self.live_status.setText("Receiving")
        self.statusBar().showMessage(
            "HackRF connected - Live reception"
        )

    def update_live_spectrum(self, data):
        self.spectrum_curve.setData(
            data["frequencies"],
            data["power"],
        )

        self.statusBar().showMessage(
            f"Live | Strongest spectral bin: "
            f"{data['peak']:.4f} MHz"
        )

    def stop_live(self):
        if self.live_worker is None:
            return

        self.live_status.setText("Stopping...")
        self.stop_button.setEnabled(False)
        self.live_worker.requestInterruption()

    def on_live_error(self, message):
        self.statusBar().showMessage(
            f"HackRF error: {message}"
        )
        QMessageBox.warning(
            self, "HackRF Error", message
        )

    def on_live_finished(self):
        self.live_worker = None

        self.set_controls_enabled(True)
        self.stop_button.setEnabled(False)

        self.live_status.setText("Disconnected")

    def closeEvent(self, event):
        if self.live_worker is not None:
            self.live_worker.requestInterruption()

            if not self.live_worker.wait(2500):
                event.ignore()
                self.live_status.setText(
                    "Waiting for HackRF to stop..."
                )
                return

        super().closeEvent(event)
