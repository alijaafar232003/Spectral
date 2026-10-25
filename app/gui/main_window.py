
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
from app.dsp.spectrum import calculate_spectrum


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SpectraIQ - Signal Analyzer")
        self.resize(1100, 700)

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # Recording configuration
        controls = QGroupBox("Recording Configuration")
        form = QFormLayout(controls)

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
            "Center frequency is known"
        )
        self.known_center.setChecked(True)
        self.known_center.toggled.connect(
            self.center_frequency.setEnabled
        )

        self.sample_format = QComboBox()
        self.sample_format.addItem(
            "Complex Float32 (cf32_le)", "cf32_le"
        )
        self.sample_format.addItem(
            "HackRF Int8 (ci8)", "ci8"
        )

        form.addRow("Sample Rate:", self.sample_rate)
        form.addRow("Center Frequency:", self.center_frequency)
        form.addRow("", self.known_center)
        form.addRow("Recording Format:", self.sample_format)

        layout.addWidget(controls)

        # File loading
        file_controls = QHBoxLayout()

        self.open_button = QPushButton("Open I/Q Recording")
        self.open_button.clicked.connect(self.open_recording)

        self.file_label = QLabel("No recording loaded")

        file_controls.addWidget(self.open_button)
        file_controls.addWidget(self.file_label, 1)

        layout.addLayout(file_controls)

        # Spectrum graph
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("#101827")
        self.spectrum_plot.setTitle("Frequency Spectrum")

        self.spectrum_plot.setLabel(
            "left", "Power Spectral Density", units="dB/Hz"
        )
        self.spectrum_plot.setLabel(
            "bottom", "Frequency", units="MHz"
        )

        self.spectrum_plot.showGrid(
            x=True, y=True, alpha=0.3
        )

        self.spectrum_plot.setMouseEnabled(x=True, y=True)

        layout.addWidget(self.spectrum_plot, 1)

        self.statusBar().showMessage("SpectraIQ Ready")

    def open_recording(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open I/Q Recording",
            "",
            "I/Q Recordings (*.raw *.iq *.cf32 *.sigmf-data);;"
            "All Files (*)",
        )

        if not file_path:
            return

        try:
            self.analyze_recording(file_path)

        except Exception as error:
            QMessageBox.critical(
                self,
                "Recording Error",
                str(error),
            )

    def analyze_recording(self, file_path):
        sample_rate_hz = self.sample_rate.value() * 1_000_000

        center_hz = (
            self.center_frequency.value() * 1_000_000
            if self.known_center.isChecked()
            else None
        )

        source = FileIQSource(
            file_path=file_path,
            sample_rate=sample_rate_hz,
            center_frequency=center_hz,
            sample_format=self.sample_format.currentData(),
        )

        source.open()

        try:
            # Read a limited sample window, not the entire file.
            block = source.read(65536)
        finally:
            source.close()

        if block is None or block.num_samples < 2048:
            raise ValueError(
                "Recording must contain at least 2048 I/Q samples."
            )

        result = calculate_spectrum(
            block,
            fft_size=2048,
        )

        # Display frequencies in MHz or relative kHz.
        if result.is_absolute_frequency:
            x_values = result.frequencies_hz / 1_000_000
            x_unit = "MHz"
        else:
            x_values = result.frequencies_hz / 1000
            x_unit = "kHz"

        self.spectrum_plot.clear()

        self.spectrum_plot.setLabel(
            "bottom", "Frequency", units=x_unit
        )

        self.spectrum_plot.plot(
            x_values,
            result.psd_db_per_hz,
            pen=pg.mkPen("#38BDF8", width=1.5),
        )

        self.spectrum_plot.enableAutoRange()

        peak_index = int(np.argmax(result.psd_db_per_hz))
        peak_frequency = x_values[peak_index]

        self.file_label.setText(Path(file_path).name)

        self.statusBar().showMessage(
            f"Loaded {block.num_samples:,} samples | "
            f"Duration: {block.duration:.4f} s | "
            f"Strongest spectral bin: "
            f"{peak_frequency:.4f} {x_unit}"
        )
