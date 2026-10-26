"""Wideband HackRF sweep adapter.

This produces spectrum survey measurements, not I/Q samples.
"""

import csv
import math

import numpy as np
from PySide6.QtCore import QObject, QProcess, QTimer, Signal


def parse_sweep_line(line: str):
    """Parse a hackrf_sweep CSV row into (timestamp, bin centers Hz, power dB)."""
    try:
        parts = next(csv.reader([line], skipinitialspace=True))
        if len(parts) < 7:
            return None

        low_hz = float(parts[2])
        high_hz = float(parts[3])
        bin_width_hz = float(parts[4])
        powers = np.asarray([float(x) for x in parts[6:]], dtype=np.float32)

        if (
            not math.isfinite(low_hz)
            or not math.isfinite(high_hz)
            or not math.isfinite(bin_width_hz)
            or bin_width_hz <= 0
            or high_hz <= low_hz
            or powers.size == 0
        ):
            return None

        frequencies = low_hz + (np.arange(powers.size) + 0.5) * bin_width_hz
        valid = (frequencies < high_hz + bin_width_hz * 0.001) & np.isfinite(powers)
        return (parts[0].strip(), parts[1].strip()), frequencies[valid], powers[valid]
    except (ValueError, TypeError, csv.Error):
        return None


class HackRFSweepController(QObject):
    """Asynchronously run hackrf_sweep and emit partial/full band spectra."""

    spectrum_ready = Signal(object)
    started = Signal()
    finished = Signal()
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.SeparateChannels)
        self.process.started.connect(self._on_started)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._on_error)
        self.process.finished.connect(self._on_finished)

        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._publish)
        self.buffer = ""
        self.stderr_tail = ""
        self.running = False
        self.stopping = False
        self.start_mhz = 0.0
        self.end_mhz = 0.0
        self.bin_width_hz = 100_000
        self.frequencies = np.empty(0)
        self.powers = np.empty(0, dtype=np.float32)
        self.timestamp = None
        self.dirty = False
        self.sweep_count = 0

    def start(self, start_mhz: float, end_mhz: float, bin_width_hz: int = 100_000):
        if self.process.state() != QProcess.NotRunning:
            raise RuntimeError("A scan is already running")
        if not (1 <= start_mhz < end_mhz <= 6000):
            raise ValueError("Choose a valid frequency range")
        if not (2445 <= bin_width_hz <= 5_000_000):
            raise ValueError("Invalid sweep bin width")

        self.start_mhz = start_mhz
        self.end_mhz = end_mhz
        self.bin_width_hz = bin_width_hz
        self.buffer = ""
        self.stderr_tail = ""
        self.timestamp = None
        self.dirty = False
        self.sweep_count = 0
        self.stopping = False

        count = int(math.ceil((end_mhz - start_mhz) * 1e6 / bin_width_hz))
        self.frequencies = (start_mhz * 1e6 + (np.arange(count) + 0.5) * bin_width_hz) / 1e6
        self.powers = np.full(count, np.nan, dtype=np.float32)

        # -n makes rows in one scan share the same timestamp.
        # Read-only operation; do not enable transmitter or antenna bias power.
        args = [
            "-f", f"{start_mhz:g}:{end_mhz:g}",
            "-w", str(bin_width_hz),
            "-n",
        ]
        self.process.start("hackrf_sweep", args)

    def stop(self):
        if self.process.state() == QProcess.NotRunning:
            return
        self.stopping = True
        self.process.terminate()
        QTimer.singleShot(1500, self._force_stop)

    def _force_stop(self):
        if self.stopping and self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _on_started(self):
        self.running = True
        self.timer.start()
        self.started.emit()

    def _read_stderr(self):
        chunk = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self.stderr_tail = (self.stderr_tail + chunk)[-2000:]

    def _read_stdout(self):
        self.buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            parsed = parse_sweep_line(line.strip())
            if parsed is None:
                continue

            timestamp, bin_hz, power = parsed
            if self.timestamp is not None and timestamp != self.timestamp:
                self.sweep_count += 1
                self._publish()
                self.powers.fill(np.nan)
            self.timestamp = timestamp

            indices = np.floor((bin_hz - self.start_mhz * 1e6) / self.bin_width_hz).astype(int)
            mask = (indices >= 0) & (indices < self.powers.size)
            self.powers[indices[mask]] = power[mask]
            self.dirty = True

        if len(self.buffer) > 100_000:
            self.buffer = ""

    def _publish(self):
        if not self.dirty or not np.any(np.isfinite(self.powers)):
            return
        self.dirty = False
        self.spectrum_ready.emit({
            "frequencies": self.frequencies.copy(),
            "power": self.powers.copy(),
            "sweeps": self.sweep_count,
            "coverage": int(np.count_nonzero(np.isfinite(self.powers))),
        })

    def _on_error(self, error):
        if self.stopping:
            return
        if error == QProcess.FailedToStart:
            self.timer.stop()
            self.running = False
            self.error.emit(
                "Cannot start hackrf_sweep. In Radioconda Prompt, check: where hackrf_sweep"
            )
            self.finished.emit()
        else:
            self.error.emit(self.process.errorString())

    def _on_finished(self, exit_code, exit_status):
        self._read_stdout()
        self._read_stderr()
        self._publish()
        self.timer.stop()
        self.running = False
        if not self.stopping and exit_code != 0:
            self.error.emit(
                "hackrf_sweep exited with an error. " + (self.stderr_tail.strip()[-400:] or f"Exit code {exit_code}")
            )
        self.stopping = False
        self.finished.emit()
