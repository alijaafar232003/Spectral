
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.acquisition.hackrf_source import HackRFIQSource
from app.dsp.spectrum import calculate_spectrum


class LiveSpectrumWorker(QThread):
    spectrum_ready = Signal(object)
    connected = Signal()
    error = Signal(str)

    def __init__(self, center_frequency, sample_rate):
        super().__init__()

        self.center_frequency = center_frequency
        self.sample_rate = sample_rate

    def run(self):
        source = HackRFIQSource(
            center_frequency=self.center_frequency,
            sample_rate=self.sample_rate,
        )

        try:
            source.open()
            self.connected.emit()

            last_update = 0.0
            consecutive_timeouts = 0

            while not self.isInterruptionRequested():
                try:
                    block = source.read(32768)
                    consecutive_timeouts = 0

                except TimeoutError:
                    consecutive_timeouts += 1

                    if consecutive_timeouts >= 3:
                        raise RuntimeError(
                            "HackRF stopped delivering samples"
                        )

                    continue

                if block is None or block.num_samples < 2048:
                    continue

                # Limit GUI updates to approximately 8 per second.
                now = time.monotonic()

                if now - last_update < 0.125:
                    continue

                result = calculate_spectrum(
                    block,
                    fft_size=2048,
                )

                peak_index = int(
                    np.argmax(result.psd_db_per_hz)
                )

                self.spectrum_ready.emit({
                    "frequencies": (
                        result.frequencies_hz / 1_000_000
                    ),
                    "power": result.psd_db_per_hz,
                    "peak": (
                        result.frequencies_hz[peak_index]
                        / 1_000_000
                    ),
                })

                last_update = now

        except Exception as exc:
            self.error.emit(str(exc))

        finally:
            try:
                source.close()
            except Exception as exc:
                self.error.emit(
                    f"Error closing HackRF: {exc}"
                )
