
from datetime import datetime, timezone

import numpy as np

from app.core.interfaces import IQSource
from app.core.models import IQBlock


class HackRFIQSource(IQSource):
    """Live receive-only HackRF source using SoapySDR."""

    def __init__(
        self,
        center_frequency: float = 100e6,
        sample_rate: float = 2e6,
    ):
        # Validate basic acquisition parameters.
        IQBlock(
            samples=np.array([0j], dtype=np.complex64),
            sample_rate=sample_rate,
            center_frequency=center_frequency,
        )

        self.center_frequency = center_frequency
        self.sample_rate = sample_rate

        self.device = None
        self.stream = None
        self.soapy = None

    def open(self) -> None:
        if self.device is not None:
            raise RuntimeError("HackRF source is already open")

        try:
            import SoapySDR
        except ImportError as exc:
            raise RuntimeError(
                "SoapySDR is not available in this Python environment"
            ) from exc

        self.soapy = SoapySDR

        try:
            # Connect specifically to a HackRF device.
            self.device = SoapySDR.Device()

            # Configure receiver channel 0.
            self.device.setSampleRate(
                SoapySDR.SOAPY_SDR_RX,
                0,
                self.sample_rate,
            )

            self.device.setFrequency(
                SoapySDR.SOAPY_SDR_RX,
                0,
                self.center_frequency,
            )

            # Read back the actual configured values.
            self.sample_rate = self.device.getSampleRate(
                SoapySDR.SOAPY_SDR_RX, 0
            )
            self.center_frequency = self.device.getFrequency(
                SoapySDR.SOAPY_SDR_RX, 0
            )

            # Receive complex float32 samples.
            self.stream = self.device.setupStream(
                SoapySDR.SOAPY_SDR_RX,
                SoapySDR.SOAPY_SDR_CF32,
            )

            self.device.activateStream(self.stream)

        except Exception:
            self.close()
            raise

    def read(self, num_samples: int) -> IQBlock | None:
        if self.device is None or self.stream is None:
            raise RuntimeError("HackRF source is not open")

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        buffer = np.empty(num_samples, dtype=np.complex64)

        result = self.device.readStream(
            self.stream,
            [buffer],
            num_samples,
            timeoutUs=1_000_000,
        )

        if result.ret == self.soapy.SOAPY_SDR_TIMEOUT:
            raise TimeoutError("No samples received before timeout")

        if result.ret < 0:
            raise RuntimeError(
                f"HackRF streaming error: {result.ret}"
            )

        if result.ret == 0:
            raise TimeoutError("No samples received")

        return IQBlock(
            samples=buffer[:result.ret].copy(),
            sample_rate=self.sample_rate,
            center_frequency=self.center_frequency,
            timestamp=datetime.now(timezone.utc),
            source_id="hackrf_0",
            metadata={
                "source_type": "live",
                "stream_format": "CF32",
            },
        )

    def close(self) -> None:
        if self.device is not None:
            if self.stream is not None:
                try:
                    self.device.deactivateStream(self.stream)
                finally:
                    self.device.closeStream(self.stream)

        self.stream = None
        self.device = None
