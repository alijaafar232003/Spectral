
from pathlib import Path

import numpy as np

from app.core.interfaces import IQSource
from app.core.models import IQBlock


#It performs four operations:
#1. Opens an existing recording.
#2. Reads a requested number of I/Q samples.
#3. Converts the samples into the format SpectraIQ understands.
#4. Closes the recording when finished.

class FileIQSource(IQSource):
    """Read raw I/Q recordings in supported formats."""

    FORMATS = {
        "cf32_le": np.dtype("<f4"),
        "ci8": np.dtype("i1"),
    }

    def __init__(
        self,
        file_path: str | Path,
        sample_rate: float,
        center_frequency: float | None = None,
        sample_format: str = "cf32_le",
    ):
        if sample_format not in self.FORMATS:
            raise ValueError(f"Unsupported sample format: {sample_format}")

        # Validate acquisition information
        IQBlock(
            samples=np.array([0j], dtype=np.complex64),
            sample_rate=sample_rate,
            center_frequency=center_frequency,
        )

        self.file_path = Path(file_path)
        self.sample_rate = sample_rate
        self.center_frequency = center_frequency
        self.sample_format = sample_format
        self._file = None

    def open(self) -> None:
        self.close()

        if not self.file_path.is_file():
            raise FileNotFoundError(self.file_path)

        # Each complex sample contains one I and one Q value.
        bytes_per_sample = self.FORMATS[self.sample_format].itemsize * 2

        file_size = self.file_path.stat().st_size

        if file_size == 0 or file_size % bytes_per_sample != 0:
            raise ValueError("Invalid or incomplete I/Q recording")

        self._file = self.file_path.open("rb")

    def read(self, num_samples: int) -> IQBlock | None:
        if self._file is None:
            raise RuntimeError("Open the source before reading")

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        raw = np.fromfile(
            self._file,
            dtype=self.FORMATS[self.sample_format],
            count=num_samples * 2,
        )

        if raw.size == 0:
            return None

        # Convert signed 8-bit components to approximately [-1, 1).
        scale = 128.0 if self.sample_format == "ci8" else 1.0

        i_samples = raw[0::2].astype(np.float32) / scale
        q_samples = raw[1::2].astype(np.float32) / scale

        samples = np.empty(i_samples.size, dtype=np.complex64)
        samples.real = i_samples
        samples.imag = q_samples

        return IQBlock(
            samples=samples,
            sample_rate=self.sample_rate,
            center_frequency=self.center_frequency,
            source_id=str(self.file_path),
            metadata={
                "source_type": "file",
                "sample_format": self.sample_format,
            },
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
