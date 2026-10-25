
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import math

import numpy as np


@dataclass #its the block having all the sample/signal data
class IQBlock:
    samples: np.ndarray
    sample_rate: float
    center_frequency: float | None = None
    timestamp: datetime | None = None
    source_id: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Require complex I/Q samples
        if not np.iscomplexobj(self.samples):
            raise ValueError("Samples must contain complex I/Q values")

        # Normalize internal representation
        self.samples = np.asarray(self.samples, dtype=np.complex64)

        if self.samples.ndim != 1 or self.samples.size == 0:
            raise ValueError("Samples must be a non-empty 1D array")

        if not np.all(np.isfinite(self.samples)):
            raise ValueError("Samples must contain finite values")

        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive and finite")

        if self.center_frequency is not None:
            if not math.isfinite(self.center_frequency) or self.center_frequency < 0:
                raise ValueError("Invalid center frequency")

    @property
    def num_samples(self) -> int:
        return self.samples.size

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate
