
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from app.dsp.spectrum import SpectrumResult


@dataclass
class SignalCandidate:
    start_hz: float
    end_hz: float
    center_hz: float
    bandwidth_hz: float
    peak_frequency_hz: float
    peak_psd_db_per_hz: float
    peak_excess_db: float


@dataclass
class DetectionResult:
    noise_floor_db_per_hz: np.ndarray
    threshold_db_per_hz: np.ndarray
    candidates: list[SignalCandidate]


def detect_signals(
    spectrum: SpectrumResult,
    threshold_db: float = 12.0,
    noise_window_bins: int = 101,
    min_bins: int = 2,
    max_gap_bins: int = 1,
) -> DetectionResult:
    """Detect candidate regions above a local noise floor."""

    frequencies = np.asarray(
        spectrum.frequencies_hz,
        dtype=np.float64,
    )
    power = np.asarray(
        spectrum.psd_db_per_hz,
        dtype=np.float64,
    )

    if (
        frequencies.ndim != 1
        or len(frequencies) < 8
        or power.shape != frequencies.shape
    ):
        raise ValueError("Invalid spectrum dimensions")

    if not (
        np.all(np.isfinite(frequencies))
        and np.all(np.isfinite(power))
    ):
        raise ValueError("Spectrum contains invalid values")

    steps = np.diff(frequencies)

    if (
        np.any(steps <= 0)
        or not np.allclose(
            steps, steps[0], rtol=1e-3, atol=1e-6
        )
    ):
        raise ValueError(
            "Frequency bins must be uniformly spaced"
        )

    if not np.isfinite(threshold_db) or threshold_db <= 0:
        raise ValueError("threshold_db must be positive")

    if noise_window_bins < 3 or noise_window_bins % 2 == 0:
        raise ValueError(
            "noise_window_bins must be odd and >= 3"
        )

    if min_bins < 1 or max_gap_bins < 0:
        raise ValueError("Invalid grouping settings")

    # Estimate the local background using a moving median.
    noise_floor = median_filter(
        power,
        size=noise_window_bins,
        mode="reflect",
    )

    threshold = noise_floor + threshold_db

    # Find frequency bins exceeding the threshold.
    active_bins = np.flatnonzero(power > threshold)

    candidates = []

    if active_bins.size:
        # Group bins, allowing small gaps between them.
        cuts = np.flatnonzero(
            np.diff(active_bins) > max_gap_bins + 1
        ) + 1

        for group in np.split(active_bins, cuts):
            left = int(group[0])
            right = int(group[-1])

            if right - left + 1 < min_bins:
                continue

            peak = left + int(
                np.argmax(power[left:right + 1])
            )

            bin_width = steps[0]

            start = frequencies[left] - bin_width / 2
            end = frequencies[right] + bin_width / 2

            candidates.append(
                SignalCandidate(
                    start_hz=float(start),
                    end_hz=float(end),
                    center_hz=float((start + end) / 2),
                    bandwidth_hz=float(end - start),
                    peak_frequency_hz=float(frequencies[peak]),
                    peak_psd_db_per_hz=float(power[peak]),
                    peak_excess_db=float(
                        power[peak] - noise_floor[peak]
                    ),
                )
            )

    return DetectionResult(
        noise_floor_db_per_hz=noise_floor,
        threshold_db_per_hz=threshold,
        candidates=candidates,
    )
