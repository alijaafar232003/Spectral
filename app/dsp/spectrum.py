
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from app.core.models import IQBlock

"""
#What does this code do?

It takes the samples from our IQBlock and generates two arrays:
- Frequencies: The frequencies represented in the spectrum, measured in Hz.
- Power values: The estimated power spectral density at each frequency, expressed on a decibel scale.
We're using Welch's method, which calculates and averages spectra from overlapping segments.
This generally gives us a smoother, more stable power estimate than using a single FFT.

The fft_size parameter controls the number of samples used in each FFT. We'll start with 2048,
 which is a practical default for our initial analyzer.

Important: These are uncalibrated power-density estimates, not absolute RF power in dBm.
Accurate dBm measurements require receiver calibration later.
"""

@dataclass
class SpectrumResult:
    frequencies_hz: np.ndarray
    psd_db_per_hz: np.ndarray
    is_absolute_frequency: bool


def calculate_spectrum(
    block: IQBlock,
    fft_size: int = 2048,
) -> SpectrumResult:
    """Calculate a two-sided power spectrum from I/Q samples."""

    if not isinstance(fft_size, int) or fft_size < 8:
        raise ValueError("fft_size must be an integer >= 8")

    if block.num_samples < fft_size:
        raise ValueError(
            f"At least {fft_size} I/Q samples are required"
        )

    # Estimate the power spectral density using Welch's method.
    frequencies, psd = welch(
        block.samples,
        fs=block.sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size // 2,
        return_onesided=False,
        scaling="density",
        detrend=False,
    )

    # Arrange negative frequencies before positive frequencies.
    frequencies = np.fft.fftshift(frequencies)
    psd = np.fft.fftshift(psd)

    # Convert power values into decibels.
    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))

    # Convert baseband frequencies to actual RF frequencies
    # when the recording's center frequency is known.
    is_absolute = block.center_frequency is not None

    if is_absolute:
        frequencies = frequencies + block.center_frequency

    return SpectrumResult(
        frequencies_hz=frequencies,
        psd_db_per_hz=psd_db,
        is_absolute_frequency=is_absolute,
    )
