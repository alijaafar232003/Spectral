
import numpy as np
import pytest

from app.core.models import IQBlock
from app.dsp.spectrum import (
    SpectrumResult,
    calculate_spectrum,
)
from app.dsp.detector import detect_signals


def test_detect_known_signal():
    sample_rate = 1_000_000
    count = 16384
    time = np.arange(count) / sample_rate

    rng = np.random.default_rng(42)

    signal = 0.7 * np.exp(
        2j * np.pi * 125_000 * time
    )

    noise = 0.07 * (
        rng.standard_normal(count)
        + 1j * rng.standard_normal(count)
    )

    block = IQBlock(
        samples=(signal + noise).astype(np.complex64),
        sample_rate=sample_rate,
        center_frequency=100e6,
    )

    spectrum = calculate_spectrum(block)
    result = detect_signals(spectrum)

    assert any(
        abs(c.peak_frequency_hz - 100_125_000) < 1000
        for c in result.candidates
    )


def test_no_signal_above_threshold():
    frequencies = np.arange(2048, dtype=float) * 1000
    power = np.full(2048, -100.0)

    spectrum = SpectrumResult(
        frequencies_hz=frequencies,
        psd_db_per_hz=power,
        is_absolute_frequency=False,
    )

    result = detect_signals(spectrum)

    assert len(result.candidates) == 0


def test_detect_two_candidates():
    frequencies = np.arange(2048, dtype=float) * 1000
    power = np.full(2048, -100.0)

    power[300:306] = -60
    power[900:908] = -65

    spectrum = SpectrumResult(
        frequencies_hz=frequencies,
        psd_db_per_hz=power,
        is_absolute_frequency=False,
    )

    result = detect_signals(spectrum)

    assert len(result.candidates) == 2
    assert result.candidates[0].center_hz < (
        result.candidates[1].center_hz
    )


def test_invalid_threshold():
    frequencies = np.arange(2048, dtype=float)
    power = np.full(2048, -100.0)

    spectrum = SpectrumResult(
        frequencies_hz=frequencies,
        psd_db_per_hz=power,
        is_absolute_frequency=False,
    )

    with pytest.raises(ValueError):
        detect_signals(spectrum, threshold_db=-5)
