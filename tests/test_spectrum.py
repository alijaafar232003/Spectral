
import numpy as np
import pytest

from app.core.models import IQBlock
from app.acquisition.file_source import FileIQSource
from app.dsp.spectrum import calculate_spectrum


"""
What are we testing?
The tests check five things:
1. Known frequency: A simulated signal at 100.125 MHz produces a spectrum peak at approximately 100.125 MHz.
2. Negative frequency offset: The analyzer correctly handles complex signals below the center frequency.
3. Spectrum output: The generated arrays contain the expected number of valid values.
4. Invalid settings: Incorrect FFT sizes are rejected.
5. Complete offline pipeline: A signal can be saved as a recording, imported using Step 4's FileIQSource, and analyzed successfully.
"""

def generate_tone(frequency, sample_rate, count=8192):
    """Create a known complex sinusoidal test signal."""
    time = np.arange(count) / sample_rate

    return np.exp(
        2j * np.pi * frequency * time
    ).astype(np.complex64)


def test_detect_known_frequency():
    samples = generate_tone(125_000, 1_000_000)

    block = IQBlock(
        samples=samples,
        sample_rate=1_000_000,
        center_frequency=100_000_000,
    )

    result = calculate_spectrum(block)

    peak_index = np.argmax(result.psd_db_per_hz)
    peak_frequency = result.frequencies_hz[peak_index]

    assert peak_frequency == pytest.approx(
        100_125_000,
        abs=500,
    )
    assert result.is_absolute_frequency


def test_negative_baseband_frequency():
    samples = generate_tone(-50_000, 1_000_000)

    block = IQBlock(
        samples=samples,
        sample_rate=1_000_000,
    )

    result = calculate_spectrum(block)

    peak_index = np.argmax(result.psd_db_per_hz)
    peak_frequency = result.frequencies_hz[peak_index]

    assert peak_frequency == pytest.approx(
        -50_000,
        abs=500,
    )
    assert not result.is_absolute_frequency


def test_spectrum_output():
    samples = generate_tone(100_000, 1_000_000)

    block = IQBlock(
        samples=samples,
        sample_rate=1_000_000,
    )

    result = calculate_spectrum(block, fft_size=1024)

    assert len(result.frequencies_hz) == 1024
    assert len(result.psd_db_per_hz) == 1024
    assert np.all(np.isfinite(result.psd_db_per_hz))


def test_invalid_fft_size():
    samples = generate_tone(100_000, 1_000_000)

    block = IQBlock(
        samples=samples,
        sample_rate=1_000_000,
    )

    with pytest.raises(ValueError):
        calculate_spectrum(block, fft_size=0)

    with pytest.raises(ValueError):
        calculate_spectrum(block, fft_size=16384)


def test_recorded_file_pipeline(tmp_path):
    # Generate a signal and save it as a raw recording.
    samples = generate_tone(75_000, 1_000_000)
    path = tmp_path / "test_signal.cf32"
    samples.astype("<c8").tofile(path)

    # Open the recording with our Step 4 importer.
    source = FileIQSource(
        file_path=path,
        sample_rate=1_000_000,
        center_frequency=100_000_000,
    )

    source.open()
    try:
        block = source.read(8192)
    finally:
        source.close()

    # Analyze the imported signal.
    result = calculate_spectrum(block)
    peak = result.frequencies_hz[
        np.argmax(result.psd_db_per_hz)
    ]

    assert peak == pytest.approx(
        100_075_000,
        abs=500,
    )
