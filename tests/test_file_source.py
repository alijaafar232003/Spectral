
import numpy as np
import pytest

from app.acquisition.file_source import FileIQSource


def test_float32_recording(tmp_path):
    path = tmp_path / "sample.cf32"

    original = np.array(
        [1 + 2j, 3 + 4j, 5 + 6j],
        dtype=np.complex64,
    )

    original.astype("<c8").tofile(path)

    source = FileIQSource(
        file_path=path,
        sample_rate=1000,
        center_frequency=100e6,
        sample_format="cf32_le",
    )

    source.open()

    block = source.read(2)
    np.testing.assert_allclose(block.samples, original[:2])
    assert block.num_samples == 2
    assert block.sample_rate == 1000

    second = source.read(2)
    np.testing.assert_allclose(second.samples, original[2:])

    assert source.read(2) is None
    source.close()


def test_int8_recording(tmp_path):
    path = tmp_path / "hackrf.raw"

    # Interleaved components: I, Q, I, Q
    raw = np.array([64, -64, 127, 0], dtype=np.int8)
    raw.tofile(path)

    source = FileIQSource(
        path,
        sample_rate=2_000_000,
        sample_format="ci8",
    )

    source.open()
    block = source.read(10)
    source.close()

    expected = np.array([
        0.5 - 0.5j,
        127 / 128 + 0j,
    ], dtype=np.complex64)

    np.testing.assert_allclose(block.samples, expected)


def test_missing_file(tmp_path):
    source = FileIQSource(
        tmp_path / "missing.raw",
        sample_rate=1000,
    )

    with pytest.raises(FileNotFoundError):
        source.open()


def test_invalid_recording(tmp_path):
    path = tmp_path / "broken.raw"
    path.write_bytes(b"\x01\x02\x03")

    source = FileIQSource(
        path,
        sample_rate=1000,
        sample_format="ci8",
    )

    with pytest.raises(ValueError):
        source.open()


def test_read_before_open(tmp_path):
    source = FileIQSource(
        tmp_path / "sample.raw",
        sample_rate=1000,
    )

    with pytest.raises(RuntimeError):
        source.read(100)


def test_invalid_sample_count(tmp_path):
    path = tmp_path / "sample.raw"
    np.array([1, 2], dtype=np.int8).tofile(path)

    source = FileIQSource(
        path,
        sample_rate=1000,
        sample_format="ci8",
    )

    source.open()

    with pytest.raises(ValueError):
        source.read(0)

    source.close()
