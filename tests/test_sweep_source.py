import numpy as np

from app.acquisition.sweep_source import parse_sweep_line


def test_valid_sweep_csv_line():
    line = (
        "2026-10-26, 12:00:00.000000, 88000000, 88500000, "
        "100000, 20, -70.0, -45.0, -65.5, -72.0, -68.0"
    )
    parsed = parse_sweep_line(line)
    assert parsed is not None
    timestamp, freqs, powers = parsed
    assert timestamp[0] == "2026-10-26"
    np.testing.assert_allclose(freqs[:2], [88_050_000, 88_150_000])
    np.testing.assert_allclose(powers[:2], [-70.0, -45.0])


def test_invalid_sweep_csv_line():
    assert parse_sweep_line("not a sweep row") is None
    assert parse_sweep_line("") is None
