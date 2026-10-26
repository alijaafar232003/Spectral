
import pytest

from app.core.bands import (
    BAND_PRESETS,
    validate_band,
    band_center,
)


def test_all_presets_are_valid():
    for start, end, antenna in BAND_PRESETS.values():
        validate_band(start, end)
        assert antenna


def test_band_center():
    assert band_center(88, 108) == 98
    assert band_center(900, 960) == 930


def test_invalid_band():
    with pytest.raises(ValueError):
        validate_band(960, 900)

    with pytest.raises(ValueError):
        validate_band(0, 100)

    with pytest.raises(ValueError):
        validate_band(100, 6100)


def test_custom_band():
    validate_band(400, 450)
    assert band_center(400, 450) == 425
