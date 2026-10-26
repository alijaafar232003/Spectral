
import math


# Name: (start MHz, end MHz, recommended antenna)
BAND_PRESETS = {
    "FM Broadcast": (
        88, 108, "VHF/FM antenna"
    ),
    "UHF 800": (
        800, 900, "Antenna covering 800–900 MHz"
    ),
    "900 MHz": (
        900, 960, "Antenna covering 900–960 MHz"
    ),
    "Aviation Navigation": (
        960, 1215, "Antenna covering 960–1215 MHz"
    ),
    "L-Band": (
        1215, 1300, "Antenna covering 1215–1300 MHz"
    ),
    "2.4 GHz ISM": (
        2400, 2483.5, "2.4 GHz antenna"
    ),
}


def validate_band(start_mhz, end_mhz):
    """Validate a range against HackRF One tuning limits."""

    if not (
        math.isfinite(start_mhz)
        and math.isfinite(end_mhz)
    ):
        raise ValueError("Frequency values must be finite")

    if not (1 <= start_mhz < end_mhz <= 6000):
        raise ValueError(
            "Choose a valid range from 1 to 6000 MHz"
        )


def band_center(start_mhz, end_mhz):
    """Calculate the center of a selected band."""

    validate_band(start_mhz, end_mhz)
    return (start_mhz + end_mhz) / 2
