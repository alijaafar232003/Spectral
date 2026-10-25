
from pathlib import Path

import numpy as np


def main():
    sample_rate = 2_000_000
    num_samples = 65536

    time = np.arange(num_samples) / sample_rate

    # Two test signals relative to a 100 MHz center.
    signal_1 = np.exp(2j * np.pi * 250_000 * time)
    signal_2 = 0.6 * np.exp(
        2j * np.pi * -350_000 * time
    )

    # Add reproducible background noise.
    rng = np.random.default_rng(42)

    noise = 0.03 * (
        rng.standard_normal(num_samples)
        + 1j * rng.standard_normal(num_samples)
    )

    samples = (
        0.5 * signal_1 + 0.5 * signal_2 + noise
    ).astype(np.complex64)

    output = Path("demo_iq.cf32")
    samples.astype("<c8").tofile(output)

    print(f"Created: {output.resolve()}")
    print("Sample rate: 2 MS/s")
    print("Center frequency: 100 MHz")
    print("Expected peaks: 99.65 MHz and 100.25 MHz")


if __name__ == "__main__":
    main()
