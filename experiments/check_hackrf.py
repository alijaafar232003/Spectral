
import numpy as np

from app.acquisition.hackrf_source import HackRFIQSource
from app.dsp.spectrum import calculate_spectrum


def main():
    source = HackRFIQSource(
        center_frequency=100e6,
        sample_rate=2e6,
    )

    try:
        print("Connecting to HackRF...")
        source.open()

        print("Reading live I/Q samples...")

        # Read enough samples to calculate a spectrum.
        samples = []
        total = 0

        while total < 8192:
            block = source.read(8192 - total)
            samples.append(block.samples)
            total += block.num_samples

        block.samples = np.concatenate(samples)

        result = calculate_spectrum(block)

        peak_index = np.argmax(result.psd_db_per_hz)
        peak_frequency = result.frequencies_hz[peak_index]

        print("HackRF reception successful!")
        print(f"Samples received: {block.num_samples}")
        print(f"Sample rate: {block.sample_rate / 1e6:.2f} MS/s")
        print(f"Center frequency: {block.center_frequency / 1e6:.2f} MHz")
        print(f"Strongest spectral bin: {peak_frequency / 1e6:.4f} MHz")

    finally:
        source.close()


if __name__ == "__main__":
    main()
