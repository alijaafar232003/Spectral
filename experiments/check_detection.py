
from pathlib import Path

from app.acquisition.file_source import FileIQSource
from app.dsp.spectrum import calculate_spectrum
from app.dsp.detector import detect_signals


def main():
    recording = Path("demo_iq.cf32")

    if not recording.exists():
        print(
            "Demo recording missing. Run:\n"
            "python experiments\\generate_demo.py"
        )
        return

    source = FileIQSource(
        file_path=recording,
        sample_rate=2_000_000,
        center_frequency=100_000_000,
        sample_format="cf32_le",
    )

    source.open()

    try:
        block = source.read(65536)
    finally:
        source.close()

    spectrum = calculate_spectrum(block)
    detection = detect_signals(spectrum)

    print(
        f"Detected {len(detection.candidates)} "
        "candidate signals\n"
    )

    for index, candidate in enumerate(
        detection.candidates, start=1
    ):
        print(f"Candidate {index}")
        print(
            f"  Peak frequency: "
            f"{candidate.peak_frequency_hz / 1e6:.4f} MHz"
        )
        print(
            f"  Detected width: "
            f"{candidate.bandwidth_hz / 1000:.2f} kHz"
        )
        print(
            f"  Peak above local background: "
            f"{candidate.peak_excess_db:.1f} dB"
        )
        print()


if __name__ == "__main__":
    main()
