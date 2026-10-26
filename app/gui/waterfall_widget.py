
import numpy as np
import pyqtgraph as pg


class WaterfallWidget(pg.PlotWidget):
    """Display recent frequency spectra as a scrolling waterfall."""

    def __init__(self, history_rows=180):
        super().__init__()

        self.history_rows = history_rows
        self.history = None

        self.setBackground("#101827")
        self.setTitle("Live Waterfall")

        self.setLabel(
            "bottom", "Frequency", units="MHz"
        )
        self.setLabel(
            "left", "History", units="frames"
        )

        self.setMouseEnabled(x=True, y=False)
        self.setYRange(0, history_rows, padding=0)

        # Display each spectrum as one horizontal image row.
        self.image = pg.ImageItem(axisOrder="row-major")

        # Dark blue/purple for weak signals,
        # green/yellow for stronger signals.
        colors = pg.colormap.get("viridis")
        self.image.setLookupTable(
            colors.getLookupTable(nPts=256)
        )

        self.addItem(self.image)
        self.image.hide()

        self.levels = None

    def reset(self):
        """Clear the waterfall when starting another session."""
        self.history = None
        self.levels = None
        self.image.hide()

    def add_spectrum(self, frequencies, power):
        frequencies = np.asarray(frequencies)
        power = np.asarray(power, dtype=np.float32)

        if len(frequencies) < 2 or len(power) != len(frequencies):
            return

        # Initialize on the first incoming spectrum.
        if self.history is None or self.history.shape[1] != len(power):
            low = float(np.percentile(power, 15))
            high = float(np.percentile(power, 99.5))

            self.levels = (low, max(high, low + 15))

            self.history = np.full(
                (self.history_rows, len(power)),
                low,
                dtype=np.float32,
            )

            # Align the image with the actual frequency axis.
            frequency_step = frequencies[1] - frequencies[0]

            start = frequencies[0] - frequency_step / 2
            width = frequency_step * len(frequencies)

            self.image.setImage(
                self.history,
                autoLevels=False,
                levels=self.levels,
            )

            self.image.setRect(
                float(start),
                0,
                float(width),
                self.history_rows,
            )

            self.setXRange(
                float(start),
                float(start + width),
                padding=0,
            )

            self.image.show()

        # Shift previous spectra upward.
        self.history[:-1] = self.history[1:].copy()

        # Add the newest spectrum at the bottom.
        self.history[-1] = power

        # Refresh the waterfall image.
        self.image.setImage(
            self.history,
            autoLevels=False,
            levels=self.levels,
        )
