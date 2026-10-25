
from abc import ABC, abstractmethod

from app.core.models import IQBlock


class IQSource(ABC): ##A common set of rules that all signal sources must follow, whether the data comes from HackRF, a recording, or a simulated signal.
    """Common interface for all I/Q data sources."""

    @abstractmethod
    def open(self) -> None:
        """Initialize the source."""
        pass

    @abstractmethod
    def read(self, num_samples: int) -> IQBlock | None:
        """Read up to num_samples complex I/Q samples.

        Return None when no more data is available.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""
        pass
