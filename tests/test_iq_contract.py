
import numpy as np
import pytest

from app.core.models import IQBlock
from app.core.interfaces import IQSource


def test_iq_block_creation():
    samples = np.array(
        [1 + 1j, 2 + 2j, 3 + 3j],
        dtype=np.complex64
    )

    block = IQBlock(
        samples=samples,
        sample_rate=1000,
        center_frequency=100e6,
        source_id="test"
    )

    assert block.num_samples == 3
    assert block.duration == pytest.approx(0.003)
    assert block.samples.dtype == np.complex64
    assert block.center_frequency == 100e6


def test_invalid_sample_rate():
    samples = np.array([1 + 1j], dtype=np.complex64)

    with pytest.raises(ValueError):
        IQBlock(samples=samples, sample_rate=0)


def test_invalid_samples():
    samples = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        IQBlock(samples=samples, sample_rate=1000)


def test_iq_source_is_abstract():
    with pytest.raises(TypeError):
        IQSource()
