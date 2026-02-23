import numpy as np
import pytest

from fedbiomed.common.analytics.accumulators._image_nd import ImageNDAccumulator
from fedbiomed.common.constants import FedbiomedError


def test_image_nd_accumulator_returns_scalars():
    acc = ImageNDAccumulator(
        {"stats": {"min": {}, "max": {}, "mean": {}, "std": {}, "count": {}}}
    )

    img1 = np.ones((4, 5, 3), dtype=np.float32)
    img2 = np.full((2, 2, 3), 2.0, dtype=np.float32)

    acc.update(img1)
    acc.update(img2)
    out = acc.finalize()

    assert isinstance(out["count"], int)
    assert isinstance(out["min"], float)
    assert isinstance(out["max"], float)
    assert isinstance(out["mean"], float)
    assert isinstance(out["std"], float)

    assert out["min"] == 1.0
    assert out["max"] == 2.0
    assert out["count"] == (4 * 5 * 3) + (2 * 2 * 3)


def test_image_nd_accumulator_ignores_aggregate_channels_for_global_scalars():
    acc = ImageNDAccumulator(
        {
            "stats": {
                "mean": {"aggregate_channels": True},
                "std": {"aggregate_channels": True},
                "count": {"aggregate_channels": True},
            }
        }
    )

    img = np.ones((4, 5, 3), dtype=np.float32)
    acc.update(img)

    out = acc.finalize()
    assert out["mean"] == 1.0
    assert out["std"] == 0.0
    assert out["count"] == 4 * 5 * 3


def test_image_nd_accumulator_rejects_non_finite_values():
    acc = ImageNDAccumulator({"stats": {"mean": {}, "count": {}, "min": {}, "max": {}}})

    img = np.array([[1.0, np.nan], [np.inf, 3.0]], dtype=np.float32)
    with pytest.raises(
        FedbiomedError,
        match="ImageNDAccumulator Error: Image contains non-numeric values",
    ):
        acc.update(img)


def test_image_nd_accumulator_rejects_empty_arrays():
    acc = ImageNDAccumulator({"stats": {"mean": {}}})
    with pytest.raises(
        FedbiomedError,
        match="ImageNDAccumulator Error: Image contains non-numeric values",
    ):
        acc.update(np.array([], dtype=np.float32))


def test_image_nd_accumulator_finalize_raises_if_no_valid_pixels_accumulated():
    acc = ImageNDAccumulator({"stats": {"mean": {}}})
    with pytest.raises(
        FedbiomedError,
        match="ImageNDAccumulator Error: No valid pixel values were accumulated",
    ):
        acc.finalize()
