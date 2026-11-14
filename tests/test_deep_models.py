"""Shape tests for src/models/deep.py (tiny inputs; no training)."""

import numpy as np
import pytest

from src.models.deep import BUILDERS, to_model_input


def test_input_is_padded_to_whole_weeks_with_missing_days():
    values = np.ones((3, 1035), dtype=np.float32)
    mask = np.zeros((3, 1035), dtype=np.uint8)
    x = to_model_input(values, mask)
    assert x.shape == (3, 1036, 2) and x.dtype == np.float32
    assert x[0, -1, 0] == 0.0 and x[0, -1, 1] == 1.0   # padded day: value 0, marked missing
    assert x[0, 0, 1] == 0.0


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_builders_output_one_probability_per_customer(name):
    model = BUILDERS[name]((28, 2))
    out = model.predict(np.zeros((4, 28, 2), dtype=np.float32), verbose=0)
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()
