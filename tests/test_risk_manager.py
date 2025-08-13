import os
import sys

import pytest

# Ensure project root is on sys.path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk.manager import compute_levels


def test_compute_levels_rounding():
    stop, target = compute_levels("long", 10.0, 0.3333, 1.0, 2.0)
    assert stop == 9.67
    assert target == 10.67
    assert round(stop, 2) == stop
    assert round(target, 2) == target

    stop, target = compute_levels("short", 10.0, 0.3333, 1.0, 2.0)
    assert stop == 10.33
    assert target == 9.33
    assert round(stop, 2) == stop
    assert round(target, 2) == target


@pytest.mark.parametrize(
    "direction,entry,atr,atr_mult,reward_mult",
    [
        ("long", 10.0, -1.0, 1.0, 2.0),  # negative ATR
        ("long", 10.0, 1.0, 1.0, 0.0),   # zero reward multiple
        ("long", 10.0, 1.0, 0.0, 2.0),   # zero ATR multiple
        ("long", 10.0, 1.0, 1.0, -1.0),  # negative reward multiple
        ("long", 0.0, 1.0, 1.0, 2.0),    # zero entry
        ("", 10.0, 1.0, 1.0, 2.0),       # invalid direction
    ],
)
def test_compute_levels_invalid_inputs(direction, entry, atr, atr_mult, reward_mult):
    assert compute_levels(direction, entry, atr, atr_mult, reward_mult) == (None, None)
