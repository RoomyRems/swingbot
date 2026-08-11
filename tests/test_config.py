from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from swingbot.config import config_fingerprint, load_config


VALID = """
symbols = ["SPY", "QQQ"]

[data]
feed = "sip"
adjustment = "all"

[risk]
initial_equity = 100000.0
risk_per_trade = 0.005
max_total_risk = 0.03
max_positions = 5
max_position_fraction = 0.20
reward_r = 2.0

[execution]
max_entry_gap_r = 0.25
slippage_bps = 5.0
commission_per_share = 0.0
"""


class ConfigTests(unittest.TestCase):
    def _path(self, directory: str, content: str) -> Path:
        path = Path(directory) / "config.toml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_config_is_strict_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._path(directory, VALID)
            first = load_config(path)
            second = load_config(path)
        self.assertEqual(first, second)
        self.assertEqual(config_fingerprint(first), config_fingerprint(second))

    def test_unknown_setting_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._path(directory, VALID + "\n[risk.typo]\nmagic = 4\n")
            with self.assertRaisesRegex(ValueError, "unknown risk setting"):
                load_config(path)

    def test_duplicate_toml_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._path(directory, 'symbols = ["SPY"]\nsymbols = ["QQQ"]\n')
            with self.assertRaisesRegex(ValueError, "invalid TOML"):
                load_config(path)

    def test_raw_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            content = VALID.replace('adjustment = "all"', 'adjustment = "raw"')
            path = self._path(directory, content)
            with self.assertRaisesRegex(ValueError, "must be 'all'"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
