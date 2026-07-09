from __future__ import annotations

import unittest

from src.data_loader import load_data


class DataLoaderTests(unittest.TestCase):
    def test_load_data_has_target_column(self) -> None:
        dataframe = load_data(n_samples=20, random_state=42)

        self.assertIn("target", dataframe.columns)
        self.assertEqual(len(dataframe), 20)

    def test_load_data_raises_for_small_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            load_data(n_samples=1)


if __name__ == "__main__":
    unittest.main()
