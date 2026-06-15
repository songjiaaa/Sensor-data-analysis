"""分析算法单元测试。"""

from __future__ import annotations

import unittest

import numpy as np

from sensor_noise.analysis import build_time_mask, compute_stats, run_pipeline


class TestComputeStats(unittest.TestCase):
    def test_known_std(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 2.0, size=1000)
        stats = compute_stats(data)
        self.assertAlmostEqual(stats.mean, 5.0, delta=0.2)
        self.assertAlmostEqual(stats.std, 2.0, delta=0.2)
        self.assertAlmostEqual(stats.rms_noise, 2.0, delta=0.2)
        self.assertEqual(stats.n_samples, 1000)


class TestBuildTimeMask(unittest.TestCase):
    def test_rejects_short_window(self) -> None:
        time_s = np.linspace(0, 10, 300)
        with self.assertRaisesRegex(ValueError, "样本不足"):
            build_time_mask(time_s, (0.0, 0.5))

    def test_accepts_valid_window(self) -> None:
        time_s = np.linspace(0, 10, 300)
        mask = build_time_mask(time_s, (0.0, 10.0))
        self.assertEqual(int(np.count_nonzero(mask)), 300)


class TestRunPipeline(unittest.TestCase):
    def test_produces_all_channels(self) -> None:
        fs = 100.0
        n = 500
        time_s = np.arange(n) / fs
        rng = np.random.default_rng(0)
        channels = {
            "ch1": rng.normal(0, 1, n),
            "ch2": rng.normal(0, 0.5, n),
        }
        results = run_pipeline(channels, time_s, fs, (0.0, time_s[-1]))
        self.assertEqual(set(results.keys()), {"ch1", "ch2"})
        for r in results.values():
            self.assertIsNotNone(r.plot)
            self.assertEqual(r.stats.n_samples, n)
            self.assertGreater(len(r.psd), 0)


if __name__ == "__main__":
    unittest.main()
