"""分析算法单元测试。"""

from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from sensor_noise.analysis import (
    build_time_mask,
    compute_psd,
    compute_stats,
    estimate_asd,
    run_pipeline,
    stats_table,
)


class TestComputeStats(unittest.TestCase):
    def test_known_std(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 2.0, size=1000)
        result = compute_stats(data)
        self.assertAlmostEqual(result.mean, 5.0, delta=0.2)
        self.assertAlmostEqual(result.std, 2.0, delta=0.2)
        self.assertAlmostEqual(result.rms_noise, 2.0, delta=0.2)
        self.assertEqual(result.n_samples, 1000)

    def test_skew_kurtosis_use_unbiased_estimator(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 1.0, size=500)
        centered = data - np.mean(data)
        result = compute_stats(data)
        self.assertAlmostEqual(result.skewness, float(stats.skew(centered, bias=False)))
        self.assertAlmostEqual(result.kurtosis, float(stats.kurtosis(centered, bias=False)))


class TestPsd(unittest.TestCase):
    def test_detects_tone_near_expected_frequency(self) -> None:
        fs = 1000.0
        n = 8000
        t = np.arange(n) / fs
        tone_hz = 50.0
        centered = 2.0 * np.sin(2 * np.pi * tone_hz * t)
        freqs, psd = compute_psd(centered, fs)
        self.assertGreater(freqs[-1], fs * 0.45)
        self.assertLess(freqs[0], fs * 0.02)
        peak_hz = float(freqs[np.argmax(psd)])
        self.assertAlmostEqual(peak_hz, tone_hz, delta=2.0)

    def test_estimate_asd_uses_geometric_mean(self) -> None:
        psd = np.logspace(0, 3, 20)
        band = psd[2:18]
        expected = float(np.sqrt(np.exp(np.mean(np.log(band)))))
        self.assertAlmostEqual(estimate_asd(psd), expected)


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
            self.assertIn("简化 Allan", r.plot.allan_title)
            self.assertIn("Hz", r.plot.psd_title)

    def test_stats_table_has_asd_column(self) -> None:
        fs = 100.0
        n = 500
        time_s = np.arange(n) / fs
        channels = {"ch1": np.random.default_rng(0).normal(0, 1, n)}
        results = run_pipeline(channels, time_s, fs, (0.0, time_s[-1]))
        rows = stats_table(results)
        self.assertIn("ASD估计", rows[0])
        self.assertFalse(np.isnan(rows[0]["ASD估计"]))


if __name__ == "__main__":
    unittest.main()
