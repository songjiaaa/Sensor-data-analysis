"""数据加载单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_noise.loaders import load_recording
from sensor_noise.session import Session


def _write_csv(path: Path, header: list[str], rows: list[list[float]], meta_lines: list[str] | None = None) -> None:
    lines: list[str] = []
    if meta_lines:
        lines.extend(meta_lines)
    lines.append(",".join(header))
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def _make_rows(n: int = 300, fs: float = 100.0) -> tuple[list[list[float]], list[list[float]], list[float]]:
    t = np.arange(n) / fs
    ch1 = t * 0.01 + 1.0
    ch2 = t * 0.02 + 2.0
    rows_ch1_first = [[t[i], ch1[i], ch2[i]] for i in range(n)]
    rows_shuffled = [[ch2[i], t[i], ch1[i]] for i in range(n)]
    return rows_ch1_first, rows_shuffled, t.tolist()


class TestCsvLoading(unittest.TestCase):
    def test_standard_column_order(self) -> None:
        rows, _, _ = _make_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "std.csv"
            _write_csv(path, ["time", "ch1", "ch2"], rows)
            rec = load_recording(path)
            self.assertEqual(rec.channel_names, ["ch1", "ch2"])
            self.assertEqual(len(rec.time_s), 300)
            self.assertAlmostEqual(rec.sample_rate_hz, 100.0, delta=1.0)

    def test_shuffled_columns_mapped_by_header(self) -> None:
        _, rows, t = _make_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shuffled.csv"
            _write_csv(path, ["ch2", "time", "ch1"], rows)
            rec = load_recording(path)
            ch1 = rec.channels["ch1"]
            ch2 = rec.channels["ch2"]
            self.assertAlmostEqual(ch1[0], t[0] * 0.01 + 1.0)
            self.assertAlmostEqual(ch2[0], t[0] * 0.02 + 2.0)

    def test_drops_nan_rows(self) -> None:
        rows, _, _ = _make_rows()
        rows[10] = [rows[10][0], float("nan"), rows[10][2]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nan.csv"
            _write_csv(path, ["time", "ch1", "ch2"], rows)
            rec = load_recording(path)
            self.assertEqual(len(rec.time_s), 299)

    def test_rejects_non_monotonic_time(self) -> None:
        rows, _, _ = _make_rows()
        rows[50], rows[51] = rows[51], rows[50]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_time.csv"
            _write_csv(path, ["time", "ch1", "ch2"], rows)
            with self.assertRaisesRegex(ValueError, "单调"):
                load_recording(path)

    def test_uses_duration_based_rate(self) -> None:
        rows, _, _ = _make_rows(n=300, fs=100.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.csv"
            _write_csv(
                path,
                ["time", "ch1", "ch2"],
                rows,
                meta_lines=["#平均帧率(Hz),100"],
            )
            rec = load_recording(path)
            self.assertAlmostEqual(rec.sample_rate_hz, 100.0, delta=1.0)
            self.assertEqual(rec.load_warnings, [])

    def test_ignores_nominal_sample_rate(self) -> None:
        rows, _, _ = _make_rows(fs=100.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nominal_only.csv"
            _write_csv(
                path,
                ["time", "ch1", "ch2"],
                rows,
                meta_lines=["#标称帧率(Hz),500"],
            )
            rec = load_recording(path)
            self.assertAlmostEqual(rec.sample_rate_hz, 100.0, delta=1.0)
            self.assertEqual(len(rec.load_warnings), 1)
            self.assertIn("标称帧率", rec.load_warnings[0])
            self.assertIn("平均帧率", rec.load_warnings[0])

    def test_warns_when_nominal_differs_from_average(self) -> None:
        rows, _, _ = _make_rows(fs=100.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mismatch.csv"
            _write_csv(
                path,
                ["time", "ch1", "ch2"],
                rows,
                meta_lines=["#平均帧率(Hz),500", "#标称帧率(Hz),500"],
            )
            rec = load_recording(path)
            self.assertAlmostEqual(rec.sample_rate_hz, 100.0, delta=1.0)
            self.assertEqual(len(rec.load_warnings), 1)
            self.assertIn("标称帧率", rec.load_warnings[0])
            self.assertIn("平均帧率", rec.load_warnings[0])
            self.assertNotIn("时长估算", rec.load_warnings[0])

    def test_rejects_duplicate_columns(self) -> None:
        rows, _, _ = _make_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup.csv"
            _write_csv(path, ["time", "ch1", "ch1"], [[r[0], r[1], r[2]] for r in rows])
            with self.assertRaisesRegex(ValueError, "重复列名"):
                load_recording(path)

    def test_english_meta_sample_rate(self) -> None:
        rows, _, _ = _make_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "en_meta.csv"
            _write_csv(
                path,
                ["time", "ch1", "ch2"],
                rows,
                meta_lines=["#avg_sample_rate_hz,100", "#nominal_sample_rate_hz,999"],
            )
            rec = load_recording(path)
            self.assertAlmostEqual(rec.sample_rate_hz, 100.0, delta=1.0)

    def test_corrects_time_when_meta_duration_mismatches(self) -> None:
        """模拟 THM060A 采集：时间列为 1 ms 步进计数，元数据时长为真实录制时长。"""
        n = 300
        true_fs = 10.0
        true_dur = (n - 1) / true_fs
        rows = [[i * 0.001, 1.0 + i * 0.01, 2.0] for i in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thm_style.csv"
            _write_csv(
                path,
                ["时间(s)", "ch1", "ch2"],
                rows,
                meta_lines=[
                    f"#时长(s),{true_dur}",
                    f"#平均帧率(Hz),{true_fs}",
                    f"#总帧数,{n}",
                ],
            )
            rec = load_recording(path)
            self.assertAlmostEqual(rec.sample_rate_hz, true_fs, delta=0.1)
            self.assertAlmostEqual(rec.time_s[-1] - rec.time_s[0], true_dur, delta=0.01)
            self.assertTrue(any("时间列时长" in w for w in rec.load_warnings))

    def test_keeps_time_column_when_meta_duration_agrees(self) -> None:
        rows, _, t = _make_rows(n=300, fs=100.0)
        true_dur = t[-1] - t[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consistent.csv"
            _write_csv(
                path,
                ["time", "ch1", "ch2"],
                rows,
                meta_lines=[f"#时长(s),{true_dur}", "#平均帧率(Hz),100"],
            )
            rec = load_recording(path)
            self.assertAlmostEqual(rec.sample_rate_hz, 100.0, delta=1.0)
            np.testing.assert_allclose(rec.time_s, t, rtol=0, atol=1e-9)
            self.assertEqual(rec.load_warnings, [])


class TestXlsxLoading(unittest.TestCase):
    def test_shuffled_columns_mapped_by_header(self) -> None:
        _, rows, t = _make_rows()
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(rows, columns=["ch2", "time", "ch1"])
            path = Path(tmp) / "shuffled.xlsx"
            df.to_excel(path, index=False, header=True)
            rec = load_recording(path)
            ch1 = rec.channels["ch1"]
            ch2 = rec.channels["ch2"]
            self.assertAlmostEqual(ch1[0], t[0] * 0.01 + 1.0)
            self.assertAlmostEqual(ch2[0], t[0] * 0.02 + 2.0)


class TestSessionTimeRange(unittest.TestCase):
    def test_rejects_short_time_window(self) -> None:
        rows, _, _ = _make_rows(n=300, fs=100.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            _write_csv(path, ["time", "ch1", "ch2"], rows)
            session = Session()
            session.load(path)
            with self.assertRaisesRegex(ValueError, "样本不足"):
                session.resolve_time_range(0.0, 0.5)


if __name__ == "__main__":
    unittest.main()
