"""数据加载单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_noise.loaders import load_recording


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

    def test_uses_meta_sample_rate(self) -> None:
        rows, _, _ = _make_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.csv"
            _write_csv(
                path,
                ["time", "ch1", "ch2"],
                rows,
                meta_lines=["#平均帧率(Hz),200"],
            )
            rec = load_recording(path)
            self.assertAlmostEqual(rec.sample_rate_hz, 200.0)


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


if __name__ == "__main__":
    unittest.main()
