"""应用状态与业务编排。"""

from __future__ import annotations

from pathlib import Path

from sensor_noise.analysis import run_pipeline, stats_table
from sensor_noise.loaders import load_recording
from sensor_noise.models import ChannelResult, Recording


class Session:
    """单一数据源：录制数据 + 分析结果 + 时间范围。"""

    def __init__(self) -> None:
        self.recording: Recording | None = None
        self.time_range: tuple[float, float] | None = None
        self.results: dict[str, ChannelResult] = {}

    @property
    def ready(self) -> bool:
        return self.recording is not None

    @property
    def analyzed(self) -> bool:
        return bool(self.results)

    def load(self, path: str | Path) -> Recording:
        self.recording = load_recording(path)
        self.results = {}
        rec = self.recording
        self.time_range = (float(rec.time_s[0]), float(rec.time_s[-1]))
        return rec

    def resolve_time_range(self, t0: float, t1: float) -> tuple[float, float]:
        if not self.recording:
            raise RuntimeError("未加载文件")
        f0, f1 = float(self.recording.time_s[0]), float(self.recording.time_s[-1])
        t0, t1 = max(t0, f0), min(t1, f1)
        if t1 <= t0:
            raise ValueError(f"时间范围 [{t0}, {t1}] 无效")
        self.time_range = (t0, t1)
        return self.time_range

    def analyze(self, on_progress=None) -> dict[str, ChannelResult]:
        if not self.recording or not self.time_range:
            raise RuntimeError("未加载文件或未设置时间范围")
        rec = self.recording
        self.results = run_pipeline(
            rec.channels,
            rec.time_s,
            rec.sample_rate_hz,
            self.time_range,
            on_progress,
        )
        return self.results

    def get(self, channel: str) -> ChannelResult | None:
        return self.results.get(channel)

    def table_rows(self) -> list[dict]:
        return stats_table(self.results)
