"""THM060A 静态采集 CSV 数据加载模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RecordingMeta:
    """录制元数据。"""

    start_time: str = ""
    end_time: str = ""
    total_frames: int = 0
    duration_s: float = 0.0
    avg_sample_rate_hz: float = 0.0
    dropped_frames: int = 0
    nominal_sample_rate_hz: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class StaticRecording:
    """静态采集数据集。"""

    path: Path
    meta: RecordingMeta
    time_s: np.ndarray
    channels: dict[str, np.ndarray]
    sample_rate_hz: float

    @property
    def channel_names(self) -> list[str]:
        return list(self.channels.keys())

    def get_channel(self, name: str) -> np.ndarray:
        return self.channels[name]


def _parse_meta_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line.startswith("#"):
        return None
    content = line.lstrip("#").strip()
    if "," not in content:
        return None
    parts = [p.strip() for p in content.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def load_static_csv(path: str | Path) -> StaticRecording:
    """加载 THM060A 静态采集 CSV 文件。"""
    path = Path(path)
    meta = RecordingMeta()
    meta_keys = {
        "录制开始": "start_time",
        "录制结束": "end_time",
        "总帧数": "total_frames",
        "时长(s)": "duration_s",
        "平均帧率(Hz)": "avg_sample_rate_hz",
        "丢弃帧数": "dropped_frames",
        "标称帧率(Hz)": "nominal_sample_rate_hz",
    }

    with path.open("r", encoding="utf-8-sig") as f:
        header_line_idx = 0
        for idx, line in enumerate(f):
            parsed = _parse_meta_line(line)
            if parsed is None:
                header_line_idx = idx
                break
            key, value = parsed
            if key in meta_keys:
                attr = meta_keys[key]
                if attr in {"total_frames", "dropped_frames"}:
                    setattr(meta, attr, int(float(value)))
                elif attr in {"duration_s", "avg_sample_rate_hz", "nominal_sample_rate_hz"}:
                    setattr(meta, attr, float(value))
                else:
                    setattr(meta, attr, value)
            else:
                meta.extra[key] = value

    df = pd.read_csv(path, skiprows=header_line_idx, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    time_col = next((c for c in df.columns if "时间" in c or c.lower() == "time"), None)
    if time_col is None:
        raise ValueError("未找到时间列")

    time_s = df[time_col].to_numpy(dtype=float)
    channel_cols = [c for c in df.columns if c.startswith("ch") or c.lower().startswith("ch")]
    if not channel_cols:
        channel_cols = [c for c in df.columns if c not in {time_col, "seq"}]

    channels = {col: df[col].to_numpy(dtype=float) for col in channel_cols}

    if meta.avg_sample_rate_hz > 0:
        sample_rate = meta.avg_sample_rate_hz
    elif meta.nominal_sample_rate_hz > 0:
        sample_rate = meta.nominal_sample_rate_hz
    elif len(time_s) > 1:
        dt = np.diff(time_s)
        sample_rate = 1.0 / np.median(dt[dt > 0])
    else:
        sample_rate = 1.0

    return StaticRecording(
        path=path,
        meta=meta,
        time_s=time_s,
        channels=channels,
        sample_rate_hz=sample_rate,
    )
