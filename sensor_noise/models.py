"""领域数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RecordingMeta:
    start_time: str = ""
    end_time: str = ""
    total_frames: int = 0
    duration_s: float = 0.0
    avg_sample_rate_hz: float = 0.0
    dropped_frames: int = 0
    nominal_sample_rate_hz: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class Recording:
    path: Path
    meta: RecordingMeta
    time_s: np.ndarray
    channels: dict[str, np.ndarray]
    sample_rate_hz: float

    @property
    def channel_names(self) -> list[str]:
        return list(self.channels.keys())


@dataclass
class NoiseStats:
    mean: float
    std: float
    rms_noise: float
    min_val: float
    max_val: float
    peak_to_peak: float
    variance: float
    skewness: float
    kurtosis: float
    n_samples: int


@dataclass
class PlotCache:
    channel: str
    time_range: tuple[float, float]
    time_x: np.ndarray
    time_y: np.ndarray
    time_title: str
    mean_line: float
    psd_x: np.ndarray
    psd_y: np.ndarray
    psd_title: str
    allan_x: np.ndarray
    allan_y: np.ndarray
    allan_title: str
    hist_bins: np.ndarray
    hist_counts: np.ndarray
    hist_bar_width: float
    hist_pdf_x: np.ndarray
    hist_pdf_y: np.ndarray
    hist_title: str


@dataclass
class ChannelResult:
    channel: str
    stats: NoiseStats
    time_s: np.ndarray
    raw_data: np.ndarray
    time_range: tuple[float, float]
    frequencies_hz: np.ndarray
    psd: np.ndarray
    allan_tau_s: np.ndarray
    allan_dev: np.ndarray
    histogram_bins: np.ndarray
    histogram_counts: np.ndarray
    gaussian_mu: float
    gaussian_sigma: float
    plot: PlotCache | None = None
