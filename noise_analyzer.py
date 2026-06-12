"""静态传感器数据噪声分析核心算法。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats


@dataclass
class BasicNoiseStats:
    """基础噪声统计量。"""

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
class ChannelNoiseResult:
    """单通道噪声分析结果。"""

    channel: str
    basic: BasicNoiseStats
    detrended: np.ndarray
    frequencies_hz: np.ndarray
    psd: np.ndarray
    allan_tau_s: np.ndarray
    allan_dev: np.ndarray
    histogram_bins: np.ndarray
    histogram_counts: np.ndarray
    gaussian_mu: float
    gaussian_sigma: float


def compute_basic_stats(data: np.ndarray) -> BasicNoiseStats:
    """计算基础噪声统计量（去均值后的 RMS 为噪声 RMS）。"""
    data = np.asarray(data, dtype=float)
    mean = float(np.mean(data))
    centered = data - mean
    std = float(np.std(centered, ddof=1))
    return BasicNoiseStats(
        mean=mean,
        std=std,
        rms_noise=std,
        min_val=float(np.min(data)),
        max_val=float(np.max(data)),
        peak_to_peak=float(np.max(data) - np.min(data)),
        variance=float(np.var(centered, ddof=1)),
        skewness=float(stats.skew(centered)),
        kurtosis=float(stats.kurtosis(centered)),
        n_samples=len(data),
    )


def compute_allan_deviation(
    data: np.ndarray,
    sample_rate_hz: float,
    max_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算重叠 Allan 偏差（OADEV）。

    适用于静态数据，用于区分白噪声、随机游走、偏置不稳定性等。
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 10:
        return np.array([]), np.array([])

    max_m = n // 4
    if max_m < 2:
        return np.array([]), np.array([])

    m_values = np.unique(
        np.logspace(0, np.log10(max_m), num=max_points, dtype=int)
    )
    m_values = m_values[m_values >= 1]

    taus = []
    adevs = []
    for m in m_values:
        m = int(m)
        n_clusters = n // m
        if n_clusters < 2:
            continue
        trimmed = data[: n_clusters * m]
        cluster_avg = trimmed.reshape(n_clusters, m).mean(axis=1)
        diffs = np.diff(cluster_avg)
        avar = 0.5 * np.mean(diffs**2)
        taus.append(m / sample_rate_hz)
        adevs.append(np.sqrt(avar))

    return np.array(taus), np.array(adevs)


def compute_psd(
    data: np.ndarray,
    sample_rate_hz: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch 法计算功率谱密度。"""
    data = np.asarray(data, dtype=float)
    centered = data - np.mean(data)
    n = len(centered)
    if nperseg is None:
        nperseg = min(4096, n // 4)
    nperseg = max(256, min(nperseg, n // 2))

    freqs, psd = signal.welch(
        centered,
        fs=sample_rate_hz,
        nperseg=nperseg,
        scaling="density",
        detrend="constant",
    )
    return freqs, psd


def compute_histogram(
    data: np.ndarray,
    bins: int = 80,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """计算去均值数据的直方图及高斯拟合参数。"""
    centered = np.asarray(data, dtype=float) - np.mean(data)
    counts, edges = np.histogram(centered, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = float(np.mean(centered))
    sigma = float(np.std(centered, ddof=1))
    return centers, counts, mu, sigma


def analyze_channel(
    channel: str,
    data: np.ndarray,
    sample_rate_hz: float,
    time_range: tuple[float, float] | None = None,
    time_s: np.ndarray | None = None,
) -> ChannelNoiseResult:
    """对单通道执行完整噪声分析。"""
    data = np.asarray(data, dtype=float)
    if time_range is not None and time_s is not None:
        mask = (time_s >= time_range[0]) & (time_s <= time_range[1])
        data = data[mask]

    detrended = data - np.mean(data)
    basic = compute_basic_stats(data)
    freqs, psd = compute_psd(detrended, sample_rate_hz)
    taus, adev = compute_allan_deviation(detrended, sample_rate_hz)
    bins, counts, mu, sigma = compute_histogram(detrended)

    return ChannelNoiseResult(
        channel=channel,
        basic=basic,
        detrended=detrended,
        frequencies_hz=freqs,
        psd=psd,
        allan_tau_s=taus,
        allan_dev=adev,
        histogram_bins=bins,
        histogram_counts=counts,
        gaussian_mu=mu,
        gaussian_sigma=sigma,
    )


def noise_density_from_psd(psd: np.ndarray, freqs: np.ndarray) -> float:
    """从 PSD 估算白噪声密度（sqrt of median PSD in flat band）。"""
    if len(psd) < 4:
        return float("nan")
    # 取中间频段避免 DC 和 Nyquist 边缘
    lo = len(psd) // 10
    hi = len(psd) * 9 // 10
    if hi <= lo:
        return float(np.sqrt(np.median(psd)))
    return float(np.sqrt(np.median(psd[lo:hi])))


def format_stats_table(results: list[ChannelNoiseResult]) -> list[dict]:
    """将多通道结果格式化为表格行。"""
    rows = []
    for r in results:
        nd = noise_density_from_psd(r.psd, r.frequencies_hz)
        rows.append(
            {
                "通道": r.channel,
                "均值": r.basic.mean,
                "标准差(σ)": r.basic.std,
                "RMS噪声": r.basic.rms_noise,
                "峰峰值": r.basic.peak_to_peak,
                "偏度": r.basic.skewness,
                "峰度": r.basic.kurtosis,
                "样本数": r.basic.n_samples,
                "PSD噪声密度": nd,
            }
        )
    return rows
