"""噪声分析算法与批处理流水线。"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import signal, stats

from sensor_noise.models import ChannelResult, NoiseStats, PlotCache

ProgressCallback = Callable[[int, int, str], None]

PSD_LOW_FRAC = 0.01
PSD_HIGH_FRAC = 0.97
PSD_EDGE_FRAC = 0.01
MIDBAND_START_FRAC = 0.10
MIDBAND_END_FRAC = 0.90


def compute_stats(data: np.ndarray) -> NoiseStats:
    data = np.asarray(data, dtype=float)
    mean = float(np.mean(data))
    centered = data - mean
    std = float(np.std(centered, ddof=1))
    return NoiseStats(
        mean=mean,
        std=std,
        rms_noise=float(np.sqrt(np.mean(centered**2))),
        min_val=float(np.min(data)),
        max_val=float(np.max(data)),
        peak_to_peak=float(np.max(data) - np.min(data)),
        variance=float(np.var(centered, ddof=1)),
        skewness=float(stats.skew(centered, bias=False)),
        kurtosis=float(stats.kurtosis(centered, bias=False)),
        n_samples=len(data),
    )


def compute_allan(data: np.ndarray, fs: float, max_points: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """简化非重叠 Allan 偏差，用于快速浏览，非 IEEE 重叠 Allan (OADEV)。"""
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 10:
        return np.array([]), np.array([])
    max_m = n // 4
    if max_m < 2:
        return np.array([]), np.array([])
    m_values = np.unique(np.logspace(0, np.log10(max_m), num=max_points, dtype=int))
    taus, adevs = [], []
    for m in m_values[m_values >= 1]:
        m = int(m)
        n_clusters = n // m
        if n_clusters < 2:
            continue
        avg = data[: n_clusters * m].reshape(n_clusters, m).mean(axis=1)
        avar = 0.5 * np.mean(np.diff(avg) ** 2)
        taus.append(m / fs)
        adevs.append(float(np.sqrt(avar)))
    return np.array(taus), np.array(adevs)


def trim_psd(freqs: np.ndarray, psd: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """裁掉直流邻域与奈奎斯特边缘，减弱窗泄漏；最高频率约为 0.97×(fs/2)。"""
    if len(freqs) < 5:
        return freqs, psd
    nyq = fs / 2.0
    mask = (freqs >= max(float(freqs[1]), nyq * PSD_LOW_FRAC)) & (freqs <= nyq * PSD_HIGH_FRAC)
    f, p = freqs[mask], psd[mask]
    edge = max(1, len(f) // int(1 / PSD_EDGE_FRAC))
    return f[edge:-edge], p[edge:-edge]


def midband_psd(psd: np.ndarray) -> np.ndarray:
    if len(psd) < 4:
        return np.array([])
    start = int(len(psd) * MIDBAND_START_FRAC)
    end = int(len(psd) * MIDBAND_END_FRAC)
    return psd[start:end]


def estimate_asd(psd: np.ndarray) -> float:
    """中频段 PSD 几何均值开方，作为幅值谱密度 (ASD) 量级估计。"""
    band = midband_psd(psd)
    if len(band) == 0:
        return float("nan")
    return float(np.sqrt(np.exp(np.mean(np.log(band)))))


def compute_psd(data: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    nperseg = max(256, min(8192, n // 2))
    freqs, psd = signal.welch(
        np.asarray(data, float),
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
        scaling="density",
        detrend=False,
    )
    return trim_psd(freqs, psd, fs)


def compute_histogram(centered: np.ndarray, bins: int = 80):
    centered = np.asarray(centered, float)
    counts, edges = np.histogram(centered, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, float(np.mean(centered)), float(np.std(centered, ddof=1))


def build_time_mask(time_s: np.ndarray, time_range: tuple[float, float]) -> np.ndarray:
    mask = (time_s >= time_range[0]) & (time_s <= time_range[1])
    if int(np.count_nonzero(mask)) < 256:
        raise ValueError(f"时间范围 {time_range} 内样本不足 256 个")
    return mask


def decimate(x: np.ndarray, y: np.ndarray, max_points: int = 4000):
    if len(x) <= max_points:
        return x, y
    return x[:: int(np.ceil(len(x) / max_points))], y[:: int(np.ceil(len(y) / max_points))]


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_plot_cache(result: ChannelResult) -> PlotCache:
    t0, t1 = result.time_range
    tx, ty = decimate(result.time_s, result.raw_data)
    note = f", 绘图{max(1, result.stats.n_samples // max(len(tx), 1))}x降采样" if len(tx) < result.stats.n_samples else ""
    hb = result.histogram_bins
    bw = float(np.diff(hb)[0]) if len(hb) > 1 else 1.0
    px = np.linspace(hb.min(), hb.max(), 200) if len(hb) > 1 else hb.copy()
    psd_x = result.frequencies_hz
    if len(psd_x) >= 2:
        psd_range = f", {psd_x[0]:.1f}~{psd_x[-1]:.1f} Hz"
    else:
        psd_range = ""
    return PlotCache(
        channel=result.channel,
        time_range=result.time_range,
        time_x=tx,
        time_y=ty,
        time_title=f"{result.channel} 时域 [{t0:.2f}~{t1:.2f}s] (σ={result.stats.std:.4g}, 峰峰值={result.stats.peak_to_peak:.4g}{note})",
        mean_line=result.stats.mean,
        psd_x=psd_x,
        psd_y=np.maximum(result.psd, np.finfo(float).tiny),
        psd_title=f"{result.channel} PSD (Welch){psd_range} [{t0:.2f}~{t1:.2f}s]",
        allan_x=result.allan_tau_s,
        allan_y=result.allan_dev,
        allan_title=f"{result.channel} 简化 Allan（非 OADEV） [{t0:.2f}~{t1:.2f}s]",
        hist_bins=hb,
        hist_counts=result.histogram_counts,
        hist_bar_width=bw,
        hist_pdf_x=px,
        hist_pdf_y=_gaussian_pdf(px, result.gaussian_mu, result.gaussian_sigma),
        hist_title=f"{result.channel} 分布 [{t0:.2f}~{t1:.2f}s] (σ={result.gaussian_sigma:.4g})",
    )


def analyze_channel(
    channel: str,
    time_s: np.ndarray,
    data: np.ndarray,
    fs: float,
    time_range: tuple[float, float],
) -> ChannelResult:
    data = np.asarray(data, float)
    centered = data - np.mean(data)
    bins, counts, mu, sigma = compute_histogram(centered)
    freqs, psd = compute_psd(centered, fs)
    taus, adev = compute_allan(centered, fs)
    result = ChannelResult(
        channel=channel,
        stats=compute_stats(data),
        time_s=time_s,
        raw_data=data,
        time_range=time_range,
        frequencies_hz=freqs,
        psd=psd,
        allan_tau_s=taus,
        allan_dev=adev,
        histogram_bins=bins,
        histogram_counts=counts,
        gaussian_mu=mu,
        gaussian_sigma=sigma,
        plot=None,
    )
    result.plot = build_plot_cache(result)
    return result


def run_pipeline(
    channels: dict[str, np.ndarray],
    time_s: np.ndarray,
    fs: float,
    time_range: tuple[float, float],
    on_progress: ProgressCallback | None = None,
) -> dict[str, ChannelResult]:
    mask = build_time_mask(time_s, time_range)
    t = time_s[mask]
    names = list(channels.keys())
    results: dict[str, ChannelResult] = {}
    for i, name in enumerate(names, 1):
        results[name] = analyze_channel(name, t, channels[name][mask], fs, time_range)
        if on_progress:
            on_progress(i, len(names), name)
    return results


def stats_table(results: dict[str, ChannelResult]) -> list[dict]:
    rows = []
    for r in results.values():
        rows.append(
            {
                "通道": r.channel,
                "均值": r.stats.mean,
                "标准差(σ)": r.stats.std,
                "RMS噪声": r.stats.rms_noise,
                "峰峰值": r.stats.peak_to_peak,
                "偏度": r.stats.skewness,
                "峰度": r.stats.kurtosis,
                "样本数": r.stats.n_samples,
                "ASD估计": estimate_asd(r.psd),
            }
        )
    return rows
