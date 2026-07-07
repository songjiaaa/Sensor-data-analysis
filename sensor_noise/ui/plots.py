"""Matplotlib 绘图控制器：只读缓存，按标签页渲染。"""

from __future__ import annotations

from pathlib import Path

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from sensor_noise.models import PlotCache


class PlotController:
    TABS = {
        "时域波形": "time",
        "功率谱密度": "psd",
        "Allan偏差": "allan",
        "噪声分布": "hist",
        "多通道对比": "multi",
    }

    def __init__(self, notebook, frames: dict[str, object]) -> None:
        self.notebook = notebook
        self.axes: dict[str, object] = {}
        self.canvas: dict[str, FigureCanvasTkAgg] = {}
        self._multi_bars: dict[str, object] = {}
        self._multi_channel_keys: tuple[str, ...] = ()
        for key, frame in frames.items():
            fig = Figure(figsize=(7, 4), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(canvas, frame)
            self.axes[key] = fig.add_subplot(111)
            self.canvas[key] = canvas

    def current_tab(self) -> str:
        selected = self.notebook.select()
        if not selected:
            tabs = self.notebook.tabs()
            if not tabs:
                return next(iter(self.TABS))
            self.notebook.select(tabs[0])
            selected = self.notebook.select()
        return str(self.notebook.tab(selected, "text"))

    def _redraw(self, key: str) -> None:
        try:
            if self.canvas[key].get_tk_widget().winfo_ismapped():
                self.canvas[key].draw_idle()
        except tk.TclError:
            pass

    def show_channel(self, cache: PlotCache | None, tab: str | None = None) -> None:
        tab = tab or self.current_tab()
        if tab == "多通道对比" or cache is None:
            return
        key = self.TABS.get(tab)
        if not key or key == "multi":
            return
        draw = {
            "time": self._time,
            "psd": self._psd,
            "allan": self._allan,
            "hist": self._hist,
        }[key]
        draw(self.axes[key], cache)
        self._redraw(key)

    def show_multi(self, std_by_channel: dict[str, float], highlight: str, full: bool = False) -> None:
        ax = self.axes["multi"]
        channel_keys = tuple(std_by_channel.keys())
        if full or not self._multi_bars or self._multi_channel_keys != channel_keys:
            self._multi_channel_keys = channel_keys
            ax.clear()
            self._multi_bars = {}
            for ch, std in std_by_channel.items():
                color = "tab:orange" if ch == highlight else "tab:blue"
                alpha = 1.0 if ch == highlight else 0.55
                self._multi_bars[ch] = ax.bar(ch, std, color=color, alpha=alpha)[0]
            ax.set_xlabel("通道")
            ax.set_ylabel("标准差 (σ)")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            for ch, bar in self._multi_bars.items():
                bar.set_color("tab:orange" if ch == highlight else "tab:blue")
                bar.set_alpha(1.0 if ch == highlight else 0.55)
        ax.set_title(f"各通道 σ 对比（高亮: {highlight}）")
        self._redraw("multi")

    def clear_all(self) -> None:
        """清空全部图表缓存与画布（不触发重绘，避免空闲回调堆积）。"""
        self._multi_bars = {}
        self._multi_channel_keys = ()
        for key in ("time", "psd", "allan", "hist", "multi"):
            ax = self.axes[key]
            ax.clear()
            ax.set_title("")

    _CHANNEL_DRAWERS = {
        "time": "_time",
        "psd": "_psd",
        "allan": "_allan",
        "hist": "_hist",
    }

    def export_channel_plots(self, cache: PlotCache, output_dir: Path, channel: str) -> None:
        """离屏渲染单通道图表，不依赖 canvas 当前状态。"""
        for key, name in self._CHANNEL_DRAWERS.items():
            fig = Figure(figsize=(7, 4), dpi=150)
            ax = fig.add_subplot(111)
            getattr(self, name)(ax, cache)
            fig.savefig(output_dir / f"{channel}_{name}.png", dpi=150, bbox_inches="tight")

    def export_multi_plot(
        self, std_by_channel: dict[str, float], highlight: str, path: Path
    ) -> None:
        """离屏渲染多通道对比图。"""
        fig = Figure(figsize=(7, 4), dpi=150)
        ax = fig.add_subplot(111)
        for ch, std in std_by_channel.items():
            color = "tab:orange" if ch == highlight else "tab:blue"
            alpha = 1.0 if ch == highlight else 0.55
            ax.bar(ch, std, color=color, alpha=alpha)
        ax.set_xlabel("通道")
        ax.set_ylabel("标准差 (σ)")
        ax.set_title(f"各通道 σ 对比（高亮: {highlight}）")
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(path, dpi=150, bbox_inches="tight")

    @staticmethod
    def _time(ax, c: PlotCache) -> None:
        ax.clear()
        ax.plot(c.time_x, c.time_y, lw=0.5, alpha=0.8, label="原始")
        ax.axhline(c.mean_line, color="r", ls="--", label=f"均值={c.mean_line:.4g}")
        ax.set(xlabel="时间 (s)", ylabel="幅值", title=c.time_title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _psd(ax, c: PlotCache) -> None:
        ax.clear()
        if len(c.psd_x):
            ax.plot(c.psd_x, c.psd_y, lw=0.8)
            ax.set_yscale("log")
            ax.set_xlim(c.psd_x[0], c.psd_x[-1])
        ax.set(xlabel="频率 (Hz)", ylabel="PSD", title=c.psd_title)
        ax.grid(True, alpha=0.3, which="both")

    @staticmethod
    def _allan(ax, c: PlotCache) -> None:
        ax.clear()
        if len(c.allan_x):
            ax.loglog(c.allan_x, c.allan_y, "o-", ms=4)
        ax.set(xlabel="平均时间 τ (s)", ylabel="Allan 偏差", title=c.allan_title)
        ax.grid(True, alpha=0.3, which="both")

    @staticmethod
    def _hist(ax, c: PlotCache) -> None:
        ax.clear()
        ax.bar(c.hist_bins, c.hist_counts, width=c.hist_bar_width, alpha=0.7, label="直方图")
        ax.plot(c.hist_pdf_x, c.hist_pdf_y, "r-", lw=1.5, label="高斯拟合")
        ax.set(xlabel="去均值幅值", ylabel="概率密度", title=c.hist_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
