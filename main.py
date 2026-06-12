"""静态传感器数据噪声分析软件 - GUI 入口。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from csv_loader import StaticRecording, load_static_csv
from noise_analyzer import analyze_channel, format_stats_table

# Windows 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class NoiseAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("静态数据噪声分析工具")
        self.geometry("1280x860")
        self.minsize(1024, 720)

        self.recording: StaticRecording | None = None
        self.results = []
        self.current_channel = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="打开 CSV 文件", command=self.open_file).pack(side=tk.LEFT, padx=(0, 8))
        self.file_label = ttk.Label(top, text="未加载文件", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(top, text="分析通道:").pack(side=tk.LEFT)
        self.channel_combo = ttk.Combobox(top, textvariable=self.current_channel, state="readonly", width=8)
        self.channel_combo.pack(side=tk.LEFT, padx=4)
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_changed)

        ttk.Button(top, text="开始分析", command=self.run_analysis).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="导出报告", command=self.export_report).pack(side=tk.LEFT, padx=4)

        # 时间范围
        range_frame = ttk.LabelFrame(self, text="分析时间范围 (秒，留空=全部)", padding=8)
        range_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.t_start = tk.StringVar()
        self.t_end = tk.StringVar()
        ttk.Label(range_frame, text="起始:").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.t_start, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(range_frame, text="结束:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Entry(range_frame, textvariable=self.t_end, width=12).pack(side=tk.LEFT, padx=4)

        self.meta_label = ttk.Label(self, text="", wraplength=1200)
        self.meta_label.pack(fill=tk.X, padx=8, pady=4)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        stats_frame = ttk.LabelFrame(left, text="噪声统计汇总", padding=4)
        stats_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_tree = ttk.Treeview(stats_frame, show="headings", height=12)
        vsb = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=vsb.set)
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_time = ttk.Frame(self.notebook)
        self.tab_psd = ttk.Frame(self.notebook)
        self.tab_allan = ttk.Frame(self.notebook)
        self.tab_hist = ttk.Frame(self.notebook)
        self.tab_multi = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_time, text="时域波形")
        self.notebook.add(self.tab_psd, text="功率谱密度")
        self.notebook.add(self.tab_allan, text="Allan偏差")
        self.notebook.add(self.tab_hist, text="噪声分布")
        self.notebook.add(self.tab_multi, text="多通道对比")

        self.fig_time = self._make_figure(self.tab_time)
        self.fig_psd = self._make_figure(self.tab_psd)
        self.fig_allan = self._make_figure(self.tab_allan)
        self.fig_hist = self._make_figure(self.tab_hist)
        self.fig_multi = self._make_figure(self.tab_multi)

        self.ax_time = self.fig_time.add_subplot(111)
        self.ax_psd = self.fig_psd.add_subplot(111)
        self.ax_allan = self.fig_allan.add_subplot(111)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.ax_multi = self.fig_multi.add_subplot(111)

    def _make_figure(self, parent: ttk.Frame) -> Figure:
        fig = Figure(figsize=(7, 4), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, parent)
        return fig

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择静态采集 CSV 文件",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self.recording = load_static_csv(path)
            self.file_label.config(text=Path(path).name, foreground="black")
            channels = self.recording.channel_names
            self.channel_combo["values"] = channels
            if channels:
                self.current_channel.set(channels[0])

            m = self.recording.meta
            self.meta_label.config(
                text=(
                    f"总帧数: {m.total_frames}  |  时长: {m.duration_s:.2f} s  |  "
                    f"采样率: {self.recording.sample_rate_hz:.2f} Hz  |  "
                    f"通道数: {len(channels)}"
                )
            )
            self.t_start.set(f"{self.recording.time_s[0]:.3f}")
            self.t_end.set(f"{self.recording.time_s[-1]:.3f}")
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))

    def _get_time_range(self) -> tuple[float, float] | None:
        try:
            if self.t_start.get().strip() and self.t_end.get().strip():
                return float(self.t_start.get()), float(self.t_end.get())
        except ValueError:
            pass
        return None

    def run_analysis(self) -> None:
        if self.recording is None:
            messagebox.showwarning("提示", "请先打开 CSV 文件")
            return

        time_range = self._get_time_range()
        self.results = []
        for ch_name, ch_data in self.recording.channels.items():
            result = analyze_channel(
                channel=ch_name,
                data=ch_data,
                sample_rate_hz=self.recording.sample_rate_hz,
                time_range=time_range,
                time_s=self.recording.time_s,
            )
            self.results.append(result)

        self._update_stats_table()
        self.on_channel_changed()
        self._plot_multi_channel()
        messagebox.showinfo("完成", f"已分析 {len(self.results)} 个通道")

    def _update_stats_table(self) -> None:
        for col in self.stats_tree["columns"]:
            self.stats_tree.heading(col, text="")
        self.stats_tree.delete(*self.stats_tree.get_children())

        rows = format_stats_table(self.results)
        if not rows:
            return

        columns = list(rows[0].keys())
        self.stats_tree["columns"] = columns
        for col in columns:
            self.stats_tree.heading(col, text=col)
            width = 100 if col != "通道" else 60
            self.stats_tree.column(col, width=width, anchor=tk.CENTER)

        for row in rows:
            values = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.6g}")
                else:
                    values.append(str(val))
            self.stats_tree.insert("", tk.END, values=values)

    def _find_result(self, channel: str):
        for r in self.results:
            if r.channel == channel:
                return r
        return None

    def on_channel_changed(self, _event=None) -> None:
        if not self.results:
            return
        ch = self.current_channel.get()
        result = self._find_result(ch)
        if result is None:
            return

        rec = self.recording
        time_range = self._get_time_range()
        if time_range and rec is not None:
            mask = (rec.time_s >= time_range[0]) & (rec.time_s <= time_range[1])
            t = rec.time_s[mask]
            raw = rec.channels[ch][mask]
        else:
            t = rec.time_s if rec else np.arange(len(result.detrended))
            raw = rec.channels[ch] if rec else result.detrended + result.basic.mean

        self._plot_time(t, raw, result)
        self._plot_psd(result)
        self._plot_allan(result)
        self._plot_hist(result)

    def _plot_time(self, t, raw, result) -> None:
        self.ax_time.clear()
        self.ax_time.plot(t, raw, linewidth=0.5, alpha=0.8, label="原始")
        self.ax_time.axhline(result.basic.mean, color="r", linestyle="--", linewidth=1, label=f"均值={result.basic.mean:.4g}")
        self.ax_time.set_xlabel("时间 (s)")
        self.ax_time.set_ylabel("幅值")
        self.ax_time.set_title(f"{result.channel} 时域波形  (σ={result.basic.std:.4g}, 峰峰值={result.basic.peak_to_peak:.4g})")
        self.ax_time.legend(loc="upper right", fontsize=8)
        self.ax_time.grid(True, alpha=0.3)
        self.fig_time.tight_layout()
        self.fig_time.canvas.draw_idle()

    def _plot_psd(self, result) -> None:
        self.ax_psd.clear()
        self.ax_psd.semilogy(result.frequencies_hz, result.psd, linewidth=0.8)
        self.ax_psd.set_xlabel("频率 (Hz)")
        self.ax_psd.set_ylabel("PSD")
        self.ax_psd.set_title(f"{result.channel} 功率谱密度 (Welch)")
        self.ax_psd.grid(True, alpha=0.3, which="both")
        self.fig_psd.tight_layout()
        self.fig_psd.canvas.draw_idle()

    def _plot_allan(self, result) -> None:
        self.ax_allan.clear()
        if len(result.allan_tau_s) > 0:
            self.ax_allan.loglog(result.allan_tau_s, result.allan_dev, "o-", markersize=4)
        self.ax_allan.set_xlabel("平均时间 τ (s)")
        self.ax_allan.set_ylabel("Allan 偏差")
        self.ax_allan.set_title(f"{result.channel} Allan 偏差")
        self.ax_allan.grid(True, alpha=0.3, which="both")
        self.fig_allan.tight_layout()
        self.fig_allan.canvas.draw_idle()

    def _plot_hist(self, result) -> None:
        self.ax_hist.clear()
        self.ax_hist.bar(
            result.histogram_bins,
            result.histogram_counts,
            width=np.diff(result.histogram_bins)[0] if len(result.histogram_bins) > 1 else 1,
            alpha=0.7,
            label="直方图",
        )
        x = np.linspace(result.histogram_bins.min(), result.histogram_bins.max(), 200)
        pdf = stats_norm_pdf(x, result.gaussian_mu, result.gaussian_sigma)
        self.ax_hist.plot(x, pdf, "r-", linewidth=1.5, label="高斯拟合")
        self.ax_hist.set_xlabel("去均值幅值")
        self.ax_hist.set_ylabel("概率密度")
        self.ax_hist.set_title(
            f"{result.channel} 噪声分布  (μ={result.gaussian_mu:.2e}, σ={result.gaussian_sigma:.4g})"
        )
        self.ax_hist.legend(fontsize=8)
        self.ax_hist.grid(True, alpha=0.3)
        self.fig_hist.tight_layout()
        self.fig_hist.canvas.draw_idle()

    def _plot_multi_channel(self) -> None:
        self.ax_multi.clear()
        for r in self.results:
            self.ax_multi.bar(r.channel, r.basic.std, alpha=0.8, label=r.channel)
        self.ax_multi.set_xlabel("通道")
        self.ax_multi.set_ylabel("标准差 (σ)")
        self.ax_multi.set_title("各通道噪声标准差对比")
        self.ax_multi.grid(True, alpha=0.3, axis="y")
        self.fig_multi.tight_layout()
        self.fig_multi.canvas.draw_idle()

    def export_report(self) -> None:
        if not self.results:
            messagebox.showwarning("提示", "请先执行分析")
            return

        out_dir = filedialog.askdirectory(title="选择报告输出目录")
        if not out_dir:
            return

        out_path = Path(out_dir)
        rows = format_stats_table(self.results)
        df = pd.DataFrame(rows)
        csv_path = out_path / "noise_summary.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        for fig, name in [
            (self.fig_time, "time_domain"),
            (self.fig_psd, "psd"),
            (self.fig_allan, "allan"),
            (self.fig_hist, "histogram"),
            (self.fig_multi, "multi_channel"),
        ]:
            fig.savefig(out_path / f"{name}.png", dpi=150, bbox_inches="tight")

        messagebox.showinfo("导出完成", f"报告已保存至:\n{out_path}")


def stats_norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main() -> None:
    app = NoiseAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
