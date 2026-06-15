"""主窗口 UI。"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

from sensor_noise.session import Session
from sensor_noise.ui.plots import PlotController

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ProgressDialog:
    def __init__(self, parent: tk.Tk, total: int) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title("正在分析")
        self.win.geometry("400x110")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()
        ttk.Label(self.win, text="正在分析全部通道...").pack(padx=12, pady=(12, 6))
        self.label = ttk.Label(self.win, text=f"0 / {total}")
        self.label.pack()
        self.var = tk.DoubleVar(value=0)
        ttk.Progressbar(self.win, maximum=total, variable=self.var, length=360).pack(padx=12, pady=10)

    def update(self, done: int, total: int, channel: str) -> None:
        self.var.set(done)
        self.label.config(text=f"{done} / {total}  {channel}")
        self.win.update_idletasks()

    def close(self) -> None:
        try:
            self.win.grab_release()
            self.win.destroy()
        except tk.TclError:
            pass


class MainWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("静态数据噪声分析工具")
        self.geometry("1280x860")
        self.session = Session()
        self.current_channel = tk.StringVar()
        self._sync = False
        self._progress: ProgressDialog | None = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        self.open_btn = ttk.Button(top, text="打开数据文件", command=self._open_file)
        self.open_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.file_label = ttk.Label(top, text="未加载", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(top, text="通道:").pack(side=tk.LEFT)
        self.channel_combo = ttk.Combobox(top, textvariable=self.current_channel, state="readonly", width=8)
        self.channel_combo.pack(side=tk.LEFT, padx=4)
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _: self._select_channel(self.current_channel.get()))
        self.analyze_btn = ttk.Button(top, text="开始分析", command=self._analyze)
        self.analyze_btn.pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="导出报告", command=self._export).pack(side=tk.LEFT)

        rf = ttk.LabelFrame(self, text="时间范围 (秒；修改后需重新分析)", padding=8)
        rf.pack(fill=tk.X, padx=8)
        self.t_start, self.t_end = tk.StringVar(), tk.StringVar()
        ttk.Label(rf, text="起始").pack(side=tk.LEFT)
        ttk.Entry(rf, textvariable=self.t_start, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(rf, text="结束").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Entry(rf, textvariable=self.t_end, width=12).pack(side=tk.LEFT, padx=4)
        self.meta = ttk.Label(self, text="", wraplength=1200)
        self.meta.pack(fill=tk.X, padx=8, pady=4)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        sf = ttk.LabelFrame(left, text="统计汇总（点击行切换通道）", padding=4)
        sf.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(sf, show="headings", height=12)
        sb = ttk.Scrollbar(sf, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._row_ids: dict[str, str] = {}

        right = ttk.Frame(paned)
        paned.add(right, weight=3)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        frames = {}
        for name in PlotController.TABS:
            f = ttk.Frame(self.notebook)
            self.notebook.add(f, text=name)
            frames[PlotController.TABS[name]] = f
        self.plots = PlotController(self.notebook, frames)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _: self._refresh_view())

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.open_btn.config(state=state)
        self.analyze_btn.config(state=state)
        self.channel_combo.config(state=tk.DISABLED if busy else "readonly")
        self.config(cursor="watch" if busy else "")

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("数据", "*.csv;*.xlsx;*.xlsm"), ("所有", "*.*")])
        if not path:
            return
        self._set_busy(True)
        self.file_label.config(text="加载中...")
        threading.Thread(target=lambda: self._load_worker(path), daemon=True).start()

    def _load_worker(self, path: str) -> None:
        try:
            rec = self.session.load(path)
            self.after(0, lambda: self._on_loaded(path, None))
        except Exception as e:
            self.after(0, lambda: self._on_loaded(path, e))

    def _on_loaded(self, path: str, err: Exception | None) -> None:
        self._set_busy(False)
        if err:
            messagebox.showerror("加载失败", str(err))
            return
        rec = self.session.recording
        assert rec is not None
        self.file_label.config(text=Path(path).name, foreground="black")
        self.channel_combo["values"] = rec.channel_names
        self.current_channel.set(rec.channel_names[0])
        self.t_start.set(f"{rec.time_s[0]:.3f}")
        self.t_end.set(f"{rec.time_s[-1]:.3f}")
        self.meta.config(
            text=f"帧数 {rec.meta.total_frames} | 采样率 {rec.sample_rate_hz:.1f} Hz | "
            f"时间 [{rec.time_s[0]:.2f}, {rec.time_s[-1]:.2f}] s | 通道 {len(rec.channel_names)}"
        )
        self.tree.delete(*self.tree.get_children())
        self._row_ids.clear()

    def _analyze(self) -> None:
        if not self.session.ready:
            messagebox.showwarning("提示", "请先打开文件")
            return
        try:
            t0 = float(self.t_start.get() or self.session.recording.time_s[0])
            t1 = float(self.t_end.get() or self.session.recording.time_s[-1])
            self.session.resolve_time_range(t0, t1)
        except ValueError as e:
            messagebox.showerror("时间无效", str(e))
            return
        total = len(self.session.recording.channel_names)
        self._set_busy(True)
        self._progress = ProgressDialog(self, total)

        def worker() -> None:
            try:
                self.session.analyze(on_progress=lambda d, t, c: self.after(0, lambda: self._progress.update(d, t, c)))
                self.after(0, self._on_analyzed_ok)
            except Exception as e:
                self.after(0, lambda: self._on_analyzed_fail(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_analyzed_ok(self) -> None:
        if self._progress:
            self._progress.close()
        self._set_busy(False)
        self._fill_table()
        ch = self.current_channel.get()
        if ch not in self.session.results:
            ch = next(iter(self.session.results))
            self.current_channel.set(ch)
        r = self.session.get(ch)
        if r and r.plot:
            self.plots.prewarm(r.plot)
        std_map = {k: v.stats.std for k, v in self.session.results.items()}
        self.plots.show_multi(std_map, ch, full=True)
        self._select_channel(ch)

    def _on_analyzed_fail(self, err: Exception) -> None:
        if self._progress:
            self._progress.close()
        self._set_busy(False)
        messagebox.showerror("分析失败", str(err))

    def _fill_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._row_ids.clear()
        rows = self.session.table_rows()
        if not rows:
            return
        cols = list(rows[0].keys())
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=90 if c != "通道" else 60, anchor=tk.CENTER)
        for row in rows:
            vals = [f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]) for c in cols]
            self._row_ids[row["通道"]] = self.tree.insert("", tk.END, values=vals)

    def _select_channel(self, channel: str) -> None:
        if channel not in self.session.results:
            return
        self._sync = True
        self.current_channel.set(channel)
        rid = self._row_ids.get(channel)
        if rid:
            self.tree.selection_set(rid)
            self.tree.see(rid)
        self._sync = False
        r = self.session.get(channel)
        if r and r.plot:
            self.plots.prewarm(r.plot)
        self._refresh_view()

    def _on_tree_select(self, _=None) -> None:
        if self._sync:
            return
        sel = self.tree.selection()
        if sel:
            ch = self.tree.item(sel[0], "values")[0]
            self._select_channel(ch)

    def _refresh_view(self) -> None:
        ch = self.current_channel.get()
        r = self.session.get(ch)
        tab = self.plots.current_tab()
        if tab == "多通道对比":
            std_map = {k: v.stats.std for k, v in self.session.results.items()}
            self.plots.show_multi(std_map, ch, full=False)
        elif r and r.plot:
            self.plots.show_channel(r.plot, tab)

    def _export(self) -> None:
        ch = self.current_channel.get()
        r = self.session.get(ch)
        if not r or not r.plot:
            messagebox.showwarning("提示", "请先分析")
            return
        out = filedialog.askdirectory()
        if not out:
            return
        p = Path(out)
        pd.DataFrame(self.session.table_rows()).to_csv(p / "noise_summary.csv", index=False, encoding="utf-8-sig")
        self.plots.export_channel_plots(r.plot, p, r.channel)
        std_map = {k: v.stats.std for k, v in self.session.results.items()}
        self.plots.export_multi_plot(std_map, ch, p / f"{r.channel}_multi.png")
        messagebox.showinfo("完成", f"已导出至 {p}")


def run_app() -> None:
    MainWindow().mainloop()
