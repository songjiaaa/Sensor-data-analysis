# 静态传感器噪声分析工具

加载 CSV / XLSX 录制数据，对多通道 IMU / 传感器信号做时域统计、Welch 功率谱密度、简化 Allan 偏差与噪声分布分析，并通过 Tkinter 桌面界面可视化与导出报告。

## 环境要求

- Python 3.10+
- Windows / Linux / macOS（推荐 Windows，已针对中文界面优化）

## 安装与运行

```bash
pip install -r requirements.txt
python main.py
```

Windows 快捷方式：

```bat
run.bat
```

## 输入文件格式

### 支持格式

- `.csv`（UTF-8，可含 BOM）
- `.xlsx` / `.xlsm`（不支持旧版 `.xls`）

### 表头要求

- **时间列**：列名为 `time` / `t` / `timestamp` / `time_s` / `时间` 等（精确匹配，避免误识别）
- **通道列**：`ch1`、`ch2`…，或 `fx` / `fy` / `fz` / `mx` / `my` / `mz`（不区分大小写）

列的物理顺序可以与表头名称不同；程序按列名映射，而非按位置。重复列名会被拒绝。

### 元数据（可选，CSV 以 `#` 开头）

| 键 | 说明 |
|----|------|
| 平均帧率(Hz) / avg_sample_rate_hz | 采样率（与时间列交叉校验） |
| 标称帧率(Hz) / nominal_sample_rate_hz | 次选采样率 |
| 总帧数 / 时长(s) 等 | 显示在界面 |

若元数据采样率与时间列估算偏差超过 5%，将**采用时间列估算值**并弹出提示。
若元数据与时间列均无法推断采样率，加载将报错。

### 数据质量

- 有效数值行至少 **256** 行
- 时间列须 **单调非递减**
- 含 NaN 的行会被自动丢弃

## 分析说明

| 功能 | 说明 |
|------|------|
| 时域统计 | 均值、标准差、RMS 噪声、峰峰值、偏度、峰度 |
| PSD | Welch 法，对去均值数据 |
| Allan 偏差 | 简化非重叠实现，适合快速浏览，非 IEEE 精密标定 |
| 噪声分布 | 直方图 + 高斯拟合曲线 |

## 导出

「导出报告」会在选定目录生成：

- `noise_summary.csv`：全部通道统计汇总
- `{通道}_time.png` / `_psd.png` / `_allan.png` / `_hist.png` / `_multi.png`：当前选中通道图表

## 打包 EXE

```bash
pip install -r requirements-dev.txt
build.bat
```

产物位于 `dist/`。

## 测试

```bash
python -m unittest discover -s tests -v
```

或使用 pytest（需先 `pip install pytest`）：

```bash
pytest tests/ -v
```

## 项目结构

```
main.py                 # 入口
sensor_noise/
  models.py             # 数据模型
  loaders.py            # 文件加载
  analysis.py           # 分析算法
  session.py            # 应用状态
  ui/app.py             # 主界面
  ui/plots.py           # 图表
tests/                  # 单元测试
```
