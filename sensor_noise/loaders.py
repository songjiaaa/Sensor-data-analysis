"""数据文件加载：CSV / XLSX。"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_noise.models import Recording, RecordingMeta

IMU_CHANNEL_NAMES = ("fx", "fy", "fz", "mx", "my", "mz")
TIME_COLUMN_EXACT = {
    "time",
    "t",
    "timestamp",
    "time_s",
    "time(s)",
    "times",
    "sec",
    "second",
    "seconds",
    "时间",
}
META_KEY_ALIASES: dict[str, str] = {
    "录制开始": "start_time",
    "录制结束": "end_time",
    "总帧数": "total_frames",
    "时长(s)": "duration_s",
    "平均帧率(hz)": "avg_sample_rate_hz",
    "平均帧率(Hz)": "avg_sample_rate_hz",
    "丢弃帧数": "dropped_frames",
    "标称帧率(hz)": "nominal_sample_rate_hz",
    "标称帧率(Hz)": "nominal_sample_rate_hz",
    "start_time": "start_time",
    "end_time": "end_time",
    "total_frames": "total_frames",
    "duration_s": "duration_s",
    "avg_sample_rate_hz": "avg_sample_rate_hz",
    "average frame rate (hz)": "avg_sample_rate_hz",
    "dropped_frames": "dropped_frames",
    "nominal_sample_rate_hz": "nominal_sample_rate_hz",
    "nominal frame rate (hz)": "nominal_sample_rate_hz",
}
SKIP_COLUMNS = {"seq", "index", "序号", "id", "frame", "帧号", "计数"}
NON_CHANNEL_PATTERN = re.compile(
    r"(status|state|mode|flag|version|comment|note|remark|label|name|"
    r"备注|状态|电压|温度|temp|voltage|current|电流|counter|计数)",
    re.IGNORECASE,
)
SAMPLE_RATE_MISMATCH_THRESHOLD = 0.05


def _normalize_col(name: object) -> str:
    return str(name).strip().lower()


def _normalize_meta_key(key: str) -> str:
    return key.strip()


def _lookup_meta_attr(key: str) -> str | None:
    normalized = _normalize_meta_key(key)
    if normalized in META_KEY_ALIASES:
        return META_KEY_ALIASES[normalized]
    lowered = normalized.lower()
    if lowered in META_KEY_ALIASES:
        return META_KEY_ALIASES[lowered]
    return None


def _is_channel_column(name: str) -> bool:
    normalized = _normalize_col(name)
    if normalized in IMU_CHANNEL_NAMES:
        return True
    return bool(re.fullmatch(r"ch\d+", normalized))


def _channel_sort_key(name: str) -> tuple:
    normalized = _normalize_col(name)
    imu_order = {axis: idx for idx, axis in enumerate(IMU_CHANNEL_NAMES)}
    if normalized in imu_order:
        return (1, imu_order[normalized], normalized)
    match = re.fullmatch(r"ch(\d+)", normalized)
    if match:
        return (0, int(match.group(1)), normalized)
    return (2, normalized, normalized)


def _find_time_column(columns: list[str]) -> str | None:
    for col in columns:
        if _normalize_col(col) in TIME_COLUMN_EXACT:
            return col
    for col in columns:
        stripped = str(col).strip()
        if stripped in {"时间", "时间(s)", "时间（s）"}:
            return col
    return None


def _looks_like_non_channel(name: str) -> bool:
    return bool(NON_CHANNEL_PATTERN.search(str(name)))


def _detect_channel_columns(columns: list[str], time_col: str | None = None) -> list[str]:
    matched = [col for col in columns if _is_channel_column(col)]
    if matched:
        return sorted(matched, key=_channel_sort_key)
    excluded = {_normalize_col(c) for c in SKIP_COLUMNS}
    if time_col is not None:
        excluded.add(_normalize_col(time_col))
    fallback = [
        col
        for col in columns
        if _normalize_col(col) not in excluded and not _looks_like_non_channel(col)
    ]
    return fallback


def _parse_csv_fields(line: str) -> list[str]:
    return [field.strip() for field in next(csv.reader(io.StringIO(line))) if field.strip()]


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


def _apply_meta_value(meta: RecordingMeta, key: str, value: str) -> None:
    attr = _lookup_meta_attr(key)
    if attr is None:
        meta.extra[key] = value
        return
    if attr in {"total_frames", "dropped_frames"}:
        setattr(meta, attr, int(float(value)))
    elif attr in {"duration_s", "avg_sample_rate_hz", "nominal_sample_rate_hz"}:
        setattr(meta, attr, float(value))
    else:
        setattr(meta, attr, value)


def _parse_meta_rows(raw_rows: list[list[str]]) -> RecordingMeta:
    meta = RecordingMeta()
    for row in raw_rows:
        cells = [str(c).strip() for c in row if str(c).strip() and str(c).lower() != "nan"]
        if not cells:
            continue
        if cells[0].startswith("#"):
            parsed = _parse_meta_line(cells[0])
            if parsed:
                _apply_meta_value(meta, parsed[0], parsed[1])
            continue
        if len(cells) >= 2 and _lookup_meta_attr(cells[0]) is not None:
            _apply_meta_value(meta, cells[0], cells[1])
            continue
        if len(cells) == 1 and "," in cells[0]:
            parts = [p.strip() for p in cells[0].split(",") if p.strip()]
            if len(parts) >= 2 and _lookup_meta_attr(parts[0]) is not None:
                _apply_meta_value(meta, parts[0], parts[1])
    return meta


def _row_looks_like_header(row: pd.Series) -> bool:
    values = [str(v).strip() for v in row.values if str(v).strip() and str(v).lower() != "nan"]
    if not values:
        return False
    time_col = _find_time_column(values)
    has_time = time_col is not None
    has_channel = len(_detect_channel_columns(values, time_col)) > 0
    return has_time and has_channel


def _find_header_row_index(raw: pd.DataFrame, max_scan: int = 100) -> int:
    scan_limit = min(max_scan, len(raw))
    for idx in range(scan_limit):
        if _row_looks_like_header(raw.iloc[idx]):
            return idx
    raise ValueError("未找到有效表头行，请确认文件包含时间列和通道列")


def _build_dataframe_from_raw(raw: pd.DataFrame, header_idx: int) -> pd.DataFrame:
    columns = [str(c).strip() for c in raw.iloc[header_idx].tolist()]
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = columns
    return df.reset_index(drop=True)


def _validate_unique_columns(columns: list[str]) -> None:
    if len(columns) != len(set(columns)):
        dupes = sorted({c for c in columns if columns.count(c) > 1})
        raise ValueError(f"表头存在重复列名: {', '.join(dupes)}")


def _sample_rate_from_time(time_s: np.ndarray) -> float:
    if len(time_s) <= 1:
        return 0.0
    positive_dt = np.diff(time_s)
    positive_dt = positive_dt[positive_dt > 0]
    if len(positive_dt) == 0:
        return 0.0
    return float(1.0 / np.median(positive_dt))


def _resolve_sample_rate(meta: RecordingMeta, time_s: np.ndarray) -> tuple[float, list[str]]:
    warnings: list[str] = []
    from_time = _sample_rate_from_time(time_s)
    meta_rate = meta.avg_sample_rate_hz if meta.avg_sample_rate_hz > 0 else meta.nominal_sample_rate_hz

    if meta_rate > 0 and from_time > 0:
        rel_diff = abs(meta_rate - from_time) / from_time
        if rel_diff > SAMPLE_RATE_MISMATCH_THRESHOLD:
            warnings.append(
                f"元数据采样率 ({meta_rate:.2f} Hz) 与时间列估算 ({from_time:.2f} Hz) "
                f"偏差 {rel_diff * 100:.1f}%，已采用时间列估算值"
            )
            return from_time, warnings
        return meta_rate, warnings

    if meta_rate > 0:
        return meta_rate, warnings
    if from_time > 0:
        return from_time, warnings
    return 0.0, warnings


def _validate_time_column(time_s: np.ndarray) -> None:
    if len(time_s) <= 1:
        return
    dt = np.diff(time_s)
    if np.any(dt < 0):
        raise ValueError("时间列必须单调非递减，请检查数据是否按时间排序")
    if not np.any(dt > 0):
        raise ValueError("时间列无有效间隔，无法推断采样率")


def _finalize_recording(path: Path, meta: RecordingMeta, df: pd.DataFrame) -> Recording:
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]

    time_col = _find_time_column(list(df.columns))
    if time_col is None:
        raise ValueError("未找到时间列（支持: 时间/time/t/timestamp 等）")

    channel_cols = _detect_channel_columns(list(df.columns), time_col)
    if not channel_cols:
        raise ValueError(
            "未找到通道列（支持: ch1/ch2...、fx/fy/fz/mx/my/mz，不区分大小写）"
        )

    selected_cols = [time_col, *channel_cols]
    _validate_unique_columns(selected_cols)

    df = df[selected_cols].copy()
    for col in selected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    if len(df) < 256:
        raise ValueError(f"有效数据仅 {len(df)} 行，至少需要 256 行")

    time_s = df[time_col].to_numpy(dtype=np.float64)
    channels = {col: df[col].to_numpy(dtype=np.float64) for col in channel_cols}
    _validate_time_column(time_s)

    sample_rate, load_warnings = _resolve_sample_rate(meta, time_s)
    if sample_rate <= 0:
        raise ValueError(
            "无法推断采样率：请在文件元数据中提供「平均帧率(Hz)」或「标称帧率(Hz)」，"
            "或确保时间列具有有效间隔"
        )

    if meta.total_frames <= 0:
        meta.total_frames = len(time_s)
    if meta.duration_s <= 0 and len(time_s) > 1:
        meta.duration_s = float(time_s[-1] - time_s[0])

    return Recording(
        path=path,
        meta=meta,
        time_s=time_s,
        channels=channels,
        sample_rate_hz=sample_rate,
        load_warnings=load_warnings,
    )


def _scan_csv_header(path: Path) -> tuple[int, RecordingMeta, list[str]]:
    """快速扫描 CSV 表头，返回数据起始行、元数据和列名。"""
    meta = RecordingMeta()
    header_line_idx = 0

    with path.open("r", encoding="utf-8-sig") as file:
        for idx, line in enumerate(file):
            parsed = _parse_meta_line(line)
            if parsed is None:
                header_line_idx = idx
                header_line = line.strip()
                break
            _apply_meta_value(meta, parsed[0], parsed[1])
        else:
            raise ValueError("CSV 文件为空")

    if header_line_idx == 0 and not meta.extra and meta.total_frames == 0:
        preview = pd.read_csv(path, header=None, nrows=120, encoding="utf-8-sig")
        header_idx = _find_header_row_index(preview)
        columns = [str(c).strip() for c in preview.iloc[header_idx].tolist()]
        return header_idx, meta, columns

    columns = _parse_csv_fields(header_line)
    return header_line_idx, meta, columns


def _read_table(path: Path, header_idx: int, columns: list[str]) -> pd.DataFrame:
    """只读取时间列和通道列，减少 IO 与内存占用。"""
    time_col = _find_time_column(columns)
    if time_col is None:
        raise ValueError("未找到时间列（支持: 时间/time/t/timestamp 等）")

    channel_cols = _detect_channel_columns(columns, time_col)
    if not channel_cols:
        raise ValueError(
            "未找到通道列（支持: ch1/ch2...、fx/fy/fz/mx/my/mz，不区分大小写）"
        )

    usecols = [time_col, *channel_cols]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(
            path,
            skiprows=header_idx,
            usecols=usecols,
            encoding="utf-8-sig",
            engine="c",
        )

    raw = pd.read_excel(path, header=None)
    if raw.empty:
        raise ValueError("Excel 文件为空")
    df = _build_dataframe_from_raw(raw, header_idx)
    return df[usecols]


def _load_csv(path: Path) -> Recording:
    header_idx, meta, columns = _scan_csv_header(path)

    if header_idx == 0 and not meta.extra and meta.total_frames == 0:
        raw = pd.read_csv(path, header=None, encoding="utf-8-sig", engine="c")
        header_idx = _find_header_row_index(raw)
        meta = _parse_meta_rows([[str(v) for v in raw.iloc[i].tolist()] for i in range(header_idx)])
        columns = [str(c).strip() for c in raw.iloc[header_idx].tolist()]
        time_col = _find_time_column(columns)
        channel_cols = _detect_channel_columns(columns, time_col)
        if time_col is None or not channel_cols:
            raise ValueError("未找到有效的时间列或通道列")
        col_idx = [columns.index(time_col), *[columns.index(c) for c in channel_cols]]
        data = raw.iloc[header_idx + 1 :, col_idx].copy()
        data.columns = [time_col, *channel_cols]
        df = data.reset_index(drop=True)
    else:
        df = _read_table(path, header_idx, columns)

    return _finalize_recording(path, meta, df)


def _load_xlsx(path: Path) -> Recording:
    raw = pd.read_excel(path, header=None)
    if raw.empty:
        raise ValueError("Excel 文件为空")

    header_idx = _find_header_row_index(raw)
    columns = [str(c).strip() for c in raw.iloc[header_idx].tolist()]
    time_col = _find_time_column(columns)
    channel_cols = _detect_channel_columns(columns, time_col)
    if time_col is None or not channel_cols:
        raise ValueError("未找到有效的时间列或通道列")

    meta = _parse_meta_rows([[str(v) for v in raw.iloc[i].tolist()] for i in range(header_idx)])
    df = _build_dataframe_from_raw(raw, header_idx)
    df = df[[time_col, *channel_cols]]
    return _finalize_recording(path, meta, df)


def load_recording(path: str | Path) -> Recording:
    """加载 CSV 或 XLSX。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return _load_xlsx(path)
    if suffix == ".xls":
        raise ValueError("不支持旧版 .xls 格式，请在 Excel 中另存为 .xlsx 后重试")
    raise ValueError(f"不支持的文件格式: {suffix}，请使用 CSV 或 XLSX")
