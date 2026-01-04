# -*- coding: utf-8 -*-
"""
时刻聚合：二面角阈值计数
----------------------
按时刻跨轨迹统计指定二面角列的绝对值：
- abs(dihedral) < pi/2 的数量
- abs(dihedral) > pi/2 的数量
写入任务的时刻表。
输入：line 模式填写二面角列名，如 `dihedral_1_2_3_4`。
输出：新增列 `cnt_<col>_abs_lt_pi2`, `cnt_<col>_abs_gt_pi2`。
"""

import math
from typing import Any, Dict, List

# 兜底导入（主程序会注入 Trajectory/SimpleTable）
try:
    Trajectory  # type: ignore
    SimpleTable  # type: ignore
except NameError:  # pragma: no cover - 静态检查/独立运行
    from md_modules.core import SimpleTable, Trajectory  # type: ignore


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _find_time_column(traj):
    cols = traj.table.columns or []
    if "t" in cols:
        return "t"
    if "time" in cols:
        return "time"
    return None


def _ensure_time_table(task):
    try:
        tt = getattr(task, "time_table", None)
        if not isinstance(tt, SimpleTable):
            tt = SimpleTable(["t"], [])
            task.time_table = tt
    except Exception:
        tt = SimpleTable(["t"], [])
        task.time_table = tt
    return tt


def _sort_time_rows(rows: List[Dict[str, Any]]):
    def as_float(v):
        try:
            return float(v)
        except Exception:
            return None

    rows.sort(
        key=lambda r: (
            as_float(r.get("t")) is None,
            (
                as_float(r.get("t"))
                if as_float(r.get("t")) is not None
                else str(r.get("t"))
            ),
        )
    )


def run_time_dihedral_counts(task, args):
    """聚合二面角绝对值阈值计数，写入时刻表。"""

    col = None
    if isinstance(args, dict):
        raw = (args.get("__raw__") or "").strip()
        col = raw or args.get("col") or args.get("column") or args.get("dihedral")
        if isinstance(col, str):
            col = col.strip()
    if not col:
        return {"process": ["缺少二面角列名"]}

    tt = _ensure_time_table(task)
    bucket: Dict[str, List[float]] = {}
    proc: List[str] = []
    trajs = sorted(task.trajectories.values(), key=lambda t: int(t.traj_id))
    if not trajs:
        return {"process": ["无轨迹可聚合"], "time_table_updated": False}

    for traj in trajs:
        tcol = _find_time_column(traj)
        if tcol is None or col not in (traj.table.columns or []):
            proc.append(
                f"[二面角计数] 轨迹ID {traj.traj_id} 缺少 t/time 或 {col}，已跳过"
            )
            continue
        for r in traj.table.rows:
            tf = _to_float(r.get(tcol))
            dv = _to_float(r.get(col))
            if tf is None or dv is None:
                continue
            key = str(tf)
            bucket.setdefault(key, []).append(dv)

    if not bucket:
        proc.append("[二面角计数] 未收集到有效数据")
        return {"process": proc, "time_table_updated": False}

    lt_col = f"cnt_{col}_abs_lt_pi2"
    gt_col = f"cnt_{col}_abs_gt_pi2"
    threshold = math.pi / 2.0

    idx: Dict[str, Dict[str, Any]] = {str(r.get("t")): r for r in tt.rows}
    for k, vals in bucket.items():
        row = idx.get(k)
        if row is None:
            row = {"t": float(k)}
            tt.rows.append(row)
            idx[k] = row
        row[lt_col] = sum(1 for v in vals if abs(v) < threshold)
        row[gt_col] = sum(1 for v in vals if abs(v) > threshold)

    cols_keep = [c for c in tt.columns if c != "t"]
    for c in (lt_col, gt_col):
        if c not in cols_keep:
            cols_keep.append(c)
    tt.columns = ["t"] + cols_keep
    _sort_time_rows(tt.rows)

    proc.append(f"[二面角计数] 已写入 {len(bucket)} 个时刻的计数列：{lt_col}, {gt_col}")
    return {"process": proc, "time_table_updated": True}


PLUGINS = [
    {
        "name": "时刻二面角计数",
        "description": "按时刻统计 |dihedral|<pi/2 与 >pi/2 数量",
        "scope": "Time-Series",
        "run": run_time_dihedral_counts,
        "input": {
            "mode": "line",
            "help": "输入二面角列名，如 dihedral_1_2_3_4",
            "example": "dihedral_1_2_3_4",
        },
    }
]
