# -*- coding: utf-8 -*-
"""
时刻聚合：状态计数
----------------
统计所有轨迹在每个时刻 state=2 与 state=1 的数量，并写入任务的时刻表。
输入：无需参数，默认使用列 `state` 及时间列 `t`（或 `time`）。
输出：在 task.time_table 中新增列 `count_state2`, `count_state1`。
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


def _to_int(v):
    try:
        return int(str(v).strip())
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


def run_time_state_counts(task, args):
    """聚合 state=2/1 计数，写入时刻表。"""

    tt = _ensure_time_table(task)
    bucket: Dict[str, List[int]] = {}
    proc: List[str] = []
    trajs = sorted(task.trajectories.values(), key=lambda t: int(t.traj_id))
    if not trajs:
        return {"process": ["无轨迹可聚合"], "time_table_updated": False}

    for traj in trajs:
        tcol = _find_time_column(traj)
        if tcol is None or "state" not in (traj.table.columns or []):
            proc.append(
                f"[state计数] 轨迹ID {traj.traj_id} 缺少 t/time 或 state，已跳过"
            )
            continue
        for r in traj.table.rows:
            tf = _to_float(r.get(tcol))
            sv = _to_int(r.get("state"))
            if tf is None or sv is None:
                continue
            key = str(tf)
            bucket.setdefault(key, []).append(sv)

    if not bucket:
        proc.append("[state计数] 未收集到有效数据")
        return {"process": proc, "time_table_updated": False}

    idx: Dict[str, Dict[str, Any]] = {str(r.get("t")): r for r in tt.rows}
    for k, states in bucket.items():
        row = idx.get(k)
        if row is None:
            row = {"t": float(k)}
            tt.rows.append(row)
            idx[k] = row
        row["count_state2"] = sum(1 for s in states if s == 2)
        row["count_state1"] = sum(1 for s in states if s == 1)

    cols_keep = [c for c in tt.columns if c != "t"]
    for c in ("count_state2", "count_state1"):
        if c not in cols_keep:
            cols_keep.append(c)
    tt.columns = ["t"] + cols_keep
    _sort_time_rows(tt.rows)

    proc.append(
        f"[state计数] 已写入 {len(bucket)} 个时刻的计数列：count_state2, count_state1"
    )
    return {"process": proc, "time_table_updated": True}


PLUGINS = [
    {
        "name": "时刻状态计数",
        "description": "跨轨迹统计每时刻 state=2/1 数量，写入时刻表",
        "scope": "Time-Series",
        "run": run_time_state_counts,
        "input": {"mode": "line", "help": "无需输入，需存在 state 列", "example": ""},
    }
]
