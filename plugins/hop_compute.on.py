
# plugins/hop_compute.on.py
# -*- coding: utf-8 -*-
"""
Hop 状态计算插件（教学示例）
=========================

本插件演示“计算类”插件的编写方法：
- 插件的 `scope` 可为 "Trajectory-All"（轨迹级）。
- 主程序会将用户输入的参数（如最大时间）传入插件；插件可遍历所有轨迹，**原地更新**元数据或表。
- 返回过程日志，便于用户查看处理进度。

本插件逻辑：
- 读取每条轨迹的时间列（优先 `t`，其次 `time`）与 `state` 列。
- 计算 Hop 情况：
  - `t1` = 最早从 state=2 跳到 state=1 的时间；
  - 若轨迹最大时间 `< 用户提供的最大时间`，则取 `t2 = 轨迹最大时间`；
  - 输出格式：`(t1,-t2)` 或 `t1` 或 `-t2` 或 `None`，写入 `traj.meta["hop_info"]`；
- 将 `hop_info` 追加为任务列表显示字段（如果尚未追加）。

你可以在此基础上改造业务逻辑或输出格式。
"""

from typing import Any, List

# 兜底导入（主程序会注入）
try:
    Trajectory  # type: ignore
    SimpleTable  # type: ignore
except NameError:
    import sys as _sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(__file__))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from md_manager import Trajectory, SimpleTable  # type: ignore

def _to_float(v):
    try: return float(v)
    except Exception: return None

def _to_int(v):
    try: return int(str(v).strip())
    except Exception: return None

def fmt_t8(x: float) -> str:
    if x is None: return ""
    return f"{x:.8f}"

def _find_time_column(traj):
    cols = traj.table.columns
    if "t" in cols: return "t"
    if "time" in cols: return "time"
    return None

def run_hop(task, args):
    """
    计算 Hop 情况。用户输入最大时间（通过 line 模式或 form 模式传参）。

    参数：
      - task: 主程序任务对象
      - args: {"__raw__": "100"} 或 {"max_time": "100"}

    返回：
      - {"process": ["..."]} 日志
    """
    raw = args.get("__raw__", "").strip()
    max_time = _to_float(raw) if raw else _to_float(args.get("max_time"))
    if max_time is None:
        raise ValueError("请输入/提供最大时间")

    proc: List[str] = []
    for tid, traj in task.trajectories.items():
        tcol = _find_time_column(traj)
        scol = "state" if "state" in traj.table.columns else None
        if tcol is None or scol is None:
            traj.meta["hop_info"] = None
            proc.append(f"[Hop计算] 轨迹ID：{tid}｜缺列：{tcol or 't'} / {scol or 'state'}")
            continue

        # 最早 2->1
        t1 = None; prev_state = None
        for r in traj.table.rows:
            cur_state = _to_int(r.get(scol))
            cur_t = _to_float(r.get(tcol))
            if prev_state == 2 and cur_state == 1 and t1 is None:
                t1 = cur_t; break
            prev_state = cur_state

        # 最大时间
        times: List[float] = []
        for r in traj.table.rows:
            tv = _to_float(r.get(tcol))
            if tv is not None:
                times.append(tv)
        t_max_val = max(times) if times else None

        t2 = None
        if (t_max_val is not None) and (max_time is not None) and (t_max_val < max_time):
            t2 = t_max_val

        if t1 is not None and t2 is not None:
            hop_info = f"({fmt_t8(t1)},-{fmt_t8(t2)})"
        elif t1 is not None:
            hop_info = fmt_t8(t1)
        elif t2 is not None:
            hop_info = f"-{fmt_t8(t2)}"
        else:
            hop_info = None

        traj.meta["hop_info"] = hop_info
        fields = task.settings.get("list_fields", [])
        if "hop_info" not in fields:
            fields.append("hop_info")
            task.settings["list_fields"] = fields
        proc.append(f"[Hop计算] 轨迹ID：{tid}｜hop_info：{hop_info}")
    return {"process": proc}

PLUGINS = [
    {
        "name": "Hop情况",
        "description": "计算 hop_info（时间8位小数）",
        "scope": "Trajectory-All",
        "run": run_hop,
        "input": {"mode": "line", "help": "直接输入最大时间", "example": "100"},
    }
]
