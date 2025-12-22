
# plugins/hop_import.on.py
# -*- coding: utf-8 -*-
"""
Hop 导入插件（教学示例）
=====================

本文件展示如何编写一个“导入类”插件：
- 插件的 `scope` 设为 "Import"。
- 主程序会按正则匹配子文件夹，依次把每个匹配到的文件夹通过 `args={"folder": <路径>}` 传给插件。
- 插件可直接拿到 `task` 引用，**原地更新**轨迹表与元数据（无需返回数据）。
- 插件应返回过程日志：`{"process": ["..."]}`，主程序会展示。

实现要点：
- 使用 `Trajectory` 与 `SimpleTable`（主程序在加载插件时已注入它们；此文件也提供兜底导入）。
- 统一格式策略：时间保留 8 位小数；整数类（step/state）按整数；其余数据按业务需要。
- 与其它导入插件协作：在同一轨迹（按 `source_folder`）上按 `step` 合并行。

编写插件的规范：
1) 顶部准备工具函数（类型转换、格式化）。
2) 编写 `_get_or_create_traj_for_folder`：以文件夹路径为键获取/创建轨迹。
3) 编写解析函数 `_parse_hop_all_time(path)`：读取目标文件得到结构化数据。
4) 在 `run_import_hop(task, args)` 中：
   - 校验 `folder`；
   - 如果文件存在，解析并将数据按 `step` 合并到轨迹表；
   - 计算时间统计（t_min/t_max/duration），写入 `traj.meta`；
   - 返回过程日志。

你可以以此为模板创建更多导入插件。
"""

import os
import re
from typing import Dict, Any, List, Tuple

# 兜底导入（主程序会注入，静态检查或独立运行时可用）
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

def fmt_int(v) -> int:
    iv = _to_int(v)
    return iv if iv is not None else None

def _get_or_create_traj_for_folder(task, folder: str, suggested_name: str = None):
    ab = os.path.abspath(folder)
    for traj in task.trajectories.values():
        if str(traj.meta.get("source_folder")) == ab:
            return traj
    tid = task.next_traj_id()
    name = suggested_name or os.path.basename(ab) or f"traj_{tid}"
    new_traj = Trajectory(tid, name, SimpleTable(columns=[], rows=[]), meta={"source_folder": ab, "traj_seq": int(tid)})
    task.add_trajectory(new_traj)
    return new_traj

def _parse_hop_all_time(path: str) -> Tuple[Dict[int, Tuple[float, int]], List[int]]:
    hop: Dict[int, Tuple[float, int]] = {}
    steps: List[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r'^\s*(\d+)\s+([0-9\.\-E]+)\s+current state\s+(\d+)\s*$', line)
            if m:
                step = int(m.group(1))
                t_val = _to_float(m.group(2))
                state = int(m.group(3))
                hop[step] = (t_val, state)
                steps.append(step)
    steps = sorted(set(steps))
    return hop, steps

def run_import_hop(task, args):
    """
    导入 Hop 数据：读取 hop_all_time.out 并按 step 合并到轨迹表。

    参数：
      - task: 主程序任务对象（完整引用，可原地修改）
      - args: {"folder": 匹配到的子目录路径}

    返回：
      - {"process": ["日志1", "日志2", ...]} 用于主程序输出
    """
    folder = args.get("folder")
    if not folder or not os.path.isdir(folder): raise ValueError("folder 无效")
    path = os.path.join(folder, "hop_all_time.out")
    proc: List[str] = []
    if not os.path.isfile(path):
        proc.append(f"[Hop] 未发现：{path}")
        return {"process": proc}

    hop, steps = _parse_hop_all_time(path)
    traj = _get_or_create_traj_for_folder(task, folder, suggested_name=os.path.basename(folder))

    rows_by_step: Dict[int, Dict[str, Any]] = { _to_int(r.get("step")): r for r in traj.table.rows if _to_int(r.get("step")) is not None }
    cols_set = set(traj.table.columns); cols_set.update({"step", "t", "state"})

    times: List[float] = []
    for s in steps:
        (t_val, state) = hop.get(s, (None, None))
        row = rows_by_step.get(s)
        if row is None:
            row = {"step": s}; traj.table.rows.append(row); rows_by_step[s] = row
        row["t"] = fmt_t8(t_val) if t_val is not None else ""
        row["state"] = fmt_int(state)
        if t_val is not None:
            times.append(t_val)

    traj.table.columns = list(cols_set)
    if times:
        tmin = min(times); tmax = max(times); dur = tmax - tmin
        traj.meta["t_min"] = fmt_t8(tmin)
        traj.meta["t_max"] = fmt_t8(tmax)
        traj.meta["duration"] = fmt_t8(dur)
    traj.meta["n_frames_hop"] = len(steps)
    traj.refresh_basic_meta()

    proc.append(f"[Hop] 文件夹：{os.path.abspath(folder)}｜轨迹ID：{traj.traj_id}｜帧(Hop)：{len(steps)}")
    return {"process": proc}

PLUGINS = [
    {
        "name": "导入-Hop",
        "description": "读取 hop_all_time.out（单文件夹）",
        "scope": "Import",
        "run": run_import_hop,
        "input": {"mode": "line", "help": "无需输入，主程序传入 folder", "example": ""},
    }
]