# -*- coding: utf-8 -*-
"""
Hop 二面角全局判断插件
-------------------

对每条轨迹：
- 解析 traj.meta['hop_info'] 的正数 t1；
- 在时间列（t 或 time）定位 t=t1 的行及“下一时刻”；
- 读取用户指定的二面角列的值，得到 d1, d2；
- 若 abs(d1) < abs(d2) -> 标记 1；否则 0；若无正数 t1 -> 标记 -1；
最后把 (count_1, count_0, count_-1) 写入任务元数据。
"""

import re
from typing import Any, Dict, List, Optional

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
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


def fmt_f10(x: float) -> str:
    if x is None:
        return ""
    return f"{x:.10f}"


def _find_time_column(traj) -> Optional[str]:
    cols = traj.table.columns or []
    if "t" in cols:
        return "t"
    if "time" in cols:
        return "time"
    return None


def _parse_hop_t1(hop_info: Any) -> Optional[float]:
    """
    支持几种格式：
      - "(t1,-t2)" -> 返回正数 t1
      - "t1"       -> 返回正数 t1
      - "-t2" 或 None -> 返回 None
    """
    if hop_info is None:
        return None
    s = str(hop_info).strip()
    if not s:
        return None
    m = re.match(r"^\(\s*([+-]?\d+(?:\.\d+)?)\s*,", s)
    if m:
        val = _to_float(m.group(1))
        return val if (val is not None and val > 0) else None
    m2 = re.match(r"^([+-]?\d+(?:\.\d+)?)$", s)
    if m2:
        val = _to_float(m2.group(1))
        return val if (val is not None and val > 0) else None
    return None


def _nearest_index(times: List[float], target: float) -> Optional[int]:
    """在 times 中找最接近 target 的索引（线性遍历）。"""
    if not times:
        return None
    best_i, best_d = None, None
    for i, t in enumerate(times):
        d = abs(t - target)
        if best_d is None or d < best_d:
            best_d, best_i = d, i
    return best_i


def run_hop_dihedral_global(task, args):
    """
    读取 hop_info 的正数 t1，比较二面角在 t1 及下一时刻的绝对值大小。
    **输入（line 模式）**：
      1) 直接两项：<二面角列> <标记字段>
         例：dihedral_1_2_3_4 hop_dihedral_is_increase
      2) 或使用 key=value：dihedral=<列名> out=<标记字段> [tol=<时间匹配公差>]
         tol 可省略（默认 1e-8）

    返回：
      - 每轨迹在 traj.meta[out] 写入 1/0/-1
      - 在 task.meta 记录统计元组：out__count_tuple = (count_1, count_0, count_-1)
      - process 日志逐轨输出与汇总
    """
    raw = (args.get("__raw__") or "").strip()
    if raw.lower() in ("q", "quit", "exit"):
        return {"process": ["[Hop二面角] 已取消"]}

    # 参数解析：优先解析“直接两项”，其次解析 key=value；均未给出则尝试 form 键
    dih_col = None
    out_name = None
    tol = 1e-8  # 默认即可，不必输入

    if raw:
        toks = [t for t in raw.split() if t.strip()]
        if len(toks) >= 2:
            dih_col = toks[0].strip()
            out_name = toks[1].strip()
            # 如用户附加 tol，可第三项写：tol 或 tol=数值
            if len(toks) >= 3:
                # 支持 "tol" 或 "tol=<value>" 或纯数值第三项
                if toks[2].lower() == "tol" and len(toks) >= 4:
                    tv = _to_float(toks[3])
                    if tv is not None:
                        tol = tv
                else:
                    m_tol = re.match(r"^tol\s*=\s*([^\s]+)$", toks[2], re.I)
                    if m_tol:
                        tv = _to_float(m_tol.group(1))
                        if tv is not None:
                            tol = tv
                    else:
                        # 若第三项就是数值，也接受
                        tv = _to_float(toks[2])
                        if tv is not None:
                            tol = tv

        # key=value 覆盖（若用户使用该写法）
        kvs = dict(re.findall(r"(\w+)\s*=\s*([^\s]+)", raw))
        if "dihedral" in kvs:
            dih_col = kvs["dihedral"].strip()
        if "out" in kvs:
            out_name = kvs["out"].strip()
        if "tol" in kvs:
            tv = _to_float(kvs["tol"])
            if tv is not None:
                tol = tv
    else:
        # 兼容 form 传参
        dih_col = (args.get("dihedral") or "").strip()
        out_name = (args.get("out") or "").strip()
        tv = _to_float(args.get("tol"))
        if tv is not None:
            tol = tv

    if not dih_col or not out_name:
        raise ValueError(
            "缺少参数：二面角列 或 标记字段（示例：dihedral_1_2_3_4 hop_dihedral_is_increase）"
        )

    # 统计计数
    c_pos = c_zero = c_neg = 0
    proc: List[str] = []

    for tid, traj in task.trajectories.items():
        tcol = _find_time_column(traj)
        hop_t1 = _parse_hop_t1(traj.meta.get("hop_info"))

        # hop_info 未给正数 t1 -> -1
        if hop_t1 is None or tcol is None:
            traj.meta[out_name] = -1
            c_neg += 1
            proc.append(f"[Hop二面角] 轨迹ID：{tid}｜标记：-1（缺 t1 或时间列）")
            continue

        # 准备时间序列（按当前行顺序）
        times_raw: List[Optional[float]] = [
            _to_float(r.get(tcol)) for r in traj.table.rows
        ]
        valid_map = [(i, t) for i, t in enumerate(times_raw) if t is not None]
        if not valid_map:
            traj.meta[out_name] = 0
            c_zero += 1
            proc.append(f"[Hop二面角] 轨迹ID：{tid}｜标记：0（无有效时间值）")
            continue

        # 在有效时间里找到最接近 hop_t1 的索引
        idx_valid = _nearest_index([t for _, t in valid_map], hop_t1)
        row_i = valid_map[idx_valid][0] if idx_valid is not None else None
        next_i = (
            (row_i + 1)
            if (row_i is not None and (row_i + 1) < len(traj.table.rows))
            else None
        )

        # 提取二面角 d1, d2
        r1 = traj.table.rows[row_i] if row_i is not None else None
        d1 = _to_float(r1.get(dih_col)) if r1 is not None else None
        d2 = None
        if next_i is not None:
            r2 = traj.table.rows[next_i]
            d2 = _to_float(r2.get(dih_col))

        # 判断与写入
        if (d1 is not None) and (d2 is not None):
            mark = 1 if abs(d1) < abs(d2) else 0
        else:
            mark = 0  # 缺值视为非增加

        traj.meta[out_name] = mark
        if mark == 1:
            c_pos += 1
        else:
            c_zero += 1

        proc.append(
            f"[Hop二面角] 轨迹ID：{tid}｜t1={fmt_f10(hop_t1)}｜d1={fmt_f10(d1)}｜d2={fmt_f10(d2)}｜标记：{mark}"
        )

    # 汇总写入任务元数据：元组 (count_1, count_0, count_-1)
    task.meta[f"{out_name}__count_tuple"] = (c_pos, c_zero, c_neg)
    proc.append(
        f"[Hop二面角] 汇总：{out_name}__count_tuple = ({c_pos}, {c_zero}, {c_neg})"
    )

    return {"process": proc}


PLUGINS = [
    {
        "name": "Hop二面角（全局）",
        "description": "读取 hop_info 的 t1，对比二面角在 t1 与下一时刻的绝对值大小，并统计 1/0/-1 数量",
        "scope": "Task-Global",
        "run": run_hop_dihedral_global,
        "input": {
            "mode": "line",
            "help": "直接输入：<二面角列> <标记字段>（tol 默认 1e-8，可忽略）；或 key=value 形式",
            "example": "dihedral_1_2_3_4 hop_dihedral_is_increase",
        },
    }
]
