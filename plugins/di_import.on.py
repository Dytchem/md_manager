# plugins/di_import.on.py
# -*- coding: utf-8 -*-
"""
Dipole & Mulliken 导入插件
==========================

本插件用于读取 di_time.out 文件，提取：
- Mulliken 电荷（每个原子）
- Dipole 矩阵元素（DMX、DMY、DMZ 的 au 和 Debye 单位）

数据按 step 合并到轨迹表中，与其它导入插件协作。
"""

import os
import re
from typing import Any, Dict, List, Tuple

# 兜底导入（主程序会注入，静态检查或独立运行时可用）
try:
    Trajectory  # type: ignore
    SimpleTable  # type: ignore
except NameError:
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


def fmt_t8(x: float) -> str:
    if x is None:
        return ""
    return f"{x:.8f}"


def fmt_f10(x: float) -> str:
    if x is None:
        return ""
    return f"{x:.10f}"


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
    new_traj = Trajectory(
        tid,
        name,
        SimpleTable(columns=[], rows=[]),
        meta={"source_folder": ab, "traj_seq": int(tid)},
    )
    task.add_trajectory(new_traj)
    return new_traj


def _parse_di_time(path: str) -> Tuple[Dict[int, Dict[str, Any]], List[int], int]:
    """
    解析 di_time.out 文件。

    返回：
        - di_data: {step: {数据字典}}
        - steps: 排序后的 step 列表
        - n_atoms: 原子数量
    """
    di_data: Dict[int, Dict[str, Any]] = {}
    steps: List[int] = []
    n_atoms = 0

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 查找 STEP 标记
        m_step = re.match(r"^STEP\s+(\d+)", line)
        if m_step:
            step = int(m_step.group(1))
            i += 1

            # 跳过分隔线
            while i < len(lines) and lines[i].strip().startswith("--"):
                i += 1

            # 读取 i_time
            if i < len(lines):
                m_itime = re.match(r"^i_time\s*=\s*(\d+)", lines[i].strip())
                if m_itime:
                    i_time = int(m_itime.group(1))
                    i += 1

            data = {"step": step, "i_time": i_time if "i_time" in locals() else None}

            # 跳过空行
            while i < len(lines) and not lines[i].strip():
                i += 1

            # 读取 Mulliken Charge
            if i < len(lines) and "Mulliken Charge" in lines[i]:
                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        i += 1
                        continue
                    if line.startswith("Dipole"):
                        break
                    m_state = re.match(r"^State\s+(\d+)\.(\d+)", line)
                    if m_state:
                        state = f"{m_state.group(1)}.{m_state.group(2)}"
                        i += 1
                        atom_charges = {}
                        while i < len(lines):
                            line = lines[i].strip()
                            if not line or re.match(r"^State\s+\d+\.\d+", line) or line.startswith("Dipole"):
                                break
                            parts = line.split()
                            if len(parts) >= 2 and parts[0].isdigit():
                                atom_idx = int(parts[0])
                                charge = _to_float(parts[1])
                                atom_charges[atom_idx] = charge
                                n_atoms = max(n_atoms, atom_idx)
                            i += 1
                        for atom_idx, charge in atom_charges.items():
                            data[f"mulliken_{state}_{atom_idx}"] = charge
                    else:
                        i += 1

            # 跳过空行
            while i < len(lines) and not lines[i].strip():
                i += 1

            # 读取 Dipole
            if i < len(lines) and "Dipole" in lines[i]:
                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line or line.startswith("--") or line.startswith("STEP"):
                        break

                    # 匹配格式：<state1.sublevel1|component|state2.sublevel2>  au_value  debye_value
                    m_dipole = re.match(
                        r"^<(\d+)\.(\d+)\|([A-Z]+)\|(\d+)\.(\d+)>\s+([0-9.\-E]+)\s+([0-9.\-E]+)",
                        line,
                    )
                    if m_dipole:
                        s1, sub1, comp, s2, sub2 = m_dipole.group(1, 2, 3, 4, 5)
                        au_val = _to_float(m_dipole.group(6))
                        debye_val = _to_float(m_dipole.group(7))

                        # 列名格式：dipole_1.1_DMX_1.1_au 和 dipole_1.1_DMX_1.1_debye
                        key_base = f"dipole_{s1}.{sub1}_{comp}_{s2}.{sub2}"
                        data[f"{key_base}_au"] = au_val
                        data[f"{key_base}_debye"] = debye_val
                    i += 1

            di_data[step] = data
            steps.append(step)
        else:
            i += 1

    steps = sorted(set(steps))
    return di_data, steps, n_atoms


def run_import_di(task, args):
    """
    导入 Dipole & Mulliken 数据：读取 di_time.out 并按 step 合并到轨迹表。

    参数：
      - task: 主程序任务对象
      - args: {"folder": 匹配到的子目录路径}

    返回：
      - {"process": ["日志..."]}
    """
    folder = args.get("folder")
    if not folder or not os.path.isdir(folder):
        raise ValueError("folder 无效")
    path = os.path.join(folder, "di_time.out")
    proc: List[str] = []
    if not os.path.isfile(path):
        proc.append(f"[DI] 未发现：{path}")
        return {"process": proc}

    di_data, steps, n_atoms = _parse_di_time(path)
    traj = _get_or_create_traj_for_folder(
        task, folder, suggested_name=os.path.basename(folder)
    )

    # 按 step 构建行索引
    rows_by_step: Dict[int, Dict[str, Any]] = {
        _to_int(r.get("step")): r
        for r in traj.table.rows
        if _to_int(r.get("step")) is not None
    }

    cols_set = set(traj.table.columns)
    cols_set.add("step")

    # 添加所有可能的列
    for s in steps:
        data = di_data.get(s, {})
        for key in data.keys():
            if key not in ("step", "i_time"):
                cols_set.add(key)

    # 合并数据
    for s in steps:
        data = di_data.get(s, {})
        row = rows_by_step.get(s)
        if row is None:
            row = {"step": s}
            traj.table.rows.append(row)
            rows_by_step[s] = row

        # 写入所有字段
        for key, value in data.items():
            if key == "step":
                continue
            elif key == "i_time":
                row[key] = fmt_int(value)
            elif key.startswith("mulliken_"):
                row[key] = fmt_f10(value) if value is not None else None
            elif "_au" in key or "_debye" in key:
                row[key] = fmt_f10(value) if value is not None else None

    traj.table.columns = list(cols_set)
    traj.meta["n_atoms"] = max(traj.meta.get("n_atoms") or 0, n_atoms)
    traj.meta["n_frames_di"] = len(steps)
    traj.refresh_basic_meta()

    proc.append(
        f"[DI] 文件夹：{os.path.abspath(folder)}｜轨迹ID：{traj.traj_id}｜帧(DI)：{len(steps)}｜原子：{n_atoms}"
    )
    return {"process": proc}


PLUGINS = [
    {
        "name": "导入-DI",
        "description": "读取 di_time.out（Mulliken电荷 & Dipole分量）",
        "scope": "Import",
        "run": run_import_di,
        "input": {"mode": "line", "help": "无需输入，主程序传入 folder", "example": ""},
    }
]
