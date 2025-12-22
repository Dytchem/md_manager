
# plugins/base.on.py
# -*- coding: utf-8 -*-

import os
import re
import math
from typing import Dict, Any, List, Tuple

# 主程序在加载时注入 Trajectory/SimpleTable；兜底导入用于静态检查或独立运行
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

def fmt_f10(x: float) -> str:
    if x is None: return ""
    return f"{x:.10f}"

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

# -------- 导入：位置 --------

def _parse_positions_traj_time(path: str) -> Tuple[Dict[int, List[Tuple[str, float, float, float]]], List[str], int, List[int]]:
    pos_data: Dict[int, List[Tuple[str, float, float, float]]] = {}
    symbols_first = None
    n_atoms = 0
    steps: List[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^(\d+)$', line)
        if not m: i += 1; continue
        n = int(m.group(1)); n_atoms = n
        i += 1
        if i >= len(lines): break
        header = lines[i]
        ms = re.search(r'Step:\s*(\d+)', header)
        step = int(ms.group(1)) if ms else None
        lst: List[Tuple[str, float, float, float]] = []
        for _ in range(n):
            i += 1
            if i >= len(lines): break
            parts = lines[i].split()
            if len(parts) >= 4:
                sym = parts[0]
                x = _to_float(parts[1]); y = _to_float(parts[2]); z = _to_float(parts[3])
                lst.append((sym, x, y, z))
        if step is not None:
            pos_data[step] = lst
            steps.append(step)
            if symbols_first is None:
                symbols_first = [s for (s, _, _, _) in lst]
        i += 1
    steps = sorted(set(steps))
    return pos_data, (symbols_first or []), n_atoms, steps

def run_import_positions(task, args):
    folder = args.get("folder")
    if not folder or not os.path.isdir(folder): raise ValueError("folder 无效")
    path = os.path.join(folder, "traj_time.out")
    proc: List[str] = []
    if not os.path.isfile(path):
        proc.append(f"[位置] 未发现：{path}")
        return {"process": proc}
    pos, symbols, n_atoms, steps = _parse_positions_traj_time(path)
    traj = _get_or_create_traj_for_folder(task, folder, suggested_name=os.path.basename(folder))
    rows_by_step: Dict[int, Dict[str, Any]] = { _to_int(r.get("step")): r for r in traj.table.rows if _to_int(r.get("step")) is not None }
    cols_set = set(traj.table.columns); cols_set.add("step")
    for i in range(1, n_atoms + 1):
        cols_set.update({f"symbol_{i}", f"x_{i}", f"y_{i}", f"z_{i}"})
    for s in steps:
        lst = pos.get(s, [])
        row = rows_by_step.get(s)
        if row is None:
            row = {"step": s}; traj.table.rows.append(row); rows_by_step[s] = row
        for idx, (sym, x, y, z) in enumerate(lst, start=1):
            row[f"symbol_{idx}"] = sym
            row[f"x_{idx}"] = fmt_f10(x)
            row[f"y_{idx}"] = fmt_f10(y)
            row[f"z_{idx}"] = fmt_f10(z)
    traj.table.columns = list(cols_set)
    traj.meta["symbols"] = symbols or []
    traj.meta["n_atoms"] = n_atoms
    traj.meta["n_frames_pos"] = len(steps)
    traj.meta["unit_pos"] = "ANG"
    traj.refresh_basic_meta()
    proc.append(f"[位置] 文件夹：{os.path.abspath(folder)}｜轨迹ID：{traj.traj_id}｜帧(位置)：{len(steps)}｜原子：{n_atoms}")
    return {"process": proc}

# -------- 导入：速度 --------

def _parse_velocities_vel_time(path: str) -> Tuple[Dict[int, List[Tuple[float, float, float]]], int, List[int]]:
    vel_data: Dict[int, List[Tuple[float, float, float]]] = {}
    n_atoms = 0; steps: List[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^(\d+)$', line)
        if not m: i += 1; continue
        n = int(m.group(1)); n_atoms = n
        i += 1
        if i >= len(lines): break
        header = lines[i]
        ms = re.search(r'Step:\s*(\d+)', header)
        step = int(ms.group(1)) if ms else None
        lst: List[Tuple[float, float, float]] = []
        for _ in range(n):
            i += 1
            if i >= len(lines): break
            parts = lines[i].split()
            if len(parts) >= 4:
                vx = _to_float(parts[1]); vy = _to_float(parts[2]); vz = _to_float(parts[3])
                lst.append((vx, vy, vz))
        if step is not None:
            vel_data[step] = lst; steps.append(step)
        i += 1
    steps = sorted(set(steps))
    return vel_data, n_atoms, steps

def run_import_velocities(task, args):
    folder = args.get("folder")
    if not folder or not os.path.isdir(folder): raise ValueError("folder 无效")
    path = os.path.join(folder, "vel_time.out")
    proc: List[str] = []
    if not os.path.isfile(path):
        proc.append(f"[速度] 未发现：{path}")
        return {"process": proc}
    vel, n_atoms, steps = _parse_velocities_vel_time(path)
    traj = _get_or_create_traj_for_folder(task, folder, suggested_name=os.path.basename(folder))
    rows_by_step: Dict[int, Dict[str, Any]] = { _to_int(r.get("step")): r for r in traj.table.rows if _to_int(r.get("step")) is not None }
    cols_set = set(traj.table.columns); cols_set.add("step")
    for i in range(1, n_atoms + 1):
        cols_set.update({f"vx_{i}", f"vy_{i}", f"vz_{i}"})
    for s in steps:
        lst = vel.get(s, [])
        row = rows_by_step.get(s)
        if row is None:
            row = {"step": s}; traj.table.rows.append(row); rows_by_step[s] = row
        for idx, (vx, vy, vz) in enumerate(lst, start=1):
            row[f"vx_{idx}"] = fmt_f10(vx)
            row[f"vy_{idx}"] = fmt_f10(vy)
            row[f"vz_{idx}"] = fmt_f10(vz)
    traj.table.columns = list(cols_set)
    traj.meta["n_frames_vel"] = len(steps)
    traj.meta["unit_vel"] = "AU"
    traj.meta["n_atoms"] = max(traj.meta.get("n_atoms") or 0, n_atoms)
    traj.refresh_basic_meta()
    proc.append(f"[速度] 文件夹：{os.path.abspath(folder)}｜轨迹ID：{traj.traj_id}｜帧(速度)：{len(steps)}｜原子：{n_atoms}")
    return {"process": proc}

# -------- 计算：几何/COM/统计/全局/表达式 --------

def _vec(r, i):
    x = _to_float(r.get(f"x_{i}")); y = _to_float(r.get(f"y_{i}")); z = _to_float(r.get(f"z_{i}"))
    return (x, y, z)

def run_geom(task, args):
    raw = args.get("__raw__", "").strip()
    if not raw:
        kind = args.get("kind"); group = args.get("group", ""); out = args.get("out", "").strip()
        raw = " ".join([kind or "", group.replace(",", " "), out]).strip()
    parts = [p for p in re.split(r'[,\s]+', raw.strip()) if p]
    if not parts: raise ValueError("命令为空")
    kind = parts[0].lower()
    if kind not in ("dist", "angle", "dihedral"): raise ValueError("类型应为 dist/angle/dihedral")
    need = 2 if kind == "dist" else (3 if kind == "angle" else 4)
    if len(parts) < (1 + need): raise ValueError("参数不足")
    nums = [int(parts[i]) for i in range(1, 1 + need)]
    out_col = parts[1 + need] if len(parts) > (1 + need) else ""
    proc: List[str] = []
    for tid, traj in task.trajectories.items():
        cols = set(traj.table.columns)
        col = out_col or (f"{kind}_" + "_".join(str(n) for n in nums))
        cols.add(col)
        for r in traj.table.rows:
            if kind == "dist":
                i, j = nums
                vi = _vec(r, i); vj = _vec(r, j)
                if None in vi or None in vj: r[col] = None
                else:
                    dx = vi[0] - vj[0]; dy = vi[1] - vj[1]; dz = vi[2] - vj[2]
                    r[col] = fmt_f10((dx*dx + dy*dy + dz*dz)**0.5)
            elif kind == "angle":
                i, j, k = nums
                vi = _vec(r, i); vj = _vec(r, j); vk = _vec(r, k)
                if None in vi or None in vj or None in vk: r[col] = None
                else:
                    vji = (vi[0]-vj[0], vi[1]-vj[1], vi[2]-vj[2])
                    vjk = (vk[0]-vj[0], vk[1]-vj[1], vk[2]-vj[2])
                    ni = (vji[0]**2 + vji[1]**2 + vji[2]**2)**0.5
                    nk = (vjk[0]**2 + vjk[1]**2 + vjk[2]**2)**0.5
                    if ni == 0 or nk == 0: r[col] = None
                    else:
                        cs = (vji[0]*vjk[0] + vji[1]*vjk[1] + vji[2]*vjk[2]) / (ni*nk)
                        cs = max(-1.0, min(1.0, cs))
                        r[col] = fmt_f10(math.acos(cs))
            else:
                i, j, k, l = nums
                ri = _vec(r, i); rj = _vec(r, j); rk = _vec(r, k); rl = _vec(r, l)
                if None in ri or None in rj or None in rk or None in rl: r[col] = None
                else:
                    b1 = (rj[0]-ri[0], rj[1]-ri[1], rj[2]-ri[2])
                    b2 = (rk[0]-rj[0], rk[1]-rj[1], rk[2]-rj[2])
                    b3 = (rl[0]-rk[0], rl[1]-rk[1], rl[2]-rk[2])
                    n1 = (b1[1]*b2[2]-b1[2]*b2[1], b1[2]*b2[0]-b1[0]*b2[2], b1[0]*b2[1]-b1[1]*b2[0])
                    n2 = (b2[1]*b3[2]-b2[2]*b3[1], b2[2]*b3[0]-b2[0]*b3[2], b2[0]*b3[1]-b2[1]*b3[0])
                    n1n = (n1[0]**2 + n1[1]**2 + n1[2]**2)**0.5
                    n2n = (n2[0]**2 + n2[1]**2 + n2[2]**2)**0.5
                    b2n = (b2[0]**2 + b2[1]**2 + b2[2]**2)**0.5
                    if n1n == 0 or n2n == 0 or b2n == 0: r[col] = None
                    else:
                        x = (n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]) / (n1n*n2n)
                        m = (n1[1]*n2[2]-n1[2]*n2[1], n1[2]*n2[0]-n1[0]*n2[2], n1[0]*n2[1]-n1[1]*n2[0])
                        y = (m[0]*b2[0] + m[1]*b2[1] + m[2]*b2[2]) / (n1n*n2n*b2n)
                        x = max(-1.0, min(1.0, x))
                        r[col] = fmt_f10(math.atan2(y, x))
        traj.table.columns = list(set(traj.table.columns) | {col})
        proc.append(f"[几何] 轨迹ID：{tid}｜生成列：{col}")
    return {"process": proc}

def run_com(task, args):
    raw = args.get("__raw__", "").strip()
    masses = None
    if raw:
        parts = raw.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "masses":
            masses = [float(x.strip()) for x in parts[1].split(",") if x.strip()]
    else:
        s = args.get("masses", "").strip()
        if s:
            masses = [float(x.strip()) for x in s.split(",") if x.strip()]
    proc: List[str] = []
    for tid, traj in task.trajectories.items():
        n = traj.meta.get("n_atoms") or 0
        mlist = masses or traj.meta.get("masses") or [1.0] * n
        cols = set(traj.table.columns); cols |= {"com_x", "com_y", "com_z"}
        for r in traj.table.rows:
            sx = sy = sz = 0.0; msum = 0.0
            for i in range(1, n + 1):
                xi = _to_float(r.get(f"x_{i}")); yi = _to_float(r.get(f"y_{i}")); zi = _to_float(r.get(f"z_{i}"))
                if None in (xi, yi, zi): continue
                mi = mlist[i - 1] if i - 1 < len(mlist) else 1.0
                sx += mi * xi; sy += mi * yi; sz += mi * zi; msum += mi
            if msum > 0:
                r["com_x"] = fmt_f10(sx / msum)
                r["com_y"] = fmt_f10(sy / msum)
                r["com_z"] = fmt_f10(sz / msum)
            else:
                r["com_x"] = r["com_y"] = r["com_z"] = None
        traj.table.columns = list(cols)
        proc.append(f"[COM] 轨迹ID：{tid}｜已计算 COM")
    return {"process": proc}

def _stats(vals):
    arr = [v for v in vals if v is not None]
    n = len(arr)
    if n == 0: return (None, None)
    mean = sum(arr) / n
    var = (sum((v - mean) * (v - mean) for v in arr) / (n - 1)) if n > 1 else None
    return (mean, var)

def run_stats(task, args):
    raw = args.get("__raw__", "").strip()
    if raw:
        parts = raw.split()
        column = parts[0]
        skip_none = parts[1].lower() not in ("false", "0", "no") if len(parts) >= 2 else True
    else:
        column = args.get("column")
        skip_none = str(args.get("skip_none", "true")).lower() != "false"
    if not column: raise ValueError("缺少列名")
    proc: List[str] = []
    list_field = f"mean__{column}"
    fields = task.settings.get("list_fields", [])
    if list_field not in fields:
        fields.append(list_field); task.settings["list_fields"] = fields
    for tid, traj in task.trajectories.items():
        if column not in traj.table.columns:
            traj.meta[list_field] = None; traj.meta[f"var__{column}"] = None
            proc.append(f"[统计] 轨迹ID：{tid}｜列不存在：{column}"); continue
        vals = [_to_float(v) for v in [r.get(column) for r in traj.table.rows]]
        if skip_none: vals = [v for v in vals if v is not None]
        mean, var = _stats(vals)
        traj.meta[list_field] = (fmt_f10(mean) if mean is not None else None)
        traj.meta[f"var__{column}"] = (fmt_f10(var) if var is not None else None)
        proc.append(f"[统计] 轨迹ID：{tid}｜列：{column}｜均值：{traj.meta[list_field]}｜方差：{traj.meta[f'var__{column}']}")
    return {"process": proc}

def run_global(task, args):
    raw = args.get("__raw__", "").strip()
    if raw:
        parts = raw.split()
        column = parts[0] if parts else None
        sync = parts[1].lower() in ("true", "1", "yes") if len(parts) >= 2 else False
        skip = True
    else:
        column = args.get("column")
        sync = str(args.get("sync_to_trajs", "false")).lower() == "true"
        skip = str(args.get("skip_none", "true")).lower() != "false"
    if not column: raise ValueError("缺少列名")
    str_list: List[Any] = []
    for _, traj in task.trajectories.items():
        if column not in traj.table.columns: continue
        for r in traj.table.rows:
            v = r.get(column)
            if v is None and skip: continue
            str_list.append(v)
    vals = [_to_float(v) for v in str_list if v is not None]
    gm = None if not vals else (sum(vals) / len(vals))
    task.meta[f"global_mean__{column}"] = (fmt_f10(gm) if gm is not None else None)
    proc: List[str] = [f"[全局] 列：{column}｜全局均值：{task.meta[f'global_mean__{column}']}"]
    if sync:
        for tid, traj in task.trajectories.items():
            traj.meta[f"global_mean__{column}"] = task.meta[f"global_mean__{column}"]
            proc.append(f"[全局] 同步到轨迹ID：{tid}")
    return {"process": proc}

SAFE = {"abs": abs, "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "pow": pow, "max": max, "min": min}

def _vals_from_traj_meta(task, key):
    xs = []
    for _, traj in task.trajectories.items():
        v = traj.meta.get(key)
        if isinstance(v, (int, float)): xs.append(float(v))
        elif isinstance(v, str):
            fv = _to_float(v)
            if fv is not None: xs.append(fv)
    return xs

def _agg(task, fname, key):
    xs = _vals_from_traj_meta(task, key)
    if not xs: return None
    if fname == "mean": return sum(xs) / len(xs)
    if fname == "sum": return sum(xs)
    if fname == "min": return min(xs)
    if fname == "max": return max(xs)
    return None

def safe_eval(expr, names, task):
    import ast
    node = ast.parse(expr, mode="eval")
    def chk(n):
        import ast
        if isinstance(n, ast.Expression): chk(n.body)
        elif isinstance(n, ast.Num): return
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)): chk(n.operand)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)): chk(n.left); chk(n.right)
        elif isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name): raise ValueError("不支持函数类型")
            fn = n.func.id
            if fn not in SAFE and fn not in ("mean", "sum", "min", "max"): raise ValueError(f"函数不允许：{fn}")
            for a in n.args: chk(a)
            if n.keywords: raise ValueError("不支持关键字参数")
        elif isinstance(n, ast.Name):
            if n.id not in names and n.id not in ("mean", "sum", "min", "max"): raise ValueError(f"未知变量：{n.id}")
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return
        else: raise ValueError("不支持表达式")
    chk(node)
    class Agg(ast.NodeTransformer):
        def visit_Call(self, call):
            import ast
            if isinstance(call.func, ast.Name) and call.func.id in ("mean", "sum", "min", "max"):
                if len(call.args) != 1 or not isinstance(call.args[0], (ast.Str, ast.Constant)):
                    raise ValueError("聚合需字符串键")
                key = call.args[0].s if isinstance(call.args[0], ast.Str) else call.args[0].value
                val = _agg(task, call.func.id, key)
                return ast.copy_location(ast.Constant(value=val), call)
            return call
    node2 = Agg().visit(node)
    ast.fix_missing_locations(node2)
    code = compile(node2, "<expr>", "eval")
    return eval(code, {"__builtins__": {}}, dict(SAFE, **names))

def run_expr(task, args):
    raw = args.get("__raw__", "").strip()
    name = None; expr = None; sync = False
    if raw:
        pairs = dict(re.findall(r'(name|expr|sync_to_trajs)\s+([^\s]+)', raw))
        name = pairs.get("name"); expr = pairs.get("expr")
        sync = (pairs.get("sync_to_trajs", "false").lower() == "true")
        kv = dict(re.findall(r'(\w+)\s*=\s*([^\s]+)', raw))
        if not name: name = kv.get("name")
        if not expr: expr = kv.get("expr")
        sync = sync or (kv.get("sync_to_trajs", "false").lower() == "true")
    else:
        name = args.get("name"); expr = args.get("expr")
        sync = str(args.get("sync_to_trajs", "false")).lower() == "true"
    if not name or not expr: raise ValueError("缺少 name 或 expr")
    names = dict(task.meta); names["traj_count"] = len(task.trajectories)
    for k, v in list(names.items()):
        if isinstance(v, str):
            fv = _to_float(v)
            if fv is not None: names[k] = fv
    val = safe_eval(expr, names, task)
    out = fmt_f10(float(val)) if isinstance(val, (int, float)) and val is not None else val
    task.meta[f"expr__{name}"] = out
    proc = [f"[表达式] 结果写入：expr__{name} = {out}"]
    if sync:
        for tid, _traj in task.trajectories.items():
            task.trajectories[tid].meta[f"expr__{name}"] = out
            proc.append(f"[表达式] 同步到轨迹ID：{tid}")
    return {"process": proc}

PLUGINS = [
    # 导入（位置、速度）——主程序会先后运行所有 Import 插件，包括 Hop 导入（在另一文件）
    {"name": "导入-位置", "description": "读取 traj_time.out（单文件夹）", "scope": "Import", "run": run_import_positions,
     "input": {"mode": "line", "help": "无需输入，主程序传入 folder", "example": ""}},
    {"name": "导入-速度", "description": "读取 vel_time.out（单文件夹）", "scope": "Import", "run": run_import_velocities,
     "input": {"mode": "line", "help": "无需输入，主程序传入 folder", "example": ""}},

    # 计算
    {"name": "几何计算", "description": "距离/角度/二面角（10位小数）", "scope": "Trajectory-Frame", "run": run_geom,
     "input": {"mode": "line", "help": "dist i j [out]｜angle i j k [out]｜dihedral i j k l [out]", "example": "dist 1 2 d_12"}},
    {"name": "质心计算", "description": "COM（10位小数）", "scope": "Trajectory-Frame", "run": run_com,
     "input": {"mode": "line", "help": "masses m1,m2,...（留空=默认质量）", "example": "masses 12,1,1,16"}},
    {"name": "轨迹统计", "description": "均值/方差（10位小数）", "scope": "Trajectory-All", "run": run_stats,
     "input": {"mode": "line", "help": "<列名> [skip_none]，如：dist_1_2 false", "example": "dist_1_2"}},
    {"name": "全局统计", "description": "任务级均值（10位小数，可同步到轨迹）", "scope": "Task-Global", "run": run_global,
     "input": {"mode": "line", "help": "<列名> [sync_to_trajs]，如：dist_1_2 true", "example": "dist_1_2"}},
    {"name": "任务表达", "description": "表达式（10位小数，安全评估）", "scope": "Task-Global", "run": run_expr,
     "input": {"mode": "line", "help": "name <结果名> expr <表达式> [sync_to_trajs true] 或 key=value", "example": "name gm expr mean(\"global_mean__dist_1_2\")"}},
]
