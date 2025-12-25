# -*- coding: utf-8 -*-

import ast
import math
from typing import Dict, Any, List, Tuple, Set

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


def _to_float_once(v):
    try:
        return float(v)
    except Exception:
        return None


def fmt_f10(x: float) -> str:
    if x is None:
        return ""
    return f"{x:.10f}"


# ---- 白名单：函数 + 常量（直接用裸名，如 sqrt、pi） ----
ALLOWED_FUNCS = {
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
    "ceil": math.ceil,
    "floor": math.floor,
}
ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}
ALLOWED_NAMES = set(ALLOWED_FUNCS.keys()) | set(ALLOWED_CONSTS.keys())

ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Call,
    ast.keyword,
)


def _validate_ast(tree: ast.AST):
    for n in ast.walk(tree):
        if not isinstance(n, ALLOWED_NODES):
            raise ValueError(f"不支持语法: {type(n).__name__}")
        if isinstance(n, ast.Attribute):
            raise ValueError("禁止属性访问（请用 sqrt/pi 等裸名）")
        if isinstance(n, ast.Call):
            f = n.func
            if not isinstance(f, ast.Name):
                raise ValueError("仅允许调用白名单函数")
            if f.id not in ALLOWED_FUNCS:
                raise ValueError(f"不允许的函数: {f.id}")


def _names_in_expr(expr_node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for n in ast.walk(expr_node):
        if isinstance(n, ast.Name):
            names.add(n.id)
    # 排除白名单函数与常量名；保留真正需要从行/任务上下文取值的变量名
    return {nm for nm in names if nm not in ALLOWED_NAMES}


class CompiledLine:
    __slots__ = ("kind", "var", "code", "need_names")

    def __init__(self, kind: str, var: str, code, need_names: Set[str]):
        self.kind = kind
        self.var = var
        self.code = code
        self.need_names = need_names


def _compile_one_line(line: str) -> CompiledLine:
    mod = ast.parse(line, mode="exec")
    _validate_ast(mod)
    assigns = [n for n in ast.walk(mod) if isinstance(n, ast.Assign)]
    if assigns:
        a = assigns[0]
        if len(a.targets) != 1 or not isinstance(a.targets[0], ast.Name):
            raise ValueError("仅支持 var = expr")
        var = a.targets[0].id
        expr_src = line.split("=", 1)[1].strip()
    else:
        var = ""
        expr_src = line

    src = f"__v__=({expr_src})"
    tree = ast.parse(src, mode="exec")
    _validate_ast(tree)
    need = _names_in_expr(tree)
    code = compile(tree, "<expr>", "exec")
    kind = "assign" if assigns else "expr"
    return CompiledLine(kind, var, code, need)


def _compile_all(raw: str) -> List[CompiledLine]:
    if not raw:
        return []
    for sep in (";", ","):
        raw = raw.replace(sep, "\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return [_compile_one_line(ln) for ln in lines]


# 作为 exec 的 globals：函数 + 常量（一次性）
GLOBALS = {"__builtins__": {}, **ALLOWED_FUNCS, **ALLOWED_CONSTS}


def _build_ctx_min(
    row: Dict[str, Any], task_meta: Dict[str, Any], names_needed: Set[str]
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for nm in names_needed:
        if nm in row:
            v = row[nm]
            if isinstance(v, str):
                fv = _to_float_once(v)
                ctx[nm] = fv if fv is not None else v
            else:
                ctx[nm] = v
    for nm in names_needed:
        if nm in ctx:
            continue
        v = task_meta.get(nm)
        if isinstance(v, (int, float)):
            ctx[nm] = float(v)
        elif isinstance(v, str):
            fv = _to_float_once(v)
            ctx[nm] = fv if fv is not None else v
        elif v is not None:
            ctx[nm] = v
    return ctx


def run_expr_frame(task, args):
    raw = (args.get("__raw__") or "").strip()
    if raw.lower() in ("q", "quit", "exit"):
        return {"process": ["[表达-帧] 已取消"]}

    compiled_lines = _compile_all(raw)
    if not compiled_lines:
        return {"process": ["[表达-帧] 未输入"]}

    names_union: Set[str] = set()
    for cl in compiled_lines:
        names_union |= cl.need_names

    traj_count = 0
    frame_count = 0
    new_cols: Set[str] = set()

    for _, traj in task.trajectories.items():
        rows = traj.table.rows
        if not rows:
            continue
        traj_count += 1
        cols_set = set(traj.table.columns)

        for r in rows:
            ctx = _build_ctx_min(r, task.meta, names_union)
            for i, cl in enumerate(compiled_lines, start=1):
                try:
                    exec(cl.code, GLOBALS, ctx)
                    val = ctx.get("__v__")
                    if isinstance(val, (int, float)) and val is not None:
                        out = fmt_f10(float(val))
                        out_for_ctx = float(val)
                    else:
                        out = val
                        out_for_ctx = (
                            _to_float_once(val) if isinstance(val, str) else val
                        )

                    if cl.kind == "assign":
                        r[cl.var] = out
                        ctx[cl.var] = out_for_ctx if out_for_ctx is not None else out
                        new_cols.add(cl.var)
                        cols_set.add(cl.var)
                    else:
                        key = f"_expr{i}"
                        r[key] = out
                        new_cols.add(key)
                        cols_set.add(key)
                except Exception as ex:
                    key = f"_error{i}"
                    r[key] = f"{type(ex).__name__}: {ex}"
                    new_cols.add(key)
                    cols_set.add(key)
            frame_count += 1

        traj.table.columns = list(cols_set)

    cols_out = ", ".join(sorted(new_cols)) if new_cols else "(无)"
    return {
        "process": [f"[表达-帧] 轨迹：{traj_count}｜帧：{frame_count}｜列：{cols_out}"]
    }


PLUGINS = [
    {
        "name": "帧表达",
        "description": "逐帧表达式（写入数据表列，10位小数）",
        "scope": "Trajectory-Frame",
        "run": run_expr_frame,
        "input": {
            "mode": "line",
            "help": "支持赋值；多条可用分号/逗号/换行分隔。可直接使用 sqrt、pi、sin、cos 等裸名。",
            "example": "x = x_1 + x_2; v = sqrt(vx_1*vx_1 + vy_1*vy_1 + vz_1*vz_1); area = pi * (dist_1_2*0.5) * (dist_1_2*0.5)",
        },
    }
]
