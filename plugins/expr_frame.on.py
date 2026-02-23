# -*- coding: utf-8 -*-
"""Per-frame expression evaluator plugin for trajectory rows."""

import ast
import math
import re
from typing import Any, Dict, List, Set, Tuple

# 主程序在加载时注入 Trajectory/SimpleTable；兜底导入用于静态检查或独立运行
try:
    Trajectory  # type: ignore
    SimpleTable  # type: ignore
except NameError:
    # 当插件独立运行或静态检查时，从 core 导入所需类型
    from md_modules.core import SimpleTable, Trajectory  # type: ignore


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
ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}
ALLOWED_NAMES = set(ALLOWED_FUNCS.keys()) | set(ALLOWED_CONSTS.keys())

ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Expression,
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
    ast.Attribute,
)


def _validate_ast(tree: ast.AST):
    for n in ast.walk(tree):
        if not isinstance(n, ALLOWED_NODES):
            raise ValueError(f"不支持语法: {type(n).__name__}")
        if isinstance(n, ast.Attribute):
            if not isinstance(n.value, ast.Name):
                raise ValueError("仅允许一级属性访问")
        if isinstance(n, ast.Call):
            f = n.func
            if not isinstance(f, ast.Name):
                raise ValueError("仅允许调用白名单函数")
            if f.id not in ALLOWED_FUNCS:
                raise ValueError(f"不允许的函数: {f.id}")


def _names_in_expr_str(expr_src: str) -> Set[str]:
    import re

    names = set()
    for match in re.finditer(r"\b[a-zA-Z_][a-zA-Z0-9_.]*\b", expr_src):
        word = match.group()
        if (
            word not in ALLOWED_NAMES
            and not word.replace(".", "").replace("_", "").isdigit()
        ):
            names.add(word)
    return names


class CompiledLine:
    __slots__ = ("kind", "var", "src", "need_names")

    def __init__(self, kind: str, var: str, src, need_names: Set[str]):
        self.kind = kind
        self.var = var
        self.src = src
        self.need_names = need_names


def _compile_one_line(line: str) -> CompiledLine:
    line = line.strip()
    if "=" in line:
        parts = line.split("=", 1)
        if len(parts) == 2:
            var_part = parts[0].strip()
            expr_src = parts[1].strip()
            if var_part and var_part.isidentifier():
                var = var_part
                kind = "assign"
            else:
                var = ""
                expr_src = line
                kind = "expr"
        else:
            var = ""
            expr_src = line
            kind = "expr"
    else:
        var = ""
        expr_src = line
        kind = "expr"

    src = f"({expr_src})"
    try:
        tree = ast.parse(src, mode="eval")
        _validate_ast(tree)
        need = _names_in_expr(tree)
    except SyntaxError:
        need = _names_in_expr_str(expr_src)

    # 替换点号名称为 get_value 调用
    expr_src_modified = expr_src
    for nm in need:
        if "." in nm:
            expr_src_modified = expr_src_modified.replace(nm, f"get_value('{nm}')")

    src = f"({expr_src_modified})"
    return CompiledLine(kind, var, src, need)


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
    # 行值
    for nm in names_needed:
        if nm in row:
            v = row[nm]
            if isinstance(v, str):
                fv = _to_float_once(v)
                ctx[nm] = fv if fv is not None else v
            else:
                ctx[nm] = v
        else:
            # 尝试替换下划线为点号，以支持列名中的点号
            dotted_nm = nm.replace("_", ".")
            if dotted_nm in row:
                v = row[dotted_nm]
                if isinstance(v, str):
                    fv = _to_float_once(v)
                    ctx[nm] = fv if fv is not None else v
                else:
                    ctx[nm] = v
    # 任务元数据补缺
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
    # 添加 get_value 函数
    ctx["get_value"] = lambda k: row.get(k, task_meta.get(k, None))
    return ctx


def _is_bad_number(val) -> bool:
    return isinstance(val, float) and (math.isnan(val) or math.isinf(val))


def _err_text(ex: Exception) -> str:
    return f"错误: {type(ex).__name__}: {ex}"


def run_expr_frame(task, args):
    raw = (args.get("__raw__") or "").strip()
    if raw.lower() in ("q", "quit", "exit"):
        return {"process": ["[表达-帧] 已取消"]}

    compiled_lines = _compile_all(raw)
    if not compiled_lines:
        return {"process": ["[表达-帧] 未输入"]}

    # 所有表达式涉及的变量名（并集）
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
                    val = eval(cl.src, GLOBALS, ctx)

                    # NaN / Inf 也视为“错误”
                    if _is_bad_number(val):
                        raise ValueError("数值非法（NaN/Inf）")

                    if isinstance(val, (int, float)) and val is not None:
                        out = fmt_f10(float(val))
                        out_for_ctx = float(val)
                    else:
                        out = val
                        out_for_ctx = (
                            _to_float_once(val) if isinstance(val, str) else val
                        )

                    if cl.kind == "assign":
                        # 尝试映射回点号版本，如果列名包含点号
                        dotted_var = cl.var.replace("_", ".")
                        if dotted_var in cols_set or dotted_var in r:
                            target_var = dotted_var
                        else:
                            target_var = cl.var
                        r[target_var] = out
                        ctx[cl.var] = out_for_ctx if out_for_ctx is not None else out
                        new_cols.add(target_var)
                        cols_set.add(target_var)
                    else:
                        key = f"_expr{i}"
                        r[key] = out
                        new_cols.add(key)
                        cols_set.add(key)

                except Exception as ex:
                    # —— 错误也生成参数，但直接把错误文本写在目标列 —— #
                    if cl.kind == "assign":
                        # 尝试映射回点号版本
                        dotted_var = cl.var.replace("_", ".")
                        if dotted_var in cols_set or dotted_var in r:
                            target_var = dotted_var
                        else:
                            target_var = cl.var
                        r[target_var] = _err_text(ex)
                        new_cols.add(target_var)
                        cols_set.add(target_var)
                    else:
                        key = f"_expr{i}"
                        r[key] = _err_text(ex)
                        new_cols.add(key)
                        cols_set.add(key)
                    # 不再生成 _errorN 列

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
