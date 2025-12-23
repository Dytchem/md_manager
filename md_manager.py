# md_manager.py
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import re
import importlib.util
from typing import Dict, List, Optional, Any, Tuple


# ========== 基础IO/工具 ==========
def clear_screen() -> None:
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass


def pause(msg: str = "按回车继续...") -> None:
    try:
        input(msg)
    except EOFError:
        pass


def input_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def is_quit(s: str) -> bool:
    return s.lower() in ("q", "quit", "exit")


def format_value(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


# ========== 表结构 ==========
class SimpleTable:
    def __init__(self, columns: List[str], rows: List[Dict[str, Any]]):
        self.columns = list(columns)
        self.rows = list(rows)

    def select_columns(self, cols: Optional[List[str]] = None) -> "SimpleTable":
        if not cols:
            return SimpleTable(self.columns, self.rows)
        cols2 = [c for c in cols if c in self.columns]
        return SimpleTable(cols2, [{c: r.get(c) for c in cols2} for r in self.rows])

    def to_csv(self, path: str, columns: Optional[List[str]] = None):
        cols = columns or self.columns
        cols = [c for c in cols if c in self.columns]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow({c: r.get(c) for c in cols})


# ========== 轨迹对象 ==========
class Trajectory:
    def __init__(
        self,
        traj_id: str,
        name: str,
        table: SimpleTable,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.traj_id = traj_id
        self.name = name
        self.table = table
        self.meta = meta or {}
        self.refresh_basic_meta()

    def list_columns(self) -> List[str]:
        return list(self.table.columns)

    def refresh_basic_meta(self):
        self.meta["traj_id"] = self.traj_id
        self.meta.setdefault("name", self.name)

    def save_to_folder(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        data = os.path.join(folder, f"data_{self.traj_id}.csv")
        meta = os.path.join(folder, f"meta_{self.traj_id}.json")
        with open(data, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.table.columns)
            w.writeheader()
            for r in self.table.rows:
                w.writerow({c: r.get(c) for c in self.table.columns})
        with open(meta, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "traj_id": self.traj_id,
                    "name": self.name,
                    "meta": self.meta,
                    "columns": self.table.columns,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def load_from_folder(folder: str, traj_id: str) -> "Trajectory":
        data = os.path.join(folder, f"data_{traj_id}.csv")
        meta = os.path.join(folder, f"meta_{traj_id}.json")
        if not (os.path.isfile(data) and os.path.isfile(meta)):
            raise FileNotFoundError(traj_id)
        with open(meta, "r", encoding="utf-8") as fh:
            mj = json.load(fh)
        name = mj.get("name", f"traj_{traj_id}")
        with open(data, "r", newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            cols = list(rd.fieldnames or mj.get("columns", []))
            rows = [dict(r) for r in rd]
        return Trajectory(traj_id, name, SimpleTable(cols, rows), mj.get("meta", {}))


# ========== 任务对象 ==========
class Task:
    def __init__(self, name: str):
        self.name = name
        self.trajectories: Dict[str, Trajectory] = {}
        self.settings = {"list_fields": ["traj_id", "name"], "page_size": 20}
        self.meta: Dict[str, Any] = {}

    def add_trajectory(self, traj: Trajectory):
        self.trajectories[traj.traj_id] = traj

    def remove_trajectory(self, traj_id: str):
        self.trajectories.pop(traj_id, None)

    def next_traj_id(self) -> str:
        m = 0
        for tid in self.trajectories.keys():
            try:
                m = max(m, int(tid))
            except Exception:
                pass
        return str(m + 1)

    def list_trajs_table(self, fields: Optional[List[str]] = None) -> SimpleTable:
        fields = fields or self.settings.get("list_fields") or ["traj_id", "name"]
        items = sorted(self.trajectories.values(), key=lambda t: int(t.traj_id))
        rows: List[Dict[str, Any]] = []
        for t in items:
            row: Dict[str, Any] = {}
            for field in fields:
                if field == "traj_id":
                    row[field] = t.traj_id
                elif field == "name":
                    row[field] = t.name
                else:
                    row[field] = t.meta.get(field)
            rows.append(row)
        return SimpleTable(fields, rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "settings": self.settings,
            "traj_ids": list(self.trajectories.keys()),
            "meta": self.meta,
        }

    def save(self, root: str = "tasks"):
        folder = os.path.join(root, self.name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "task.json"), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        for traj in self.trajectories.values():
            traj.save_to_folder(folder)

    @staticmethod
    def load(name: str, root: str = "tasks") -> "Task":
        folder = os.path.join(root, name)
        tj = os.path.join(folder, "task.json")
        if not os.path.isfile(tj):
            raise FileNotFoundError(name)
        with open(tj, "r", encoding="utf-8") as fh:
            tjson = json.load(fh)
        task = Task(tjson.get("name", name))
        task.settings = tjson.get("settings", task.settings)
        task.meta = tjson.get("meta", {})
        for tid in tjson.get("traj_ids", []):
            task.add_trajectory(Trajectory.load_from_folder(folder, tid))
        return task


# ========== 解析/选择 ==========
def parse_mixed_selection(line: str, options: List[str]) -> Optional[List[str]]:
    if not line or not line.strip():
        return None
    toks = [tok for tok in re.split(r"[,\s]+", line.strip()) if tok]
    chosen: List[str] = []
    seen = set()
    for tok in toks:
        if tok.isdigit():
            idx = int(tok)
            if 1 <= idx <= len(options):
                c = options[idx - 1]
                if c not in seen:
                    chosen.append(c)
                    seen.add(c)
        else:
            if tok in options and tok not in seen:
                chosen.append(tok)
                seen.add(tok)
    return chosen if chosen else None


def parse_tid_values(line: str, options: List[str]) -> List[str]:
    if not line or not line.strip():
        return []
    op_set = set(options)
    tids: List[str] = []
    for tok in re.split(r"[,\s]+", line.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            for v in range(a, b + 1):
                sv = str(v)
                if sv in op_set and sv not in tids:
                    tids.append(sv)
            continue
        if tok in op_set and tok not in tids:
            tids.append(tok)
    return tids


def parse_index_spec(line: str, max_n: int) -> List[int]:
    """解析 '1,2,5-10' 为行号列表（1-based）。"""
    if not line or not line.strip():
        return []
    idxs: List[int] = []
    for tok in re.split(r"[,\s]+", line.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            for v in range(a, b + 1):
                if 1 <= v <= max_n and v not in idxs:
                    idxs.append(v)
        elif tok.isdigit():
            v = int(tok)
            if 1 <= v <= max_n and v not in idxs:
                idxs.append(v)
    return idxs


def build_value_pred(spec: str) -> Tuple[bool, Any]:
    """
    从 '1,2,3,5-10' 构造匹配谓词：
    返回 (is_range, predicate_or_set)
    - 若包含范围，尝试按数值比较，否则按字符串等值。
    """
    toks = [t for t in re.split(r"[,\s]+", spec.strip()) if t]
    has_range = any(re.fullmatch(r"\d+-\d+", t) for t in toks)
    if has_range:
        ranges: List[Tuple[float, float]] = []
        values: set = set()
        for t in toks:
            m = re.fullmatch(r"(\d+)-(\d+)", t)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                if a > b:
                    a, b = b, a
                ranges.append((a, b))
            elif re.fullmatch(r"\d+", t):
                values.add(float(t))
            else:
                # 非数字，作为字符串等值
                values.add(t)

        def pred(val: Any) -> bool:
            # 优先数值比较
            try:
                fv = float(val)
                for a, b in ranges:
                    if a <= fv <= b:
                        return True
                if fv in values:
                    return True
            except Exception:
                s = str(val)
                if s in values:
                    return True
            return False

        return True, pred
    else:
        return False, set(toks)


# ========== 排序工具 ==========
_nat_re = re.compile(r"\d+|\D+")


def _natural_key(x: Any):
    if isinstance(x, str):
        parts = _nat_re.findall(x)
        key = []
        for p in parts:
            key.append(int(p) if p.isdigit() else p.lower())
        return key
    return x


def _apply_sort_dict_rows(
    rows: List[Dict[str, Any]],
    key_name: str,
    order: str = "asc",
    keyword: Optional[str] = None,
) -> List[Dict[str, Any]]:
    kw = (keyword or "").strip().lower()

    def sort_key(row: Dict[str, Any]):
        v = row.get(key_name)
        hit = (
            0
            if (kw and isinstance(v, (str, int, float)) and kw in str(v).lower())
            else 1
        )
        return (hit, _natural_key(v))

    reverse = order == "desc"
    return sorted(rows, key=sort_key, reverse=reverse)


# ========== 表视图（统一字母，支持删/抽/导） ==========
class TableViewer:
    @staticmethod
    def _w(columns: List[str], rows: List[Dict[str, Any]]) -> Dict[str, int]:
        w = {c: len(c) for c in columns}
        for r in rows:
            for c in columns:
                v = format_value(r.get(c))
                w[c] = max(w[c], len(v))
        return w

    @staticmethod
    def _print(columns: List[str], rows: List[Dict[str, Any]], start: int, end: int):
        w = TableViewer._w(columns, rows[start:end])
        header = " | ".join(c.ljust(w[c]) for c in columns)
        sep = "-+-".join("-" * w[c] for c in columns)
        print(header)
        print(sep)
        for r in rows[start:end]:
            print(" | ".join(format_value(r.get(c)).ljust(w[c]) for c in columns))

    @staticmethod
    def _print_cols_with_idx(columns: List[str]):
        for i, c in enumerate(columns, 1):
            print(f"{i}. {c}")

    @staticmethod
    def _select_rows_by_spec(
        rows_local: List[Dict[str, Any]], cols: List[str], spec: str
    ) -> List[Dict[str, Any]]:
        """支持：序号列表/范围；或 字段=值列表/范围；精确匹配（=）。"""
        spec = (spec or "").strip()
        if not spec:
            return []
        # 字段=值列表/范围（精确）
        if "=" in spec:
            f, v = spec.split("=", 1)
            f = f.strip()
            v = v.strip()
            if f not in cols:
                return []
            is_range, payload = build_value_pred(v)
            selected = []
            for idx, r in enumerate(rows_local, 1):
                val = r.get(f)
                if is_range:
                    if payload(val):
                        selected.append(r)
                else:
                    if str(val) in payload:
                        selected.append(r)
            return selected
        # 序号列表/范围
        idxs = parse_index_spec(spec, len(rows_local))
        return [rows_local[i - 1] for i in idxs]

    @staticmethod
    def run(
        table: SimpleTable,
        default_columns: Optional[List[str]] = None,
        page_size: int = 20,
        title: str = "",
        export_all_handler=None,  # 单轨：导出全部轨迹（合并到一个文件）
        delete_handler=None,  # 单轨：删除行
        export_page_option: bool = True,  # 单轨：仅导整表 -> False；列表：允许导当前页 -> True
    ):
        rows_local = list(table.rows)  # 与外部 rows 引用共享
        cols = default_columns or table.columns
        cols = [c for c in cols if c in table.columns]
        page = 0
        ps = max(1, page_size)
        sort_state = {"key": None, "order": "asc", "keyword": None}

        def refresh_total():
            return max(1, (len(rows_local) + ps - 1) // ps)

        while True:
            total = refresh_total()
            clear_screen()
            print(title if title else "数据表")
            if not rows_local:
                print("（空表）")
            else:
                s = page * ps
                e = min(len(rows_local), (page + 1) * ps)
                print(
                    f"显示列：{', '.join(cols)} | 页：{page + 1}/{total} | 行：{s}-{e}"
                )
                if sort_state["key"]:
                    kw_tip = (
                        f"，关键字优先：{sort_state['keyword']!r}"
                        if sort_state["keyword"]
                        else ""
                    )
                    print(f"排序：{sort_state['key']}（{sort_state['order']}）{kw_tip}")
                print("-" * 80)
                TableViewer._print(cols, rows_local, s, e)
                print("-" * 80)

            # 统一字母，两个字名称（有功能才显示）
            menu_items = [
                "列设(c)",
                "下页(n)",
                "上页(p)",
                "行数(r)",
                "跳页(g)",
                "排序(s)",
                "抽取(x)",
                "导出(e)",
                "返回(q)",
            ]
            if export_all_handler is not None:
                menu_items.insert(-1, "全导(a)")
            if delete_handler is not None:
                menu_items.insert(-1, "删行(d)")
            print("命令：" + "｜".join(menu_items))

            cmd = input_line("> ").lower()
            if is_quit(cmd) or cmd == "q":
                return
            elif cmd == "n":
                if page + 1 < total:
                    page += 1
            elif cmd == "p":
                if page > 0:
                    page -= 1
            elif cmd == "r":
                new_ps = input_line("新的每页行数：")
                if new_ps.isdigit() and int(new_ps) > 0:
                    ps = int(new_ps)
                    page = 0
            elif cmd == "g":
                to_page = input_line("跳转到第几页：")
                if to_page.isdigit():
                    tp = int(to_page)
                    if 1 <= tp <= total:
                        page = tp - 1
            elif cmd == "c":
                TableViewer._print_cols_with_idx(table.columns)
                s = input_line("列编号或列名（空格/逗号；空=全部）：")
                if is_quit(s):
                    continue
                if not s.strip():
                    cols = list(table.columns)
                else:
                    mixed = parse_mixed_selection(s, table.columns)
                    if mixed:
                        cols = mixed
                    else:
                        toks = [t for t in re.split(r"[,\s]+", s.strip()) if t]
                        cols = [
                            table.columns[int(t) - 1]
                            for t in toks
                            if t.isdigit() and 1 <= int(t) <= len(table.columns)
                        ]
            elif cmd == "s":
                key = input_line("排序字段：")
                if key not in table.columns:
                    print("字段不存在")
                    pause()
                    continue
                order = input_line("顺序（asc/desc）：").lower() or "asc"
                if order not in ("asc", "desc"):
                    order = "asc"
                kw = input_line("关键字（可空；命中者优先）：")
                rows_local = _apply_sort_dict_rows(rows_local, key, order, kw)
                sort_state = {"key": key, "order": order, "keyword": kw or None}
                page = 0
            elif cmd == "x":
                spec = input_line("抽取条件（序号或 字段=值列表/范围；输入 q 取消）：")
                if is_quit(spec):
                    continue
                sel = TableViewer._select_rows_by_spec(rows_local, cols, spec)
                if not sel:
                    print("未匹配到行")
                    pause()
                    continue
                rows_local = sel
                page = 0
            elif cmd == "d" and delete_handler is not None:
                spec = input_line("删除条件（序号或 字段=值列表/范围；输入 q 取消）：")
                if is_quit(spec):
                    continue
                sel = TableViewer._select_rows_by_spec(rows_local, cols, spec)
                if not sel:
                    print("未匹配到行")
                    pause()
                    continue
                try:
                    delete_handler(sel)  # 修改外部表 rows
                    rows_local = list(table.rows)  # 重新载入
                    print(f"已删除 {len(sel)} 行")
                except Exception as ex:
                    print(f"删除失败：{ex}")
                pause()
                page = 0
            elif cmd == "a" and export_all_handler is not None:
                path = input_line(
                    "输出CSV（默认 all_trajs_view.csv；输入 q 取消）："
                ).strip()
                if is_quit(path):
                    continue
                if not path:
                    path = "all_trajs_view.csv"
                try:
                    export_all_handler(cols, path)  # 合并所有轨迹为一个文件
                    print(f"已导出：{os.path.abspath(path)}")
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()
            elif cmd == "e":
                # 单轨视图：仅导整表；列表视图可允许导当前页（由 export_page_option 控制）
                if export_page_option:
                    opt = (
                        input_line("导出范围（all/page；回车=all，输入 q 取消）：")
                        .strip()
                        .lower()
                    )
                    if is_quit(opt):  # 取消
                        continue
                    if opt not in ("all", "page", ""):
                        opt = "all"
                else:
                    opt = "all"
                path = input_line("输出CSV（默认 view.csv；输入 q 取消）：").strip()
                if is_quit(path):
                    continue
                if not path:
                    path = "view.csv"
                s = page * ps
                e = min(len(rows_local), (page + 1) * ps)
                out_rows = (
                    rows_local[s:e]
                    if (export_page_option and opt == "page")
                    else rows_local
                )
                try:
                    SimpleTable(cols, out_rows).to_csv(path, columns=cols)
                    print(f"已导出：{os.path.abspath(path)}")
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()


# ========== 插件 ==========
class Plugin:
    def __init__(self, module, name, description, scope, run_func, input_spec):
        self.module = module
        self.name = name
        self.description = description
        self.scope = scope
        self.run = run_func
        self.input = input_spec or {}


class PluginManager:
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = plugins_dir
        self.plugins: List[Plugin] = []

    def load_plugins(self):
        self.plugins.clear()
        os.makedirs(self.plugins_dir, exist_ok=True)
        files = sorted(
            [f for f in os.listdir(self.plugins_dir) if f.endswith(".on.py")]
        )
        for fname in files:
            path = os.path.join(self.plugins_dir, fname)
            try:
                spec = importlib.util.spec_from_file_location(fname[:-3], path)
                module = importlib.util.module_from_spec(spec)
                setattr(module, "Trajectory", Trajectory)
                setattr(module, "SimpleTable", SimpleTable)
                spec.loader.exec_module(module)  # type: ignore
                loaded = getattr(module, "PLUGINS", [])
                for pd in loaded:
                    name = pd.get("name") or fname
                    desc = pd.get("description") or ""
                    scope = pd.get("scope") or "Unknown"
                    run = pd.get("run")
                    input_spec = pd.get("input") or {}
                    if not callable(run):
                        continue
                    self.plugins.append(
                        Plugin(module, name, desc, scope, run, input_spec)
                    )
            except Exception as ex:
                print(f"[插件失败] {fname}: {ex}")

    def list_plugins(self, scope_filter: Optional[str] = None) -> List[Plugin]:
        return (
            [p for p in self.plugins if p.scope == scope_filter]
            if scope_filter
            else self.plugins
        )


# ========== 插件输入 ==========
def prompt_args_by_input_spec(plugin: Plugin) -> Dict[str, Any]:
    spec = plugin.input or {}
    mode = (spec.get("mode") or "form").lower()
    if mode == "line":
        help_text = spec.get("help") or ""
        example = spec.get("example") or ""
        if help_text:
            print(help_text)
        if example:
            print(f"示例：{example}")
        line = input_line("> ")
        return {"__raw__": line}
    else:
        fields = spec.get("fields") or []
        args: Dict[str, Any] = {}
        for f in fields:
            name = f.get("name")
            prompt = f.get("prompt") or name or ""
            default = f.get("default", "")
            val = input_line(f"{prompt}（默认 {default}）：") or default
            args[name] = val
        return args


# ========== 插件结果应用 ==========
def apply_plugin_result(task: Task, result: Any):
    if isinstance(result, dict):
        proc = result.get("process")
        if isinstance(proc, list) and proc:
            for line in proc:
                print(line)

    if isinstance(result, list):
        datasets = result
        added = 0
        for ds in datasets:
            cols = ds.get("columns")
            rows = ds.get("rows")
            name = ds.get("name") or None
            meta = ds.get("meta") or {}
            if not cols or not rows:
                continue
            tid = task.next_traj_id()
            if not name:
                name = f"traj_{tid}"
            table = SimpleTable(cols, rows)
            meta["traj_seq"] = int(tid)
            traj = Trajectory(tid, name, table, meta)
            task.add_trajectory(traj)
            added += 1
        print(f"导入完成：新增轨迹 {added} 条。")
        return

    if not isinstance(result, dict):
        return

    datasets = result.get("datasets")
    if isinstance(datasets, list):
        added = 0
        for ds in datasets:
            cols = ds.get("columns")
            rows = ds.get("rows")
            name = ds.get("name") or None
            meta = ds.get("meta") or {}
            if not cols or not rows:
                continue
            tid = task.next_traj_id()
            if not name:
                name = f"traj_{tid}"
            table = SimpleTable(cols, rows)
            meta["traj_seq"] = int(tid)
            traj = Trajectory(tid, name, table, meta)
            task.add_trajectory(traj)
            added += 1
        print(f"导入完成：新增轨迹 {added} 条。")

    traj_tables = result.get("traj_tables") or {}
    for tid, tdata in traj_tables.items():
        tid_str = str(tid)
        traj = task.trajectories.get(tid_str)
        if not traj:
            continue
        cols = tdata.get("columns") or traj.table.columns
        rows = tdata.get("rows") or traj.table.rows
        traj.table = SimpleTable(cols, rows)
        traj.refresh_basic_meta()

    traj_meta = result.get("traj_meta") or {}
    for tid, kv in traj_meta.items():
        tid_str = str(tid)
        traj = task.trajectories.get(tid_str)
        if not traj:
            continue
        for k, v in kv.items():
            traj.meta[k] = v

    task_meta = result.get("task_meta") or {}
    for k, v in task_meta.items():
        task.meta[k] = v

    lf = result.get("list_field")
    mp = result.get("values") or {}
    if lf:
        fields = task.settings.get("list_fields", [])
        if lf not in fields:
            fields.append(lf)
            task.settings["list_fields"] = fields
        for tid, val in mp.items():
            tid_str = str(tid)
            traj = task.trajectories.get(tid_str)
            if traj:
                traj.meta[lf] = val


# ========== CLI ==========
class MDManagerCLI:
    def __init__(self):
        self.current_task = Task(name="默认任务")
        self.tasks_root = "tasks"
        self.pm = PluginManager()
        self.pm.load_plugins()

    @staticmethod
    def choose_columns(all_columns: List[str]) -> List[str]:
        clear_screen()
        print("选择显示列（空格/逗号均可；可混合序号与列名；空=精简）")
        TableViewer._print_cols_with_idx(all_columns)
        s = input_line("> ")
        if is_quit(s):
            s = ""
        if not s.strip():
            return all_columns[: min(8, len(all_columns))]
        mixed = parse_mixed_selection(s, all_columns)
        if mixed:
            return mixed
        toks = [t for t in re.split(r"[,\s]+", s.strip()) if t]
        return [
            all_columns[int(t) - 1]
            for t in toks
            if t.isdigit() and 1 <= int(t) <= len(all_columns)
        ]

    def run(self):
        while True:
            clear_screen()
            print("=== 分子动力学任务管理 ===")
            print(
                f"任务名：{self.current_task.name}｜轨迹数：{len(self.current_task.trajectories)}"
            )
            print("菜单：导入(i)｜列表(l)｜计算(c)｜参数(p)｜保存(s)｜切换(w)｜退出(q)")
            cmd = input_line("> ").strip().lower()
            if is_quit(cmd) or cmd in ("退出", "q"):
                print("已退出。")
                break
            elif cmd in ("导入", "i"):
                self.menu_import()
            elif cmd in ("列表", "l"):
                self.menu_trajectory_list()
            elif cmd in ("计算", "c"):
                self.menu_compute()
            elif cmd in ("参数", "p"):
                self.menu_view_task_params()
            elif cmd in ("保存", "s"):
                self.menu_save()
            elif cmd in ("切换", "w"):
                self.menu_switch()

    # ===== 导入 =====
    def menu_import(self):
        clear_screen()
        print("=== 数据导入（正则匹配文件夹，所有 Import 插件串联运行） ===")
        imps = self.pm.list_plugins(scope_filter="Import")
        if not imps:
            print("未发现导入插件")
            pause()
            return
        root = input_line("批量根目录：").strip()
        if is_quit(root) or not root or not os.path.isdir(root):
            print("根目录无效")
            pause()
            return
        pattern = input_line("子目录正则（默认 ^\\d+$）：").strip() or r"^\d+$"
        try:
            rx = re.compile(pattern)
        except Exception:
            print(f"正则无效：{pattern}")
            pause()
            return

        subs = sorted(
            [
                d
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d)) and rx.match(d)
            ]
        )
        if not subs:
            print("未匹配到子目录")
            pause()
            return

        print(f"匹配到 {len(subs)} 个子目录，串联运行 {len(imps)} 个导入插件...")
        logs: List[str] = []
        processed = 0
        for d in subs:
            folder = os.path.join(root, d)
            for plugin in imps:
                try:
                    result = plugin.run(self.current_task, {"folder": folder})
                    if isinstance(result, dict) and "process" in result:
                        logs.extend(result["process"])
                    apply_plugin_result(self.current_task, result)
                except Exception as ex:
                    logs.append(f"[跳过] {folder} / {plugin.name}: {ex}")
            processed += 1

        if logs:
            print("\n--- 过程日志 ---")
            for line in logs:
                print(line)
        print(f"\n完成：处理子目录 {processed} 个。")
        pause()

    # ===== 轨迹列表（统一字母；精确匹配；抽取/导出/删除/排序保持） =====
    def menu_trajectory_list(self):
        fields = self.current_task.settings.get("list_fields", ["traj_id", "name"])
        order_tids = sorted(self.current_task.trajectories.keys(), key=lambda x: int(x))
        page_size = max(1, int(self.current_task.settings.get("page_size", 20)))
        page = 0
        sort_state = {"key": None, "order": "asc", "keyword": None}

        def build_rows(order_ids: List[str], use_fields: List[str]):
            rows: List[Dict[str, Any]] = []
            for tid in order_ids:
                t = self.current_task.trajectories.get(tid)
                if not t:
                    continue
                row = {"_tid": tid}
                for f in use_fields:
                    if f == "traj_id":
                        row[f] = t.traj_id
                    elif f == "name":
                        row[f] = t.name
                    else:
                        row[f] = t.meta.get(f)
                rows.append(row)
            return rows, use_fields[:]

        def apply_sort(
            rows: List[Dict[str, Any]],
            key: Optional[str],
            order: str,
            keyword: Optional[str],
        ) -> List[str]:
            if not key or key not in fields:
                return [r["_tid"] for r in rows]
            return [r["_tid"] for r in _apply_sort_dict_rows(rows, key, order, keyword)]

        rows_local, cols_local = build_rows(order_tids, fields)
        if sort_state["key"]:
            order_tids = apply_sort(
                rows_local,
                sort_state["key"],
                sort_state["order"],
                sort_state["keyword"],
            )
            rows_local, cols_local = build_rows(order_tids, fields)

        while True:
            clear_screen()
            print("=== 轨迹列表 ===")
            total = max(1, (len(rows_local) + page_size - 1) // page_size)
            start = page * page_size
            end = min(len(rows_local), (page + 1) * page_size)

            w = {c: len(c) for c in cols_local}
            for r in rows_local[start:end]:
                for c in cols_local:
                    w[c] = max(w[c], len(format_value(r.get(c))))
            header = " | ".join(c.ljust(w[c]) for c in cols_local)
            sep = "-+-".join("-" * w[c] for c in cols_local)
            print(header)
            print(sep)
            for r in rows_local[start:end]:
                print(
                    " | ".join(format_value(r.get(c)).ljust(w[c]) for c in cols_local)
                )

            if sort_state["key"]:
                kw_tip = (
                    f"，关键字优先：{sort_state['keyword']!r}"
                    if sort_state["keyword"]
                    else ""
                )
                print(
                    f"\n页：{page + 1}/{total}；每页：{page_size}；排序：{sort_state['key']}（{sort_state['order']}）{kw_tip}"
                )
            else:
                print(f"\n页：{page + 1}/{total}；每页：{page_size}")

            print(
                "命令：查看(v)｜删除(d)｜列设(c)｜下页(n)｜上页(p)｜行数(r)｜跳页(g)｜排序(s)｜抽取(x)｜导出(e)｜返回(q)"
            )
            cmd = input_line("> ").strip().lower()

            if is_quit(cmd) or cmd == "q":
                return
            elif cmd == "n":
                if page + 1 < total:
                    page += 1
            elif cmd == "p":
                if page > 0:
                    page -= 1
            elif cmd == "r":
                new_ps = input_line("新的每页行数：")
                if new_ps.isdigit() and int(new_ps) > 0:
                    page_size = int(new_ps)
                    page = 0
            elif cmd == "g":
                to_page = input_line("跳转到第几页：")
                if to_page.isdigit():
                    tp = int(to_page)
                    if 1 <= tp <= total:
                        page = tp - 1
            elif cmd == "s":
                key = input_line("排序字段（当前显示列之一）：")
                if key and key not in cols_local:
                    print("字段不在当前显示列（排序未更改）。")
                    pause()
                    continue
                order = input_line("顺序（asc/desc）：").lower() or "asc"
                if order not in ("asc", "desc"):
                    order = "asc"
                kw = input_line("关键字（可空；命中者优先）：")
                sort_state = {
                    "key": key if key else None,
                    "order": order,
                    "keyword": kw or None,
                }
                order_tids = apply_sort(
                    rows_local,
                    sort_state["key"],
                    sort_state["order"],
                    sort_state["keyword"],
                )
                rows_local, cols_local = build_rows(order_tids, fields)
                page = 0
            elif cmd == "c":
                fields_all = ["traj_id", "name"]
                calc_cols = sorted(
                    {
                        k
                        for t in self.current_task.trajectories.values()
                        for k in t.meta.keys()
                    }
                )
                fields_all += [c for c in calc_cols if c not in fields_all]
                for i, f in enumerate(fields_all, 1):
                    print(f"{i}. {f}")
                s2 = input_line("字段编号或列名（空格/逗号均可；空=不变）：")
                if is_quit(s2) or not s2.strip():
                    continue
                mixed = parse_mixed_selection(s2, fields_all)
                chosen = (
                    mixed
                    if mixed
                    else [
                        c.strip()
                        for c in s2.split(",")
                        if c.strip() and c.strip() in fields_all
                    ]
                )
                self.current_task.settings["list_fields"] = chosen
                fields = chosen[:]
                rows_local, cols_local = build_rows(order_tids, fields)
                page = 0
            elif cmd == "x":
                spec = input_line("抽取条件（序号或 字段=值列表/范围；输入 q 取消）：")
                if is_quit(spec):
                    continue
                # 复用 TableViewer 的选择逻辑
                tmp_rows = [
                    {"_tid": r["_tid"], **{c: r.get(c) for c in cols_local}}
                    for r in rows_local
                ]
                sel = TableViewer._select_rows_by_spec(tmp_rows, cols_local, spec)
                if not sel:
                    print("未匹配到条目")
                    pause()
                    continue
                keep_ids = [r["_tid"] for r in sel]
                order_tids = [tid for tid in order_tids if tid in keep_ids]
                rows_local, cols_local = build_rows(order_tids, fields)
                page = 0
            elif cmd == "e":
                path = input_line(
                    "输出CSV（默认 traj_list.csv；输入 q 取消）："
                ).strip()
                if is_quit(path):
                    continue
                if not path:
                    path = "traj_list.csv"
                try:
                    SimpleTable(cols_local, rows_local).to_csv(path, columns=cols_local)
                    print(f"已导出：{os.path.abspath(path)}")
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()
            elif cmd == "v":
                q = input_line("输入条件（字段=值 或 关键字；留空=输入轨ID）：").strip()
                target_tid: Optional[str] = None
                if not q:
                    tid_in = input_line("轨迹ID：").strip()
                    if tid_in in self.current_task.trajectories:
                        target_tid = tid_in
                else:
                    matches: List[Dict[str, Any]] = []
                    if "=" in q:
                        f, v = q.split("=", 1)
                        f = f.strip()
                        v = v.strip()
                        # 精确匹配
                        for r in rows_local:
                            if f in r and str(r.get(f, "")) == v:
                                matches.append(r)
                    else:
                        # 关键字模糊（当前显示列）
                        v = q.lower()
                        for r in rows_local:
                            for c in cols_local:
                                if v in str(r.get(c, "")).lower():
                                    matches.append(r)
                                    break
                    if not matches:
                        print("未匹配到轨迹")
                        pause()
                        continue
                    if len(matches) == 1:
                        target_tid = matches[0]["_tid"]
                    else:
                        print("匹配多条，单选：")
                        for i, r in enumerate(matches, 1):
                            name_val = r.get("name", None)
                            if name_val is None:
                                t = self.current_task.trajectories.get(r["_tid"])
                                name_val = t.name if t else ""
                            print(f"{i}. [{r['_tid']}] {name_val}")
                        sel = input_line("编号：").strip()
                        if sel.isdigit():
                            idx = int(sel)
                            if 1 <= idx <= len(matches):
                                target_tid = matches[idx - 1]["_tid"]
                if not target_tid:
                    continue
                traj = self.current_task.trajectories.get(target_tid)
                if not traj:
                    print("轨迹不存在")
                    pause()
                    continue
                self.menu_view_trajectory(traj)
            elif cmd == "d":
                q = input_line("删除条件（字段=值 或 关键字；留空=按轨ID）：").strip()
                del_tids: List[str] = []
                if not q:
                    tid_in = input_line("轨迹ID（逗号/范围，如 10-20）：").strip()
                    options = list(self.current_task.trajectories.keys())
                    del_tids = parse_tid_values(tid_in, options)
                else:
                    matches: List[Dict[str, Any]] = []
                    if "=" in q:
                        f, v = q.split("=", 1)
                        f = f.strip()
                        v = v.strip()
                        for r in rows_local:
                            if f in r and str(r.get(f, "")) == v:
                                matches.append(r)
                    else:
                        v = q.lower()
                        for r in rows_local:
                            for c in cols_local:
                                if v in str(r.get(c, "")).lower():
                                    matches.append(r)
                                    break
                    if not matches:
                        print("未匹配到轨迹")
                        pause()
                        continue
                    opts = [r["_tid"] for r in matches]
                    print("匹配如下：")
                    for i, r in enumerate(matches, 1):
                        name_val = r.get("name", None)
                        if name_val is None:
                            t = self.current_task.trajectories.get(r["_tid"])
                            name_val = t.name if t else ""
                        print(f"{i}. [{r['_tid']}] {name_val}")
                    sel = (
                        input_line("输入编号或轨ID（1,3,5 / 2-6 / all）：")
                        .strip()
                        .lower()
                    )
                    if sel == "all":
                        del_tids = opts
                    else:
                        chosen = parse_mixed_selection(sel, opts)
                        if chosen:
                            del_tids = chosen
                if not del_tids:
                    continue
                for tid in del_tids:
                    self.current_task.remove_trajectory(tid)
                order_tids = [
                    tid for tid in order_tids if tid in self.current_task.trajectories
                ]
                rows_local, cols_local = build_rows(order_tids, fields)
                page = 0

    # ===== 轨迹详情（表视图统一字母；支持删/抽/全导） =====
    def menu_view_trajectory(self, traj: Trajectory):
        while True:
            clear_screen()
            print(f"=== 轨迹详情：{traj.traj_id}（{traj.name}） ===")
            if not traj.meta:
                print("（插件未写入任何参数）")
            else:
                for k in sorted(traj.meta.keys()):
                    print(f"{k}: {traj.meta.get(k)}")
            print("命令：表视图(t)｜返回(q)")
            cmd = input_line("> ").strip().lower()
            if is_quit(cmd) or cmd in ("返回", "q"):
                return
            elif cmd in ("表视图", "t"):
                all_cols = traj.list_columns()
                cols = self.choose_columns(all_cols)
                page = int(self.current_task.settings.get("page_size", 20))

                def export_all_handler(current_cols: List[str], out_path: str):
                    # 单轨视图额外：导出全部轨迹为一个文件（同列）
                    all_rows = []
                    header = ["traj_id"] + current_cols
                    for t in sorted(
                        self.current_task.trajectories.values(),
                        key=lambda x: int(x.traj_id),
                    ):
                        for r in t.table.rows:
                            row = {"traj_id": t.traj_id}
                            for c in current_cols:
                                row[c] = r.get(c) if c in t.table.columns else None
                            all_rows.append(row)
                    with open(out_path, "w", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=header)
                        w.writeheader()
                        for r in all_rows:
                            w.writerow(r)

                def delete_handler(rows_to_del: List[Dict[str, Any]]):
                    # rows_to_del 与 traj.table.rows 中的字典对象是同引用，可直接删
                    ids = set(id(r) for r in rows_to_del)
                    new_rows = [r for r in traj.table.rows if id(r) not in ids]
                    traj.table.rows = new_rows

                # 单轨表视图：导出仅导整表（不导当前页）；支持删行/抽取/全导
                TableViewer.run(
                    SimpleTable(traj.table.columns, traj.table.rows),
                    default_columns=cols,
                    page_size=page,
                    title=f"数据表：{traj.traj_id}",
                    export_all_handler=export_all_handler,
                    delete_handler=delete_handler,
                    export_page_option=False,
                )

    # ===== 计算 =====
    def menu_compute(self):
        while True:
            clear_screen()
            print("=== 参数计算 ===")
            print("类别：")
            print("  f - 时刻参数（表行级）")
            print("  a - 轨迹参数（轨迹级）")
            print("  g - 任务参数（任务级）")
            print("  q - 返回")
            cat = input_line("> ").strip().lower()
            if is_quit(cat) or cat in ("返回", "q"):
                return
            scope = (
                "Trajectory-Frame"
                if cat in ("时刻参数", "f")
                else ("Trajectory-All" if cat in ("轨迹参数", "a") else "Task-Global")
            )
            clist = self.pm.list_plugins(scope_filter=scope)
            if not clist:
                print("无可用插件")
                pause()
                continue
            print("可用插件：")
            for i, p in enumerate(clist, 1):
                print(f"{i}. {p.name} - {p.description}")
            s2 = input_line("选择插件编号：")
            if is_quit(s2):
                continue
            if not s2.isdigit() or not (1 <= int(s2) <= len(clist)):
                continue
            plugin = clist[int(s2) - 1]
            args = prompt_args_by_input_spec(plugin)
            if not args and plugin.input.get("mode") == "form":
                continue
            try:
                result = plugin.run(self.current_task, args)
                apply_plugin_result(self.current_task, result)
                print("计算完成。")
            except Exception as ex:
                print(f"执行失败：{ex}")
            pause()
            continue

    # ===== 参数查看 =====
    def menu_view_task_params(self):
        clear_screen()
        print("=== 任务参数 ===")
        if not self.current_task.meta:
            print("暂无任务参数")
        else:
            cols = ["键", "值"]
            rows = [
                {"键": k, "值": self.current_task.meta[k]}
                for k in sorted(self.current_task.meta.keys())
            ]
            w = {c: len(c) for c in cols}
            for r in rows:
                for c in cols:
                    w[c] = max(w[c], len(format_value(r.get(c))))
            header = " | ".join(c.ljust(w[c]) for c in cols)
            sep = "-+-".join("-" * w[c] for c in cols)
            print(header)
            print(sep)
            for r in rows:
                print(" | ".join(format_value(r.get(c)).ljust(w[c]) for c in cols))
        pause()

    # ===== 保存 =====
    def menu_save(self):
        clear_screen()
        name = input_line(f"任务名（当前 {self.current_task.name}，留空=不变）：")
        if not is_quit(name) and name.strip():
            self.current_task.name = name.strip()
        try:
            self.current_task.save(root=self.tasks_root)
            print(
                f"已保存：{os.path.abspath(os.path.join(self.tasks_root, self.current_task.name))}"
            )
        except Exception as ex:
            print(f"保存失败：{ex}")
        pause()

    # ===== 切换 =====
    def menu_switch(self):
        clear_screen()
        root = self.tasks_root
        os.makedirs(root, exist_ok=True)
        names = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
        if not names:
            print("暂无任务")
            pause()
            return
        for i, n in enumerate(names, 1):
            print(f"{i}. {n}")
        s = input_line("任务名或编号：")
        if is_quit(s):
            return
        chosen = None
        if s.isdigit():
            idx = int(s)
            chosen = names[idx - 1] if 1 <= idx <= len(names) else None
        elif s in names:
            chosen = s
        if not chosen:
            return
        try:
            self.current_task = Task.load(name=chosen, root=root)
            self.pm.load_plugins()
            print(f"已切换：{self.current_task.name}")
        except Exception as ex:
            print(f"切换失败：{ex}")
        pause()


# ========== 入口 ==========
def main():
    MDManagerCLI().run()


if __name__ == "__main__":
    main()
