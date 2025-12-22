
# md_manager.py
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import re
import importlib.util
from typing import Dict, List, Optional, Any

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

class Trajectory:
    def __init__(self, traj_id: str, name: str, table: SimpleTable, meta: Optional[Dict[str, Any]] = None):
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
                {"traj_id": self.traj_id, "name": self.name, "meta": self.meta, "columns": self.table.columns},
                fh, ensure_ascii=False, indent=2
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
        return {"name": self.name, "settings": self.settings, "traj_ids": list(self.trajectories.keys()), "meta": self.meta}

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

def parse_mixed_selection(line: str, options: List[str]) -> Optional[List[str]]:
    if not line or not line.strip():
        return None
    toks = [tok for tok in re.split(r'[,\s]+', line.strip()) if tok]
    chosen: List[str] = []
    seen = set()
    for tok in toks:
        if tok.isdigit():
            idx = int(tok)
            if 1 <= idx <= len(options):
                c = options[idx - 1]
                if c not in seen:
                    chosen.append(c); seen.add(c)
        else:
            if tok in options and tok not in seen:
                chosen.append(tok); seen.add(tok)
    return chosen if chosen else None

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
        print(header); print(sep)
        for r in rows[start:end]:
            print(" | ".join(format_value(r.get(c)).ljust(w[c]) for c in columns))

    @staticmethod
    def _print_cols_with_idx(columns: List[str]):
        for i, c in enumerate(columns, 1):
            print(f"{i}. {c}")

    @staticmethod
    def run(table: SimpleTable, default_columns: Optional[List[str]] = None, page_size: int = 20, title: str = "", export_all_handler=None):
        cols = default_columns or table.columns
        cols = [c for c in cols if c in table.columns]
        page = 0
        total = max(1, (len(table.rows) + page_size - 1) // page_size)
        while True:
            clear_screen()
            print(title if title else "数据表")
            if not table.rows:
                print("（空表）")
            else:
                s = page * page_size
                e = min(len(table.rows), (page + 1) * page_size)
                print(f"显示列：{', '.join(cols)} | 页：{page + 1}/{total} | 行：{s}-{e}")
                print("-" * 80)
                TableViewer._print(cols, table.rows, s, e)
                print("-" * 80)
            print("命令：设置列(c)｜下一页(n)｜上一页(p)｜导出(e)｜导出全部(ae)｜返回(q)")
            cmd = input_line("> ").lower()
            if is_quit(cmd) or cmd in ("返回", "q"):
                return
            elif cmd in ("下一页", "n"):
                if page + 1 < total: page += 1
            elif cmd in ("上一页", "p"):
                if page > 0: page -= 1
            elif cmd in ("设置列", "c"):
                TableViewer._print_cols_with_idx(table.columns)
                s = input_line("列编号或列名（空格/逗号均可；空=全部）：")
                if is_quit(s): continue
                if not s.strip():
                    cols = list(table.columns)
                else:
                    mixed = parse_mixed_selection(s, table.columns)
                    if mixed:
                        cols = mixed
                    else:
                        toks = [t for t in re.split(r'[,\s]+', s.strip()) if t]
                        cols = [table.columns[int(t) - 1] for t in toks if t.isdigit() and 1 <= int(t) <= len(table.columns)]
            elif cmd in ("导出", "e"):
                path = input_line("输出CSV（默认 view.csv）：") or "view.csv"
                try:
                    SimpleTable(cols, table.rows).to_csv(path, columns=cols)
                    print(f"已导出：{os.path.abspath(path)}")
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()
            elif cmd in ("导出全部", "ae"):
                if export_all_handler is None:
                    print("该视图未提供“导出全部”处理器。"); pause(); continue
                path = input_line("整合导出CSV（默认 all_trajs_view.csv）：") or "all_trajs_view.csv"
                try:
                    export_all_handler(cols, path)
                    print(f"已导出全部轨迹视图：{os.path.abspath(path)}")
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()

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
        files = sorted([f for f in os.listdir(self.plugins_dir) if f.endswith(".on.py")])
        for fname in files:
            path = os.path.join(self.plugins_dir, fname)
            try:
                spec = importlib.util.spec_from_file_location(fname[:-3], path)
                module = importlib.util.module_from_spec(spec)
                # 注入核心类，避免插件侧未定义报错
                setattr(module, 'Trajectory', Trajectory)
                setattr(module, 'SimpleTable', SimpleTable)
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
                    self.plugins.append(Plugin(module, name, desc, scope, run, input_spec))
            except Exception as ex:
                print(f"[插件失败] {fname}: {ex}")

    def list_plugins(self, scope_filter: Optional[str] = None) -> List[Plugin]:
        return [p for p in self.plugins if p.scope == scope_filter] if scope_filter else self.plugins

def prompt_args_by_input_spec(plugin: Plugin) -> Dict[str, Any]:
    spec = plugin.input or {}
    mode = (spec.get("mode") or "form").lower()
    if mode == "line":
        help_text = spec.get("help") or ""
        example = spec.get("example") or ""
        if help_text: print(help_text)
        if example: print(f"示例：{example}")
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

def apply_plugin_result(task: Task, result: Any):
    if isinstance(result, dict):
        proc = result.get("process")
        if isinstance(proc, list) and proc:
            for line in proc:
                print(line)

    # 兼容旧返回
    if isinstance(result, list):
        datasets = result
        added = 0
        for ds in datasets:
            cols = ds.get("columns"); rows = ds.get("rows")
            name = ds.get("name") or None; meta = ds.get("meta") or {}
            if not cols or not rows: continue
            tid = task.next_traj_id()
            if not name: name = f"traj_{tid}"
            table = SimpleTable(cols, rows)
            meta["traj_seq"] = int(tid)
            traj = Trajectory(tid, name, table, meta)
            task.add_trajectory(traj); added += 1
        print(f"导入完成：新增轨迹 {added} 条。")
        return

    if not isinstance(result, dict):
        return

    datasets = result.get("datasets")
    if isinstance(datasets, list):
        added = 0
        for ds in datasets:
            cols = ds.get("columns"); rows = ds.get("rows")
            name = ds.get("name") or None; meta = ds.get("meta") or {}
            if not cols or not rows: continue
            tid = task.next_traj_id()
            if not name: name = f"traj_{tid}"
            table = SimpleTable(cols, rows)
            meta["traj_seq"] = int(tid)
            traj = Trajectory(tid, name, table, meta)
            task.add_trajectory(traj); added += 1
        print(f"导入完成：新增轨迹 {added} 条。")

    traj_tables = result.get("traj_tables") or {}
    for tid, tdata in traj_tables.items():
        tid_str = str(tid)
        traj = task.trajectories.get(tid_str)
        if not traj: continue
        cols = tdata.get("columns") or traj.table.columns
        rows = tdata.get("rows") or traj.table.rows
        traj.table = SimpleTable(cols, rows)
        traj.refresh_basic_meta()

    traj_meta = result.get("traj_meta") or {}
    for tid, kv in traj_meta.items():
        tid_str = str(tid)
        traj = task.trajectories.get(tid_str)
        if not traj: continue
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
        if is_quit(s): s = ""
        if not s.strip():
            return all_columns[: min(8, len(all_columns))]
        mixed = parse_mixed_selection(s, all_columns)
        if mixed: return mixed
        toks = [t for t in re.split(r'[,\s]+', s.strip()) if t]
        return [all_columns[int(t) - 1] for t in toks if t.isdigit() and 1 <= int(t) <= len(all_columns)]

    def run(self):
        while True:
            clear_screen()
            print("=== 分子动力学任务管理 ===")
            print(f"任务名称：{self.current_task.name}｜轨迹数量：{len(self.current_task.trajectories)}")
            print("菜单：数据导入(i)｜轨迹列表(l)｜参数计算(c)｜任务参数(p)｜保存(s)｜切换(w)｜退出(q)")
            cmd = input_line("> ").strip().lower()
            if is_quit(cmd) or cmd in ("退出", "q"):
                print("已退出。"); break
            elif cmd in ("数据导入", "i"):
                self.menu_import()
            elif cmd in ("轨迹列表", "l"):
                self.menu_trajectory_list()
            elif cmd in ("参数计算", "c"):
                self.menu_compute()
            elif cmd in ("任务参数", "p"):
                self.menu_view_task_params()
            elif cmd in ("保存", "s"):
                self.menu_save()
            elif cmd in ("切换", "w"):
                self.menu_switch()

    def menu_import(self):
        clear_screen()
        print("=== 数据导入（正则匹配文件夹，所有 Import 插件串联运行） ===")
        imps = self.pm.list_plugins(scope_filter="Import")
        if not imps:
            print("未发现导入插件"); pause(); return
        root = input_line("批量根目录：").strip()
        if is_quit(root) or not root or not os.path.isdir(root):
            print("根目录无效"); pause(); return
        pattern = input_line("子目录正则（默认 ^\\d+$）：").strip() or r"^\d+$"
        try:
            rx = re.compile(pattern)
        except Exception:
            print(f"正则无效：{pattern}"); pause(); return

        subs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and rx.match(d)])
        if not subs:
            print("未匹配到子目录"); pause(); return

        print(f"匹配到 {len(subs)} 个子目录，开始串联运行 {len(imps)} 个导入插件...")
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

    def menu_trajectory_list(self):
        while True:
            clear_screen()
            print("=== 轨迹列表 ===")
            tbl = self.current_task.list_trajs_table()
            if tbl.rows:
                w = {c: len(c) for c in tbl.columns}
                for r in tbl.rows:
                    for c in tbl.columns:
                        w[c] = max(w[c], len(format_value(r.get(c))))
                header = " | ".join(c.ljust(w[c]) for c in tbl.columns)
                sep = "-+-".join("-" * w[c] for c in tbl.columns)
                print(header); print(sep)
                for r in tbl.rows:
                    print(" | ".join(format_value(r.get(c)).ljust(w[c]) for c in tbl.columns))
            print("命令：轨迹查看(v)｜删除(d)｜导出(e)｜字段设置(f)｜返回(q)")
            cmdline = input_line("> ").strip().lower()
            if is_quit(cmdline) or cmdline in ("返回", "q"):
                return
            toks = cmdline.split()
            cmd = toks[0] if toks else ""
            arg = toks[1] if len(toks) > 1 else None
            if cmd in ("轨迹查看", "v"):
                if not arg: continue
                traj = self.current_task.trajectories.get(arg)
                if not traj:
                    print("轨迹ID不存在"); pause(); continue
                self.menu_view_trajectory(traj)
            elif cmd in ("删除", "d"):
                if not arg: continue
                self.current_task.remove_trajectory(arg)
            elif cmd in ("导出", "e"):
                path = arg or "traj_list.csv"
                try:
                    tbl.to_csv(path, columns=tbl.columns)
                    print(f"已导出：{os.path.abspath(path)}")
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()
            elif cmd in ("字段设置", "f"):
                fields_all = ["traj_id", "name"]
                calc_cols = sorted({k for t in self.current_task.trajectories.values() for k in t.meta.keys()})
                fields_all += [c for c in calc_cols if c not in fields_all]
                for i, f in enumerate(fields_all, 1):
                    print(f"{i}. {f}")
                s2 = input_line("字段编号或列名（空格/逗号均可；空=不变）：")
                if is_quit(s2) or not s2.strip():
                    continue
                mixed = parse_mixed_selection(s2, fields_all)
                chosen = mixed if mixed else [c.strip() for c in s2.split(",") if c.strip() and c.strip() in fields_all]
                if "traj_id" not in chosen:
                    chosen = ["traj_id"] + chosen
                else:
                    chosen = ["traj_id"] + [c for c in chosen if c != "traj_id"]
                self.current_task.settings["list_fields"] = chosen

    def menu_view_trajectory(self, traj: Trajectory):
        while True:
            clear_screen()
            print(f"=== 轨迹详情：{traj.traj_id}（{traj.name}） ===")
            if not traj.meta:
                print("（插件未写入任何参数）")
            else:
                for k in sorted(traj.meta.keys()):
                    print(f"{k}: {traj.meta.get(k)}")
            print("命令：表格视图(t)｜返回(q)")
            cmd = input_line("> ").strip().lower()
            if is_quit(cmd) or cmd in ("返回", "q"):
                return
            elif cmd in ("表格视图", "t"):
                all_cols = traj.list_columns()
                cols = self.choose_columns(all_cols)
                page = int(self.current_task.settings.get("page_size", 20))

                def export_all_handler(current_cols: List[str], out_path: str):
                    all_rows = []
                    header = ["traj_id"] + current_cols
                    for t in sorted(self.current_task.trajectories.values(), key=lambda x: int(x.traj_id)):
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

                TableViewer.run(
                    SimpleTable(traj.table.columns, traj.table.rows),
                    default_columns=cols,
                    page_size=page,
                    title=f"数据表：{traj.traj_id}",
                    export_all_handler=export_all_handler,
                )

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
            scope = "Trajectory-Frame" if cat in ("时刻参数", "f") else ("Trajectory-All" if cat in ("轨迹参数", "a") else "Task-Global")
            clist = self.pm.list_plugins(scope_filter=scope)
            if not clist:
                print("无可用插件"); pause(); continue
            print("可用插件：")
            for i, p in enumerate(clist, 1):
                print(f"{i}. {p.name} - {p.description}")
            s2 = input_line("选择插件编号：")
            if is_quit(s2): continue
            if not s2.isdigit() or not (1 <= int(s2) <= len(clist)): continue
            plugin = clist[int(s2) - 1]
            args = prompt_args_by_input_spec(plugin)
            if not args and plugin.input.get("mode") == "form": continue
            try:
                result = plugin.run(self.current_task, args)
                apply_plugin_result(self.current_task, result)
                print("计算完成。")
            except Exception as ex:
                print(f"执行失败：{ex}")
            pause()
            continue

    def menu_view_task_params(self):
        clear_screen()
        print("=== 任务参数 ===")
        if not self.current_task.meta:
            print("暂无任务参数")
        else:
            cols = ["键", "值"]
            rows = [{"键": k, "值": self.current_task.meta[k]} for k in sorted(self.current_task.meta.keys())]
            w = {c: len(c) for c in cols}
            for r in rows:
                for c in cols:
                    w[c] = max(w[c], len(format_value(r.get(c))))
            header = " | ".join(c.ljust(w[c]) for c in cols)
            sep = "-+-".join("-" * w[c] for c in cols)
            print(header); print(sep)
            for r in rows:
                print(" | ".join(format_value(r.get(c)).ljust(w[c]) for c in cols))
        pause()

    def menu_save(self):
        clear_screen()
        name = input_line(f"任务名称（当前 {self.current_task.name}，留空=不变）：")
        if not is_quit(name) and name.strip():
            self.current_task.name = name.strip()
        try:
            self.current_task.save(root=self.tasks_root)
            print(f"已保存：{os.path.abspath(os.path.join(self.tasks_root, self.current_task.name))}")
        except Exception as ex:
            print(f"保存失败：{ex}")
        pause()

    def menu_switch(self):
        clear_screen()
        root = self.tasks_root
        os.makedirs(root, exist_ok=True)
        names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if not names:
            print("暂无任务"); pause(); return
        for i, n in enumerate(names, 1):
            print(f"{i}. {n}")
        s = input_line("任务名称或编号：")
        if is_quit(s): return
        chosen = None
        if s.isdigit():
            idx = int(s)
            chosen = names[idx - 1] if 1 <= idx <= len(names) else None
        elif s in names:
            chosen = s
        if not chosen: return
        try:
            self.current_task = Task.load(name=chosen, root=root)
            self.pm.load_plugins()
            print(f"已切换：{self.current_task.name}")
        except Exception as ex:
            print(f"切换失败：{ex}")
        pause()

def main():
    MDManagerCLI().run()

if __name__ == "__main__":
    main()
