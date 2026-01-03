# core.py
# 数据模型与通用工具
from typing import Dict, List, Optional, Any, Tuple
import os
import csv
import json
import re


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
    """在给定 options 顺序下解析编号/列名混合选择。"""
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
    """解析 traj_id 列表/范围。"""
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
    """解析行号列表/范围（1-based）。"""
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
    """把 '1,2,3,5-10' 转换为匹配谓词（支持数值范围与等值）。"""
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
                values.add(t)

        def pred(val: Any) -> bool:
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


# ========== 排序工具（自然排序/混合类型） ==========
_nat_re = re.compile(r"\d+|\D+")


def _natural_key_parts(s: str) -> Tuple:
    parts = _nat_re.findall(s)
    key = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return tuple(key)


def _as_float_if_possible(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return None
    return None


def _sort_key_for_value(v: Any) -> Tuple[int, Tuple]:
    """统一生成可比较的排序键。"""
    if v is None:
        return (3, (float("inf"),))
    fv = _as_float_if_possible(v)
    if fv is not None:
        return (0, (fv,))
    if isinstance(v, str):
        return (1, _natural_key_parts(v))
    return (2, (str(v).lower(),))


def _apply_sort_dict_rows(
    rows: List[Dict[str, Any]],
    key_name: str,
    order: str = "asc",
    keyword: Optional[str] = None,
) -> List[Dict[str, Any]]:
    kw = (keyword or "").strip().lower()

    def sort_key(row: Dict[str, Any]):
        v = row.get(key_name)
        hit = 0 if (kw and kw in str(v).lower()) else 1
        type_rank, val_key = _sort_key_for_value(v)
        return (hit, type_rank, val_key)

    reverse = order == "desc"
    return sorted(rows, key=sort_key, reverse=reverse)


def apply_plugin_result(task: Task, result: Any):
    """Apply plugin result dict/list to Task (moved from md_manager)."""
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
