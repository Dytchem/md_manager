# viewer.py (restored full-featured table viewer)
from typing import List, Dict, Any, Optional
import re
import os
import csv
from ui_utils import clear_screen, pause, input_line, is_quit
from core import (
    format_value,
    build_value_pred,
    parse_index_spec,
    parse_tid_values,
    parse_mixed_selection,
    _as_float_if_possible,
    _natural_key_parts,
    _apply_sort_dict_rows,
    SimpleTable,
)
from recorder import ActionRecorder


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
    def _select_rows_by_spec_default_firstcol(rows_local: List[Dict[str, Any]], first_col: str, spec: str) -> List[Dict[str, Any]]:
        spec = (spec or "").strip()
        if not spec:
            return []
        if spec.startswith("#"):
            idxs = parse_index_spec(spec[1:], len(rows_local))
            return [rows_local[i - 1] for i in idxs]
        is_range, payload = build_value_pred(spec)
        sel: List[Dict[str, Any]] = []
        for r in rows_local:
            val = r.get(first_col)
            if is_range:
                if payload(val):
                    sel.append(r)
            else:
                for tok in payload:
                    fvq = _as_float_if_possible(tok)
                    fvv = _as_float_if_possible(val)
                    if fvq is not None and fvv is not None:
                        if fvv == fvq:
                            sel.append(r)
                            break
                    else:
                        if str(val) == str(tok):
                            sel.append(r)
                            break
        return sel

    @staticmethod
    def _select_rows_by_spec(rows_local: List[Dict[str, Any]], cols: List[str], spec: str, default_first_col: Optional[str] = None) -> List[Dict[str, Any]]:
        spec = (spec or "").strip()
        if not spec:
            return []
        if "=" in spec and not spec.startswith("#"):
            f, v = spec.split("=", 1)
            f = f.strip()
            v = v.strip()
            if f not in cols:
                return []
            is_range, payload = build_value_pred(v)
            selected: List[Dict[str, Any]] = []
            for r in rows_local:
                val = r.get(f)
                if is_range:
                    if payload(val):
                        selected.append(r)
                else:
                    fvq = _as_float_if_possible(v)
                    fvv = _as_float_if_possible(val)
                    if fvq is not None and fvv is not None:
                        if fvv == fvq:
                            selected.append(r)
                    else:
                        if str(val) == v:
                            selected.append(r)
            return selected
        if default_first_col:
            return TableViewer._select_rows_by_spec_default_firstcol(rows_local, default_first_col, spec)
        idxs = parse_index_spec(spec, len(rows_local))
        return [rows_local[i - 1] for i in idxs]

    @staticmethod
    def run(table: SimpleTable, default_columns: Optional[List[str]] = None, page_size: int = 20, title: str = "", export_all_handler=None, delete_handler=None, export_page_option: bool = True, recorder: Optional[ActionRecorder] = None, context: Optional[Dict[str, Any]] = None):
        rows_local = list(table.rows)
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
                print(f"显示列：{', '.join(cols)} | 页：{page + 1}/{total} | 行：{s}-{e}")
                if sort_state["key"]:
                    kw_tip = (f"，关键字优先：{sort_state['keyword']!r}" if sort_state["keyword"] else "")
                    print(f"排序：{sort_state['key']}（{sort_state['order']}）{kw_tip}")
                print("-" * 80)
                TableViewer._print(cols, rows_local, s, e)
                print("-" * 80)

            menu_items = ["列设(c)", "下页(n)", "上页(p)", "行数(r)", "跳页(g)", "排序(s)", "抽取(x)", "导出(e)", "返回(q)"]
            if export_all_handler is not None:
                menu_items.insert(-1, "全导(a)")
            if delete_handler is not None:
                menu_items.insert(-1, "删行(d)")
            print("命令：" + "｜".join(menu_items))

            raw_cmd = input_line(" > ").strip()
            if not raw_cmd:
                continue
            parts = raw_cmd.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("q", "退出", "返回"):
                return
            elif cmd == "n":
                if page + 1 < refresh_total():
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
                    if 1 <= tp <= refresh_total():
                        page = tp - 1
            elif cmd == "c":
                cols_sorted = sorted(table.columns, key=_natural_key_parts)
                TableViewer._print_cols_with_idx(cols_sorted)
                s = input_line("列编号或列名（空格/逗号；空=全部）：")
                if not s.strip():
                    cols = list(table.columns)
                else:
                    mixed = parse_mixed_selection(s, cols_sorted)
                    if mixed:
                        cols = [c for c in mixed if c in table.columns]
                    else:
                        toks = [t for t in re.split(r"[,\s]+", s.strip()) if t]
                        cols = [cols_sorted[int(t) - 1] for t in toks if t.isdigit() and 1 <= int(t) <= len(cols_sorted)]
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
                sort_state = {"key": key, "order": order, "keyword": kw or None}
                rows_local = _apply_sort_dict_rows(rows_local, key, order, kw or None)
                page = 0
            elif cmd == "x":
                print(f"提示：默认按“{cols[0]}”筛选；其它列请用“列名=值”。行号用“#1,3,5-10”。")
                spec = input_line("抽取条件：")
                if is_quit(spec):
                    continue
                tmp_rows = [{c: r.get(c) for c in cols} for r in rows_local]
                sel = TableViewer._select_rows_by_spec(tmp_rows, cols, spec, default_first_col=cols[0])
                if not sel:
                    print("未匹配到条目")
                    pause()
                    continue
                keep_ids = set()
                # find rows matching
                new_rows = []
                for r in rows_local:
                    match = False
                    for srow in sel:
                        ok = True
                        for c in cols:
                            if str(srow.get(c)) != str(r.get(c)):
                                ok = False
                                break
                        if ok:
                            match = True
                            break
                    if match:
                        new_rows.append(r)
                rows_local = new_rows
                page = 0
            elif cmd == "e":
                if arg:
                    path = arg
                else:
                    path = input_line("输出CSV（默认 table_view.csv；输入 q 取消）：").strip()
                if is_quit(path):
                    continue
                if not path:
                    path = "table_view.csv"
                try:
                    SimpleTable(cols, rows_local).to_csv(path, columns=cols)
                    print(f"已导出：{os.path.abspath(path)}")
                    try:
                        if recorder is not None:
                            recorder.record("export", {"type": "traj_view", "path": os.path.abspath(path), "cols": cols, "context": context or {}})
                    except Exception:
                        pass
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()
            elif cmd == "a" and export_all_handler is not None:
                path = input_line("输出CSV（默认 all_trajs_view.csv；输入 q 取消)：").strip()
                if is_quit(path):
                    continue
                if not path:
                    path = "all_trajs_view.csv"
                try:
                    export_all_handler(cols, path)
                    print(f"已导出：{os.path.abspath(path)}")
                    try:
                        if recorder is not None:
                            recorder.record("export", {"type": "traj_view_all", "path": os.path.abspath(path), "cols": cols, "context": context or {}})
                    except Exception:
                        pass
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()
            elif cmd == "d" and delete_handler is not None:
                print(f"提示：默认按“{cols[0]}”筛选；其它列请用“列名=值”。行号用“#1,3,5-10”。")
                spec = input_line("删除条件：")
                if is_quit(spec):
                    continue
                tmp_rows = [{c: r.get(c) for c in cols} for r in rows_local]
                matches = TableViewer._select_rows_by_spec(tmp_rows, cols, spec, default_first_col=cols[0])
                if not matches:
                    print("未匹配到行")
                    pause()
                    continue
                # map matches back to actual rows by equality
                to_delete = []
                for r in rows_local:
                    for m in matches:
                        ok = True
                        for c in cols:
                            if str(m.get(c)) != str(r.get(c)):
                                ok = False
                                break
                        if ok:
                            to_delete.append(r)
                            break
                try:
                    delete_handler(to_delete)
                    rows_local = [r for r in rows_local if r not in to_delete]
                    print(f"已删除 {len(to_delete)} 行")
                except Exception as ex:
                    print(f"删除失败：{ex}")
                pause()


def menu_trajectory_list(manager):
    fields = manager.current_task.settings.get("list_fields", ["traj_id", "name"])[:]
    order_tids = sorted(manager.current_task.trajectories.keys(), key=lambda x: int(x))
    page_size = max(1, int(manager.current_task.settings.get("page_size", 20)))
    page = 0
    sort_state = {"key": None, "order": "asc", "keyword": None}

    def build_rows(order_ids: List[str], use_fields: List[str]):
        rows: List[Dict[str, Any]] = []
        for tid in order_ids:
            t = manager.current_task.trajectories.get(tid)
            if not t:
                continue
            row = {"traj_id": t.traj_id, "name": t.name}
            for f in use_fields:
                if f in ("traj_id", "name"):
                    continue
                row[f] = t.meta.get(f)
            rows.append(row)
        return rows, ["traj_id", "name"] + [f for f in use_fields if f not in ("traj_id", "name")]

    rows_local, cols_local = build_rows(order_tids, fields)

    def refresh_total():
        return max(1, (len(rows_local) + page_size - 1) // page_size)

    while True:
        total = refresh_total()
        clear_screen()
        print("=== 轨迹列表 ===")
        # print table
        start = page * page_size
        end = min(len(rows_local), (page + 1) * page_size)
        if rows_local:
            TableViewer._print(cols_local, rows_local, start, end)
        else:
            print("（空）")

        if sort_state["key"]:
            kw_tip = (f"，关键字优先：{sort_state['keyword']!r}" if sort_state["keyword"] else "")
            print(f"\n页：{page + 1}/{total}；每页：{page_size}；排序：{sort_state['key']}（{sort_state['order']}）{kw_tip}")
        else:
            print(f"\n页：{page + 1}/{total}；每页：{page_size}")

        print("命令：查看(v)｜删除(d)｜列设(c)｜下页(n)｜上页(p)｜行数(r)｜跳页(g)｜排序(s)｜抽取(x)｜导出(e)｜返回(q)")
        cmd = input_line("> ").strip()
        if not cmd:
            continue
        parts = cmd.split(None, 1)
        base = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if is_quit(base) or base == "q":
            return
        elif base == "n":
            if page + 1 < total:
                page += 1
        elif base == "p":
            if page > 0:
                page -= 1
        elif base == "r":
            new_ps = input_line("新的每页行数：")
            if new_ps.isdigit() and int(new_ps) > 0:
                page_size = int(new_ps)
                page = 0
        elif base == "g":
            to_page = input_line("跳转到第几页：")
            if to_page.isdigit():
                tp = int(to_page)
                if 1 <= tp <= total:
                    page = tp - 1
        elif base == "c":
            # choose fields to display
            cols_all = ["traj_id", "name"]
            calc_cols = sorted({k for t in manager.current_task.trajectories.values() for k in t.meta.keys()}, key=_natural_key_parts)
            for c in calc_cols:
                if c not in cols_all:
                    cols_all.append(c)
            cols_sorted = sorted(cols_all, key=_natural_key_parts)
            TableViewer._print_cols_with_idx(cols_sorted)
            s2 = input_line("字段编号或列名（空格/逗号均可；空=不变）：")
            if is_quit(s2) or not s2.strip():
                continue
            mixed = parse_mixed_selection(s2, cols_sorted)
            chosen = mixed if mixed else [c.strip() for c in re.split(r"[,\s]+", s2) if c.strip() and c.strip() in cols_sorted]
            manager.current_task.settings["list_fields"] = chosen
            fields = chosen[:]
            rows_local, cols_local = build_rows(order_tids, fields)
            page = 0
        elif base == "s":
            key = input_line("排序字段：")
            if key and key not in cols_local:
                print("字段不在当前显示列（排序未更改）。")
                pause()
                continue
            order = input_line("顺序（asc/desc）：").lower() or "asc"
            if order not in ("asc", "desc"):
                order = "asc"
            kw = input_line("关键字（可空；命中者优先）：")
            if key:
                rows_local = _apply_sort_dict_rows(rows_local, key, order, kw or None)
                sort_state = {"key": key, "order": order, "keyword": kw or None}
                page = 0
        elif base == "x":
            print(f"提示：默认按“{cols_local[0]}”筛选；其它列请用“列名=值”。行号用“#1,3,5-10”。")
            spec = input_line("抽取条件：")
            if is_quit(spec):
                continue
            sel = TableViewer._select_rows_by_spec(rows_local, cols_local, spec, default_first_col=cols_local[0])
            if not sel:
                print("未匹配到条目")
                pause()
                continue
            # keep only matched tids
            keep = set()
            for r in sel:
                keep.add(str(r.get("traj_id")))
            order_tids = [tid for tid in order_tids if tid in keep]
            rows_local, cols_local = build_rows(order_tids, fields)
            page = 0
        elif base == "e":
            path = input_line("输出CSV（默认 traj_list.csv；输入 q 取消）：").strip()
            if is_quit(path):
                continue
            if not path:
                path = "traj_list.csv"
            try:
                SimpleTable(cols_local, rows_local).to_csv(path)
                print(f"已导出：{os.path.abspath(path)}")
                try:
                    manager.recorder.record("export", {"type": "traj_list", "path": os.path.abspath(path), "cols": cols_local})
                except Exception:
                    pass
            except Exception as ex:
                print(f"导出失败：{ex}")
            pause()
        elif base == "v":
            # view single trajectory
            q = arg if arg else input_line("查看条件（轨迹ID 或 表达式）：").strip()
            target_tid: Optional[str] = None
            if not q:
                tid_in = input_line("轨迹ID：").strip()
                if tid_in in manager.current_task.trajectories:
                    target_tid = tid_in
            else:
                matches = TableViewer._select_rows_by_spec(rows_local, cols_local, q, default_first_col=cols_local[0])
                if not matches:
                    print("未匹配到轨迹")
                    pause()
                    continue
                if len(matches) == 1:
                    target_tid = str(matches[0].get("traj_id"))
                else:
                    print("匹配多条，单选：")
                    for i, r in enumerate(matches, 1):
                        name_val = r.get("name") or ""
                        print(f"{i}. [{r.get('traj_id')}] {name_val}")
                    sel_idx = input_line("编号：").strip()
                    if sel_idx.isdigit():
                        idx = int(sel_idx)
                        if 1 <= idx <= len(matches):
                            target_tid = str(matches[idx - 1].get("traj_id"))
            if not target_tid:
                continue
            traj = manager.current_task.trajectories.get(target_tid)
            if not traj:
                print("轨迹不存在")
                pause()
                continue
            menu_view_trajectory(manager, traj)
        elif base == "d":
            print(f"提示：默认按“{cols_local[0]}”筛选；其它列请用“列名=值”。行号用“#1,3,5-10”。")
            q = arg if arg else input_line("删除条件：").strip()
            del_tids: List[str] = []
            if not q:
                tid_in = input_line("轨迹ID（逗号/范围，如 10-20）：").strip()
                options = list(manager.current_task.trajectories.keys())
                del_tids = parse_tid_values(tid_in, options)
            else:
                matches = TableViewer._select_rows_by_spec(rows_local, cols_local, q, default_first_col=cols_local[0])
                if not matches:
                    print("未匹配到轨迹")
                    pause()
                    continue
                opts = [str(r.get("traj_id")) for r in matches]
                print("匹配如下：")
                for i, r in enumerate(matches, 1):
                    name_val = r.get("name") or ""
                    print(f"{i}. [{r.get('traj_id')}] {name_val}")
                sel = input_line("输入编号或轨ID（1,3,5 / 2-6 / all）：").strip().lower()
                if sel == "all":
                    del_tids = opts
                else:
                    chosen = parse_mixed_selection(sel, opts)
                    if chosen:
                        del_tids = chosen
            if not del_tids:
                continue
            for tid in del_tids:
                manager.current_task.remove_trajectory(tid)
            order_tids = [tid for tid in order_tids if tid in manager.current_task.trajectories]
            rows_local, cols_local = build_rows(order_tids, fields)
            page = 0


def menu_view_trajectory(manager, traj):
    while True:
        clear_screen()
        print(f"=== 轨迹详情：{traj.traj_id}（{traj.name}） ===")
        if not traj.meta:
            print("（插件未写入任何参数）")
        else:
            for k in sorted(traj.meta.keys(), key=_natural_key_parts):
                print(f"{k}: {traj.meta.get(k)}")
        print("命令：表视图(t)｜返回(q)")
        cmd = input_line("> ").strip().lower()
        if is_quit(cmd) or cmd in ("返回", "q"):
            return
        elif cmd in ("表视图", "t"):
            all_cols = traj.list_columns()
            cols = manager.choose_columns(all_cols)
            page = int(manager.current_task.settings.get("page_size", 20))

            def export_all_handler(current_cols: List[str], out_path: str):
                all_rows = []
                header = ["traj_id"] + current_cols
                for t in sorted(manager.current_task.trajectories.values(), key=lambda x: int(x.traj_id)):
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
                ids = set(id(r) for r in rows_to_del)
                traj.table.rows = [r for r in traj.table.rows if id(r) not in ids]

            TableViewer.run(SimpleTable(traj.table.columns, traj.table.rows), default_columns=cols, page_size=page, title=f"数据表：{traj.traj_id}", export_all_handler=export_all_handler, delete_handler=delete_handler, export_page_option=True, recorder=manager.recorder, context={"traj_id": traj.traj_id})
            continue