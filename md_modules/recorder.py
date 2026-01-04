# recorder.py (package)
import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional


class ActionRecorder:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def record(self, action: str, params: Dict[str, Any]):
        try:
            self.history.append(
                {
                    "time": datetime.datetime.utcnow().isoformat() + "Z",
                    "action": action,
                    "params": params,
                }
            )
        except Exception:
            pass

    def export_code(self, out_path: str):
        lines: List[str] = []
        lines.append("# -*- coding: utf-8 -*-")
        lines.append("import os, re, json, sys")
        # emit path variable definitions near the top of main()
        if path_order:
            try:
                insert_at = lines.index("    pm.load_plugins()") + 1
            except ValueError:
                insert_at = 4
            defs = ["    # path variables used below:"]
            for pth in path_order:
                var = path_vars.get(pth)
                if var:
                    defs.append(f"    {var} = {json.dumps(pth, ensure_ascii=False)}")
            for i, d in enumerate(reversed(defs)):
                lines.insert(insert_at, d)

        lines.append("")
        # hardcode project path so exported script can import md_manager reliably
        lines.append(
            "# hardcoded project path so imports work when this script is run from any directory"
        )
        lines.append("sys.path.insert(0, r'E:\\GitHub\\md_manager')")
        lines.append("")
        # import the packaged modules in the exported script
        lines.append("import md_modules.plugins as plugins")
        lines.append("import md_modules.core as core")
        lines.append("")
        lines.append("def main():")
        lines.append("    pm = plugins.PluginManager()")
        lines.append("    pm.load_plugins()")
        lines.append("    task = core.Task('replay_task')")
        lines.append("")

        # Pre-scan history to identify import actions and plugin_run entries
        # and collect all file-system paths that appear so we can define
        # shared variables for them in the exported script.
        seen_imports = []
        path_order: List[str] = []
        path_set: set = set()

        def _add_path(pth: Optional[str]):
            if not pth:
                return
            try:
                a = os.path.abspath(pth)
            except Exception:
                a = pth
            if a not in path_set:
                path_set.add(a)
                path_order.append(a)

        for ev in self.history:
            if ev.get("action") == "import":
                p = ev.get("params") or {}
                root = p.get("root")
                pattern = p.get("pattern")
                plugin_names = p.get("plugins") or []
                subs = p.get("subs") or []
                try:
                    cre = re.compile(pattern)
                except Exception:
                    cre = None
                if root:
                    r_abs = os.path.abspath(root)
                    r_norm = os.path.normcase(os.path.normpath(r_abs))
                    subs_set = set()
                    for s in subs:
                        try:
                            full = os.path.normcase(
                                os.path.normpath(os.path.abspath(os.path.join(root, s)))
                            )
                            subs_set.add(full)
                            _add_path(full)
                        except Exception:
                            pass
                    seen_imports.append((r_norm, cre, [str(n) for n in plugin_names], subs_set))
                    _add_path(root)
            elif ev.get("action") == "plugin_run":
                p = ev.get("params") or {}
                args = p.get("args") or {}
                if isinstance(args, dict):
                    folder_arg = args.get("folder")
                    if isinstance(folder_arg, str):
                        _add_path(folder_arg)
            elif ev.get("action") == "export":
                p = ev.get("params") or {}
                _add_path(p.get("path"))
            elif ev.get("action") == "task_params":
                p = ev.get("params") or {}
                _add_path(p.get("path"))

        skip_plugin_run_idxs = set()
        for idx, ev in enumerate(self.history):
            if ev.get("action") != "plugin_run":
                continue
            p = ev.get("params") or {}
            args = p.get("args") or {}
            if not isinstance(args, dict):
                continue
            folder_arg = args.get("folder")
            if not folder_arg:
                continue
            try:
                fa = os.path.normcase(os.path.normpath(os.path.abspath(folder_arg)))
            except Exception:
                fa = folder_arg
            pname = p.get("plugin")
            for r_norm, cre, plugin_names, subs_set in seen_imports:
                if fa in subs_set:
                    if not plugin_names or str(pname) in plugin_names:
                        skip_plugin_run_idxs.add(idx)
                        break
                if fa == r_norm or (isinstance(r_norm, str) and fa.startswith(r_norm + os.sep)):
                    name = os.path.basename(fa)
                    if cre is None or (name and cre.match(name)):
                        if not plugin_names or str(pname) in plugin_names:
                            skip_plugin_run_idxs.add(idx)
                            break

        # Build path variable names to avoid repeated literal duplication
        path_vars: Dict[str, str] = {}
        for i, pth in enumerate(path_order, 1):
            # safe var name
            var = f"PATH_{i}"
            path_vars[pth] = var

        for idx, ev in enumerate(self.history):
            if idx in skip_plugin_run_idxs:
                continue
            a = ev.get("action")
            p = ev.get("params") or {}
            if a == "import":
                root = p.get("root")
                pattern = p.get("pattern")
                plugin_names = p.get("plugins") or []
                lines.append(
                    f"    # import: root={json.dumps(root, ensure_ascii=False)} pattern={json.dumps(pattern, ensure_ascii=False)}"
                )
                lines.append(f"    rx = re.compile({json.dumps(pattern, ensure_ascii=False)})")
                root_abs = os.path.abspath(root) if root else root
                root_var = path_vars.get(root_abs) if root_abs else None
                if root_var:
                    lines.append(
                        f"    subs = sorted([d for d in os.listdir({root_var}) if os.path.isdir(os.path.join({root_var}, d)) and rx.match(d)])"
                    )
                    lines.append(f"    for d in subs:")
                    lines.append(f"        folder = os.path.join({root_var}, d)")
                else:
                    lines.append(
                        f"    subs = sorted([d for d in os.listdir({json.dumps(root, ensure_ascii=False)}) if os.path.isdir(os.path.join({json.dumps(root, ensure_ascii=False)}, d)) and rx.match(d)])"
                    )
                    lines.append(f"    for d in subs:")
                    lines.append(f"        folder = os.path.join({json.dumps(root, ensure_ascii=False)}, d)")
                lines.append(f"        for pname in {json.dumps(plugin_names, ensure_ascii=False)}:")
                lines.append(
                    f"            plugin = next((pp for pp in pm.list_plugins(scope_filter='Import') if pp.name==pname), None)"
                )
                lines.append(f"            if plugin:")
                lines.append("                result = plugin.run(task, {'folder': folder})")
                lines.append("                core.apply_plugin_result(task, result)")
                lines.append("")
                continue
            if a == "plugin_run":
                pname = p.get("plugin")
                scope = p.get("scope")
                args = p.get("args") or {}
                lines.append(f"    # plugin run: {pname} scope={scope}")
                lines.append(
                    f"    plugin = next((pp for pp in pm.list_plugins(scope_filter={json.dumps(scope, ensure_ascii=False)}) if pp.name=={json.dumps(pname, ensure_ascii=False)}), None)"
                )
                lines.append("    if plugin:")
                try:
                    # build a Python literal for args where any known paths
                    # are replaced by their variable names (unquoted)
                    if isinstance(args, dict):
                        parts = []
                        for k, v in args.items():
                            if isinstance(v, str):
                                try:
                                    v_abs = os.path.abspath(v)
                                except Exception:
                                    v_abs = v
                                var = path_vars.get(v_abs)
                                if var:
                                    parts.append(f"{json.dumps(k)}: {var}")
                                    continue
                            parts.append(f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}")
                        args_json = "{" + ", ".join(parts) + "}"
                    else:
                        args_json = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args_json = json.dumps(args, ensure_ascii=False)
                lines.append(f"        result = plugin.run(task, {args_json})")
                lines.append("        core.apply_plugin_result(task, result)")
                lines.append("")
            elif a == "export":
                ptype = p.get("type")
                path = p.get("path")
                cols = p.get("cols") or []
                ctx = p.get("context") or {}
                if ptype == "traj_list":
                    lines.append(f"    # export traj list -> {path}")
                    lines.append("    rows = []")
                    lines.append(
                        "    for tid in sorted(task.trajectories.keys(), key=lambda x: int(x)):"
                    )
                    lines.append("        t = task.trajectories.get(tid)")
                    lines.append("        if not t: continue")
                    lines.append("        row = {}")
                    for c in cols:
                        if c == "traj_id":
                            lines.append("        row['traj_id'] = t.traj_id")
                        elif c == "name":
                            lines.append("        row['name'] = t.name")
                        else:
                            lines.append(
                                f"        row[{json.dumps(c)}] = t.meta.get({json.dumps(c)})"
                            )
                    lines.append("        rows.append(row)")
                    p_abs = os.path.abspath(path) if path else path
                    pvar = path_vars.get(p_abs)
                    if pvar:
                        lines.append(f"    core.SimpleTable({json.dumps(cols, ensure_ascii=False)}, rows).to_csv({pvar})")
                    else:
                        lines.append(
                            f"    core.SimpleTable({json.dumps(cols, ensure_ascii=False)}, rows).to_csv({json.dumps(path, ensure_ascii=False)})"
                        )
                    lines.append("")
                elif ptype in ("traj_view", "traj_view_all"):
                    tid = None
                    if isinstance(ctx, dict):
                        tid = ctx.get("traj_id")
                    if tid is None:
                        lines.append(f"    # skip export (no traj_id) -> {path}")
                        lines.append("")
                    else:
                        lines.append(f"    # export traj {tid} -> {path}")
                        lines.append(
                            f"    t = task.trajectories.get({json.dumps(str(tid))})"
                        )
                        lines.append("    if t:")
                        p_abs = os.path.abspath(path) if path else path
                        pvar = path_vars.get(p_abs)
                        if pvar:
                            lines.append(
                                f"        core.SimpleTable({json.dumps(cols, ensure_ascii=False)}, t.table.rows).to_csv({pvar})"
                            )
                        else:
                            lines.append(
                                f"        core.SimpleTable({json.dumps(cols, ensure_ascii=False)}, t.table.rows).to_csv({json.dumps(path, ensure_ascii=False)})"
                            )
                        lines.append("")
                else:
                    lines.append(f"    # unknown export type {ptype} -> {path}")
                    lines.append("")
                if ptype == "task_params":
                    lines.append(f"    # load task parameters from {path}")
                    lines.append(f"    try:")
                    p_abs = os.path.abspath(path) if path else path
                    pvar = path_vars.get(p_abs)
                    if pvar:
                        lines.append(f"        with open({pvar}, 'r', encoding='utf-8') as fh:")
                    else:
                        lines.append(
                            f"        with open({json.dumps(path, ensure_ascii=False)}, 'r', encoding='utf-8') as fh:"
                        )
                    lines.append(f"            payload = json.load(fh)")
                    lines.append("        task.name = payload.get('name', task.name)")
                    lines.append(
                        "        task.settings.update(payload.get('settings', {}))"
                    )
                    lines.append("        task.meta.update(payload.get('meta', {}))")
                    lines.append(f"    except Exception as __ex:")
                    lines.append(f"        print('加载任务参数失败：', __ex)")
                    lines.append("")
            else:
                lines.append(f"    # action {a} skipped")
                lines.append("")

        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    main()")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
