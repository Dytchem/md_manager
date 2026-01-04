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
        seen_imports = []
        for ev in self.history:
            if ev.get("action") != "import":
                continue
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
                        subs_set.add(
                            os.path.normcase(
                                os.path.normpath(os.path.abspath(os.path.join(root, s)))
                            )
                        )
                    except Exception:
                        pass
                seen_imports.append(
                    (r_norm, cre, [str(n) for n in plugin_names], subs_set)
                )

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
            fa = os.path.normcase(os.path.normpath(os.path.abspath(folder_arg)))
            pname = p.get("plugin")
            for r_norm, cre, plugin_names, subs_set in seen_imports:
                if fa in subs_set:
                    if not plugin_names or str(pname) in plugin_names:
                        skip_plugin_run_idxs.add(idx)
                        break
                if fa == r_norm or fa.startswith(r_norm + os.sep):
                    name = os.path.basename(fa)
                    if cre is None or (name and cre.match(name)):
                        if not plugin_names or str(pname) in plugin_names:
                            skip_plugin_run_idxs.add(idx)
                            break

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
                    f"    # import: root={json.dumps(root, ensure_ascii=False)} pattern={json.dumps(pattern)}"
                )
                lines.append(f"    rx = re.compile({json.dumps(pattern)})")
                lines.append(
                    f"    subs = sorted([d for d in os.listdir({ 'r'+json.dumps(root) }) if os.path.isdir(os.path.join({ 'r'+json.dumps(root) }, d)) and rx.match(d)])"
                )
                lines.append(f"    for d in subs:")
                lines.append(
                    f"        folder = os.path.join({ 'r'+json.dumps(root) }, d)"
                )
                lines.append(
                    f"        for pname in {json.dumps(plugin_names, ensure_ascii=False)}:"
                )
                lines.append(
                    f"            plugin = next((pp for pp in pm.list_plugins(scope_filter='Import') if pp.name==pname), None)"
                )
                lines.append(f"            if plugin:")
                lines.append(
                    f"                result = plugin.run(task, { {'folder': 'folder'} })"
                )
                lines.append("                core.apply_plugin_result(task, result)")
                lines.append("")
                continue
            if a == "plugin_run":
                pname = p.get("plugin")
                scope = p.get("scope")
                args = p.get("args") or {}
                lines.append(f"    # plugin run: {pname} scope={scope}")
                lines.append(
                    f"    plugin = next((pp for pp in pm.list_plugins(scope_filter={json.dumps(scope)}) if pp.name=={json.dumps(pname)}), None)"
                )
                lines.append("    if plugin:")
                try:
                    args_json = json.dumps(args, ensure_ascii=False)
                    if (
                        isinstance(args, dict)
                        and "folder" in args
                        and isinstance(args.get("folder"), str)
                    ):
                        args_json = args_json.replace(
                            json.dumps(args.get("folder")),
                            "r" + json.dumps(args.get("folder")),
                        )
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
                    lines.append(
                        f"    core.SimpleTable({json.dumps(cols)}, rows).to_csv({ 'r'+json.dumps(path) })"
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
                        lines.append(
                            f"        core.SimpleTable({json.dumps(cols)}, t.table.rows).to_csv({ 'r'+json.dumps(path) })"
                        )
                        lines.append("")
                else:
                    lines.append(f"    # unknown export type {ptype} -> {path}")
                    lines.append("")
                if ptype == "task_params":
                    lines.append(f"    # load task parameters from {path}")
                    lines.append(f"    try:")
                    lines.append(
                        f"        with open({ 'r'+json.dumps(path) }, 'r', encoding='utf-8') as fh:"
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
