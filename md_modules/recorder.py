"""
recorder.py - Trajectory analysis action recorder and export code generator.

Records user interactions (imports, plugin runs, exports) and generates
reproducible Python scripts that can replay the analysis pipeline.
Exported scripts deduplicate paths using shared variables.
"""
import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional


class ActionRecorder:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def record(self, action: str, params: Dict[str, Any]):
        """Record an action with its parameters in the history.
        
        Args:
            action: Type of action (e.g., 'import', 'plugin_run', 'export')
            params: Action parameters (e.g., root path, plugin name, export type)
        """
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

        # Phase 1: Scan history to extract paths for deduplication.
        # We identify import roots and export targets, then generate deduplicated
        # PATH_N variables to avoid repeating identical paths in the output script.
        seen_imports = []
        path_order: List[str] = []
        # keep a normalized key set to deduplicate across case/sep differences
        path_norm_set: set = set()
        norm_to_original: Dict[str, str] = {}

        def _add_path(pth: Optional[str]):
            if not pth:
                return
            try:
                a = os.path.abspath(pth)
            except Exception:
                a = pth
            try:
                a_norm = os.path.normcase(os.path.normpath(a))
            except Exception:
                a_norm = a
            if a_norm not in path_norm_set:
                path_norm_set.add(a_norm)
                norm_to_original[a_norm] = a
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
                            # Skip adding per-subdir paths to avoid bloating PATH vars.
                            # (import subdirs are internally tracked but not exported)
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
                        # Defer plugin_run folder addition: check later if it is
                        # covered by an import root to avoid redundant PATH variables.
                        try:
                            plugin_folder_candidates.append(folder_arg)
                        except NameError:
                            plugin_folder_candidates = [folder_arg]
            elif ev.get("action") == "export":
                p = ev.get("params") or {}
                _add_path(p.get("path"))
            elif ev.get("action") == "task_params":
                p = ev.get("params") or {}
                _add_path(p.get("path"))
        # Include project root to ensure exported scripts can load the md_modules package.
        try:
            _add_path(os.path.abspath(os.getcwd()))
        except Exception:
            pass

        # Phase 2: Filter plugin_run folders to exclude those already covered by imports.
        # This prevents duplicate PATH variables for subdirectories.
        try:
            candidates = plugin_folder_candidates
        except NameError:
            candidates = []
        for folder_arg in candidates:
            try:
                fa_norm = os.path.normcase(os.path.normpath(os.path.abspath(folder_arg)))
            except Exception:
                fa_norm = folder_arg
            covered = False
            for r_norm, cre, plugin_names, subs_set in seen_imports:
                # if this folder is within the import root or one of its subs,
                # skip adding it as a separate PATH
                if fa_norm == r_norm or (isinstance(r_norm, str) and fa_norm.startswith(r_norm + os.sep)):
                    covered = True
                    break
                if fa_norm in subs_set:
                    covered = True
                    break
            if not covered:
                _add_path(folder_arg)
        # Phase 3: Build indices of plugin_run actions to skip (those covered by import).
        # If a plugin_run's folder matches an import root or subdir with the same plugin,
        # it will be generated as part of the import loop and should be skipped here.
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

        # Phase 4: Build deduplicated PATH_N variable map.
        # Uses normalized keys for deduplication but preserves original path strings.
        path_vars: Dict[str, str] = {}
        for i, pth in enumerate(path_order, 1):
            var = f"PATH_{i}"
            path_vars[pth] = var

        # Phase 5: Generate Python code for each recorded action.
        # Uses PATH_N variables to deduplicate filesystem paths in exported script.
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
                    # handle unknown export types gracefully
                    if ptype == "task_params":
                        # Export task parameters: write current task name, settings, and meta to JSON.
                        lines.append(f"    # export task parameters -> {path}")
                        lines.append("    try:")
                        p_abs = os.path.abspath(path) if path else path
                        pvar = path_vars.get(p_abs)
                        target = pvar if pvar else json.dumps(path, ensure_ascii=False)
                        lines.append(f"        payload = {{'name': task.name, 'settings': task.settings, 'meta': task.meta}}")
                        lines.append(f"        with open({target}, 'w', encoding='utf-8') as fh:")
                        lines.append("            json.dump(payload, fh, ensure_ascii=False, indent=2)")
                        lines.append("    except Exception as __ex:")
                        lines.append("        print('写入任务参数失败：', __ex)")
                        lines.append("")
                    else:
                        lines.append(f"    # unknown export type {ptype} -> {path}")
                        lines.append("")
            else:
                lines.append(f"    # action {a} skipped")
                lines.append("")

        # Phase 6: Emit collected PATH_N variable definitions near the start of main().
        # This ensures all paths are defined before being referenced in action code.
        if path_order:
            try:
                insert_at = lines.index("    pm.load_plugins()") + 1
            except ValueError:
                insert_at = 4
            defs = ["    # Path variables (deduplicated filesystem paths):"]
            for pth in path_order:
                var = path_vars.get(pth)
                if var:
                    # Emit as raw string literal to preserve backslashes and avoid JSON escapes.
                    lit = pth.replace("'", "\\'")
                    defs.append(f"    {var} = r'{lit}'")
            for i, d in enumerate(reversed(defs)):
                lines.insert(insert_at, d)

        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    main()")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
