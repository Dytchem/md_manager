# cli.py (package version)
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional

from . import viewer
from .core import (
    SimpleTable,
    Task,
    Trajectory,
    _apply_sort_dict_rows,
    _natural_key_parts,
    apply_plugin_result,
    build_value_pred,
    format_value,
    parse_index_spec,
    parse_mixed_selection,
    parse_tid_values,
)
from .plugins import PluginManager, prompt_args_by_input_spec
from .recorder import ActionRecorder
from .ui_utils import clear_screen, input_line, is_quit, pause


class MDManagerCLI:
    def __init__(self):
        self.current_task = Task(name="默认任务")
        self.tasks_root = "tasks"
        self.pm = PluginManager()
        self.pm.load_plugins()
        self.recorder = ActionRecorder()

    @staticmethod
    def choose_columns(all_columns: List[str]) -> List[str]:
        clear_screen()
        print("选择显示列（空格/逗号均可；可混合序号与列名；空=精简）")
        ordered = sorted(all_columns, key=_natural_key_parts)
        viewer.TableViewer._print_cols_with_idx(ordered)
        s = input_line("> ")
        if is_quit(s):
            s = ""
        if not s.strip():
            return ordered[: min(8, len(ordered))]
        mixed = parse_mixed_selection(s, ordered)
        if mixed:
            return mixed
        toks = [t for t in re.split(r"[,\s]+", s.strip()) if t]
        return [
            ordered[int(t) - 1]
            for t in toks
            if t.isdigit() and 1 <= int(t) <= len(ordered)
        ]

    def run(self):
        while True:
            clear_screen()
            print("=== 分子动力学任务管理 ===")
            print(
                f"任务名：{self.current_task.name}｜轨迹数：{len(self.current_task.trajectories)}"
            )
            # compact single-line main menu
            print(
                "导入(i) | 列表(l) | 计算(c) | 参数(p) | 保存(s) | 切换(w) | 导出代码(o) | 退出(q)"
            )

            raw = input_line("> ").strip()
            if not raw:
                continue
            parts = raw.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("退出", "q") or is_quit(cmd):
                confirm = input_line("确认退出请输入 yes：").strip().lower()
                if confirm == "yes":
                    print("已退出。")
                    break
                else:
                    continue
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
            elif cmd in ("导出代码", "o"):
                self.menu_export_code()

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

        try:
            self.recorder.record(
                "import",
                {
                    "root": root,
                    "pattern": pattern,
                    "plugins": [p.name for p in imps],
                    "subs": subs,
                },
            )
        except Exception:
            pass

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
                    try:
                        self.recorder.record(
                            "plugin_run",
                            {
                                "plugin": plugin.name,
                                "scope": plugin.scope,
                                "args": {"folder": folder},
                            },
                        )
                    except Exception:
                        pass
                except Exception as ex:
                    logs.append(f"[跳过] {folder} / {plugin.name}: {ex}")
            processed += 1

        if logs:
            print("\n--- 过程日志 ---")
            for line in logs:
                print(line)
        print(f"\n完成：处理子目录 {processed} 个。")
        pause()

    # ===== 轨迹列表（拆至 viewer.menu_trajectory_list） =====
    def menu_trajectory_list(self):
        return viewer.menu_trajectory_list(self)

    # ===== 单轨详情/表视图（同字母；默认按首列筛选） =====
    def menu_view_trajectory(self, traj: Trajectory):
        return viewer.menu_view_trajectory(self, traj)

    # ===== 计算/参数/保存/切换 =====
    def menu_compute(self):
        while True:
            clear_screen()
            print("=== 参数计算 ===")
            print("类别：")
            print("  f - 时刻参数（表行级）")
            print("  a - 轨迹参数（轨迹级）")
            print("  g - 任务参数（任务级）")
            print("  q - 返回")
            cat = input_line(" > ").strip().lower()
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
            s2 = input_line("选择编号：")
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
                try:
                    self.recorder.record(
                        "plugin_run",
                        {"plugin": plugin.name, "scope": plugin.scope, "args": args},
                    )
                except Exception:
                    pass
                print("计算完成。")
            except Exception as ex:
                print(f"执行失败：{ex}")
            pause()
            continue

    def menu_view_task_params(self):
        while True:
            clear_screen()
            print("=== 任务参数 ===")
            if not self.current_task.meta:
                print("暂无任务参数")
            else:
                cols = ["键", "值"]
                rows = [
                    {"键": k, "值": self.current_task.meta[k]}
                    for k in sorted(
                        self.current_task.meta.keys(), key=_natural_key_parts
                    )
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

            print("命令：导出(e)｜返回(q)")
            cmd = input_line(" > ").strip().lower()
            if is_quit(cmd) or cmd in ("返回", "q"):
                return
            elif cmd in ("导出", "e"):
                path = input_line(
                    "输出参数文件（默认 task_params.json；输入 q 取消）："
                ).strip()
                if is_quit(path):
                    continue
                if not path:
                    path = "task_params.json"
                try:
                    payload = {
                        "name": self.current_task.name,
                        "settings": self.current_task.settings,
                        "meta": self.current_task.meta,
                        "traj_ids": list(self.current_task.trajectories.keys()),
                    }
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, ensure_ascii=False, indent=2)
                    print(f"已导出任务参数：{os.path.abspath(path)}")
                    try:
                        self.recorder.record(
                            "export",
                            {
                                "type": "task_params",
                                "path": os.path.abspath(path),
                                "context": {"task": self.current_task.name},
                            },
                        )
                    except Exception:
                        pass
                except Exception as ex:
                    print(f"导出失败：{ex}")
                pause()
            else:
                # 其它输入回到上层
                return

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

    def menu_export_code(self):
        clear_screen()
        print("=== 导出复现脚本 ===")
        if not self.recorder.history:
            print("暂无记录操作，先执行一次导入/导出流程后再导出脚本。")
            pause()
            return
        path = input_line("输出脚本路径（默认 replay_script.py）：").strip()
        if is_quit(path):
            return
        if not path:
            path = "replay_script.py"
        try:
            self.recorder.export_code(path)
            print(f"已生成脚本：{os.path.abspath(path)}")
        except Exception as ex:
            print(f"导出脚本失败：{ex}")
        pause()

    def menu_switch(self):
        clear_screen()
        root = self.tasks_root
        os.makedirs(root, exist_ok=True)
        names = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
            key=_natural_key_parts,
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
