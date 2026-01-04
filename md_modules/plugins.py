"""
plugins.py - Plugin loader and manager for trajectory analysis.

Loads .on.py plugin files from the plugins/ directory, manages plugin
registration, discovery by scope, and user input prompting.
"""
import importlib.util
import os
import re
from typing import Any, Dict, List, Optional

from .core import SimpleTable, Trajectory
from .ui_utils import input_line


class Plugin:
    """Represents a loaded trajectory analysis plugin with metadata and execution function."""
    def __init__(self, module, name, description, scope, run_func, input_spec):
        self.module = module
        self.name = name
        self.description = description
        self.scope = scope
        self.run = run_func
        self.input = input_spec or {}


class PluginManager:
    """Loads, manages, and discovers plugins from the plugins/ directory."""
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
