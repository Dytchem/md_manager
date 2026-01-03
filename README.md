# MD Manager

命令行工具，用于管理分子动力学（MD）任务：批量导入轨迹、按插件计算参数、查看与导出结果。

本仓库近期重构为若干模块，并增加了“操作记录与导出（生成可重放的 Python 脚本）”功能，方便复现一整套导入/计算/导出流程。

**主要功能**

- 批量导入轨迹（按子目录正则匹配）
- 插件化架构：支持导入、逐帧计算、整轨计算与任务级计算
- 终端交互式查看：轨迹列表、分页、排序、抽取、导出、全导、删行等（已恢复为传统界面）
- 操作记录器：记录用户的导入、插件运行与导出操作，并能导出简短的重放脚本（使用目录/模式匹配替代逐文件枚举）

快捷文件与脚本

- `md_manager.py`：程序入口（轻量化 launcher，交由 `cli.py` 驱动）
- `core.py`：核心数据结构（`Task`、`Trajectory`、`SimpleTable`）与工具函数

# MD Manager

命令行工具，用于管理分子动力学（MD）任务：批量导入轨迹、通过插件计算参数、在终端查看并导出结果。

主要特性

- 批量导入轨迹（按用户指定的路径或子目录匹配）
- 插件化架构：支持导入（Import）、逐帧计算（Trajectory-Frame）、整轨计算（Trajectory-All）和任务级计算（Task-Global）
- 交互式查看：轨迹列表与单轨迹表视图，支持分页、排序、抽取与导出
- 操作记录与重放：记录用户的导入/插件运行/导出操作，生成可重放的 Python 脚本；生成的脚本会使用用户在界面中指定的导入/导出路径和插件参数，以便精确复现工作流程

主要文件说明

- `md_manager.py`：程序入口（launcher，启动 CLI）
- `cli.py`：主交互菜单与命令分发
- `core.py`：核心数据结构与工具（`Task`、`Trajectory`、`SimpleTable` 等）
# MD Manager

命令行工具，用于管理分子动力学（MD）任务：批量导入轨迹、通过插件计算参数、在终端查看并导出结果。

主要功能

- 批量导入轨迹（按用户指定的根目录和子目录正则匹配）
- 插件化架构：支持四类 scope——`Import`、`Trajectory-Frame`、`Trajectory-All`、`Task-Global`
- 交互式查看：轨迹列表、单轨迹表视图，支持分页、排序、抽取（支持正则/范围语法）、导出与删除
- 操作记录与重放：记录导入、插件运行与导出操作，生成可重放的 Python 脚本以精确复现流程（导出脚本保留用户输入的路径和参数）

快速开始

```bash
cd md_manager
python md_manager.py
```

主要文件

- `md_manager.py`：程序入口（launcher）
- `cli.py`：主交互菜单与命令分发
- `core.py`：核心数据结构与工具（`Task`、`Trajectory`、`SimpleTable`）
- `plugins.py`：插件加载与管理
- `viewer.py`：表格与轨迹列表视图
- `recorder.py`：`ActionRecorder`，记录操作并导出重放脚本
- `plugins/*.on.py`：插件示例（如 `hop_import.on.py`、`hop_compute.on.py` 等）

插件开发要点

- 将插件放入 `plugins/`，文件名以 `.on.py` 结尾，导出 `PLUGINS` 列表
- 插件应包含至少：`name`、`description`、`scope`、`run(task, args)`；可选 `input` 描述用于交互式参数输入

导出与格式化

- 导出生成的回放脚本会包含用户指定的导入/导出路径、插件参数与任务参数，并在脚本顶部确保项目路径可被导入。
- 建议使用 `black` 与 `isort` 保持代码风格：

```bash
python -m pip install --upgrade black isort
python -m black .
python -m isort .
```

更多

如需帮助或想添加插件示例，请查看 `plugins/` 下的示例插件，或直接打开 Issue。 
