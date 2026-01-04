# MD Manager

命令行工具，用于批量导入分子动力学（MD）轨迹、通过插件计算/分析、交互式查看并导出结果，还可记录操作生成可重放的脚本。

## 功能

- 导入：按根目录+子目录正则批量读取轨迹数据。
- 插件：五类 scope——`Import`、`Trajectory-Frame`、`Trajectory-All`、`Task-Global`、`Time-Series`，可串联运行。
- 查看：终端表格视图支持分页、排序、抽取（正则/范围）、导出与删除。
- 重放：`ActionRecorder` 记录导入/插件/导出步骤，生成可重放的 Python 脚本（保留用户输入的路径和参数）。

## 目录结构

- `md_manager.py`：入口；启动交互式 CLI。
- `md_modules/cli.py`：主菜单与命令分发。
- `md_modules/core.py`：核心数据结构（`Task`、`Trajectory`、`SimpleTable`）。
- `md_modules/plugins.py`：插件加载、发现与管理。
- `md_modules/viewer.py`：表格/轨迹列表视图与交互。
- `md_modules/recorder.py`：操作记录与回放脚本生成。
- `plugins/*.on.py`：插件实现与示例。
- `tasks/`、`test/`：示例数据与测试脚本/输出。

## 插件

- 放置位置：`plugins/`，文件名以 `.on.py` 结尾。
- 导出符号：文件需暴露 `PLUGINS` 列表。
- 必备字段：`name`、`description`、`scope`、`run(task, args)`；可选 `input` 描述用于交互式参数提示。
- Scope 约定：
  - `Import`：从目录/文件读取数据，构建轨迹。
  - `Trajectory-Frame`：逐帧处理。
  - `Trajectory-All`：单轨迹的整体处理。
  - `Task-Global`：跨轨迹的全局统计或导出。
  - `Time-Series`：基于任务级时间序列的统计与导出。
- 已内置插件示例：导入（`base.on.py`、`hop_import.on.py`）、逐帧与整轨计算（`expr_frame.on.py`、`hop_compute.on.py`）、全局统计（`hop_dihedral_global.on.py`），时间序列统计（`time_state_counts.on.py`、`time_dihedral_counts.on.py`）。

## 快速开始

```bash
cd md_manager
python md_manager.py
```

## 说明（关于 AI Coding）

- 本项目代码与文档可能在 AI 助手协助下生成或修改，请在关键逻辑处自行审阅与测试。
- 若需严格审计，可对生成的回放脚本与插件逻辑进行逐项验证。
