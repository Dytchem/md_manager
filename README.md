# MD Manager (AI Coding)

一个命令行工具，用于管理分子动力学任务，支持插件化扩展，方便导入轨迹和进行参数计算。

## 功能特性

- 支持批量导入轨迹数据（通过正则匹配子目录）。
- 插件化设计，易于扩展导入和计算功能。
- 命令行交互，提供轨迹查看、任务保存、参数计算等操作。

## 目录结构

```

md_manager/
├─ md_manager.py              # 主程序
├─ plugins/
│  ├─ base.on.py              # 基础插件：导入位置/速度 + 几何/质心/统计/表达式
│  ├─ hop_import.on.py        # 独立插件：导入 Hop 数据（含教程）
│  └─ hop_compute.on.py       # 独立插件：计算 Hop 状态（含教程）
├─ README.md
└─ .gitignore

````

## 快速开始

```bash
# 进入项目目录
cd md_manager

# 运行主程序
python md_manager.py
````

### 主菜单操作

*   **数据导入 (i)**：输入根目录和子目录正则，批量导入轨迹。
*   **轨迹列表 (l)**：查看轨迹、导出数据、调整显示字段。
*   **参数计算 (c)**：选择插件进行几何、质心、统计、Hop 状态等计算。
*   **任务参数 (p)**：查看任务级参数。
*   **保存 (s)**：保存当前任务。
*   **切换 (w)**：切换已保存任务。

## 插件开发指南

*   插件文件放在 `plugins/` 目录，命名以 `.on.py` 结尾。
*   每个插件需定义 `PLUGINS` 列表，包含：
    *   `name`：插件名称
    *   `description`：描述
    *   `scope`：作用范围（Import / Trajectory-Frame / Trajectory-All / Task-Global）
    *   `run(task, args)`：执行逻辑
    *   `input`：输入规范（mode、help、example）
*   参考 `hop_import.on.py` 和 `hop_compute.on.py` 文件中的示例和文档。
