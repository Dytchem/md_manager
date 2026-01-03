# md_manager.py
# -*- coding: utf-8 -*-

import csv
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import viewer
from cli import MDManagerCLI
from core import (SimpleTable, Task, Trajectory, _apply_sort_dict_rows,
                  _natural_key_parts, build_value_pred, format_value,
                  parse_index_spec, parse_mixed_selection, parse_tid_values)
from recorder import ActionRecorder
from ui_utils import clear_screen, input_line, is_quit, pause
from viewer import TableViewer


# md_manager.py 作为兼容启动器，委托给 cli.MDManagerCLI
def main():
    MDManagerCLI().run()


if __name__ == "__main__":
    main()
