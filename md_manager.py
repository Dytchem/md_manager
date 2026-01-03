# md_manager.py
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import re
import importlib.util
from typing import Dict, List, Optional, Any, Tuple
from recorder import ActionRecorder
from core import (
    format_value,
    SimpleTable,
    Trajectory,
    Task,
    parse_mixed_selection,
    parse_tid_values,
    parse_index_spec,
    build_value_pred,
    _natural_key_parts,
    _apply_sort_dict_rows,
)
import viewer
from viewer import TableViewer
from ui_utils import clear_screen, pause, input_line, is_quit

from cli import MDManagerCLI


# md_manager.py 作为兼容启动器，委托给 cli.MDManagerCLI
def main():
    MDManagerCLI().run()


if __name__ == "__main__":
    main()
