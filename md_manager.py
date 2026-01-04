# md_manager.py
# -*- coding: utf-8 -*-

"""
md_manager.py - repository root launcher
This file delegates to the package implementation under `md_modules`.
"""

from md_modules.cli import MDManagerCLI


def main():
    MDManagerCLI().run()


if __name__ == "__main__":
    main()
