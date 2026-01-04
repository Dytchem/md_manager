#!python3
# -*- coding: utf-8 -*-
"""
md_manager - Main entry point for MD trajectory analysis and visualization.
Delegates to the package implementation under md_modules for core functionality.
"""

from md_modules.cli import MDManagerCLI


def main():
    MDManagerCLI().run()


if __name__ == "__main__":
    main()
