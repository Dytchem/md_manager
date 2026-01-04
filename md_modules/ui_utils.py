"""Minimal console UI helpers for the CLI interface."""

import os
import sys


_LAST_INPUT_EOF = False


def clear_screen():
    try:
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
    except Exception:
        pass


def pause():
    input("按回车继续...")


def input_line(prompt: str = "> ") -> str:
    """Read a line; track if the user sent EOF (Ctrl+D/Ctrl+Z)."""
    global _LAST_INPUT_EOF
    try:
        _LAST_INPUT_EOF = False
        return input(prompt)
    except EOFError:
        _LAST_INPUT_EOF = True
        return ""


def last_input_was_eof() -> bool:
    return _LAST_INPUT_EOF


def is_quit(s: str) -> bool:
    return str(s).strip().lower() in ("q", "quit", "exit")
