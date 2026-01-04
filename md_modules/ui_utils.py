"""Minimal console UI helpers for the CLI interface."""

import os
import sys


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
    try:
        return input(prompt)
    except EOFError:
        return ""


def is_quit(s: str) -> bool:
    return str(s).strip().lower() in ("q", "quit", "exit")
