import os


def clear_screen() -> None:
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass


def pause(msg: str = "按回车继续...") -> None:
    try:
        input(msg)
    except EOFError:
        pass


def input_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def is_quit(s: str) -> bool:
    return s.lower() in ("q", "quit", "exit")
