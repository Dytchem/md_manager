# shim for backward compatibility: re-export CLI from md_modules package
from md_modules.cli import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        MDManagerCLI().run()
    except Exception:
        raise
