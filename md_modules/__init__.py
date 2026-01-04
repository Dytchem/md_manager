"""Package init: re-export core CLI types for convenient imports."""

from .cli import MDManagerCLI  # noqa: F401
from .core import *  # noqa: F401,F403
from .plugins import PluginManager  # noqa: F401
from .recorder import ActionRecorder  # noqa: F401
from .viewer import TableViewer  # noqa: F401
