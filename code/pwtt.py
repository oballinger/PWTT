"""Backward-compatible shim — re-exports everything from the pwtt package."""
import sys
import os

# Add the parent directory to sys.path so the pwtt *package* is importable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Remove this shim module from sys.modules so the real package can be loaded
_self_key = __name__
if _self_key in sys.modules:
    del sys.modules[_self_key]

# Import the real pwtt package and re-export everything
from pwtt import *  # noqa: F401,F403,E402
from pwtt import __version__  # noqa: F401,E402
