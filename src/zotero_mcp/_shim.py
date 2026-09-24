"""Support for deprecated module paths. See docs/architecture.md, "Shims"."""

from __future__ import annotations

import warnings
from importlib import import_module

# The release that first ships the moved paths, and the one that removes every shim at once. The
# release plan is written here and nowhere else -- shim docstrings, comments and docs/architecture.md
# name no version -- so a change of plan is a change to these two lines.
MOVED_IN = "0.14.0"
REMOVED_IN = "0.15.0"


class MovedModuleWarning(DeprecationWarning):
    """A name was imported from a module path that moved in ``MOVED_IN``."""


def forwarder(old: str, new: str | dict[str, str]):
    """Build a PEP 562 ``__getattr__`` forwarding ``old``'s names to ``new`` (one module path, or a
    name -> module-path map for a split module). Nothing is imported until a name is used, so importing
    the old path costs nothing; each use warns at the caller's line. Patching a name on the old module
    does not reach the package: patch the new path."""

    def __getattr__(name: str):
        if name.startswith("__"):  # inspect/pytest probing __path__ etc. must not import the target
            raise AttributeError(name)
        target = new if isinstance(new, str) else new.get(name)
        if target is None:
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        value = getattr(import_module(target), name)
        warnings.warn(
            f"{old}.{name} moved to {target}.{name} in {MOVED_IN}; this alias is removed in {REMOVED_IN}",
            MovedModuleWarning,
            stacklevel=2,
        )
        return value

    return __getattr__
