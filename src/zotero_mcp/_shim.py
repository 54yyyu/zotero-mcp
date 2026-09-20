"""Support for deprecated module paths. See docs/architecture.md, "Shims"."""

from __future__ import annotations

import warnings
from importlib import import_module


class MovedModuleWarning(DeprecationWarning):
    """A name was imported from a module path that moved in 0.13.0."""


def forwarder(old: str, new: str | dict[str, str], removed_in: str = "0.14.0"):
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
            f"{old}.{name} moved to {target}.{name} in 0.13.0; this alias is removed in {removed_in}",
            MovedModuleWarning,
            stacklevel=2,
        )
        return value

    return __getattr__
