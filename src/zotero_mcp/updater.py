"""
Update functionality for zotero-mcp.

This module provides intelligent updating that detects the original installation
method and preserves all user configurations.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
import logging

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


def _is_uv_tool_installation() -> bool:
    """Check if zotero-mcp is currently installed as a uv tool."""
    if not shutil.which("uv"):
        return False

    try:
        result = subprocess.run(
            ["uv", "tool", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and (
            "zotero-mcp-server" in result.stdout or "zotero-mcp" in result.stdout
        )
    except Exception:
        return False


def detect_installation_method() -> str:
    """
    Detect how zotero-mcp was originally installed.

    Returns:
        Installation method: 'uv', 'pipx', 'conda', or 'pip'
    """
    # Check for uv tool installs first (most reliable uv signal).
    if _is_uv_tool_installation():
        return "uv"

    # Check for pipx installation.
    if is_pipx_installation():
        return "pipx"

    # Check for uv virtualenv/project installs.
    if shutil.which("uv"):
        # Check if we're in a uv-managed project
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / "pyproject.toml").exists():
                try:
                    with open(parent / "pyproject.toml") as f:
                        content = f.read()
                        if "uv" in content.lower() or "[tool.uv" in content:
                            return "uv"
                except Exception:
                    pass

            if (parent / "uv.lock").exists():
                return "uv"

        # Check if we're in a uv virtual environment
        if "VIRTUAL_ENV" in os.environ:
            venv_path = Path(os.environ["VIRTUAL_ENV"])
            pyvenv_cfg = venv_path / "pyvenv.cfg"
            if pyvenv_cfg.exists():
                try:
                    with open(pyvenv_cfg) as f:
                        content = f.read()
                        if "uv" in content.lower():
                            return "uv"
                except Exception:
                    pass

    # Check for conda environment
    if "CONDA_DEFAULT_ENV" in os.environ or "CONDA_PREFIX" in os.environ:
        return "conda"

    # Default to pip
    return "pip"


def is_pipx_installation() -> bool:
    """Check if zotero-mcp was installed via pipx."""
    try:
        # Check if pipx is available
        if not shutil.which("pipx"):
            return False

        # Try to get pipx list
        result = subprocess.run(
            ["pipx", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return "zotero-mcp-server" in result.stdout or "zotero-mcp" in result.stdout

    except Exception:
        pass

    return False


def get_current_version() -> str | None:
    """Get the currently installed version of zotero-mcp."""
    try:
        from zotero_mcp._version import __version__
        return __version__
    except ImportError:
        # Fallback to pip show
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "zotero-mcp-server"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    return None


class ReceiptError(Exception):
    """A uv tool receipt exists but cannot be read or understood."""


def get_installed_version(python: str | os.PathLike[str] | None = None) -> str | None:
    """Read the version that is installed on disk right now.

    ``get_current_version`` answers from the already-imported module, so after
    an install it still reports the version this process started with. A fresh
    interpreter reads whatever the install step actually left behind. ``-E``
    and a neutral working directory keep ``PYTHONPATH`` and the cwd from
    shadowing the install without hiding the user site (``-I`` would, and a
    ``pip install --user`` install lives there). Distribution metadata is read
    rather than the package imported, so a release whose import is broken
    still reports the version it installed (``verify_installation`` is where
    the import is checked).

    Args:
        python: Interpreter of the environment that was updated. Defaults to
            this process's interpreter, which is right for pip/pipx/conda and
            for a uv tool invoked through its own console script.
    """
    try:
        interpreter = str(python or sys.executable)
        result = subprocess.run(
            [
                interpreter,
                "-E",
                "-c",
                "import importlib.metadata as m; print(m.version('zotero-mcp-server'))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_neutral_cwd(interpreter),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    version = result.stdout.strip().splitlines()
    return version[-1].strip() if version else None


def _neutral_cwd(interpreter: str) -> str:
    """A working directory with no Python packages in it: the interpreter's own.

    ``python -c`` and ``python -m`` put the cwd on ``sys.path``, so a source
    checkout in the cwd would shadow the install being measured.
    """
    return str(Path(interpreter).parent)


def _uv_tool_dir() -> Path | None:
    """Directory uv keeps its tool environments and receipts in."""
    try:
        result = subprocess.run(
            ["uv", "tool", "dir"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip())


def _uv_tool_python(tool_dir: Path | None = None) -> Path | None:
    """Interpreter of the ``zotero-mcp-server`` uv tool environment, if present."""
    tool_dir = tool_dir or _uv_tool_dir()
    if tool_dir is None:
        return None
    env = tool_dir / "zotero-mcp-server"
    for candidate in (env / "bin" / "python", env / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    return None


def _read_uv_tool_receipt() -> dict[str, Any] | None:
    """Return the extras, version specifier and Python of the uv tool install.

    uv records the requirement it was installed with in ``uv-receipt.toml`` and
    resolves ``uv tool upgrade`` against it.

    Returns ``None`` when there is no receipt. Raises :class:`ReceiptError`
    when a receipt exists but cannot be read, so the caller never mistakes an
    unreadable pin for the absence of one.
    """
    tool_dir = _uv_tool_dir()
    if tool_dir is None:
        raise ReceiptError("`uv tool dir` failed, so the receipt could not be located")
    receipt_path = tool_dir / "zotero-mcp-server" / "uv-receipt.toml"
    if not receipt_path.exists():
        return None

    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as e:
            raise ReceiptError(f"no TOML parser available to read {receipt_path}") from e

    try:
        with open(receipt_path, "rb") as f:
            receipt = tomllib.load(f)
    except Exception as e:
        raise ReceiptError(f"could not read {receipt_path}: {e}") from e

    tool = receipt.get("tool", {})
    requirement = next(
        (
            r for r in tool.get("requirements", [])
            if re.sub(r"[-_.]+", "-", str(r.get("name", ""))).lower() == "zotero-mcp-server"
        ),
        None,
    )
    if requirement is None:
        raise ReceiptError(f"{receipt_path} has no zotero-mcp-server requirement")
    return {
        "extras": list(requirement.get("extras", [])),
        "specifier": str(requirement.get("specifier", "")),
        "python": tool.get("python") or None,
    }


def _is_exact_pin(specifier: str) -> bool:
    """True for ``==X.Y.Z``: the one constraint ``uv tool upgrade`` can never move.

    A range (``>=0.9,<1``) can still upgrade within its bounds, and a wildcard
    (``==0.9.*``) within its prefix, so those are left to ``uv tool upgrade``
    and are not overridden.
    """
    spec = specifier.strip()
    return spec.startswith("==") and not spec.endswith(".*") and "," not in spec


def _uv_tool_reinstall_command(receipt: dict[str, Any]) -> list[str]:
    """Reinstall at ``@latest``, keeping the extras and Python the user chose.

    ``@latest`` is uv's own escape from a pinned receipt (it says so in the
    hint it prints), and it rewrites the receipt without the pin, so the next
    ``uv tool upgrade`` works normally. A bare ``uv tool install --force
    zotero-mcp-server`` would drop the extras and let uv pick a Python.
    """
    cmd = ["uv", "tool", "install", "--force"]
    if receipt.get("python"):
        cmd += ["--python", str(receipt["python"])]
    extras = receipt.get("extras") or []
    spec = "zotero-mcp-server" + (f"[{','.join(extras)}]" if extras else "")
    cmd.append(f"{spec}@latest")
    return cmd


def get_latest_version() -> str | None:
    """Get the latest version from PyPI (with GitHub releases as fallback)."""
    if not requests:
        logger.warning("requests library not available, cannot check for updates")
        return None

    # Try PyPI first
    try:
        response = requests.get(
            "https://pypi.org/pypi/zotero-mcp-server/json",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("info", {}).get("version")
    except Exception as e:
        logger.warning(f"Could not fetch latest version from PyPI: {e}")

    # Fallback to GitHub releases
    try:
        response = requests.get(
            "https://api.github.com/repos/54yyyu/zotero-mcp/releases/latest",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            tag_name = data.get("tag_name", "")
            return tag_name.lstrip("v")
    except Exception as e:
        logger.warning(f"Could not fetch latest version from GitHub: {e}")

    return None


def _normalize_version(version: str) -> str:
    """Strip whitespace and a leading 'v' from a version string."""
    return version.strip().lstrip("v").strip()


def is_newer_version(current: str, latest: str) -> bool:
    """Return True only if ``latest`` is strictly newer than ``current``.

    A plain ``current != latest`` check treats "ahead of PyPI" as "needs
    update", so anyone on a git/dev install is told an update is available and
    ``zotero-mcp update`` silently downgrades them to the last release. Compare
    ordering instead, so being ahead reports as up to date.

    ``packaging`` is not a declared dependency (it is merely present in most
    environments as a transitive one), so fall back to a numeric-tuple compare
    when it is missing. The fallback stops at the first non-numeric component,
    which is enough to order plain X.Y.Z releases.
    """
    current_norm = _normalize_version(current)
    latest_norm = _normalize_version(latest)

    try:
        from packaging.version import Version

        return Version(latest_norm) > Version(current_norm)
    except Exception:
        def as_tuple(value: str) -> tuple[int, ...]:
            parts: list[int] = []
            for component in value.split("."):
                try:
                    parts.append(int(component))
                except ValueError:
                    break
            return tuple(parts)

        return as_tuple(latest_norm) > as_tuple(current_norm)


def backup_configurations() -> Path:
    """
    Backup current configurations before update.

    Returns:
        Path to backup directory
    """
    backup_dir = Path(tempfile.mkdtemp(prefix="zotero_mcp_backup_"))

    # Backup Claude Desktop configs (all known build locations, issue #392)
    from zotero_mcp.setup_helper import claude_config_candidates

    for config_path in claude_config_candidates():
        if config_path.exists():
            try:
                backup_path = backup_dir / "claude_desktop_config.json"
                shutil.copy2(config_path, backup_path)
                print(f"Backed up Claude Desktop config from: {config_path}")
                break
            except Exception as e:
                logger.warning(f"Could not backup Claude config from {config_path}: {e}")

    # Backup semantic search config
    semantic_config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
    if semantic_config_path.exists():
        try:
            backup_semantic_path = backup_dir / "semantic_config.json"
            shutil.copy2(semantic_config_path, backup_semantic_path)
            print(f"Backed up semantic search config")
        except Exception as e:
            logger.warning(f"Could not backup semantic search config: {e}")

    # Backup ChromaDB database (if exists)
    chroma_db_path = Path.home() / ".config" / "zotero-mcp" / "chroma_db"
    if chroma_db_path.exists():
        try:
            backup_chroma_path = backup_dir / "chroma_db"
            shutil.copytree(chroma_db_path, backup_chroma_path)
            print(f"Backed up ChromaDB database")
        except Exception as e:
            logger.warning(f"Could not backup ChromaDB database: {e}")

    return backup_dir


def restore_configurations(backup_dir: Path) -> bool:
    """
    Restore configurations from backup.

    Args:
        backup_dir: Path to backup directory

    Returns:
        True if restore was successful
    """
    success = True

    # Restore Claude Desktop config
    claude_backup = backup_dir / "claude_desktop_config.json"
    if claude_backup.exists():
        # Find the current Claude config location
        from zotero_mcp.setup_helper import find_claude_config

        try:
            current_config_path = find_claude_config(verbose=True)
            if current_config_path:
                shutil.copy2(claude_backup, current_config_path)
                print(f"Restored Claude Desktop config to: {current_config_path}")
        except Exception as e:
            logger.error(f"Could not restore Claude Desktop config: {e}")
            success = False

    # Restore semantic search config
    semantic_backup = backup_dir / "semantic_config.json"
    if semantic_backup.exists():
        try:
            semantic_config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
            semantic_config_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(semantic_backup, semantic_config_path)
            print(f"Restored semantic search config")
        except Exception as e:
            logger.error(f"Could not restore semantic search config: {e}")
            success = False

    # Restore ChromaDB database
    chroma_backup = backup_dir / "chroma_db"
    if chroma_backup.exists():
        try:
            chroma_db_path = Path.home() / ".config" / "zotero-mcp" / "chroma_db"
            if chroma_db_path.exists():
                shutil.rmtree(chroma_db_path)
            shutil.copytree(chroma_backup, chroma_db_path)
            print(f"Restored ChromaDB database")
        except Exception as e:
            logger.error(f"Could not restore ChromaDB database: {e}")
            success = False

    return success


def update_via_method(method: str, force: bool = False) -> tuple[bool, str]:
    """
    Update zotero-mcp using the specified method.

    Args:
        method: Installation method ('pip', 'uv', 'conda', 'pipx')
        force: Force update even if already latest

    Returns:
        Tuple of (success, message)
    """
    package_name = "zotero-mcp-server"

    try:
        if method == "uv":
            if _is_uv_tool_installation():
                before = get_current_version()
                try:
                    receipt = _read_uv_tool_receipt()
                    receipt_error = None
                except ReceiptError as e:
                    receipt, receipt_error = None, str(e)

                if receipt and _is_exact_pin(receipt["specifier"]):
                    # `uv tool upgrade` resolves against the specifier recorded
                    # at install time, so an exact pin can never move.
                    if sys.platform == "win32":
                        return False, _windows_reinstall_message(receipt)
                    print(
                        f"uv tool receipt pins zotero-mcp-server to '{receipt['specifier']}'; "
                        "reinstalling at @latest instead of upgrading."
                    )
                    cmd = _uv_tool_reinstall_command(receipt)
                else:
                    upgrade_result = subprocess.run(
                        ["uv", "tool", "upgrade", "zotero-mcp-server"],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if upgrade_result.returncode != 0:
                        return False, f"Update failed: {upgrade_result.stderr}"

                    # Exit 0 also covers "Nothing to upgrade". Only a changed
                    # version on disk counts as an upgrade.
                    after = get_installed_version(python=_uv_tool_python())
                    if after is None:
                        return False, (
                            "`uv tool upgrade` ran but the installed version could not be read "
                            "afterwards; run 'zotero-mcp version' to check"
                        )
                    if before is None or is_newer_version(before, after):
                        return True, "Updated successfully via uv tool"

                    # A no-op. Reinstalling at @latest discards whatever the
                    # receipt says, so only do it when the receipt is known
                    # and unconstrained.
                    manual = "uv tool install --force 'zotero-mcp-server[all]@latest'"
                    if receipt is None:
                        why = receipt_error or "no uv tool receipt found for zotero-mcp-server"
                        return False, (
                            f"`uv tool upgrade` changed nothing and the uv tool receipt could not "
                            f"be used ({why}); reinstall by hand, e.g. {manual}"
                        )
                    if receipt["specifier"]:
                        return False, (
                            f"`uv tool upgrade` changed nothing: the uv tool receipt constrains "
                            f"zotero-mcp-server to '{receipt['specifier']}', which excludes newer "
                            f"releases. Reinstall by hand with a wider constraint, e.g. {manual}"
                        )
                    if sys.platform == "win32":
                        return False, _windows_reinstall_message(receipt)
                    print("`uv tool upgrade` changed nothing; reinstalling at @latest.")
                    cmd = _uv_tool_reinstall_command(receipt)
            else:
                cmd = ["uv", "pip", "install", "--upgrade", package_name]
        elif method == "pip":
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package_name]
        elif method == "conda":
            # Use pip within conda environment
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package_name]
        elif method == "pipx":
            # First try to upgrade, if that fails, reinstall
            try:
                result = subprocess.run(
                    ["pipx", "upgrade", "zotero-mcp-server"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    return True, "Updated successfully via pipx"
            except Exception:
                pass

            # Fall back to reinstall
            cmd = ["pipx", "install", "--force", package_name]
        else:
            return False, f"Unknown installation method: {method}"

        if (
            force
            and method != "pipx"
            and cmd[:3] != ["uv", "tool", "install"]
        ):
            cmd.append("--force-reinstall")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return True, f"Successfully updated via {method}"
        else:
            return False, f"Update failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Update timed out"
    except Exception as e:
        return False, f"Update error: {str(e)}"


def _windows_reinstall_message(receipt: dict[str, Any]) -> str:
    """Explain why the reinstall is left to the user on Windows.

    ``uv tool install --force`` deletes and recreates the tool environment,
    and on Windows that environment holds the ``python.exe`` this updater is
    running from, which cannot be deleted while it runs. Doing it from inside
    would fail part-way and leave the environment broken.
    """
    manual = subprocess.list2cmdline(_uv_tool_reinstall_command(receipt))
    return (
        "`uv tool upgrade` cannot move a pinned uv tool, and reinstalling from inside the "
        "running environment is not safe on Windows. Run this from a separate shell:\n  "
        + manual
    )


def verify_installation(python: str | os.PathLike[str] | None = None) -> tuple[bool, str]:
    """
    Verify that the updated installation is working.

    Args:
        python: Interpreter of the environment that was updated; see
            :func:`get_installed_version`.

    Returns:
        Tuple of (success, message)
    """
    try:
        # Run a basic command in a fresh interpreter, so it exercises the
        # files the install step left behind rather than this process's
        # already-imported module.
        interpreter = str(python or sys.executable)
        result = subprocess.run(
            [interpreter, "-E", "-m", "zotero_mcp.cli", "version"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_neutral_cwd(interpreter),
        )

        if result.returncode == 0:
            return True, "Installation verified successfully"
        else:
            return False, f"Installation verification failed: {result.stderr}"

    except Exception as e:
        return False, f"Installation verification error: {str(e)}"


def _probe_python(method: str) -> Path | None:
    """Interpreter to read the post-update version from, for the given method."""
    if method == "uv" and _is_uv_tool_installation():
        return _uv_tool_python()
    return None


def update_zotero_mcp(check_only: bool = False,
                     force: bool = False,
                     method: str | None = None) -> dict[str, Any]:
    """
    Main update function for zotero-mcp.

    Args:
        check_only: Only check for updates without installing
        force: Force update even if already latest
        method: Override auto-detected installation method

    Returns:
        Dictionary with update results
    """
    result = {
        "success": False,
        "current_version": None,
        "latest_version": None,
        "installed_version": None,
        "method": None,
        "message": "",
        "needs_update": False
    }

    # Get current version
    current_version = get_current_version()
    result["current_version"] = current_version

    if not current_version:
        result["message"] = "Could not determine current version"
        return result

    # Get latest version
    latest_version = get_latest_version()
    result["latest_version"] = latest_version

    if not latest_version:
        result["message"] = "Could not check for latest version"
        return result

    # Check if update is needed. Only a strictly newer release counts — a
    # current version ahead of PyPI (git or dev install) must not be "updated"
    # into a downgrade.
    is_ahead = is_newer_version(latest_version, current_version)
    needs_update = is_newer_version(current_version, latest_version) or force
    result["needs_update"] = needs_update

    if is_ahead:
        up_to_date_message = (
            f"Already up to date (version {current_version} is ahead of the "
            f"latest release {latest_version})"
        )
    else:
        up_to_date_message = f"Already up to date (version {current_version})"

    if not needs_update and not force:
        result["success"] = True
        result["message"] = up_to_date_message
        return result

    if check_only:
        if needs_update:
            result["message"] = f"Update available: {current_version} → {latest_version}"
        else:
            result["message"] = up_to_date_message
        result["success"] = True
        return result

    # Detect installation method
    detected_method = method or detect_installation_method()
    result["method"] = detected_method

    print(f"Detected installation method: {detected_method}")
    print(f"Current version: {current_version}")
    print(f"Latest version: {latest_version}")

    if not needs_update:
        print(up_to_date_message)
        if not force:
            result["success"] = True
            result["message"] = up_to_date_message
            return result

    if is_ahead and force:
        print(
            f"Warning: --force will replace {current_version} with the older "
            f"released version {latest_version}."
        )

    # Backup configurations
    print("Backing up configurations...")
    try:
        backup_dir = backup_configurations()
        result["backup_dir"] = str(backup_dir)
    except Exception as e:
        result["message"] = f"Failed to backup configurations: {e}"
        return result

    # Perform update
    print(f"Updating zotero-mcp using {detected_method}...")
    try:
        update_success, update_message = update_via_method(detected_method, force)

        if not update_success:
            result["message"] = update_message
            return result

        print(update_message)

        # Restore configurations
        print("Restoring configurations...")
        restore_success = restore_configurations(backup_dir)

        if not restore_success:
            result["message"] = "Update succeeded but configuration restore had issues"
            return result

        # Verify installation
        print("Verifying installation...")
        probe_python = _probe_python(detected_method)
        verify_success, verify_message = verify_installation(python=probe_python)

        if not verify_success:
            result["message"] = f"Update completed but verification failed: {verify_message}"
            return result

        print(verify_message)

        # The outcome is what is on disk now, not what PyPI said was latest.
        # An install step can exit 0 without changing anything — `uv tool
        # upgrade` under a pinned receipt does exactly that.
        installed_version = get_installed_version(python=probe_python)
        result["installed_version"] = installed_version

        if not installed_version:
            result["message"] = (
                "Update ran but the installed version could not be read afterwards; "
                "run 'zotero-mcp version' to check"
            )
            return result

        installed = _normalize_version(installed_version)
        is_latest = installed == _normalize_version(latest_version)
        if is_newer_version(current_version, installed_version):
            message = f"Successfully updated from {current_version} to {installed_version}"
            if not is_latest:
                message += f" (latest release is {latest_version})"
        elif installed == _normalize_version(current_version):
            if not is_latest:
                result["message"] = (
                    f"Update ran without error but version {installed_version} is still installed "
                    f"(latest is {latest_version}). Check how zotero-mcp-server is pinned or "
                    f"reinstall it by hand."
                )
                return result
            message = f"Reinstalled version {installed_version}"
        else:
            if not (force and is_latest):
                result["message"] = (
                    f"Update left version {installed_version} installed, which is older than the "
                    f"{current_version} that was there before (latest is {latest_version})"
                )
                return result
            message = f"Replaced {current_version} with the older released version {installed_version}"

        # Cleanup backup
        try:
            shutil.rmtree(backup_dir)
        except Exception:
            pass  # Not critical if cleanup fails

        result["success"] = True
        result["message"] = message

    except Exception as e:
        result["message"] = f"Update failed: {str(e)}"

    return result
