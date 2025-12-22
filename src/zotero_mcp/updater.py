"""
Update functionality for zotero-mcp.

This module provides intelligent updating that detects the original installation
method and preserves all user configurations.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


def detect_installation_method() -> str:
    """
    Detect how zotero-mcp was originally installed.

    Returns:
        Installation method: 'uv', 'pipx', 'conda', or 'pip'
    """
    # Check for uv
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

    # Check for pipx installation
    if is_pipx_installation():
        return "pipx"

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
            return "zotero-mcp" in result.stdout

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
                [sys.executable, "-m", "pip", "show", "zotero-mcp"],
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


def get_latest_version() -> str | None:
    """Get the latest version from GitHub releases."""
    if not requests:
        logger.warning("requests library not available, cannot check for updates")
        return None

    try:
        response = requests.get(
            "https://api.github.com/repos/54yyyu/zotero-mcp/releases/latest",
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            tag_name = data.get("tag_name", "")
            # Remove 'v' prefix if present
            return tag_name.lstrip("v")

    except Exception as e:
        logger.warning(f"Could not fetch latest version: {e}")

    return None


def backup_configurations() -> Path:
    """
    Backup current configurations before update.

    Returns:
        Path to backup directory
    """
    backup_dir = Path(tempfile.mkdtemp(prefix="zotero_mcp_backup_"))

    # Backup Claude Desktop configs
    claude_config_paths = [
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        Path.home() / "Library" / "Application Support" / "Claude Desktop" / "claude_desktop_config.json",
        Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json",
        Path(os.environ.get("APPDATA", "")) / "Claude Desktop" / "claude_desktop_config.json",
        Path.home() / ".config" / "Claude" / "claude_desktop_config.json",
        Path.home() / ".config" / "Claude Desktop" / "claude_desktop_config.json",
    ]

    for config_path in claude_config_paths:
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
            current_config_path = find_claude_config()
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
    repo_url = "git+https://github.com/54yyyu/zotero-mcp.git"

    try:
        if method == "uv":
            cmd = ["uv", "pip", "install", "--upgrade", repo_url]
        elif method == "pip":
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", repo_url]
        elif method == "conda":
            # Use pip within conda environment
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", repo_url]
        elif method == "pipx":
            # pipx requires special handling for git URLs
            # First try to upgrade, if that fails, reinstall
            try:
                result = subprocess.run(
                    ["pipx", "upgrade", "zotero-mcp"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    return True, "Updated successfully via pipx"
            except Exception:
                pass

            # Fall back to reinstall
            cmd = ["pipx", "install", "--force", repo_url]
        else:
            return False, f"Unknown installation method: {method}"

        if force and method != "pipx":
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


def verify_installation() -> tuple[bool, str]:
    """
    Verify that the updated installation is working.

    Returns:
        Tuple of (success, message)
    """
    try:
        # Try to import the module
        import zotero_mcp

        # Try to get version
        from zotero_mcp._version import __version__

        # Try to run a basic command
        result = subprocess.run(
            [sys.executable, "-m", "zotero_mcp.cli", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return True, f"Installation verified successfully (version {__version__})"
        else:
            return False, f"Installation verification failed: {result.stderr}"

    except Exception as e:
        return False, f"Installation verification error: {str(e)}"


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

    # Check if update is needed
    needs_update = current_version != latest_version or force
    result["needs_update"] = needs_update

    if not needs_update and not force:
        result["success"] = True
        result["message"] = f"Already up to date (version {current_version})"
        return result

    if check_only:
        if needs_update:
            result["message"] = f"Update available: {current_version} → {latest_version}"
        else:
            result["message"] = f"Already up to date (version {current_version})"
        result["success"] = True
        return result

    # Detect installation method
    detected_method = method or detect_installation_method()
    result["method"] = detected_method

    print(f"Detected installation method: {detected_method}")
    print(f"Current version: {current_version}")
    print(f"Latest version: {latest_version}")

    if not needs_update:
        print("Already up to date!")
        if not force:
            result["success"] = True
            result["message"] = "Already up to date"
            return result

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
        verify_success, verify_message = verify_installation()

        if not verify_success:
            result["message"] = f"Update completed but verification failed: {verify_message}"
            return result

        print(verify_message)

        # Cleanup backup
        try:
            shutil.rmtree(backup_dir)
        except Exception:
            pass  # Not critical if cleanup fails

        result["success"] = True
        result["message"] = f"Successfully updated from {current_version} to {latest_version}"

    except Exception as e:
        result["message"] = f"Update failed: {str(e)}"

    return result