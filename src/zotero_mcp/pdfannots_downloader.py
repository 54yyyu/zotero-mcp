"""
Utility for downloading and installing the pdfannots2json tool.
"""

import os
import sys
import platform
import shutil
import tempfile
import tarfile
import zipfile
import urllib.request
from pathlib import Path

# Constants
CURRENT_VERSION = "1.0.15"
BASE_URL = f"https://github.com/mgmeyers/pdfannots2json/releases/download/{CURRENT_VERSION}/"

# Download URLs based on platform and architecture
DOWNLOAD_URLS = {
    "darwin": {
        "x86_64": f"{BASE_URL}pdfannots2json.Mac.Intel.tar.gz",
        "arm64": f"{BASE_URL}pdfannots2json.Mac.M1.tar.gz"
    },
    "linux": {
        "x86_64": f"{BASE_URL}pdfannots2json.Linux.x64.tar.gz"
    },
    "win32": {
        "x86_64": f"{BASE_URL}pdfannots2json.Windows.x64.zip",
        "AMD64": f"{BASE_URL}pdfannots2json.Windows.x64.zip"  # Windows reports AMD64 instead of x86_64
    }
}

def get_executable_name():
    """Get the name of the executable based on the platform"""
    if platform.system().lower() == "windows":
        return "pdfannots2json.exe"
    else:
        return f"pdfannots2json-{platform.system().lower()}-{platform.machine()}"

def get_install_dir():
    """Get the directory to install the executable"""
    return os.path.expanduser("~/.pdfannots2json")

def get_executable_path():
    """Get the full path to the executable"""
    return os.path.join(get_install_dir(), get_executable_name())

def get_download_url():
    """Get the download URL for the current platform and architecture"""
    system = platform.system().lower()
    if system == "darwin":
        system = "darwin"  # macOS
    elif system == "windows":
        system = "win32"

    machine = platform.machine()

    # Map architecture names
    if machine == "amd64":
        machine = "x86_64"

    # Check if we have a URL for this platform/architecture
    if system in DOWNLOAD_URLS and machine in DOWNLOAD_URLS[system]:
        return DOWNLOAD_URLS[system][machine]

    return None

def make_executable(path):
    """Make a file executable"""
    if platform.system().lower() != "windows":
        current_mode = os.stat(path).st_mode
        os.chmod(path, current_mode | 0o111)  # Add executable bit

def exists():
    """Check if the executable exists"""
    return os.path.exists(get_executable_path())

def download_and_install():
    """Download and extract the executable

    Returns:
        bool: True if successful, False otherwise
    """
    install_dir = get_install_dir()
    url = get_download_url()
    if not url:
        print(f"No download URL available for {platform.system()} {platform.machine()}")
        return False

    print(f"Downloading pdfannots2json from {url}")

    try:
        # Create install directory if it doesn't exist
        os.makedirs(install_dir, exist_ok=True)

        # Remove any existing executable
        if exists():
            os.remove(get_executable_path())

        # Create a temporary directory for the download
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the file
            archive_path = os.path.join(temp_dir, "download.archive")
            urllib.request.urlretrieve(url, archive_path)

            # Extract based on file type
            if url.endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=install_dir)
            elif url.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zip_file:
                    zip_file.extractall(path=install_dir)

            # Make sure the executable is executable
            exe_path = get_executable_path()
            if os.path.exists(exe_path):
                make_executable(exe_path)

            # Legacy file handling
            legacy_exe = os.path.join(install_dir, "pdfannots2json")
            if os.path.exists(legacy_exe) and not os.path.exists(exe_path):
                os.rename(legacy_exe, exe_path)
                make_executable(exe_path)

        print(f"Successfully installed pdfannots2json to {exe_path}")
        return True

    except Exception as e:
        print(f"Error downloading pdfannots2json: {e}")
        return False
