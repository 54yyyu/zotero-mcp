"""
Utility for downloading and installing the pdfannots2json tool.
"""

import os
import platform
import tempfile
import tarfile
import zipfile
import hashlib
import urllib.request

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

# Pinned SHA256 hashes for upstream release binaries.
EXPECTED_SHA256 = {
    "pdfannots2json.Linux.x64.tar.gz": "f5cc05baa70ac15da2cc358c79acb296b8630cdc654ed304acf50bd9489a94bd",
    "pdfannots2json.Mac.Intel.tar.gz": "ce42fee021b37c38fe131db236ca7711282af899a82eaaaf074bdcc6aebb6c74",
    "pdfannots2json.Mac.M1.tar.gz": "c230a4e578e1c2ff2475cb7f6f59d5ee92e4769e7a3b0ef1cee96444922bc5d5",
    "pdfannots2json.Windows.x64.zip": "0d1496dce3518a4f6523af784051ddd1f1a2083690da41d39da7a198199aa4f3",
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


def _verify_archive_checksum(archive_path: str, url: str) -> bool:
    """Verify downloaded archive checksum against pinned values."""
    asset_name = os.path.basename(url)
    expected = EXPECTED_SHA256.get(asset_name)
    if not expected:
        print(f"No pinned checksum available for {asset_name}")
        return False

    hasher = hashlib.sha256()
    with open(archive_path, "rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            hasher.update(chunk)

    actual = hasher.hexdigest()
    if actual != expected:
        print(
            f"Checksum mismatch for {asset_name}. "
            f"Expected {expected}, got {actual}"
        )
        return False
    return True


def _safe_extract_tar(archive_path: str, destination: str) -> None:
    """Safely extract tar archives while blocking path traversal/symlinks."""
    dest_real = os.path.realpath(destination)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.realpath(os.path.join(destination, member.name))
            if not member_path.startswith(dest_real + os.sep) and member_path != dest_real:
                raise ValueError(f"Unsafe tar member path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing to extract symlink from archive: {member.name}")
        tar.extractall(path=destination)


def _safe_extract_zip(archive_path: str, destination: str) -> None:
    """Safely extract zip archives while blocking path traversal."""
    dest_real = os.path.realpath(destination)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        for member in zip_file.namelist():
            member_path = os.path.realpath(os.path.join(destination, member))
            if not member_path.startswith(dest_real + os.sep) and member_path != dest_real:
                raise ValueError(f"Unsafe zip member path: {member}")
        zip_file.extractall(path=destination)


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
            if not _verify_archive_checksum(archive_path, url):
                return False

            # Extract based on file type
            if url.endswith(".tar.gz"):
                _safe_extract_tar(archive_path, install_dir)
            elif url.endswith(".zip"):
                _safe_extract_zip(archive_path, install_dir)

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
