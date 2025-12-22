"""
Helper functions for PDF annotation extraction using pdfannots2json.
"""

import os
import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Constants
PDFANNOTS_VERSION = "1.0.15"
PDFANNOTS_BASE_URL = f"https://github.com/mgmeyers/pdfannots2json/releases/download/{PDFANNOTS_VERSION}/"

# Download URLs based on platform and architecture
PDFANNOTS_URLS = {
    "darwin": {
        "x86_64": f"{PDFANNOTS_BASE_URL}pdfannots2json.Mac.Intel.tar.gz",
        "arm64": f"{PDFANNOTS_BASE_URL}pdfannots2json.Mac.M1.tar.gz"
    },
    "linux": {
        "x86_64": f"{PDFANNOTS_BASE_URL}pdfannots2json.Linux.x64.tar.gz"
    },
    "win32": {
        "x86_64": f"{PDFANNOTS_BASE_URL}pdfannots2json.Windows.x64.zip",
        "AMD64": f"{PDFANNOTS_BASE_URL}pdfannots2json.Windows.x64.zip"
    }
}

def get_pdfannots_dir() -> str:
    """Get the directory where pdfannots2json is installed"""
    return os.path.expanduser("~/.pdfannots2json")

def get_pdfannots_executable() -> str:
    """Get the path to the pdfannots2json executable"""
    base_dir = get_pdfannots_dir()
    system = platform.system().lower()
    machine = platform.machine()

    if system == "windows":
        return os.path.join(base_dir, "pdfannots2json.exe")
    else:
        return os.path.join(base_dir, f"pdfannots2json-{system}-{machine}")

def is_pdfannots_installed() -> bool:
    """Check if pdfannots2json is installed"""
    return os.path.exists(get_pdfannots_executable())

def ensure_pdfannots_installed() -> bool:
    """Ensure pdfannots2json is installed, downloading if necessary"""
    if is_pdfannots_installed():
        return True

    # If not installed, use the downloader script to install it
    try:
        from zotero_mcp import pdfannots_downloader
        success = pdfannots_downloader.download_and_install()
        return success
    except Exception as e:
        print(f"Error installing pdfannots2json: {e}")
        return False

def extract_annotations_from_pdf(
    pdf_path: str | Path,
    output_dir: str | None = None,
    image_format: str = "jpg",
    image_dpi: int = 120,
    image_quality: int = 90
) -> list[dict[str, Any]]:
    """
    Extract annotations directly from a PDF file using pdfannots2json

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images (if None, uses a temp dir)
        image_format: Format for extracted images (jpg, png)
        image_dpi: DPI for extracted images
        image_quality: Quality for extracted images (1-100)

    Returns:
        List of annotation objects
    """
    if not ensure_pdfannots_installed():
        print("Error: pdfannots2json is not installed")
        return []

    # Create temporary output directory if none provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Get the path to the executable
    exe_path = get_pdfannots_executable()

    # Construct command similar to what the plugin uses
    cmd = [
        exe_path,
        str(pdf_path),
        "-o", output_dir,
        "-n", "annotation",
        "-f", image_format,
        "-d", str(image_dpi),
        "-q", str(image_quality)
    ]

    try:
        # Run the command and capture JSON output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        annotations = json.loads(result.stdout)
        print(f"Extracted {len(annotations)} annotations from PDF")
        return annotations
    except subprocess.CalledProcessError as e:
        print(f"Error extracting annotations: {e}")
        print(f"stderr: {e.stderr}")
        return []
    except json.JSONDecodeError:
        print("Error parsing JSON output from pdfannots2json")
        return []
