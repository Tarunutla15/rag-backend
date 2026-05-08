"""Utility helper functions."""
import os
import uuid
from pathlib import Path


def ensure_upload_dir(upload_dir: str) -> Path:
    """Ensure upload directory exists."""
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


def generate_file_id() -> str:
    """Generate a unique file ID."""
    return str(uuid.uuid4())


def get_file_path(upload_dir: str, filename: str) -> Path:
    """Get full path for a file in upload directory."""
    return Path(upload_dir) / filename

