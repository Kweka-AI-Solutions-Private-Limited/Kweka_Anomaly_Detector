"""
InspectAI Storage Service
-------------------------
Local storage abstraction for reference images, test inspection images,
and model version artifacts.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Tuple, List, Optional
from fastapi import UploadFile, HTTPException

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

STORAGE_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"
REFERENCES_DIR = STORAGE_BASE_DIR / "references"
INSPECTIONS_DIR = STORAGE_BASE_DIR / "inspections"
ARTIFACTS_DIR = STORAGE_BASE_DIR / "artifacts"


def get_storage_base_dir() -> Path:
    """Returns the base storage directory path."""
    return STORAGE_BASE_DIR

# Ensure directories exist
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_checksum(file_path: Path) -> str:
    """Calculates SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_reference_image(model_id: str, upload_file: UploadFile) -> Tuple[Path, str, int, str]:
    """
    Saves an uploaded reference image to storage/references/{model_id}/.
    Returns: (absolute_path, relative_uri, file_size, checksum)
    """
    ext = Path(upload_file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image extension '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    model_dir = REFERENCES_DIR / str(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Secure filename
    safe_filename = f"{hashlib.md5(upload_file.filename.encode()).hexdigest()[:10]}_{Path(upload_file.filename).name}"
    target_path = model_dir / safe_filename

    # Save content
    file_size = 0
    with open(target_path, "wb") as out_file:
        while chunk := upload_file.file.read(65536):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE_BYTES:
                out_file.close()
                target_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")
            out_file.write(chunk)

    checksum = calculate_checksum(target_path)
    relative_uri = f"storage/references/{model_id}/{safe_filename}"
    return target_path, relative_uri, file_size, checksum


def save_inspection_image(inspection_id: str, upload_file: UploadFile) -> Tuple[Path, str, int, str]:
    """
    Saves an uploaded test sample image to storage/inspections/{inspection_id}/.
    Returns: (absolute_path, relative_uri, file_size, checksum)
    """
    ext = Path(upload_file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image extension '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    insp_dir = INSPECTIONS_DIR / str(inspection_id)
    insp_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{Path(upload_file.filename).name}"
    target_path = insp_dir / safe_filename

    file_size = 0
    with open(target_path, "wb") as out_file:
        while chunk := upload_file.file.read(65536):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE_BYTES:
                out_file.close()
                target_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")
            out_file.write(chunk)

    checksum = calculate_checksum(target_path)
    relative_uri = f"storage/inspections/{inspection_id}/{safe_filename}"
    return target_path, relative_uri, file_size, checksum


def get_version_artifacts_dir(model_id: str, version_number: int) -> Path:
    """Returns directory path for storing a version's PatchCore artifacts."""
    v_dir = ARTIFACTS_DIR / str(model_id) / f"v{version_number}"
    v_dir.mkdir(parents=True, exist_ok=True)
    return v_dir
