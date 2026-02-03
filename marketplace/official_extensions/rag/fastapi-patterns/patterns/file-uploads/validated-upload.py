"""Secure File Upload Pattern with Validation.

Keywords: file upload, multipart, image upload, validation, security

CRITICAL: File uploads are a common attack vector.
Always validate:
1. File type (using magic bytes, not just extension)
2. File size
3. Sanitize filename

Requirements:
    pip install fastapi python-multipart python-magic

For production, also consider:
- Virus scanning
- Image re-encoding to strip metadata
- Rate limiting uploads
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
import uuid
import os

# Optional: python-magic for better file type detection
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

app = FastAPI()

# Configuration
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

ALLOWED_CONTENT_TYPES = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/gif": [".gif"],
    "image/webp": [".webp"],
    "application/pdf": [".pdf"],
}

# Magic bytes for common file types
MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # WebP starts with RIFF
    b"%PDF": "application/pdf",
}


def detect_file_type(content: bytes) -> str | None:
    """Detect file type from magic bytes.

    More reliable than Content-Type header which can be spoofed.
    """
    if HAS_MAGIC:
        # Use python-magic if available (more reliable)
        mime = magic.from_buffer(content, mime=True)
        return mime

    # Fallback to manual magic byte detection
    for magic_bytes, mime_type in MAGIC_BYTES.items():
        if content.startswith(magic_bytes):
            return mime_type

    return None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    NEVER use user-provided filename directly.
    Generate a safe filename with UUID.
    """
    # Get original extension (for reference only)
    original_ext = Path(filename).suffix.lower() if filename else ""

    # Generate safe filename with UUID
    safe_name = f"{uuid.uuid4()}{original_ext}"

    return safe_name


async def validate_upload(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> bytes:
    """Validate uploaded file.

    Returns file content if valid, raises HTTPException otherwise.
    """
    # Read file content
    content = await file.read()

    # Check size
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size / 1024 / 1024:.1f} MB"
        )

    # Detect actual file type from content
    detected_type = detect_file_type(content)

    if detected_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {list(ALLOWED_CONTENT_TYPES.keys())}"
        )

    # Verify extension matches detected type
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        allowed_extensions = ALLOWED_CONTENT_TYPES.get(detected_type, [])

        if ext and ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File extension doesn't match content type"
            )

    return content


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file with validation.

    - Validates file type using magic bytes
    - Validates file size
    - Sanitizes filename
    - Stores with safe UUID-based name
    """
    # Validate and get content
    content = await validate_upload(file)

    # Generate safe filename
    safe_filename = sanitize_filename(file.filename or "upload")

    # Determine subdirectory based on type
    detected_type = detect_file_type(content)
    type_dir = detected_type.split("/")[0] if detected_type else "other"

    # Create type-specific directory
    save_dir = UPLOAD_DIR / type_dir
    save_dir.mkdir(exist_ok=True)

    # Save file
    save_path = save_dir / safe_filename
    save_path.write_bytes(content)

    return {
        "filename": safe_filename,
        "original_filename": file.filename,
        "content_type": detected_type,
        "size": len(content),
        "path": str(save_path.relative_to(UPLOAD_DIR)),
    }


@app.post("/upload/multiple")
async def upload_multiple_files(files: list[UploadFile] = File(...)):
    """Upload multiple files."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per request")

    results = []
    for file in files:
        try:
            content = await validate_upload(file)
            safe_filename = sanitize_filename(file.filename)

            save_path = UPLOAD_DIR / safe_filename
            save_path.write_bytes(content)

            results.append({
                "filename": safe_filename,
                "original_filename": file.filename,
                "size": len(content),
                "status": "success",
            })
        except HTTPException as e:
            results.append({
                "original_filename": file.filename,
                "status": "failed",
                "error": e.detail,
            })

    return {"uploads": results}
