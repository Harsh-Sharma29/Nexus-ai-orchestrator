"""Filesystem-safe naming for workspace document uploads."""

from __future__ import annotations

import os
import re
from pathlib import Path


def sanitize_workspace_filename(filename: str) -> str:
    """Return a safe basename for workspace storage.

    Strips path components and replaces spaces, brackets, and other
    characters that break cross-platform paths (e.g. ``05) Foo ().pdf``).
    """
    name = Path(filename).name
    stem, ext = os.path.splitext(name)
    ext = ext.lower()

    stem = re.sub(r"[\s()\[\]{}]+", "_", stem)
    stem = re.sub(r"[^\w.\-]+", "_", stem, flags=re.ASCII)
    stem = re.sub(r"_+", "_", stem).strip("._")
    if not stem:
        stem = "document"

    return f"{stem}{ext}"


def resolve_workspace_doc_path(doc_path: str) -> str:
    """Resolve a document path, including legacy unsanitized filenames."""
    if os.path.exists(doc_path):
        return doc_path

    directory = os.path.dirname(doc_path)
    sanitized = os.path.join(directory, sanitize_workspace_filename(os.path.basename(doc_path)))
    if os.path.exists(sanitized):
        return sanitized

    return doc_path
