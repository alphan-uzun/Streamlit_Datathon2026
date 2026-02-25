"""Text normalization helpers for STAPLE."""

from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, and remove punctuation."""
    text = text.lower().strip()
    chars = []
    for ch in text:
        if unicodedata.category(ch).startswith("P"):
            continue
        chars.append(ch)
    return _WS_RE.sub(" ", "".join(chars)).strip()

